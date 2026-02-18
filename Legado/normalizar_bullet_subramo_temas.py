#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normaliza as colunas `bullet_points` e `subramo` para uso no Notion.

Regras:
1) bullet_points:
   - antes de cada bullet (linha iniciada por "•"), a linha anterior deve
     terminar com vírgula.
   - se a linha anterior terminar com ".", troca por ",".
   - se não terminar com "." nem ",", acrescenta ",".

2) subramo:
   - no máximo 3 itens.
   - prioridade para a hierarquia (título/subtítulo) obtida do HTML do ramo
     a partir dos links (`link_1`, `link_2`, `link_3`).
   - completa com itens já existentes de `subramo` se ainda faltar.
   - separador final: vírgula.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
from urllib.parse import urlparse


SJUR_PREFIX = "sjur-servicos.tse.jus.br/sjur-servicos/rest/download/pdf/"
CIT_RE = re.compile(
    r"(Ac\.|Res\.|EDcl|Embargos?|AgR|MS|REspE?|REspe|RO-?El|RMS|RCEd|AE|TutCautAnt|MC|AAg|Rec\.)",
    re.IGNORECASE,
)
CANONICAL_RE = re.compile(
    r"<link\s+rel=[\"']canonical[\"']\s+href=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(ch) != "Mn"
    )


def normalize_key(text: str) -> str:
    s = strip_accents((text or "").lower())
    s = s.replace("_", ": ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s:]", " ", s)
    return normalize_ws(s)


def clean_text(text: str) -> str:
    return normalize_ws(html.unescape(text or "").replace("\xa0", " "))


def normalize_token_for_dedupe(text: str) -> str:
    s = strip_accents((text or "").lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return normalize_ws(s)


def sanitize_subramo_item(text: str) -> str:
    # Evita vírgulas internas para que o separador CSV/Notion seja inequívoco.
    s = normalize_ws(text)
    s = re.sub(r"\s*,\s*", " - ", s)
    s = re.sub(r"\s*;\s*", " - ", s)
    return normalize_ws(s)


def normalize_bullet_points(value: str) -> str:
    if not (value or "").strip():
        return value or ""

    lines = (value or "").splitlines()
    if len(lines) < 2:
        return value or ""

    out = list(lines)
    for i in range(1, len(out)):
        current = out[i].lstrip()
        if not current.startswith("•"):
            continue

        prev = out[i - 1].rstrip()
        if not prev:
            continue

        if prev.endswith("."):
            prev = prev[:-1] + ","
        elif not prev.endswith(","):
            prev = prev + ","

        out[i - 1] = prev

    return "\n".join(out)


def parse_subramo_tokens(value: str) -> List[str]:
    raw = (value or "").strip()
    if not raw:
        return []

    # remove wrappers comuns
    raw = raw.strip("[]")
    raw = raw.replace("';", ",").replace('";', ",")
    raw = raw.replace(";", ",")

    parts = [normalize_ws(p) for p in re.split(r"[,|\n\r]+", raw) if normalize_ws(p)]
    cleaned: List[str] = []
    seen: Set[str] = set()
    for p in parts:
        p = p.strip(" '\"")
        p = normalize_ws(p)
        if not p:
            continue
        p = sanitize_subramo_item(p)
        if not p:
            continue
        k = normalize_token_for_dedupe(p)
        if not k or k in seen:
            continue
        seen.add(k)
        cleaned.append(p)
    return cleaned


def dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        t = sanitize_subramo_item(v)
        if not t:
            continue
        k = normalize_token_for_dedupe(t)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def map_ramos_to_html(ramos: Sequence[str], html_paths: Sequence[Path]) -> Dict[str, Path]:
    labels_by_path: Dict[Path, List[str]] = {}
    for path in html_paths:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        labels = [path.name.replace(" — Temas Selecionados.html", "").strip()]

        mt = re.search(r"<title>(.*?)</title>", txt, re.IGNORECASE | re.DOTALL)
        if mt:
            title = clean_text(re.sub(r"<[^>]+>", " ", mt.group(1)))
            labels.append(title)
            if "—" in title:
                labels.append(title.split("—")[0].strip())

        mh1 = re.search(r"<h1[^>]*>(.*?)</h1>", txt, re.IGNORECASE | re.DOTALL)
        if mh1:
            labels.append(clean_text(re.sub(r"<[^>]+>", " ", mh1.group(1))))

        labels_by_path[path] = dedupe_preserve_order(labels)

    fallback_target = normalize_key("Enfrentamento à desinformação eleitoral")
    mapping: Dict[str, Path] = {}
    used: Set[Path] = set()

    for ramo in ramos:
        rk = normalize_key(ramo)

        # fallback explícito
        if rk == fallback_target:
            chosen = None
            for p in html_paths:
                if "repositorio" in normalize_key(p.name) and "desinformacao eleitoral" in normalize_key(p.name):
                    chosen = p
                    break
            if chosen is None:
                raise RuntimeError(f"Fallback não encontrou HTML para ramo '{ramo}'.")
            if chosen in used:
                raise RuntimeError(f"HTML já usado por outro ramo: {chosen.name}")
            mapping[ramo] = chosen
            used.add(chosen)
            continue

        best_score = -1
        best_path: Path | None = None
        for p in html_paths:
            score = 0
            for label in labels_by_path[p]:
                lk = normalize_key(label)
                if rk == lk:
                    score = max(score, 100)
                elif rk in lk or lk in rk:
                    score = max(score, 80)
                else:
                    overlap = len(set(rk.split()) & set(lk.split()))
                    score = max(score, overlap)
            if score > best_score:
                best_score = score
                best_path = p

        if best_path is None or best_score < 3:
            raise RuntimeError(f"Não foi possível mapear ramo '{ramo}' para HTML.")
        if best_path in used:
            raise RuntimeError(f"Mapeamento duplicado de HTML detectado: {best_path.name}")
        mapping[ramo] = best_path
        used.add(best_path)

    return mapping


class HierarchyParser(HTMLParser):
    def __init__(self, base_path: str):
        super().__init__(convert_charrefs=False)
        self.base_path = (base_path or "").strip("/")
        self.in_a = False
        self.a_href = ""
        self.a_text: List[str] = []
        self.labels_by_depth: Dict[int, str] = {}
        self.href_to_label_paths: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() != "a":
            return
        self.in_a = True
        self.a_href = dict(attrs).get("href", "")
        self.a_text = []

    def handle_data(self, data: str):
        if self.in_a:
            self.a_text.append(data)

    def handle_endtag(self, tag: str):
        if tag.lower() != "a" or not self.in_a:
            return

        href = clean_text(self.a_href)
        text = clean_text("".join(self.a_text))
        self.in_a = False
        self.a_href = ""
        self.a_text = []

        if not href or not text:
            return

        if SJUR_PREFIX in href:
            if CIT_RE.search(text):
                labels = tuple(self.labels_by_depth[d] for d in sorted(self.labels_by_depth) if self.labels_by_depth.get(d))
                self.href_to_label_paths[href].append(labels)
            return

        if "temasselecionados.tse.jus.br" not in href:
            return
        if not self.base_path:
            return

        parsed = urlparse(href)
        path = (parsed.path or "").strip("/")
        if not path.startswith(self.base_path):
            return

        rest = path[len(self.base_path):].strip("/")
        if not rest:
            return

        depth = len([seg for seg in rest.split("/") if seg])
        if depth < 1:
            return

        self.labels_by_depth[depth] = text
        for k in list(self.labels_by_depth):
            if k > depth:
                del self.labels_by_depth[k]


def extract_hierarchy_map_for_html(path: Path) -> Dict[str, List[str]]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = CANONICAL_RE.search(txt)
    base = ""
    if m:
        base = urlparse(m.group(1)).path

    parser = HierarchyParser(base_path=base)
    parser.feed(txt)

    href_to_best_labels: Dict[str, List[str]] = {}
    for href, label_paths in parser.href_to_label_paths.items():
        if not label_paths:
            continue
        counts = Counter(label_paths)
        # prioriza caminho mais profundo e mais frequente
        best = max(counts.items(), key=lambda kv: (len(kv[0]), kv[1]))[0]
        href_to_best_labels[href] = dedupe_preserve_order(best)
    return href_to_best_labels


def build_ramo_href_hierarchy_map(ramo_to_html: Dict[str, Path]) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for ramo, html_path in ramo_to_html.items():
        out[ramo] = extract_hierarchy_map_for_html(html_path)
    return out


def select_top_subramo(
    row: Dict[str, str],
    href_hierarchy_by_ramo: Dict[str, Dict[str, List[str]]],
    max_items: int,
) -> List[str]:
    ramo = normalize_ws(row.get("ramo", ""))
    ramo_map = href_hierarchy_by_ramo.get(ramo, {})

    preferred: List[str] = []
    for col in ("link_1", "link_2", "link_3"):
        href = normalize_ws(row.get(col, ""))
        if not href:
            continue
        labels = ramo_map.get(href, [])
        # remove prefixo "Parte ..." se houver muitas camadas; prioriza título/subtítulo
        labels_clean = [l for l in labels if not re.match(r"^\s*Parte\b", l, re.IGNORECASE)] or labels
        preferred.extend(labels_clean)

    preferred = dedupe_preserve_order(preferred)

    existing = parse_subramo_tokens(row.get("subramo", ""))
    merged = dedupe_preserve_order(preferred + existing)
    return merged[:max_items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normaliza bullet_points e subramo para integração com Notion."
    )
    parser.add_argument(
        "--csv-in",
        default="temas_selec_TSE_all_2.csv",
        help="CSV de entrada (default: temas_selec_TSE_all_2.csv)",
    )
    parser.add_argument(
        "--csv-out",
        default="temas_selec_TSE_all_2_normalizado.csv",
        help="CSV de saída (default: temas_selec_TSE_all_2_normalizado.csv)",
    )
    parser.add_argument(
        "--html-glob",
        default="* — Temas Selecionados.html",
        help="Glob para HTMLs locais (default: * — Temas Selecionados.html)",
    )
    parser.add_argument(
        "--max-subramo",
        type=int,
        default=3,
        help="Máximo de itens em subramo (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    args = parse_args()

    csv_in = Path(args.csv_in)
    csv_out = Path(args.csv_out)
    html_paths = sorted(Path(".").glob(args.html_glob))
    max_sub = max(1, int(args.max_subramo))

    if not csv_in.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_in}")
    if not html_paths:
        raise RuntimeError(f"Nenhum HTML encontrado com glob: {args.html_glob}")

    with csv_in.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise RuntimeError("CSV de entrada está vazio.")
    if "bullet_points" not in fieldnames or "subramo" not in fieldnames:
        raise RuntimeError("CSV precisa conter colunas 'bullet_points' e 'subramo'.")

    ramos = sorted({normalize_ws(r.get("ramo", "")) for r in rows if normalize_ws(r.get("ramo", ""))})
    ramo_to_html = map_ramos_to_html(ramos, html_paths)
    href_hierarchy_by_ramo = build_ramo_href_hierarchy_map(ramo_to_html)

    bullet_changed = 0
    subramo_changed = 0
    subramo_from_hierarchy = 0

    for row in rows:
        original_bp = row.get("bullet_points", "") or ""
        new_bp = normalize_bullet_points(original_bp)
        if new_bp != original_bp:
            bullet_changed += 1
            row["bullet_points"] = new_bp

        original_sub = normalize_ws(row.get("subramo", ""))
        selected = select_top_subramo(row, href_hierarchy_by_ramo, max_items=max_sub)
        if selected:
            # detecta se veio algo de hierarquia
            ramo = normalize_ws(row.get("ramo", ""))
            ramo_map = href_hierarchy_by_ramo.get(ramo, {})
            has_hier = False
            for col in ("link_1", "link_2", "link_3"):
                href = normalize_ws(row.get(col, ""))
                if href and href in ramo_map:
                    has_hier = True
                    break
            if has_hier:
                subramo_from_hierarchy += 1
        new_sub = ", ".join(selected)
        if new_sub != original_sub:
            subramo_changed += 1
            row["subramo"] = new_sub

    with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV entrada: {csv_in}")
    print(f"CSV saída: {csv_out}")
    print(f"Ramos mapeados: {len(ramo_to_html)}")
    print(f"bullet_points alterados: {bullet_changed}")
    print(f"subramo alterados: {subramo_changed}")
    print(f"linhas com subramo baseado em hierarquia HTML: {subramo_from_hierarchy}")
    print(f"max_subramo aplicado: {max_sub}")


if __name__ == "__main__":
    main()
