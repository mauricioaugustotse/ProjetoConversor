#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enriquecimento global dos temas selecionados do TSE.

Objetivo:
- Ler `temas_selec_TSE_all.csv`
- Mapear cada `ramo` para o HTML local correspondente
- Normalizar `data_julgamento` para M/D/AAAA
- Extrair links de julgados e preencher `link_1`, `link_2`, `link_3` (top-k)
- Gerar `temas_selec_TSE_all_2.csv` sem duplicar linhas
- Emitir auditorias globais
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import re
import sys
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


MIN_CONTAINMENT_LEN = 40

# Reconhecimento básico de citação jurídica
CIT_TOKEN = (
    r"(?:Ac\.|Res\.|EDcl|Embargos?|AgR(?:-[A-Za-z]+)?|MS|REspE?|REspe|REsp|"
    r"RO-?El|RMS|RCEd|AE|TutCautAnt|MC|AAg|Rec\.?)"
)

ANCHOR_TAG_RE = re.compile(
    r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
HAS_CITATION_RE = re.compile(CIT_TOKEN, re.IGNORECASE)
DATE_IN_TEXT_RE = re.compile(r"\bde\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.IGNORECASE)
RELATOR_RE = re.compile(
    r"\brel\.\s*(?:Min\.|Des\.|Juiz(?:a)?|in\.)?\s*([^\).;]+)",
    re.IGNORECASE,
)
CANONICAL_RE = re.compile(
    r"<link\s+rel=[\"']canonical[\"']\s+href=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)

# Classes processuais para extração mais precisa de número/tipo.
PROCESS_CLASS_TOKEN = (
    r"(?:AREspE|REspE?|REspe|REsp|RO-?El|RMS|RCEd|RCED|AE|TutCautAnt|MC|AAg|"
    r"Rec\.?|Cta|Pet|PA|Inst\.?|MS|AI|Ag|RO|RE|PA)"
)
PROCESS_CLASS_AND_NUM_RE = re.compile(
    rf"\b({PROCESS_CLASS_TOKEN})\s*(?:n[ºo°.]?\s*)?(\d[\d./-]*)",
    re.IGNORECASE,
)

# Citações no texto_original
ROW_CITATION_PAREN_RE = re.compile(
    rf"\({CIT_TOKEN}[\s\S]{{0,360}}?rel\.[\s\S]{{0,180}}?\)",
    re.IGNORECASE,
)
ROW_CITATION_FALLBACK_RE = re.compile(
    rf"{CIT_TOKEN}[\s\S]{{0,260}}?rel\.[^\n\.]{{0,140}}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HtmlCitation:
    ramo: str
    source: str
    href: str
    texto_citacao_raw: str
    texto_citacao_norm: str
    numero_norm: str
    tipo_de_processo: str
    relator_raw: str
    relator_norm: str
    data_norm_mdy: str


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_ramo_key(text: str) -> str:
    s = strip_accents((text or "").lower())
    s = s.replace("_", ": ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s:]", " ", s)
    return normalize_ws(s)


def clean_anchor_inner_html(inner_html: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", inner_html or "")
    unescaped = html.unescape(without_tags).replace("\xa0", " ")
    return normalize_ws(unescaped)


def normalize_citation_text(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("−", "-")
    cleaned = strip_accents(cleaned.lower())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return normalize_ws(cleaned)


def normalize_process_number(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    digits = digits.lstrip("0")
    return digits or ""


def normalize_relator(text: str) -> str:
    s = strip_accents((text or "").lower())
    s = re.sub(
        r"\b(min\.|des\.|juiz(?:a)?|in\.|red\.?|designado|designada)\b",
        " ",
        s,
    )
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [t for t in normalize_ws(s).split() if t not in {"de", "da", "do", "dos", "das", "e"}]
    if len(tokens) > 3:
        tokens = tokens[-3:]
    return " ".join(tokens)


def format_relator_display(relator_raw: str) -> str:
    r = normalize_ws(relator_raw)
    if not r:
        return ""
    if re.match(r"^(Min\.|Des\.|Juiz|Juíza)\s+", r, re.IGNORECASE):
        return r
    return f"Min. {r}"


def parse_date_to_mdy(raw_date: str, prefer_day_first: bool) -> Tuple[str, str]:
    """
    Retorna (data_mdy, status) onde status in {'parsed','empty','unparsed'}.
    """
    raw = normalize_ws((raw_date or "").replace("º", ""))
    if not raw:
        return "", "empty"

    m = re.fullmatch(r"(\d{1,2})([./-])(\d{1,2})\2(\d{2,4})", raw)
    if not m:
        return "", "unparsed"

    first_s, sep, second_s, year_s = m.groups()
    first = int(first_s)
    second = int(second_s)
    year = int(year_s)
    if len(year_s) == 2:
        year = 2000 + year if year <= 30 else 1900 + year

    if sep in ".-":
        day, month = first, second
    else:
        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            month, day = first, second
        elif first > 12 and second > 12:
            return "", "unparsed"
        else:
            if prefer_day_first:
                day, month = first, second
            else:
                month, day = first, second

    try:
        dt.date(year, month, day)
    except ValueError:
        return "", "unparsed"

    return f"{month}/{day}/{year}", "parsed"


def extract_relator_from_text(citation_text: str) -> str:
    match = RELATOR_RE.search(citation_text or "")
    return normalize_ws(match.group(1)) if match else ""


def extract_process_from_href(href: str) -> str:
    match = re.search(r"processoNumero=(\d+)", href or "", re.IGNORECASE)
    if not match:
        return ""
    return normalize_process_number(match.group(1))


def normalize_tipo_de_processo(sigla: str) -> str:
    s = normalize_ws(sigla or "")
    s = s.replace("–", "-").replace("—", "-")
    return s


def extract_tipo_from_text(citation_text: str) -> str:
    text = normalize_ws(citation_text or "")
    # Prioriza classe processual vinculada ao número.
    m = PROCESS_CLASS_AND_NUM_RE.search(text)
    if m:
        return normalize_tipo_de_processo(m.group(1))
    # Fallback: primeiro token jurídico detectado.
    m2 = HAS_CITATION_RE.search(text)
    if m2:
        return normalize_tipo_de_processo(m2.group(0))
    return ""


def extract_process_from_text(citation_text: str, href: str = "") -> str:
    from_href = extract_process_from_href(href)
    if from_href:
        return from_href

    text = normalize_ws(citation_text or "")

    # 1) número associado a classe processual (evita capturar nº do acórdão/resolução)
    class_matches = list(PROCESS_CLASS_AND_NUM_RE.finditer(text))
    if class_matches:
        candidates = [normalize_process_number(m.group(2)) for m in class_matches]
        candidates = [c for c in candidates if c]
        if candidates:
            # Em geral o primeiro match já é o processo principal; usa maior por robustez.
            return max(candidates, key=len)

    # 2) número após n./nº
    n_matches = list(re.finditer(r"\bn[ºo°.]?\s*(\d[\d./-]*)", text, re.IGNORECASE))
    if n_matches:
        # Se houver múltiplos, evita valores que parecem data.
        filtered = [
            normalize_process_number(m.group(1))
            for m in n_matches
            if not re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", m.group(1))
        ]
        filtered = [f for f in filtered if f]
        if filtered:
            return max(filtered, key=len)

    # 3) trecho após no/na/nos/nas
    no_seg = re.search(r"\b(?:no|na|nos|nas)\b([^\)]{0,120})", text, re.IGNORECASE)
    if no_seg:
        tokens = re.findall(r"\d[\d./-]*", no_seg.group(1))
        tokens = [
            t for t in tokens
            if not re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", t)
        ]
        if tokens:
            return normalize_process_number(max(tokens, key=lambda x: len(re.sub(r"\D", "", x))))

    # 4) fallback geral antes de rel.
    before_rel = text.split("rel.", 1)[0]
    tokens = re.findall(r"\d[\d./-]*", before_rel)
    tokens = [
        t for t in tokens
        if not re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", t)
    ]
    if not tokens:
        return ""
    return normalize_process_number(max(tokens, key=lambda x: len(re.sub(r"\D", "", x))))


def extract_date_from_text(citation_text: str) -> str:
    match = DATE_IN_TEXT_RE.search(citation_text or "")
    if not match:
        return ""
    date_str, status = parse_date_to_mdy(match.group(1), prefer_day_first=True)
    return date_str if status == "parsed" else ""


def extract_canonical_url(raw_html: str) -> str:
    m = CANONICAL_RE.search(raw_html or "")
    return normalize_ws(m.group(1)) if m else ""


def fetch_url_text(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def extract_row_citations(texto_original: str) -> List[Tuple[str, str]]:
    text = texto_original or ""
    raw_candidates = [normalize_ws(m.group(0)) for m in ROW_CITATION_PAREN_RE.finditer(text)]
    if not raw_candidates:
        raw_candidates = [normalize_ws(m.group(0)) for m in ROW_CITATION_FALLBACK_RE.finditer(text)]

    dedup: List[Tuple[str, str]] = []
    seen_norm: Set[str] = set()
    for raw in raw_candidates:
        norm = normalize_citation_text(raw)
        if len(norm) < 12:
            continue
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        dedup.append((raw, norm))
    return dedup


def summarize_text(value: str, max_len: int = 280) -> str:
    clean = normalize_ws(value or "")
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def detect_html_label_candidates(html_path: Path) -> List[str]:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    labels: List[str] = []

    # 1) nome do arquivo
    labels.append(html_path.name.replace(" — Temas Selecionados.html", "").strip())

    # 2) title
    title_m = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if title_m:
        title = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip()
        if "—" in title:
            labels.append(title.split("—")[0].strip())
        labels.append(title)

    # 3) h1
    h1_m = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.IGNORECASE | re.DOTALL)
    if h1_m:
        h1 = html.unescape(re.sub(r"<[^>]+>", "", h1_m.group(1))).strip()
        if h1:
            labels.append(h1)

    # limpeza de vazios + dedupe preservando ordem
    out: List[str] = []
    seen: Set[str] = set()
    for lbl in labels:
        key = normalize_ws(lbl)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def map_ramos_to_html(ramos: Sequence[str], html_paths: Sequence[Path]) -> Dict[str, Path]:
    if not html_paths:
        raise RuntimeError("Nenhum HTML encontrado para mapeamento de ramos.")

    path_to_labels: Dict[Path, List[str]] = {
        p: detect_html_label_candidates(p) for p in html_paths
    }

    # fallback explícito
    fallback_contains = {
        normalize_ramo_key("Enfrentamento à desinformação eleitoral"): "Repositório - Enfrentamento à desinformação eleitoral",
    }

    mapping: Dict[str, Path] = {}
    used_paths: Set[Path] = set()

    for ramo in ramos:
        ramo_key = normalize_ramo_key(ramo)

        # 0) fallback explícito por substring no nome do arquivo
        forced_path: Path | None = None
        if ramo_key in fallback_contains:
            needle = normalize_ramo_key(fallback_contains[ramo_key])
            for p in html_paths:
                if needle in normalize_ramo_key(p.name):
                    forced_path = p
                    break
            if forced_path is None:
                raise RuntimeError(
                    f"Fallback de mapeamento não encontrou HTML para ramo '{ramo}'."
                )
            if forced_path in used_paths:
                raise RuntimeError(
                    f"Fallback mapeou HTML já usado por outro ramo: '{forced_path.name}'."
                )
            mapping[ramo] = forced_path
            used_paths.add(forced_path)
            continue

        best: Tuple[int, Path | None] = (-1, None)
        for p in html_paths:
            labels = path_to_labels[p]
            score = 0
            for lbl in labels:
                lbl_key = normalize_ramo_key(lbl)
                if ramo_key == lbl_key:
                    score = max(score, 100)
                elif ramo_key in lbl_key or lbl_key in ramo_key:
                    score = max(score, 80)
                else:
                    overlap = len(set(ramo_key.split()) & set(lbl_key.split()))
                    score = max(score, overlap)
            if score > best[0]:
                best = (score, p)

        best_score, best_path = best
        if best_path is None or best_score < 3:
            raise RuntimeError(
                f"Não foi possível mapear ramo '{ramo}' para nenhum HTML com confiança."
            )
        if best_path in used_paths:
            raise RuntimeError(
                f"Mapeamento duplicado detectado: ramo '{ramo}' e outro ramo apontam para '{best_path.name}'."
            )
        mapping[ramo] = best_path
        used_paths.add(best_path)

    missing = [r for r in ramos if r not in mapping]
    if missing:
        raise RuntimeError(f"Ramos sem HTML mapeado: {missing}")
    return mapping


def extract_html_citations_for_ramo(
    ramo: str,
    html_path: Path,
    raw_html: str | None = None,
    source: str = "local",
) -> List[HtmlCitation]:
    if raw_html is None:
        raw_html = html_path.read_text(encoding="utf-8", errors="ignore")

    out: List[HtmlCitation] = []
    seen: Set[Tuple[str, str]] = set()

    for href_raw, inner_html in ANCHOR_TAG_RE.findall(raw_html):
        href = normalize_ws(html.unescape(href_raw))
        if "sjur-servicos.tse.jus.br/sjur-servicos/rest/download/pdf/" not in href:
            continue

        citation_raw = clean_anchor_inner_html(inner_html)
        if not citation_raw:
            continue
        if not HAS_CITATION_RE.search(citation_raw):
            continue

        citation_norm = normalize_citation_text(citation_raw)
        if not citation_norm:
            continue

        dedupe_key = (citation_norm, href)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        out.append(
            HtmlCitation(
                ramo=ramo,
                source=source,
                href=href,
                texto_citacao_raw=citation_raw,
                texto_citacao_norm=citation_norm,
                numero_norm=extract_process_from_text(citation_raw, href=href),
                tipo_de_processo=extract_tipo_from_text(citation_raw),
                relator_raw=extract_relator_from_text(citation_raw),
                relator_norm=normalize_relator(extract_relator_from_text(citation_raw)),
                data_norm_mdy=extract_date_from_text(citation_raw),
            )
        )
    return out


def build_indexes_for_ramo(
    citations: Sequence[HtmlCitation],
) -> Dict[str, object]:
    by_citation: Dict[str, Set[str]] = defaultdict(set)
    by_proc_rel_date: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    by_proc_rel: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    by_proc: Dict[str, Set[str]] = defaultdict(set)
    citation_records: Dict[str, List[HtmlCitation]] = defaultdict(list)
    href_records: Dict[str, List[HtmlCitation]] = defaultdict(list)

    for rec in citations:
        by_citation[rec.texto_citacao_norm].add(rec.href)
        citation_records[rec.texto_citacao_norm].append(rec)
        href_records[rec.href].append(rec)

        if rec.numero_norm:
            by_proc[rec.numero_norm].add(rec.href)
        if rec.numero_norm and rec.relator_norm:
            by_proc_rel[(rec.numero_norm, rec.relator_norm)].add(rec.href)
        if rec.numero_norm and rec.relator_norm and rec.data_norm_mdy:
            by_proc_rel_date[(rec.numero_norm, rec.relator_norm, rec.data_norm_mdy)].add(rec.href)

    return {
        "by_citation": by_citation,
        "by_proc_rel_date": by_proc_rel_date,
        "by_proc_rel": by_proc_rel,
        "by_proc": by_proc,
        "citation_records": citation_records,
        "href_records": href_records,
        "citation_keys": list(by_citation.keys()),
    }


def containment_candidates(
    row_citation_norms: Sequence[str],
    html_citation_keys: Sequence[str],
    html_by_citation: Dict[str, Set[str]],
) -> Set[str]:
    result: Set[str] = set()
    for rc in row_citation_norms:
        if len(rc) < MIN_CONTAINMENT_LEN:
            continue
        for hk in html_citation_keys:
            if len(hk) < MIN_CONTAINMENT_LEN:
                continue
            if rc in hk or hk in rc:
                result.update(html_by_citation[hk])
    return result


def metadata_candidates(
    numero_norm: str,
    relator_norm: str,
    data_norm_mdy: str,
    idx_proc_rel_date: Dict[Tuple[str, str, str], Set[str]],
    idx_proc_rel: Dict[Tuple[str, str], Set[str]],
    idx_proc: Dict[str, Set[str]],
) -> Tuple[Set[str], str]:
    if numero_norm and relator_norm and data_norm_mdy:
        cands = idx_proc_rel_date.get((numero_norm, relator_norm, data_norm_mdy), set())
        if cands:
            return set(cands), "C_proc_rel_data"
    if numero_norm and relator_norm:
        cands = idx_proc_rel.get((numero_norm, relator_norm), set())
        if cands:
            return set(cands), "C_proc_rel"
    if numero_norm:
        cands = idx_proc.get(numero_norm, set())
        if cands:
            return set(cands), "C_processo"
    return set(), "none"


def narrow_candidates_by_metadata(
    candidates: Set[str],
    row_num: str,
    row_rel: str,
    row_date: str,
    href_records: Dict[str, List[HtmlCitation]],
) -> Set[str]:
    """
    Reduz ambiguidades:
    - se houver candidatos com mesmo processo, filtra para eles
    - depois relator
    - depois data
    """
    if not candidates:
        return candidates

    narrowed = set(candidates)

    if row_num:
        with_num = {
            h for h in narrowed
            if any(rec.numero_norm == row_num for rec in href_records.get(h, []))
        }
        if with_num:
            narrowed = with_num

    if row_rel:
        with_rel = {
            h for h in narrowed
            if any(rec.relator_norm == row_rel for rec in href_records.get(h, []))
        }
        if with_rel:
            narrowed = with_rel

    if row_date:
        with_date = {
            h for h in narrowed
            if any(rec.data_norm_mdy == row_date for rec in href_records.get(h, []))
        }
        if with_date:
            narrowed = with_date

    return narrowed


def rank_href_candidate(
    href: str,
    stage: str,
    row_num: str,
    row_rel: str,
    row_date: str,
    row_citations: Sequence[str],
    href_records: Dict[str, List[HtmlCitation]],
) -> Tuple[int, int, str]:
    base_stage_score = {
        "A_exato": 300,
        "B_contencao": 220,
        "C_proc_rel_data": 180,
        "C_proc_rel": 140,
        "C_processo": 100,
        "none": 0,
    }.get(stage, 0)

    strong_bonus = 0
    weak_bonus = 0

    recs = href_records.get(href, [])
    if row_num and any(r.numero_norm == row_num for r in recs):
        strong_bonus += 55
    if row_rel and any(r.relator_norm == row_rel for r in recs):
        strong_bonus += 25
    if row_date and any(r.data_norm_mdy == row_date for r in recs):
        strong_bonus += 20

    if row_num and f"processoNumero={row_num}" in href:
        weak_bonus += 15

    best_overlap = 0
    if row_citations and recs:
        for rc in row_citations:
            for hr in recs:
                hk = hr.texto_citacao_norm
                if not rc or not hk:
                    continue
                if rc == hk:
                    best_overlap = max(best_overlap, 999)
                elif len(rc) >= MIN_CONTAINMENT_LEN and len(hk) >= MIN_CONTAINMENT_LEN and (rc in hk or hk in rc):
                    best_overlap = max(best_overlap, min(len(rc), len(hk)))
    if best_overlap >= 999:
        weak_bonus += 40
    elif best_overlap > 0:
        weak_bonus += min(20, best_overlap // 20)

    if re.search(r"/pdf/\d+$", href):
        weak_bonus += 5

    score = base_stage_score + strong_bonus + weak_bonus
    return score, strong_bonus, href


def parse_args() -> argparse.Namespace:
    today = dt.date.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(
        description="Enriquecimento global de temas TSE com link_1..link_3 e auditoria."
    )
    parser.add_argument(
        "--csv-in",
        default="temas_selec_TSE_all.csv",
        help="CSV de entrada (default: temas_selec_TSE_all.csv)",
    )
    parser.add_argument(
        "--html-glob",
        default="* — Temas Selecionados.html",
        help="Glob para descobrir HTMLs locais (default: '* — Temas Selecionados.html')",
    )
    parser.add_argument(
        "--csv-out",
        default="temas_selec_TSE_all_2.csv",
        help="CSV de saída final (default: temas_selec_TSE_all_2.csv)",
    )
    parser.add_argument(
        "--audit-prefix",
        default=f"auditoria_temas_all_2_{today}",
        help="Prefixo para arquivos de auditoria",
    )
    parser.add_argument(
        "--top-k-links",
        type=int,
        default=3,
        help="Quantidade máxima de links por linha (default: 3)",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Não consulta páginas online; usa apenas HTMLs locais.",
    )
    parser.add_argument(
        "--skip-insert-missing",
        action="store_true",
        help="Não insere linhas novas para casos ausentes detectados nas citações.",
    )
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    args = parse_args()

    csv_in = Path(args.csv_in)
    csv_out = Path(args.csv_out)
    top_k = max(1, int(args.top_k_links))
    audit_prefix = args.audit_prefix
    html_paths = sorted(Path(".").glob(args.html_glob))

    if not csv_in.exists():
        raise FileNotFoundError(f"CSV de entrada não encontrado: {csv_in}")
    if not html_paths:
        raise RuntimeError(f"Nenhum HTML encontrado com glob: {args.html_glob}")

    with csv_in.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise RuntimeError("CSV de entrada está vazio.")

    # 1) descoberta de ramos e mapeamento
    ramos = sorted({normalize_ws(r.get("ramo", "")) for r in rows if normalize_ws(r.get("ramo", ""))})
    ramo_to_html = map_ramos_to_html(ramos, html_paths)

    # 2) parse HTMLs por ramo + índices (local + live opcional)
    ramo_indexes: Dict[str, Dict[str, object]] = {}
    all_html_citations: List[HtmlCitation] = []
    live_diff_by_ramo: Dict[str, Dict[str, int]] = {}
    for ramo in ramos:
        html_path = ramo_to_html[ramo]
        local_raw = html_path.read_text(encoding="utf-8", errors="ignore")
        local_citations = extract_html_citations_for_ramo(
            ramo,
            html_path,
            raw_html=local_raw,
            source="local",
        )

        merged: List[HtmlCitation] = list(local_citations)
        local_keys = {(c.texto_citacao_norm, c.href) for c in local_citations}
        live_only = 0
        local_only = 0

        if not args.skip_live:
            canonical = extract_canonical_url(local_raw)
            if canonical:
                try:
                    live_raw = fetch_url_text(canonical)
                    live_citations = extract_html_citations_for_ramo(
                        ramo,
                        html_path,
                        raw_html=live_raw,
                        source="live",
                    )
                    live_keys = {(c.texto_citacao_norm, c.href) for c in live_citations}
                    live_only = len(live_keys - local_keys)
                    local_only = len(local_keys - live_keys)

                    seen = set(local_keys)
                    for c in live_citations:
                        k = (c.texto_citacao_norm, c.href)
                        if k in seen:
                            continue
                        seen.add(k)
                        merged.append(c)
                except Exception:
                    # Falha de rede não interrompe execução local.
                    live_only = -1
                    local_only = -1

        live_diff_by_ramo[ramo] = {
            "local_citations": len(local_citations),
            "merged_citations": len(merged),
            "live_only": live_only,
            "local_only": local_only,
        }

        ramo_indexes[ramo] = build_indexes_for_ramo(merged)
        all_html_citations.extend(merged)

    # índices globais para fallback de metadados em no-match
    global_indexes = build_indexes_for_ramo(all_html_citations)

    # schema output: remove link_de_acesso, append link_1..link_3
    fieldnames = [c for c in original_fieldnames if c != "link_de_acesso" and c not in {"link_1", "link_2", "link_3"}]
    fieldnames.extend(["link_1", "link_2", "link_3"])
    for row in rows:
        row.pop("link_de_acesso", None)
        row["link_1"] = ""
        row["link_2"] = ""
        row["link_3"] = ""

    # auditorias
    linhas_sem_match: List[Dict[str, str]] = []
    linhas_multi: List[Dict[str, str]] = []
    linhas_truncadas: List[Dict[str, str]] = []
    datas_nao_interpretaveis: List[Dict[str, str]] = []
    csv_citation_entries: List[Dict[str, str]] = []

    # métricas
    counters = Counter()

    # 3/4/5/6: processar linhas
    for idx, row in enumerate(rows):
        ramo = normalize_ws(row.get("ramo", ""))
        if not ramo:
            # ramo vazio não terá índice; trata como sem match
            ramo = ""

        raw_date = normalize_ws(row.get("data_julgamento", ""))
        normalized_date, date_status = parse_date_to_mdy(raw_date, prefer_day_first=False)
        if date_status == "parsed":
            row["data_julgamento"] = normalized_date
            if normalized_date != raw_date:
                counters["date_changed"] += 1
        elif date_status == "empty":
            row["data_julgamento"] = ""
            counters["date_empty"] += 1
        else:
            # mantém original por segurança e audita
            row["data_julgamento"] = raw_date
            counters["date_unparsed"] += 1
            datas_nao_interpretaveis.append(
                {
                    "row_idx": str(idx),
                    "ID": normalize_ws(row.get("ID", "")),
                    "ramo": ramo,
                    "numero_processo": normalize_ws(row.get("numero_processo", "")),
                    "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                    "relator": normalize_ws(row.get("relator", "")),
                    "data_original": raw_date,
                    "texto_original_resumo": summarize_text(row.get("texto_original", "")),
                }
            )

        citation_pairs = extract_row_citations(row.get("texto_original", ""))
        row_citation_norms = [norm for _, norm in citation_pairs]

        for raw_cit, norm_cit in citation_pairs:
            csv_citation_entries.append(
                {
                    "ramo": ramo,
                    "row_idx": str(idx),
                    "ID": normalize_ws(row.get("ID", "")),
                    "numero_processo": normalize_ws(row.get("numero_processo", "")),
                    "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                    "relator": normalize_ws(row.get("relator", "")),
                    "data_julgamento": normalize_ws(row.get("data_julgamento", "")),
                    "citacao_csv_raw": raw_cit,
                    "citacao_csv_norm": norm_cit,
                    "texto_original_resumo": summarize_text(row.get("texto_original", "")),
                }
            )

        idx_data = ramo_indexes.get(ramo)
        if idx_data is None:
            # ramo sem mapeamento deve ser impossível devido validação.
            candidates: Set[str] = set()
            stage = "none"
            row_num = normalize_process_number(row.get("numero_processo", ""))
            row_rel = normalize_relator(row.get("relator", ""))
            row_date_norm, _ = parse_date_to_mdy(row.get("data_julgamento", ""), prefer_day_first=False)
            href_records: Dict[str, List[HtmlCitation]] = {}
        else:
            by_citation = idx_data["by_citation"]  # type: ignore[assignment]
            by_proc_rel_date = idx_data["by_proc_rel_date"]  # type: ignore[assignment]
            by_proc_rel = idx_data["by_proc_rel"]  # type: ignore[assignment]
            by_proc = idx_data["by_proc"]  # type: ignore[assignment]
            citation_keys = idx_data["citation_keys"]  # type: ignore[assignment]
            href_records = idx_data["href_records"]  # type: ignore[assignment]

            row_num = normalize_process_number(row.get("numero_processo", ""))
            row_rel = normalize_relator(row.get("relator", ""))
            row_date_norm, _ = parse_date_to_mdy(row.get("data_julgamento", ""), prefer_day_first=False)

            candidates = set()
            stage = "none"

            for cn in row_citation_norms:
                candidates.update(by_citation.get(cn, set()))
            if candidates:
                stage = "A_exato"
            else:
                candidates = containment_candidates(row_citation_norms, citation_keys, by_citation)
                if candidates:
                    stage = "B_contencao"
                else:
                    candidates, stage = metadata_candidates(
                        row_num,
                        row_rel,
                        row_date_norm,
                        by_proc_rel_date,
                        by_proc_rel,
                        by_proc,
                    )

            # Fallback global por metadados se ramo não retornar candidato.
            if not candidates:
                g_candidates, g_stage = metadata_candidates(
                    row_num,
                    row_rel,
                    row_date_norm,
                    global_indexes["by_proc_rel_date"],  # type: ignore[index]
                    global_indexes["by_proc_rel"],  # type: ignore[index]
                    global_indexes["by_proc"],  # type: ignore[index]
                )
                if g_candidates:
                    candidates = g_candidates
                    stage = f"G_{g_stage}"
                    href_records = global_indexes["href_records"]  # type: ignore[assignment]

        sorted_candidates: List[str] = []
        if candidates:
            candidates = narrow_candidates_by_metadata(
                candidates,
                row_num=row_num,
                row_rel=row_rel,
                row_date=row_date_norm,
                href_records=href_records,
            )
            ranked = [
                rank_href_candidate(
                    href=h,
                    stage=stage,
                    row_num=row_num,
                    row_rel=row_rel,
                    row_date=row_date_norm,
                    row_citations=row_citation_norms,
                    href_records=href_records,
                )
                for h in candidates
            ]
            # score desc, strong_bonus desc, href asc
            ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
            sorted_candidates = [r[2] for r in ranked]

        top_links = sorted_candidates[:top_k]
        if len(top_links) >= 1:
            row["link_1"] = top_links[0]
        if len(top_links) >= 2:
            row["link_2"] = top_links[1]
        if len(top_links) >= 3:
            row["link_3"] = top_links[2]

        counters["rows_total"] += 1
        if len(sorted_candidates) == 0:
            counters["rows_no_match"] += 1
            linhas_sem_match.append(
                {
                    "row_idx": str(idx),
                    "ID": normalize_ws(row.get("ID", "")),
                    "ramo": ramo,
                    "numero_processo": normalize_ws(row.get("numero_processo", "")),
                    "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                    "relator": normalize_ws(row.get("relator", "")),
                    "data_julgamento": normalize_ws(row.get("data_julgamento", "")),
                    "match_stage": stage,
                    "qtd_citacoes_extraidas": str(len(row_citation_norms)),
                    "citacao_principal": citation_pairs[0][0] if citation_pairs else "",
                    "texto_original_resumo": summarize_text(row.get("texto_original", "")),
                }
            )
        else:
            counters["rows_with_any_link"] += 1
            counters["total_candidate_links"] += len(sorted_candidates)

            if len(sorted_candidates) > 1:
                counters["rows_multi_candidates"] += 1
                linhas_multi.append(
                    {
                        "row_idx": str(idx),
                        "ID": normalize_ws(row.get("ID", "")),
                        "ramo": ramo,
                        "numero_processo": normalize_ws(row.get("numero_processo", "")),
                        "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                        "relator": normalize_ws(row.get("relator", "")),
                        "data_julgamento": normalize_ws(row.get("data_julgamento", "")),
                        "match_stage": stage,
                        "qtd_candidatos_total": str(len(sorted_candidates)),
                        "link_1": row["link_1"],
                        "link_2": row["link_2"],
                        "link_3": row["link_3"],
                        "hrefs_ordenados_todos": " | ".join(sorted_candidates),
                    }
                )
            if len(sorted_candidates) > top_k:
                counters["rows_truncated_topk"] += 1
                linhas_truncadas.append(
                    {
                        "row_idx": str(idx),
                        "ID": normalize_ws(row.get("ID", "")),
                        "ramo": ramo,
                        "numero_processo": normalize_ws(row.get("numero_processo", "")),
                        "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                        "relator": normalize_ws(row.get("relator", "")),
                        "data_julgamento": normalize_ws(row.get("data_julgamento", "")),
                        "qtd_candidatos_total": str(len(sorted_candidates)),
                        "top_k": str(top_k),
                        "link_1": row["link_1"],
                        "link_2": row["link_2"],
                        "link_3": row["link_3"],
                        "hrefs_descartados": " | ".join(sorted_candidates[top_k:]),
                    }
                )

    # 7) inserção opcional de casos ausentes (inclui diferenças live/local)
    casos_novos_inseridos: List[Dict[str, str]] = []
    if not args.skip_insert_missing:
        row_idx_proc: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        row_idx_pr: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
        row_idx_prd: Dict[Tuple[str, str, str, str], Set[int]] = defaultdict(set)
        ramo_numero_no_ramo_max: Dict[str, int] = defaultdict(int)

        max_id = 0
        for i, row in enumerate(rows):
            rid = normalize_ws(row.get("ID", ""))
            if rid.isdigit():
                max_id = max(max_id, int(rid))

            ramo = normalize_ws(row.get("ramo", ""))
            proc = normalize_process_number(row.get("numero_processo", ""))
            rel = normalize_relator(row.get("relator", ""))
            d, _ = parse_date_to_mdy(row.get("data_julgamento", ""), prefer_day_first=False)

            nr = normalize_ws(row.get("numero_no_ramo", ""))
            if nr.isdigit():
                ramo_numero_no_ramo_max[ramo] = max(ramo_numero_no_ramo_max[ramo], int(nr))

            if proc:
                row_idx_proc[(ramo, proc)].add(i)
            if proc and rel:
                row_idx_pr[(ramo, proc, rel)].add(i)
            if proc and rel and d:
                row_idx_prd[(ramo, proc, rel, d)].add(i)

        missing_groups: Dict[Tuple[str, str, str, str, str], List[HtmlCitation]] = defaultdict(list)

        for rec in all_html_citations:
            ramo = normalize_ws(rec.ramo)
            proc = rec.numero_norm
            rel = rec.relator_norm
            d = rec.data_norm_mdy
            tipo = normalize_ws(rec.tipo_de_processo)

            # Sem número de processo não é seguro inserir automaticamente.
            if not proc:
                continue

            represented = False
            if proc and rel and d and row_idx_prd.get((ramo, proc, rel, d)):
                represented = True
            elif proc and rel and row_idx_pr.get((ramo, proc, rel)):
                represented = True
            elif proc and row_idx_proc.get((ramo, proc)):
                represented = True

            if represented:
                continue

            case_key = (ramo, proc, rel, d, tipo)
            missing_groups[case_key].append(rec)

        # Seleciona 1 citação por caso (melhor href disponível) e cria linha nova.
        for case_key, recs in sorted(missing_groups.items(), key=lambda kv: kv[0]):
            ramo, proc, rel_norm, d, tipo = case_key

            def _rec_priority(rec: HtmlCitation) -> Tuple[int, int, int, str]:
                score = 0
                if rec.numero_norm and f"processoNumero={rec.numero_norm}" in rec.href:
                    score += 30
                if rec.source == "live":
                    score += 6
                if re.search(r"/pdf/\d+$", rec.href):
                    score += 3
                if rec.data_norm_mdy:
                    score += 2
                if rec.relator_norm:
                    score += 1
                return (-score, 0, 0, rec.href)

            best = sorted(recs, key=_rec_priority)[0]

            max_id += 1
            ramo_numero_no_ramo_max[ramo] += 1

            new_row = {k: "" for k in fieldnames}
            new_row["ID"] = str(max_id)
            new_row["numero_no_ramo"] = str(ramo_numero_no_ramo_max[ramo])
            new_row["ramo"] = ramo
            new_row["numero_processo"] = proc
            new_row["tipo_de_processo"] = tipo or best.tipo_de_processo
            new_row["relator"] = format_relator_display(best.relator_raw)
            new_row["data_julgamento"] = d
            new_row["texto_original"] = best.texto_citacao_raw
            new_row["contexto"] = "Incluído automaticamente por auditoria de omissões."
            new_row["link_1"] = best.href
            new_row["link_2"] = ""
            new_row["link_3"] = ""

            rows.append(new_row)
            counters["inserted_missing_rows"] += 1
            counters["rows_with_any_link"] += 1
            counters["total_candidate_links"] += 1

            # Atualiza índices para evitar inserção duplicada.
            new_idx = len(rows) - 1
            row_idx_proc[(ramo, proc)].add(new_idx)
            if rel_norm:
                row_idx_pr[(ramo, proc, rel_norm)].add(new_idx)
            if rel_norm and d:
                row_idx_prd[(ramo, proc, rel_norm, d)].add(new_idx)

            casos_novos_inseridos.append(
                {
                    "ID_novo": new_row["ID"],
                    "ramo": ramo,
                    "numero_processo": proc,
                    "tipo_de_processo": new_row["tipo_de_processo"],
                    "relator": new_row["relator"],
                    "data_julgamento": d,
                    "link_1": best.href,
                    "fonte_html": best.source,
                    "texto_citacao_raw": best.texto_citacao_raw,
                }
            )

        # Inclui citações inseridas na base de comparação de omissões.
        for item in casos_novos_inseridos:
            raw_cit = item["texto_citacao_raw"]
            csv_citation_entries.append(
                {
                    "ramo": item["ramo"],
                    "row_idx": "",
                    "ID": item["ID_novo"],
                    "numero_processo": item["numero_processo"],
                    "tipo_de_processo": item["tipo_de_processo"],
                    "relator": item["relator"],
                    "data_julgamento": item["data_julgamento"],
                    "citacao_csv_raw": raw_cit,
                    "citacao_csv_norm": normalize_citation_text(raw_cit),
                    "texto_original_resumo": summarize_text(raw_cit),
                }
            )

    # 8) auditoria de omissões HTML vs CSV (base metadados)
    omissoes_html_nao_no_csv: List[Dict[str, str]] = []
    csv_nao_no_html: List[Dict[str, str]] = []

    row_idx_proc: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    row_idx_pr: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
    row_idx_prd: Dict[Tuple[str, str, str, str], Set[int]] = defaultdict(set)
    for i, row in enumerate(rows):
        ramo = normalize_ws(row.get("ramo", ""))
        proc = normalize_process_number(row.get("numero_processo", ""))
        rel = normalize_relator(row.get("relator", ""))
        d, _ = parse_date_to_mdy(row.get("data_julgamento", ""), prefer_day_first=False)
        if proc:
            row_idx_proc[(ramo, proc)].add(i)
        if proc and rel:
            row_idx_pr[(ramo, proc, rel)].add(i)
        if proc and rel and d:
            row_idx_prd[(ramo, proc, rel, d)].add(i)

    html_idx_proc: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    html_idx_pr: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    html_idx_prd: Dict[Tuple[str, str, str, str], Set[str]] = defaultdict(set)
    for rec in all_html_citations:
        ramo = normalize_ws(rec.ramo)
        proc = rec.numero_norm
        rel = rec.relator_norm
        d = rec.data_norm_mdy
        if proc:
            html_idx_proc[(ramo, proc)].add(rec.href)
        if proc and rel:
            html_idx_pr[(ramo, proc, rel)].add(rec.href)
        if proc and rel and d:
            html_idx_prd[(ramo, proc, rel, d)].add(rec.href)

    # HTML não representado no CSV final
    seen_om: Set[Tuple[str, str, str, str, str]] = set()
    for rec in all_html_citations:
        ramo = normalize_ws(rec.ramo)
        proc = rec.numero_norm
        rel = rec.relator_norm
        d = rec.data_norm_mdy
        if not proc:
            continue

        represented = False
        if proc and rel and d and row_idx_prd.get((ramo, proc, rel, d)):
            represented = True
        elif proc and rel and row_idx_pr.get((ramo, proc, rel)):
            represented = True
        elif proc and row_idx_proc.get((ramo, proc)):
            represented = True
        if represented:
            continue

        dedupe = (ramo, proc, rel, d, rec.href)
        if dedupe in seen_om:
            continue
        seen_om.add(dedupe)
        omissoes_html_nao_no_csv.append(
            {
                "ramo": ramo,
                "texto_citacao_raw": rec.texto_citacao_raw,
                "texto_citacao_norm": rec.texto_citacao_norm,
                "href": rec.href,
                "numero_norm": proc,
                "relator_norm": rel,
                "data_norm_mdy": d,
                "fonte_html": rec.source,
            }
        )

    # CSV final não representado no HTML
    for i, row in enumerate(rows):
        ramo = normalize_ws(row.get("ramo", ""))
        proc = normalize_process_number(row.get("numero_processo", ""))
        rel = normalize_relator(row.get("relator", ""))
        d, _ = parse_date_to_mdy(row.get("data_julgamento", ""), prefer_day_first=False)
        if not proc:
            continue

        represented = False
        if proc and rel and d and html_idx_prd.get((ramo, proc, rel, d)):
            represented = True
        elif proc and rel and html_idx_pr.get((ramo, proc, rel)):
            represented = True
        elif proc and html_idx_proc.get((ramo, proc)):
            represented = True
        if represented:
            continue

        csv_nao_no_html.append(
            {
                "ramo": ramo,
                "ID": normalize_ws(row.get("ID", "")),
                "numero_processo": normalize_ws(row.get("numero_processo", "")),
                "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                "relator": normalize_ws(row.get("relator", "")),
                "data_julgamento": normalize_ws(row.get("data_julgamento", "")),
                "citacao_csv_raw": summarize_text(row.get("texto_original", "")),
                "citacao_csv_norm": "",
                "texto_original_resumo": summarize_text(row.get("texto_original", "")),
            }
        )

    # salvar output final
    write_csv(csv_out, rows, fieldnames)

    # salvar auditorias
    sem_match_path = Path(f"{audit_prefix}_linhas_sem_match.csv")
    multi_path = Path(f"{audit_prefix}_linhas_com_multiplos_candidatos.csv")
    trunc_path = Path(f"{audit_prefix}_linhas_truncadas_top3.csv")
    omissoes_path = Path(f"{audit_prefix}_omissoes_html_nao_no_csv.csv")
    csv_nao_html_path = Path(f"{audit_prefix}_csv_nao_no_html.csv")
    datas_path = Path(f"{audit_prefix}_datas_nao_interpretaveis.csv")
    novos_path = Path(f"{audit_prefix}_casos_novos_inseridos.csv")
    live_diff_path = Path(f"{audit_prefix}_comparativo_live_local.csv")
    resumo_path = Path(f"{audit_prefix}_resumo.txt")

    write_csv(
        sem_match_path,
        linhas_sem_match,
        [
            "row_idx",
            "ID",
            "ramo",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_julgamento",
            "match_stage",
            "qtd_citacoes_extraidas",
            "citacao_principal",
            "texto_original_resumo",
        ],
    )
    write_csv(
        multi_path,
        linhas_multi,
        [
            "row_idx",
            "ID",
            "ramo",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_julgamento",
            "match_stage",
            "qtd_candidatos_total",
            "link_1",
            "link_2",
            "link_3",
            "hrefs_ordenados_todos",
        ],
    )
    write_csv(
        trunc_path,
        linhas_truncadas,
        [
            "row_idx",
            "ID",
            "ramo",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_julgamento",
            "qtd_candidatos_total",
            "top_k",
            "link_1",
            "link_2",
            "link_3",
            "hrefs_descartados",
        ],
    )
    write_csv(
        omissoes_path,
        omissoes_html_nao_no_csv,
        [
            "ramo",
            "texto_citacao_raw",
            "texto_citacao_norm",
            "href",
            "numero_norm",
            "relator_norm",
            "data_norm_mdy",
            "fonte_html",
        ],
    )
    write_csv(
        csv_nao_html_path,
        csv_nao_no_html,
        [
            "ramo",
            "ID",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_julgamento",
            "citacao_csv_raw",
            "citacao_csv_norm",
            "texto_original_resumo",
        ],
    )
    write_csv(
        datas_path,
        datas_nao_interpretaveis,
        [
            "row_idx",
            "ID",
            "ramo",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_original",
            "texto_original_resumo",
        ],
    )
    write_csv(
        novos_path,
        casos_novos_inseridos,
        [
            "ID_novo",
            "ramo",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_julgamento",
            "link_1",
            "fonte_html",
            "texto_citacao_raw",
        ],
    )
    write_csv(
        live_diff_path,
        [
            {
                "ramo": ramo,
                "arquivo_html": str(ramo_to_html[ramo]),
                "citacoes_local": str(stats["local_citations"]),
                "citacoes_usadas": str(stats["merged_citations"]),
                "citacoes_somente_live": str(stats["live_only"]),
                "citacoes_somente_local": str(stats["local_only"]),
            }
            for ramo, stats in sorted(live_diff_by_ramo.items())
        ],
        [
            "ramo",
            "arquivo_html",
            "citacoes_local",
            "citacoes_usadas",
            "citacoes_somente_live",
            "citacoes_somente_local",
        ],
    )

    # resumo
    live_only_total = sum(
        max(0, stats["live_only"]) for stats in live_diff_by_ramo.values()
    )
    resumo_lines = [
        f"CSV entrada: {csv_in}",
        f"CSV saída: {csv_out}",
        f"HTML glob: {args.html_glob}",
        f"HTMLs encontrados: {len(html_paths)}",
        f"Ramos mapeados: {len(ramos)}",
        f"Top-k links: {top_k}",
        f"Consulta live habilitada: {'não' if args.skip_live else 'sim'}",
        "",
        f"Linhas totais (originais): {counters['rows_total']}",
        f"Linhas inseridas por omissão: {counters['inserted_missing_rows']}",
        f"Linhas totais (final): {len(rows)}",
        f"Linhas com >=1 link: {counters['rows_with_any_link']}",
        f"Linhas sem match: {counters['rows_no_match']}",
        f"Linhas com múltiplos candidatos: {counters['rows_multi_candidates']}",
        f"Linhas truncadas top-k: {counters['rows_truncated_topk']}",
        f"Total de candidatos (antes do corte): {counters['total_candidate_links']}",
        "",
        f"Datas alteradas por normalização: {counters['date_changed']}",
        f"Datas vazias: {counters['date_empty']}",
        f"Datas não interpretáveis: {counters['date_unparsed']}",
        "",
        f"Citações somente live (todas as páginas): {live_only_total}",
        f"Omissões HTML não no CSV: {len(omissoes_html_nao_no_csv)}",
        f"Citações CSV não no HTML: {len(csv_nao_no_html)}",
        "",
        f"Arquivos de auditoria:",
        f"- {sem_match_path}",
        f"- {multi_path}",
        f"- {trunc_path}",
        f"- {omissoes_path}",
        f"- {csv_nao_html_path}",
        f"- {datas_path}",
        f"- {novos_path}",
        f"- {live_diff_path}",
    ]

    resumo_path.write_text("\n".join(resumo_lines) + "\n", encoding="utf-8")
    print("\n".join(resumo_lines))


if __name__ == "__main__":
    main()
