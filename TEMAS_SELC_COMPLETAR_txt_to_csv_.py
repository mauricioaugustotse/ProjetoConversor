#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detecta acréscimos em HTMLs de Temas Selecionados e gera CSV complementar.

Saída principal:
- temas_selec_TSE_complementar.csv

Auditorias:
- <prefix>_novos_detectados.csv
- <prefix>_ja_existentes_descartados.csv
- <prefix>_incompletos_sem_chave.csv
- <prefix>_api_falhas.csv
- <prefix>_resumo.txt
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import html
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI


EXPECTED_SCHEMA = [
    "ID",
    "numero_no_ramo",
    "tema",
    "punchline",
    "tese",
    "bullet_points",
    "data_julgamento",
    "numero_processo",
    "ramo",
    "relator",
    "subramo",
    "texto_original",
    "contexto",
    "tipo_de_processo",
    "link_1",
    "link_2",
    "link_3",
]

SJUR_PREFIX = "sjur-servicos.tse.jus.br/sjur-servicos/rest/download/pdf/"
MIN_TEXT_LEN = 30

CIT_TOKEN = (
    r"(?:Ac\.|Res\.|EDcl|Embargos?|AgR(?:-[A-Za-z]+)?|MS|REspE?|REspe|REsp|"
    r"RO-?El|RMS|RCEd|AE|TutCautAnt|MC|AAg|Rec\.?)"
)
HAS_CITATION_RE = re.compile(CIT_TOKEN, re.IGNORECASE)
ANCHOR_RE = re.compile(
    r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
P_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
CANONICAL_RE = re.compile(
    r"<link\s+rel=[\"']canonical[\"']\s+href=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)

# Cobertura de classes processuais usada para extrair tipo+número.
PROCESS_CLASS_TOKEN = (
    r"(?:AgR|Ag|AREspE|AREspEl|REspEl|REspE?|REspe|REsp|RO-?El|RMS|RCEd|RCED|"
    r"AE|TutCautAnt|MC|AAg|Rec\.?|Cta|Pet|PA|Inst\.?|MS|AI|RO|RE|EDcl)"
)
PROCESS_CLASS_AND_NUM_RE = re.compile(
    rf"\b(({PROCESS_CLASS_TOKEN})(?:-[A-Za-z]+)*)\s*(?:n[ºo°.]?\s*)?(\d[\d./-]*)",
    re.IGNORECASE,
)
RELATOR_RE = re.compile(
    r"\brel\.\s*(?:Min\.|Des\.|Juiz(?:a)?|in\.)?\s*([^\).;]+)",
    re.IGNORECASE,
)
DATE_TEXT_RE = re.compile(r"\bde\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.IGNORECASE)
DECISAO_DATA_RE = re.compile(r"decisaoData=(\d{8})", re.IGNORECASE)
PROC_FROM_HREF_RE = re.compile(r"processoNumero=(\d+)", re.IGNORECASE)
TIPO_FROM_HREF_RE = re.compile(r"processoClasse=([A-Za-z]+)", re.IGNORECASE)
CITATION_LIKE_RE = re.compile(rf"^\s*\({CIT_TOKEN}", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    ramo: str
    source: str  # local | live
    href: str
    citacao_raw: str
    citacao_norm: str
    numero_norm: str
    tipo: str
    relator: str
    data_mdy: str
    texto_original: str
    subramo: str


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_nul(text: str) -> str:
    return (text or "").replace("\x00", "")


def strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_key(text: str) -> str:
    s = strip_accents((text or "").lower())
    s = s.replace("_", ": ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s:]", " ", s)
    return normalize_ws(s)


def normalize_token(text: str) -> str:
    s = strip_accents((text or "").lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return normalize_ws(s)


def normalize_tipo(tipo: str) -> str:
    s = normalize_ws(clean_nul(tipo)).replace("–", "-").replace("—", "-")
    s = s.rstrip(".")
    return s


def normalize_tipo_key(tipo: str) -> str:
    return normalize_tipo(tipo).upper()


def normalize_process_number(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    digits = digits.lstrip("0")
    return digits or ""


def clean_html_text(raw: str) -> str:
    txt = html.unescape(TAG_RE.sub(" ", raw or "")).replace("\xa0", " ")
    return normalize_ws(clean_nul(txt))


def parse_date_to_mdy(raw_date: str, prefer_day_first: bool) -> Tuple[str, str]:
    """
    Retorna (data_mdy, status) com status in {'parsed','empty','unparsed'}.
    """
    raw = normalize_ws((raw_date or "").replace("º", ""))
    if not raw:
        return "", "empty"

    m = re.fullmatch(r"(\d{1,2})([./-])(\d{1,2})\2(\d{2,4})", raw)
    if m:
        a_s, sep, b_s, y_s = m.groups()
        a = int(a_s)
        b = int(b_s)
        y = int(y_s)
        if len(y_s) == 2:
            y = 2000 + y if y <= 30 else 1900 + y

        if sep in ".-":
            d, mm = a, b
        else:
            if a > 12 and b <= 12:
                d, mm = a, b
            elif b > 12 and a <= 12:
                mm, d = a, b
            elif a > 12 and b > 12:
                return "", "unparsed"
            else:
                if prefer_day_first:
                    d, mm = a, b
                else:
                    mm, d = a, b
        try:
            dt.date(y, mm, d)
        except ValueError:
            return "", "unparsed"
        return f"{mm}/{d}/{y}", "parsed"

    ymd = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", raw.replace("-", "/").replace(".", "/"))
    if ymd:
        y, mm, d = map(int, ymd.groups())
        try:
            dt.date(y, mm, d)
        except ValueError:
            return "", "unparsed"
        return f"{mm}/{d}/{y}", "parsed"

    return "", "unparsed"


def extract_date_from_href(href: str) -> str:
    m = DECISAO_DATA_RE.search(href or "")
    if not m:
        return ""
    yyyymmdd = m.group(1)
    y = int(yyyymmdd[:4])
    mm = int(yyyymmdd[4:6])
    d = int(yyyymmdd[6:8])
    try:
        dt.date(y, mm, d)
    except ValueError:
        return ""
    return f"{mm}/{d}/{y}"


def extract_date_from_text(citation_text: str, href: str) -> str:
    m = DATE_TEXT_RE.search(citation_text or "")
    if m:
        date_mdy, status = parse_date_to_mdy(m.group(1), prefer_day_first=True)
        if status == "parsed":
            return date_mdy
    return extract_date_from_href(href)


def extract_relator(citation_text: str) -> str:
    m = RELATOR_RE.search(citation_text or "")
    if not m:
        return ""
    rel = normalize_ws(m.group(1))
    if not rel:
        return ""
    if re.match(r"^(Min\.|Des\.|Juiz|Juíza)\s+", rel, re.IGNORECASE):
        return rel
    return f"Min. {rel}"


def extract_tipo_from_href(href: str) -> str:
    m = TIPO_FROM_HREF_RE.search(href or "")
    if not m:
        return ""
    return normalize_tipo(m.group(1))


def extract_tipo_from_text(citation_text: str, href: str) -> str:
    text = normalize_ws(citation_text or "")
    m = PROCESS_CLASS_AND_NUM_RE.search(text)
    if m:
        return normalize_tipo(m.group(1))

    # Padrão "no AgR-REspEl nº 968" (token começa por letra).
    m2 = re.search(
        r"\b(?:no|na|nos|nas)\s+([A-Za-z][A-Za-z.-]*(?:-[A-Za-z][A-Za-z.-]*)*)\s*(?:n[ºo°.]?\s*)?\d",
        text,
        re.IGNORECASE,
    )
    if m2:
        cand = normalize_tipo(m2.group(1))
        if normalize_tipo_key(cand) not in {"AC", "RES"}:
            return cand

    from_href = extract_tipo_from_href(href)
    if from_href:
        return from_href

    return ""


def extract_process_from_text(citation_text: str, href: str) -> str:
    href_num = PROC_FROM_HREF_RE.search(href or "")
    if href_num:
        return normalize_process_number(href_num.group(1))

    text = normalize_ws(citation_text or "")
    class_matches = list(PROCESS_CLASS_AND_NUM_RE.finditer(text))
    if class_matches:
        nums = [normalize_process_number(m.group(3)) for m in class_matches]
        nums = [n for n in nums if n]
        if nums:
            return max(nums, key=len)

    # nº <numero>
    n_matches = list(re.finditer(r"\bn[ºo°.]?\s*(\d[\d./-]*)", text, re.IGNORECASE))
    filtered = []
    for m in n_matches:
        token = m.group(1)
        if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", token):
            continue
        pn = normalize_process_number(token)
        if pn:
            filtered.append(pn)
    if filtered:
        return max(filtered, key=len)

    # fallback geral, excluindo datas.
    tokens = re.findall(r"\d[\d./-]*", text)
    tokens = [
        t for t in tokens
        if not re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", t)
    ]
    if not tokens:
        return ""
    return normalize_process_number(max(tokens, key=lambda x: len(re.sub(r"\D", "", x))))


def normalize_citation_text(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("−", "-")
    cleaned = strip_accents(cleaned.lower())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return normalize_ws(cleaned)


def extract_canonical_url(raw_html: str) -> str:
    m = CANONICAL_RE.search(raw_html or "")
    return normalize_ws(m.group(1)) if m else ""


def fetch_url_text(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def is_housekeeping_text(text: str) -> bool:
    t = normalize_ws(text)
    if not t:
        return True
    if re.match(r"^(Atualizado em\b|NE\s*:|Vide\b|Nota explicativa\b)", t, re.IGNORECASE):
        return True
    return False


def is_substantive_text(text: str) -> bool:
    t = normalize_ws(text)
    if not t:
        return False
    if len(t) < MIN_TEXT_LEN:
        return False
    if CITATION_LIKE_RE.search(t):
        return False
    if is_housekeeping_text(t):
        return False
    return True


class HierarchyParser(HTMLParser):
    def __init__(self, base_path: str):
        super().__init__(convert_charrefs=False)
        self.base_path = (base_path or "").strip("/")
        self.in_a = False
        self.a_href = ""
        self.a_text: List[str] = []
        self.labels_by_depth: Dict[int, str] = {}
        self.href_to_paths: Dict[str, List[Tuple[str, ...]]] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]):
        if tag.lower() != "a":
            return
        self.in_a = True
        self.a_href = dict(attrs).get("href", "") or ""
        self.a_text = []

    def handle_data(self, data: str):
        if self.in_a:
            self.a_text.append(data)

    def handle_endtag(self, tag: str):
        if tag.lower() != "a" or not self.in_a:
            return
        href = normalize_ws(html.unescape(self.a_href))
        text = clean_html_text("".join(self.a_text))
        self.in_a = False
        self.a_href = ""
        self.a_text = []
        if not href or not text:
            return

        if SJUR_PREFIX in href:
            labels = tuple(
                self.labels_by_depth[d]
                for d in sorted(self.labels_by_depth)
                if self.labels_by_depth.get(d)
            )
            self.href_to_paths.setdefault(href, []).append(labels)
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


def sanitize_subramo_item(text: str) -> str:
    s = normalize_ws(text)
    s = re.sub(r"\s*,\s*", " - ", s)
    s = re.sub(r"\s*;\s*", " - ", s)
    return normalize_ws(s)


def dedupe_preserve(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        s = sanitize_subramo_item(v)
        if not s:
            continue
        key = normalize_token(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def extract_hierarchy_map(raw_html: str) -> Dict[str, List[str]]:
    canonical = extract_canonical_url(raw_html)
    base_path = urlparse(canonical).path
    parser = HierarchyParser(base_path=base_path)
    parser.feed(raw_html)
    out: Dict[str, List[str]] = {}
    for href, paths in parser.href_to_paths.items():
        if not paths:
            continue
        # preferência por caminho mais profundo; empate por frequência.
        counts: Dict[Tuple[str, ...], int] = {}
        for p in paths:
            counts[p] = counts.get(p, 0) + 1
        best = max(counts.items(), key=lambda kv: (len(kv[0]), kv[1]))[0]
        labels = [lbl for lbl in best if not re.match(r"^\s*Parte\b", lbl, re.IGNORECASE)]
        if not labels:
            labels = list(best)
        out[href] = dedupe_preserve(labels)[:3]
    return out


def choose_preceding_paragraph(
    paragraphs: Sequence[str],
    p_end_positions: Sequence[int],
    anchor_start: int,
) -> str:
    idx = bisect.bisect_left(p_end_positions, anchor_start) - 1
    if idx < 0:
        return ""
    candidates: List[str] = []
    j = idx
    while j >= 0 and len(candidates) < 8:
        ptxt = paragraphs[j]
        if ptxt:
            candidates.append(ptxt)
        j -= 1
    if not candidates:
        return ""
    best = candidates[0]
    if is_housekeeping_text(best) or CITATION_LIKE_RE.search(best):
        for cand in candidates[1:]:
            if not is_housekeeping_text(cand) and not CITATION_LIKE_RE.search(cand):
                best = cand
                break
    return best


def extract_candidates_from_html(
    ramo: str,
    raw_html: str,
    source: str,
) -> List[Candidate]:
    hierarchy_map = extract_hierarchy_map(raw_html)
    p_matches = list(P_RE.finditer(raw_html))
    p_end_positions = [m.end() for m in p_matches]
    paragraphs = [clean_html_text(m.group(1)) for m in p_matches]

    out: List[Candidate] = []
    for m in ANCHOR_RE.finditer(raw_html):
        href_raw = m.group(1)
        inner_html = m.group(2)
        href = normalize_ws(html.unescape(href_raw))
        if SJUR_PREFIX not in href:
            continue

        citation_raw = clean_html_text(inner_html)
        if not citation_raw:
            continue
        if not HAS_CITATION_RE.search(citation_raw):
            continue

        numero_norm = extract_process_from_text(citation_raw, href)
        tipo = extract_tipo_from_text(citation_raw, href)
        relator = extract_relator(citation_raw)
        data_mdy = extract_date_from_text(citation_raw, href)
        texto_original = choose_preceding_paragraph(paragraphs, p_end_positions, m.start())
        subramo = ", ".join(hierarchy_map.get(href, [])[:3])

        out.append(
            Candidate(
                ramo=ramo,
                source=source,
                href=href,
                citacao_raw=citation_raw,
                citacao_norm=normalize_citation_text(citation_raw),
                numero_norm=numero_norm,
                tipo=tipo,
                relator=relator,
                data_mdy=data_mdy,
                texto_original=texto_original,
                subramo=subramo,
            )
        )
    return out


def detect_html_label_candidates(html_path: Path) -> List[str]:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    labels: List[str] = [html_path.name.replace(" — Temas Selecionados.html", "").strip()]

    m_title = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if m_title:
        title = clean_html_text(m_title.group(1))
        labels.append(title)
        if "—" in title:
            labels.append(title.split("—")[0].strip())

    m_h1 = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.IGNORECASE | re.DOTALL)
    if m_h1:
        labels.append(clean_html_text(m_h1.group(1)))

    return dedupe_preserve(labels)


def map_ramos_to_html(ramos: Sequence[str], html_paths: Sequence[Path]) -> Dict[str, Path]:
    if not html_paths:
        raise RuntimeError("Nenhum HTML encontrado para mapeamento.")

    labels_by_path: Dict[Path, List[str]] = {
        p: detect_html_label_candidates(p) for p in html_paths
    }

    fallback_target = normalize_key("Enfrentamento à desinformação eleitoral")
    mapping: Dict[str, Path] = {}
    used: Set[Path] = set()

    for ramo in ramos:
        rk = normalize_key(ramo)
        if rk == fallback_target:
            forced: Path | None = None
            for p in html_paths:
                name_k = normalize_key(p.name)
                if "repositorio" in name_k and "desinformacao eleitoral" in name_k:
                    forced = p
                    break
            if forced is None:
                raise RuntimeError(f"Fallback não encontrou HTML para ramo '{ramo}'.")
            if forced in used:
                raise RuntimeError(f"Fallback gerou duplicidade de HTML: {forced.name}")
            mapping[ramo] = forced
            used.add(forced)
            continue

        best_score = -1
        best_path: Path | None = None
        for p in html_paths:
            score = 0
            for lbl in labels_by_path[p]:
                lk = normalize_key(lbl)
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
            raise RuntimeError(f"Não foi possível mapear ramo '{ramo}' com confiança.")
        if best_path in used:
            raise RuntimeError(f"Mapeamento duplicado detectado para HTML '{best_path.name}'.")
        mapping[ramo] = best_path
        used.add(best_path)

    missing = [r for r in ramos if r not in mapping]
    if missing:
        raise RuntimeError(f"Ramos sem HTML mapeado: {missing}")
    return mapping


def load_prompt_from_temas_script(script_path: Path) -> str:
    if not script_path.exists():
        raise FileNotFoundError(f"Script base não encontrado: {script_path}")
    txt = script_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"PROMPT_EXTRACAO\s*=\s*([\"']{3})([\s\S]*?)\1", txt, re.MULTILINE)
    if not m:
        raise RuntimeError("Não foi possível extrair PROMPT_EXTRACAO de TEMAS_SELC_txt_to_csv.py")
    return m.group(2)


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {}
    json_str = ""
    if "```json" in raw_text:
        try:
            json_str = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
        except IndexError:
            json_str = ""
    if not json_str:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            json_str = raw_text[start : end + 1]
    if not json_str:
        return {}
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}

    def normalize_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k).strip().lower(): normalize_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize_keys(x) for x in obj]
        return obj

    return normalize_keys(data)


def normalize_subramo_value(value: Any, max_items: int = 3) -> str:
    parts: List[str] = []

    def _walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            s = normalize_ws(clean_nul(v))
            if not s:
                return
            chunks = [normalize_ws(x) for x in re.split(r"[,;\n\r|]+", s) if normalize_ws(x)]
            parts.extend(chunks)
            return
        if isinstance(v, list):
            for x in v:
                _walk(x)
            return
        _walk(str(v))

    _walk(value)
    out: List[str] = []
    seen: Set[str] = set()
    for p in parts:
        p = sanitize_subramo_item(p)
        key = normalize_token(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= max_items:
            break
    return ", ".join(out)


def _next_non_empty(lines: Sequence[str], idx: int) -> int:
    for j in range(idx + 1, len(lines)):
        if lines[j].strip():
            return j
    return -1


def normalize_bullets_for_notion(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        lines = [normalize_ws(clean_nul(str(x))) for x in value if normalize_ws(clean_nul(str(x)))]
    else:
        raw = clean_nul(str(value))
        lines = [normalize_ws(ln) for ln in raw.splitlines() if normalize_ws(ln)]

    if not lines:
        return ""

    # normaliza prefixo bullet
    normalized: List[str] = []
    for ln in lines:
        if not ln.startswith("•"):
            ln = f"• {ln.lstrip('- ').strip()}"
        normalized.append(ln)

    token = "__BSEP__"
    out: List[str] = []
    for i, line in enumerate(normalized):
        work = line.rstrip()
        next_idx = _next_non_empty(normalized, i)
        next_is_bullet = next_idx >= 0 and normalized[next_idx].lstrip().startswith("•")

        # preserva separador inline ", •"
        work = re.sub(r",(?=\s*•)", token, work)
        # preserva separador final antes do próximo bullet
        if next_is_bullet and work.strip():
            if re.search(r",\s*$", work):
                work = re.sub(r",\s*$", token, work)
            elif re.search(r"\s-\s*$", work):
                work = re.sub(r"\s-\s*$", token, work)

        work = re.sub(r"\s*,\s*", " - ", work)
        work = work.replace(token, ",")
        out.append(work)
    return "\n".join(out)


def normalize_api_field(col: str, value: Any) -> str:
    if col == "bullet_points":
        return normalize_bullets_for_notion(value)
    if col == "subramo":
        return normalize_subramo_value(value, max_items=3)
    if value is None:
        return ""
    return normalize_ws(clean_nul(str(value)))


def parse_api_analytics(data: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in ("tema", "contexto", "bullet_points", "tese", "punchline"):
        out[col] = normalize_api_field(col, data.get(col, ""))
    return out


def call_api_single(
    client: OpenAI,
    model: str,
    prompt_base: str,
    texto_julgado: str,
    retries: int,
    retry_wait_s: float,
) -> Tuple[bool, Dict[str, str], str]:
    messages = [
        {
            "role": "system",
            "content": "Você é um assistente de jurimetria especializado em jurisprudência eleitoral brasileira.",
        },
        {
            "role": "user",
            "content": prompt_base.format(texto_julgado=texto_julgado),
        },
    ]
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = parse_json_response(content)
            fields = parse_api_analytics(parsed)
            req = ("tema", "contexto", "bullet_points", "tese", "punchline")
            if all(normalize_ws(fields.get(k, "")) for k in req):
                return True, fields, ""
            last_error = "Resposta da API incompleta para colunas analíticas."
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if attempt < retries:
            time.sleep(retry_wait_s * attempt)
    return False, {}, last_error or "Falha desconhecida na API."


def write_csv_atomic(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_dict_csv(path: Path, rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


def candidate_key(c: Candidate) -> Tuple[str, str, str, str]:
    return (
        normalize_key(c.ramo),
        normalize_process_number(c.numero_norm),
        normalize_ws(c.data_mdy),
        normalize_tipo_key(c.tipo),
    )


def candidate_has_full_key(c: Candidate) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if not normalize_process_number(c.numero_norm):
        missing.append("numero_processo")
    if not normalize_ws(c.data_mdy):
        missing.append("data_julgamento")
    if not normalize_tipo_key(c.tipo):
        missing.append("tipo_de_processo")
    return (len(missing) == 0), missing


def candidate_rank(c: Candidate) -> Tuple[int, int, int, int, str]:
    return (
        2 if c.source == "local" else 1,
        1 if is_substantive_text(c.texto_original) else 0,
        1 if normalize_ws(c.relator) else 0,
        len(normalize_ws(c.texto_original)),
        c.href,
    )


def ensure_schema(fieldnames: Sequence[str]) -> None:
    if list(fieldnames) != EXPECTED_SCHEMA:
        raise RuntimeError(
            "Schema do CSV-base divergente do esperado.\n"
            f"Esperado: {EXPECTED_SCHEMA}\n"
            f"Encontrado: {list(fieldnames)}"
        )


def build_audit_prefix(user_prefix: str) -> str:
    if user_prefix:
        return user_prefix
    return f"auditoria_temas_complementar_{dt.date.today().strftime('%Y%m%d')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera temas_selec_TSE_complementar.csv com acréscimos detectados nos HTMLs."
    )
    parser.add_argument("--base-csv", default="temas_selec_TSE_all_5.csv")
    parser.add_argument("--html-glob", default="* — Temas Selecionados.html")
    parser.add_argument("--csv-out", default="temas_selec_TSE_complementar.csv")
    parser.add_argument("--audit-prefix", default="")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait-s", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--temas-script", default="TEMAS_SELC_txt_to_csv.py")
    parser.add_argument("--max-new-rows", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--use-live-fallback", action="store_true", default=True)
    parser.add_argument("--no-use-live-fallback", dest="use_live_fallback", action="store_false")
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    load_dotenv()
    args = parse_args()

    base_csv = Path(args.base_csv)
    csv_out = Path(args.csv_out)
    html_paths = sorted(Path(".").glob(args.html_glob))
    audit_prefix = build_audit_prefix(args.audit_prefix)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(f"{args.csv_out}.checkpoint.json")

    if not base_csv.exists():
        raise FileNotFoundError(f"CSV-base não encontrado: {base_csv}")
    if not html_paths:
        raise RuntimeError(f"Nenhum HTML encontrado com glob: {args.html_glob}")

    prompt_base = load_prompt_from_temas_script(Path(args.temas_script))

    # Resume
    if checkpoint.exists() and csv_out.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        fieldnames = state.get("fieldnames", [])
        rows_out = state.get("rows_out", [])
        novos_detectados = state.get("novos_detectados", [])
        descartados = state.get("ja_existentes_descartados", [])
        incompletos = state.get("incompletos_sem_chave", [])
        api_falhas = state.get("api_falhas", [])
        done_indices = set(int(x) for x in state.get("done_indices", []))
        source_stats = state.get("source_stats", {})
        print(f"[resume] checkpoint carregado: done={len(done_indices)} de {len(rows_out)}")
    else:
        with base_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            base_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        ensure_schema(fieldnames)

        # baseline
        max_id = 0
        max_nr_by_ramo: Dict[str, int] = {}
        base_key_set: Set[Tuple[str, str, str, str]] = set()
        base_ramo_href_set: Set[Tuple[str, str]] = set()
        ramos: Set[str] = set()
        for row in base_rows:
            rid = normalize_ws(row.get("ID", ""))
            if rid.isdigit():
                max_id = max(max_id, int(rid))

            ramo = normalize_ws(row.get("ramo", ""))
            if ramo:
                ramos.add(ramo)
            nr = normalize_ws(row.get("numero_no_ramo", ""))
            if ramo and nr.isdigit():
                max_nr_by_ramo[ramo] = max(max_nr_by_ramo.get(ramo, 0), int(nr))

            num = normalize_process_number(row.get("numero_processo", ""))
            tipo = normalize_tipo_key(row.get("tipo_de_processo", ""))
            data_mdy, status = parse_date_to_mdy(row.get("data_julgamento", ""), prefer_day_first=False)
            if status == "parsed":
                data = data_mdy
            else:
                data = normalize_ws(row.get("data_julgamento", ""))
            if ramo and num and data and tipo:
                base_key_set.add((normalize_key(ramo), num, data, tipo))
            for col in ("link_1", "link_2", "link_3"):
                href = normalize_ws(row.get(col, ""))
                if href:
                    base_ramo_href_set.add((normalize_key(ramo), href))

        ramo_to_html = map_ramos_to_html(sorted(ramos), html_paths)

        # extrai candidatos local + live
        all_candidates: List[Candidate] = []
        source_stats = {
            "html_local_count": len(html_paths),
            "live_pages_ok": 0,
            "live_pages_fail": 0,
            "candidates_local": 0,
            "candidates_live": 0,
        }

        for ramo, hpath in sorted(ramo_to_html.items(), key=lambda x: x[0]):
            local_raw = hpath.read_text(encoding="utf-8", errors="ignore")
            local_cands = extract_candidates_from_html(ramo=ramo, raw_html=local_raw, source="local")
            all_candidates.extend(local_cands)
            source_stats["candidates_local"] += len(local_cands)

            if args.use_live_fallback:
                canonical = extract_canonical_url(local_raw)
                if canonical:
                    try:
                        live_raw = fetch_url_text(canonical)
                        live_cands = extract_candidates_from_html(ramo=ramo, raw_html=live_raw, source="live")
                        all_candidates.extend(live_cands)
                        source_stats["candidates_live"] += len(live_cands)
                        source_stats["live_pages_ok"] += 1
                    except Exception:
                        source_stats["live_pages_fail"] += 1

        # dedupe interno por chave principal
        grouped: Dict[Tuple[str, str, str, str], List[Candidate]] = {}
        incompletos: List[Dict[str, Any]] = []
        descartados: List[Dict[str, Any]] = []
        novos_raw: List[Candidate] = []

        for cand in all_candidates:
            ok_key, missing_fields = candidate_has_full_key(cand)
            if not ok_key:
                incompletos.append(
                    {
                        "ramo": cand.ramo,
                        "href": cand.href,
                        "source": cand.source,
                        "citacao_raw": cand.citacao_raw,
                        "numero_processo": cand.numero_norm,
                        "tipo_de_processo": cand.tipo,
                        "data_julgamento": cand.data_mdy,
                        "missing_fields": ", ".join(missing_fields),
                        "motivo": "Chave incompleta",
                    }
                )
                continue
            if not is_substantive_text(cand.texto_original):
                incompletos.append(
                    {
                        "ramo": cand.ramo,
                        "href": cand.href,
                        "source": cand.source,
                        "citacao_raw": cand.citacao_raw,
                        "numero_processo": cand.numero_norm,
                        "tipo_de_processo": cand.tipo,
                        "data_julgamento": cand.data_mdy,
                        "missing_fields": "",
                        "motivo": "texto_original sem conteúdo substantivo",
                    }
                )
                continue
            key = candidate_key(cand)
            grouped.setdefault(key, []).append(cand)

        selected: List[Candidate] = []
        for key, cands in grouped.items():
            best = sorted(cands, key=candidate_rank, reverse=True)[0]
            selected.append(best)

        # aplica regra incremental por ramo+proc+data+tipo
        for cand in selected:
            ramo_norm = normalize_key(cand.ramo)
            if (ramo_norm, cand.href) in base_ramo_href_set:
                descartados.append(
                    {
                        "ramo": cand.ramo,
                        "numero_processo": cand.numero_norm,
                        "tipo_de_processo": cand.tipo,
                        "data_julgamento": cand.data_mdy,
                        "href": cand.href,
                        "source": cand.source,
                        "motivo": "Href já existente no CSV-base para o mesmo ramo",
                    }
                )
                continue
            key = candidate_key(cand)
            if key in base_key_set:
                descartados.append(
                    {
                        "ramo": cand.ramo,
                        "numero_processo": cand.numero_norm,
                        "tipo_de_processo": cand.tipo,
                        "data_julgamento": cand.data_mdy,
                        "href": cand.href,
                        "source": cand.source,
                        "motivo": "Já existente no CSV-base para a mesma chave",
                    }
                )
            else:
                novos_raw.append(cand)

        novos_raw = sorted(
            novos_raw,
            key=lambda c: (
                normalize_key(c.ramo),
                c.numero_norm,
                c.data_mdy,
                normalize_tipo_key(c.tipo),
                c.href,
            ),
        )
        if args.max_new_rows > 0:
            novos_raw = novos_raw[: args.max_new_rows]

        # prepara linhas de saída
        rows_out: List[Dict[str, Any]] = []
        novos_detectados: List[Dict[str, Any]] = []
        next_id = max_id
        next_nr_by_ramo = dict(max_nr_by_ramo)

        for i, cand in enumerate(novos_raw):
            next_id += 1
            next_nr_by_ramo[cand.ramo] = next_nr_by_ramo.get(cand.ramo, 0) + 1
            row = {k: "" for k in fieldnames}
            row["ID"] = str(next_id)
            row["numero_no_ramo"] = str(next_nr_by_ramo[cand.ramo])
            row["ramo"] = cand.ramo
            row["numero_processo"] = cand.numero_norm
            row["tipo_de_processo"] = normalize_tipo(cand.tipo)
            row["relator"] = cand.relator
            row["data_julgamento"] = cand.data_mdy
            row["subramo"] = cand.subramo
            row["texto_original"] = clean_nul(cand.texto_original)
            row["tema"] = ""
            row["contexto"] = ""
            row["bullet_points"] = ""
            row["tese"] = ""
            row["punchline"] = ""
            row["link_1"] = cand.href
            row["link_2"] = ""
            row["link_3"] = ""
            rows_out.append(row)

            novos_detectados.append(
                {
                    "row_idx_out": str(i),
                    "ID": row["ID"],
                    "numero_no_ramo": row["numero_no_ramo"],
                    "ramo": cand.ramo,
                    "numero_processo": cand.numero_norm,
                    "tipo_de_processo": normalize_tipo(cand.tipo),
                    "data_julgamento": cand.data_mdy,
                    "relator": cand.relator,
                    "link_1": cand.href,
                    "source": cand.source,
                    "texto_original_preview": row["texto_original"][:300],
                    "api_status": "pending",
                }
            )

        api_falhas: List[Dict[str, Any]] = []
        done_indices: Set[int] = set()

        write_csv_atomic(csv_out, rows_out, fieldnames)
        write_json_atomic(
            checkpoint,
            {
                "fieldnames": fieldnames,
                "rows_out": rows_out,
                "done_indices": sorted(done_indices),
                "novos_detectados": novos_detectados,
                "ja_existentes_descartados": descartados,
                "incompletos_sem_chave": incompletos,
                "api_falhas": api_falhas,
                "source_stats": source_stats,
            },
        )
        print(f"[start] novos detectados: {len(rows_out)}")

    # Se não houver novos, finaliza com auditorias.
    if not rows_out:
        write_csv_atomic(csv_out, rows_out, fieldnames)
        write_dict_csv(
            Path(f"{audit_prefix}_novos_detectados.csv"),
            [],
            [
                "row_idx_out",
                "ID",
                "numero_no_ramo",
                "ramo",
                "numero_processo",
                "tipo_de_processo",
                "data_julgamento",
                "relator",
                "link_1",
                "source",
                "texto_original_preview",
                "api_status",
            ],
        )
        write_dict_csv(
            Path(f"{audit_prefix}_ja_existentes_descartados.csv"),
            descartados,
            ["ramo", "numero_processo", "tipo_de_processo", "data_julgamento", "href", "source", "motivo"],
        )
        write_dict_csv(
            Path(f"{audit_prefix}_incompletos_sem_chave.csv"),
            incompletos,
            [
                "ramo",
                "href",
                "source",
                "numero_processo",
                "tipo_de_processo",
                "data_julgamento",
                "missing_fields",
                "motivo",
                "citacao_raw",
            ],
        )
        write_dict_csv(
            Path(f"{audit_prefix}_api_falhas.csv"),
            [],
            ["row_idx_out", "ID", "numero_processo", "ramo", "link_1", "erro"],
        )
        summary = [
            f"CSV base: {base_csv}",
            f"CSV saída: {csv_out}",
            f"Novos detectados: 0",
            f"Descartados por já existir: {len(descartados)}",
            f"Incompletos/sem chave: {len(incompletos)}",
            f"API sucesso: 0",
            f"API falhas: 0",
            f"HTML local lidos: {source_stats.get('html_local_count', 0)}",
            f"Páginas live OK: {source_stats.get('live_pages_ok', 0)}",
            f"Páginas live falha: {source_stats.get('live_pages_fail', 0)}",
        ]
        Path(f"{audit_prefix}_resumo.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
        if checkpoint.exists():
            checkpoint.unlink()
        print("Concluído. Nenhum acréscimo detectado.")
        return

    # API stage
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrada para preenchimento analítico.")
    client = OpenAI(api_key=api_key)

    pending = [i for i in range(len(rows_out)) if i not in done_indices]
    print(f"[api] pendentes: {len(pending)} / {len(rows_out)}")
    lock = threading.Lock()
    since_save = 0

    def _save_state() -> None:
        write_csv_atomic(csv_out, rows_out, fieldnames)
        write_json_atomic(
            checkpoint,
            {
                "fieldnames": fieldnames,
                "rows_out": rows_out,
                "done_indices": sorted(done_indices),
                "novos_detectados": novos_detectados,
                "ja_existentes_descartados": descartados,
                "incompletos_sem_chave": incompletos,
                "api_falhas": api_falhas,
                "source_stats": source_stats,
            },
        )

    def _worker(idx: int) -> Dict[str, Any]:
        row = rows_out[idx]
        ok, fields, err = call_api_single(
            client=client,
            model=args.model,
            prompt_base=prompt_base,
            texto_julgado=row.get("texto_original", ""),
            retries=max(1, int(args.retries)),
            retry_wait_s=max(0.5, float(args.retry_wait_s)),
        )
        return {"idx": idx, "ok": ok, "fields": fields, "err": err}

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        fut_map = {ex.submit(_worker, idx): idx for idx in pending}
        for fut in as_completed(fut_map):
            res = fut.result()
            idx = int(res["idx"])
            row = rows_out[idx]
            if res["ok"]:
                fields = res["fields"]
                row["tema"] = normalize_api_field("tema", fields.get("tema", ""))
                row["contexto"] = normalize_api_field("contexto", fields.get("contexto", ""))
                row["tese"] = normalize_api_field("tese", fields.get("tese", ""))
                row["punchline"] = normalize_api_field("punchline", fields.get("punchline", ""))
                row["bullet_points"] = normalize_api_field("bullet_points", fields.get("bullet_points", ""))
                novos_detectados[idx]["api_status"] = "ok"
            else:
                api_falhas.append(
                    {
                        "row_idx_out": str(idx),
                        "ID": row.get("ID", ""),
                        "numero_processo": row.get("numero_processo", ""),
                        "ramo": row.get("ramo", ""),
                        "link_1": row.get("link_1", ""),
                        "erro": res.get("err", "erro desconhecido"),
                    }
                )
                novos_detectados[idx]["api_status"] = "fail"

            with lock:
                done_indices.add(idx)
                since_save += 1
                if since_save >= max(1, int(args.save_every)):
                    _save_state()
                    since_save = 0
                    print(f"[progress] API concluído: {len(done_indices)}/{len(rows_out)}")

    # normalização final bullets notion
    for row in rows_out:
        row["bullet_points"] = normalize_bullets_for_notion(row.get("bullet_points", ""))

    _save_state()

    # auditorias finais
    novos_path = Path(f"{audit_prefix}_novos_detectados.csv")
    desc_path = Path(f"{audit_prefix}_ja_existentes_descartados.csv")
    inc_path = Path(f"{audit_prefix}_incompletos_sem_chave.csv")
    fail_path = Path(f"{audit_prefix}_api_falhas.csv")
    resumo_path = Path(f"{audit_prefix}_resumo.txt")

    write_dict_csv(
        novos_path,
        novos_detectados,
        [
            "row_idx_out",
            "ID",
            "numero_no_ramo",
            "ramo",
            "numero_processo",
            "tipo_de_processo",
            "data_julgamento",
            "relator",
            "link_1",
            "source",
            "texto_original_preview",
            "api_status",
        ],
    )
    write_dict_csv(
        desc_path,
        descartados,
        ["ramo", "numero_processo", "tipo_de_processo", "data_julgamento", "href", "source", "motivo"],
    )
    write_dict_csv(
        inc_path,
        incompletos,
        [
            "ramo",
            "href",
            "source",
            "numero_processo",
            "tipo_de_processo",
            "data_julgamento",
            "missing_fields",
            "motivo",
            "citacao_raw",
        ],
    )
    write_dict_csv(
        fail_path,
        api_falhas,
        ["row_idx_out", "ID", "numero_processo", "ramo", "link_1", "erro"],
    )

    ok_count = sum(1 for x in novos_detectados if x.get("api_status") == "ok")
    fail_count = sum(1 for x in novos_detectados if x.get("api_status") == "fail")
    summary_lines = [
        f"CSV base: {base_csv}",
        f"CSV saída: {csv_out}",
        f"Novos detectados: {len(novos_detectados)}",
        f"Descartados por já existir: {len(descartados)}",
        f"Incompletos/sem chave: {len(incompletos)}",
        f"API sucesso: {ok_count}",
        f"API falhas: {fail_count}",
        f"HTML local lidos: {source_stats.get('html_local_count', 0)}",
        f"Candidatos local: {source_stats.get('candidates_local', 0)}",
        f"Candidatos live: {source_stats.get('candidates_live', 0)}",
        f"Páginas live OK: {source_stats.get('live_pages_ok', 0)}",
        f"Páginas live falha: {source_stats.get('live_pages_fail', 0)}",
        f"Auditoria novos: {novos_path}",
        f"Auditoria descartados: {desc_path}",
        f"Auditoria incompletos: {inc_path}",
        f"Auditoria API falhas: {fail_path}",
    ]
    resumo_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if checkpoint.exists():
        checkpoint.unlink()

    print("\nConcluído.")
    print(f"Saída: {csv_out}")
    print(f"Novos={len(novos_detectados)} | API ok={ok_count} | API falhas={fail_count}")


if __name__ == "__main__":
    main()
