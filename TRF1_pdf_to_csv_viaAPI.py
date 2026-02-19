#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRF1 PDF -> CSV com enriquecimento OpenAI + Perplexity.

Objetivo:
- Ler PDFs de boletins do TRF1 (pastas 2020..2026 ou seleção do usuário).
- Extrair julgados com parser determinístico (classe, processo, relator, data, texto etc.).
- Gerar CSV único: boletins_de_jurisprudencia_TRF1.csv.
- Enriquecer colunas analíticas via OpenAI (gpt-5-mini) e notícia via Perplexity.
- Executar em lotes com checkpoint/resume e logs amigáveis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERRO: pdfplumber não encontrado. Execute: pip install pdfplumber") from exc

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERRO: openai não encontrado. Execute: pip install openai") from exc


OUTPUT_NAME = "boletins_de_jurisprudencia_TRF1.csv"
CHECKPOINT_NAME = ".boletins_de_jurisprudencia_TRF1.checkpoint.json"
QUALITY_REPORT_NAME = ".boletins_de_jurisprudencia_TRF1.quality_report.json"
CHECKPOINT_VERSION = 3

CSV_COLUMNS: List[str] = [
    "classe",
    "numero_processo",
    "relator(a)",
    "orgao_julgador",
    "data_julgamento",
    "data_publicacao",
    "contexto",
    "tese",
    "tema",
    "texto_do_boletim",
    "decisao",
    "tribunal",
    "informativo",
    "ramo_do_direito",
    "subramo_do_direito",
    "punchline",
    "bullet_points",
    "jurisprudência",
    "legislacao",
    "noticia",
]

INTERNAL_KEYS = ["_row_id", "_openai_done", "_perplexity_done", "_perplexity_reason", "_source_pdf"]
MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a"}
OPENAI_CRITICAL_COLUMNS = [
    "contexto",
    "tese",
    "tema",
    "ramo_do_direito",
    "subramo_do_direito",
    "punchline",
    "bullet_points",
]
TEMA_MIN_WORDS = 8
TEMA_MAX_WORDS = 16

ORG_HEAD_RE = re.compile(
    r"^(primeira|segunda|terceira|quarta|quinta|sexta|s[eé]tima|oitava|nona|d[eé]cima|d[eé]cima primeira|d[eé]cima segunda|d[eé]cima terceira|d[eé]cima quarta|d[eé]cima quinta|d[eé]cima sexta|d[eé]cima s[eé]tima|d[eé]cima oitava|d[eé]cima nona|vig[eé]sima)\s+(turma|se[cç][aã]o)$",
    re.IGNORECASE,
)

CITATION_RE = re.compile(
    r"\((?P<cit>(?:[^()]|\([^()]*\))*?"
    r"(?P<tipo>[A-Za-z][A-Za-z0-9]{0,20}(?:[A-Za-z0-9\- ]{0,30})?)\s+"
    r"(?P<numero>\d{2,7}\s*-\s*\d{2}\s*\.\s*\d{4}\s*\.\s*\d\s*\.\s*\d{2}\s*\.\s*\d{4})"
    r"(?:[^()]|\([^()]*\))*?rel\.\s*(?P<relator>(?:[^(),]|\([^()]*\)){3,150}?)\s*,\s*em\s+"
    r"(?P<datefrag>(?:sess[ãa]o[^)]*?\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{1,2}\s*a\s*\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s*a\s*\d{1,2}/\d{1,2}/\d{4}))"
    r"(?:[^()]|\([^()]*\))*?)\)\.?,?",
    flags=re.IGNORECASE,
)

URL_RE = re.compile(r"https?://[^\s\]\)>,;\"']+", re.IGNORECASE)
DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
DATE_RANGE_MIXED_RE = re.compile(r"(\d{1,2})/(\d{1,2})\s*a\s*(\d{1,2})/(\d{1,2})/(\d{4})")
DATE_RANGE_COMPACT_RE = re.compile(r"(\d{1,2})\s*a\s*(\d{1,2})/(\d{1,2})/(\d{4})")
CNJ_FUZZY_RE = re.compile(
    r"(?P<n1>\d{6,9})\s*[-.]?\s*(?P<n2>\d{2})?\s*[-.]?\s*"
    r"(?P<ano>(?:19|20)\d{2})\s*[-.]?\s*(?P<j>\d)\s*[-.]?\s*(?P<tr>\d{2})\s*[-.]?\s*(?P<orig>\d{4})",
    re.IGNORECASE,
)
CITATION_FUZZY_RE = re.compile(
    r"\((?P<cit>[^()]{20,520}?)\)\.?,?",
    flags=re.IGNORECASE,
)
RELATOR_INLINE_RE = re.compile(
    r"(?i)\b(?:desembargador(?:a)?|des\.?|ju[ií]z(?:a)?(?: federal)?|min\.?)\s+([^,)\n]{3,140})"
)
PROCESS_NUMBER_IN_TEXT_RE = re.compile(r"\d{2,7}\s*[-.]\s*\d{2}\s*[-.]?\s*\d{4}\s*[-.]?\s*\d\s*[-.]?\s*\d{2}\s*[-.]?\s*\d{4}")
CITATION_THEN_CASE_RE = re.compile(
    r"\([^)]{20,360}\brel\.\s*[^)]*(?:\bem\s+)?\d{1,2}/\d{1,2}/+\d{4}[^)]*\)\s+[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ]",
    re.IGNORECASE,
)
HEADER_NOISE_PREFIX_RE = re.compile(r"^oletim\s+nformativo\s+de\s+urisprud", re.IGNORECASE)
CASE_LIKELY_START_RE = re.compile(
    r"\b[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]{2,}(?:\s+[a-záàâãéèêíìîóòôõúùûç]{2,})?\."
)
SPACE_RE = re.compile(r"\s+")
NON_NEWS_EXT_RE = re.compile(r"\.(?:pdf|doc|docx|xls|xlsx|zip|rar)(?:$|[?#])", re.IGNORECASE)
NEWS_HINT_TOKENS = (
    "noticia",
    "noticias",
    "news",
    "reportagem",
    "materia",
)
NON_NEWS_URL_TOKENS = (
    "acordao",
    "jurisprud",
    "consulta",
    "inteiro-teor",
    "processo",
    "pje",
    "documento",
    "download",
    "busca",
)
MAINSTREAM_NEWS_DOMAIN_SUFFIXES = (
    "g1.globo.com",
    "oglobo.globo.com",
    "valor.globo.com",
    "folha.uol.com.br",
    "uol.com.br",
    "estadao.com.br",
    "cnnbrasil.com.br",
    "terra.com.br",
    "r7.com",
    "band.uol.com.br",
    "correiobraziliense.com.br",
    "metropoles.com",
    "agenciabrasil.ebc.com.br",
    "veja.abril.com.br",
    "exame.com",
    "istoe.com.br",
    "poder360.com.br",
    "jovempan.com.br",
    "gazetadopovo.com.br",
    "nexojornal.com.br",
    "cartacapital.com.br",
    "noticias.uol.com.br",
)
OFFICIAL_INSTITUTION_DOMAIN_SUFFIXES = (
    "gov.br",
    "jus.br",
    "leg.br",
    "mp.br",
    "def.br",
    "mil.br",
)
OFFICIAL_NEWS_HINT_TOKENS = (
    "noticia",
    "noticias",
    "imprensa",
    "sala-de-imprensa",
    "comunicacao",
    "comunicados",
    "assuntos/noticias",
    "portal",
)
PERPLEXITY_DOMAIN_POLICY_CHOICES = ("consagrados_oficiais", "consagrados_apenas")
PERPLEXITY_DEFAULT_DOMAIN_POLICY = "consagrados_oficiais"
PERPLEXITY_ACCEPT_REASON_PREFIX = "accepted_stage"
PERPLEXITY_REJECTABLE_STAGE1_REASONS = {
    "domain_rejected",
    "path_rejected",
    "score_below_threshold",
    "no_candidates_stage1",
}
NON_EDITORIAL_SECTION_TOKENS = (
    "/blog/",
    "/blogs/",
    "/coluna/",
    "/colunas/",
    "/opiniao/",
    "/opinion/",
)
PT_STOPWORDS = {
    "para",
    "com",
    "sobre",
    "pela",
    "pelas",
    "pelos",
    "entre",
    "contra",
    "apenas",
    "caso",
    "julgado",
    "julgamento",
    "direito",
    "processo",
    "acao",
    "ação",
    "tema",
    "boletim",
    "noticia",
    "noticias",
    "jurisprudencia",
    "jurisprudência",
    "direitos",
    "publico",
    "público",
    "publica",
    "pública",
    "federal",
    "regiao",
    "região",
    "tribunal",
    "regional",
    "justica",
    "justiça",
    "administracao",
    "administração",
    "administrativo",
    "administrativa",
    "civil",
    "penal",
    "constitucional",
    "processual",
    "recurso",
    "decisao",
    "decisão",
    "acordao",
    "acórdão",
    "sentenca",
    "sentença",
}
DECISAO_MARK_RE = re.compile(r"(?i)\b(?:un[âa]nime|unanimidade|por maioria|maioria)\b\.?")


def normalize_ws(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def fold_text(text: str) -> str:
    base = unicodedata.normalize("NFKD", text or "")
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    base = base.lower()
    return normalize_ws(base)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in MISSING_TOKENS


def read_secret_from_file(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not raw:
        return ""
    first_line = raw.splitlines()[0].strip()
    if "=" in first_line:
        first_line = first_line.split("=", 1)[1].strip()
    return first_line.strip("\"' ")


def resolve_openai_key(cli_value: str) -> str:
    if cli_value.strip():
        return cli_value.strip()
    env = os.getenv("OPENAI_API_KEY", "").strip()
    if env:
        return env
    for fallback in ("CHAVE_SECRETA_API_Mauricio_local.txt",):
        val = read_secret_from_file(fallback)
        if val:
            return val
    return ""


def resolve_perplexity_key(cli_value: str) -> str:
    if cli_value.strip():
        return cli_value.strip()
    env = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if env:
        return env
    for fallback in ("Chave_secreta_Perplexity.txt",):
        val = read_secret_from_file(fallback)
        if val:
            return val
    return ""


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_informativo_from_filename(path: Path) -> str:
    m = re.search(r"Bij[_-]?(\d+)$", path.stem, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def normalize_cnj_number(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 20:
        return (
            f"{digits[0:7]}-{digits[7:9]}.{digits[9:13]}."
            f"{digits[13]}.{digits[14:16]}.{digits[16:20]}"
        )
    text = normalize_ws(raw)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    return text


def normalize_relator(raw: str) -> str:
    text = normalize_ws(raw)
    if not text:
        return ""
    lower = text.lower()
    convocado_marker = ""
    if re.search(r"(?i)\(convocad[oa]\)", text):
        convocado_marker = "convocada" if re.search(r"(?i)\(convocada\)", text) else "convocado"

    is_min = bool(re.search(r"(?i)\b(min\.?|ministro(?:a)?)\b", text))
    is_des = bool(re.search(r"(?i)\b(des\.?|desembargador(?:a)?)\b", text))

    cleaned = re.sub(r"(?i)\(convocad[oa]\)", "", text)
    cleaned = re.sub(
        r"(?i)^(?:desembargador(?:a)?|des\.?|ju[ií]z(?:a)?|min\.?|ministro(?:a)?)\s+(?:federal\s+)?",
        "",
        cleaned,
    )
    cleaned = normalize_ws(cleaned).strip(" ,.;")
    if not cleaned:
        return ""

    if convocado_marker:
        return f"{cleaned} ({convocado_marker})"
    if is_min or "min." in lower or lower.startswith("min"):
        return f"Min. {cleaned}"
    if is_des:
        return f"Des. {cleaned}"
    return cleaned


def normalize_date_to_mdy(value: str, source_hint: str = "auto") -> str:
    text = normalize_ws(value).replace("//", "/")
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if not m:
        return ""

    a = int(m.group(1))
    b = int(m.group(2))
    y = int(m.group(3))

    mdy_ok = True
    dmy_ok = True
    try:
        datetime(y, a, b)
    except ValueError:
        mdy_ok = False
    try:
        datetime(y, b, a)
    except ValueError:
        dmy_ok = False

    if source_hint == "dmy":
        if dmy_ok:
            return f"{b}/{a}/{y}"
        if mdy_ok:
            return f"{a}/{b}/{y}"
        return ""
    if source_hint == "mdy":
        if mdy_ok:
            return f"{a}/{b}/{y}"
        if dmy_ok:
            return f"{b}/{a}/{y}"
        return ""

    # auto: resolve casos claros e, em ambiguidade, preserva interpretação M/D.
    if a > 12 and dmy_ok:
        return f"{b}/{a}/{y}"
    if b > 12 and mdy_ok:
        return f"{a}/{b}/{y}"
    if mdy_ok:
        return f"{a}/{b}/{y}"
    if dmy_ok:
        return f"{b}/{a}/{y}"
    return ""


def normalize_orgao(text: str) -> str:
    cleaned = normalize_ws(text)
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def parse_date_fragment_to_mdy(fragment: str) -> str:
    if not fragment:
        return ""
    frag = normalize_ws(fragment).replace("//", "/")
    matches = DATE_RE.findall(frag)
    if matches:
        d, m, y = matches[-1]
        return normalize_date_to_mdy(f"{d}/{m}/{y}", source_hint="dmy")
    # fallback para padrões de intervalo
    m = DATE_RANGE_MIXED_RE.search(frag)
    if m:
        end_day = int(m.group(3))
        month = int(m.group(4))
        year = int(m.group(5))
        return normalize_date_to_mdy(f"{month}/{end_day}/{year}", source_hint="mdy")
    m = DATE_RANGE_COMPACT_RE.search(frag)
    if m:
        end_day = int(m.group(2))
        month = int(m.group(3))
        year = int(m.group(4))
        return normalize_date_to_mdy(f"{month}/{end_day}/{year}", source_hint="mdy")
    return ""


def extract_publication_date_from_first_page(page_text: str) -> str:
    if not page_text:
        return ""
    lines = [normalize_ws(x) for x in page_text.splitlines() if normalize_ws(x)]
    header = " ".join(lines[:30])
    m = re.search(
        r"JUSTI[ÇC]A\s+FEDERAL.*?(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})",
        header,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_date_to_mdy(m.group(2), source_hint="dmy")
    first_dates = re.findall(r"\d{2}/\d{2}/\d{4}", " ".join(lines[:25]))
    if len(first_dates) >= 2:
        return normalize_date_to_mdy(first_dates[1], source_hint="dmy")
    return ""


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    low = line.lower()
    compact = re.sub(r"[^a-z0-9]+", "", low)

    if re.fullmatch(r"\d+", line):
        return True
    if "boletim informativo de jurisprud" in low:
        return True
    if "tribunal regional federal da 1" in low:
        return True
    if "este informativo contém notícias não oficiais" in low:
        return True
    if "este informativo contem noticias nao oficiais" in low:
        return True
    if "informativo" in low and re.search(r"not[ií]c", low) and ("não oficial" in low or "nao oficial" in low):
        return True
    if "conteúdo efetivo das decisões" in low or "conteudo efetivo das decisoes" in low:
        return True
    if "fornecidas pelos gabinetes" in low:
        return True
    if "notas tomadas nas sessões de julgamento" in low or "notas tomadas nas sessoes de julgamento" in low:
        return True
    if "servidores da jurisprud" in low:
        return True
    if "sessões de julgamento" in low and "jurisprud" in low:
        return True
    if "não consistindo em repositório oficial" in low or "nao consistindo em repositorio oficial" in low:
        return True
    if "finalidade de antecipar decisões" in low or "finalidade de antecipar decisoes" in low:
        return True
    if re.search(r"justi[çc]a federal.*\d{2}/\d{2}/\d{4}.*\d{2}/\d{2}/\d{4}", low):
        return True
    if "publicação no e-djf1" in low or "publicacao no e-djf1" in low:
        return True
    if "publicação no pje" in low or "publicacao no pje" in low:
        return True
    if "jurisprud" in compact and "informativo" in compact:
        return True
    if "informativo" in compact and "naooficiais" in compact:
        return True
    if "informativo" in compact and "nooficiais" in compact:
        return True
    if "informativo" in compact and "notcias" in compact and "oficiais" in compact:
        return True
    if "fornecidaspelosgabinetes" in compact:
        return True
    if "notastomadasnassessoesdejulgamento" in compact:
        return True
    if "naoconsistindoemrepositoriooficial" in compact:
        return True
    if "finalidadedeantecipardecisoes" in compact:
        return True
    if "trf1" in compact and "informativo" in compact:
        return True
    if "trf" in compact and "jurisprud" in compact and "informativo" in compact:
        return True
    if "oletimnformativodeurisprud" in compact:
        return True
    if "esteinformativocontemnoticiasnaooficiais" in compact:
        return True
    if compact in {"bij", "n", "essoesdea", "essoesda", "ssoesdea", "essesdea"}:
        return True
    return False


def strip_inline_header_noise(line_raw: str) -> str:
    if not line_raw:
        return ""
    line = line_raw.strip()
    if not line:
        return ""
    low = line.lower()
    compact = re.sub(r"[^a-z0-9]+", "", low)
    has_header_mark = (
        "informativo" in low
        or "jurisprud" in low
        or "justiça federal" in low
        or "justica federal" in low
        or "repositorio oficial" in low
        or "naoconsistindoemrepositoriooficial" in compact
        or bool(re.search(r"\d{2}/\d{2}/\d{4}\s+\d{2}/\d{2}/\d{4}", line))
    )
    if not has_header_mark:
        return line_raw
    m = CASE_LIKELY_START_RE.search(line)
    if m and m.start() > 0:
        candidate = line[m.start() :].lstrip()
        if len(candidate) >= 20:
            return candidate
    return line_raw


def extract_date_fragment(text: str) -> str:
    if not text:
        return ""
    clean = text.replace("//", "/")
    m = re.search(
        r"(?i)(sess[ãa]o[^)]*?\d{4}|\d{1,2}/\d{1,2}\s*a\s*\d{1,2}/\d{1,2}/+\d{4}|\d{1,2}\s*a\s*\d{1,2}/\d{1,2}/+\d{4}|\d{1,2}/\d{1,2}/+\d{4})",
        clean,
    )
    if not m:
        return ""
    return cleanup_text(m.group(1).replace("//", "/"))


def extract_relator_from_citation_text(citation_text: str) -> str:
    if not citation_text:
        return ""
    m = re.search(r"(?i)\brel\.\s*([^,\n)]{3,160})", citation_text)
    if m:
        return normalize_relator(m.group(1))
    m = RELATOR_INLINE_RE.search(citation_text)
    if m:
        return normalize_relator(m.group(0))
    return ""


def extract_tipo_from_citation_text(citation_text: str, numero_raw: str) -> str:
    if not citation_text:
        return ""
    if numero_raw:
        pos = citation_text.find(numero_raw)
        if pos > 0:
            prefix = citation_text[:pos]
            mt = re.search(r"([A-Za-z][A-Za-z0-9./°º\-]*(?:\s+[A-Za-z][A-Za-z0-9./°º\-]*){0,2})\s*$", prefix.strip())
            if mt:
                return cleanup_text(mt.group(1))
    m = re.search(r"\(\s*([A-Za-z][A-Za-z0-9./°º\-]*(?:\s+[A-Za-z][A-Za-z0-9./°º\-]*){0,2})\s+", citation_text)
    if m:
        return cleanup_text(m.group(1))
    return ""


def canonical_cnj_from_parts(n1: str, n2: str, ano: str, j: str, tr: str, orig: str) -> str:
    n1_clean = re.sub(r"\D", "", n1 or "")
    n2_clean = re.sub(r"\D", "", n2 or "")
    ano_clean = re.sub(r"\D", "", ano or "")
    j_clean = re.sub(r"\D", "", j or "")
    tr_clean = re.sub(r"\D", "", tr or "")
    orig_clean = re.sub(r"\D", "", orig or "")
    if len(n1_clean) == 6:
        n1_clean = n1_clean.zfill(7)
    if len(n1_clean) != 7 or len(n2_clean) != 2:
        return ""
    if len(ano_clean) != 4 or len(j_clean) != 1 or len(tr_clean) != 2 or len(orig_clean) != 4:
        return ""
    return f"{n1_clean}-{n2_clean}.{ano_clean}.{j_clean}.{tr_clean}.{orig_clean}"


def normalize_cnj_number_fuzzy(raw_text: str) -> Tuple[str, str]:
    if not raw_text:
        return "", ""
    clean = re.sub(r"(?<=\d)\s+(?=\d)", "", raw_text.replace("–", "-"))

    # Caminho rápido: número já está regular após limpeza.
    fast = PROCESS_NUMBER_IN_TEXT_RE.search(clean)
    if fast:
        normalized = normalize_cnj_number(fast.group(0))
        if re.fullmatch(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}", normalized):
            return normalized, fast.group(0)

    best = ("", "")
    for m in CNJ_FUZZY_RE.finditer(clean):
        n1 = m.group("n1")
        n2 = m.group("n2") or ""
        ano = m.group("ano")
        j = m.group("j")
        tr = m.group("tr")
        orig = m.group("orig")

        if not n2:
            if len(n1) == 9:
                n1, n2 = n1[:7], n1[7:]
            elif len(n1) == 8:
                n1, n2 = n1[:6], n1[6:]
            else:
                continue

        candidate = canonical_cnj_from_parts(n1, n2, ano, j, tr, orig)
        if candidate:
            return candidate, m.group(0)
        if not best[0]:
            best = (candidate, m.group(0))
    return best


def span_overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def find_citation_segments(full_text: str) -> List[Dict[str, str]]:
    segments: List[Dict[str, str]] = []
    occupied: List[Tuple[int, int]] = []

    for m in CITATION_RE.finditer(full_text):
        numero = normalize_cnj_number(m.group("numero"))
        data = parse_date_fragment_to_mdy(m.group("datefrag"))
        if not numero or not data:
            continue
        seg = {
            "start": str(m.start()),
            "end": str(m.end()),
            "citation_text": m.group(0),
            "tipo": cleanup_text(m.group("tipo")),
            "numero": numero,
            "relator": normalize_relator(m.group("relator")),
            "datefrag": cleanup_text(m.group("datefrag")),
            "source": "strict",
        }
        segments.append(seg)
        occupied.append((m.start(), m.end()))

    for m in CITATION_FUZZY_RE.finditer(full_text):
        span = (m.start(), m.end())
        if any(span_overlaps(span, taken) for taken in occupied):
            continue
        citation_text = m.group(0)
        datefrag = extract_date_fragment(citation_text)
        data = parse_date_fragment_to_mdy(datefrag)
        numero, numero_raw = normalize_cnj_number_fuzzy(citation_text)
        if not data or not numero:
            continue
        tipo = extract_tipo_from_citation_text(citation_text, numero_raw)
        relator = extract_relator_from_citation_text(citation_text)
        segments.append(
            {
                "start": str(span[0]),
                "end": str(span[1]),
                "citation_text": citation_text,
                "tipo": tipo,
                "numero": numero,
                "relator": relator,
                "datefrag": datefrag,
                "source": "fuzzy",
            }
        )
        occupied.append(span)

    segments.sort(key=lambda x: int(x["start"]))
    deduped: List[Dict[str, str]] = []
    last_end = -1
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        if start < last_end:
            continue
        deduped.append(seg)
        last_end = end
    return deduped


def cleanup_text(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\u00ad", "")
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)  # hifenização de quebra de linha
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\s+,", ",", s)
    s = normalize_ws(s)
    return s.strip(" \n\t")


def sanitize_boletim_text(text: str) -> str:
    if not text:
        return ""
    # Normaliza EOL e remove quebras para manter texto contínuo no CSV.
    s = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    return cleanup_text(s)


def infer_tema_from_texto(texto_do_boletim: str) -> str:
    base = cleanup_text(texto_do_boletim)
    if not base:
        return ""
    tokens = [t for t in normalize_ws(base).split(" ") if t]
    if not tokens:
        return ""
    if len(tokens) < TEMA_MIN_WORDS:
        filler = ["em", "julgamento", "no", "TRF1", "sobre", "controvérsia", "jurídica"]
        idx = 0
        while len(tokens) < TEMA_MIN_WORDS:
            tokens.append(filler[idx % len(filler)])
            idx += 1
    tokens = tokens[:TEMA_MAX_WORDS]
    return " ".join(tokens)[:180]


def extract_texto_do_boletim_from_block(block: str, citation_text: str) -> str:
    if not block:
        return ""
    cut = len(citation_text or "")
    if cut <= 0:
        return sanitize_boletim_text(block)
    base = block[: max(0, len(block) - cut)]
    return sanitize_boletim_text(base)


def parse_decisao(block_text: str) -> str:
    text = cleanup_text(block_text)
    if not text:
        return ""
    if re.search(r"\b(?:un[âa]nime|unanimidade)\b", text, flags=re.IGNORECASE):
        return "unanimidade"
    if re.search(r"\bpor\s+maioria\b", text, flags=re.IGNORECASE):
        return "maioria"
    # Captura forma abreviada no fechamento: "... . Maioria."
    tail = text[-240:] if len(text) > 240 else text
    if re.search(r"(?:^|[.;:])\s*maioria\.?\s*$", tail, flags=re.IGNORECASE):
        return "maioria"
    # Última sentença/tokens.
    sentences = [normalize_ws(s) for s in re.split(r"[.;:]\s*", tail) if normalize_ws(s)]
    if sentences:
        last = sentences[-1].lower()
        if last == "maioria":
            return "maioria"
    return ""


def tema_word_count(tema: str) -> int:
    return len([t for t in normalize_ws(tema).split(" ") if t])


def is_tema_detailed(tema: str) -> bool:
    wc = tema_word_count(tema)
    return TEMA_MIN_WORDS <= wc <= TEMA_MAX_WORDS


def generate_row_id(row: Dict[str, str]) -> str:
    texto_fingerprint = cleanup_text(row.get("texto_do_boletim", ""))[:180]
    relator_value = row.get("relator(a)", "") or row.get("relator", "")
    base = "|".join(
        [
            row.get("numero_processo", ""),
            row.get("informativo", ""),
            relator_value,
            row.get("data_julgamento", ""),
            texto_fingerprint,
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def build_openai_request_key(row: Dict[str, str], text_max_chars: int) -> str:
    text_fp = compact_for_prompt(row.get("texto_do_boletim", ""), text_max_chars)
    base = "|".join(
        [
            normalize_ws(row.get("numero_processo", "")),
            normalize_ws(row.get("classe", "")),
            normalize_ws(row.get("relator(a)", "")),
            normalize_ws(text_fp),
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def build_perplexity_request_key(row: Dict[str, str], text_max_chars: int) -> str:
    text_fp = compact_for_prompt(row.get("texto_do_boletim", ""), min(text_max_chars, 2400))
    base = "|".join(
        [
            normalize_ws(row.get("numero_processo", "")),
            normalize_ws(row.get("relator(a)", "")),
            normalize_ws(row.get("tema", "")),
            normalize_ws(text_fp),
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def classify_openai_error(err_text: str) -> str:
    low = (err_text or "").lower()
    if "429" in low or "rate limit" in low:
        return "rate_limit"
    if "timeout" in low or "timed out" in low or "read timed out" in low:
        return "timeout"
    if "503" in low or "502" in low:
        return "upstream_unavailable"
    return "error"


def base_empty_row() -> Dict[str, str]:
    row = {col: "" for col in CSV_COLUMNS}
    row["tribunal"] = "TRF1"
    return row


def extract_rows_from_pdf(pdf_path: Path, logger: logging.Logger) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    info = extract_informativo_from_filename(pdf_path)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return rows

            first_page_text = pdf.pages[0].extract_text() or ""
            data_pub = extract_publication_date_from_first_page(first_page_text)

            current_orgao = ""
            marks: List[Tuple[int, str]] = [(0, "")]
            parts: List[str] = []
            text_len = 0

            for page in pdf.pages:
                page_text = page.extract_text() or ""
                lines = page_text.splitlines()
                for raw_line in lines:
                    line_raw = strip_inline_header_noise(raw_line.rstrip("\r"))
                    line_norm = normalize_ws(line_raw)
                    if not line_norm:
                        continue
                    if ORG_HEAD_RE.match(line_norm):
                        current_orgao = normalize_orgao(line_norm)
                        marks.append((text_len, current_orgao))
                        continue
                    if is_noise_line(line_norm):
                        continue
                    parts.append(line_raw)
                    text_len += len(line_raw) + 1

            full_text = "\n".join(parts)
            if not full_text.strip():
                return rows

            segments = find_citation_segments(full_text)
            if not segments:
                logger.debug("Sem citações finais detectáveis em %s", pdf_path.name)
                return rows

            prev_end = 0
            mark_idx = 0
            strict_count = 0
            fuzzy_count = 0
            for seg in segments:
                start = int(seg["start"])
                end = int(seg["end"])
                block = full_text[prev_end:end]
                prev_end = end
                if len(normalize_ws(block)) < 40:
                    continue

                while mark_idx + 1 < len(marks) and marks[mark_idx + 1][0] <= start:
                    mark_idx += 1
                orgao = marks[mark_idx][1] if marks else ""

                tipo = cleanup_text(seg.get("tipo", ""))
                numero = normalize_cnj_number(seg.get("numero", ""))
                relator = normalize_relator(seg.get("relator", ""))
                datefrag = cleanup_text(seg.get("datefrag", ""))
                data_julgamento = parse_date_fragment_to_mdy(datefrag)
                if not numero or not data_julgamento:
                    continue

                citation_text = seg.get("citation_text", "")
                texto_do_boletim = extract_texto_do_boletim_from_block(block, citation_text)
                if not texto_do_boletim:
                    continue

                row = base_empty_row()
                row["numero_processo"] = numero
                row["classe"] = tipo
                row["relator(a)"] = relator
                row["orgao_julgador"] = orgao
                row["data_julgamento"] = data_julgamento
                row["data_publicacao"] = data_pub
                row["texto_do_boletim"] = texto_do_boletim
                row["decisao"] = parse_decisao(texto_do_boletim)
                row["tribunal"] = "TRF1"
                row["informativo"] = info
                row["_source_pdf"] = str(pdf_path)
                row["_openai_done"] = "0"
                row["_perplexity_done"] = "0"
                row["_perplexity_reason"] = ""
                row["_row_id"] = generate_row_id(row)
                rows.append(row)
                if seg.get("source") == "fuzzy":
                    fuzzy_count += 1
                else:
                    strict_count += 1
            logger.debug(
                "%s -> citações extraídas strict=%d fuzzy=%d",
                pdf_path.name,
                strict_count,
                fuzzy_count,
            )
    except Exception as exc:
        logger.warning("Falha ao extrair %s: %s", pdf_path, exc)
    return rows


def deduplicate_rows(rows: Sequence[Dict[str, str]], logger: logging.Logger) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    dup = 0
    for row in rows:
        rid = row.get("_row_id", "")
        if not rid:
            rid = generate_row_id(row)
            row["_row_id"] = rid
        if rid in seen:
            dup += 1
            continue
        seen.add(rid)
        out.append(row)
    if dup:
        logger.info("Linhas duplicadas removidas: %d", dup)
    return out


def normalize_tag_item(text: str) -> str:
    s = normalize_ws(text)
    s = s.strip("\"' ")
    s = s.replace("\"", "").replace("'", "")
    s = re.sub(r"\s*,\s*", " ", s)
    s = normalize_ws(s)
    return s


def split_multi_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [normalize_ws(str(x)) for x in value if normalize_ws(str(x))]
        return items
    text = str(value or "")
    if not text:
        return []
    parts = [normalize_ws(x) for x in re.split(r"[,;\n\r|]+", text) if normalize_ws(x)]
    return parts


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        key = normalize_ws(item).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(normalize_ws(item))
    return out


def normalize_subramo_two(value: Any) -> str:
    parts = dedupe_keep_order(split_multi_value(value))
    parts = [normalize_tag_item(x) for x in parts if normalize_tag_item(x)]
    if len(parts) >= 2:
        return f"{parts[0]}, {parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return ""


def normalize_bullet_points_three(value: Any) -> str:
    parts: List[str]
    if isinstance(value, str) and "•" in value:
        # Se vier com bullets agregados na mesma linha, quebra pelos marcadores.
        raw = value.replace("\r\n", "\n").replace("\r", "\n")
        parts = [normalize_ws(x) for x in re.split(r"\s*•\s*", raw) if normalize_ws(x)]
    else:
        parts = dedupe_keep_order(split_multi_value(value))
    cleaned: List[str] = []
    for p in parts:
        x = normalize_ws(str(p))
        x = x.strip("\"' ")
        x = re.sub(r"\s*•\s*", "", x)
        x = re.sub(r"\s*,\s*", " - ", x)
        x = x.replace("\"", "").replace("'", "")
        x = normalize_ws(x).strip(" -.,;:")
        if x:
            cleaned.append(f"• {x},")
    cleaned = dedupe_keep_order(cleaned)
    if not cleaned:
        return ""
    if len(cleaned) > 3:
        cleaned = cleaned[:3]
    return "\n".join(cleaned)


def normalize_legislacao(value: Any) -> str:
    parts = dedupe_keep_order(split_multi_value(value))
    cleaned = [normalize_tag_item(x) for x in parts if normalize_tag_item(x)]
    cleaned = [re.sub(r"\s*,\s*", " ", x) for x in cleaned]
    cleaned = dedupe_keep_order(cleaned)
    return ", ".join(cleaned)


def normalize_jurisprudencia(value: Any) -> str:
    parts = dedupe_keep_order(split_multi_value(value))
    cleaned = [normalize_tag_item(x) for x in parts if normalize_tag_item(x)]
    cleaned = dedupe_keep_order(cleaned)
    return ", ".join(cleaned)


def truncate_to_words(text: str, max_words: int) -> str:
    tokens = [t for t in normalize_ws(text).split(" ") if t]
    if len(tokens) <= max_words:
        return " ".join(tokens)
    return " ".join(tokens[:max_words])


def compact_for_prompt(text: str, max_chars: int, *, keep_tail: int = 420) -> str:
    """
    Compacta texto para prompt preservando início e cauda (útil para trecho decisório).
    """
    cleaned = sanitize_boletim_text(text)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    keep_tail = max(120, min(keep_tail, max_chars // 2))
    head_len = max(120, max_chars - keep_tail - 24)
    head = cleaned[:head_len].rstrip()
    tail = cleaned[-keep_tail:].lstrip()
    return f"{head} ... [trecho omitido] ... {tail}"


def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def normalize_host(host: str) -> str:
    h = (host or "").strip().lower()
    if "@" in h:
        h = h.split("@", 1)[-1]
    if ":" in h:
        h = h.split(":", 1)[0]
    if h.startswith("www."):
        h = h[4:]
    return h


def is_mainstream_news_domain(host: str) -> bool:
    h = normalize_host(host)
    if not h:
        return False
    for allowed in MAINSTREAM_NEWS_DOMAIN_SUFFIXES:
        a = normalize_host(allowed)
        if h == a or h.endswith(f".{a}"):
            return True
    return False


def is_official_institution_domain(host: str) -> bool:
    h = normalize_host(host)
    if not h:
        return False
    for allowed in OFFICIAL_INSTITUTION_DOMAIN_SUFFIXES:
        a = normalize_host(allowed)
        if h == a or h.endswith(f".{a}"):
            return True
    return False


def classify_source_domain(host: str, domain_policy: str = PERPLEXITY_DEFAULT_DOMAIN_POLICY) -> str:
    if is_mainstream_news_domain(host):
        return "mainstream"
    if domain_policy == "consagrados_oficiais" and is_official_institution_domain(host):
        return "official"
    return "other"


def evaluate_news_url_policy(
    url: str,
    domain_policy: str = PERPLEXITY_DEFAULT_DOMAIN_POLICY,
) -> Tuple[bool, str, str]:
    if not validate_url(url):
        return False, "path_rejected", "other"
    parsed = urlparse(url.strip())
    host = normalize_host(parsed.netloc or "")
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    full = f"{host}{path}?{query}"
    source_type = classify_source_domain(host, domain_policy=domain_policy)
    if source_type == "other":
        return False, "domain_rejected", source_type
    if NON_NEWS_EXT_RE.search(path):
        return False, "path_rejected", source_type
    if any(tok in full for tok in NON_NEWS_URL_TOKENS):
        return False, "path_rejected", source_type
    if any(tok in path for tok in NON_EDITORIAL_SECTION_TOKENS):
        return False, "path_rejected", source_type
    if path in {"", "/"}:
        return False, "path_rejected", source_type
    if source_type == "official":
        if any(tok in full for tok in OFFICIAL_NEWS_HINT_TOKENS):
            return True, "ok", source_type
        if re.search(r"/20\d{2}/", path):
            return True, "ok", source_type
        if len([x for x in path.split("/") if x]) >= 2:
            return True, "ok", source_type
        return False, "path_rejected", source_type
    if any(tok in full for tok in NEWS_HINT_TOKENS):
        return True, "ok", source_type
    return True, "ok", source_type


def is_probably_news_url(url: str, domain_policy: str = PERPLEXITY_DEFAULT_DOMAIN_POLICY) -> bool:
    accepted, _, _ = evaluate_news_url_policy(url, domain_policy=domain_policy)
    return accepted


def extract_relator_surname(relator_raw: str) -> str:
    text = fold_text(relator_raw)
    text = re.sub(r"\(convocad[oa]\)", "", text)
    text = re.sub(r"\b(des\.?|min\.?)\b", " ", text)
    tokens = re.findall(r"[a-z]{2,}", text)
    return tokens[-1] if tokens else ""


def topic_keywords_from_row(row: Dict[str, str], limit: int = 6) -> List[str]:
    base = fold_text(f"{row.get('tema', '')} {row.get('contexto', '')} {row.get('tese', '')}")
    toks = re.findall(r"[a-z0-9]{4,}", base)
    out: List[str] = []
    for t in toks:
        if t in PT_STOPWORDS:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def case_match_features(row: Dict[str, str], evidence: str, url: str = "") -> Dict[str, Any]:
    ev = fold_text(evidence)
    url_folded = fold_text(url)
    combined = f"{ev} {url_folded}".strip()
    proc = re.sub(r"\D", "", row.get("numero_processo", ""))
    ev_digits = re.sub(r"\D", "", evidence or "")
    url_digits = re.sub(r"\D", "", url or "")

    proc_full_in_ev = bool(proc and proc in ev_digits)
    proc_full_in_url = bool(proc and proc in url_digits)
    proc_short = proc[:11] if proc and len(proc) >= 11 else ""
    proc_short_in_ev = bool(proc_short and proc_short in ev_digits)
    proc_short_in_url = bool(proc_short and proc_short in url_digits)

    surname = extract_relator_surname(row.get("relator(a)", ""))
    surname_in_ev = bool(surname and surname in ev)
    surname_in_url = bool(surname and surname in url_folded)

    keys = topic_keywords_from_row(row, limit=8)
    key_hits_ev = sum(1 for k in keys if k in ev)
    key_hits_url = sum(1 for k in keys if k in url_folded)
    key_hits_total = sum(1 for k in keys if k in combined)
    trf1_in_ev = bool("trf1" in ev or "tribunal regional federal da 1" in ev)
    trf1_in_url = bool("trf1" in url_folded or "tribunal regional federal da 1" in url_folded)
    trf1_in_combined = bool(trf1_in_ev or trf1_in_url)

    anchor_url = proc_full_in_url or proc_short_in_url or surname_in_url
    anchor_evidence = proc_full_in_ev or proc_short_in_ev or surname_in_ev
    reliable_anchor = bool(anchor_url or (anchor_evidence and (key_hits_url >= 1 or trf1_in_url)))

    return {
        "combined": combined,
        "proc_full_in_ev": proc_full_in_ev,
        "proc_full_in_url": proc_full_in_url,
        "proc_short_in_ev": proc_short_in_ev,
        "proc_short_in_url": proc_short_in_url,
        "surname_in_ev": surname_in_ev,
        "surname_in_url": surname_in_url,
        "key_hits_ev": key_hits_ev,
        "key_hits_url": key_hits_url,
        "key_hits_total": key_hits_total,
        "trf1_in_ev": trf1_in_ev,
        "trf1_in_url": trf1_in_url,
        "trf1_in_combined": trf1_in_combined,
        "reliable_anchor": reliable_anchor,
    }


def case_match_score(row: Dict[str, str], evidence: str, url: str = "") -> int:
    feats = case_match_features(row, evidence, url)
    if not feats.get("combined"):
        return 0
    score = 0
    if feats["proc_full_in_url"]:
        score += 4
    elif feats["proc_full_in_ev"]:
        score += 3

    if feats["proc_short_in_url"]:
        score += 2
    elif feats["proc_short_in_ev"]:
        score += 1

    if feats["surname_in_url"]:
        score += 3
    elif feats["surname_in_ev"]:
        score += 2

    if feats["key_hits_url"] >= 2:
        score += 2
    elif feats["key_hits_url"] == 1:
        score += 1

    if feats["key_hits_ev"] >= 2:
        score += 1

    if feats["trf1_in_combined"]:
        score += 1
    if feats["reliable_anchor"] and feats["key_hits_total"] >= 1:
        score += 1
    return score


def evidence_matches_case(row: Dict[str, str], evidence: str, url: str = "") -> bool:
    return case_match_score(row, evidence, url) >= 2


def normalize_confidence(value: str) -> str:
    conf = normalize_ws(value).lower()
    return conf if conf in {"alta", "baixa"} else "baixa"


def is_strict_noticia_candidate(
    row: Dict[str, str],
    url: str,
    evidence: str,
    confianca: str,
    min_score_mainstream: int = 2,
    min_score_official: int = 1,
    domain_policy: str = PERPLEXITY_DEFAULT_DOMAIN_POLICY,
) -> bool:
    accepted, _, source_type = evaluate_news_url_policy(url, domain_policy=domain_policy)
    if not accepted:
        return False
    feats = case_match_features(row, evidence, url)
    if source_type == "official":
        if (not feats["reliable_anchor"]) and not (feats["key_hits_url"] >= 2 and feats["trf1_in_combined"]):
            return False
    elif (not feats["reliable_anchor"]) and feats["key_hits_url"] < 1:
        return False
    score = case_match_score(row, evidence, url)
    min_score = min_score_official if source_type == "official" else min_score_mainstream
    return score >= max(0, min_score)


def dedupe_candidates(candidates: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for cand in candidates:
        url = normalize_ws(cand.get("url", ""))
        if not url:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "url": url,
                "evidencia": normalize_ws(cand.get("evidencia", "")),
                "confianca": normalize_confidence(cand.get("confianca", "")),
            }
        )
    return out


def select_best_noticia_candidate(
    row: Dict[str, str],
    candidates: Sequence[Dict[str, str]],
    *,
    min_score_mainstream: int,
    min_score_official: int,
    domain_policy: str,
    stage: int,
) -> Tuple[str, str]:
    if not candidates:
        return "", f"no_candidates_stage{stage}"

    seen_domain_reject = False
    seen_path_reject = False
    seen_score_below = False
    accepted_mainstream: List[Tuple[int, int, Dict[str, str]]] = []
    accepted_official: List[Tuple[int, int, Dict[str, str]]] = []

    for cand in candidates:
        url = normalize_ws(cand.get("url", ""))
        evidence = normalize_ws(cand.get("evidencia", ""))
        conf = normalize_confidence(cand.get("confianca", ""))
        allowed, reject_reason, source_type = evaluate_news_url_policy(url, domain_policy=domain_policy)
        if not allowed:
            if reject_reason == "domain_rejected":
                seen_domain_reject = True
            elif reject_reason == "path_rejected":
                seen_path_reject = True
            continue

        feats = case_match_features(row, evidence, url)
        if stage == 1:
            if source_type == "official":
                semantic_ok = feats["reliable_anchor"] or (feats["key_hits_url"] >= 2 and feats["trf1_in_combined"])
            else:
                semantic_ok = feats["reliable_anchor"] or (feats["key_hits_url"] >= 1)
        else:
            # Stage2 (fallback): relaxa um pouco sem abrir para links genéricos.
            # Exige vínculo contextual consistente no texto+URL.
            semantic_ok = feats["reliable_anchor"] or (feats["trf1_in_combined"] and feats["key_hits_total"] >= 2)
        if not semantic_ok:
            seen_score_below = True
            continue

        score = case_match_score(row, evidence, url)
        threshold = min_score_official if source_type == "official" else min_score_mainstream
        if stage >= 2:
            threshold = max(0, threshold - 1)
        if score < threshold:
            seen_score_below = True
            continue
        conf_rank = 1 if conf == "alta" else 0
        item = (score, conf_rank, {"url": url, "source_type": source_type})
        if source_type == "mainstream":
            accepted_mainstream.append(item)
        else:
            accepted_official.append(item)

    if accepted_mainstream:
        best = sorted(accepted_mainstream, key=lambda x: (x[0], x[1]), reverse=True)[0][2]
        return best["url"], f"accepted_stage{stage}_mainstream"
    if accepted_official:
        best = sorted(accepted_official, key=lambda x: (x[0], x[1]), reverse=True)[0][2]
        return best["url"], f"accepted_stage{stage}_official"

    if seen_score_below:
        return "", "score_below_threshold"
    if seen_path_reject:
        return "", "path_rejected"
    if seen_domain_reject:
        return "", "domain_rejected"
    return "", f"no_candidates_stage{stage}"


def normalize_openai_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    ramo = normalize_ws(str(payload.get("ramo_do_direito", "") or ""))
    subramo = normalize_subramo_two(payload.get("subramo_do_direito", ""))
    punchline = truncate_to_words(str(payload.get("punchline", "") or ""), 20)
    bullets = normalize_bullet_points_three(payload.get("bullet_points", ""))
    jurisprudencia = normalize_jurisprudencia(payload.get("jurisprudência", payload.get("jurisprudencia", "")))
    legislacao = normalize_legislacao(payload.get("legislacao", ""))

    return {
        "contexto": normalize_ws(str(payload.get("contexto", "") or "")),
        "tese": normalize_ws(str(payload.get("tese", "") or "")),
        "tema": normalize_ws(str(payload.get("tema", "") or "")),
        "ramo_do_direito": ramo,
        "subramo_do_direito": subramo,
        "punchline": punchline,
        "bullet_points": bullets,
        "jurisprudência": jurisprudencia,
        "legislacao": legislacao,
    }


OPENAI_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "contexto": {"type": "string"},
        "tese": {"type": "string"},
        "tema": {"type": "string"},
        "ramo_do_direito": {"type": "string"},
        "subramo_do_direito": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
        "punchline": {"type": "string"},
        "bullet_points": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "jurisprudência": {"type": "array", "items": {"type": "string"}},
        "legislacao": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "contexto",
        "tese",
        "tema",
        "ramo_do_direito",
        "subramo_do_direito",
        "punchline",
        "bullet_points",
        "jurisprudência",
        "legislacao",
    ],
}


def build_openai_prompt(row: Dict[str, str], *, text_max_chars: int = 0) -> str:
    texto_base = compact_for_prompt(row.get("texto_do_boletim", ""), text_max_chars)
    legislacao_rules = """
Você é um assistente jurídico ultrapreciso, focado em extrair e formatar EXCLUSIVAMENTE citações de DISPOSITIVOS LEGAIS como tags para o Notion.
# TAREFA
Analise o texto-base e extraia SOMENTE as menções diretas a dispositivos legais (leis, códigos, constituição). A regra mais importante é que a tag final NÃO PODE CONTER VÍRGULAS OU ASPAS.
# REGRAS DE VEDAÇÃO (O QUE IGNORAR)
- IGNORE FRASES COMPLETAS: Não extraia teses, resumos de julgados ou qualquer frase longa.
- IGNORE JURISPRUDÊNCIA.
- FOCO TOTAL NO PADRÃO: Tag válida deve conter número de lei, artigo, ou diploma conhecido.
# REGRAS DE FORMATAÇÃO DAS TAGS
- Formato: <DIPLOMA> <restante da citação com espaços>, sem vírgulas e sem aspas.
# REGRAS DE NORMALIZAÇÃO DE DIPLOMAS
- Constituição: CF/88
- Códigos: CTB, CPC/2015, CPP, CP
- Leis: Lei nº X.XXX/AAAA
- Decretos-Lei: DL nº XXX/AAAA
# REGRAS ADICIONAIS
- Expansão: enumerações em tags separadas.
- Deduplicação: não repetir tags.
"""
    return f"""
Analise o julgado do TRF1 e preencha os campos solicitados.

REGRAS IMPORTANTES:
1) data_julgamento já está normalizada e não deve ser alterada.
2) ramo_do_direito: uma única grande área.
3) subramo_do_direito: exatamente 2 subramos distintos.
4) tema: entre 8 e 16 palavras, contendo objeto jurídico e controvérsia central.
   Exemplo ruim: "Ensino superior"
   Exemplo bom: "Antecipação de colação de grau para candidato aprovado em concurso público"
5) bullet_points: exatamente 3 itens, cada item em linha separada iniciando por "• " e terminando com vírgula. Evite vírgulas dentro do texto do item.
6) jurisprudência: apenas precedentes/jurisprudência citados.
7) legislacao: seguir rigorosamente as regras abaixo.
8) TEXTO PRINCIPAL para análise: o campo "texto_do_boletim". Use metadados apenas para desambiguação.
9) punchline: no máximo 20 palavras.

{legislacao_rules}

DADOS DO JULGADO:
- classe: {row.get('classe', '')}
- numero_processo: {row.get('numero_processo', '')}
- relator(a): {row.get('relator(a)', row.get('relator', ''))}
- orgao_julgador: {row.get('orgao_julgador', '')}
- data_julgamento: {row.get('data_julgamento', '')}
- informativo: {row.get('informativo', '')}
- texto_do_boletim: {texto_base}

Retorne APENAS JSON válido no schema solicitado.
""".strip()


def parse_openai_message_content(content_obj: Any) -> str:
    if isinstance(content_obj, str):
        return content_obj.strip()
    if isinstance(content_obj, list):
        chunks: List[str] = []
        for item in content_obj:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks).strip()
    return ""


def openai_call_single(
    client: OpenAI,
    model: str,
    row: Dict[str, str],
    timeout: int,
    retries: int,
    max_completion_tokens: int,
    text_max_chars: int,
    logger: logging.Logger,
) -> Tuple[bool, Dict[str, str], str]:
    prompt = build_openai_prompt(row, text_max_chars=text_max_chars)
    last_err = ""
    completion_tokens = max(0, int(max_completion_tokens))
    for attempt in range(1, retries + 1):
        try:
            req_payload: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Você é um assistente jurídico de jurimetria. "
                            "Responda somente com JSON válido."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trf1_enrichment",
                        "strict": True,
                        "schema": OPENAI_JSON_SCHEMA,
                    },
                },
                "timeout": timeout,
            }
            if completion_tokens > 0:
                req_payload["max_completion_tokens"] = completion_tokens
            resp = client.chat.completions.create(
                **req_payload,
            )
            choice = resp.choices[0]
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            raw = parse_openai_message_content(choice.message.content)
            if not raw:
                if finish_reason == "length":
                    raise ValueError("empty_json_due_length")
                refusal = getattr(choice.message, "refusal", None)
                if refusal:
                    raise ValueError(f"refusal: {refusal}")
                raise ValueError("empty_json")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("JSON de resposta não é objeto.")
            normalized = normalize_openai_payload(parsed)
            return True, normalized, ""
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if "empty_json_due_length" in last_err and completion_tokens > 0:
                completion_tokens = min(3200, max(completion_tokens + 350, int(completion_tokens * 1.5)))
            if attempt < retries:
                wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.35)
                logger.debug("OpenAI retry %d/%d em %.2fs: %s", attempt, retries, wait, exc)
                time.sleep(wait)
    return False, {}, last_err


PERPLEXITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidatos": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {"type": "string"},
                    "evidencia": {"type": "string"},
                    "confianca": {"type": "string", "enum": ["alta", "baixa"]},
                },
                "required": ["url", "evidencia", "confianca"],
            },
            "maxItems": 5,
        },
        "noticia": {"type": "string"},
        "evidencia": {"type": "string"},
        "confianca": {"type": "string", "enum": ["alta", "baixa"]},
    },
    "required": ["candidatos"],
}


def build_perplexity_prompt(row: Dict[str, str], stage: int = 1, *, text_max_chars: int = 0) -> str:
    texto_base = compact_for_prompt(row.get("texto_do_boletim", ""), text_max_chars)
    stage_hint = "ETAPA 1 (primária)" if stage == 1 else "ETAPA 2 (fallback controlado)"
    strategy_hint = (
        "Faça busca ampla controlada, priorizando número do processo, relator(a) e tema para aumentar cobertura sem perder aderência."
        if stage == 2
        else "Faça busca precisa com foco em correspondência direta ao caso."
    )
    return f"""
{stage_hint}: Encontre ATÉ 5 links de notícia potencialmente relacionados ao MESMO julgado abaixo.
{strategy_hint}
Use "texto_do_boletim" como fonte principal de contexto.

REGRAS ESTRITAS:
1) Só aceite notícia com evidência textual clara de correspondência ao caso.
2) Fontes aceitas: veículos de imprensa consagrados e páginas oficiais de órgãos públicos/tribunais (gov.br, jus.br etc.) com conteúdo noticioso.
3) Não retorne blogs, colunas de opinião, sites pessoais, repositórios jurídicos ou documentos técnicos.
4) Correspondência mínima desejável: número do processo OU relator(a) + fato central compatível.
5) REJEITE links de consulta processual, acórdão, inteiro teor, PDF, busca ou documento técnico.
6) Se houver dúvida, retorne `candidatos` vazio.

FORMATO DE RESPOSTA (JSON):
- candidatos: lista (0 a 5) de objetos {{url, evidencia, confianca}}
- Para cada candidato, `evidencia` deve ter no máximo 25 palavras e mencionar preferencialmente processo ou relator(a).

Contexto:
- classe: {row.get('classe', '')}
- processo: {row.get('numero_processo', '')}
- relator(a): {row.get('relator(a)', row.get('relator', ''))}
- data: {row.get('data_julgamento', '')}
- tema: {row.get('tema', '')}
- contexto: {row.get('contexto', '')}
- tese: {row.get('tese', '')}
- ramo: {row.get('ramo_do_direito', '')}
- subramo: {row.get('subramo_do_direito', '')}
- texto_do_boletim: {texto_base}
""".strip()


def parse_perplexity_content(content: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"candidatos": [], "noticia": "", "evidencia": "", "confianca": "baixa"}
    text = (content or "").strip()
    if not text:
        return out
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            raw_cands = obj.get("candidatos")
            candidates: List[Dict[str, str]] = []
            if isinstance(raw_cands, list):
                for cand in raw_cands:
                    if not isinstance(cand, dict):
                        continue
                    candidates.append(
                        {
                            "url": normalize_ws(str(cand.get("url", "") or "")),
                            "evidencia": normalize_ws(str(cand.get("evidencia", "") or "")),
                            "confianca": normalize_confidence(str(cand.get("confianca", "") or "")),
                        }
                    )
            out["candidatos"] = dedupe_candidates(candidates)
            out["noticia"] = normalize_ws(str(obj.get("noticia", "") or ""))
            out["evidencia"] = normalize_ws(str(obj.get("evidencia", "") or ""))
            out["confianca"] = normalize_confidence(str(obj.get("confianca", "") or ""))
            if out["noticia"] and not out["candidatos"]:
                out["candidatos"] = dedupe_candidates(
                    [
                        {
                            "url": out["noticia"],
                            "evidencia": out["evidencia"],
                            "confianca": out["confianca"],
                        }
                    ]
                )
            return out
    except Exception:
        pass
    urls = URL_RE.findall(text)
    if urls:
        out["candidatos"] = dedupe_candidates(
            [
                {
                    "url": u.strip(),
                    "evidencia": "",
                    "confianca": "baixa",
                }
                for u in urls[:5]
            ]
        )
    return out


def perplexity_call_single(
    session: requests.Session,
    model: str,
    row: Dict[str, str],
    *,
    stage: int,
    min_score_mainstream: int,
    min_score_official: int,
    domain_policy: str,
    timeout: int,
    retries: int,
    max_tokens: int,
    text_max_chars: int,
    logger: logging.Logger,
) -> Tuple[bool, str, bool, str, str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Responda apenas JSON válido sem texto adicional.",
            },
            {
                "role": "user",
                "content": build_perplexity_prompt(row, stage=stage, text_max_chars=text_max_chars),
            },
        ],
        "search_mode": "web",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "trf1_noticia",
                "strict": True,
                "schema": PERPLEXITY_SCHEMA,
            },
        },
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    last_err = ""
    saw_rate_limit = False
    reason = ""
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                timeout=(8, timeout),
            )
            if resp.status_code == 429:
                saw_rate_limit = True
                last_err = "429 rate limit"
                if attempt < retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.35)
                    time.sleep(wait)
                    continue
                return False, "", True, last_err, "rate_limit"
            resp.raise_for_status()
            data = resp.json()
            content = ""
            try:
                content = str(((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or "").strip()
            except Exception:
                content = ""
            parsed = parse_perplexity_content(content)
            citations = data.get("citations") or []
            candidates: List[Dict[str, str]] = list(parsed.get("candidatos", []))
            parsed_url = normalize_ws(str(parsed.get("noticia", "") or ""))
            parsed_evidence = normalize_ws(str(parsed.get("evidencia", "") or ""))
            parsed_conf = normalize_confidence(str(parsed.get("confianca", "") or ""))
            if parsed_url:
                candidates.append(
                    {
                        "url": parsed_url,
                        "evidencia": parsed_evidence,
                        "confianca": parsed_conf,
                    }
                )
            if isinstance(citations, list):
                for c in citations:
                    url = normalize_ws(str(c))
                    if url:
                        candidates.append(
                            {
                                "url": url,
                                "evidencia": parsed_evidence,
                                "confianca": parsed_conf,
                            }
                        )
            candidates = dedupe_candidates(candidates)
            selected_url, selected_reason = select_best_noticia_candidate(
                row,
                candidates,
                min_score_mainstream=min_score_mainstream,
                min_score_official=min_score_official,
                domain_policy=domain_policy,
                stage=stage,
            )
            return True, selected_url, False, "", selected_reason
        except requests.Timeout as exc:
            saw_rate_limit = True
            last_err = str(exc)
            reason = "timeout"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            reason = "error"
            logger.debug("Perplexity erro tentativa %d/%d: %s", attempt, retries, exc)
        if attempt < retries:
            wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.35)
            time.sleep(wait)
    fallback_reason = "rate_limit" if saw_rate_limit else (reason or "error")
    return False, "", saw_rate_limit, last_err, fallback_reason


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def checkpoint_payload(
    manifest: Sequence[Dict[str, str]],
    rows: Sequence[Dict[str, str]],
    perplexity_state: Dict[str, Any],
) -> Dict[str, Any]:
    payload_rows = []
    for row in rows:
        obj = {k: row.get(k, "") for k in (CSV_COLUMNS + INTERNAL_KEYS)}
        payload_rows.append(obj)
    return {
        "version": CHECKPOINT_VERSION,
        "manifest": list(manifest),
        "rows": payload_rows,
        "perplexity_state": perplexity_state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(path)


def read_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def same_manifest(a: Sequence[Dict[str, str]], b: Sequence[Dict[str, str]]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.get("path") != y.get("path") or x.get("sha1") != y.get("sha1"):
            return False
    return True


def row_from_checkpoint(raw: Dict[str, Any]) -> Dict[str, str]:
    row = base_empty_row()
    for col in CSV_COLUMNS:
        if col in raw and raw.get(col) is not None:
            row[col] = str(raw.get(col))
    if is_missing(row.get("relator(a)")) and not is_missing(raw.get("relator")):
        row["relator(a)"] = str(raw.get("relator", ""))
    for key in INTERNAL_KEYS:
        row[key] = str(raw.get(key, "") if raw.get(key) is not None else "")
    if not row.get("_row_id"):
        row["_row_id"] = generate_row_id(row)
    return row


def discover_input_files(
    input_dirs: Sequence[str],
    input_files: Sequence[str],
    logger: logging.Logger,
) -> List[Path]:
    files: List[Path] = []

    for p in input_files:
        pp = Path(p).expanduser()
        if pp.exists() and pp.is_file() and pp.suffix.lower() == ".pdf":
            files.append(pp.resolve())

    for d in input_dirs:
        dd = Path(d).expanduser()
        if not dd.exists() or not dd.is_dir():
            continue
        files.extend(sorted(x.resolve() for x in dd.glob("*.pdf")))

    if not files:
        for year in ("2020", "2021", "2022", "2023", "2024", "2025", "2026"):
            dd = Path(year)
            if dd.exists() and dd.is_dir():
                files.extend(sorted(x.resolve() for x in dd.glob("*.pdf")))

    files = sorted(files, key=lambda p: (p.parent.name, p.name))
    if not files:
        return []

    # dedupe por hash SHA-1
    unique: List[Path] = []
    seen_hash: Dict[str, Path] = {}
    for f in files:
        try:
            h = sha1_file(f)
        except Exception as exc:
            logger.warning("Falha hash %s: %s", f, exc)
            continue
        if h in seen_hash:
            logger.info("PDF duplicado ignorado: %s (igual a %s)", f, seen_hash[h])
            continue
        seen_hash[h] = f
        unique.append(f)
    return unique


def compute_manifest(files: Sequence[Path]) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    for f in files:
        manifest.append({"path": str(f), "sha1": sha1_file(f)})
    return manifest


def count_list_items(csv_value: str) -> int:
    return len([x for x in split_multi_value(csv_value) if normalize_ws(x)])


def has_valid_openai_critical_fields(row: Dict[str, str]) -> bool:
    for col in OPENAI_CRITICAL_COLUMNS:
        if is_missing(row.get(col)):
            return False
    if not is_tema_detailed(row.get("tema", "")):
        return False
    if count_list_items(row.get("subramo_do_direito", "")) < 2:
        return False
    if count_list_items(row.get("bullet_points", "")) < 3:
        return False
    return True


def refresh_done_flags(rows: Sequence[Dict[str, str]]) -> None:
    for row in rows:
        row["_openai_done"] = "1" if has_valid_openai_critical_fields(row) else "0"
        if row.get("noticia") and validate_url(row.get("noticia", "")):
            row["_perplexity_done"] = "1"
            if not row.get("_perplexity_reason"):
                row["_perplexity_reason"] = "ok"
        else:
            row["_perplexity_done"] = "0"
            if not row.get("_perplexity_reason"):
                row["_perplexity_reason"] = ""


def needs_openai(row: Dict[str, str]) -> bool:
    return not has_valid_openai_critical_fields(row)


def needs_perplexity(row: Dict[str, str]) -> bool:
    return is_missing(row.get("noticia"))


def validate_rows(rows: Sequence[Dict[str, str]], logger: logging.Logger) -> None:
    date_re = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
    invalid_julg = 0
    invalid_pub = 0
    for row in rows:
        d = normalize_ws(row.get("data_julgamento", ""))
        if d and not date_re.match(d):
            invalid_julg += 1
        p = normalize_ws(row.get("data_publicacao", ""))
        if p and not date_re.match(p):
            invalid_pub += 1
    if invalid_julg:
        logger.warning("data_julgamento fora do padrão M/D/AAAA: %d linhas", invalid_julg)
    if invalid_pub:
        logger.warning("data_publicacao fora do padrão M/D/AAAA: %d linhas", invalid_pub)
    if not invalid_julg and not invalid_pub:
        logger.info("Validação: datas em padrão M/D/AAAA nas linhas não vazias.")


def build_quality_report(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    fill_rate: Dict[str, Dict[str, Any]] = {}
    for col in CSV_COLUMNS:
        filled = sum(1 for row in rows if not is_missing(row.get(col)))
        pct = (filled / total * 100.0) if total else 0.0
        fill_rate[col] = {"filled": filled, "missing": total - filled, "fill_pct": round(pct, 2)}

    starts_with_header_noise = 0
    multiple_unanime = 0
    citation_then_new_sentence = 0
    openai_done_but_missing_critical = 0
    data_julgamento_invalid = 0
    data_publicacao_invalid = 0
    bullet_blank_like = 0
    tema_too_short_count = 0
    perplexity_reasons: Dict[str, int] = {}
    perplexity_reject_by_reason: Dict[str, int] = {}
    perplexity_stage1_accept_count = 0
    perplexity_stage2_accept_count = 0

    for row in rows:
        texto_ref = row.get("texto_do_boletim", "")
        if HEADER_NOISE_PREFIX_RE.search(texto_ref):
            starts_with_header_noise += 1
        if len(re.findall(r"\bun[âa]nime\b", texto_ref, flags=re.IGNORECASE)) >= 2:
            multiple_unanime += 1
        if CITATION_THEN_CASE_RE.search(texto_ref):
            citation_then_new_sentence += 1
        if row.get("_openai_done") == "1" and not has_valid_openai_critical_fields(row):
            openai_done_but_missing_critical += 1
        dj = normalize_ws(row.get("data_julgamento", ""))
        if dj and not re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", dj):
            data_julgamento_invalid += 1
        dp = normalize_ws(row.get("data_publicacao", ""))
        if dp and not re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", dp):
            data_publicacao_invalid += 1
        if re.fullmatch(r"\s*,\s*,\s*", row.get("bullet_points", "")):
            bullet_blank_like += 1
        tema_wc = tema_word_count(row.get("tema", ""))
        if tema_wc and tema_wc < TEMA_MIN_WORDS:
            tema_too_short_count += 1
        reason = normalize_ws(row.get("_perplexity_reason", ""))
        if reason:
            perplexity_reasons[reason] = perplexity_reasons.get(reason, 0) + 1
            if reason.startswith("accepted_stage1_"):
                perplexity_stage1_accept_count += 1
            elif reason.startswith("accepted_stage2_"):
                perplexity_stage2_accept_count += 1
            else:
                perplexity_reject_by_reason[reason] = perplexity_reject_by_reason.get(reason, 0) + 1

    openai_done = sum(1 for row in rows if row.get("_openai_done") == "1")
    pplx_done = sum(1 for row in rows if row.get("_perplexity_done") == "1")
    noticia_fill_rate_pct = round((pplx_done / total * 100.0), 2) if total else 0.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": total,
        "fill_rate": fill_rate,
        "metrics": {
            "starts_with_oletim_nformativo": starts_with_header_noise,
            "multiple_unanime": multiple_unanime,
            "citation_then_new_sentence": citation_then_new_sentence,
            "openai_done_but_missing_critical": openai_done_but_missing_critical,
            "data_julgamento_invalid": data_julgamento_invalid,
            "data_publicacao_invalid": data_publicacao_invalid,
            "bullet_blank_like": bullet_blank_like,
            "tema_too_short_count": tema_too_short_count,
            "openai_done": openai_done,
            "perplexity_done": pplx_done,
            "noticia_fill_rate_pct": noticia_fill_rate_pct,
            "perplexity_stage1_accept_count": perplexity_stage1_accept_count,
            "perplexity_stage2_accept_count": perplexity_stage2_accept_count,
            "perplexity_reject_by_reason": perplexity_reject_by_reason,
            "perplexity_empty_or_error_reasons": perplexity_reasons,
        },
    }


def write_quality_report(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    report = build_quality_report(rows)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def gui_select_inputs(logger: logging.Logger) -> Tuple[List[str], List[str]]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:
        logger.warning("GUI indisponível (tkinter): %s", exc)
        return [], []

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    mode = messagebox.askyesnocancel(
        "TRF1 - Seleção de PDFs",
        "Sim: selecionar uma pasta inteira.\nNão: selecionar PDFs específicos.\nCancelar: usar modo padrão (pastas 2020..2026).",
    )
    selected_dirs: List[str] = []
    selected_files: List[str] = []
    if mode is True:
        seen_dirs: set[str] = set()
        while True:
            folder = filedialog.askdirectory(title="Selecione uma pasta com PDFs do TRF1")
            if folder:
                norm = str(Path(folder).expanduser().resolve())
                if norm not in seen_dirs:
                    seen_dirs.add(norm)
                    selected_dirs.append(norm)
            if not folder and not selected_dirs:
                # Usuário cancelou sem selecionar pasta.
                break
            add_more = messagebox.askyesno(
                "TRF1 - Seleção de pastas",
                "Deseja adicionar outra pasta?",
            )
            if not add_more:
                break
    elif mode is False:
        files = filedialog.askopenfilenames(
            title="Selecione PDFs específicos do TRF1",
            filetypes=[("PDF", "*.pdf"), ("Todos os arquivos", "*.*")],
        )
        selected_files = list(files)
    root.destroy()
    return selected_dirs, selected_files


@dataclass
class OpenAIConfig:
    enabled: bool
    api_key: str
    model: str
    batch_size: int
    max_workers: int
    delay: float
    retries: int
    timeout: int
    max_workers_cap: int = 14
    max_completion_tokens: int = 700
    text_max_chars: int = 0


@dataclass
class PerplexityConfig:
    enabled: bool
    api_key: str
    model: str
    batch_size: int
    max_workers: int
    delay: float
    retries: int
    timeout: int
    max_workers_cap: int
    fallback_enabled: bool
    fallback_max_fraction: float
    min_score_mainstream: int
    min_score_official: int
    domain_policy: str
    max_tokens: int = 280
    text_max_chars: int = 0


def run_openai_enrichment(
    rows: List[Dict[str, str]],
    config: OpenAIConfig,
    output_csv: Path,
    checkpoint_path: Path,
    manifest: Sequence[Dict[str, str]],
    perplexity_state: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    if not config.enabled:
        return
    pending_all = [r for r in rows if r.get("_openai_done", "0") != "1" and needs_openai(r)]
    # Evita chamadas repetidas para linhas idênticas no mesmo run.
    dedupe_map: Dict[str, Dict[str, str]] = {}
    duplicates_by_primary: Dict[str, List[Dict[str, str]]] = {}
    pending: List[Dict[str, str]] = []
    for row in pending_all:
        key = build_openai_request_key(row, config.text_max_chars)
        primary = dedupe_map.get(key)
        if primary is None:
            dedupe_map[key] = row
            pending.append(row)
            continue
        primary_id = primary.get("_row_id") or str(id(primary))
        duplicates_by_primary.setdefault(primary_id, []).append(row)

    if not pending:
        logger.info("OpenAI: nada pendente.")
        return
    duplicates_count = len(pending_all) - len(pending)
    logger.info(
        "OpenAI: %d linhas pendentes (%d chamadas únicas; %d duplicadas reaproveitadas).",
        len(pending_all),
        len(pending),
        max(0, duplicates_count),
    )
    client = OpenAI(api_key=config.api_key)
    workers = max(1, int(config.max_workers))
    workers_cap = max(workers, int(config.max_workers_cap))
    delay = max(0.0, float(config.delay))
    total = len(pending)
    done = 0

    def apply_payload(row: Dict[str, str], payload: Dict[str, str]) -> None:
        for col, value in payload.items():
            if is_missing(row.get(col)):
                row[col] = value
        row["subramo_do_direito"] = normalize_subramo_two(row.get("subramo_do_direito", ""))
        row["bullet_points"] = normalize_bullet_points_three(row.get("bullet_points", ""))
        row["_openai_done"] = "1" if has_valid_openai_critical_fields(row) else "0"

    for start in range(0, total, config.batch_size):
        end = min(start + config.batch_size, total)
        batch = pending[start:end]
        batch_started = time.perf_counter()
        logger.info(
            "OpenAI lote %d-%d/%d | workers=%d delay=%.2fs",
            start + 1,
            end,
            total,
            workers,
            delay,
        )
        futures = {}
        batch_errors = 0
        batch_rate_limited = 0
        batch_timeouts = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for row in batch:
                futures[
                    ex.submit(
                        openai_call_single,
                        client,
                        config.model,
                        row,
                        config.timeout,
                        config.retries,
                        config.max_completion_tokens,
                        config.text_max_chars,
                        logger,
                    )
                ] = row

            for fut in as_completed(futures):
                row = futures[fut]
                ok, payload, err = fut.result()
                if ok:
                    apply_payload(row, payload)
                    if row["_openai_done"] != "1":
                        logger.debug("OpenAI retorno parcial row=%s", row.get("_row_id"))
                else:
                    row["_openai_done"] = "0"
                    batch_errors += 1
                    kind = classify_openai_error(err)
                    if kind == "rate_limit":
                        batch_rate_limited += 1
                    elif kind == "timeout":
                        batch_timeouts += 1
                    logger.debug("OpenAI falha row=%s: %s", row.get("_row_id"), err)
                done += 1
                primary_id = row.get("_row_id") or str(id(row))
                for dup in duplicates_by_primary.get(primary_id, []):
                    if ok:
                        apply_payload(dup, payload)
                    else:
                        dup["_openai_done"] = "0"
                    done += 1

        elapsed = max(0.001, time.perf_counter() - batch_started)
        throughput = len(batch) / elapsed
        error_ratio = batch_errors / max(1, len(batch))
        if batch_rate_limited > 0 or error_ratio >= 0.25:
            workers = max(1, workers - 1)
            delay = min(2.5, delay * 1.35 + 0.05)
            logger.info("OpenAI autoajuste: reduzindo workers=%d delay=%.2f", workers, delay)
        elif batch_errors == 0 and workers < workers_cap:
            workers += 1
            delay = max(0.0, delay * 0.9)
            logger.info("OpenAI autoajuste: aumentando workers=%d delay=%.2f", workers, delay)
        logger.info(
            "OpenAI lote concluído: %.2f req/s | erros=%d timeout=%d",
            throughput,
            batch_errors,
            batch_timeouts,
        )

        write_csv(output_csv, rows)
        write_checkpoint(
            checkpoint_path,
            checkpoint_payload(manifest=manifest, rows=rows, perplexity_state=perplexity_state),
        )
        logger.info("OpenAI progresso: %d/%d", done, len(pending_all))
        if end < total and delay > 0:
            time.sleep(delay)


def run_perplexity_enrichment(
    rows: List[Dict[str, str]],
    config: PerplexityConfig,
    output_csv: Path,
    checkpoint_path: Path,
    manifest: Sequence[Dict[str, str]],
    perplexity_state: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    if not config.enabled:
        return perplexity_state
    pending = [r for r in rows if r.get("_perplexity_done", "0") != "1" and needs_perplexity(r)]
    if not pending:
        logger.info("Perplexity: nada pendente.")
        return perplexity_state

    session = requests.Session()
    pool_size = max(16, int(config.max_workers_cap) * 2)
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
    )

    workers = int(perplexity_state.get("workers", config.max_workers))
    delay = float(perplexity_state.get("delay", config.delay))

    total = len(pending)
    logger.info("Perplexity: %d linhas pendentes.", total)

    def apply_perplexity_result(row: Dict[str, str], url: str, reason: str, stage: int) -> None:
        row["noticia"] = url
        row["_perplexity_done"] = "1" if url else "0"
        row["_perplexity_reason"] = reason or (f"no_candidates_stage{stage}")

    def process_perplexity_stage(stage_rows: List[Dict[str, str]], stage: int) -> int:
        nonlocal workers, delay
        if not stage_rows:
            return 0

        stage_total = len(stage_rows)
        dedupe_map: Dict[str, Dict[str, str]] = {}
        duplicates_by_primary: Dict[str, List[Dict[str, str]]] = {}
        stage_unique: List[Dict[str, str]] = []
        for row in stage_rows:
            key = build_perplexity_request_key(row, config.text_max_chars)
            primary = dedupe_map.get(key)
            if primary is None:
                dedupe_map[key] = row
                stage_unique.append(row)
                continue
            primary_id = primary.get("_row_id") or str(id(primary))
            duplicates_by_primary.setdefault(primary_id, []).append(row)

        stage_call_total = len(stage_unique)
        stage_done = 0
        stage_name = f"stage{stage}"
        logger.info(
            "Perplexity %s: %d linhas (%d chamadas únicas).",
            stage_name,
            stage_total,
            stage_call_total,
        )

        for start in range(0, stage_call_total, config.batch_size):
            end = min(start + config.batch_size, stage_call_total)
            batch = stage_unique[start:end]
            batch_started = time.perf_counter()
            logger.info(
                "Perplexity %s lote %d-%d/%d | workers=%d delay=%.2fs",
                stage_name,
                start + 1,
                end,
                stage_call_total,
                workers,
                delay,
            )
            batch_rate_limited = 0
            batch_errors = 0
            futures = {}
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                for row in batch:
                    futures[
                        ex.submit(
                            perplexity_call_single,
                            session,
                            config.model,
                            row,
                            stage=stage,
                            min_score_mainstream=config.min_score_mainstream,
                            min_score_official=config.min_score_official,
                            domain_policy=config.domain_policy,
                            timeout=config.timeout,
                            retries=config.retries,
                            max_tokens=config.max_tokens,
                            text_max_chars=config.text_max_chars,
                            logger=logger,
                        )
                    ] = row

                for fut in as_completed(futures):
                    row = futures[fut]
                    ok, url, rate_limited, err, reason = fut.result()
                    if ok:
                        apply_perplexity_result(row, url, reason, stage)
                    else:
                        batch_errors += 1
                        row["_perplexity_done"] = "0"
                        row["_perplexity_reason"] = reason or "error"
                        if rate_limited:
                            batch_rate_limited += 1
                        logger.debug("Perplexity falha row=%s stage=%s: %s", row.get("_row_id"), stage, err)
                    stage_done += 1
                    primary_id = row.get("_row_id") or str(id(row))
                    for dup in duplicates_by_primary.get(primary_id, []):
                        if ok:
                            apply_perplexity_result(dup, url, reason, stage)
                        else:
                            dup["_perplexity_done"] = "0"
                            dup["_perplexity_reason"] = reason or "error"
                        stage_done += 1

            error_ratio = batch_errors / max(1, len(batch))
            if batch_rate_limited > 0 or error_ratio >= 0.25:
                workers = max(1, workers - 1)
                delay = min(5.0, delay * 1.5 + 0.1)
                logger.info("Perplexity autoajuste: reduzindo workers=%d delay=%.2f", workers, delay)
            elif batch_errors == 0 and workers < config.max_workers_cap:
                workers += 1
                delay = max(config.delay, delay * 0.9)
                logger.info("Perplexity autoajuste: aumentando workers=%d delay=%.2f", workers, delay)

            elapsed = max(0.001, time.perf_counter() - batch_started)
            throughput = len(batch) / elapsed
            perplexity_state["workers"] = workers
            perplexity_state["delay"] = delay

            write_csv(output_csv, rows)
            write_checkpoint(
                checkpoint_path,
                checkpoint_payload(manifest=manifest, rows=rows, perplexity_state=perplexity_state),
            )
            logger.info(
                "Perplexity %s lote concluído: %.2f req/s | erros=%d",
                stage_name,
                throughput,
                batch_errors,
            )
            logger.info("Perplexity %s progresso: %d/%d", stage_name, stage_done, stage_total)
            if end < stage_call_total and delay > 0:
                time.sleep(delay)
        return stage_done

    process_perplexity_stage(pending, stage=1)

    fallback_budget = int(math.ceil(total * max(0.0, min(1.0, config.fallback_max_fraction))))
    if config.fallback_enabled and fallback_budget > 0:
        fallback_pool = [
            row
            for row in pending
            if row.get("_perplexity_done", "0") != "1"
            and normalize_ws(row.get("_perplexity_reason", "")) in PERPLEXITY_REJECTABLE_STAGE1_REASONS
        ]
        fallback_rows = fallback_pool[:fallback_budget]
        if fallback_rows:
            logger.info(
                "Perplexity fallback: %d elegíveis | budget=%d | executando=%d",
                len(fallback_pool),
                fallback_budget,
                len(fallback_rows),
            )
            process_perplexity_stage(fallback_rows, stage=2)
        else:
            logger.info("Perplexity fallback: sem linhas elegíveis.")
    elif config.fallback_enabled:
        logger.info("Perplexity fallback: budget=0, nenhuma linha reprocessada.")

    session.close()
    return perplexity_state


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TRF1 PDF para CSV híbrido com OpenAI + Perplexity (retoma checkpoint compatível).",
    )
    p.add_argument("--no-gui", action="store_true", help="Desativa GUI e usa apenas argumentos CLI.")
    p.add_argument("--input-dirs", nargs="*", default=[], help="Pastas contendo PDFs.")
    p.add_argument("--input-files", nargs="*", default=[], help="Lista de PDFs específicos.")
    p.add_argument("--output-dir", default=".", help="Pasta de saída do CSV/checkpoint.")

    p.add_argument("--verbose", action="store_true", help="Log detalhado.")
    p.add_argument("--quiet", action="store_true", help="Reduz logs.")

    p.add_argument("--openai-api-key", default="", help="API key OpenAI.")
    p.add_argument("--openai-model", default="gpt-5-mini", help="Modelo OpenAI.")
    p.add_argument("--openai-batch-size", type=int, default=30)
    p.add_argument("--openai-max-workers", type=int, default=10)
    p.add_argument("--openai-max-workers-cap", type=int, default=14)
    p.add_argument("--openai-delay", type=float, default=0.05)
    p.add_argument("--openai-retries", type=int, default=3)
    p.add_argument("--openai-timeout", type=int, default=50)
    p.add_argument("--openai-max-completion-tokens", type=int, default=700)
    p.add_argument(
        "--openai-text-max-chars",
        type=int,
        default=0,
        help="Máximo de caracteres do texto_do_boletim no prompt OpenAI (0 = sem corte).",
    )
    p.add_argument("--disable-openai", action="store_true")

    p.add_argument("--perplexity-api-key", default="", help="API key Perplexity.")
    p.add_argument("--perplexity-model", default="sonar", help="Modelo Perplexity.")
    p.add_argument("--perplexity-batch-size", type=int, default=20)
    p.add_argument("--perplexity-max-workers", type=int, default=8)
    p.add_argument("--perplexity-max-workers-cap", type=int, default=12)
    p.add_argument("--perplexity-delay", type=float, default=0.25)
    p.add_argument("--perplexity-retries", type=int, default=3)
    p.add_argument("--perplexity-timeout", type=int, default=30)
    p.add_argument("--perplexity-max-tokens", type=int, default=280)
    p.add_argument(
        "--perplexity-text-max-chars",
        type=int,
        default=0,
        help="Máximo de caracteres do texto_do_boletim no prompt Perplexity (0 = sem corte).",
    )
    p.add_argument(
        "--perplexity-fallback-enabled",
        dest="perplexity_fallback_enabled",
        action="store_true",
        default=True,
        help="Habilita fallback da Perplexity em 2ª etapa para linhas sem match.",
    )
    p.add_argument(
        "--no-perplexity-fallback",
        dest="perplexity_fallback_enabled",
        action="store_false",
        help="Desabilita fallback da Perplexity.",
    )
    p.add_argument("--perplexity-fallback-max-fraction", type=float, default=0.60)
    p.add_argument("--perplexity-min-score-mainstream", type=int, default=2)
    p.add_argument("--perplexity-min-score-official", type=int, default=1)
    p.add_argument(
        "--perplexity-domain-policy",
        default=PERPLEXITY_DEFAULT_DOMAIN_POLICY,
        choices=list(PERPLEXITY_DOMAIN_POLICY_CHOICES),
    )
    p.add_argument("--disable-perplexity", action="store_true")
    return p


def setup_logger(verbose: bool, quiet: bool) -> logging.Logger:
    logger = logging.getLogger("trf1_hibrido")
    logger.handlers.clear()
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def ensure_output_rows_format(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        for col in CSV_COLUMNS + INTERNAL_KEYS:
            row.setdefault(col, "")
        row["classe"] = normalize_ws(row.get("classe", ""))
        row["texto_do_boletim"] = sanitize_boletim_text(row.get("texto_do_boletim", ""))
        row["decisao"] = normalize_ws(row.get("decisao", "")) or parse_decisao(row.get("texto_do_boletim", ""))
        if is_missing(row.get("tema")):
            tema_fallback = infer_tema_from_texto(row.get("texto_do_boletim", ""))
            row["tema"] = tema_fallback if tema_fallback else "Tema não identificado automaticamente"
        row["data_julgamento"] = normalize_date_to_mdy(row.get("data_julgamento", ""), source_hint="auto")
        row["data_publicacao"] = normalize_date_to_mdy(row.get("data_publicacao", ""), source_hint="auto")
        row["relator(a)"] = normalize_relator(row.get("relator(a)", "") or row.get("relator", ""))
        row["subramo_do_direito"] = normalize_subramo_two(row.get("subramo_do_direito", ""))
        row["bullet_points"] = normalize_bullet_points_three(row.get("bullet_points", ""))
        row["legislacao"] = normalize_legislacao(row.get("legislacao", ""))
        row["jurisprudência"] = normalize_jurisprudencia(row.get("jurisprudência", ""))
        row["punchline"] = truncate_to_words(row.get("punchline", ""), 20)
        if row.get("noticia") and not validate_url(row.get("noticia", "")):
            row["noticia"] = ""
        row["_perplexity_reason"] = normalize_ws(row.get("_perplexity_reason", ""))
    refresh_done_flags(rows)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logger = setup_logger(args.verbose, args.quiet)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / OUTPUT_NAME
    checkpoint_path = output_dir / CHECKPOINT_NAME
    quality_report_path = output_dir / QUALITY_REPORT_NAME

    input_dirs = list(args.input_dirs or [])
    input_files = list(args.input_files or [])
    if not args.no_gui and not input_dirs and not input_files:
        gui_dirs, gui_files = gui_select_inputs(logger)
        input_dirs = gui_dirs or input_dirs
        input_files = gui_files or input_files

    files = discover_input_files(input_dirs=input_dirs, input_files=input_files, logger=logger)
    if not files:
        raise SystemExit("Nenhum PDF encontrado para processar.")
    logger.info("PDFs selecionados: %d", len(files))

    manifest = compute_manifest(files)
    rows: List[Dict[str, str]] = []
    perplexity_state: Dict[str, Any] = {"workers": args.perplexity_max_workers, "delay": args.perplexity_delay}
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint:
        cp_version = int(checkpoint.get("version", 0) or 0)
        cp_manifest = checkpoint.get("manifest", [])
        if cp_version == CHECKPOINT_VERSION and isinstance(cp_manifest, list) and same_manifest(cp_manifest, manifest):
            saved_rows = checkpoint.get("rows", [])
            if isinstance(saved_rows, list) and saved_rows:
                rows = [row_from_checkpoint(r) for r in saved_rows if isinstance(r, dict)]
                if isinstance(checkpoint.get("perplexity_state"), dict):
                    perplexity_state.update(checkpoint.get("perplexity_state", {}))
                logger.info("Checkpoint compatível carregado: %d linhas. Retomando processamento.", len(rows))
            else:
                logger.info("Checkpoint encontrado, mas sem linhas úteis. Reextraindo base.")
        else:
            logger.info("Checkpoint ignorado (incompatível com versão atual ou conjunto de PDFs).")

    if not rows:
        logger.info("Modo clean-run ativo: iniciando extração base sem reaproveitamento de CSV legado.")
        extracted: List[Dict[str, str]] = []
        for idx, pdf in enumerate(files, start=1):
            logger.info("Extraindo %d/%d: %s", idx, len(files), pdf.name)
            extracted.extend(extract_rows_from_pdf(pdf, logger))
        extracted = deduplicate_rows(extracted, logger)
        rows = extracted
        ensure_output_rows_format(rows)
        write_csv(output_csv, rows)
        write_checkpoint(
            checkpoint_path,
            checkpoint_payload(manifest=manifest, rows=rows, perplexity_state=perplexity_state),
        )
        logger.info("Extração base concluída: %d julgados.", len(rows))

    ensure_output_rows_format(rows)

    openai_key = resolve_openai_key(args.openai_api_key)
    perplexity_key = resolve_perplexity_key(args.perplexity_api_key)

    openai_cfg = OpenAIConfig(
        enabled=(not args.disable_openai) and bool(openai_key),
        api_key=openai_key,
        model=args.openai_model.strip() or "gpt-5-mini",
        batch_size=max(1, int(args.openai_batch_size)),
        max_workers=max(1, int(args.openai_max_workers)),
        max_workers_cap=max(1, int(args.openai_max_workers_cap)),
        delay=max(0.0, float(args.openai_delay)),
        retries=max(1, int(args.openai_retries)),
        timeout=max(5, int(args.openai_timeout)),
        max_completion_tokens=max(0, int(args.openai_max_completion_tokens)),
        text_max_chars=max(0, int(args.openai_text_max_chars)),
    )
    perplexity_cfg = PerplexityConfig(
        enabled=(not args.disable_perplexity) and bool(perplexity_key),
        api_key=perplexity_key,
        model=args.perplexity_model.strip() or "sonar",
        batch_size=max(1, int(args.perplexity_batch_size)),
        max_workers=max(1, int(args.perplexity_max_workers)),
        delay=max(0.0, float(args.perplexity_delay)),
        retries=max(1, int(args.perplexity_retries)),
        timeout=max(5, int(args.perplexity_timeout)),
        max_workers_cap=max(1, int(args.perplexity_max_workers_cap)),
        fallback_enabled=bool(args.perplexity_fallback_enabled),
        fallback_max_fraction=max(0.0, float(args.perplexity_fallback_max_fraction)),
        min_score_mainstream=max(0, int(args.perplexity_min_score_mainstream)),
        min_score_official=max(0, int(args.perplexity_min_score_official)),
        domain_policy=args.perplexity_domain_policy,
        max_tokens=max(0, int(args.perplexity_max_tokens)),
        text_max_chars=max(0, int(args.perplexity_text_max_chars)),
    )

    if not openai_cfg.enabled and not args.disable_openai:
        logger.warning("OpenAI desabilitado por ausência de API key.")
    if not perplexity_cfg.enabled and not args.disable_perplexity:
        logger.warning("Perplexity desabilitado por ausência de API key.")

    run_openai_enrichment(
        rows=rows,
        config=openai_cfg,
        output_csv=output_csv,
        checkpoint_path=checkpoint_path,
        manifest=manifest,
        perplexity_state=perplexity_state,
        logger=logger,
    )

    perplexity_state = run_perplexity_enrichment(
        rows=rows,
        config=perplexity_cfg,
        output_csv=output_csv,
        checkpoint_path=checkpoint_path,
        manifest=manifest,
        perplexity_state=perplexity_state,
        logger=logger,
    )

    ensure_output_rows_format(rows)
    write_csv(output_csv, rows)
    write_checkpoint(
        checkpoint_path,
        checkpoint_payload(manifest=manifest, rows=rows, perplexity_state=perplexity_state),
    )
    write_quality_report(quality_report_path, rows)
    report = build_quality_report(rows)
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    logger.info(
        (
            "Métricas-chave: tema_curto=%s | noticia_fill_rate_pct=%s | "
            "pplx_stage1_accept=%s | pplx_stage2_accept=%s"
        ),
        metrics.get("tema_too_short_count", 0),
        metrics.get("noticia_fill_rate_pct", 0),
        metrics.get("perplexity_stage1_accept_count", 0),
        metrics.get("perplexity_stage2_accept_count", 0),
    )
    validate_rows(rows, logger)

    total = len(rows)
    openai_done = sum(1 for r in rows if r.get("_openai_done", "0") == "1")
    pplx_done = sum(1 for r in rows if r.get("_perplexity_done", "0") == "1")
    logger.info("Concluído. Linhas: %d | OpenAI completas: %d | Perplexity completas: %d", total, openai_done, pplx_done)
    logger.info("CSV final: %s", output_csv)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Relatório de qualidade: %s", quality_report_path)


if __name__ == "__main__":
    main()
