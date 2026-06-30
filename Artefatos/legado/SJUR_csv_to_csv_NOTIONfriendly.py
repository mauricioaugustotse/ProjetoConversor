#!/usr/bin/env python3
"""Converte CSVs do SJUR para layout amigável ao Notion.

Fluxo de trabalho:
1. Lê o CSV de entrada e normaliza encoding, delimitador, cabeçalhos e células.
2. Aplica padronizações jurídicas por coluna (CNJ, partes, composição e resultado).
3. Remove colunas descartadas e alinha o esquema final para importação no Notion.
4. Opcionalmente preenche `tema`/`punchline` (OpenAI) e URLs de notícia (Gemini/Perplexity).
5. Gera CSVs de saída por arquivo e um consolidado, com checkpoint e backup.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import html
import hashlib
import json
import os
import logging
import re
import sys
import time
import threading
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence
from urllib.parse import urlparse

import requests
from Artefatos.scripts.openai_progress_utils import (
    build_file_signature,
    make_backup,
    read_json_dict,
    same_file_signature,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
)
from Artefatos.scripts.openai_log_utils import configure_standard_logging, install_print_logger_bridge

# Decisoes do DJe podem ter textoDecisao/textoEmenta acima do limite padrao do
# modulo csv (128 KB), causando "field larger than field limit (131072)" na
# leitura dos CSVs brutos. Eleva o limite ao maximo suportado pela plataforma
# (no Windows o C long e 32 bits, entao sys.maxsize estoura e decaimos ate caber).
_CSV_FIELD_LIMIT = sys.maxsize
while True:
    try:
        csv.field_size_limit(_CSV_FIELD_LIMIT)
        break
    except OverflowError:
        _CSV_FIELD_LIMIT //= 10

ENCODINGS_TO_TRY = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
SPACE_RE = re.compile(r"\s+")
HEADER_ALLOWED_RE = re.compile(r"[^0-9A-Za-z_]+")
PARTES_SPLIT_RE = re.compile(r"[;,]")
RELATOR_PREFIX_RE = re.compile(r"^\s*relator(?:\(a\)|a|o)?\s*[:.\-]?\s*", re.IGNORECASE)
GENERAL_NEWS_COLUMNS = [f"noticia_geral_{i}" for i in range(1, 3)]
REMOVED_GENERAL_NEWS_COLUMNS = [f"noticia_geral_{i}" for i in range(3, 10)]
EXCLUDED_COLUMNS = {
    "numeroDecisao",
    "numeroProtocolo",
    "naturezaDocumento",
    "resuultado",
    *REMOVED_GENERAL_NEWS_COLUMNS,
}
OUTPUT_HEADER_REMAP = {"relatores": "relator"}
URL_COLUMNS = ["noticia_TSE", "noticia_TRE", *GENERAL_NEWS_COLUMNS]
THEME_COLUMNS = ["tema", "punchline"]
DEFAULT_PRESERVE_COLUMNS = [*THEME_COLUMNS, *URL_COLUMNS]
ROW_PROGRESS_EVERY = 100
DEFAULT_PERPLEXITY_KEY_FILE = "Chave_secreta_Perplexity.txt"
DEFAULT_GEMINI_KEY_FILE = "Chave_Gemini.txt"
DEFAULT_OPENAI_KEY_FILE = "Chave_secreta_OpenAI.txt"
OPENAI_KEY_FALLBACK_FILES = (
    "CHAVE_SECRETA_API_Mauricio_local.txt",
    "Chave Secreta API_Mauricio_local.txt",
    "Chave_Secreta_API_Mauricio_local.txt",
    "chave_secreta_api_mauricio_local.txt",
)
PERPLEXITY_KEY_FALLBACK_FILES = (
    "Chave_secreta_Perplexity.txt",
    "Chave secreta Perplexity.txt",
    "chave_secreta_perplexity.txt",
)
GEMINI_KEY_FALLBACK_FILES = (
    "Chave_Gemini.txt",
    "Chave_Google_API.txt",
    "chave_gemini.txt",
    "chave_google_api.txt",
    "../JULES-IA/Chave_Gemini.txt",
    "../JULES-IA/Chave_Google_API.txt",
    "../OneDrive/Documentos/API Key Gemini GitHub.txt",
)
OPENAI_DEFAULT_MAX_WORKERS = 10
OPENAI_DEFAULT_TIMEOUT = 45
OPENAI_DEFAULT_BATCH_SIZE = 40
OPENAI_DEFAULT_DELAY = 0.05
OPENAI_DEFAULT_RETRIES = 3
OPENAI_DEFAULT_TARGET_RPM = 180
DEFAULT_COMBINED_MULTI = "DJe_consolidado.csv"
CHECKPOINT_VERSION = 3
PERPLEXITY_NEWS_STRATEGY_VERSION = 10
PERPLEXITY_DEFAULT_MAX_TOKENS = 180
PERPLEXITY_EMENTA_CONTEXT_MAX_CHARS = 700
PERPLEXITY_DECISAO_CONTEXT_MAX_CHARS = 900
PERPLEXITY_DEFAULT_TARGET_RPM = 240
PERPLEXITY_DEFAULT_RETRIES = 3
DEFAULT_GEMINI_NEWS_MODEL = os.getenv("GEMINI_NEWS_MODEL") or "gemini-3.1-flash-lite"
GEMINI_DEFAULT_TIMEOUT = 25
GEMINI_DEFAULT_MAX_WORKERS = 2
GEMINI_DEFAULT_BATCH_SIZE = 10
GEMINI_DEFAULT_DELAY = 0.2
GEMINI_DEFAULT_MAX_OUTPUT_TOKENS = 900
GEMINI_DEFAULT_TARGET_RPM = 40
GEMINI_DEFAULT_RETRIES = 1
GEMINI_REST_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
PERPLEXITY_SEARCH_MAX_RESULTS = 5
PERPLEXITY_SEARCH_MAX_TOKENS_PER_PAGE = 256
PERPLEXITY_PAGE_FETCH_TIMEOUT = 12
PERPLEXITY_PAGE_TEXT_MAX_CHARS = 18000
PERPLEXITY_PROGRESS_LOG_INTERVAL_SECONDS = 1.5
PERPLEXITY_CACHE_FILENAME = f".sjur_perplexity_news_cache.v{PERPLEXITY_NEWS_STRATEGY_VERSION}.json"
# TTLs do cache de noticias: "filled" e permanente; "no_match" e reavaliado apos o TTL
# (a noticia pode ser publicada depois da decisao); erro retryable expira rapido.
NEWS_CACHE_NO_MATCH_TTL_DAYS = 7
NEWS_CACHE_RETRYABLE_TTL_HOURS = 24
NEWS_PROMPT_VALUE_MAX_CHARS = 360
LOGGER = logging.getLogger("SJUR_csv_to_csv_NOTIONfriendly")
SCRIPT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = (SCRIPT_DIR / "Artefatos").resolve()
# CSVs de origem SJUR/DJe usam datas no padrao brasileiro (D/M/AAAA).
CSV_INPUT_DATES_DAY_FIRST = True
EXCLUDED_COLUMNS_NORMALIZED = {
    re.sub(r"[_\s]+", "", name).casefold()
    for name in (
        *EXCLUDED_COLUMNS,
        "natureza_documento",
        "naturezadocumento",
    )
}

CANONICAL_MINISTERS: list[tuple[str, tuple[str, ...]]] = [
    ("Min. Cármen Lúcia", ("carmen lucia",)),
    ("Min. Nunes Marques", ("nunes marques",)),
    ("Min. André Mendonça", ("andre mendonca",)),
    ("Min. Antonio Carlos Ferreira", ("antonio carlos ferreira",)),
    ("Min. Villas Bôas Cueva", ("villas boas cueva", "vilas boas cueva")),
    ("Min. Floriano de Azevedo Marques", ("floriano de azevedo marques",)),
    ("Min. Estela Aranha", ("estela aranha",)),
]

COMPOSICAO_TRIGGER_RE = re.compile(r"composi[çc][aã]o\s*:|acompanharam o relator", re.IGNORECASE)
COMPOSICAO_SEGMENT_RE = re.compile(
    r"composi[çc][aã]o\s*:[^.]{0,800}|acompanharam o relator[^.]{0,800}",
    re.IGNORECASE,
)
ADVOGADOS_BLOCK_RE = re.compile(
    r"(?is)advogad(?:o|a|os|as)\s*:\s*(.*?)(?=(?:\b(?:advogad(?:o|a|os|as)|"
    r"recorrente|recorrido|recorrida|agravante|agravado|agravada|representante|"
    r"executad[oa]s?|interessad[oa]s?|impetrante|impetrado|requerente|requerid[oa]s?|"
    r"relator(?:a)?|minist[eé]rio p[úu]blico(?: eleitoral| federal)?|"
    r"procuradoria-?geral eleitoral|decis[aã]o|ac[oó]rd[aã]o|composi[çc][aã]o|ementa)\b\s*:|"
    r"\b(?:decis[aã]o|ac[oó]rd[aã]o|ementa)\b|$))"
)
ADVOGADO_NOME_OAB_RE = re.compile(
    r"([A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]+(?:\s+[A-ZÀ-ÖØ-Ýa-zà-öø-ÿ'`´^~.-]+){1,}?)"
    r"\s*(?:[-–—]\s*)?OAB\b",
    re.UNICODE,
)
ADVOGADO_TRAILING_ROLE_RE = re.compile(
    r"(?is)\b(?:recorrente|recorrid[oa]s?|agravant[ea]s?|agravad[oa]s?|representante|"
    r"executad[oa]s?|interessad[oa]s?|impetrante|impetrado|requerente|requerid[oa]s?|"
    r"minist[eé]rio p[úu]blico(?: eleitoral| federal)?|procuradoria-?geral eleitoral|"
    r"decis[aã]o|ac[oó]rd[aã]o|ementa)\b.*$"
)
ADVOGADO_OUTROS_SUFFIX_RE = re.compile(r"(?is)\s+e\s+outr[oa]s?\b.*$")
ADVOGADO_CANDIDATE_NAME_RE = re.compile(
    r"([A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]+(?:\s+(?:"
    r"[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]+|da|de|do|das|dos|e|d[aeo])){1,10})"
)
ADVOGADO_BAD_PREFIX_RE = re.compile(
    r"(?i)^(?:recorr|agrav|represent|executad|interessad|impetr|requer|decis|ac[oó]rd|ementa|"
    r"minist[eé]rio|procuradoria)"
)
ADVOGADO_BAD_ENTITY_RE = re.compile(
    r"(?i)\b(?:partido|coliga[cç][aã]o|minist[eé]rio|procuradoria|advocacia-?geral|"
    r"uni[aã]o brasil|nacional|municipal)\b"
)
LEG_SPLIT_RE = re.compile(r"\s*,\s*(?=LEG\.\s*:)", re.IGNORECASE)
LEGAL_ENTRY_RE = re.compile(
    r"(?i)(?P<tipo>LEI\s+COMPLEMENTAR|LEI\s+ORDIN[ÁA]RIA|LEI|DECRETO-LEI|"
    r"RESOLU[CÇ][AÃ]O\s+DO\s+TRIBUNAL\s+SUPERIOR\s+ELEITORAL|EMENDA\s+CONSTITUCIONAL|"
    r"CONSTITUI[CÇ][AÃ]O\s+FEDERAL)\s*N[ºo°]\.?\s*:\s*(?P<num>\d+)\s*Ano\s*:\s*(?P<ano>\d{4})"
)
ARTICLE_RE = re.compile(
    r"(?i)\bart\.?\s*\d+[º°]?(?:\s*,\s*(?:§\s*\d+[º°]?|inc\.?\s*[IVXLC]+|[IVXLC]+|al[íi]nea\s*[a-z]))*"
)
ARTICLE_BLOCK_RE = re.compile(r"(?i)\barts?\.?\s*(?P<body>[^;\n()]{1,180})")
ARTICLE_TOKEN_RE = re.compile(r"(?i)\barts?\.?")
ARTICLE_TO_LAW_CUT_RE = re.compile(
    r"(?i)\b(?:da|do|dos|das|de|na|no|nos|nas|pela|pelo|pelas|pelos)\b"
    r"(?=\s+(?:lei\b|lc\b|res\b|resolu[cç][aã]o\b|c[oó]digo\b|constitui[cç][aã]o\b|"
    r"cf/?88\b|ec\b|decreto-lei\b))"
)
ARTICLE_UNIT_RE = re.compile(
    r"(?i)(?:\bart\.?\s*)?"
    r"(?P<num>\d+(?:\.\d+)*(?:[º°])?(?:\s*[-–—]\s*[A-Za-z0-9]+)?)"
    r"(?P<mods>(?:\s*,?\s*§+\s*\d+[º°]?)*(?:\s*,?\s*(?:inc\.?\s*)?[IVXLC]+)?"
    r"(?:\s*,?\s*al[íi]nea\s*[a-z])?)"
)
LC_TEXT_RE = re.compile(
    r"(?ix)\b(?:lei\s+complementar|lc)\s*(?:n(?:[.ºo°]|úmero)?\s*)?"
    r"(?P<num>\d{1,4}(?:\.\d{3})?)(?:\s*(?:/|-)\s*(?P<ano>\d{2,4}))?"
)
LEI_TEXT_RE = re.compile(
    r"(?ix)\blei(?:\s+ordin[áa]ria)?\s*(?:n(?:[.ºo°]|úmero)?\s*)?"
    r"(?P<num>\d{1,5}(?:\.\d{3})?)(?:\s*(?:/|-)\s*(?P<ano>\d{2,4}))?"
)
DECRETO_LEI_TEXT_RE = re.compile(
    r"(?ix)\bdecreto\s*-\s*lei\s*(?:n(?:[.ºo°]|úmero)?\s*)?"
    r"(?P<num>\d{1,5}(?:\.\d{3})?)(?:\s*(?:/|-)\s*(?P<ano>\d{2,4}))?"
)
RES_TSE_TEXT_RE = re.compile(
    r"(?ix)\b(?:"
    r"res(?:olu[cç][aã]o)?\.?\s*[-–—]?\s*tse|"
    r"resolu[cç][aã]o\s+(?:do\s+)?(?:tribunal\s+superior\s+eleitoral|tse)|"
    r"resolu[cç][aã]o(?=[^.;\n]{0,20}/\s*tse)"
    r")\s*(?:n(?:[.ºo°]|úmero)?\s*)?"
    r"(?P<num>\d{1,5}(?:\.\d{3})?)(?:\s*(?:/|-)\s*(?P<ano>\d{2,4}))?(?:\s*/\s*tse)?"
)
EC_TEXT_RE = re.compile(
    r"(?ix)\b(?:emenda\s+constitucional|ec)\s*(?:n(?:[.ºo°]|úmero)?\s*)?"
    r"(?P<num>\d{1,4})(?:\s*(?:/|-)\s*(?P<ano>\d{2,4}))?"
)
CF_ALIAS_RE = re.compile(r"(?i)\b(?:constitui[cç][aã]o federal|constitui[cç][aã]o da rep[úu]blica|cf/?88)\b")
LEI_DAS_ELEICOES_ALIAS_RE = re.compile(r"(?i)\blei\s+das\s+elei[cç][oõ]es\b")
LEI_INELEGIBILIDADES_ALIAS_RE = re.compile(r"(?i)\blei\s+de\s+inelegibilidades\b")
CODIGO_ELEITORAL_ALIAS_RE = re.compile(r"(?i)\bc[oó]digo\s+eleitoral\b")
CPC_ALIAS_RE = re.compile(r"(?i)\b(?:c[oó]digo\s+de\s+processo\s+civil|cpc)\b")
LEI_ELEICOES_LABEL = "Lei n° 9.504/97"
LC_64_LABEL = "LC n° 64/90"
CODIGO_ELEITORAL_LABEL = "Lei n° 4.737/65"
CPC_LABEL = "Lei n° 13.105/15"
UNANIME_RE = re.compile(r"\bpor\s+unanimidade\b|\b[aà]\s+unanimidade\b", re.IGNORECASE)
MAIORIA_RE = re.compile(r"\bpor\s+maioria\b|\bmaioria\s+de\s+votos\b", re.IGNORECASE)
MONOCRATICA_RE = re.compile(r"\bdecis[aã]o\s+monocr[aá]tica\b", re.IGNORECASE)
NAO_CONHECIMENTO_NORM_RE = re.compile(
    r"\b(?:nao\s+conhec(?:imento|id[oa]s?|o|er(?:am|a|ei|emos|ia|iam)?|eu)|"
    r"nao-?conhec(?:imento|id[oa]s?)|nao\s+se\s+conhece|nao\s+foi\s+conhecid[oa]|"
    r"neg(?:o|ou|a(?:r)?|aram)\s+conhecimento)\b",
    re.IGNORECASE,
)
ADV_TITLES_PREFIX_RE = re.compile(r"^\s*(?:dr|dra|doutor|doutora)\.?\s+", re.IGNORECASE)
CNJ_DIGITS_RE = re.compile(r"\D+")
HTML_COMMENT_RE = re.compile(r"(?is)<!--.*?-->")
HTML_TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
HTML_OG_TITLE_RE = re.compile(
    r'(?is)<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']'
)
HTML_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style|noscript|svg|iframe).*?>.*?</\1>")
HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
CNJ_PROCESS_RE = re.compile(r"\b\d{7}-\d{2}\.\d{4}\.6\.\d{2}\.\d{4}\b")
NEWS_NOT_FOUND_RE = re.compile(
    r"(?i)\b(?:pagina nao encontrada|p[aá]gina n[aã]o encontrada|page not found|erro 404|error 404|404 not found|conteudo nao encontrado|conte[uú]do n[aã]o encontrado)\b"
)
NEWS_GENERIC_TITLE_MARKERS = (
    "pauta de julgamentos",
    "confira a pauta",
    "recebe mais de",
    "concurso unificado",
    "diario da justica",
    "diário da justiça",
    "aplicativo pardal",
    "ultimas noticias",
    "solenidade pela posse",
    "presidente do tse apresenta",
    "observatorio de direitos",
    "saiba o que diz a lei",
    "calendario eleitoral",
    "calendário eleitoral",
    "regras da eleicao",
    "regras da eleição",
    "ano judiciario",
    "ano judiciário",
)
NEWS_WEAK_EDITORIAL_URL_MARKERS = (
    "pauta da sessao",
    "pauta de sessao",
    "pauta da sessao plenaria",
    "pauta da sessao de julgamento",
    "pauta de julgamento",
    "pautas de julgamento",
    "confira a pauta",
    "sessao plenaria do dia",
    "sessao de julgamento do dia",
    "calendario eleitoral",
    "diario da justica",
    "ano judiciario",
)
HEADER_END_RE = re.compile(r"(?i)\b(?:decis[aã]o|ac[oó]rd[aã]o|ementa|voto)\b")
HEADER_LABEL_RE = re.compile(
    r"(?i)\b("
    r"recorrentes?|recorrid[oa]s?|agravantes?|agravad[oa]s?|impetrantes?|impetrad[oa]s?|"
    r"requerentes?|requerid[oa]s?|exequentes?|executad[oa]s?|embargantes?|embargad[oa]s?|"
    r"apelantes?|apelad[oa]s?|autores?|r[eé]us?|interessad[oa]s?|representad[oa]s?|"
    r"org[aã]o\s+coator|autoridade\s+coator[ao]|"
    r"advogad(?:o|a|os|as)|"
    r"representantes?\s+do\s*\(a\)\s+[a-zà-öø-ÿ0-9._\-/ ]{1,40}"
    r")\s*:",
    re.IGNORECASE,
)
HEADER_PARTY_ROLE_PREFIXES = (
    "recorr",
    "agrav",
    "impetr",
    "requer",
    "exequ",
    "execut",
    "embarg",
    "apel",
    "autor",
    "réu",
    "reu",
    "interessad",
    "representad",
    "orgao coator",
    "autoridade coator",
)
HEADER_ADV_ROLE_PREFIXES = (
    "advogad",
    "representante do (a)",
    "representantes do (a)",
)
ENTITY_SPLIT_RE = re.compile(
    r"\s*;\s*|\s*,\s*(?=(?:[A-ZÀ-ÖØ-Ý]|Minist[eé]rio|Partido|Coliga[cç][aã]o|"
    r"Federa[cç][aã]o|Uni[aã]o|Ju[ií]zo|Tribunal|Defensoria|Procuradoria))"
)
ENTITY_OUTROS_SUFFIX_RE = re.compile(r"(?is)\s+e\s+outr[oa]s?\b.*$")
ENTITY_OAB_TAIL_RE = re.compile(r"(?is)\s*(?:[-–—]\s*)?OAB\b[^,;]*")
ENTITY_UF_NUM_SUFFIX_RE = re.compile(r"\s*[-–—]\s*[A-Z]{2}\d{3,}(?:-[A-Z])?$")
ENTITY_ID_TAIL_RE = re.compile(r"(?is)\s+id\.?\s*\d+.*$")
ENTITY_CNPJ_TAIL_RE = re.compile(r"\s+\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b.*$")
ENTITY_INLINE_LABEL_CUT_RE = re.compile(
    r"(?is)\b(?:advogad(?:o|a|os|as)|representantes?\s+do\s*\(a\)|"
    r"autoridade\s+coator[ao]|decis[aã]o|ac[oó]rd[aã]o|ementa)\b"
)
ENTITY_DECISION_NOISE_RE = re.compile(
    r"(?i)\b(?:decis[aã]o|ac[oó]rd[aã]o|ementa|tribunal|elei[cç][oõ]es|"
    r"despacho|ante\s+o\s+exposto|publique-se|intimem-se)\b"
)
ENTITY_INSTITUTION_RE = re.compile(
    r"(?i)\b(?:minist[eé]rio|procuradoria|defensoria|advocacia-?geral|ju[ií]zo|tribunal)\b"
)
DEFAULT_METADATA_HEADER_MAX_CHARS = 1200
DEFAULT_ASSUNTOS_MAX_ITEMS = 2
ASSUNTOS_TAXONOMY_CHOICES = ("controlled", "mixed", "free")

ASSUNTOS_CANONICOS = (
    "Abuso de poder",
    "AIJE",
    "Captação de sufrágio",
    "Condições de elegibilidade",
    "Condutas vedadas a agente público",
    "Consulta",
    "Contas de campanha",
    "Contas de exercício financeiro",
    "Cota gênero",
    "Crime eleitoral",
    "Criação de zona eleitoral",
    "Doação",
    "Filiação partidária",
    "Fundo Partidário",
    "Inelegibilidade",
    "Infidelidade partidária",
    "Intempestividade",
    "Lista tríplice",
    "Matéria administrativa",
    "Matéria processual",
    "Pesquisa eleitoral",
    "Propaganda eleitoral antecipada",
    "Propaganda eleitoral irregular",
    "Registro de candidatura",
)

ASSUNTOS_ALIAS_TO_CANONICAL = {
    "aije": "AIJE",
    "acao de investigacao judicial eleitoral": "AIJE",
    "captacao ilicita de sufragio": "Captação de sufrágio",
    "captacao de sufragio": "Captação de sufrágio",
    "condutas vedadas a agente publico": "Condutas vedadas a agente público",
    "cota de genero": "Cota gênero",
    "contas de exercicio financeiro": "Contas de exercício financeiro",
    "filiacao partidaria": "Filiação partidária",
    "fundo partidario": "Fundo Partidário",
    "lista triplice": "Lista tríplice",
    "materia administrativa": "Matéria administrativa",
    "materia processual": "Matéria processual",
    "propaganda eleitoral antecipada": "Propaganda eleitoral antecipada",
    "propaganda eleitoral irregular": "Propaganda eleitoral irregular",
}

ASSUNTOS_RULES_DESCRICAO = (
    ("Lista tríplice", re.compile(r"(?i)\blista\s+tr[ií]plice\b")),
    ("Consulta", re.compile(r"(?i)\bconsulta\b")),
    ("Contas de exercício financeiro", re.compile(r"(?i)\bpresta[cç][aã]o\s+de\s+contas\s+anual\b")),
    ("Matéria administrativa", re.compile(r"(?i)\bpeti[cç][aã]o\s+c[ií]vel\b")),
)

ASSUNTOS_RULES_TEXT = (
    ("AIJE", re.compile(r"(?i)\b(?:aije|a[cç][aã]o\s+de\s+investiga[cç][aã]o\s+judicial\s+eleitoral)\b")),
    ("Abuso de poder", re.compile(r"(?i)\babuso\s+de\s+poder\b")),
    ("Captação de sufrágio", re.compile(r"(?i)\bcapta[cç][aã]o(?:\s+il[ií]cita)?\s+de\s+sufr[aá]gio\b")),
    ("Condições de elegibilidade", re.compile(r"(?i)\bcondi[cç][oõ]es\s+de\s+elegibilidade\b")),
    ("Condutas vedadas a agente público", re.compile(r"(?i)\bcondutas?\s+vedadas?\b")),
    ("Contas de campanha", re.compile(r"(?i)\b(?:presta[cç][aã]o\s+de\s+contas\s+de\s+campanha|contas?\s+de\s+campanha)\b")),
    ("Contas de exercício financeiro", re.compile(r"(?i)\b(?:presta[cç][aã]o\s+de\s+contas\s+anual|exerc[ií]cio\s+financeiro)\b")),
    ("Cota gênero", re.compile(r"(?i)\b(?:cota\s+de\s+g[eê]nero|fraude\s+[aà]\s+cota\s+de\s+g[eê]nero)\b")),
    ("Crime eleitoral", re.compile(r"(?i)\bcrime\s+eleitoral\b")),
    ("Criação de zona eleitoral", re.compile(r"(?i)\bcria[cç][aã]o\s+de\s+zona\s+eleitoral\b")),
    ("Doação", re.compile(r"(?i)\bdoa[cç][aã]o\b")),
    ("Filiação partidária", re.compile(r"(?i)\bfilia[cç][aã]o\s+partid[aá]ria\b")),
    ("Fundo Partidário", re.compile(r"(?i)\bfundo\s+partid[aá]rio\b")),
    ("Inelegibilidade", re.compile(r"(?i)\binelegibil(?:idade|idades)\b")),
    ("Infidelidade partidária", re.compile(r"(?i)\binfidelidade\s+partid[aá]ria\b")),
    ("Intempestividade", re.compile(r"(?i)\bintempestiv(?:idade|o|a|os|as)?\b")),
    ("Lista tríplice", re.compile(r"(?i)\blista\s+tr[ií]plice\b")),
    ("Matéria administrativa", re.compile(r"(?i)\bmat[eé]ria\s+administrativa\b")),
    ("Matéria processual", re.compile(r"(?i)\b(?:mat[eé]ria\s+processual|embargos?\s+de\s+declara[cç][aã]o|agravo\s+interno)\b")),
    ("Pesquisa eleitoral", re.compile(r"(?i)\bpesquisa\s+eleitoral\b")),
    ("Propaganda eleitoral antecipada", re.compile(r"(?i)\bpropaganda\s+eleitoral\s+antecipad[ao]?\b")),
    ("Propaganda eleitoral irregular", re.compile(r"(?i)\b(?:propaganda\s+eleitoral\s+irregular|derram(?:e|amento)\s+de\s+santinh)\b")),
    ("Registro de candidatura", re.compile(r"(?i)\bregistro\s+de\s+candidatur\w*\b")),
)

VEICULOS_GERAIS = [
    "folha de s.paulo",
    "estadao",
    "cnn brasil",
    "g1",
    "o globo",
    "terra",
    "r7",
    "band",
    "correio braziliense",
    "veja",
    "exame",
    "istoe",
    "jovem pan",
    "gazeta do povo",
    "nexo jornal",
    "carta capital",
    "conjur",
    "migalhas",
    "uol",
    "poder360",
    "metropoles",
    "agencia brasil",
    "jota",
    "valor economico",
]

VEICULOS_DOMINIOS = [
    "folha.uol.com.br",
    "estadao.com.br",
    "cnnbrasil.com.br",
    "g1.globo.com",
    "oglobo.globo.com",
    "terra.com.br",
    "r7.com",
    "band.uol.com.br",
    "correiobraziliense.com.br",
    "veja.abril.com.br",
    "exame.com",
    "istoe.com.br",
    "jovempan.com.br",
    "gazetadopovo.com.br",
    "nexojornal.com.br",
    "cartacapital.com.br",
    "conjur.com.br",
    "migalhas.com.br",
    "uol.com.br",
    "noticias.uol.com.br",
    "poder360.com.br",
    "metropoles.com",
    "agenciabrasil.ebc.com.br",
    "jota.info",
    "valor.globo.com",
]

GENERAL_MEDIA_MAINSTREAM_DOMAINS = (
    "g1.globo.com",
    "oglobo.globo.com",
    "agenciabrasil.ebc.com.br",
    "cnnbrasil.com.br",
    "metropoles.com",
    "folha.uol.com.br",
    "uol.com.br",
    "noticias.uol.com.br",
    "estadao.com.br",
    "valor.globo.com",
    "terra.com.br",
    "r7.com",
    "band.uol.com.br",
    "correiobraziliense.com.br",
    "veja.abril.com.br",
    "exame.com",
    "istoe.com.br",
    "jovempan.com.br",
    "gazetadopovo.com.br",
    "nexojornal.com.br",
    "cartacapital.com.br",
)

GENERAL_MEDIA_LEGAL_DOMAINS = (
    "conjur.com.br",
    "migalhas.com.br",
    "jota.info",
    "poder360.com.br",
)

GENERAL_MEDIA_DOMAIN_ORDER = (
    *GENERAL_MEDIA_MAINSTREAM_DOMAINS,
    *GENERAL_MEDIA_LEGAL_DOMAINS,
)

GENERAL_MEDIA_WEAK_TOPIC_MARKERS = (
    "matéria processual",
    "materia processual",
    "matéria administrativa",
    "materia administrativa",
    "agravo em recurso especial eleitoral",
    "recurso especial eleitoral",
    "agravo regimental",
    "embargos de declaração",
    "embargos de declaracao",
    "petição cível",
    "peticao civel",
)

NEWS_CONFIDENCE_LEVELS = ("none", "low", "medium", "high")
NEWS_MATCH_STRONG_FIELDS = {"numero_unico", "numero_processo"}
NEWS_MATCH_HARD_CONTEXT_FIELDS = {
    "data_decisao",
    "tribunal",
    "origem",
    "sigla_uf",
    "nome_municipio",
    "descricao_classe",
    "nome_tipo_processo",
    "relator",
}
NEWS_MATCH_SOFT_CONTEXT_FIELDS = {
    "assuntos",
    "partes",
    "tema",
    "punchline",
}
NEWS_MATCH_CONTEXT_FIELDS = {
    *NEWS_MATCH_HARD_CONTEXT_FIELDS,
    *NEWS_MATCH_SOFT_CONTEXT_FIELDS,
}
GENERAL_NEWS_GEO_ANCHOR_FIELDS = {
    "origem",
    "nome_municipio",
}
NEWS_GENERIC_SEGMENTS_BLOCKLIST = {
    "busca",
    "search",
    "tag",
    "tags",
    "categoria",
    "categorias",
    "arquivo",
    "arquivos",
    "acervo",
    "clipping",
    "blog",
    "blogs",
    "coluna",
    "colunas",
    "opiniao",
    "opinion",
    "editorial",
    "agenda",
    "servicos",
    "servico",
    "institucional",
    "jurisprudencia",
    "pje",
    "consulta-processual",
    "processo",
    "processos",
    "busca-processual",
    "documentos",
    "pdf",
    "fotos",
    "fotogaleria",
    "videos",
    "video",
    "podcasts",
    "podcast",
    "tv",
    "amanhecidas",
    "informativo",
}
NEWS_GENERIC_LAST_SEGMENTS = {
    "noticias",
    "news",
    "politica",
    "brasil",
    "eleicoes",
    "home",
    "index",
    "index.html",
    "ultimas-noticias",
    "ultimas",
    "editoria",
    "editorias",
    "ao-vivo",
    "ao-vivo.html",
}


@dataclass
class ProcessSummary:
    input_path: Path
    output_path: Path
    rows: int
    truncated_cells: int
    encoding: str
    delimiter: str
    fields: list[str]


@dataclass
class WebLookupConfig:
    enabled: bool = False
    provider: str = "perplexity"
    api_key: str = ""
    model: str = "sonar"
    timeout_seconds: int = 15
    max_workers: int = 4
    batch_size: int = 20
    delay_between_batches: float = 0.3
    max_tokens: int = PERPLEXITY_DEFAULT_MAX_TOKENS


@dataclass
class TemaPunchlineConfig:
    enabled: bool = False
    api_key: str = ""
    model: str = "gpt-5.1"
    timeout_seconds: int = OPENAI_DEFAULT_TIMEOUT
    max_workers: int = OPENAI_DEFAULT_MAX_WORKERS
    batch_size: int = OPENAI_DEFAULT_BATCH_SIZE
    delay_between_batches: float = OPENAI_DEFAULT_DELAY
    retries: int = OPENAI_DEFAULT_RETRIES
    target_rpm: int = OPENAI_DEFAULT_TARGET_RPM


@dataclass
class LocalTemaPunchlineConfig:
    enabled: bool = False


@dataclass
class MetadataExtractionConfig:
    include_institutional_entities: bool = True
    header_max_chars: int = DEFAULT_METADATA_HEADER_MAX_CHARS
    max_entity_words: int = 14


@dataclass
class AssuntosEnrichmentConfig:
    enabled: bool = False
    api_key: str = ""
    model: str = "gpt-5.1"
    timeout_seconds: int = OPENAI_DEFAULT_TIMEOUT
    max_workers: int = OPENAI_DEFAULT_MAX_WORKERS
    batch_size: int = OPENAI_DEFAULT_BATCH_SIZE
    delay_between_batches: float = OPENAI_DEFAULT_DELAY
    retries: int = OPENAI_DEFAULT_RETRIES
    target_rpm: int = OPENAI_DEFAULT_TARGET_RPM
    max_items: int = DEFAULT_ASSUNTOS_MAX_ITEMS
    taxonomy_mode: str = "mixed"


def detect_csv_format(path: Path) -> tuple[str, str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS_TO_TRY:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(8192)
            delimiter = ","
            if sample:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ","
            return encoding, delimiter
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"Nao foi possivel detectar encoding de {path}: {last_error}")


def sanitize_header(name: str | None, used: set[str]) -> str:
    text = (name or "").replace("\ufeff", "").strip()
    text = CONTROL_CHAR_RE.sub("", text)
    text = SPACE_RE.sub("_", text)
    text = HEADER_ALLOWED_RE.sub("", text)

    if not text:
        text = "coluna"

    base = text
    counter = 2
    while text in used:
        text = f"{base}_{counter}"
        counter += 1

    used.add(text)
    return text


def remap_output_header(name: str) -> str:
    return OUTPUT_HEADER_REMAP.get(name, name)


def is_excluded_column(name: str) -> bool:
    normalized = re.sub(r"[_\s]+", "", str(name or "")).casefold()
    return normalized in EXCLUDED_COLUMNS_NORMALIZED


def sanitize_cell(value: object, max_chars: int, replace_newlines: bool) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "")
    text = text.replace("¿", "\"")
    text = CONTROL_CHAR_RE.sub(" ", text)

    if replace_newlines:
        text = (
            text.replace("\r\n", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("\t", " ")
        )

    text = SPACE_RE.sub(" ", text).strip()
    truncated = False
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    return text, truncated


def normalize_date_to_mdy(value: object, *, prefer_day_first: bool) -> str:
    raw = SPACE_RE.sub(" ", str(value or "")).strip()
    if not raw:
        return ""
    text = re.sub(r"(?i)^dje\s+de\s+", "", raw).strip()

    match_iso = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", text)
    if match_iso:
        year, month, day = map(int, match_iso.groups())
        try:
            date(year, month, day)
            return f"{month}/{day}/{year}"
        except ValueError:
            return raw

    match = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b", text)
    if not match:
        return raw

    a, b, year = map(int, match.groups())
    mdy_ok = True
    dmy_ok = True
    try:
        date(year, a, b)
    except ValueError:
        mdy_ok = False
    try:
        date(year, b, a)
    except ValueError:
        dmy_ok = False

    if a > 12 and dmy_ok:
        return f"{b}/{a}/{year}"
    if b > 12 and mdy_ok:
        return f"{a}/{b}/{year}"
    if prefer_day_first:
        if dmy_ok:
            return f"{b}/{a}/{year}"
        if mdy_ok:
            return f"{a}/{b}/{year}"
    else:
        if mdy_ok:
            return f"{a}/{b}/{year}"
        if dmy_ok:
            return f"{b}/{a}/{year}"
    return raw


def read_secret_from_file(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        return ""
    raw = ""
    for encoding in (*ENCODINGS_TO_TRY, "utf-16", "utf-16-le", "utf-16-be"):
        try:
            raw = path.read_text(encoding=encoding).strip()
            break
        except UnicodeError:
            continue
        except Exception:
            return ""
    if not raw:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""
    if not raw:
        return ""

    first_line = raw.splitlines()[0].strip()
    if "=" in first_line:
        first_line = first_line.split("=", 1)[1].strip()
    first_line = first_line.lstrip("\ufeff").strip("\"' ")
    return first_line


def _resolve_existing_key_file_path(primary_path: str, fallback_names: Sequence[str]) -> str:
    base_path = Path(primary_path).expanduser() if primary_path else (SCRIPT_DIR / fallback_names[0])
    if not base_path.is_absolute():
        base_path = SCRIPT_DIR / base_path
    base_path = base_path.resolve()
    if base_path.exists() and base_path.is_file():
        return str(base_path)

    search_dir = base_path.parent if base_path.parent.exists() else SCRIPT_DIR
    tried: set[Path] = set()
    for name in fallback_names:
        candidate = (search_dir / name).resolve()
        if candidate in tried:
            continue
        tried.add(candidate)
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    primary_name_norm = normalize_for_match(base_path.name).replace(" ", "")
    if primary_name_norm:
        for candidate in search_dir.glob("*"):
            if not candidate.is_file():
                continue
            candidate_name_norm = normalize_for_match(candidate.name).replace(" ", "")
            if candidate_name_norm == primary_name_norm:
                return str(candidate.resolve())
    return str(base_path)


def resolve_perplexity_api_key(cli_value: str, key_file_path: str) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    env_value = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if env_value:
        return env_value
    resolved_path = _resolve_existing_key_file_path(key_file_path, PERPLEXITY_KEY_FALLBACK_FILES)
    file_value = read_secret_from_file(resolved_path)
    if file_value:
        return file_value
    return ""


def resolve_gemini_api_key(cli_value: str, key_file_path: str) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value
    resolved_path = _resolve_existing_key_file_path(key_file_path, GEMINI_KEY_FALLBACK_FILES)
    file_value = read_secret_from_file(resolved_path)
    if file_value:
        return file_value
    return ""


def resolve_openai_api_key(cli_value: str, key_file_path: str) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    env_value = os.getenv("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value
    resolved_path = _resolve_existing_key_file_path(key_file_path, OPENAI_KEY_FALLBACK_FILES)
    file_value = read_secret_from_file(resolved_path)
    if file_value:
        return file_value
    return ""


class RequestPacer:
    def __init__(self, target_rpm: int) -> None:
        rpm = max(0, int(target_rpm))
        self._min_interval = (60.0 / float(rpm)) if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait_turn(self) -> None:
        if self._min_interval <= 0.0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_at:
                    self._next_at = now + self._min_interval
                    return
                sleep_for = max(0.0, self._next_at - now)
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.2))


class GerenciadorRequisicoes:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_workers: int = 4,
        retries: int = PERPLEXITY_DEFAULT_RETRIES,
        target_rpm: int = PERPLEXITY_DEFAULT_TARGET_RPM,
        provider: str = "perplexity",
    ) -> None:
        self.provider = str(provider or "perplexity").strip().lower()
        self.api_key = api_key
        self.model = model
        self.retries = max(1, int(retries))
        self.pacer = RequestPacer(target_rpm)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.api_session = requests.Session()
        self.api_session.headers.update({"Content-Type": "application/json"})
        if self.provider == "perplexity":
            self.api_session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.page_session = requests.Session()
        self.page_session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/134.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
        self._page_cache: dict[str, Optional[dict[str, object]]] = {}
        self._page_cache_lock = threading.Lock()

    @staticmethod
    def _extract_message_content(result: dict[str, object]) -> str:
        content_obj = ((result.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "")
        if isinstance(content_obj, list):
            chunks: list[str] = []
            for item in content_obj:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks).strip()
        return str(content_obj or "").strip()

    @staticmethod
    def _extract_gemini_text(result: dict[str, object]) -> str:
        chunks: list[str] = []
        for candidate in result.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else []
            for part in parts or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, object]:
        source = str(text or "").strip()
        source = re.sub(r"^```json\s*", "", source, flags=re.IGNORECASE | re.MULTILINE)
        source = re.sub(r"^```\s*", "", source, flags=re.MULTILINE)
        source = re.sub(r"\s*```$", "", source).strip()
        try:
            payload = json.loads(source)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            match = re.search(r"\{.*\}", source, flags=re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                    return payload if isinstance(payload, dict) else {}
                except Exception:
                    return {}
        return {}

    def _post_json_with_retries(
        self,
        *,
        endpoint: str,
        payload: dict[str, object],
        timeout: int,
    ) -> Optional[dict[str, object]]:
        backoff = 1.0
        for attempt in range(self.retries):
            try:
                self.pacer.wait_turn()
                response = self.api_session.post(endpoint, json=payload, timeout=timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(response=response)
                response.raise_for_status()
                result = response.json()
                if isinstance(result, dict):
                    return result
                return None
            except requests.exceptions.Timeout:
                pass
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if status_code and status_code < 500 and status_code != 429:
                    return None
            except Exception:
                pass
            if attempt < self.retries - 1:
                time.sleep(min(backoff, 8.0))
                backoff *= 2.0
        return None

    def _fetch_page_summary_sync(self, url: str, timeout: int) -> Optional[dict[str, object]]:
        with self._page_cache_lock:
            if url in self._page_cache:
                return self._page_cache[url]
        try:
            response = self.page_session.get(url, timeout=timeout, allow_redirects=True)
            content_type = str(response.headers.get("Content-Type", "") or "").lower()
            raw_html = response.text if "html" in content_type or not content_type else ""
            title = _extract_html_title(raw_html)
            text = _extract_html_text(raw_html, max_chars=PERPLEXITY_PAGE_TEXT_MAX_CHARS)
            summary: Optional[dict[str, object]] = {
                "url": _limpar_url_bruta(str(response.url or "")) or url,
                "status_code": int(response.status_code),
                "title": title,
                "text": text,
                "content_type": content_type,
            }
        except requests.exceptions.Timeout:
            summary = None
        except Exception:
            summary = None
        with self._page_cache_lock:
            self._page_cache[url] = summary
        return summary

    async def call_perplexity(self, prompt: str, timeout: int = 15) -> Optional[dict[str, object]]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
        }
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: self._post_json_with_retries(
                endpoint="https://api.perplexity.ai/chat/completions",
                payload=payload,
                timeout=timeout,
            ),
        )
        if not isinstance(result, dict):
            return None
        content = self._extract_message_content(result)
        citations = result.get("citations") or []
        if not isinstance(citations, list):
            citations = []
        return {"content": content, "citations": citations}

    async def search_news_results(
        self,
        *,
        query: str,
        timeout: int,
        max_results: int = PERPLEXITY_SEARCH_MAX_RESULTS,
        max_tokens_per_page: int = PERPLEXITY_SEARCH_MAX_TOKENS_PER_PAGE,
        search_domain_order: Optional[Sequence[str]] = None,
    ) -> Optional[list[dict[str, str]]]:
        payload = {
            "query": query,
            "max_results": max(1, int(max_results)),
            "max_tokens_per_page": max(64, int(max_tokens_per_page)),
        }
        ordered_domains = [
            str(domain or "").strip()
            for domain in (search_domain_order or ())
            if str(domain or "").strip()
        ]
        if ordered_domains:
            payload["search_domain_order"] = ordered_domains
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: self._post_json_with_retries(
                endpoint="https://api.perplexity.ai/search",
                payload=payload,
                timeout=timeout,
            ),
        )
        if not isinstance(result, dict):
            return None
        return _normalize_search_results(result.get("results", []))

    async def fetch_page_summary(self, url: str, timeout: int = PERPLEXITY_PAGE_FETCH_TIMEOUT) -> Optional[dict[str, object]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self._fetch_page_summary_sync(url, timeout),
        )

    async def call_news_lookup(
        self,
        *,
        prompt: str,
        timeout: int,
        max_tokens: int,
        max_general_urls: int,
    ) -> Optional[dict[str, object]]:
        if self.provider == "gemini":
            thinking_config = (
                {"thinkingLevel": "minimal"}
                if str(self.model or "").startswith("gemini-3")
                else {"thinkingBudget": 0}
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": max(128, int(max_tokens)),
                    "responseMimeType": "application/json",
                    "thinkingConfig": thinking_config,
                },
                "tools": [{"googleSearch": {}}],
            }
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self._post_json_with_retries(
                    endpoint=f"{GEMINI_REST_BASE_URL}/models/{self.model}:generateContent?key={self.api_key}",
                    payload=payload,
                    timeout=timeout,
                ),
            )
            if not isinstance(result, dict):
                return None
            usage = result.get("usageMetadata")
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
                output_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
                self.news_tokens_prompt_total = getattr(self, "news_tokens_prompt_total", 0) + prompt_tokens
                self.news_tokens_output_total = getattr(self, "news_tokens_output_total", 0) + output_tokens
                LOGGER.debug(
                    "[Gemini noticias] tokens: prompt=%d saida=%d | acumulado prompt=%d saida=%d",
                    prompt_tokens,
                    output_tokens,
                    self.news_tokens_prompt_total,
                    self.news_tokens_output_total,
                )
            parsed = self._extract_json_object(self._extract_gemini_text(result))
            if not parsed:
                return None
            tse = parsed.get("tse", {})
            tre = parsed.get("tre", {})
            raw_gerais = parsed.get("gerais", [])
            gerais: list[dict[str, object]] = []
            if isinstance(raw_gerais, list):
                for item in raw_gerais[: max(1, int(max_general_urls))]:
                    if isinstance(item, dict):
                        gerais.append(dict(item))
            return {"tse": tse, "tre": tre, "gerais": gerais, "citations": []}

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Voce faz pesquisa juridico-jornalistica eleitoral. "
                        "Pesquise primeiro com o numero do processo exato entre aspas e com os filtros "
                        "de dominio sugeridos pelo usuario. "
                        "Ignore resultados genericos sobre regras eleitorais, calendario, resolucoes, "
                        "pautas ou paginas institucionais sem aderencia ao caso. "
                        "Responda exclusivamente em JSON valido, sem markdown, "
                        "sem comentarios e sem texto adicional."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "search_mode": "web",
            "max_tokens": max(1, int(max_tokens)),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sjur_news_lookup",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "tse": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "url": {"type": "string"},
                                    "confidence": {"type": "string", "enum": list(NEWS_CONFIDENCE_LEVELS)},
                                    "matched_fields": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "maxItems": 8,
                                    },
                                },
                                "required": ["url", "confidence", "matched_fields"],
                            },
                            "tre": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "url": {"type": "string"},
                                    "confidence": {"type": "string", "enum": list(NEWS_CONFIDENCE_LEVELS)},
                                    "matched_fields": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "maxItems": 8,
                                    },
                                },
                                "required": ["url", "confidence", "matched_fields"],
                            },
                            "gerais": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "url": {"type": "string"},
                                        "confidence": {"type": "string", "enum": list(NEWS_CONFIDENCE_LEVELS)},
                                        "matched_fields": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "maxItems": 8,
                                        },
                                    },
                                    "required": ["url", "confidence", "matched_fields"],
                                },
                                "maxItems": max(1, int(max_general_urls)),
                            },
                        },
                        "required": ["tse", "tre", "gerais"],
                    },
                },
            },
        }
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: self._post_json_with_retries(
                endpoint="https://api.perplexity.ai/chat/completions",
                payload=payload,
                timeout=timeout,
            ),
        )
        if not isinstance(result, dict):
            return None
        content = self._extract_message_content(result)
        citations = result.get("citations") or []
        if not isinstance(citations, list):
            citations = []
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        tse = parsed.get("tse", {})
        tre = parsed.get("tre", {})
        raw_gerais = parsed.get("gerais", [])
        gerais: list[dict[str, object]] = []
        if isinstance(raw_gerais, list):
            for item in raw_gerais:
                if isinstance(item, dict):
                    gerais.append(dict(item))
        return {
            "tse": tse,
            "tre": tre,
            "gerais": gerais,
            "citations": citations,
        }

    def close(self) -> None:
        self.api_session.close()
        self.page_session.close()
        self.executor.shutdown(wait=True)


class GerenciadorOpenAI:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_workers: int = OPENAI_DEFAULT_MAX_WORKERS,
        retries: int = OPENAI_DEFAULT_RETRIES,
        target_rpm: int = OPENAI_DEFAULT_TARGET_RPM,
    ) -> None:
        self.model = model
        self.retries = max(1, int(retries))
        self.pacer = RequestPacer(target_rpm)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.sessao = requests.Session()
        self.sessao.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def _extract_message_content(result: dict[str, object]) -> str:
        content_obj = ((result.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "")
        if isinstance(content_obj, list):
            chunks: list[str] = []
            for item in content_obj:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks).strip()
        return str(content_obj or "").strip()

    async def _call_json_schema(
        self,
        *,
        prompt: str,
        timeout: int,
        schema_name: str,
        schema: dict[str, object],
    ) -> Optional[dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Voce e um assistente juridico especializado em jurisprudencia eleitoral. "
                        "Responda exclusivamente em JSON valido, sem markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        loop = asyncio.get_event_loop()
        for attempt in range(1, self.retries + 1):
            try:
                self.pacer.wait_turn()
                response = await loop.run_in_executor(
                    self.executor,
                    lambda: self.sessao.post(
                        "https://api.openai.com/v1/chat/completions",
                        json=payload,
                        timeout=timeout,
                    ),
                )
                response.raise_for_status()
                result = response.json()
                raw_content = self._extract_message_content(result if isinstance(result, dict) else {})
                if not raw_content:
                    if attempt < self.retries:
                        await asyncio.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                        continue
                    return None
                parsed = json.loads(raw_content)
                if not isinstance(parsed, dict):
                    if attempt < self.retries:
                        await asyncio.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                        continue
                    return None
                return parsed
            except requests.exceptions.Timeout:
                if attempt < self.retries:
                    await asyncio.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                    continue
                return None
            except requests.exceptions.HTTPError as exc:
                status = int(exc.response.status_code) if exc.response is not None else 0
                retryable = status in (429, 500, 502, 503, 504)
                if retryable and attempt < self.retries:
                    await asyncio.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                    continue
                return None
            except Exception:
                if attempt < self.retries:
                    await asyncio.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                    continue
                return None

    async def call_tema_punchline(self, prompt: str, timeout: int = OPENAI_DEFAULT_TIMEOUT) -> Optional[dict[str, str]]:
        parsed = await self._call_json_schema(
            prompt=prompt,
            timeout=timeout,
            schema_name="tema_punchline",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tema": {"type": "string"},
                    "punchline": {"type": "string"},
                },
                "required": ["tema", "punchline"],
            },
        )
        if not isinstance(parsed, dict):
            return None
        tema = SPACE_RE.sub(" ", str(parsed.get("tema", "") or "")).strip()
        punchline = SPACE_RE.sub(" ", str(parsed.get("punchline", "") or "")).strip()
        return {"tema": tema, "punchline": punchline}

    async def call_assuntos(
        self,
        prompt: str,
        *,
        max_items: int,
        timeout: int = OPENAI_DEFAULT_TIMEOUT,
    ) -> Optional[dict[str, list[str]]]:
        max_items_safe = max(1, int(max_items))
        parsed = await self._call_json_schema(
            prompt=prompt,
            timeout=timeout,
            schema_name="assuntos_enrichment",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "assuntos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": max_items_safe,
                    }
                },
                "required": ["assuntos"],
            },
        )
        if not isinstance(parsed, dict):
            return None
        raw_items = parsed.get("assuntos")
        if not isinstance(raw_items, list):
            return None
        assuntos = [SPACE_RE.sub(" ", str(item or "")).strip() for item in raw_items]
        return {"assuntos": assuntos}

    def close(self) -> None:
        self.sessao.close()
        self.executor.shutdown(wait=True)


def _limpar_url_bruta(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip().strip(").,;:!?]}>\"'")
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return None


def limpar_url(texto: str) -> Optional[str]:
    if not texto:
        return None
    match = re.search(r"https?://\S+", texto)
    if not match:
        return None
    return _limpar_url_bruta(match.group(0))


def extrair_urls_multiplas(texto: str) -> Optional[list[str]]:
    if not texto:
        return None
    encontrados = re.findall(r"https?://\S+", texto)
    if not encontrados:
        return None
    urls: list[str] = []
    vistos: set[str] = set()
    for url in encontrados:
        limpa = _limpar_url_bruta(url)
        if limpa and limpa not in vistos:
            vistos.add(limpa)
            urls.append(limpa)
    return urls if urls else None


def extrair_urls_de_citations(citations: object) -> list[str]:
    if not isinstance(citations, list):
        return []
    urls: list[str] = []
    vistos: set[str] = set()
    for item in citations:
        if not isinstance(item, str):
            continue
        limpa = _limpar_url_bruta(item)
        if limpa and limpa not in vistos:
            vistos.add(limpa)
            urls.append(limpa)
    return urls


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _parsed_url(url: str):
    try:
        return urlparse(url)
    except Exception:
        return None


def _path_segments(url: str) -> list[str]:
    parsed = _parsed_url(url)
    if parsed is None:
        return []
    return [segment for segment in (parsed.path or "").lower().split("/") if segment]


def _query_looks_like_listing(url: str) -> bool:
    parsed = _parsed_url(url)
    if parsed is None:
        return True
    query = (parsed.query or "").lower()
    if not query:
        return False
    return any(
        marker in query
        for marker in ("q=", "query=", "search=", "tag=", "tags=", "categoria=", "page=")
    )


def _path_has_sequence(segments: Sequence[str], expected: Sequence[str]) -> bool:
    expected_list = [item for item in expected if item]
    if not expected_list or len(segments) < len(expected_list):
        return False
    target = list(expected_list)
    for idx in range(0, len(segments) - len(target) + 1):
        if list(segments[idx : idx + len(target)]) == target:
            return True
    return False


def _last_path_segment(segments: Sequence[str]) -> str:
    return segments[-1] if segments else ""


def _clean_slug(segment: str) -> str:
    return re.sub(r"\.(?:html?|ghtml|shtml)$", "", str(segment or "").strip().lower())


def _news_url_marker_text(url: str) -> str:
    segments = [_clean_slug(segment).replace("-", " ").replace("_", " ") for segment in _path_segments(url)]
    return normalize_for_match(" ".join(segments))


def is_editorially_weak_news_url(url: str) -> bool:
    marker_text = _news_url_marker_text(url)
    if not marker_text:
        return True
    if any(marker in marker_text for marker in NEWS_WEAK_EDITORIAL_URL_MARKERS):
        return True
    return False


def _looks_like_article_terminal(segment: str) -> bool:
    slug = _clean_slug(segment)
    if not slug or slug in NEWS_GENERIC_LAST_SEGMENTS:
        return False
    if len(slug) < 8:
        return False
    if "/" in slug:
        return False
    if "." in slug and not segment.lower().endswith((".html", ".htm", ".ghtml", ".shtml")):
        return False
    if slug.isdigit():
        return False
    return ("-" in slug) or any(ch.isdigit() for ch in slug) or len(slug) >= 16


def _is_probably_article_url(
    url: str,
    *,
    min_segments: int,
    required_sequence: Optional[Sequence[str]] = None,
) -> bool:
    if is_editorially_weak_news_url(url):
        return False
    segments = _path_segments(url)
    if len(segments) < min_segments:
        return False
    if required_sequence is not None and not _path_has_sequence(segments, required_sequence):
        return False
    if _query_looks_like_listing(url):
        return False
    if any(segment in NEWS_GENERIC_SEGMENTS_BLOCKLIST for segment in segments):
        return False
    return _looks_like_article_terminal(_last_path_segment(segments))


def _normalize_match_field_name(value: object) -> str:
    normalized = normalize_for_match(str(value or ""))
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized


def _extract_news_candidate(raw_value: object) -> tuple[Optional[str], str, set[str]]:
    if isinstance(raw_value, dict):
        url = _limpar_url_bruta(str(raw_value.get("url", "") or ""))
        confidence = _normalize_lookup_text(raw_value.get("confidence", "")).lower()
        if confidence not in NEWS_CONFIDENCE_LEVELS:
            confidence = "none"
        matched_fields_raw = raw_value.get("matched_fields", [])
        matched_fields: set[str] = set()
        if isinstance(matched_fields_raw, list):
            for item in matched_fields_raw:
                field_name = _normalize_match_field_name(item)
                if field_name:
                    matched_fields.add(field_name)
        return url, confidence, matched_fields
    if isinstance(raw_value, str):
        return _limpar_url_bruta(raw_value), "none", set()
    return None, "none", set()


def _news_candidate_passes_gate(
    confidence: str,
    matched_fields: set[str],
    *,
    official_source: bool,
) -> bool:
    if confidence not in {"high", "medium"}:
        return False
    if matched_fields & NEWS_MATCH_STRONG_FIELDS:
        return True
    hard_matches = matched_fields & NEWS_MATCH_HARD_CONTEXT_FIELDS
    soft_matches = matched_fields & NEWS_MATCH_SOFT_CONTEXT_FIELDS
    context_matches = matched_fields & NEWS_MATCH_CONTEXT_FIELDS
    if official_source:
        if confidence == "high":
            return bool(hard_matches) or len(context_matches) >= 2
        return (bool(hard_matches) and len(context_matches) >= 2) or len(context_matches) >= 3
    if confidence == "high":
        return (bool(hard_matches) and len(context_matches) >= 2) or len(context_matches) >= 3
    return bool(hard_matches) and bool(soft_matches) and len(context_matches) >= 2


def _is_tse_url(url: str) -> bool:
    return _host(url).endswith("tse.jus.br")


def _is_tse_news_url(url: str) -> bool:
    host = _host(url)
    if not host.endswith("tse.jus.br"):
        return False
    return _is_probably_article_url(
        url,
        min_segments=5,
        required_sequence=("comunicacao", "noticias"),
    )


def _is_tre_url(url: str) -> bool:
    host = _host(url)
    return ".jus.br" in host and "tre-" in host


def _is_tre_news_url(url: str) -> bool:
    if not _is_tre_url(url):
        return False
    segments = _path_segments(url)
    if _path_has_sequence(segments, ("comunicacao", "noticias")):
        return _is_probably_article_url(
            url,
            min_segments=5,
            required_sequence=("comunicacao", "noticias"),
        )
    if _path_has_sequence(segments, ("noticias",)):
        return _is_probably_article_url(
            url,
            min_segments=3,
            required_sequence=("noticias",),
        )
    return False


def _normalize_tse_news_url(value: object) -> Optional[str]:
    url = _limpar_url_bruta(str(value or ""))
    if not url or not _is_tse_news_url(url):
        return None
    return url


def _normalize_tre_news_url(value: object) -> Optional[str]:
    url = _limpar_url_bruta(str(value or ""))
    if not url or not _is_tre_news_url(url):
        return None
    return url


OFFICIAL_NEWS_COLUMN_NORMALIZERS = {
    "noticia_TSE": _normalize_tse_news_url,
    "noticia_TRE": _normalize_tre_news_url,
}


def sanitize_official_news_columns_in_row(row: dict[str, str]) -> int:
    cleared = 0
    for column, normalizer in OFFICIAL_NEWS_COLUMN_NORMALIZERS.items():
        raw_value = str(row.get(column, "") or "")
        normalized = normalizer(raw_value) or ""
        if raw_value != normalized:
            row[column] = normalized
            if raw_value and not normalized:
                cleared += 1
    return cleared


def sanitize_official_news_columns(rows: Iterable[dict[str, str]]) -> int:
    cleared_total = 0
    for row in rows:
        cleared_total += sanitize_official_news_columns_in_row(row)
    return cleared_total


def _is_general_media_url(url: str) -> bool:
    host = _host(url)
    if not any(host == domain or host.endswith("." + domain) for domain in VEICULOS_DOMINIOS):
        return False
    return _is_probably_article_url(url, min_segments=2)


def _pick_first(urls: Iterable[str], predicate: Callable[[str], bool]) -> Optional[str]:
    for url in urls:
        if predicate(url):
            return url
    return None


def build_lookup_payload(row: dict[str, str]) -> dict[str, str]:
    tribunal = " ".join(
        part for part in (row.get("siglaTribunalJE", ""), row.get("origemDecisao", "")) if part
    ).strip()
    sigla_uf = row.get("siglaUF") or ""
    nome_municipio = row.get("nomeMunicipio") or ""
    origem = " ".join(part for part in (sigla_uf, nome_municipio) if part).strip()
    return {
        "numero_unico": row.get("numeroUnico") or row.get("numeroProcesso") or "",
        "data_decisao": row.get("dataDecisao") or "",
        "assuntos": row.get("assuntos") or "",
        "partes": row.get("partes") or "",
        "relator": row.get("relator") or row.get("relatores") or "",
        "texto_decisao": row.get("textoDecisao") or "",
        "texto_ementa": row.get("textoEmenta") or "",
        "descricao_classe": row.get("descricaoClasse") or "",
        "nome_tipo_processo": row.get("nomeTipoProcesso") or "",
        "sigla_uf": sigla_uf,
        "nome_municipio": nome_municipio,
        "tribunal": tribunal,
        "origem": origem,
        "tema": row.get("tema") or "",
        "punchline": row.get("punchline") or "",
    }


def resolve_artifacts_dir(*parts: str) -> Path:
    path = Path(ARTIFACTS_ROOT).expanduser().resolve()
    for part in parts:
        path /= part
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_checkpoint_artifact_path(output_path: Path) -> Path:
    return (resolve_artifacts_dir("checkpoints") / f".{output_path.stem}.openai.checkpoint.json").resolve()


def resolve_report_artifact_path(output_path: Path) -> Path:
    return (resolve_artifacts_dir("reports") / f".{output_path.stem}.openai.report.json").resolve()


def resolve_backup_artifacts_dir() -> Path:
    return resolve_artifacts_dir("backups")


def resolve_intermediate_csv_dir() -> Path:
    return resolve_artifacts_dir("intermediarios", "csv")


def resolve_web_lookup_cache_path(base_dir: Optional[Path] = None) -> Path:
    _ = base_dir
    return (resolve_artifacts_dir("cache") / PERPLEXITY_CACHE_FILENAME).resolve()


def _normalize_lookup_text(value: object) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def _truncate_lookup_text(value: object, max_chars: int) -> str:
    text = _normalize_lookup_text(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def _quote_search_term(value: object, max_chars: int = 120) -> str:
    text = _truncate_lookup_text(value, max_chars)
    text = text.replace('"', " ").strip()
    return f'"{text}"' if text else ""


def _derive_tre_domain_hint(page_data: dict[str, object]) -> str:
    sigla_uf = _normalize_lookup_text(page_data.get("sigla_uf", "")).upper()
    if not sigla_uf:
        tribunal = _normalize_lookup_text(page_data.get("tribunal", "")).upper()
        match = re.search(r"\bTRE[-\s/]?([A-Z]{2})\b", tribunal)
        if match:
            sigla_uf = match.group(1)
    if re.fullmatch(r"[A-Z]{2}", sigla_uf):
        return f"tre-{sigla_uf.lower()}.jus.br"
    return "tre-xx.jus.br"


def _join_query_terms(*values: object) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        quoted = _quote_search_term(value)
        if not quoted:
            continue
        normalized = quoted.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(quoted)
    return " ".join(terms)


def build_news_query_hints(page_data: dict[str, object]) -> dict[str, str]:
    numero_unico = _normalize_lookup_text(page_data.get("numero_unico", ""))
    tema = _truncate_lookup_text(page_data.get("tema", ""), 120)
    assuntos = _truncate_lookup_text(page_data.get("assuntos", ""), 80)
    relator = _truncate_lookup_text(page_data.get("relator", ""), 80)
    nome_municipio = _truncate_lookup_text(page_data.get("nome_municipio", ""), 60)
    sigla_uf = _normalize_lookup_text(page_data.get("sigla_uf", "")).upper()
    descricao_classe = _truncate_lookup_text(page_data.get("descricao_classe", ""), 60)
    partes = _truncate_lookup_text(page_data.get("partes", ""), 80)
    tre_domain = _derive_tre_domain_hint(page_data)
    return {
        "tre_domain": tre_domain,
        "tse_exact": "site:tse.jus.br/comunicacao/noticias "
        + _join_query_terms(numero_unico, tema),
        "tre_exact": f"site:{tre_domain}/comunicacao/noticias "
        + _join_query_terms(numero_unico, tema, nome_municipio, sigla_uf),
        "general_exact": _join_query_terms(numero_unico, tema, nome_municipio, sigla_uf),
        "tse_context": "site:tse.jus.br/comunicacao/noticias "
        + _join_query_terms(tema, descricao_classe, nome_municipio, sigla_uf, relator),
        "tre_context": f"site:{tre_domain} "
        + _join_query_terms(tema, descricao_classe, nome_municipio, sigla_uf, relator),
        "general_context": _join_query_terms(*_build_general_media_focus_terms(page_data, max_topics=2, max_parties=2)),
    }


def _extract_html_title(raw_html: str) -> str:
    if not raw_html:
        return ""
    match = HTML_TITLE_RE.search(raw_html) or HTML_OG_TITLE_RE.search(raw_html)
    if match is None:
        return ""
    title = html.unescape(str(match.group(1) or ""))
    title = CONTROL_CHAR_RE.sub(" ", title)
    return SPACE_RE.sub(" ", title).strip()


def _extract_html_text(raw_html: str, *, max_chars: int) -> str:
    if not raw_html:
        return ""
    cleaned = HTML_COMMENT_RE.sub(" ", raw_html)
    cleaned = HTML_SCRIPT_STYLE_RE.sub(" ", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = CONTROL_CHAR_RE.sub(" ", cleaned)
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip()
    return cleaned


def _normalize_search_results(raw_results: object) -> list[dict[str, str]]:
    if not isinstance(raw_results, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = _limpar_url_bruta(str(item.get("url", "") or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        normalized.append(
            {
                "url": url,
                "title": _normalize_lookup_text(item.get("title", "")),
                "snippet": _normalize_lookup_text(item.get("snippet", "")),
                "date": _normalize_lookup_text(item.get("date", "") or item.get("last_updated", "")),
            }
        )
    return normalized


def _extract_news_parties(page_data: dict[str, object], *, max_items: int = 4) -> list[str]:
    parties = split_multiselect_values(str(page_data.get("partes", "") or ""))
    prioritized: list[str] = []
    fallback: list[str] = []
    seen: set[str] = set()
    ignored_markers = (
        "ministerio publico",
        "uniao federal",
        "procuradoria",
        "advocacia geral",
    )
    institutional_markers = (
        "partido",
        "coligacao",
        "federação",
        "federacao",
        "frente",
        "união brasil",
        "uniao brasil",
    )
    for party in parties:
        normalized = normalize_for_match(party)
        if not normalized or any(marker in normalized for marker in ignored_markers):
            continue
        key = normalized.strip()
        if key in seen:
            continue
        seen.add(key)
        target = fallback if any(marker in normalized for marker in institutional_markers) else prioritized
        target.append(party)
    ordered = prioritized + fallback
    return ordered[: max(1, int(max_items))]


def _extract_general_media_topics(page_data: dict[str, object], *, max_items: int = 2) -> list[str]:
    candidates: list[str] = []
    for field_name, max_chars in (("tema", 100), ("punchline", 100)):
        value = _truncate_lookup_text(page_data.get(field_name, ""), max_chars)
        if value:
            candidates.append(value)
    candidates.extend(
        _truncate_lookup_text(item, 80)
        for item in split_multiselect_values(str(page_data.get("assuntos", "") or ""))
    )
    for field_name in ("nome_tipo_processo", "descricao_classe"):
        value = _truncate_lookup_text(page_data.get(field_name, ""), 80)
        if value:
            candidates.append(value)

    selected: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_for_match(candidate)
        if (
            not normalized
            or normalized in seen
            or any(marker in normalized for marker in GENERAL_MEDIA_WEAK_TOPIC_MARKERS)
        ):
            continue
        seen.add(normalized)
        selected.append(candidate)
        if len(selected) >= max(1, int(max_items)):
            break
    return selected


def _build_general_media_focus_terms(
    page_data: dict[str, object],
    *,
    max_topics: int = 2,
    max_parties: int = 1,
) -> list[str]:
    terms: list[str] = []
    terms.extend(_extract_general_media_topics(page_data, max_items=max_topics))
    nome_municipio = _truncate_lookup_text(page_data.get("nome_municipio", ""), 60)
    sigla_uf = _normalize_lookup_text(page_data.get("sigla_uf", "")).upper()
    if nome_municipio:
        terms.append(nome_municipio)
    if sigla_uf:
        terms.append(sigla_uf)
    parties = _extract_news_parties(page_data, max_items=max_parties)
    if parties:
        terms.extend(parties)
    else:
        relator = _truncate_lookup_text(page_data.get("relator", ""), 60)
        if relator:
            terms.append(relator)
    return terms


def _build_site_or_filter(domains: Sequence[str]) -> str:
    clean_domains = [str(domain or "").strip() for domain in domains if str(domain or "").strip()]
    if not clean_domains:
        return ""
    return "(" + " OR ".join(f"site:{domain}" for domain in clean_domains) + ")"


def _target_general_lookup_count(
    *,
    precisa_tse: bool,
    precisa_tre: bool,
    missing_general_count: int,
) -> int:
    count = max(0, int(missing_general_count))
    if count <= 0:
        return 0
    if not precisa_tse and not precisa_tre:
        return 1
    return count


def _search_domain_order_for_query_label(query_label: str) -> tuple[str, ...]:
    label = str(query_label or "")
    if label == "general_editorial_mainstream":
        return GENERAL_MEDIA_MAINSTREAM_DOMAINS
    if label == "general_editorial_legal":
        return GENERAL_MEDIA_LEGAL_DOMAINS
    if label.startswith("general_"):
        return GENERAL_MEDIA_DOMAIN_ORDER
    return ()


def _search_max_tokens_per_page_for_query_label(query_label: str) -> int:
    if str(query_label or "").startswith("general_"):
        return 128
    return PERPLEXITY_SEARCH_MAX_TOKENS_PER_PAGE


def _entity_match_variants(value: str) -> list[str]:
    text = SPACE_RE.sub(" ", str(value or "")).strip()
    if not text:
        return []
    variants: list[str] = []
    full = normalize_for_match(re.sub(r"\([^)]*\)", " ", text))
    if full:
        variants.append(full)
    for acronym in re.findall(r"\(([A-Z]{2,})\)", text):
        norm = normalize_for_match(acronym)
        if norm:
            variants.append(norm)
    words = [word for word in re.split(r"[^a-z0-9]+", full) if word]
    if len(words) >= 2:
        variants.append(f"{words[0]} {words[-1]}")
    if len(words) >= 3:
        variants.append(" ".join(words[-2:]))
    return dedupe_preserve(
        [variant.strip() for variant in variants if len(variant.strip()) >= 4],
        key_func=lambda item: item,
    )


def _relator_match_variants(value: str) -> list[str]:
    text = RELATOR_PREFIX_RE.sub("", str(value or "")).strip()
    text = re.sub(r"(?i)\bmin(?:istro|istra)?\.?\b", " ", text)
    normalized = normalize_for_match(text)
    if not normalized:
        return []
    words = [word for word in re.split(r"[^a-z0-9]+", normalized) if word]
    variants = [normalized]
    if len(words) >= 2:
        variants.append(" ".join(words[-2:]))
        variants.append(f"{words[0]} {words[-1]}")
    return dedupe_preserve(
        [variant.strip() for variant in variants if len(variant.strip()) >= 6],
        key_func=lambda item: item,
    )


def _free_text_match_variants(value: object, *, min_chars: int = 6) -> list[str]:
    normalized = normalize_for_match(str(value or ""))
    if not normalized or len(normalized) < min_chars:
        return []
    variants = [normalized]
    shortened = re.sub(r"\beleitoral\b", " ", normalized)
    shortened = SPACE_RE.sub(" ", shortened).strip()
    if shortened and shortened != normalized and len(shortened) >= min_chars:
        variants.append(shortened)
    return dedupe_preserve(variants, key_func=lambda item: item)


def _match_sigla_uf_in_text(raw_text: str, sigla_uf: str, url: str) -> bool:
    normalized_uf = _normalize_lookup_text(sigla_uf).upper()
    if not normalized_uf or not re.fullmatch(r"[A-Z]{2}", normalized_uf):
        return False
    host = _host(url)
    if host.endswith(f"tre-{normalized_uf.lower()}.jus.br"):
        return True
    raw_lower = str(raw_text or "").lower()
    return re.search(rf"(?<![a-z0-9])(?:{normalized_uf.lower()}|/{normalized_uf.lower()}|\({normalized_uf.lower()}\))(?![a-z0-9])", raw_lower) is not None


def _match_news_candidate_fields(
    page_data: dict[str, object],
    raw_text: str,
    *,
    url: str,
    category: str,
) -> set[str]:
    matched: set[str] = set()
    normalized_text = normalize_for_match(raw_text)
    digits_text = re.sub(r"\D+", "", str(raw_text or ""))
    numero_unico = clean_numero_processo(str(page_data.get("numero_unico", "") or ""))
    if numero_unico and numero_unico in digits_text:
        matched.add("numero_unico")

    tribunal = normalize_for_match(str(page_data.get("tribunal", "") or ""))
    if category == "tse" and "tse" in tribunal:
        matched.add("tribunal")

    origem_variants = _free_text_match_variants(page_data.get("origem", ""), min_chars=5)
    if origem_variants and any(variant in normalized_text for variant in origem_variants):
        matched.add("origem")

    if _match_sigla_uf_in_text(raw_text, str(page_data.get("sigla_uf", "") or ""), url):
        matched.add("sigla_uf")

    municipio_variants = _free_text_match_variants(page_data.get("nome_municipio", ""), min_chars=4)
    if municipio_variants and any(variant in normalized_text for variant in municipio_variants):
        matched.add("nome_municipio")

    classe_variants = _free_text_match_variants(page_data.get("descricao_classe", ""), min_chars=6)
    if classe_variants and any(variant in normalized_text for variant in classe_variants):
        matched.add("descricao_classe")

    tipo_variants = _free_text_match_variants(page_data.get("nome_tipo_processo", ""), min_chars=8)
    if tipo_variants and any(variant in normalized_text for variant in tipo_variants):
        matched.add("nome_tipo_processo")

    relator_variants = _relator_match_variants(str(page_data.get("relator", "") or ""))
    if relator_variants and any(variant in normalized_text for variant in relator_variants):
        matched.add("relator")

    for party in _extract_news_parties(page_data):
        variants = _entity_match_variants(party)
        if variants and any(variant in normalized_text for variant in variants):
            matched.add("partes")
            break

    for field_name in ("assuntos", "tema", "punchline"):
        variants = _free_text_match_variants(page_data.get(field_name, ""))
        if variants and any(variant in normalized_text for variant in variants):
            matched.add(field_name)

    return matched


def _score_news_candidate_match(matched_fields: set[str], *, rank: int, query_label: str) -> int:
    score = 0
    if matched_fields & NEWS_MATCH_STRONG_FIELDS:
        score += 100
    score += 14 * len(matched_fields & NEWS_MATCH_HARD_CONTEXT_FIELDS)
    score += 9 * len(matched_fields & NEWS_MATCH_SOFT_CONTEXT_FIELDS)
    score += max(0, 6 - max(0, int(rank)))
    if query_label.startswith(("tse_", "tre_")):
        score += 3
    return score


def _derive_news_candidate_confidence(matched_fields: set[str], *, official_source: bool) -> str:
    if matched_fields & NEWS_MATCH_STRONG_FIELDS:
        return "high"
    hard_matches = matched_fields & NEWS_MATCH_HARD_CONTEXT_FIELDS
    soft_matches = matched_fields & NEWS_MATCH_SOFT_CONTEXT_FIELDS
    context_matches = matched_fields & NEWS_MATCH_CONTEXT_FIELDS
    if official_source:
        if len(hard_matches) >= 2 and len(context_matches) >= 4:
            return "high"
        if len(hard_matches) >= 1 and len(context_matches) >= 3:
            return "medium"
        return "low"
    if len(hard_matches) >= 2 and len(soft_matches) >= 1 and len(context_matches) >= 4:
        return "high"
    if len(hard_matches) >= 1 and len(soft_matches) >= 1 and len(context_matches) >= 2:
        return "medium"
    return "low"


def _looks_like_generic_news_page(summary: dict[str, object]) -> bool:
    title = normalize_for_match(str(summary.get("title", "") or ""))
    text = str(summary.get("text", "") or "")
    if any(marker in title for marker in NEWS_GENERIC_TITLE_MARKERS):
        return True
    cnj_refs = {match.group(0) for match in CNJ_PROCESS_RE.finditer(text[:8000])}
    return len(cnj_refs) >= 3


def _is_valid_news_page_summary(summary: Optional[dict[str, object]]) -> bool:
    if not isinstance(summary, dict):
        return False
    if int(summary.get("status_code", 0) or 0) != 200:
        return False
    text = str(summary.get("text", "") or "")
    title = str(summary.get("title", "") or "")
    if not text or len(text) < 120:
        return False
    if NEWS_NOT_FOUND_RE.search(normalize_for_match(f"{title} {text[:800]}")):
        return False
    if _looks_like_generic_news_page(summary):
        return False
    return True


async def _validate_official_news_url_live(
    gerenciador: "GerenciadorRequisicoes",
    url: str,
    *,
    category: str,
    timeout: int = PERPLEXITY_PAGE_FETCH_TIMEOUT,
) -> Optional[str]:
    normalized = _limpar_url_bruta(str(url or ""))
    if category == "tse":
        normalized = _normalize_tse_news_url(normalized) or ""
    elif category == "tre":
        normalized = _normalize_tre_news_url(normalized) or ""
    else:
        return None
    if not normalized:
        return None

    summary = await gerenciador.fetch_page_summary(normalized, timeout=timeout)
    if not _is_valid_news_page_summary(summary):
        return None

    final_url = _limpar_url_bruta(str((summary or {}).get("url") or normalized))
    if category == "tse":
        return _normalize_tse_news_url(final_url) or normalized
    if category == "tre":
        return _normalize_tre_news_url(final_url) or normalized
    return normalized


_OFFICIAL_NEWS_LIVE_CACHE: dict[str, Optional[bool]] = {}
_OFFICIAL_NEWS_LIVE_CACHE_LOCK = threading.Lock()
_OFFICIAL_NEWS_LIVE_SESSION: Optional[requests.Session] = None
_OFFICIAL_NEWS_LIVE_SESSION_LOCK = threading.Lock()


def _get_official_news_live_session() -> requests.Session:
    global _OFFICIAL_NEWS_LIVE_SESSION
    with _OFFICIAL_NEWS_LIVE_SESSION_LOCK:
        if _OFFICIAL_NEWS_LIVE_SESSION is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
            )
            _OFFICIAL_NEWS_LIVE_SESSION = session
        return _OFFICIAL_NEWS_LIVE_SESSION


def official_news_url_live_status(
    url: str,
    *,
    timeout: int = PERPLEXITY_PAGE_FETCH_TIMEOUT,
) -> Optional[bool]:
    normalized = _limpar_url_bruta(str(url or ""))
    if _is_tse_news_url(normalized):
        normalized = _normalize_tse_news_url(normalized) or ""
    elif _is_tre_news_url(normalized):
        normalized = _normalize_tre_news_url(normalized) or ""
    else:
        return None
    if not normalized:
        return False

    with _OFFICIAL_NEWS_LIVE_CACHE_LOCK:
        if normalized in _OFFICIAL_NEWS_LIVE_CACHE:
            return _OFFICIAL_NEWS_LIVE_CACHE[normalized]

    try:
        response = _get_official_news_live_session().get(
            normalized,
            timeout=timeout,
            allow_redirects=True,
        )
        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        raw_html = response.text if "html" in content_type or not content_type else ""
        summary: Optional[dict[str, object]] = {
            "url": _limpar_url_bruta(str(response.url or "")) or normalized,
            "status_code": int(response.status_code),
            "title": _extract_html_title(raw_html),
            "text": _extract_html_text(raw_html, max_chars=PERPLEXITY_PAGE_TEXT_MAX_CHARS),
            "content_type": content_type,
        }
        status: Optional[bool] = _is_valid_news_page_summary(summary)
    except requests.exceptions.Timeout:
        status = None
    except Exception:
        status = None

    with _OFFICIAL_NEWS_LIVE_CACHE_LOCK:
        _OFFICIAL_NEWS_LIVE_CACHE[normalized] = status
    return status


def _news_category_from_url(url: str) -> str:
    if _is_tse_news_url(url):
        return "tse"
    if _is_tre_news_url(url):
        return "tre"
    if _is_general_media_url(url):
        return "gerais"
    return ""


def _register_search_candidates(
    candidates: dict[str, dict[str, dict[str, object]]],
    page_data: dict[str, object],
    results: Sequence[dict[str, str]],
    *,
    query_label: str,
    precisa_tse: bool,
    precisa_tre: bool,
    precisa_gerais: bool,
) -> None:
    for rank, result in enumerate(results):
        url = _limpar_url_bruta(str(result.get("url", "") or ""))
        if not url:
            continue
        category = _news_category_from_url(url)
        if category == "tse" and not precisa_tse:
            continue
        if category == "tre" and not precisa_tre:
            continue
        if category == "gerais" and not precisa_gerais:
            continue
        if not category:
            continue
        combined_text = " ".join(
            part for part in (result.get("title", ""), result.get("snippet", "")) if part
        )
        matched_fields = _match_news_candidate_fields(
            page_data,
            combined_text,
            url=url,
            category=category,
        )
        score = _score_news_candidate_match(matched_fields, rank=rank, query_label=query_label)
        existing = candidates[category].get(url)
        if existing is None:
            candidates[category][url] = {
                "url": url,
                "title": str(result.get("title", "") or ""),
                "snippet": str(result.get("snippet", "") or ""),
                "matched_fields": set(matched_fields),
                "score": score,
                "rank": rank,
            }
            continue
        existing_fields = existing.get("matched_fields", set())
        if not isinstance(existing_fields, set):
            existing_fields = set(existing_fields) if isinstance(existing_fields, (list, tuple, set)) else set()
        existing_fields.update(matched_fields)
        existing["matched_fields"] = existing_fields
        existing["score"] = max(int(existing.get("score", 0) or 0), score)
        existing["rank"] = min(int(existing.get("rank", rank) or rank), rank)
        if not existing.get("title") and result.get("title"):
            existing["title"] = str(result.get("title", "") or "")
        if not existing.get("snippet") and result.get("snippet"):
            existing["snippet"] = str(result.get("snippet", "") or "")


async def _select_validated_candidate_urls(
    gerenciador: GerenciadorRequisicoes,
    page_data: dict[str, object],
    candidate_map: dict[str, dict[str, object]],
    *,
    category: str,
    limit: int,
) -> list[str]:
    official_source = category in {"tse", "tre"}
    if not candidate_map:
        return []
    selected: list[str] = []
    seen: set[str] = set()
    ordered_candidates = sorted(
        candidate_map.values(),
        key=lambda item: (
            -int(item.get("score", 0) or 0),
            int(item.get("rank", 999) or 999),
            str(item.get("url", "") or ""),
        ),
    )
    for candidate in ordered_candidates:
        raw_url = _limpar_url_bruta(str(candidate.get("url", "") or ""))
        if not raw_url:
            continue
        summary = await gerenciador.fetch_page_summary(raw_url, timeout=PERPLEXITY_PAGE_FETCH_TIMEOUT)
        if not _is_valid_news_page_summary(summary):
            continue
        final_url = _limpar_url_bruta(str((summary or {}).get("url", "") or raw_url))
        if category == "tse":
            final_url = _normalize_tse_news_url(final_url) or ""
        elif category == "tre":
            final_url = _normalize_tre_news_url(final_url) or ""
        else:
            final_url = final_url if final_url and _is_general_media_url(final_url) else ""
        if not final_url or final_url in seen:
            continue
        page_blob = " ".join(
            part
            for part in (
                str(candidate.get("title", "") or ""),
                str(candidate.get("snippet", "") or ""),
                str((summary or {}).get("title", "") or ""),
                str((summary or {}).get("text", "") or ""),
            )
            if part
        )
        matched_fields = candidate.get("matched_fields", set())
        if not isinstance(matched_fields, set):
            matched_fields = set(matched_fields) if isinstance(matched_fields, (list, tuple, set)) else set()
        matched_fields = set(matched_fields)
        matched_fields.update(
            _match_news_candidate_fields(
                page_data,
                page_blob,
                url=final_url,
                category=category,
            )
        )
        case_anchor_fields = {
            "nome_municipio",
            "sigla_uf",
            "origem",
            "descricao_classe",
            "nome_tipo_processo",
        }
        detail_anchor_fields = {
            "relator",
            "descricao_classe",
            "nome_tipo_processo",
            "assuntos",
            "tema",
            "punchline",
        }
        official_specific_fields = {
            "relator",
            "nome_tipo_processo",
            "origem",
            "assuntos",
            "tema",
            "punchline",
        }
        general_specific_fields = {
            "relator",
            "nome_tipo_processo",
            "origem",
            "assuntos",
            "tema",
            "punchline",
        }
        has_strong_match = bool(matched_fields & NEWS_MATCH_STRONG_FIELDS)
        if not has_strong_match and not (matched_fields & case_anchor_fields):
            continue
        if not has_strong_match and (
            "partes" not in matched_fields or not (matched_fields & detail_anchor_fields)
        ):
            continue
        if official_source and not has_strong_match and len(matched_fields & official_specific_fields) < 2:
            continue
        if not official_source and not has_strong_match and not (matched_fields & general_specific_fields):
            continue
        if not official_source and not has_strong_match and not (
            matched_fields & GENERAL_NEWS_GEO_ANCHOR_FIELDS
        ):
            continue
        if official_source and not (matched_fields & NEWS_MATCH_SOFT_CONTEXT_FIELDS):
            continue
        confidence = _derive_news_candidate_confidence(
            matched_fields,
            official_source=official_source,
        )
        if not _news_candidate_passes_gate(
            confidence,
            matched_fields,
            official_source=official_source,
        ):
            continue
        seen.add(final_url)
        selected.append(final_url)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def build_news_search_queries(
    page_data: dict[str, object],
    *,
    precisa_tse: bool,
    precisa_tre: bool,
    precisa_gerais: bool,
) -> list[tuple[str, str]]:
    hints = build_news_query_hints(page_data)
    general_only = precisa_gerais and not precisa_tse and not precisa_tre
    numero_unico = _normalize_lookup_text(page_data.get("numero_unico", ""))
    nome_municipio = _normalize_lookup_text(page_data.get("nome_municipio", ""))
    sigla_uf = _normalize_lookup_text(page_data.get("sigla_uf", "")).upper()
    descricao_classe = _normalize_lookup_text(page_data.get("descricao_classe", ""))
    relator = _normalize_lookup_text(page_data.get("relator", ""))
    tema = _normalize_lookup_text(page_data.get("tema", ""))
    assuntos = _normalize_lookup_text(page_data.get("assuntos", ""))
    partes = _extract_news_parties(page_data, max_items=2)
    general_focus_terms = _build_general_media_focus_terms(page_data, max_topics=2, max_parties=1)
    editorial_focus_terms = _build_general_media_focus_terms(page_data, max_topics=2, max_parties=2)
    queries: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, query: str) -> None:
        normalized_query = SPACE_RE.sub(" ", str(query or "")).strip()
        if not normalized_query or normalized_query in seen:
            return
        seen.add(normalized_query)
        queries.append((label, normalized_query))

    if general_only:
        editorial_mainstream_query = _join_query_terms(*general_focus_terms)
        editorial_legal_query = _join_query_terms(numero_unico, *editorial_focus_terms)
        mainstream_filter = _build_site_or_filter(GENERAL_MEDIA_MAINSTREAM_DOMAINS)
        legal_filter = _build_site_or_filter(GENERAL_MEDIA_LEGAL_DOMAINS)
        general_context_query = _join_query_terms(*general_focus_terms) or hints["general_context"]
        if editorial_mainstream_query and mainstream_filter:
            add("general_editorial_mainstream", f"{mainstream_filter} {editorial_mainstream_query}")
        if editorial_legal_query and legal_filter:
            add("general_editorial_legal", f"{legal_filter} {editorial_legal_query}")
        add("general_context", general_context_query)
        return queries

    add(
        "shared_exact",
        _join_query_terms(numero_unico, *partes, nome_municipio, descricao_classe),
    )
    if precisa_tse:
        add(
            "tse_exact",
            "site:tse.jus.br/comunicacao/noticias "
            + _join_query_terms(numero_unico, *partes, nome_municipio, descricao_classe, relator),
        )
    if precisa_tre:
        tre_domain = _derive_tre_domain_hint(page_data)
        add(
            "tre_exact",
            f"site:{tre_domain} " + _join_query_terms(numero_unico, *partes, nome_municipio, descricao_classe),
        )
    if precisa_gerais:
        general_context_query = _join_query_terms(*general_focus_terms) or hints["general_context"]
        add("general_context", general_context_query)
    add(
        "shared_context",
        _join_query_terms(*partes, nome_municipio, sigla_uf, descricao_classe, relator, tema, assuntos),
    )
    if precisa_tse:
        add(
            "tse_context",
            "site:tse.jus.br/comunicacao/noticias "
            + _join_query_terms(*partes, nome_municipio, sigla_uf, descricao_classe, relator),
        )
    if precisa_tre:
        add(
            "tre_context",
            f"site:{tre_domain} "
            + _join_query_terms(*partes, nome_municipio, sigla_uf, descricao_classe, relator),
        )
    if precisa_gerais:
        editorial_query = _join_query_terms(*editorial_focus_terms)
        mainstream_filter = _build_site_or_filter(GENERAL_MEDIA_MAINSTREAM_DOMAINS)
        legal_filter = _build_site_or_filter(GENERAL_MEDIA_LEGAL_DOMAINS)
        if editorial_query and mainstream_filter:
            add("general_editorial_mainstream", f"{mainstream_filter} {editorial_query}")
        editorial_legal_query = _join_query_terms(numero_unico, *editorial_focus_terms)
        if editorial_legal_query and legal_filter:
            add("general_editorial_legal", f"{legal_filter} {editorial_legal_query}")
    return queries


def _news_lookup_targets_satisfied(
    *,
    precisa_tse: bool,
    precisa_tre: bool,
    precisa_gerais: bool,
    needed_general_count: int,
    tse_urls: Sequence[str],
    tre_urls: Sequence[str],
    gerais_urls: Sequence[str],
) -> bool:
    if precisa_tse and not tse_urls:
        return False
    if precisa_tre and not tre_urls:
        return False
    if precisa_gerais and len(gerais_urls) < max(1, int(needed_general_count or 0)):
        return False
    return True


def build_news_lookup_request(
    page_data: dict[str, str],
    *,
    model: str,
    provider: str = "perplexity",
    precisa_tse: bool = True,
    precisa_tre: bool = True,
    needed_general_count: int = 0,
) -> dict[str, object]:
    tema = _normalize_lookup_text(page_data.get("tema", ""))
    punchline = _normalize_lookup_text(page_data.get("punchline", ""))
    precisa_contexto_textual = not tema or not punchline
    texto_ementa_contexto = ""
    texto_decisao_contexto = ""
    if precisa_contexto_textual:
        texto_ementa_contexto = _truncate_lookup_text(
            page_data.get("texto_ementa", ""),
            PERPLEXITY_EMENTA_CONTEXT_MAX_CHARS,
        )
        if not texto_ementa_contexto:
            texto_decisao_contexto = _truncate_lookup_text(
                page_data.get("texto_decisao", ""),
                PERPLEXITY_DECISAO_CONTEXT_MAX_CHARS,
            )
    return {
        "strategy_version": PERPLEXITY_NEWS_STRATEGY_VERSION,
        "provider": _normalize_lookup_text(provider),
        "model": _normalize_lookup_text(model),
        "needs_tse": bool(precisa_tse),
        "needs_tre": bool(precisa_tre),
        "needed_general_count": max(0, int(needed_general_count)),
        "numero_unico": _normalize_lookup_text(page_data.get("numero_unico", "")),
        "data_decisao": _normalize_lookup_text(page_data.get("data_decisao", "")),
        "assuntos": _normalize_lookup_text(page_data.get("assuntos", "")),
        "partes": _normalize_lookup_text(page_data.get("partes", "")),
        "advogados": _normalize_lookup_text(page_data.get("advogados", "")),
        "relator": _normalize_lookup_text(page_data.get("relator", "")),
        "descricao_classe": _normalize_lookup_text(page_data.get("descricao_classe", "")),
        "nome_tipo_processo": _normalize_lookup_text(page_data.get("nome_tipo_processo", "")),
        "sigla_uf": _normalize_lookup_text(page_data.get("sigla_uf", "")),
        "nome_municipio": _normalize_lookup_text(page_data.get("nome_municipio", "")),
        "tribunal": _normalize_lookup_text(page_data.get("tribunal", "")),
        "origem": _normalize_lookup_text(page_data.get("origem", "")),
        "tema": tema,
        "punchline": punchline,
        "texto_ementa_contexto": texto_ementa_contexto,
        "texto_decisao_contexto": texto_decisao_contexto,
    }


def build_news_request_key(request_payload: dict[str, object]) -> str:
    serialized = json.dumps(request_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _normalize_general_news_urls(urls: object) -> list[str]:
    out: list[str] = []
    vistos: set[str] = set()
    if isinstance(urls, str):
        urls_iter: Iterable[object] = [urls]
    elif isinstance(urls, list):
        urls_iter = urls
    else:
        urls_iter = []
    for item in urls_iter:
        url = _limpar_url_bruta(str(item or ""))
        if not url or url in vistos or is_editorially_weak_news_url(url):
            continue
        vistos.add(url)
        out.append(url)
    return out[: len(GENERAL_NEWS_COLUMNS)]


def normalize_news_cache_entry(raw: object) -> Optional[dict[str, object]]:
    if not isinstance(raw, dict):
        return None
    status = _normalize_lookup_text(raw.get("status", "")).lower()
    if status not in {"filled", "no_match", "error_retryable"}:
        return None
    tse = _normalize_tse_news_url(raw.get("tse", "")) or ""
    tre = _normalize_tre_news_url(raw.get("tre", "")) or ""
    gerais = _normalize_general_news_urls(raw.get("gerais", []))
    if status == "no_match" and any([tse, tre, gerais]):
        status = "filled"
    if status == "filled" and not any([tse, tre, gerais]):
        return None
    return {
        "status": status,
        "tse": tse,
        "tre": tre,
        "gerais": gerais,
        "updated_at": _normalize_lookup_text(raw.get("updated_at", "")) or utc_now_iso(),
    }


def _news_cache_entry_age_days(cache_entry: dict) -> float:
    raw = _normalize_lookup_text(cache_entry.get("updated_at", ""))
    if not raw:
        return float("inf")
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0)


def _cache_entry_can_skip_lookup(
    cache_entry: object,
    *,
    precisa_tse: bool,
    precisa_tre: bool,
    precisa_gerais: bool,
    needed_general_count: int,
    treat_retryable_as_terminal: bool = False,
) -> bool:
    if not isinstance(cache_entry, dict):
        return False
    status = _normalize_lookup_text(cache_entry.get("status", "")).lower()
    if status == "no_match":
        return _news_cache_entry_age_days(cache_entry) <= NEWS_CACHE_NO_MATCH_TTL_DAYS
    if status == "error_retryable" and treat_retryable_as_terminal:
        return _news_cache_entry_age_days(cache_entry) <= NEWS_CACHE_RETRYABLE_TTL_HOURS / 24.0
    if status != "filled":
        return False
    tse = _normalize_tse_news_url(cache_entry.get("tse", ""))
    tre = _normalize_tre_news_url(cache_entry.get("tre", ""))
    gerais = _normalize_general_news_urls(cache_entry.get("gerais", []))
    if precisa_tse and not tse:
        return False
    if precisa_tre and not tre:
        return False
    if precisa_gerais and len(gerais) < max(1, int(needed_general_count or 0)):
        return False
    return any([tse, tre, gerais])


def read_news_cache(path: Path) -> dict[str, dict[str, object]]:
    payload = read_json_dict(path)
    version = int(payload.get("version", 0) or 0)
    strategy_version = int(payload.get("strategy_version", 0) or 0)
    if version != 1 or strategy_version != PERPLEXITY_NEWS_STRATEGY_VERSION:
        return {}
    entries_raw = payload.get("entries", {})
    if not isinstance(entries_raw, dict):
        return {}
    entries: dict[str, dict[str, object]] = {}
    for key, value in entries_raw.items():
        if not isinstance(key, str):
            continue
        entry = normalize_news_cache_entry(value)
        if entry is not None:
            entries[key] = entry
    return entries


def write_news_cache(path: Path, entries: dict[str, dict[str, object]]) -> None:
    write_json_atomic(
        path,
        {
            "version": 1,
            "strategy_version": PERPLEXITY_NEWS_STRATEGY_VERSION,
            "updated_at": utc_now_iso(),
            "entries": entries,
        },
    )


def _is_terminal_news_cache_status(status: object) -> bool:
    return _normalize_lookup_text(status).lower() in {"filled", "no_match"}


def estimate_news_api_calls(
    rows: Sequence[dict[str, str]],
    *,
    lookup_payloads: Optional[Sequence[dict[str, str]]] = None,
    model: str = "sonar",
    provider: str = "perplexity",
    cache_path: Optional[Path] = None,
) -> dict[str, int]:
    if lookup_payloads is not None and len(lookup_payloads) != len(rows):
        raise ValueError("lookup_payloads precisa ter o mesmo tamanho de rows.")
    news_cache = read_news_cache(cache_path) if cache_path is not None else {}
    request_keys_needed: set[str] = set()
    skipped_existing = 0
    terminal_cache_hits = 0
    for idx, row in enumerate(rows):
        sanitize_official_news_columns_in_row(row)
        precisa_tse = not bool((row.get("noticia_TSE") or "").strip())
        precisa_tre = not bool((row.get("noticia_TRE") or "").strip())
        missing_general_count = sum(
            not bool((row.get(col) or "").strip()) for col in GENERAL_NEWS_COLUMNS
        )
        required_general_count = _target_general_lookup_count(
            precisa_tse=precisa_tse,
            precisa_tre=precisa_tre,
            missing_general_count=missing_general_count,
        )
        precisa_gerais = required_general_count > 0
        if not (precisa_tse or precisa_tre or precisa_gerais):
            skipped_existing += 1
            continue
        lookup_payload = (
            dict(lookup_payloads[idx])
            if lookup_payloads is not None
            else build_lookup_payload(row)
        )
        request_key = build_news_request_key(
            build_news_lookup_request(
                lookup_payload,
                model=model,
                provider=provider,
                precisa_tse=precisa_tse,
                precisa_tre=precisa_tre,
                needed_general_count=required_general_count,
            )
        )
        cache_entry = news_cache.get(request_key)
        if _cache_entry_can_skip_lookup(
            cache_entry,
            precisa_tse=precisa_tse,
            precisa_tre=precisa_tre,
            precisa_gerais=precisa_gerais,
            needed_general_count=required_general_count,
            treat_retryable_as_terminal=(provider == "gemini"),
        ):
            terminal_cache_hits += 1
            continue
        request_keys_needed.add(request_key)
    return {
        "estimated_api_calls": len(request_keys_needed),
        "estimated_cache_hits": terminal_cache_hits,
        "estimated_skipped_existing": skipped_existing,
    }


TEMA_PUNCHLINE_PROMPT_EMENTA_CHARS = 3200
TEMA_PUNCHLINE_PROMPT_HEAD_CHARS = 2800
TEMA_PUNCHLINE_PROMPT_TAIL_CHARS = 2200


def _tema_punchline_prompt_contexts(page_data: dict[str, str]) -> tuple[str, str]:
    texto_decisao = _normalize_lookup_text(page_data.get("texto_decisao", ""))
    texto_ementa = _normalize_lookup_text(page_data.get("texto_ementa", ""))
    if not texto_ementa and texto_decisao:
        try:
            texto_ementa = _local_ementa_source(texto_decisao, "")
        except Exception:
            texto_ementa = ""
    texto_ementa = _truncate_lookup_text(texto_ementa, TEMA_PUNCHLINE_PROMPT_EMENTA_CHARS)

    if len(texto_decisao) <= TEMA_PUNCHLINE_PROMPT_HEAD_CHARS + TEMA_PUNCHLINE_PROMPT_TAIL_CHARS + 500:
        return texto_ementa, texto_decisao
    head = texto_decisao[:TEMA_PUNCHLINE_PROMPT_HEAD_CHARS].rstrip()
    tail = texto_decisao[-TEMA_PUNCHLINE_PROMPT_TAIL_CHARS:].lstrip()
    return texto_ementa, f"{head}\n\n[trecho intermediario omitido para economia]\n\n{tail}"


def gerar_prompt_tema_punchline(page_data: dict[str, str]) -> str:
    numero_unico = page_data.get("numero_unico", "")
    data_decisao = page_data.get("data_decisao", "")
    assuntos = page_data.get("assuntos", "")
    partes = page_data.get("partes", "")
    relator = page_data.get("relator", "")
    texto_ementa, texto_decisao = _tema_punchline_prompt_contexts(page_data)
    sigla_uf = page_data.get("sigla_uf", "")
    nome_municipio = page_data.get("nome_municipio", "")
    tribunal = page_data.get("tribunal", "")
    origem = page_data.get("origem", "")

    return f"""Com base nos dados abaixo, gere os campos "tema" e "punchline" do julgamento.

Contexto do caso:
numeroUnico: {numero_unico}
dataDecisao: {data_decisao}
assuntos: {assuntos}
partes: {partes}
relator: {relator}
siglaUF: {sigla_uf}
nomeMunicipio: {nome_municipio}
tribunal: {tribunal}
origem: {origem}
textoEmenta: {texto_ementa}
textoDecisao: {texto_decisao}

Instrucoes de saida:
- "tema": produza uma descricao contextual de 110 a 220 caracteres, em uma unica frase nominal, com o instituto eleitoral, o cargo/eleicao quando houver, o ponto juridico central e o efeito pratico relevante. Evite lista de palavras soltas.
- "punchline": em 1 ou 2 frases, descreva a ratio decidendi e a consequencia pratica do julgamento. Priorize o que o TSE decidiu, por que decidiu e o que foi mantido, afastado ou determinado.
- Use portugues juridico claro e objetivo, sem caixa alta excessiva.
- Nao invente fatos fora do texto fornecido.
- Se a decisao for monocratica ou de nao conhecimento/desprovimento, preserve essa informacao quando relevante.
- Retorne somente JSON valido no formato:
{{"tema":"...","punchline":"..."}}"""


LOCAL_SECTION_STOP_RE = re.compile(
    r"(?i)\b(?:relat[oó]rio|voto|[eé]\s+o\s+relat[oó]rio|decido|ante\s+o\s+exposto|publique-?se|intimem-?se)\b"
)
LOCAL_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|;\s+")
LOCAL_HEADER_PREFIX_RE = re.compile(
    r"(?is)^.*?\b(?:relator(?:a)?|agravante|agravado|recorrente|recorrido|requerente|requerido|"
    r"embargante|embargado|impetrante|impetrado|advogad[oa]s?)\s*:"
)
LOCAL_CONCLUSION_RE = re.compile(
    r"(?is)\b(?:ante\s+o\s+exposto|diante\s+do\s+exposto|por\s+essas\s+raz[oõ]es|"
    r"por\s+tais\s+fundamentos|assim)\b.{0,1400}?(?:publique-?se|intimem-?se|comunique-?se|$)"
)
LOCAL_DECISION_MARKER_RE = re.compile(r"(?i)\b(?:decis[aã]o|ac[oó]rd[aã]o|despacho)\b")
LOCAL_TRAILING_STOPWORDS = {
    "a",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "por",
    "que",
}


def _local_compact_text(value: object) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip(" \t\r\n")


def _local_sentence_case(value: str) -> str:
    text = _local_compact_text(value).strip(" .;,:")
    if not text:
        return ""
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
        if upper_ratio > 0.78:
            text = text.lower()
            return text[:1].upper() + text[1:]
    return text[:1].upper() + text[1:]


def _local_trim_at_boundary(value: str, max_chars: int) -> str:
    text = _local_compact_text(value).strip(" .;,:")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    for patterns in ((r"\.\s+", r";\s+"), (r",\s+",), (r"\s+",)):
        candidates: list[int] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if int(max_chars * 0.55) <= match.start() <= max_chars:
                    candidates.append(match.start())
        if candidates:
            cut = max(candidates)
            trimmed = text[:cut].rstrip(" .;,:")
            break
    else:
        trimmed = text[:max_chars].rstrip(" .;,:")
    while True:
        words = trimmed.split()
        if not words or words[-1].casefold().strip(".,;:") not in LOCAL_TRAILING_STOPWORDS:
            return trimmed
        next_trimmed = " ".join(words[:-1]).rstrip(" .;,:")
        if len(next_trimmed) < int(max_chars * 0.45):
            return trimmed
        trimmed = next_trimmed


def _local_ementa_source(texto_decisao: str, texto_ementa: str) -> str:
    ementa = _local_compact_text(texto_ementa)
    if ementa:
        return ementa

    text = _local_compact_text(texto_decisao)
    if not text:
        return ""
    for pattern in (r"(?i)\bementa\b\s*[:\-]?\s*", r"(?i)\b(?:assim\s+)?ementad[oa]\s*:\s*"):
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = text[match.end() :]
        stop = LOCAL_SECTION_STOP_RE.search(candidate)
        if stop and stop.start() >= 80:
            candidate = candidate[: stop.start()]
        if len(candidate) >= 60:
            return candidate

    marker = LOCAL_DECISION_MARKER_RE.search(text)
    if marker and marker.start() < 1800:
        candidate = text[marker.end() :]
        candidate = LOCAL_HEADER_PREFIX_RE.sub("", candidate).strip()
        if len(candidate) >= 60:
            return candidate
    return text[:3000]


def _local_units(value: str, *, max_units: int = 6) -> list[str]:
    source = _local_compact_text(value)
    if not source:
        return []
    raw_units = LOCAL_SENTENCE_SPLIT_RE.split(source)
    units: list[str] = []
    for raw in raw_units:
        unit = _local_sentence_case(re.sub(r"^\d+\s*[.)-]\s*", "", raw))
        if len(unit) < 14:
            continue
        if re.search(r"(?i)^(tribunal|relator|agravante|agravado|recorrente|recorrido|advogad)", unit):
            continue
        units.append(unit)
        if len(units) >= max_units:
            break
    return units


def gerar_tema_punchline_local(page_data: dict[str, str]) -> dict[str, str]:
    texto_decisao = page_data.get("texto_decisao", "")
    texto_ementa = page_data.get("texto_ementa", "")
    descricao_classe = _local_sentence_case(page_data.get("descricao_classe", ""))
    nome_tipo = _local_sentence_case(page_data.get("nome_tipo_processo", ""))
    assuntos = [_local_sentence_case(item) for item in split_multiselect_values(page_data.get("assuntos", ""))]
    resultado = _local_sentence_case(page_data.get("resultado", ""))

    source = _local_ementa_source(texto_decisao, texto_ementa)
    units = _local_units(source, max_units=8)
    if units:
        tema_source = "; ".join(units[:5])
    else:
        fallback_parts = [item for item in [assuntos[0] if assuntos else "", descricao_classe or nome_tipo] if item]
        tema_source = " - ".join(fallback_parts)

    tema = _local_trim_at_boundary(tema_source, 185)
    if not tema:
        numero = page_data.get("numero_unico", "") or "processo sem numero"
        tema = _local_sentence_case(" - ".join(item for item in [descricao_classe or nome_tipo, numero] if item))

    tail = _local_compact_text(texto_decisao)[-4500:]
    conclusion_match = None
    for match in LOCAL_CONCLUSION_RE.finditer(tail):
        conclusion_match = match
    if conclusion_match:
        punchline_source = conclusion_match.group(0)
    elif len(units) > 1:
        punchline_source = ". ".join(units[1:5])
    else:
        punchline_source = tema
    punchline = _local_trim_at_boundary(_local_sentence_case(punchline_source), 620)
    if resultado and resultado.casefold() not in punchline.casefold():
        candidate = f"{punchline}. Resultado: {resultado}"
        punchline = _local_trim_at_boundary(candidate, 700)
    return {"tema": tema, "punchline": punchline}


def preencher_rows_com_tema_punchline_local(
    rows: list[dict[str, str]],
    lookup_payloads: Sequence[dict[str, str]],
) -> dict[str, int]:
    stats = {"tema_preenchidos": 0, "punchline_preenchidos": 0}
    for idx, row in enumerate(rows):
        payload = dict(lookup_payloads[idx]) if idx < len(lookup_payloads) else build_lookup_payload(row)
        payload["resultado"] = row.get("resultado", "")
        generated = gerar_tema_punchline_local(payload)
        if generated.get("tema") and not (row.get("tema") or "").strip():
            row["tema"] = generated["tema"]
            stats["tema_preenchidos"] += 1
        if generated.get("punchline") and not (row.get("punchline") or "").strip():
            row["punchline"] = generated["punchline"]
            stats["punchline_preenchidos"] += 1
    return stats


def gerar_prompt_assuntos(
    page_data: dict[str, str],
    *,
    max_items: int,
    taxonomy_mode: str,
) -> str:
    numero_unico = page_data.get("numero_unico", "")
    data_decisao = page_data.get("data_decisao", "")
    assuntos = page_data.get("assuntos", "")
    partes = page_data.get("partes", "")
    relator = page_data.get("relator", "")
    texto_ementa = page_data.get("texto_ementa", "")
    texto_decisao = page_data.get("texto_decisao", "")
    tribunal = page_data.get("tribunal", "")
    origem = page_data.get("origem", "")
    classe = page_data.get("descricao_classe", "")
    tipo = page_data.get("nome_tipo_processo", "")
    canonic = ", ".join(ASSUNTOS_CANONICOS)

    modo_instrucao = (
        "Use apenas assuntos canônicos da lista."
        if taxonomy_mode == "controlled"
        else (
            "Priorize assuntos canônicos da lista. Se nenhum canônico representar bem o caso, "
            "voce pode incluir um assunto novo curto."
            if taxonomy_mode == "mixed"
            else "Pode usar assuntos canônicos ou novos, mantendo alta precisão."
        )
    )

    return f"""Com base no caso abaixo, preencha o campo "assuntos" com ate {max_items} etiquetas.

Contexto do caso:
numeroUnico: {numero_unico}
dataDecisao: {data_decisao}
classe: {classe}
tipoProcesso: {tipo}
assuntos_existentes: {assuntos}
partes: {partes}
relator: {relator}
tribunal: {tribunal}
origem: {origem}
textoEmenta: {texto_ementa}
textoDecisao: {texto_decisao}

Lista canonica preferencial:
{canonic}

Regras obrigatorias:
- {modo_instrucao}
- Retorne no maximo {max_items} itens.
- Nao repita itens.
- Nao invente fatos fora do texto.
- Responda somente JSON valido no formato:
{{"assuntos":["item1","item2"]}}
"""


def _compact_prompt_value(value: object, max_chars: int = NEWS_PROMPT_VALUE_MAX_CHARS) -> str:
    text = SPACE_RE.sub(" ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.strip(" ,;:")


def gerar_prompt_noticias(page_data: dict[str, object], *, max_general_urls: int) -> str:
    domains_str = ", ".join(VEICULOS_DOMINIOS)
    partes = str(page_data.get("partes", "") or "")
    advogados = str(page_data.get("advogados", "") or "")
    tema = str(page_data.get("tema", "") or "")
    punchline = str(page_data.get("punchline", "") or "")
    texto_ementa_contexto = str(page_data.get("texto_ementa_contexto", "") or "")
    texto_decisao_contexto = str(page_data.get("texto_decisao_contexto", "") or "")
    hints = build_news_query_hints(page_data)
    # Compactacao na renderizacao do prompt (economia de tokens). A chave do cache
    # usa o payload integro de build_news_lookup_request, entao nada e invalidado.
    contexto_campos = [
        ("numeroUnico", page_data.get("numero_unico", "")),
        ("dataDecisao", page_data.get("data_decisao", "")),
        ("tribunal", page_data.get("tribunal", "")),
        ("origem", page_data.get("origem", "")),
        ("siglaUF", page_data.get("sigla_uf", "")),
        ("nomeMunicipio", page_data.get("nome_municipio", "")),
        ("descricaoClasse", page_data.get("descricao_classe", "")),
        ("nomeTipoProcesso", page_data.get("nome_tipo_processo", "")),
        ("assuntos", page_data.get("assuntos", "")),
        ("partes", partes),
        ("advogados", advogados),
        ("relator", page_data.get("relator", "")),
        ("tema", tema),
        ("punchline", punchline),
        ("textoEmentaContexto", texto_ementa_contexto),
        ("textoDecisaoContexto", texto_decisao_contexto),
    ]
    contexto_linhas = "\n".join(
        f"{nome}: {valor_compacto}"
        for nome, valor in contexto_campos
        for valor_compacto in [_compact_prompt_value(valor)]
        if valor_compacto
    )
    return f"""Encontre noticias sobre o MESMO caso eleitoral abaixo e responda APENAS em JSON.

Objetivo:
- "tse": objeto com URL de noticia oficial do TSE.
- "tre": objeto com URL de noticia oficial de TRE estadual.
- "gerais": ate {max_general_urls} objetos com URLs de noticias da imprensa geral.

Fluxo obrigatorio de busca:
1. Pesquise primeiro pelo numero do processo exato entre aspas.
2. Para "tse", use como prioridade absoluta os caminhos de noticia em `tse.jus.br/comunicacao/noticias`.
3. Para "tre", priorize o dominio oficial `{hints["tre_domain"]}` e caminhos com `comunicacao/noticias` ou `noticias`.
4. Para "gerais", use primeiro o numero do processo; se ele nao aparecer em nenhuma materia, aceite noticia com aderencia forte ao mesmo fato, municipio/UF, classe, partes ou relator.
5. Ignore totalmente resultados sobre regras gerais das eleicoes, calendario eleitoral, resolucoes de 2026, IA nas eleicoes, pautas de julgamento genericas, propaganda em abstrato ou paginas meramente institucionais.

Consultas sugeridas:
- TSE exata: {hints["tse_exact"]}
- TRE exata: {hints["tre_exact"]}
- Gerais exata: {hints["general_exact"]}
- TSE contextual: {hints["tse_context"]}
- TRE contextual: {hints["tre_context"]}
- Gerais contextual: {hints["general_context"]}

Regras de resposta:
1. "tse" so pode conter noticia oficial em tse.jus.br com perfil noticioso. Rejeite paginas genericas, arquivo de eleicoes, busca, acordaos, PDFs e paginas administrativas.
2. "tre" so pode conter noticia oficial em dominio tre-xx.jus.br com perfil noticioso. Rejeite consulta processual, busca, PDF, acordao, edital e pagina institucional generica.
3. "gerais" so pode conter URLs nos dominios: {domains_str}
4. Rejeite URLs duplicadas, opiniao, blog, clipping, pagina generica, busca, tag, liveblog ou pagina sem aderencia clara ao caso.
5. Prefira correspondencia por numero do processo. Na falta disso, aceite a noticia quando houver combinacao consistente de relator, tribunal/origem, municipio/UF, classe, partes, data e fato central.
6. Prefira retornar vazio a arriscar link fraco, mas nao exija perfeicao literal. Se a aderencia estiver boa porem sem prova absoluta, use confidence "medium" com matched_fields honestos.
7. Use confidence "high" quando a aderencia ao MESMO caso estiver forte. Use "medium" quando a aderencia estiver plausivel e consistente. Use "low" ou "none" quando houver fragilidade relevante.
8. Em matched_fields, liste apenas nomes deste conjunto quando de fato ajudarem a confirmar o caso: numero_unico, data_decisao, tribunal, origem, sigla_uf, nome_municipio, descricao_classe, nome_tipo_processo, assuntos, partes, relator, tema, punchline.
9. Para "gerais", pode retornar 1 ou 2 links quando ambos forem claramente aderentes e nao duplicados.

Contexto principal:
{contexto_linhas}

Formato JSON obrigatorio:
{{
  "tse": {{"url": "", "confidence": "none", "matched_fields": []}},
  "tre": {{"url": "", "confidence": "none", "matched_fields": []}},
  "gerais": [
    {{"url": "", "confidence": "high", "matched_fields": ["numero_unico", "tema"]}}
  ]
}}
""".strip()


def _normalize_news_lookup_response(
    response: object,
    *,
    precisa_tse: bool,
    precisa_tre: bool,
    precisa_gerais: bool,
) -> tuple[Optional[str], Optional[str], Optional[list[str]]]:
    tse_url = None
    tre_url = None
    gerais_urls: list[str] = []
    if isinstance(response, dict):
        if precisa_tse:
            raw_tse, confidence_tse, matched_tse = _extract_news_candidate(response.get("tse", {}))
            if (
                raw_tse
                and _is_tse_news_url(raw_tse)
                and _news_candidate_passes_gate(
                    confidence_tse,
                    matched_tse,
                    official_source=True,
                )
            ):
                tse_url = raw_tse
        if precisa_tre:
            raw_tre, confidence_tre, matched_tre = _extract_news_candidate(response.get("tre", {}))
            if (
                raw_tre
                and _is_tre_news_url(raw_tre)
                and _news_candidate_passes_gate(
                    confidence_tre,
                    matched_tre,
                    official_source=True,
                )
            ):
                tre_url = raw_tre
        if precisa_gerais:
            raw_gerais = response.get("gerais", [])
            vistos: set[str] = set()
            if isinstance(raw_gerais, list):
                for item in raw_gerais:
                    url, confidence, matched_fields = _extract_news_candidate(item)
                    if (
                        not url
                        or not _is_general_media_url(url)
                        or not _news_candidate_passes_gate(
                            confidence,
                            matched_fields,
                            official_source=False,
                        )
                        or url in vistos
                    ):
                        continue
                    vistos.add(url)
                    gerais_urls.append(url)
    return tse_url, tre_url, (gerais_urls if gerais_urls else None)


async def buscar_todas_noticias_async(
    gerenciador: GerenciadorRequisicoes,
    page_data: dict[str, str],
    precisa_tse: bool,
    precisa_tre: bool,
    precisa_gerais: bool,
    timeout_seconds: int,
    max_tokens: int,
    needed_general_count: int = 0,
) -> dict[str, object]:
    _ = max_tokens
    required_general_count = max(0, int(needed_general_count if precisa_gerais else 0))
    if precisa_gerais and required_general_count <= 0:
        required_general_count = _target_general_lookup_count(
            precisa_tse=precisa_tse,
            precisa_tre=precisa_tre,
            missing_general_count=len(GENERAL_NEWS_COLUMNS),
        )
    if getattr(gerenciador, "provider", "perplexity") == "gemini":
        response = await gerenciador.call_news_lookup(
            prompt=gerar_prompt_noticias(page_data, max_general_urls=max(1, required_general_count or 1)),
            timeout=timeout_seconds,
            max_tokens=max(GEMINI_DEFAULT_MAX_OUTPUT_TOKENS, int(max_tokens or 0)),
            max_general_urls=max(1, required_general_count or 1),
        )
        if response is None:
            return {"status": "ok", "tse": None, "tre": None, "gerais": None}
        tse_url, tre_url, gerais_urls = _normalize_news_lookup_response(
            response,
            precisa_tse=precisa_tse,
            precisa_tre=precisa_tre,
            precisa_gerais=precisa_gerais,
        )
        if tse_url:
            tse_url = await _validate_official_news_url_live(
                gerenciador,
                tse_url,
                category="tse",
                timeout=timeout_seconds,
            )
        if tre_url:
            tre_url = await _validate_official_news_url_live(
                gerenciador,
                tre_url,
                category="tre",
                timeout=timeout_seconds,
            )
        return {
            "status": "ok",
            "tse": tse_url,
            "tre": tre_url,
            "gerais": gerais_urls,
        }
    search_queries = build_news_search_queries(
        page_data,
        precisa_tse=precisa_tse,
        precisa_tre=precisa_tre,
        precisa_gerais=precisa_gerais,
    )
    if not search_queries:
        return {
            "status": "ok",
            "tse": None,
            "tre": None,
            "gerais": None,
        }
    had_retryable_error = False
    candidates: dict[str, dict[str, dict[str, object]]] = {
        "tse": {},
        "tre": {},
        "gerais": {},
    }
    tse_urls: list[str] = []
    tre_urls: list[str] = []
    gerais_urls: list[str] = []
    for query_label, query in search_queries:
        need_tse = precisa_tse and not tse_urls
        need_tre = precisa_tre and not tre_urls
        need_gerais = precisa_gerais and len(gerais_urls) < required_general_count
        if not (need_tse or need_tre or need_gerais):
            break
        if query_label.startswith("tse_") and not need_tse:
            continue
        if query_label.startswith("tre_") and not need_tre:
            continue
        results = await gerenciador.search_news_results(
            query=query,
            timeout=timeout_seconds,
            max_tokens_per_page=_search_max_tokens_per_page_for_query_label(query_label),
            search_domain_order=_search_domain_order_for_query_label(query_label),
        )
        if results is None:
            had_retryable_error = True
            continue
        _register_search_candidates(
            candidates,
            page_data,
            results,
            query_label=query_label,
            precisa_tse=need_tse,
            precisa_tre=need_tre,
            precisa_gerais=need_gerais,
        )
        if need_tse:
            tse_urls = await _select_validated_candidate_urls(
                gerenciador,
                page_data,
                candidates["tse"],
                category="tse",
                limit=1,
            )
        if need_tre:
            tre_urls = await _select_validated_candidate_urls(
                gerenciador,
                page_data,
                candidates["tre"],
                category="tre",
                limit=1,
            )
        if need_gerais:
            gerais_urls = await _select_validated_candidate_urls(
                gerenciador,
                page_data,
                candidates["gerais"],
                category="gerais",
                limit=required_general_count,
            )
        if _news_lookup_targets_satisfied(
            precisa_tse=precisa_tse,
            precisa_tre=precisa_tre,
            precisa_gerais=precisa_gerais,
            needed_general_count=required_general_count,
            tse_urls=tse_urls,
            tre_urls=tre_urls,
            gerais_urls=gerais_urls,
        ):
            break

    if any([tse_urls, tre_urls, gerais_urls]):
        return {
            "status": "ok",
            "tse": tse_urls[0] if tse_urls else None,
            "tre": tre_urls[0] if tre_urls else None,
            "gerais": gerais_urls or None,
        }
    if had_retryable_error:
        return {
            "status": "error_retryable",
            "tse": None,
            "tre": None,
            "gerais": None,
        }
    return {
        "status": "ok",
        "tse": None,
        "tre": None,
        "gerais": None,
    }


def _aplicar_urls_no_row(row: dict[str, str], url_tse: Optional[str], url_tre: Optional[str], urls_gerais: Optional[list[str]]) -> None:
    sanitize_official_news_columns_in_row(row)
    url_tse = _normalize_tse_news_url(url_tse) if url_tse else None
    url_tre = _normalize_tre_news_url(url_tre) if url_tre else None
    if url_tse and not row.get("noticia_TSE"):
        row["noticia_TSE"] = url_tse
    if url_tre and not row.get("noticia_TRE"):
        row["noticia_TRE"] = url_tre

    gerais_normalizadas: list[str] = []
    vistos_gerais = {
        _limpar_url_bruta(str(row.get(col, "") or ""))
        for col in GENERAL_NEWS_COLUMNS
        if _limpar_url_bruta(str(row.get(col, "") or ""))
    }
    for raw_url in urls_gerais or []:
        url = _limpar_url_bruta(str(raw_url or ""))
        if not url or url in vistos_gerais or is_editorially_weak_news_url(url):
            continue
        vistos_gerais.add(url)
        gerais_normalizadas.append(url)
    if not gerais_normalizadas:
        return

    idx_url = 0
    for col in GENERAL_NEWS_COLUMNS:
        if row.get(col):
            continue
        if idx_url >= len(gerais_normalizadas):
            break
        row[col] = gerais_normalizadas[idx_url]
        idx_url += 1


async def enriquecer_rows_com_urls_async(
    rows: list[dict[str, str]],
    logger: Callable[[str], None],
    config: WebLookupConfig,
    lookup_payloads: Optional[Sequence[dict[str, str]]] = None,
    on_batch_done: Optional[Callable[[int, int, int, Dict[str, int]], None]] = None,
    cache_path: Optional[Path] = None,
    row_indices: Optional[Sequence[int]] = None,
    row_progress: Optional[dict[str, dict[str, object]]] = None,
) -> dict[str, int]:
    if not rows:
        return {
            "perplexity_api_calls": 0,
            "perplexity_cache_hits": 0,
            "perplexity_no_match": 0,
            "perplexity_retryable_errors": 0,
            "perplexity_skipped_existing": 0,
        }
    if lookup_payloads is not None and len(lookup_payloads) != len(rows):
        raise ValueError("lookup_payloads precisa ter o mesmo tamanho de rows.")
    if row_indices is not None and len(row_indices) != len(rows):
        raise ValueError("row_indices precisa ter o mesmo tamanho de rows.")
    gerenciador: Optional[GerenciadorRequisicoes] = None
    sanitize_official_news_columns(rows)
    news_cache = read_news_cache(cache_path) if cache_path is not None else {}
    provider_label = "Gemini" if config.provider == "gemini" else "Perplexity"
    metrics = {
        "perplexity_api_calls": 0,
        "perplexity_cache_hits": 0,
        "perplexity_no_match": 0,
        "perplexity_retryable_errors": 0,
        "perplexity_skipped_existing": 0,
    }
    try:
        total = len(rows)
        for start in range(0, total, config.batch_size):
            end = min(start + config.batch_size, total)
            lote = rows[start:end]
            lote_payloads = (
                list(lookup_payloads[start:end])
                if lookup_payloads is not None
                else [build_lookup_payload(row) for row in lote]
            )
            logger(f"[{provider_label}] Processando lote {start + 1}-{end} de {total}")
            batch_stage = f"{config.provider}_batch_{start + 1}_{end}_of_{total}"
            planos: list[dict[str, object]] = []
            tarefas_por_chave: dict[str, asyncio.Task] = {}
            for offset, (row, lookup_data) in enumerate(zip(lote, lote_payloads)):
                absolute_index = row_indices[start + offset] if row_indices is not None else start + offset
                sanitize_official_news_columns_in_row(row)
                precisa_tse = not bool((row.get("noticia_TSE") or "").strip())
                precisa_tre = not bool((row.get("noticia_TRE") or "").strip())
                missing_general_count = sum(
                    not bool((row.get(col) or "").strip()) for col in GENERAL_NEWS_COLUMNS
                )
                required_general_count = _target_general_lookup_count(
                    precisa_tse=precisa_tse,
                    precisa_tre=precisa_tre,
                    missing_general_count=missing_general_count,
                )
                precisa_gerais = required_general_count > 0
                if not (precisa_tse or precisa_tre or precisa_gerais):
                    plano = {
                        "source": "existing",
                        "row": row,
                        "absolute_index": absolute_index,
                        "request_key": "",
                    }
                    planos.append(plano)
                    if row_progress is not None:
                        row_progress[str(absolute_index)] = {
                            "stage": batch_stage,
                            "request_key": "",
                            "news_status": "skipped_existing",
                        }
                    continue
                request_payload = build_news_lookup_request(
                    lookup_data,
                    model=config.model,
                    provider=config.provider,
                    precisa_tse=precisa_tse,
                    precisa_tre=precisa_tre,
                    needed_general_count=required_general_count,
                )
                request_key = build_news_request_key(request_payload)
                cache_entry = news_cache.get(request_key)
                if _cache_entry_can_skip_lookup(
                    cache_entry,
                    precisa_tse=precisa_tse,
                    precisa_tre=precisa_tre,
                    precisa_gerais=precisa_gerais,
                    needed_general_count=required_general_count,
                    treat_retryable_as_terminal=(config.provider == "gemini"),
                ):
                    planos.append(
                        {
                            "source": "cache",
                            "row": row,
                            "absolute_index": absolute_index,
                            "request_key": request_key,
                            "cache_entry": cache_entry,
                        }
                    )
                    if row_progress is not None:
                        row_progress[str(absolute_index)] = {
                            "stage": batch_stage,
                            "request_key": request_key,
                            "news_status": str(cache_entry.get("status", "")),
                        }
                    continue
                if gerenciador is None:
                    gerenciador = GerenciadorRequisicoes(
                        api_key=config.api_key,
                        model=config.model,
                        max_workers=config.max_workers,
                        retries=GEMINI_DEFAULT_RETRIES if config.provider == "gemini" else PERPLEXITY_DEFAULT_RETRIES,
                        target_rpm=GEMINI_DEFAULT_TARGET_RPM if config.provider == "gemini" else PERPLEXITY_DEFAULT_TARGET_RPM,
                        provider=config.provider,
                    )
                planos.append(
                    {
                        "source": "api",
                        "row": row,
                        "absolute_index": absolute_index,
                        "request_key": request_key,
                        "task_key": request_key,
                    }
                )
                if request_key not in tarefas_por_chave:
                    tarefas_por_chave[request_key] = asyncio.create_task(
                        buscar_todas_noticias_async(
                            gerenciador,
                            lookup_data,
                            precisa_tse=precisa_tse,
                            precisa_tre=precisa_tre,
                            precisa_gerais=precisa_gerais,
                            needed_general_count=required_general_count,
                            timeout_seconds=config.timeout_seconds,
                            max_tokens=config.max_tokens,
                        )
                    )
                if row_progress is not None:
                    row_progress[str(absolute_index)] = {
                        "stage": batch_stage,
                        "request_key": request_key,
                        "news_status": "pending",
                    }
            resultados_por_chave: dict[str, dict[str, object]] = {}
            if tarefas_por_chave:
                async def _resolve_task(
                    request_key: str,
                    task: asyncio.Task,
                ) -> tuple[str, dict[str, object]]:
                    result = await task
                    if isinstance(result, dict):
                        return request_key, result
                    return request_key, {
                        "status": "error_retryable",
                        "tse": None,
                        "tre": None,
                        "gerais": None,
                    }

                wrapped_tasks = [
                    asyncio.create_task(_resolve_task(request_key, task))
                    for request_key, task in tarefas_por_chave.items()
                ]
                completed_api = 0
                total_api = len(wrapped_tasks)
                last_progress_log = 0.0
                for completed in asyncio.as_completed(wrapped_tasks):
                    request_key, result = await completed
                    resultados_por_chave[request_key] = result
                    completed_api += 1
                    now = time.monotonic()
                    if (
                        completed_api == 1
                        or completed_api == total_api
                        or now - last_progress_log >= PERPLEXITY_PROGRESS_LOG_INTERVAL_SECONDS
                    ):
                        logger(
                            f"[{provider_label}] Lote "
                            f"{start + 1}-{end} progresso | consultas={completed_api}/{total_api}"
                        )
                        last_progress_log = now
            batch_tse = 0
            batch_tre = 0
            batch_gerais = 0
            batch_api_calls = len(tarefas_por_chave)
            batch_cache_hits = 0
            batch_no_match = 0
            batch_retryable_errors = 0
            batch_skipped_existing = 0
            cache_dirty = False
            for plano in planos:
                row = plano["row"]
                absolute_index = int(plano.get("absolute_index", 0) or 0)
                request_key = str(plano.get("request_key", "") or "")
                before_tse = bool((row.get("noticia_TSE") or "").strip())
                before_tre = bool((row.get("noticia_TRE") or "").strip())
                before_gerais = sum(bool((row.get(col) or "").strip()) for col in GENERAL_NEWS_COLUMNS)
                source = str(plano.get("source", "") or "")
                url_tse = None
                url_tre = None
                urls_gerais = None
                news_status = "pending"
                if source == "existing":
                    batch_skipped_existing += 1
                    metrics["perplexity_skipped_existing"] += 1
                    news_status = "skipped_existing"
                elif source == "cache":
                    cache_entry = plano.get("cache_entry", {})
                    batch_cache_hits += 1
                    metrics["perplexity_cache_hits"] += 1
                    if isinstance(cache_entry, dict):
                        url_tse = _normalize_tse_news_url(cache_entry.get("tse", "")) or None
                        url_tre = _normalize_tre_news_url(cache_entry.get("tre", "")) or None
                        urls_gerais = _normalize_general_news_urls(cache_entry.get("gerais", [])) or None
                        news_status = str(cache_entry.get("status", "") or "pending")
                    if news_status == "no_match":
                        batch_no_match += 1
                        metrics["perplexity_no_match"] += 1
                elif source == "api":
                    resultado = resultados_por_chave.get(
                        str(plano.get("task_key", "") or ""),
                        {"status": "error_retryable", "tse": None, "tre": None, "gerais": None},
                    )
                    if isinstance(resultado, dict):
                        raw_status = str(resultado.get("status", "") or "").strip().lower()
                        if raw_status == "ok":
                            url_tse = _normalize_tse_news_url(resultado.get("tse", "")) or None
                            url_tre = _normalize_tre_news_url(resultado.get("tre", "")) or None
                            urls_gerais = _normalize_general_news_urls(resultado.get("gerais", [])) or None
                            news_status = "filled" if any([url_tse, url_tre, urls_gerais]) else "no_match"
                            if news_status == "no_match":
                                batch_no_match += 1
                                metrics["perplexity_no_match"] += 1
                        else:
                            news_status = "error_retryable"
                            batch_retryable_errors += 1
                            metrics["perplexity_retryable_errors"] += 1
                    else:
                        news_status = "error_retryable"
                        batch_retryable_errors += 1
                        metrics["perplexity_retryable_errors"] += 1
                    if request_key:
                        news_cache[request_key] = {
                            "status": news_status,
                            "tse": url_tse or "",
                            "tre": url_tre or "",
                            "gerais": urls_gerais or [],
                            "updated_at": utc_now_iso(),
                        }
                        cache_dirty = True
                _aplicar_urls_no_row(row, url_tse, url_tre, urls_gerais)
                after_tse = bool((row.get("noticia_TSE") or "").strip())
                after_tre = bool((row.get("noticia_TRE") or "").strip())
                after_gerais = sum(bool((row.get(col) or "").strip()) for col in GENERAL_NEWS_COLUMNS)
                if not before_tse and after_tse:
                    batch_tse += 1
                if not before_tre and after_tre:
                    batch_tre += 1
                if after_gerais > before_gerais:
                    batch_gerais += after_gerais - before_gerais
                if row_progress is not None:
                    row_progress[str(absolute_index)] = {
                        "stage": batch_stage,
                        "request_key": request_key,
                        "news_status": news_status,
                    }
            metrics["perplexity_api_calls"] += batch_api_calls
            if cache_dirty and cache_path is not None:
                write_news_cache(cache_path, news_cache)
            logger(
                f"[{provider_label}] Lote "
                f"{start + 1}-{end} concluido | novos_tse={batch_tse} | "
                f"novos_tre={batch_tre} | novas_gerais={batch_gerais} | "
                f"api={batch_api_calls} | cache={batch_cache_hits} | "
                f"no_match={batch_no_match} | retryable={batch_retryable_errors}"
            )
            if on_batch_done is not None:
                on_batch_done(
                    start,
                    end,
                    total,
                    {
                        "novos_tse": batch_tse,
                        "novos_tre": batch_tre,
                        "novas_gerais": batch_gerais,
                        "perplexity_api_calls": batch_api_calls,
                        "perplexity_cache_hits": batch_cache_hits,
                        "perplexity_no_match": batch_no_match,
                        "perplexity_retryable_errors": batch_retryable_errors,
                        "perplexity_skipped_existing": batch_skipped_existing,
                    },
                )
            if end < total and config.delay_between_batches > 0:
                await asyncio.sleep(config.delay_between_batches)
    finally:
        if gerenciador is not None:
            tokens_prompt = int(getattr(gerenciador, "news_tokens_prompt_total", 0) or 0)
            tokens_output = int(getattr(gerenciador, "news_tokens_output_total", 0) or 0)
            if tokens_prompt or tokens_output:
                logger(
                    f"[{provider_label}] Tokens consumidos na busca de noticias: "
                    f"prompt={tokens_prompt} | saida={tokens_output}"
                )
            gerenciador.close()
    return metrics


def _aplicar_assuntos_no_row(
    row: dict[str, str],
    assuntos: Sequence[str],
    *,
    max_items: int,
    taxonomy_mode: str,
) -> None:
    if (row.get("assuntos") or "").strip():
        return
    sanitized: list[str] = []
    for item in assuntos:
        normalized = normalize_assunto_value(str(item or ""), taxonomy_mode=taxonomy_mode)
        if not normalized:
            continue
        sanitized.append(normalized)
    sanitized = dedupe_preserve(sanitized, key_func=normalize_for_match)
    if not sanitized:
        return
    row["assuntos"] = ",".join(sanitized[: max(1, int(max_items))])


async def enriquecer_rows_com_assuntos_openai_async(
    rows: list[dict[str, str]],
    logger: Callable[[str], None],
    config: AssuntosEnrichmentConfig,
    lookup_payloads: Optional[Sequence[dict[str, str]]] = None,
    on_batch_done: Optional[Callable[[int, int, int, Dict[str, int]], None]] = None,
) -> None:
    if not rows:
        return
    if lookup_payloads is not None and len(lookup_payloads) != len(rows):
        raise ValueError("lookup_payloads precisa ter o mesmo tamanho de rows.")
    gerenciador = GerenciadorOpenAI(
        api_key=config.api_key,
        model=config.model,
        max_workers=config.max_workers,
        retries=config.retries,
        target_rpm=config.target_rpm,
    )
    try:
        total = len(rows)
        for start in range(0, total, config.batch_size):
            end = min(start + config.batch_size, total)
            lote = rows[start:end]
            lote_payloads = (
                list(lookup_payloads[start:end])
                if lookup_payloads is not None
                else [build_lookup_payload(row) for row in lote]
            )
            logger(f"[Assuntos/OpenAI] Processando lote {start + 1}-{end} de {total}")
            tarefas = []
            linhas_consultadas = 0
            for row, lookup_data in zip(lote, lote_payloads):
                if (row.get("assuntos") or "").strip():
                    tarefas.append(asyncio.sleep(0, result=None))
                    continue
                linhas_consultadas += 1
                prompt = gerar_prompt_assuntos(
                    lookup_data,
                    max_items=config.max_items,
                    taxonomy_mode=config.taxonomy_mode,
                )
                tarefas.append(
                    gerenciador.call_assuntos(
                        prompt,
                        max_items=config.max_items,
                        timeout=config.timeout_seconds,
                    )
                )
            resultados = await asyncio.gather(*tarefas)

            assuntos_preenchidos = 0
            sem_retorno = 0
            for row, resultado in zip(lote, resultados):
                antes = bool((row.get("assuntos") or "").strip())
                if not isinstance(resultado, dict):
                    if not antes:
                        sem_retorno += 1
                    continue
                raw_assuntos = resultado.get("assuntos", [])
                if not isinstance(raw_assuntos, list):
                    if not antes:
                        sem_retorno += 1
                    continue
                _aplicar_assuntos_no_row(
                    row,
                    [str(item or "") for item in raw_assuntos],
                    max_items=config.max_items,
                    taxonomy_mode=config.taxonomy_mode,
                )
                depois = bool((row.get("assuntos") or "").strip())
                if not antes and depois:
                    assuntos_preenchidos += 1
            logger(
                "[Assuntos/OpenAI] Lote "
                f"{start + 1}-{end} concluido | linhas_consultadas={linhas_consultadas} | "
                f"assuntos_preenchidos={assuntos_preenchidos} | sem_retorno={sem_retorno}"
            )
            if on_batch_done is not None:
                on_batch_done(
                    start,
                    end,
                    total,
                    {
                        "linhas_consultadas": linhas_consultadas,
                        "assuntos_preenchidos": assuntos_preenchidos,
                        "sem_retorno": sem_retorno,
                    },
                )
            if end < total and config.delay_between_batches > 0:
                await asyncio.sleep(config.delay_between_batches)
    finally:
        gerenciador.close()


def _aplicar_tema_punchline_no_row(row: dict[str, str], tema: str, punchline: str) -> None:
    tema_limpo, _ = sanitize_cell(tema, max_chars=0, replace_newlines=True)
    punchline_limpo, _ = sanitize_cell(punchline, max_chars=0, replace_newlines=True)
    if tema_limpo and not (row.get("tema") or "").strip():
        row["tema"] = tema_limpo
    if punchline_limpo and not (row.get("punchline") or "").strip():
        row["punchline"] = punchline_limpo


async def enriquecer_rows_com_tema_punchline_async(
    rows: list[dict[str, str]],
    logger: Callable[[str], None],
    config: TemaPunchlineConfig,
    lookup_payloads: Optional[Sequence[dict[str, str]]] = None,
    on_batch_done: Optional[Callable[[int, int, int, Dict[str, int]], None]] = None,
) -> None:
    if not rows:
        return
    if lookup_payloads is not None and len(lookup_payloads) != len(rows):
        raise ValueError("lookup_payloads precisa ter o mesmo tamanho de rows.")
    gerenciador = GerenciadorOpenAI(
        api_key=config.api_key,
        model=config.model,
        max_workers=config.max_workers,
        retries=config.retries,
        target_rpm=config.target_rpm,
    )
    try:
        total = len(rows)
        for start in range(0, total, config.batch_size):
            end = min(start + config.batch_size, total)
            lote = rows[start:end]
            lote_payloads = (
                list(lookup_payloads[start:end])
                if lookup_payloads is not None
                else [build_lookup_payload(row) for row in lote]
            )
            logger(f"[ChatGPT] Processando lote {start + 1}-{end} de {total}")
            tarefas = []
            requested_rows = 0
            for row, lookup_data in zip(lote, lote_payloads):
                precisa_tema = not bool((row.get("tema") or "").strip())
                precisa_punchline = not bool((row.get("punchline") or "").strip())
                if not (precisa_tema or precisa_punchline):
                    tarefas.append(asyncio.sleep(0, result=None))
                    continue
                requested_rows += 1
                prompt = gerar_prompt_tema_punchline(lookup_data)
                tarefas.append(gerenciador.call_tema_punchline(prompt, timeout=config.timeout_seconds))

            resultados = await asyncio.gather(*tarefas)
            tema_preenchidos = 0
            punchline_preenchidos = 0
            sem_retorno = 0
            for row, resultado in zip(lote, resultados):
                tema_antes = bool((row.get("tema") or "").strip())
                punchline_antes = bool((row.get("punchline") or "").strip())
                if not isinstance(resultado, dict):
                    if not tema_antes or not punchline_antes:
                        sem_retorno += 1
                    continue
                _aplicar_tema_punchline_no_row(
                    row,
                    tema=str(resultado.get("tema", "") or ""),
                    punchline=str(resultado.get("punchline", "") or ""),
                )
                tema_depois = bool((row.get("tema") or "").strip())
                punchline_depois = bool((row.get("punchline") or "").strip())
                if not tema_antes and tema_depois:
                    tema_preenchidos += 1
                if not punchline_antes and punchline_depois:
                    punchline_preenchidos += 1
            logger(
                "[ChatGPT] Lote "
                f"{start + 1}-{end} concluido | linhas_consultadas={requested_rows} | "
                f"tema_preenchidos={tema_preenchidos} | "
                f"punchline_preenchidos={punchline_preenchidos} | "
                f"sem_retorno={sem_retorno}"
            )
            if on_batch_done is not None:
                on_batch_done(
                    start,
                    end,
                    total,
                    {
                        "linhas_consultadas": requested_rows,
                        "tema_preenchidos": tema_preenchidos,
                        "punchline_preenchidos": punchline_preenchidos,
                        "sem_retorno": sem_retorno,
                    },
                )
            if end < total and config.delay_between_batches > 0:
                await asyncio.sleep(config.delay_between_batches)
    finally:
        gerenciador.close()


def format_partes_as_multiselect(value: str) -> str:
    if not value:
        return ""

    return ", ".join(split_multiselect_values(value))


def clean_relator_prefix(value: str) -> str:
    if not value:
        return ""
    cleaned = RELATOR_PREFIX_RE.sub("", value)
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def format_numero_unico_cnj(value: str) -> str:
    if not value:
        return ""
    digits = CNJ_DIGITS_RE.sub("", value)
    if len(digits) != 20:
        return value
    return (
        f"{digits[0:7]}-{digits[7:9]}.{digits[9:13]}."
        f"{digits[13]}.{digits[14:16]}.{digits[16:20]}"
    )


def clean_numero_processo(value: str) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D+", "", value)
    if digits:
        return digits
    return SPACE_RE.sub(" ", value).strip()


def normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized


def dedupe_preserve(items: Iterable[str], key_func: Callable[[str], str] | None = None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    key_callable = key_func or (lambda item: item)
    for item in items:
        key = key_callable(item)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def split_multiselect_values(value: str) -> list[str]:
    if not value:
        return []
    items: list[str] = []
    for part in PARTES_SPLIT_RE.split(value):
        item = SPACE_RE.sub(" ", part).strip(" ,;")
        if item:
            items.append(item)
    return dedupe_preserve(items, key_func=normalize_for_match)


def merge_multiselect_values(*values: str) -> str:
    merged: list[str] = []
    for value in values:
        merged.extend(split_multiselect_values(value))
    merged = dedupe_preserve(merged, key_func=normalize_for_match)
    return ", ".join(merged)


def format_br_number(number: str) -> str:
    digits = re.sub(r"\D", "", number or "")
    if not digits:
        return ""
    return f"{int(digits):,}".replace(",", ".")


def format_law_reference(law_type: str, number: str, year: str) -> str:
    law_type_norm = normalize_for_match(law_type)
    number_fmt = format_br_number(number)
    year_short = year[-2:] if year else ""
    year_suffix = f"/{year_short}" if year_short else ""

    if "lei complementar" in law_type_norm:
        return f"LC n° {number_fmt}{year_suffix}"
    if "decreto-lei" in law_type_norm:
        return f"Decreto-Lei n° {number_fmt}{year_suffix}"
    if "resolucao do tribunal superior eleitoral" in law_type_norm:
        return f"Res. TSE n° {number_fmt}{year_suffix}"
    if "emenda constitucional" in law_type_norm:
        return f"EC n° {number_fmt}{year_suffix}"
    if "constituicao federal" in law_type_norm:
        return "CF/88"
    return f"Lei n° {number_fmt}{year_suffix}"


def normalize_article(article_text: str) -> str:
    text = SPACE_RE.sub(" ", article_text).strip(" ,;")
    text = re.sub(r"(?i)^art\.?", "art.", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text


def _normalize_reference_year(value: str) -> str:
    digits = re.sub(r"\D+", "", value or "")
    if len(digits) == 4:
        return digits
    if len(digits) == 2:
        return f"20{digits}" if int(digits) <= 30 else f"19{digits}"
    return ""


def _infer_reference_year(law_type: str, number: str) -> str:
    law_type_norm = normalize_for_match(law_type)
    digits = re.sub(r"\D+", "", number or "")
    if not digits:
        return ""
    if "lei complementar" in law_type_norm and digits == "64":
        return "1990"
    if "lei" in law_type_norm and digits == "9504":
        return "1997"
    if "lei" in law_type_norm and digits == "4737":
        return "1965"
    if "lei" in law_type_norm and digits == "13105":
        return "2015"
    return ""


def _format_detected_law_reference(law_type: str, number: str, year: str) -> str:
    year_norm = _normalize_reference_year(year) or _infer_reference_year(law_type, number)
    return format_law_reference(law_type=law_type, number=number, year=year_norm)


def _canonical_law_label_from_fragment(fragment: str) -> str:
    text = SPACE_RE.sub(" ", fragment).strip(" ,;")
    if not text:
        return ""
    if CF_ALIAS_RE.search(text):
        return "CF/88"
    if LEI_DAS_ELEICOES_ALIAS_RE.search(text):
        return LEI_ELEICOES_LABEL
    if LEI_INELEGIBILIDADES_ALIAS_RE.search(text):
        return LC_64_LABEL
    if CODIGO_ELEITORAL_ALIAS_RE.search(text):
        return CODIGO_ELEITORAL_LABEL
    if CPC_ALIAS_RE.search(text):
        return CPC_LABEL

    for pattern, law_type in (
        (LC_TEXT_RE, "lei complementar"),
        (DECRETO_LEI_TEXT_RE, "decreto-lei"),
        (RES_TSE_TEXT_RE, "resolucao do tribunal superior eleitoral"),
        (EC_TEXT_RE, "emenda constitucional"),
        (LEI_TEXT_RE, "lei"),
    ):
        match = pattern.search(text)
        if not match:
            continue
        label = _format_detected_law_reference(
            law_type=law_type,
            number=match.group("num"),
            year=match.groupdict().get("ano", ""),
        )
        if label:
            return label
    return ""


def _extract_law_mentions(text: str) -> list[tuple[int, int, str]]:
    if not text:
        return []

    mentions: list[tuple[int, int, str]] = []
    for match in LEGAL_ENTRY_RE.finditer(text):
        label = format_law_reference(
            law_type=match.group("tipo"),
            number=match.group("num"),
            year=match.group("ano"),
        )
        if label:
            mentions.append((match.start(), match.end(), label))

    for pattern, law_type in (
        (LC_TEXT_RE, "lei complementar"),
        (DECRETO_LEI_TEXT_RE, "decreto-lei"),
        (RES_TSE_TEXT_RE, "resolucao do tribunal superior eleitoral"),
        (EC_TEXT_RE, "emenda constitucional"),
        (LEI_TEXT_RE, "lei"),
    ):
        for match in pattern.finditer(text):
            label = _format_detected_law_reference(
                law_type=law_type,
                number=match.group("num"),
                year=match.groupdict().get("ano", ""),
            )
            if label:
                mentions.append((match.start(), match.end(), label))

    for pattern, label in (
        (CF_ALIAS_RE, "CF/88"),
        (LEI_DAS_ELEICOES_ALIAS_RE, LEI_ELEICOES_LABEL),
        (LEI_INELEGIBILIDADES_ALIAS_RE, LC_64_LABEL),
        (CODIGO_ELEITORAL_ALIAS_RE, CODIGO_ELEITORAL_LABEL),
        (CPC_ALIAS_RE, CPC_LABEL),
    ):
        for match in pattern.finditer(text):
            mentions.append((match.start(), match.end(), label))

    return sorted(mentions, key=lambda item: (item[0], item[1], item[2]))


def _normalize_article_label(number: str, mods: str) -> str:
    number_clean = SPACE_RE.sub("", (number or "").replace("–", "-").replace("—", "-"))
    number_clean = number_clean.replace("°", "º")
    mods_clean = SPACE_RE.sub(" ", mods or "").strip(" ,;")
    mods_clean = re.sub(r"\s*,\s*", " ", mods_clean)
    mods_clean = re.sub(r"(?i)\binc\.?\s*", "", mods_clean)
    mods_clean = mods_clean.replace("°", "º")
    mods_clean = re.sub(
        r"(?i)(?<!al[íi]nea\s)\b([ivxlcdm]+)\b",
        lambda match: match.group(1).upper(),
        mods_clean,
    )
    mods_clean = SPACE_RE.sub(" ", mods_clean).strip()
    if not number_clean:
        return ""
    return f"art. {number_clean}" if not mods_clean else f"art. {number_clean} {mods_clean}"


def _extract_article_mentions(text: str) -> list[tuple[int, int, str]]:
    if not text:
        return []

    mentions: list[tuple[int, int, str]] = []
    for token_match in ARTICLE_TOKEN_RE.finditer(text):
        segment = text[token_match.end() : token_match.end() + 180]
        if not segment:
            continue
        cut_match = ARTICLE_TO_LAW_CUT_RE.search(segment)
        if cut_match:
            body = segment[: cut_match.start()]
        else:
            sentence_cut = re.search(r"(?:[;\n)]|\.(?=\s+[A-ZÀ-ÖØ-Ý]))", segment)
            body = segment[: sentence_cut.start()] if sentence_cut else segment
        body = re.sub(r"(?i)\b(?:ambos?|todas?|todos?)\b.*$", "", body).strip(" ,;")
        if not body:
            continue
        body_start = token_match.end()
        for unit_match in ARTICLE_UNIT_RE.finditer(body):
            label = _normalize_article_label(unit_match.group("num"), unit_match.group("mods"))
            if not label:
                continue
            mentions.append((body_start + unit_match.start(), body_start + unit_match.end(), label))
    return mentions


def _pair_article_mentions_with_laws(
    text: str,
    articles: Sequence[tuple[int, int, str]],
    laws: Sequence[tuple[int, int, str]],
) -> list[tuple[str, str]]:
    if not text or not articles or not laws:
        return []

    pairs: list[tuple[str, str]] = []
    for article_start, article_end, article_label in articles:
        best_choice: tuple[tuple[int, int, int], str] | None = None
        for law_start, law_end, law_label in laws:
            gap = text[min(article_end, law_end) : max(article_start, law_start)]
            if len(gap) > 140:
                continue
            if re.search(r"(?:[;\n]|\.(?=\s+[A-ZÀ-ÖØ-Ý]))", gap):
                continue
            score = (len(gap), 0 if law_start >= article_start else 1, law_start)
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, law_label)
        if best_choice is None:
            continue
        pairs.append((article_label, best_choice[1]))
    return pairs


def extract_referencias_legislativas(*texts: str) -> str:
    final_labels: list[str] = []
    seen_labels: set[str] = set()
    paired_laws: set[str] = set()

    for raw_text in texts:
        text = SPACE_RE.sub(" ", raw_text or "").strip()
        if not text:
            continue
        laws = _extract_law_mentions(text)
        articles = _extract_article_mentions(text)
        for article_label, law_label in _pair_article_mentions_with_laws(text, articles, laws):
            combined = f"{article_label} {law_label}"
            combined = SPACE_RE.sub(" ", combined).strip()
            combined_key = normalize_for_match(combined)
            if combined_key in seen_labels:
                continue
            seen_labels.add(combined_key)
            paired_laws.add(normalize_for_match(law_label))
            final_labels.append(combined)

        for _start, _end, law_label in laws:
            law_key = normalize_for_match(law_label)
            if law_key in paired_laws or law_key in seen_labels:
                continue
            seen_labels.add(law_key)
            final_labels.append(law_label)

    return ", ".join(final_labels)


def clean_referencias_legislativas(value: str) -> str:
    if not value:
        return ""

    text = SPACE_RE.sub(" ", value).strip()
    return extract_referencias_legislativas(text)


def extract_composicao_multiselect(*texts: str) -> str:
    source = " ".join(item for item in texts if item)
    if not source:
        return ""
    if not COMPOSICAO_TRIGGER_RE.search(source):
        return ""

    segments = [match.group(0) for match in COMPOSICAO_SEGMENT_RE.finditer(source)]
    if not segments:
        segments = [source]

    search_space = normalize_for_match(" ".join(segments))
    found: list[str] = []
    for canonical_name, aliases in CANONICAL_MINISTERS:
        if any(alias in search_space for alias in aliases):
            found.append(canonical_name)

    return ",".join(found)


def _prepare_metadata_header_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    cleaned = SPACE_RE.sub(" ", text).strip()
    if not cleaned:
        return ""
    end_match = HEADER_END_RE.search(cleaned)
    if end_match:
        cleaned = cleaned[: end_match.start()]
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned.strip()


def _split_candidate_on_conjunction(candidate: str) -> list[str]:
    if not candidate or " e " not in candidate.lower():
        return [candidate]
    if ENTITY_INSTITUTION_RE.search(candidate):
        return [candidate]
    parts = [piece.strip(" ,;.-") for piece in re.split(r"(?i)\s+e\s+", candidate) if piece.strip(" ,;.-")]
    if len(parts) == 2 and all(len(part.split()) >= 2 for part in parts):
        return parts
    return [candidate]


def _clean_entity_candidate(raw_value: str) -> str:
    cleaned = SPACE_RE.sub(" ", raw_value or "").strip(" ,;.-")
    if not cleaned:
        return ""
    cleaned = ENTITY_INLINE_LABEL_CUT_RE.split(cleaned)[0]
    cleaned = SPACE_RE.sub(" ", cleaned).strip(" ,;.-")
    if not cleaned:
        return ""
    cleaned = ENTITY_OUTROS_SUFFIX_RE.sub("", cleaned)
    cleaned = ENTITY_OAB_TAIL_RE.sub("", cleaned)
    cleaned = ENTITY_UF_NUM_SUFFIX_RE.sub("", cleaned)
    cleaned = ENTITY_ID_TAIL_RE.sub("", cleaned)
    cleaned = ENTITY_CNPJ_TAIL_RE.sub("", cleaned)
    cleaned = ADV_TITLES_PREFIX_RE.sub("", cleaned)
    cleaned = SPACE_RE.sub(" ", cleaned).strip(" ,;.-")
    return cleaned


def _split_header_block_entities(block: str, config: MetadataExtractionConfig) -> list[str]:
    if not block:
        return []
    trimmed = SPACE_RE.sub(" ", block).strip(" ,;.-")
    if not trimmed:
        return []
    trimmed = ENTITY_INLINE_LABEL_CUT_RE.split(trimmed)[0]
    trimmed = SPACE_RE.sub(" ", trimmed).strip(" ,;.-")
    if not trimmed:
        return []

    entities: list[str] = []
    for chunk in ENTITY_SPLIT_RE.split(trimmed):
        cleaned_chunk = _clean_entity_candidate(chunk)
        if not cleaned_chunk:
            continue
        for candidate in _split_candidate_on_conjunction(cleaned_chunk):
            final_candidate = _clean_entity_candidate(candidate)
            if not final_candidate:
                continue
            if ENTITY_DECISION_NOISE_RE.search(final_candidate):
                continue
            if len(final_candidate.split()) > max(2, int(config.max_entity_words)):
                continue
            entities.append(final_candidate)
    return dedupe_preserve(entities, key_func=normalize_for_match)


def _is_adv_role(role_norm: str) -> bool:
    return role_norm.startswith("advogad") or role_norm.startswith("representante do") or role_norm.startswith(
        "representantes do"
    )


def _is_party_role(role_norm: str) -> bool:
    return any(role_norm.startswith(prefix) for prefix in HEADER_PARTY_ROLE_PREFIXES)


# Ruido de CABECALHO de processo (PJe) que vazava para advogados/partes:
# "CLASSE (codigo) Nº numero-CNJ (PJe) - COMARCA - UF", "Referencia: Peticao id.",
# etc. Nenhum desses padroes aparece em nome de pessoa, entao e seguro rejeitar.
PROCESS_HEADER_NOISE_RE = re.compile(
    r"\(\s*pje\s*\)"                                   # (PJe)
    r"|\d{7}-\d{2}\.\d{4}\.\d"                         # numero CNJ
    r"|\bn[ºo°]\s*\d{4,}"                              # No 060...
    r"|\brefer[êe]ncia\s*:|\bid\.?\s*\d{6,}"           # Referencia:/Peticao id. 1652...
    r"|\(\s*\d{4,6}\s*\)"                              # codigo de classe (11550)
    r"|\b(?:recurso\s+(?:ordin[aá]rio|especial)|embargos\s+de\s+declara[cç][aã]o|"
    r"agravo\s+(?:interno|regimental|de\s+instrumento)|mandado\s+de\s+seguran[cç]a|"
    r"a[cç][aã]o\s+(?:cautelar|rescis[oó]ria)|presta[cç][aã]o\s+de\s+contas|"
    r"registro\s+de\s+candidatura|habeas\s+corpus|"
    r"tutela\s+(?:cautelar|antecipada|de\s+urg[êe]ncia))\b",
    re.IGNORECASE,
)


def _is_valid_advogado_entity(entity: str, *, include_institutional_entities: bool) -> bool:
    cleaned = SPACE_RE.sub(" ", str(entity or "")).strip(" ,;.-")
    if not cleaned:
        return False
    if PROCESS_HEADER_NOISE_RE.search(cleaned):
        return False
    if ENTITY_DECISION_NOISE_RE.search(cleaned):
        return False
    if re.search(r"(?i)\b(?:art\.?|s[uú]mula|ac[oó]rd[aã]o|lei|fls?\.?)\b", cleaned):
        return False
    if re.search(r"(?i)\b(?:ministr[oa]|relator(?:a)?)\b", cleaned):
        return False
    if re.search(r"(?i)\b(?:fundo\s+partid[aá]rio|inelegibil|propaganda|mat[eé]ria|doa[cç][aã]o)\b", cleaned):
        return False
    if ENTITY_INSTITUTION_RE.search(cleaned):
        return include_institutional_entities

    words = cleaned.split()
    if len(words) < 2:
        return False
    lower_words = 0
    for word in words:
        token = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", word)
        if not token:
            continue
        if token.lower() in {"de", "da", "do", "das", "dos", "e", "d"}:
            continue
        if not token[0].isupper():
            lower_words += 1
    return lower_words <= 1


def sanitize_advogados_multiselect(value: str, config: MetadataExtractionConfig) -> str:
    if not value:
        return ""
    cleaned_items: list[str] = []
    for item in split_multiselect_values(value):
        candidate = _clean_entity_candidate(item)
        if not _is_valid_advogado_entity(
            candidate,
            include_institutional_entities=config.include_institutional_entities,
        ):
            continue
        cleaned_items.append(candidate)
    cleaned_items = dedupe_preserve(cleaned_items, key_func=normalize_for_match)
    return ", ".join(cleaned_items)


def extract_header_metadata(
    *texts: str,
    config: Optional[MetadataExtractionConfig] = None,
) -> dict[str, list[str]]:
    metadata_config = config or MetadataExtractionConfig()
    partes: list[str] = []
    advogados: list[str] = []

    for text in texts:
        header_text = _prepare_metadata_header_text(text, max_chars=metadata_config.header_max_chars)
        if not header_text:
            continue
        labels = list(HEADER_LABEL_RE.finditer(header_text))
        if not labels:
            continue
        for index, label_match in enumerate(labels):
            role_norm = normalize_for_match(label_match.group(1))
            start = label_match.end()
            end = labels[index + 1].start() if index + 1 < len(labels) else len(header_text)
            entities = _split_header_block_entities(header_text[start:end], metadata_config)
            if not entities:
                continue
            if _is_adv_role(role_norm):
                for entity in entities:
                    if not _is_valid_advogado_entity(
                        entity,
                        include_institutional_entities=metadata_config.include_institutional_entities,
                    ):
                        continue
                    advogados.append(entity)
                continue
            if _is_party_role(role_norm):
                partes.extend(entities)

    return {
        "partes": dedupe_preserve(partes, key_func=normalize_for_match),
        "advogados": dedupe_preserve(advogados, key_func=normalize_for_match),
    }


def extract_partes_multiselect(*texts: str, config: Optional[MetadataExtractionConfig] = None) -> str:
    metadata = extract_header_metadata(*texts, config=config)
    return ", ".join(metadata.get("partes", []))


def extract_advogados_multiselect(*texts: str, config: Optional[MetadataExtractionConfig] = None) -> str:
    metadata = extract_header_metadata(*texts, config=config)
    return ", ".join(metadata.get("advogados", []))


def _canonical_assuntos_map() -> dict[str, str]:
    canonical = {normalize_for_match(item): item for item in ASSUNTOS_CANONICOS}
    for alias, canonical_label in ASSUNTOS_ALIAS_TO_CANONICAL.items():
        canonical[normalize_for_match(alias)] = canonical_label
    return canonical


def normalize_assunto_value(value: str, taxonomy_mode: str) -> str:
    raw = SPACE_RE.sub(" ", str(value or "")).strip(" ,;")
    if not raw:
        return ""
    if len(raw.split()) > 8:
        return ""
    if ENTITY_DECISION_NOISE_RE.search(raw):
        return ""
    canonical_map = _canonical_assuntos_map()
    normalized_key = normalize_for_match(raw)
    canonical = canonical_map.get(normalized_key, "")
    if canonical:
        return canonical
    if taxonomy_mode == "controlled":
        return ""
    if taxonomy_mode not in ASSUNTOS_TAXONOMY_CHOICES:
        taxonomy_mode = "mixed"
    # mixed/free: aceita etiqueta nova curta e limpa.
    return raw


def extract_assuntos_deterministic(
    *,
    descricao_classe: str,
    nome_tipo_processo: str,
    texto_decisao: str,
    texto_ementa: str,
    max_items: int,
    taxonomy_mode: str,
) -> list[str]:
    max_items_safe = max(1, int(max_items))
    scores: dict[str, int] = defaultdict(int)
    order = {label: idx for idx, label in enumerate(ASSUNTOS_CANONICOS)}

    descricao_scope = " ".join((descricao_classe or "", nome_tipo_processo or ""))
    for label, pattern in ASSUNTOS_RULES_DESCRICAO:
        if pattern.search(descricao_scope):
            scores[label] += 5

    texto_scope = " ".join(
        (
            _prepare_metadata_header_text(texto_decisao or "", max_chars=900),
            (texto_ementa or "")[:1200],
        )
    )
    for label, pattern in ASSUNTOS_RULES_TEXT:
        if pattern.search(texto_scope):
            weight = 1 if label == "Matéria processual" else 3
            scores[label] += weight

    if not scores:
        return []

    sorted_labels = sorted(scores.items(), key=lambda item: (-item[1], order.get(item[0], 10_000), item[0]))
    extracted: list[str] = []
    for label, _score in sorted_labels:
        normalized = normalize_assunto_value(label, taxonomy_mode=taxonomy_mode)
        if not normalized:
            continue
        extracted.append(normalized)
        if len(extracted) >= max_items_safe:
            break
    return dedupe_preserve(extracted, key_func=normalize_for_match)


def classify_resultado(texto_decisao: str, descricao_tipo_decisao: str = "") -> str:
    scope = texto_decisao[:5000] if texto_decisao else ""
    descricao_norm = normalize_for_match(descricao_tipo_decisao)
    labels: list[str] = []

    # Em decisoes monocraticas, menções a "por unanimidade"/"por maioria"
    # costumam descrever o julgamento na origem, nao o ato atual do TSE.
    if "monocratica" in descricao_norm:
        labels.append("monocrática")
    elif UNANIME_RE.search(scope):
        labels.append("unânime")
    elif MAIORIA_RE.search(scope):
        labels.append("por maioria")
    elif MONOCRATICA_RE.search(scope):
        labels.append("monocrática")
    else:
        if "monocratica" in descricao_norm:
            labels.append("monocrática")
        else:
            labels.append("monocrática")

    scope_norm = normalize_for_match(scope)
    if NAO_CONHECIMENTO_NORM_RE.search(scope_norm):
        labels.append("não conhecido")

    return ",".join(dedupe_preserve(labels, key_func=lambda item: item.casefold()))


def process_one_csv(
    input_path: Path,
    out_dir: Path,
    max_texto_chars: int,
    replace_newlines: bool,
    web_lookup_config: WebLookupConfig,
    local_tema_punchline_config: Optional[LocalTemaPunchlineConfig] = None,
    tema_punchline_config: Optional[TemaPunchlineConfig] = None,
    assuntos_enrichment_config: Optional[AssuntosEnrichmentConfig] = None,
    metadata_extraction_config: Optional[MetadataExtractionConfig] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> ProcessSummary:
    local_tema_punchline_config = local_tema_punchline_config or LocalTemaPunchlineConfig(enabled=False)
    tema_punchline_config = tema_punchline_config or TemaPunchlineConfig(enabled=False, api_key="")
    assuntos_enrichment_config = assuntos_enrichment_config or AssuntosEnrichmentConfig(enabled=False, api_key="")
    metadata_extraction_config = metadata_extraction_config or MetadataExtractionConfig()
    logger = logger or (lambda _message: None)

    encoding, delimiter = detect_csv_format(input_path)
    logger(f"[Lendo] {input_path.name} (encoding={encoding}, delimiter='{delimiter}')")

    with input_path.open("r", encoding=encoding, newline="") as src:
        reader = csv.DictReader(src, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{input_path.name}: sem cabecalho valido")

        used_headers: set[str] = set()
        clean_headers = [sanitize_header(name, used_headers) for name in reader.fieldnames]
        mapping = list(zip(reader.fieldnames, clean_headers))
        source_key_by_clean = {clean: original for original, clean in mapping}

        output_path = out_dir / f"{input_path.stem}_notion.csv"
        checkpoint_path = resolve_checkpoint_artifact_path(output_path)
        report_path = resolve_report_artifact_path(output_path)
        cache_path = resolve_web_lookup_cache_path(out_dir)
        source_sig = build_file_signature(input_path)
        started_at = time.time()
        output_fields: list[str] = []
        seen_output_fields: set[str] = set()
        for header in clean_headers:
            mapped = remap_output_header(header)
            if is_excluded_column(mapped):
                continue
            if mapped in seen_output_fields:
                continue
            output_fields.append(mapped)
            seen_output_fields.add(mapped)
        for derived in ("composicao", "advogados", "resultado", "referenciasLegislativas", *THEME_COLUMNS):
            if derived not in output_fields:
                output_fields.append(derived)
        for url_column in URL_COLUMNS:
            if url_column not in output_fields:
                output_fields.append(url_column)

        if output_path.exists() and not checkpoint_path.exists():
            backup = make_backup(
                output_path,
                backup_dir=resolve_backup_artifacts_dir(),
                label="startup_backup",
            )
            if backup is not None:
                logger(f"[Backup] {output_path.name} -> {backup.name}")

        processed_rows: list[dict[str, str]] = []
        lookup_payloads: list[dict[str, str]] = []
        truncated_cells = 0
        partes_added_total = 0
        advogados_added_total = 0
        assuntos_filled_rules_total = 0
        assuntos_filled_openai_total = 0
        restored_from_output_total = 0
        row_progress: dict[str, dict[str, object]] = {}
        perplexity_metrics_total = {
            "perplexity_api_calls": 0,
            "perplexity_cache_hits": 0,
            "perplexity_no_match": 0,
            "perplexity_retryable_errors": 0,
            "perplexity_skipped_existing": 0,
        }

        for row_index, source_row in enumerate(reader, start=1):
            clean_row: dict[str, str] = {}
            for original_key, clean_key in mapping:
                if is_excluded_column(clean_key):
                    continue

                per_cell_limit = max_texto_chars if clean_key in {"textoEmenta", "textoDecisao"} else 0
                clean_value, was_truncated = sanitize_cell(
                    source_row.get(original_key, ""),
                    max_chars=per_cell_limit,
                    replace_newlines=replace_newlines,
                )
                if clean_key == "partes":
                    clean_value = format_partes_as_multiselect(clean_value)
                elif clean_key in {"relatores", "relator"}:
                    clean_value = clean_relator_prefix(clean_value)
                elif clean_key == "referenciasLegislativas":
                    clean_value = clean_referencias_legislativas(clean_value)
                elif clean_key == "numeroUnico":
                    clean_value = format_numero_unico_cnj(clean_value)
                elif clean_key == "numeroProcesso":
                    clean_value = clean_numero_processo(clean_value)
                clean_row[clean_key] = clean_value
                if was_truncated:
                    truncated_cells += 1

            texto_decisao_key = source_key_by_clean.get("textoDecisao")
            texto_ementa_key = source_key_by_clean.get("textoEmenta")
            referencias_key = source_key_by_clean.get("referenciasLegislativas")
            descricao_tipo_decisao_key = source_key_by_clean.get("descricaoTipoDecisao")
            numero_unico_key = source_key_by_clean.get("numeroUnico")
            numero_processo_key = source_key_by_clean.get("numeroProcesso")
            data_decisao_key = source_key_by_clean.get("dataDecisao")
            assuntos_key = source_key_by_clean.get("assuntos")
            partes_key = source_key_by_clean.get("partes")
            advogados_key = source_key_by_clean.get("advogados")
            relator_key = source_key_by_clean.get("relator") or source_key_by_clean.get("relatores")
            descricao_classe_key = source_key_by_clean.get("descricaoClasse")
            nome_tipo_processo_key = source_key_by_clean.get("nomeTipoProcesso")
            sigla_tribunal_je_key = source_key_by_clean.get("siglaTribunalJE")
            origem_decisao_key = source_key_by_clean.get("origemDecisao")
            sigla_uf_key = source_key_by_clean.get("siglaUF")
            nome_municipio_key = source_key_by_clean.get("nomeMunicipio")
            texto_decisao_raw = source_row.get(texto_decisao_key, "") if texto_decisao_key else ""
            texto_ementa_raw = source_row.get(texto_ementa_key, "") if texto_ementa_key else ""
            referencias_raw = source_row.get(referencias_key, "") if referencias_key else ""
            descricao_tipo_decisao_raw = (
                source_row.get(descricao_tipo_decisao_key, "") if descricao_tipo_decisao_key else ""
            )
            numero_unico_raw = source_row.get(numero_unico_key, "") if numero_unico_key else ""
            numero_processo_raw = source_row.get(numero_processo_key, "") if numero_processo_key else ""
            data_decisao_raw = source_row.get(data_decisao_key, "") if data_decisao_key else ""
            assuntos_raw = source_row.get(assuntos_key, "") if assuntos_key else ""
            partes_raw = source_row.get(partes_key, "") if partes_key else ""
            advogados_raw = source_row.get(advogados_key, "") if advogados_key else ""
            relator_raw = source_row.get(relator_key, "") if relator_key else ""
            descricao_classe_raw = source_row.get(descricao_classe_key, "") if descricao_classe_key else ""
            nome_tipo_processo_raw = source_row.get(nome_tipo_processo_key, "") if nome_tipo_processo_key else ""
            sigla_tribunal_je_raw = source_row.get(sigla_tribunal_je_key, "") if sigla_tribunal_je_key else ""
            origem_decisao_raw = source_row.get(origem_decisao_key, "") if origem_decisao_key else ""
            sigla_uf_raw = source_row.get(sigla_uf_key, "") if sigla_uf_key else ""
            nome_municipio_raw = source_row.get(nome_municipio_key, "") if nome_municipio_key else ""
            texto_decisao_full, _ = sanitize_cell(texto_decisao_raw, max_chars=0, replace_newlines=replace_newlines)
            texto_ementa_full, _ = sanitize_cell(texto_ementa_raw, max_chars=0, replace_newlines=replace_newlines)
            referencias_full, _ = sanitize_cell(referencias_raw, max_chars=0, replace_newlines=replace_newlines)
            descricao_tipo_decisao_full, _ = sanitize_cell(
                descricao_tipo_decisao_raw,
                max_chars=0,
                replace_newlines=replace_newlines,
            )
            numero_unico_full, _ = sanitize_cell(numero_unico_raw, max_chars=0, replace_newlines=replace_newlines)
            numero_processo_full, _ = sanitize_cell(numero_processo_raw, max_chars=0, replace_newlines=replace_newlines)
            data_decisao_full, _ = sanitize_cell(data_decisao_raw, max_chars=0, replace_newlines=replace_newlines)
            assuntos_full, _ = sanitize_cell(assuntos_raw, max_chars=0, replace_newlines=replace_newlines)
            partes_full, _ = sanitize_cell(partes_raw, max_chars=0, replace_newlines=replace_newlines)
            advogados_full, _ = sanitize_cell(advogados_raw, max_chars=0, replace_newlines=replace_newlines)
            relator_full, _ = sanitize_cell(relator_raw, max_chars=0, replace_newlines=replace_newlines)
            descricao_classe_full, _ = sanitize_cell(descricao_classe_raw, max_chars=0, replace_newlines=replace_newlines)
            nome_tipo_processo_full, _ = sanitize_cell(
                nome_tipo_processo_raw,
                max_chars=0,
                replace_newlines=replace_newlines,
            )
            sigla_tribunal_je_full, _ = sanitize_cell(
                sigla_tribunal_je_raw,
                max_chars=0,
                replace_newlines=replace_newlines,
            )
            origem_decisao_full, _ = sanitize_cell(
                origem_decisao_raw,
                max_chars=0,
                replace_newlines=replace_newlines,
            )
            sigla_uf_full, _ = sanitize_cell(sigla_uf_raw, max_chars=0, replace_newlines=replace_newlines)
            nome_municipio_full, _ = sanitize_cell(nome_municipio_raw, max_chars=0, replace_newlines=replace_newlines)
            numero_unico_full = format_numero_unico_cnj(numero_unico_full)
            numero_processo_full = clean_numero_processo(numero_processo_full)
            data_decisao_full = normalize_date_to_mdy(
                data_decisao_full,
                prefer_day_first=CSV_INPUT_DATES_DAY_FIRST,
            )
            numero_referencia = numero_unico_full or numero_processo_full
            partes_full = format_partes_as_multiselect(partes_full)
            advogados_full = sanitize_advogados_multiselect(advogados_full, metadata_extraction_config)
            relator_full = clean_relator_prefix(relator_full)
            metadata = extract_header_metadata(
                texto_decisao_full,
                texto_ementa_full,
                config=metadata_extraction_config,
            )
            partes_extraidas = ", ".join(metadata.get("partes", []))
            advogados_extraidos = ", ".join(metadata.get("advogados", []))
            partes_merged = merge_multiselect_values(partes_full, partes_extraidas)
            advogados_merged = merge_multiselect_values(advogados_full, advogados_extraidos)
            assuntos_deterministic = extract_assuntos_deterministic(
                descricao_classe=descricao_classe_full,
                nome_tipo_processo=nome_tipo_processo_full,
                texto_decisao=texto_decisao_full,
                texto_ementa=texto_ementa_full,
                max_items=assuntos_enrichment_config.max_items,
                taxonomy_mode=assuntos_enrichment_config.taxonomy_mode,
            )
            assuntos_merged = assuntos_full.strip()
            if not assuntos_merged and assuntos_deterministic:
                assuntos_merged = ",".join(assuntos_deterministic)
                assuntos_filled_rules_total += 1

            partes_added_total += max(
                0,
                len(split_multiselect_values(partes_merged)) - len(split_multiselect_values(partes_full)),
            )
            advogados_added_total += max(
                0,
                len(split_multiselect_values(advogados_merged)) - len(split_multiselect_values(advogados_full)),
            )

            for date_key in list(clean_row.keys()):
                if date_key.lower() in {"datadecisao", "data_decisao", "datajulgamento", "data_publicacao"}:
                    clean_row[date_key] = normalize_date_to_mdy(
                        clean_row.get(date_key, ""),
                        prefer_day_first=CSV_INPUT_DATES_DAY_FIRST,
                    )

            if "relatores" in clean_row or "relator" in clean_row:
                relator_value = clean_row.get("relator", "") or clean_row.get("relatores", "")
                clean_row["relator"] = relator_value
                clean_row.pop("relatores", None)

            if "partes" in clean_row or "partes" in output_fields:
                clean_row["partes"] = partes_merged
            if "assuntos" in clean_row or "assuntos" in output_fields:
                clean_row["assuntos"] = assuntos_merged
            clean_row["referenciasLegislativas"] = extract_referencias_legislativas(
                referencias_full,
                texto_decisao_full,
                texto_ementa_full,
            )
            clean_row["composicao"] = extract_composicao_multiselect(texto_decisao_full, texto_ementa_full)
            clean_row["advogados"] = advogados_merged
            clean_row["resultado"] = classify_resultado(
                texto_decisao=texto_decisao_full,
                descricao_tipo_decisao=descricao_tipo_decisao_full,
            )
            for theme_column in THEME_COLUMNS:
                clean_row[theme_column] = clean_row.get(theme_column, "")

            for url_column in URL_COLUMNS:
                clean_row[url_column] = clean_row.get(url_column, "")
            sanitize_official_news_columns_in_row(clean_row)

            processed_rows.append(clean_row)
            lookup_payloads.append(
                {
                    "numero_unico": numero_referencia,
                    "data_decisao": data_decisao_full,
                    "assuntos": assuntos_merged,
                    "partes": partes_merged,
                    "advogados": advogados_merged,
                    "relator": relator_full,
                    "descricao_classe": descricao_classe_full,
                    "nome_tipo_processo": nome_tipo_processo_full,
                    "texto_decisao": texto_decisao_full,
                    "texto_ementa": texto_ementa_full,
                    "sigla_uf": sigla_uf_full,
                    "nome_municipio": nome_municipio_full,
                    "tribunal": " ".join(
                        part for part in (sigla_tribunal_je_full, origem_decisao_full) if part
                    ).strip(),
                    "origem": " ".join(part for part in (sigla_uf_full, nome_municipio_full) if part).strip(),
                    "tema": clean_row.get("tema", "") or "",
                    "punchline": clean_row.get("punchline", "") or "",
                }
            )
            if row_index % ROW_PROGRESS_EVERY == 0:
                logger(
                    f"[Processamento] {input_path.name}: {row_index} linhas preparadas "
                    f"(truncadas={truncated_cells})"
                )

        logger(
            f"[Processamento] {input_path.name}: total de {len(processed_rows)} linhas preparadas "
            f"(truncadas={truncated_cells})"
        )

        def _count_filled(col: str) -> int:
            return sum(1 for row in processed_rows if (row.get(col) or "").strip())

        def _count_empty_assuntos() -> int:
            return sum(1 for row in processed_rows if not (row.get("assuntos") or "").strip())

        def _count_filled_gerais() -> int:
            total = 0
            for row in processed_rows:
                for col in GENERAL_NEWS_COLUMNS:
                    if (row.get(col) or "").strip():
                        total += 1
            return total

        def _refresh_lookup_payloads_from_rows() -> None:
            for idx, row in enumerate(processed_rows):
                if idx >= len(lookup_payloads):
                    break
                lookup_payloads[idx]["assuntos"] = row.get("assuntos", "") or ""
                lookup_payloads[idx]["partes"] = row.get("partes", "") or ""
                lookup_payloads[idx]["advogados"] = row.get("advogados", "") or ""
                lookup_payloads[idx]["tema"] = row.get("tema", "") or ""
                lookup_payloads[idx]["punchline"] = row.get("punchline", "") or ""

        def _save_state(status: str, stage: str, extra: Optional[Dict[str, int]] = None) -> None:
            completed_at = utc_now_iso() if status == "completed" else ""
            write_csv_atomic(output_path, output_fields, processed_rows)
            write_json_atomic(
                checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "source_signature": source_sig,
                    "input_csv": str(input_path.resolve()),
                    "output_csv": str(output_path.resolve()),
                    "status": status,
                    "stage": stage,
                    "tema_enabled": bool(tema_punchline_config.enabled),
                    "assuntos_openai_enabled": bool(assuntos_enrichment_config.enabled),
                    "perplexity_enabled": bool(web_lookup_config.enabled),
                    "web_lookup_provider": web_lookup_config.provider,
                    "strategy_version": PERPLEXITY_NEWS_STRATEGY_VERSION,
                    "cache_file": str(cache_path.resolve()),
                    "row_progress": row_progress,
                    "processed_rows": processed_rows,
                    "completed_at": completed_at,
                    "updated_at": utc_now_iso(),
                },
            )
            write_json_atomic(
                report_path,
                {
                    "script": "SJUR_csv_to_csv_NOTIONfriendly.py",
                    "input_csv": str(input_path.resolve()),
                    "output_csv": str(output_path.resolve()),
                    "checkpoint_file": str(checkpoint_path.resolve()),
                    "cache_file": str(cache_path.resolve()),
                    "status": status,
                    "stage": stage,
                    "strategy_version": PERPLEXITY_NEWS_STRATEGY_VERSION,
                    "web_lookup_provider": web_lookup_config.provider,
                    "rows_total": len(processed_rows),
                    "tema_filled": _count_filled("tema"),
                    "punchline_filled": _count_filled("punchline"),
                    "partes_added": partes_added_total,
                    "advogados_added": advogados_added_total,
                    "assuntos_filled_rules": assuntos_filled_rules_total,
                    "assuntos_filled_openai": assuntos_filled_openai_total,
                    "rows_still_empty_assuntos": _count_empty_assuntos(),
                    "url_tse_filled": _count_filled("noticia_TSE"),
                    "url_tre_filled": _count_filled("noticia_TRE"),
                    "url_gerais_filled_total": _count_filled_gerais(),
                    "restored_from_output_csv": restored_from_output_total,
                    "perplexity_api_calls": perplexity_metrics_total["perplexity_api_calls"],
                    "perplexity_cache_hits": perplexity_metrics_total["perplexity_cache_hits"],
                    "perplexity_no_match": perplexity_metrics_total["perplexity_no_match"],
                    "perplexity_retryable_errors": perplexity_metrics_total["perplexity_retryable_errors"],
                    "perplexity_skipped_existing": perplexity_metrics_total["perplexity_skipped_existing"],
                    "elapsed_seconds": round(max(0.0, time.time() - started_at), 2),
                    "completed_at": completed_at,
                    "batch_stats": extra or {},
                    "updated_at": utc_now_iso(),
                },
            )

        cp = read_json_dict(checkpoint_path)
        cp_sig = cp.get("source_signature", {})
        cp_rows = cp.get("processed_rows", [])
        if (
            int(cp.get("version", 0) or 0) == CHECKPOINT_VERSION
            and same_file_signature(cp_sig, source_sig)
            and isinstance(cp_rows, list)
            and len(cp_rows) == len(processed_rows)
        ):
            restored = 0
            for idx, saved_row in enumerate(cp_rows):
                if not isinstance(saved_row, dict):
                    continue
                for key in output_fields:
                    if key in saved_row:
                        processed_rows[idx][key] = str(saved_row.get(key, "") or "")
                restored += 1
            sanitize_official_news_columns(processed_rows)
            if restored:
                logger(f"[resume] checkpoint aplicado: {restored}/{len(processed_rows)} linhas.")
            cp_row_progress = cp.get("row_progress", {})
            if isinstance(cp_row_progress, dict):
                for idx_key, saved_progress in cp_row_progress.items():
                    if not isinstance(saved_progress, dict):
                        continue
                    try:
                        idx_value = int(str(idx_key))
                    except Exception:
                        continue
                    if idx_value < 0 or idx_value >= len(processed_rows):
                        continue
                    row_progress[str(idx_value)] = {
                        "stage": str(saved_progress.get("stage", "") or ""),
                        "request_key": str(saved_progress.get("request_key", "") or ""),
                        "news_status": str(saved_progress.get("news_status", "") or ""),
                    }
        elif output_path.exists() and not checkpoint_path.exists():
            preserved_stats = preserve_columns_from_reference_rows(
                target_rows=processed_rows,
                reference_csv=output_path,
                columns=DEFAULT_PRESERVE_COLUMNS,
            )
            restored_from_output_total = int(preserved_stats.get("applied_total", 0) or 0)
            if restored_from_output_total > 0:
                logger(
                    f"[resume] reaproveitado do CSV existente: {restored_from_output_total} "
                    f"celulas em {output_path.name}."
                )

        _refresh_lookup_payloads_from_rows()

        if local_tema_punchline_config.enabled:
            local_stats = preencher_rows_com_tema_punchline_local(processed_rows, lookup_payloads)
            if local_stats["tema_preenchidos"] or local_stats["punchline_preenchidos"]:
                logger(
                    "[Tema/local] preenchimento heuristico concluido | "
                    f"tema_preenchidos={local_stats['tema_preenchidos']} | "
                    f"punchline_preenchidos={local_stats['punchline_preenchidos']}"
                )
            _refresh_lookup_payloads_from_rows()

        _save_state("running", "prepared")

        if assuntos_enrichment_config.enabled:
            logger(f"[Assuntos/OpenAI] Iniciando enriquecimento para {len(processed_rows)} linhas...")
            logger(
                "[Assuntos/OpenAI] Config: "
                f"model={assuntos_enrichment_config.model}, workers={assuntos_enrichment_config.max_workers}, "
                f"batch={assuntos_enrichment_config.batch_size}, timeout={assuntos_enrichment_config.timeout_seconds}s, "
                f"delay={assuntos_enrichment_config.delay_between_batches}s, "
                f"max_itens={assuntos_enrichment_config.max_items}, "
                f"taxonomy_mode={assuntos_enrichment_config.taxonomy_mode}"
            )
            assuntos_filled_antes_openai = _count_filled("assuntos")
            asyncio.run(
                enriquecer_rows_com_assuntos_openai_async(
                    processed_rows,
                    logger,
                    assuntos_enrichment_config,
                    lookup_payloads=lookup_payloads,
                    on_batch_done=lambda s, e, t, stats: _save_state(
                        "running",
                        f"openai_assuntos_batch_{s + 1}_{e}_of_{t}",
                        stats,
                    ),
                )
            )
            assuntos_preenchidos_pos_openai = _count_filled("assuntos")
            assuntos_filled_openai_total = max(
                0,
                assuntos_preenchidos_pos_openai - assuntos_filled_antes_openai,
            )
            _refresh_lookup_payloads_from_rows()
            _save_state("running", "after_openai_assuntos")

        if tema_punchline_config.enabled:
            logger(f"[ChatGPT] Iniciando geracao de tema/punchline para {len(processed_rows)} linhas...")
            logger(
                "[ChatGPT] Config: "
                f"model={tema_punchline_config.model}, workers={tema_punchline_config.max_workers}, "
                f"batch={tema_punchline_config.batch_size}, timeout={tema_punchline_config.timeout_seconds}s, "
                f"delay={tema_punchline_config.delay_between_batches}s"
            )
            asyncio.run(
                enriquecer_rows_com_tema_punchline_async(
                    processed_rows,
                    logger,
                    tema_punchline_config,
                    lookup_payloads=lookup_payloads,
                    on_batch_done=lambda s, e, t, stats: _save_state(
                        "running",
                        f"openai_batch_{s + 1}_{e}_of_{t}",
                        stats,
                    ),
                )
            )
            _refresh_lookup_payloads_from_rows()
            _save_state("running", "after_openai")

        if web_lookup_config.enabled:
            provider_label = "Gemini" if web_lookup_config.provider == "gemini" else "Perplexity"
            estimate = estimate_news_api_calls(
                processed_rows,
                lookup_payloads=lookup_payloads,
                model=web_lookup_config.model,
                provider=web_lookup_config.provider,
                cache_path=cache_path,
            )
            logger(f"[{provider_label}] Iniciando busca de URLs para {len(processed_rows)} linhas...")
            if web_lookup_config.provider == "gemini":
                logger(
                    "[Gemini] Modo econômico: uma chamada grounded por caso pendente, com cache local e validação de confiança."
                )
            else:
                logger(
                    "[Perplexity] Politica balanceada: aceita confidence=high ou medium quando houver "
                    "matched_fields suficientes e validacao local de pagina noticiosa."
                )
            logger(
                f"[{provider_label}] Config: "
                f"model={web_lookup_config.model}, workers={web_lookup_config.max_workers}, "
                f"batch={web_lookup_config.batch_size}, timeout={web_lookup_config.timeout_seconds}s, "
                f"delay={web_lookup_config.delay_between_batches}s"
            )
            logger(
                f"[{provider_label}] Estimativa antes do run: "
                f"api={estimate['estimated_api_calls']} | "
                f"cache_terminal={estimate['estimated_cache_hits']} | "
                f"ja_preenchidas={estimate['estimated_skipped_existing']}"
            )
            perplexity_metrics_total = asyncio.run(
                enriquecer_rows_com_urls_async(
                    processed_rows,
                    logger,
                    web_lookup_config,
                    lookup_payloads=lookup_payloads,
                    on_batch_done=lambda s, e, t, stats: _save_state(
                        "running",
                        f"{web_lookup_config.provider}_batch_{s + 1}_{e}_of_{t}",
                        stats,
                    ),
                    cache_path=cache_path,
                    row_progress=row_progress,
                )
            )
            _save_state("running", f"after_{web_lookup_config.provider}")

        write_csv_atomic(output_path, output_fields, processed_rows)
        _save_state("completed", "final")
        cleanup_processing_artifacts(
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            logger=logger,
        )

        rows = len(processed_rows)

    logger(f"[Gerado] {output_path.name} | linhas={rows} | celulas_truncadas={truncated_cells}")
    return ProcessSummary(
        input_path=input_path,
        output_path=output_path,
        rows=rows,
        truncated_cells=truncated_cells,
        encoding=encoding,
        delimiter=delimiter,
        fields=output_fields,
    )


def compile_csvs(processed_paths: Sequence[Path], combined_output: Path, logger: Callable[[str], None]) -> int:
    if not processed_paths:
        raise ValueError("Nenhum arquivo processado para compilar.")

    ordered_headers: list[str] = []
    for path in processed_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            for header in headers:
                if header not in ordered_headers:
                    ordered_headers.append(header)

    total_rows = 0
    with combined_output.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=ordered_headers)
        writer.writeheader()

        for path in processed_paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    writer.writerow({header: row.get(header, "") for header in ordered_headers})
                    total_rows += 1

    logger(f"[Compilado] {combined_output.name} | linhas_totais={total_rows}")
    return total_rows


def parse_columns_arg(raw_value: str, fallback: Sequence[str]) -> list[str]:
    if not raw_value.strip():
        return [item for item in fallback]
    columns: list[str] = []
    for part in re.split(r"[;,]", raw_value):
        name = part.strip()
        if name and name not in columns:
            columns.append(name)
    return columns or [item for item in fallback]


def _normalize_match_digits(value: object) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def _build_row_match_candidates(row: dict[str, str]) -> list[str]:
    numero_unico = _normalize_match_digits(row.get("numeroUnico", ""))
    numero_processo = _normalize_match_digits(row.get("numeroProcesso", ""))
    candidates: list[str] = []
    if numero_unico:
        candidates.append(f"numeroUnico:{numero_unico}")
    if numero_processo:
        candidates.append(f"numeroProcesso:{numero_processo}")
    return candidates


def _apply_preserved_columns_to_rows(
    *,
    target_rows: list[dict[str, str]],
    reference_rows: Sequence[dict[str, str]],
    columns: Sequence[str],
) -> dict[str, object]:
    usable_columns = [col for col in columns if any(col in row for row in target_rows)]
    stats: dict[str, object] = {
        "rows_total": len(target_rows),
        "reference_rows_total": len(reference_rows),
        "usable_columns": usable_columns,
        "matched_by_key": 0,
        "fallback_by_index": len(reference_rows) == len(target_rows),
        "applied_total": 0,
        "applied_by_column": {col: 0 for col in usable_columns},
    }
    if not target_rows or not reference_rows or not usable_columns:
        return stats

    key_to_ref_indexes: dict[str, list[int]] = defaultdict(list)
    for ref_index, ref_row in enumerate(reference_rows):
        for key in _build_row_match_candidates(ref_row):
            key_to_ref_indexes[key].append(ref_index)

    matched_by_key = 0
    used_reference_indexes: set[int] = set()
    fallback_by_index = bool(stats["fallback_by_index"])
    applied_by_column = dict(stats["applied_by_column"])
    applied_total = 0

    for row_idx, target_row in enumerate(target_rows):
        sanitize_official_news_columns_in_row(target_row)
        reference_row: Optional[dict[str, str]] = None
        for key in _build_row_match_candidates(target_row):
            bucket = key_to_ref_indexes.get(key, [])
            while bucket and bucket[0] in used_reference_indexes:
                bucket.pop(0)
            if bucket:
                ref_index = bucket.pop(0)
                used_reference_indexes.add(ref_index)
                reference_row = dict(reference_rows[ref_index])
                matched_by_key += 1
                break
        if reference_row is None and fallback_by_index and row_idx < len(reference_rows):
            reference_row = dict(reference_rows[row_idx])
        if reference_row is None:
            continue
        for col in usable_columns:
            value = str(reference_row.get(col, "") or "").strip()
            normalizer = OFFICIAL_NEWS_COLUMN_NORMALIZERS.get(col)
            if normalizer is not None:
                value = normalizer(value) or ""
            if not value:
                continue
            if (target_row.get(col, "") or "").strip() == value:
                continue
            target_row[col] = value
            applied_by_column[col] = int(applied_by_column.get(col, 0) or 0) + 1
            applied_total += 1

    stats["matched_by_key"] = matched_by_key
    stats["applied_total"] = applied_total
    stats["applied_by_column"] = applied_by_column
    return stats


def preserve_columns_from_reference_rows(
    *,
    target_rows: list[dict[str, str]],
    reference_csv: Path,
    columns: Sequence[str],
) -> dict[str, object]:
    if not columns or not reference_csv.exists():
        return {
            "rows_total": len(target_rows),
            "reference_rows_total": 0,
            "usable_columns": [],
            "matched_by_key": 0,
            "fallback_by_index": False,
            "applied_total": 0,
            "applied_by_column": {},
        }
    ref_encoding, ref_delimiter = detect_csv_format(reference_csv)
    with reference_csv.open("r", encoding=ref_encoding, newline="") as handle:
        ref_reader = csv.DictReader(handle, delimiter=ref_delimiter)
        reference_rows = [dict(row) for row in ref_reader]
    return _apply_preserved_columns_to_rows(
        target_rows=target_rows,
        reference_rows=reference_rows,
        columns=columns,
    )


def preserve_columns_from_reference_csv(
    target_csv: Path,
    reference_csv: Path,
    columns: Sequence[str],
    logger: Callable[[str], None],
) -> int:
    if not columns:
        return 0

    if not reference_csv.exists():
        raise FileNotFoundError(f"Arquivo de referencia nao encontrado: {reference_csv}")

    with target_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        target_reader = csv.DictReader(handle)
        target_headers = target_reader.fieldnames or []
        target_rows = [dict(row) for row in target_reader]

    if not target_headers:
        raise ValueError(f"{target_csv.name}: sem cabecalho valido para preservacao.")

    stats = preserve_columns_from_reference_rows(
        target_rows=target_rows,
        reference_csv=reference_csv,
        columns=columns,
    )
    if not any(stats.get("usable_columns", [])):
        logger("[Preservacao] Nenhuma coluna de preservacao encontrada no CSV de destino.")
        return len(target_rows)
    if int(stats.get("reference_rows_total", 0) or 0) == 0:
        logger(f"[Preservacao] {reference_csv.name}: sem linhas para preservar.")
        return len(target_rows)

    write_csv_atomic(target_csv, target_headers, target_rows)
    usable_columns = [str(item) for item in stats.get("usable_columns", [])]
    matched_by_key = int(stats.get("matched_by_key", 0) or 0)
    fallback_by_index = bool(stats.get("fallback_by_index"))
    applied_total = int(stats.get("applied_total", 0) or 0)
    applied_by_column = {
        str(key): int(value or 0)
        for key, value in dict(stats.get("applied_by_column", {})).items()
    }
    logger(
        f"[Preservacao] {target_csv.name}: colunas={','.join(usable_columns)} | "
        f"linhas={len(target_rows)} | chaves={matched_by_key} | "
        f"fallback_index={'sim' if fallback_by_index else 'nao'} | "
        f"celulas_atualizadas={applied_total}"
    )
    for col in usable_columns:
        logger(f"[Preservacao] coluna {col}: {applied_by_column[col]} celulas atualizadas.")

    return len(target_rows)


def cleanup_processing_artifacts(
    *,
    output_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    logger: Callable[[str], None],
) -> None:
    removed: list[str] = []

    candidates = [
        output_path.with_suffix(output_path.suffix + ".tmp"),
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp"),
        report_path.with_suffix(report_path.suffix + ".tmp"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
                removed.append(candidate.name)
        except Exception:
            continue

    backup_pattern = f"{output_path.name}.startup_backup_*"
    for backup_path in resolve_backup_artifacts_dir().glob(backup_pattern):
        try:
            if backup_path.is_file():
                backup_path.unlink()
                removed.append(backup_path.name)
        except Exception:
            continue

    if removed:
        logger(
            f"[Limpeza] {output_path.name}: removidos {len(removed)} artefatos auxiliares "
            f"({', '.join(removed[:6])}{' ...' if len(removed) > 6 else ''})"
        )


def cleanup_global_notion_artifacts(out_dir: Path, logger: Callable[[str], None]) -> None:
    locations_and_patterns = (
        (out_dir, ("*_notion.csv.tmp", ".tmp_validacao*.csv", "*_limpeza_check.csv")),
        (resolve_backup_artifacts_dir(), ("*_notion.csv.startup_backup_*", "*_APIenriched.csv.apienriched_backup_*")),
    )
    removed: list[str] = []
    for location, patterns in locations_and_patterns:
        for pattern in patterns:
            for path in location.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                        removed.append(path.name)
                except Exception:
                    continue
    if removed:
        logger(
            f"[Limpeza] Remocao global de artefatos: {len(removed)} arquivos "
            f"({', '.join(removed[:6])}{' ...' if len(removed) > 6 else ''})"
        )


def infer_combined_name(input_paths: Sequence[Path], requested_name: str) -> str:
    cleaned = SPACE_RE.sub(" ", str(requested_name or "")).strip()
    if cleaned:
        return cleaned
    if len(input_paths) == 1:
        return f"{input_paths[0].stem}_consolidado.csv"
    return DEFAULT_COMBINED_MULTI


def cleanup_intermediate_processed_csvs(
    summaries: Sequence[ProcessSummary],
    *,
    combined_path: Path,
    logger: Callable[[str], None],
) -> None:
    _ = summaries
    _ = combined_path
    logger(
        "[Limpeza] CSVs intermediarios *_notion.csv preservados em Artefatos para retomada "
        "e enriquecimento posterior."
    )


def normalize_input_paths(file_paths: Iterable[str]) -> list[Path]:
    normalized: list[Path] = []
    seen: set[Path] = set()
    for item in file_paths:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {item}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Arquivo ignorado (nao e .csv): {path.name}")
        if path not in seen:
            normalized.append(path)
            seen.add(path)
    return normalized


def run_batch(
    files: Sequence[str],
    out_dir: str,
    max_texto_chars: int,
    combined_name: str,
    replace_newlines: bool,
    web_lookup_config: WebLookupConfig,
    local_tema_punchline_config: Optional[LocalTemaPunchlineConfig] = None,
    tema_punchline_config: Optional[TemaPunchlineConfig] = None,
    assuntos_enrichment_config: Optional[AssuntosEnrichmentConfig] = None,
    metadata_extraction_config: Optional[MetadataExtractionConfig] = None,
    logger: Optional[Callable[[str], None]] = None,
    preserve_from_csv: str = "",
    preserve_columns: Sequence[str] = (),
) -> tuple[list[ProcessSummary], Path, int]:
    local_tema_punchline_config = local_tema_punchline_config or LocalTemaPunchlineConfig(enabled=False)
    tema_punchline_config = tema_punchline_config or TemaPunchlineConfig(enabled=False, api_key="")
    assuntos_enrichment_config = assuntos_enrichment_config or AssuntosEnrichmentConfig(enabled=False, api_key="")
    metadata_extraction_config = metadata_extraction_config or MetadataExtractionConfig()
    logger = logger or (lambda _message: None)

    if not files:
        raise ValueError("Nenhum CSV informado para processamento.")

    if max_texto_chars < 0:
        raise ValueError("max_texto_chars nao pode ser negativo. Use 0 para sem truncamento.")
    web_lookup_config.provider = str(web_lookup_config.provider or "perplexity").strip().lower()
    if web_lookup_config.enabled and not web_lookup_config.api_key:
        if web_lookup_config.provider == "gemini":
            raise ValueError(
                "Busca de URLs ativada, mas sem API key do Gemini. "
                "Use --gemini-api-key, a variavel de ambiente GEMINI_API_KEY/GOOGLE_API_KEY "
                f"ou o arquivo {DEFAULT_GEMINI_KEY_FILE}."
            )
        raise ValueError(
            "Busca de URLs ativada, mas sem API key da Perplexity. "
            "Use --perplexity-api-key, a variavel de ambiente PERPLEXITY_API_KEY "
            f"ou o arquivo {DEFAULT_PERPLEXITY_KEY_FILE}."
        )
    openai_api_key = (tema_punchline_config.api_key or assuntos_enrichment_config.api_key).strip()
    if (tema_punchline_config.enabled or assuntos_enrichment_config.enabled) and not openai_api_key:
        raise ValueError(
            "Recursos OpenAI ativados (tema/punchline e/ou assuntos), mas sem API key da OpenAI. "
            "Use --openai-api-key, a variavel de ambiente OPENAI_API_KEY "
            f"ou o arquivo {DEFAULT_OPENAI_KEY_FILE}."
        )
    if tema_punchline_config.enabled and not tema_punchline_config.api_key:
        tema_punchline_config.api_key = openai_api_key
    if assuntos_enrichment_config.enabled and not assuntos_enrichment_config.api_key:
        assuntos_enrichment_config.api_key = openai_api_key

    input_paths = normalize_input_paths(files)
    output_dir = Path(out_dir).expanduser().resolve() if str(out_dir or "").strip() else SCRIPT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_output_dir = resolve_intermediate_csv_dir()

    chosen_combined_name = infer_combined_name(input_paths, combined_name)
    if not chosen_combined_name.lower().endswith(".csv"):
        chosen_combined_name = f"{chosen_combined_name}.csv"
    combined_path = output_dir / chosen_combined_name

    logger(
        "[Saida] CSVs por arquivo/etapa gravados em "
        f"{processed_output_dir}. Apenas o consolidado final sai em {output_dir}."
    )

    summaries: list[ProcessSummary] = []
    for input_path in input_paths:
        summary = process_one_csv(
            input_path=input_path,
            out_dir=processed_output_dir,
            max_texto_chars=max_texto_chars,
            replace_newlines=replace_newlines,
            web_lookup_config=web_lookup_config,
            local_tema_punchline_config=local_tema_punchline_config,
            tema_punchline_config=tema_punchline_config,
            assuntos_enrichment_config=assuntos_enrichment_config,
            metadata_extraction_config=metadata_extraction_config,
            logger=logger,
        )
        summaries.append(summary)

    compiled_rows = compile_csvs(
        processed_paths=[summary.output_path for summary in summaries],
        combined_output=combined_path,
        logger=logger,
    )

    preserve_source = preserve_from_csv.strip()
    if preserve_source:
        preserve_path = Path(preserve_source).expanduser()
        if not preserve_path.is_absolute():
            preserve_path = SCRIPT_DIR / preserve_path
        preserve_path = preserve_path.resolve()
        columns_to_preserve = list(preserve_columns) if preserve_columns else [*DEFAULT_PRESERVE_COLUMNS]
        compiled_rows = preserve_columns_from_reference_csv(
            target_csv=combined_path,
            reference_csv=preserve_path,
            columns=columns_to_preserve,
            logger=logger,
        )

    cleanup_intermediate_processed_csvs(summaries, combined_path=combined_path, logger=logger)
    cleanup_global_notion_artifacts(output_dir, logger)

    return summaries, combined_path, compiled_rows


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Tkinter nao esta disponivel neste Python. "
            "Instale/suporte tkinter ou use modo CLI com --no-gui."
        ) from exc

    class App:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("SJUR CSV Pipeline")
            self.root.geometry("1040x860")
            self.root.minsize(920, 760)

            self.file_vars: dict[str, tk.BooleanVar] = {}
            self.busy = False
            self.busy_widgets: list[tk.Widget] = []

            self.output_dir_var = tk.StringVar(value=str(SCRIPT_DIR))
            self.max_texto_chars_var = tk.StringVar(value="9000")
            self.combined_name_var = tk.StringVar(value="")
            self.replace_newlines_var = tk.BooleanVar(value=True)
            self.buscar_urls_var = tk.BooleanVar(value=False)
            self.gerar_tema_punchline_var = tk.BooleanVar(value=False)
            self.enriquecer_assuntos_var = tk.BooleanVar(value=False)

            self.enrich_out_dir_var = tk.StringVar(value="")
            self.enrich_suffix_var = tk.StringVar(value="_APIenriched")
            self.enrich_in_place_var = tk.BooleanVar(value=False)
            self.enrich_no_backup_var = tk.BooleanVar(value=False)
            self.enrich_fill_tema_punchline_var = tk.BooleanVar(value=True)
            self.enrich_fill_assuntos_var = tk.BooleanVar(value=False)
            self.enrich_fill_urls_var = tk.BooleanVar(value=True)

            self.assuntos_max_itens_var = tk.StringVar(value=str(DEFAULT_ASSUNTOS_MAX_ITEMS))
            self.assuntos_taxonomy_mode_var = tk.StringVar(value="mixed")
            self.openai_key_file_var = tk.StringVar(
                value=_resolve_existing_key_file_path(DEFAULT_OPENAI_KEY_FILE, OPENAI_KEY_FALLBACK_FILES)
            )
            self.openai_api_key_var = tk.StringVar(value=resolve_openai_api_key("", self.openai_key_file_var.get()))
            self.openai_model_var = tk.StringVar(value="gpt-5.1")
            self.openai_workers_var = tk.StringVar(value=str(OPENAI_DEFAULT_MAX_WORKERS))
            self.openai_timeout_var = tk.StringVar(value=str(OPENAI_DEFAULT_TIMEOUT))
            self.openai_batch_size_var = tk.StringVar(value=str(OPENAI_DEFAULT_BATCH_SIZE))
            self.openai_delay_var = tk.StringVar(value=str(OPENAI_DEFAULT_DELAY))
            self.openai_retries_var = tk.StringVar(value=str(OPENAI_DEFAULT_RETRIES))
            self.openai_target_rpm_var = tk.StringVar(value=str(OPENAI_DEFAULT_TARGET_RPM))
            self.perplexity_key_file_var = tk.StringVar(
                value=_resolve_existing_key_file_path(DEFAULT_PERPLEXITY_KEY_FILE, PERPLEXITY_KEY_FALLBACK_FILES)
            )
            self.perplexity_api_key_var = tk.StringVar(
                value=resolve_perplexity_api_key("", self.perplexity_key_file_var.get())
            )
            self.perplexity_model_var = tk.StringVar(value="sonar")
            self.perplexity_workers_var = tk.StringVar(value="4")
            self.perplexity_timeout_var = tk.StringVar(value="15")
            self.perplexity_batch_size_var = tk.StringVar(value="20")
            self.perplexity_delay_var = tk.StringVar(value="0.3")
            self.verbose_terminal_var = tk.BooleanVar(value=True)

            self.prep_resume_var = tk.StringVar(
                value="Resume: selecione arquivos para ver checkpoints e saidas reaproveitaveis."
            )
            self.enrich_resume_var = tk.StringVar(
                value="Resume: a aba de enriquecimento procura checkpoints e saidas existentes."
            )
            self.enrich_estimate_var = tk.StringVar(
                value="Estimativa Perplexity: selecione arquivos e habilite noticias para calcular."
            )

            self._build_ui()
            self._attach_hint_traces()

        def _register_busy_widget(self, widget: tk.Widget) -> tk.Widget:
            self.busy_widgets.append(widget)
            return widget

        def _set_busy(self, busy: bool) -> None:
            self.busy = busy
            state = "disabled" if busy else "normal"
            for widget in self.busy_widgets:
                try:
                    widget.configure(state=state)
                except Exception:
                    continue

        def _build_ui(self) -> None:
            main = ttk.Frame(self.root, padding=12)
            main.pack(fill="both", expand=True)

            files_box = ttk.LabelFrame(main, text="Arquivos", padding=8)
            files_box.pack(fill="x")

            file_controls = ttk.Frame(files_box)
            file_controls.pack(fill="x")
            self._register_busy_widget(
                ttk.Button(file_controls, text="Selecionar CSVs", command=self.add_files)
            ).pack(side="left", padx=(0, 8))
            self._register_busy_widget(
                ttk.Button(file_controls, text="Adicionar pasta", command=self.add_folder)
            ).pack(side="left", padx=(0, 8))
            self._register_busy_widget(
                ttk.Button(file_controls, text="Marcar todos", command=self.check_all)
            ).pack(side="left", padx=(0, 8))
            self._register_busy_widget(
                ttk.Button(file_controls, text="Desmarcar todos", command=self.uncheck_all)
            ).pack(side="left", padx=(0, 8))
            self._register_busy_widget(
                ttk.Button(file_controls, text="Limpar lista", command=self.clear_files)
            ).pack(side="left")

            self.canvas = tk.Canvas(files_box, height=180)
            scrollbar = ttk.Scrollbar(files_box, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=scrollbar.set)
            self.canvas.pack(side="left", fill="both", expand=True, pady=(8, 0))
            scrollbar.pack(side="right", fill="y", pady=(8, 0))

            self.files_frame = ttk.Frame(self.canvas)
            self.files_frame_window = self.canvas.create_window((0, 0), window=self.files_frame, anchor="nw")
            self.files_frame.bind(
                "<Configure>",
                lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
            )
            self.canvas.bind("<Configure>", self._on_files_canvas_configure)

            notebook = ttk.Notebook(main)
            notebook.pack(fill="x", expand=False, pady=(12, 0))

            prep_tab = ttk.Frame(notebook, padding=12)
            enrich_tab = ttk.Frame(notebook, padding=12)
            notebook.add(prep_tab, text="Preparacao")
            notebook.add(enrich_tab, text="Enriquecimento")

            self._build_preparation_tab(prep_tab)
            self._build_enrichment_tab(enrich_tab)

            advanced = ttk.LabelFrame(main, text="Avancado (APIs)", padding=10)
            advanced.pack(fill="x", pady=(12, 0))
            self._build_advanced_panel(advanced)

            log_box = ttk.LabelFrame(main, text="Log", padding=8)
            log_box.pack(fill="both", expand=True, pady=(12, 0))
            self.log_widget = tk.Text(log_box, height=14, wrap="word")
            self.log_widget.pack(fill="both", expand=True)
            self.log_widget.configure(state="disabled")

            self.refresh_file_list()

        def _build_preparation_tab(self, parent: ttk.Frame) -> None:
            parent.columnconfigure(1, weight=1)

            ttk.Label(parent, text="Pasta de saida do CSV final consolidado:").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Entry(parent, textvariable=self.output_dir_var, width=70).grid(
                row=0, column=1, sticky="ew", padx=8
            )
            self._register_busy_widget(
                ttk.Button(
                    parent,
                    text="Selecionar...",
                    command=lambda: self.choose_directory(
                        self.output_dir_var,
                        "Selecione a pasta de saida da preparacao",
                    ),
                )
            ).grid(row=0, column=2, sticky="e")

            ttk.Label(parent, text="Limite textoDecisao/textoEmenta (0 = sem truncamento):").grid(
                row=1, column=0, sticky="w", pady=(8, 0)
            )
            ttk.Entry(parent, textvariable=self.max_texto_chars_var, width=12).grid(
                row=1, column=1, sticky="w", padx=8, pady=(8, 0)
            )

            ttk.Label(parent, text="Nome do compilado (vazio = automatico):").grid(
                row=2, column=0, sticky="w", pady=(8, 0)
            )
            ttk.Entry(parent, textvariable=self.combined_name_var, width=28).grid(
                row=2, column=1, sticky="w", padx=8, pady=(8, 0)
            )

            ttk.Checkbutton(
                parent,
                text="Substituir quebras de linha e tabs por espaco",
                variable=self.replace_newlines_var,
            ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

            ttk.Label(
                parent,
                text="Padrao recomendado: gerar primeiro o consolidado final; intermediarios e artefatos vao para Artefatos.",
            ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

            ttk.Checkbutton(
                parent,
                text="Gerar tema/punchline com OpenAI durante a preparacao",
                variable=self.gerar_tema_punchline_var,
            ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

            ttk.Checkbutton(
                parent,
                text="Enriquecer assuntos vazios com OpenAI durante a preparacao",
                variable=self.enriquecer_assuntos_var,
            ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(2, 0))

            ttk.Checkbutton(
                parent,
                text="Buscar noticias com Perplexity durante a preparacao",
                variable=self.buscar_urls_var,
            ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(2, 0))

            prep_status = ttk.LabelFrame(parent, text="Resume detectado", padding=8)
            prep_status.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))
            prep_status.columnconfigure(0, weight=1)
            ttk.Label(
                prep_status,
                textvariable=self.prep_resume_var,
                justify="left",
                wraplength=920,
            ).grid(row=0, column=0, sticky="w")

            self._register_busy_widget(
                ttk.Button(parent, text="Rodar preparacao", command=self.process_preparation)
            ).grid(row=9, column=0, sticky="w", pady=(12, 0))

        def _build_enrichment_tab(self, parent: ttk.Frame) -> None:
            parent.columnconfigure(1, weight=1)

            ttk.Label(parent, text="Pasta de saida (vazio = pasta do CSV resolvido):").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Entry(parent, textvariable=self.enrich_out_dir_var, width=70).grid(
                row=0, column=1, sticky="ew", padx=8
            )
            self._register_busy_widget(
                ttk.Button(
                    parent,
                    text="Selecionar...",
                    command=lambda: self.choose_directory(
                        self.enrich_out_dir_var,
                        "Selecione a pasta de saida do enriquecimento",
                    ),
                )
            ).grid(row=0, column=2, sticky="e")

            ttk.Label(parent, text="Sufixo do arquivo final:").grid(row=1, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(parent, textvariable=self.enrich_suffix_var, width=18).grid(
                row=1, column=1, sticky="w", padx=8, pady=(8, 0)
            )

            ttk.Checkbutton(
                parent,
                text="Sobrescrever o proprio CSV resolvido (--in-place)",
                variable=self.enrich_in_place_var,
            ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

            ttk.Checkbutton(
                parent,
                text="Nao criar backup quando o arquivo de saida ja existir",
                variable=self.enrich_no_backup_var,
            ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

            ttk.Label(
                parent,
                text=(
                    "Se o item selecionado nao for *_notion.csv, a aba tenta usar o correspondente "
                    "gerado na pasta da Preparacao."
                ),
            ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

            ttk.Checkbutton(
                parent,
                text="Preencher tema/punchline vazios com OpenAI",
                variable=self.enrich_fill_tema_punchline_var,
            ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

            ttk.Checkbutton(
                parent,
                text="Preencher assuntos vazios com OpenAI",
                variable=self.enrich_fill_assuntos_var,
            ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(2, 0))

            ttk.Checkbutton(
                parent,
                text="Preencher noticias vazias com Perplexity",
                variable=self.enrich_fill_urls_var,
            ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(2, 0))

            enrich_status = ttk.LabelFrame(parent, text="Resume detectado", padding=8)
            enrich_status.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))
            enrich_status.columnconfigure(0, weight=1)
            ttk.Label(
                enrich_status,
                textvariable=self.enrich_resume_var,
                justify="left",
                wraplength=920,
            ).grid(row=0, column=0, sticky="w")

            estimate_box = ttk.LabelFrame(parent, text="Estimativa Perplexity", padding=8)
            estimate_box.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(10, 0))
            estimate_box.columnconfigure(0, weight=1)
            ttk.Label(
                estimate_box,
                textvariable=self.enrich_estimate_var,
                justify="left",
                wraplength=920,
            ).grid(row=0, column=0, sticky="w")

            self._register_busy_widget(
                ttk.Button(parent, text="Rodar enriquecimento", command=self.process_enrichment)
            ).grid(row=10, column=0, sticky="w", pady=(12, 0))

        def _build_advanced_panel(self, parent: ttk.LabelFrame) -> None:
            parent.columnconfigure(1, weight=1)

            ttk.Label(parent, text="OpenAI API key:").grid(row=0, column=0, sticky="w")
            ttk.Entry(parent, textvariable=self.openai_api_key_var, width=62, show="*").grid(
                row=0, column=1, sticky="ew", padx=8
            )

            ttk.Label(parent, text="Arquivo da chave OpenAI:").grid(row=1, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(parent, textvariable=self.openai_key_file_var, width=62).grid(
                row=1, column=1, sticky="ew", padx=8, pady=(8, 0)
            )

            openai_frame = ttk.Frame(parent)
            openai_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
            ttk.Label(openai_frame, text="Modelo OpenAI").grid(row=0, column=0, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_model_var, width=12).grid(
                row=0, column=1, padx=(4, 14)
            )
            ttk.Label(openai_frame, text="Workers").grid(row=0, column=2, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_workers_var, width=5).grid(
                row=0, column=3, padx=(4, 14)
            )
            ttk.Label(openai_frame, text="Timeout(s)").grid(row=0, column=4, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_timeout_var, width=6).grid(
                row=0, column=5, padx=(4, 14)
            )
            ttk.Label(openai_frame, text="Batch").grid(row=0, column=6, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_batch_size_var, width=6).grid(
                row=0, column=7, padx=(4, 14)
            )
            ttk.Label(openai_frame, text="Delay(s)").grid(row=0, column=8, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_delay_var, width=6).grid(
                row=0, column=9, padx=(4, 14)
            )
            ttk.Label(openai_frame, text="Retries").grid(row=1, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(openai_frame, textvariable=self.openai_retries_var, width=6).grid(
                row=1, column=1, padx=(4, 14), pady=(6, 0)
            )
            ttk.Label(openai_frame, text="Target RPM").grid(row=1, column=2, sticky="w", pady=(6, 0))
            ttk.Entry(openai_frame, textvariable=self.openai_target_rpm_var, width=8).grid(
                row=1, column=3, padx=(4, 14), pady=(6, 0)
            )
            ttk.Label(openai_frame, text="Assuntos max/linha").grid(row=1, column=4, sticky="w", pady=(6, 0))
            ttk.Entry(openai_frame, textvariable=self.assuntos_max_itens_var, width=6).grid(
                row=1, column=5, padx=(4, 14), pady=(6, 0)
            )
            ttk.Label(openai_frame, text="Taxonomia").grid(row=1, column=6, sticky="w", pady=(6, 0))
            ttk.Combobox(
                openai_frame,
                textvariable=self.assuntos_taxonomy_mode_var,
                values=list(ASSUNTOS_TAXONOMY_CHOICES),
                state="readonly",
                width=12,
            ).grid(row=1, column=7, padx=(4, 0), pady=(6, 0))

            ttk.Label(parent, text="Perplexity API key:").grid(row=3, column=0, sticky="w", pady=(10, 0))
            ttk.Entry(parent, textvariable=self.perplexity_api_key_var, width=62, show="*").grid(
                row=3, column=1, sticky="ew", padx=8, pady=(10, 0)
            )

            ttk.Label(parent, text="Arquivo da chave Perplexity:").grid(row=4, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(parent, textvariable=self.perplexity_key_file_var, width=62).grid(
                row=4, column=1, sticky="ew", padx=8, pady=(8, 0)
            )

            perf_frame = ttk.Frame(parent)
            perf_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
            ttk.Label(perf_frame, text="Modelo Perplexity").grid(row=0, column=0, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_model_var, width=10).grid(
                row=0, column=1, padx=(4, 14)
            )
            ttk.Label(perf_frame, text="Workers").grid(row=0, column=2, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_workers_var, width=5).grid(
                row=0, column=3, padx=(4, 14)
            )
            ttk.Label(perf_frame, text="Timeout(s)").grid(row=0, column=4, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_timeout_var, width=6).grid(
                row=0, column=5, padx=(4, 14)
            )
            ttk.Label(perf_frame, text="Batch").grid(row=0, column=6, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_batch_size_var, width=6).grid(
                row=0, column=7, padx=(4, 14)
            )
            ttk.Label(perf_frame, text="Delay(s)").grid(row=0, column=8, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_delay_var, width=6).grid(
                row=0, column=9, padx=(4, 14)
            )
            ttk.Checkbutton(
                perf_frame,
                text="Verbose no terminal",
                variable=self.verbose_terminal_var,
            ).grid(row=0, column=10, sticky="w")

        def _attach_hint_traces(self) -> None:
            tracked = [
                self.output_dir_var,
                self.buscar_urls_var,
                self.gerar_tema_punchline_var,
                self.enriquecer_assuntos_var,
                self.perplexity_model_var,
                self.enrich_out_dir_var,
                self.enrich_suffix_var,
                self.enrich_in_place_var,
                self.enrich_fill_urls_var,
            ]
            for variable in tracked:
                variable.trace_add("write", lambda *_args: self.root.after_idle(self.refresh_runtime_hints))

        def _on_files_canvas_configure(self, event: tk.Event) -> None:
            self.canvas.itemconfigure(self.files_frame_window, width=event.width)

        def log(self, message: str) -> None:
            def append() -> None:
                self.log_widget.configure(state="normal")
                self.log_widget.insert("end", f"{message}\n")
                self.log_widget.see("end")
                self.log_widget.configure(state="disabled")
                self.root.update_idletasks()

            if self.verbose_terminal_var.get():
                print(message, flush=True)
            if threading.current_thread() is threading.main_thread():
                append()
            else:
                self.root.after(0, append)

        def add_files(self) -> None:
            raw_selection = filedialog.askopenfilenames(
                title="Selecione os arquivos CSV",
                filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
            )
            if not raw_selection:
                return
            try:
                file_paths = list(self.root.tk.splitlist(raw_selection))
            except Exception:
                file_paths = [str(item) for item in raw_selection] if isinstance(raw_selection, (list, tuple)) else []
            for raw_path in file_paths:
                file_path = str(Path(raw_path).expanduser().resolve())
                if file_path not in self.file_vars:
                    self.file_vars[file_path] = tk.BooleanVar(value=True)
                    self.file_vars[file_path].trace_add(
                        "write",
                        lambda *_args: self.root.after_idle(self.refresh_runtime_hints),
                    )
            self.refresh_file_list()

        def add_folder(self) -> None:
            folder = filedialog.askdirectory(title="Selecione a pasta com CSVs")
            if not folder:
                return
            for file_path in sorted(Path(folder).expanduser().rglob("*.csv")):
                file_str = str(file_path.resolve())
                if file_str not in self.file_vars:
                    self.file_vars[file_str] = tk.BooleanVar(value=True)
                    self.file_vars[file_str].trace_add(
                        "write",
                        lambda *_args: self.root.after_idle(self.refresh_runtime_hints),
                    )
            self.refresh_file_list()

        def clear_files(self) -> None:
            self.file_vars.clear()
            self.refresh_file_list()

        def check_all(self) -> None:
            for variable in self.file_vars.values():
                variable.set(True)
            self.refresh_runtime_hints()

        def uncheck_all(self) -> None:
            for variable in self.file_vars.values():
                variable.set(False)
            self.refresh_runtime_hints()

        def choose_directory(self, variable: tk.StringVar, title: str) -> None:
            selected_dir = filedialog.askdirectory(title=title)
            if selected_dir:
                variable.set(selected_dir)
                self.refresh_runtime_hints()

        def refresh_file_list(self) -> None:
            for child in self.files_frame.winfo_children():
                child.destroy()

            if not self.file_vars:
                ttk.Label(self.files_frame, text="Nenhum CSV selecionado.").grid(
                    row=0, column=0, sticky="w", padx=4, pady=2
                )

            for row_index, file_path in enumerate(sorted(self.file_vars)):
                ttk.Checkbutton(
                    self.files_frame,
                    text=Path(file_path).name,
                    variable=self.file_vars[file_path],
                ).grid(row=row_index, column=0, sticky="w", padx=4, pady=2)

            self.files_frame.update_idletasks()
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)
            self.refresh_runtime_hints()

        def _selected_files(self) -> list[str]:
            return [path for path, variable in sorted(self.file_vars.items()) if variable.get()]

        def _selected_paths(self) -> list[Path]:
            return [Path(path) for path in self._selected_files()]

        def _resolve_preparation_output_dir(self) -> Path:
            raw = self.output_dir_var.get().strip()
            return Path(raw).expanduser().resolve() if raw else SCRIPT_DIR.resolve()

        def _resolve_prep_output_path(self, input_path: Path) -> Path:
            return resolve_intermediate_csv_dir() / f"{input_path.stem}_notion.csv"

        def _resolve_prep_artifacts(self, input_path: Path) -> tuple[Path, Path, Path, Path]:
            output_path = self._resolve_prep_output_path(input_path)
            checkpoint_path = resolve_checkpoint_artifact_path(output_path)
            report_path = resolve_report_artifact_path(output_path)
            cache_path = resolve_web_lookup_cache_path(output_path.parent)
            return output_path, checkpoint_path, report_path, cache_path

        def _resolve_enrichment_input(self, selected_path: Path) -> Path:
            if selected_path.name.lower().endswith("_notion.csv"):
                return selected_path
            candidate = self._resolve_prep_output_path(selected_path)
            return candidate if candidate.exists() else selected_path

        def _resolve_enrich_output_path(self, selected_path: Path) -> Path:
            input_path = self._resolve_enrichment_input(selected_path)
            if self.enrich_in_place_var.get():
                return input_path
            base_dir = (
                Path(self.enrich_out_dir_var.get().strip()).expanduser().resolve()
                if self.enrich_out_dir_var.get().strip()
                else input_path.parent
            )
            suffix = self.enrich_suffix_var.get().strip() or "_APIenriched"
            return base_dir / f"{input_path.stem}{suffix}{input_path.suffix}"

        def _resolve_enrich_artifacts(self, selected_path: Path) -> tuple[Path, Path, Path, Path, Path]:
            input_path = self._resolve_enrichment_input(selected_path)
            output_path = self._resolve_enrich_output_path(selected_path)
            checkpoint_path = resolve_checkpoint_artifact_path(output_path)
            report_path = resolve_report_artifact_path(output_path)
            cache_path = resolve_web_lookup_cache_path(output_path.parent)
            return input_path, output_path, checkpoint_path, report_path, cache_path

        def _load_rows_from_csv(self, path: Path) -> list[dict[str, str]]:
            if not path.exists():
                return []
            encoding, delimiter = detect_csv_format(path)
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                rows: list[dict[str, str]] = []
                for raw in reader:
                    if raw is None:
                        continue
                    rows.append({str(key or ""): str(value or "") for key, value in raw.items()})
            return rows

        def _restore_rows_for_estimate(self, input_path: Path, output_path: Path, checkpoint_path: Path) -> list[dict[str, str]]:
            checkpoint = read_json_dict(checkpoint_path)
            cp_rows = checkpoint.get("processed_rows", [])
            if isinstance(cp_rows, list) and cp_rows and all(isinstance(row, dict) for row in cp_rows):
                return [{str(key or ""): str(value or "") for key, value in row.items()} for row in cp_rows]

            rows = self._load_rows_from_csv(input_path)
            if rows and output_path.exists() and output_path.resolve() != input_path.resolve():
                preserve_columns_from_reference_rows(
                    target_rows=rows,
                    reference_csv=output_path,
                    columns=[*DEFAULT_PRESERVE_COLUMNS, "assuntos"],
                )
            return rows

        def _summarize_row_progress(self, checkpoint: dict[str, object], report: dict[str, object]) -> str:
            row_progress = checkpoint.get("row_progress", {})
            if not isinstance(row_progress, dict) or not row_progress:
                return ""
            rows_total_raw = report.get("rows_total", checkpoint.get("rows_total", 0))
            try:
                rows_total = int(rows_total_raw or 0)
            except Exception:
                rows_total = 0
            statuses = defaultdict(int)
            for item in row_progress.values():
                if isinstance(item, dict):
                    statuses[str(item.get("news_status", "") or "unknown")] += 1
            parts: list[str] = []
            if statuses.get("pending", 0):
                parts.append(f"pending={statuses['pending']}")
            if statuses.get("error_retryable", 0):
                parts.append(f"retryable={statuses['error_retryable']}")
            if statuses.get("no_match", 0):
                parts.append(f"no_match={statuses['no_match']}")
            if statuses.get("filled", 0):
                parts.append(f"filled={statuses['filled']}")
            if statuses.get("skipped_existing", 0):
                parts.append(f"skip_existing={statuses['skipped_existing']}")
            if rows_total > len(row_progress):
                parts.append(f"sem_progresso={rows_total - len(row_progress)}")
            return ", ".join(parts)

        def _describe_resume(self, file_label: str, output_path: Path, checkpoint_path: Path, report_path: Path) -> str:
            checkpoint = read_json_dict(checkpoint_path)
            report = read_json_dict(report_path)
            if checkpoint or report:
                status = str(checkpoint.get("status", report.get("status", "")) or "-")
                stage = str(checkpoint.get("stage", report.get("stage", "")) or "-")
                progress = self._summarize_row_progress(checkpoint, report)
                suffix = f" | {progress}" if progress else ""
                return f"{file_label}: status={status} | stage={stage}{suffix}"
            if output_path.exists():
                return f"{file_label}: saida existente em {output_path.name} (sem checkpoint)."
            return ""

        def _format_resume_lines(self, lines: list[str], *, empty_message: str) -> str:
            cleaned = [line for line in lines if line]
            if not cleaned:
                return empty_message
            limit = 3
            if len(cleaned) > limit:
                extra = len(cleaned) - limit
                cleaned = cleaned[:limit] + [f"+{extra} arquivo(s) adicional(is) com resume ou saida existente."]
            return "\n".join(cleaned)

        def refresh_runtime_hints(self) -> None:
            selected_paths = self._selected_paths()
            if not selected_paths:
                self.prep_resume_var.set(
                    "Resume: selecione arquivos para ver checkpoints e saidas reaproveitaveis."
                )
                self.enrich_resume_var.set(
                    "Resume: a aba de enriquecimento procura checkpoints e saidas existentes."
                )
                self.enrich_estimate_var.set(
                    "Estimativa Perplexity: selecione arquivos e habilite noticias para calcular."
                )
                return

            prep_lines: list[str] = []
            for path in selected_paths:
                output_path, checkpoint_path, report_path, _cache_path = self._resolve_prep_artifacts(path)
                prep_lines.append(self._describe_resume(output_path.name, output_path, checkpoint_path, report_path))
            self.prep_resume_var.set(
                self._format_resume_lines(
                    prep_lines,
                    empty_message="Nenhum checkpoint da preparacao foi encontrado para os arquivos selecionados.",
                )
            )

            enrich_lines: list[str] = []
            estimated_api_calls = 0
            estimated_cache_hits = 0
            estimated_skipped_existing = 0
            estimate_errors: list[str] = []
            for path in selected_paths:
                input_path, output_path, checkpoint_path, report_path, cache_path = self._resolve_enrich_artifacts(path)
                enrich_lines.append(self._describe_resume(output_path.name, output_path, checkpoint_path, report_path))
                if not self.enrich_fill_urls_var.get():
                    continue
                try:
                    rows = self._restore_rows_for_estimate(input_path, output_path, checkpoint_path)
                    lookup_payloads = [build_lookup_payload(row) for row in rows]
                    estimate = estimate_news_api_calls(
                        rows,
                        lookup_payloads=lookup_payloads,
                        model=self.perplexity_model_var.get().strip() or "sonar",
                        cache_path=cache_path,
                    )
                    estimated_api_calls += int(estimate.get("estimated_api_calls", 0) or 0)
                    estimated_cache_hits += int(estimate.get("estimated_cache_hits", 0) or 0)
                    estimated_skipped_existing += int(estimate.get("estimated_skipped_existing", 0) or 0)
                except Exception as exc:
                    estimate_errors.append(f"{input_path.name}: {exc}")

            self.enrich_resume_var.set(
                self._format_resume_lines(
                    enrich_lines,
                    empty_message="Nenhum checkpoint do enriquecimento foi encontrado para os arquivos selecionados.",
                )
            )

            if not self.enrich_fill_urls_var.get():
                self.enrich_estimate_var.set(
                    "Noticias desabilitadas nesta aba. Estimativa Perplexity = 0 chamadas."
                )
            else:
                summary = (
                    f"Chamadas previstas={estimated_api_calls} | "
                    f"cache_terminal={estimated_cache_hits} | "
                    f"ja_preenchidas={estimated_skipped_existing}"
                )
                if estimate_errors:
                    summary += f"\nAvisos: {' | '.join(estimate_errors[:2])}"
                    if len(estimate_errors) > 2:
                        summary += f" | +{len(estimate_errors) - 2} arquivo(s)"
                self.enrich_estimate_var.set(summary)

        def _parse_int(self, value: str, label: str, *, allow_zero: bool = False) -> int:
            try:
                parsed = int(str(value).strip())
            except ValueError as exc:
                raise ValueError(f"{label} deve ser um numero inteiro.") from exc
            min_value = 0 if allow_zero else 1
            if parsed < min_value:
                suffix = "nao pode ser negativo." if allow_zero else "deve ser maior que zero."
                raise ValueError(f"{label} {suffix}")
            return parsed

        def _parse_float(self, value: str, label: str) -> float:
            try:
                parsed = float(str(value).strip())
            except ValueError as exc:
                raise ValueError(f"{label} deve ser numerico.") from exc
            if parsed < 0:
                raise ValueError(f"{label} nao pode ser negativo.")
            return parsed

        def _build_shared_settings(self) -> dict[str, object]:
            assuntos_max_itens = self._parse_int(
                self.assuntos_max_itens_var.get(),
                "Assuntos: max itens por linha",
            )
            assuntos_taxonomy_mode = (self.assuntos_taxonomy_mode_var.get().strip() or "mixed").lower()
            if assuntos_taxonomy_mode not in ASSUNTOS_TAXONOMY_CHOICES:
                raise ValueError(
                    f"Assuntos: taxonomia invalida. Use {', '.join(ASSUNTOS_TAXONOMY_CHOICES)}."
                )
            return {
                "openai_workers": self._parse_int(self.openai_workers_var.get(), "OpenAI: workers"),
                "openai_timeout": self._parse_int(self.openai_timeout_var.get(), "OpenAI: timeout"),
                "openai_batch_size": self._parse_int(self.openai_batch_size_var.get(), "OpenAI: batch"),
                "openai_delay": self._parse_float(self.openai_delay_var.get(), "OpenAI: delay"),
                "openai_retries": self._parse_int(self.openai_retries_var.get(), "OpenAI: retries"),
                "openai_target_rpm": self._parse_int(
                    self.openai_target_rpm_var.get(),
                    "OpenAI: target RPM",
                    allow_zero=True,
                ),
                "assuntos_max_itens": assuntos_max_itens,
                "assuntos_taxonomy_mode": assuntos_taxonomy_mode,
                "perplexity_workers": self._parse_int(self.perplexity_workers_var.get(), "Perplexity: workers"),
                "perplexity_timeout": self._parse_int(self.perplexity_timeout_var.get(), "Perplexity: timeout"),
                "perplexity_batch_size": self._parse_int(
                    self.perplexity_batch_size_var.get(),
                    "Perplexity: batch",
                ),
                "perplexity_delay": self._parse_float(self.perplexity_delay_var.get(), "Perplexity: delay"),
                "resolved_openai_api_key": resolve_openai_api_key(
                    self.openai_api_key_var.get().strip(),
                    self.openai_key_file_var.get().strip(),
                ),
                "resolved_perplexity_api_key": resolve_perplexity_api_key(
                    self.perplexity_api_key_var.get().strip(),
                    self.perplexity_key_file_var.get().strip(),
                ),
            }

        def _run_background(
            self,
            *,
            label: str,
            worker: Callable[[], object],
            on_success: Callable[[object], None],
        ) -> None:
            if self.busy:
                messagebox.showwarning("Processamento em andamento", "Aguarde a operacao atual terminar.")
                return

            self._set_busy(True)
            self.log(f"[GUI] Iniciando {label}...")

            def _target() -> None:
                try:
                    result = worker()
                except Exception as exc:  # pylint: disable=broad-except
                    self.root.after(0, lambda: self._finish_background_error(label, exc))
                    return
                self.root.after(0, lambda: self._finish_background_success(label, result, on_success))

            threading.Thread(target=_target, daemon=True).start()

        def _finish_background_error(self, label: str, exc: Exception) -> None:
            self._set_busy(False)
            self.log(f"[GUI] {label} falhou: {exc}")
            self.refresh_runtime_hints()
            messagebox.showerror(f"Erro em {label}", str(exc))

        def _finish_background_success(
            self,
            label: str,
            result: object,
            on_success: Callable[[object], None],
        ) -> None:
            self._set_busy(False)
            self.log(f"[GUI] {label} concluido.")
            self.refresh_runtime_hints()
            on_success(result)

        def process_preparation(self) -> None:
            selected_files = self._selected_files()
            if not selected_files:
                messagebox.showwarning("Aviso", "Selecione ao menos um arquivo CSV.")
                return
            try:
                max_texto_chars = self._parse_int(
                    self.max_texto_chars_var.get(),
                    "Limite textoDecisao/textoEmenta",
                    allow_zero=True,
                )
                shared = self._build_shared_settings()
            except ValueError as exc:
                messagebox.showerror("Erro", str(exc))
                return

            tema_punchline_config = TemaPunchlineConfig(
                enabled=self.gerar_tema_punchline_var.get(),
                api_key=str(shared["resolved_openai_api_key"]),
                model=self.openai_model_var.get().strip() or "gpt-5.1",
                timeout_seconds=int(shared["openai_timeout"]),
                max_workers=int(shared["openai_workers"]),
                batch_size=int(shared["openai_batch_size"]),
                delay_between_batches=float(shared["openai_delay"]),
                retries=int(shared["openai_retries"]),
                target_rpm=int(shared["openai_target_rpm"]),
            )
            assuntos_enrichment_config = AssuntosEnrichmentConfig(
                enabled=self.enriquecer_assuntos_var.get(),
                api_key=str(shared["resolved_openai_api_key"]),
                model=self.openai_model_var.get().strip() or "gpt-5.1",
                timeout_seconds=int(shared["openai_timeout"]),
                max_workers=int(shared["openai_workers"]),
                batch_size=int(shared["openai_batch_size"]),
                delay_between_batches=float(shared["openai_delay"]),
                retries=int(shared["openai_retries"]),
                target_rpm=int(shared["openai_target_rpm"]),
                max_items=int(shared["assuntos_max_itens"]),
                taxonomy_mode=str(shared["assuntos_taxonomy_mode"]),
            )
            web_lookup_config = WebLookupConfig(
                enabled=self.buscar_urls_var.get(),
                api_key=str(shared["resolved_perplexity_api_key"]),
                model=self.perplexity_model_var.get().strip() or "sonar",
                timeout_seconds=int(shared["perplexity_timeout"]),
                max_workers=int(shared["perplexity_workers"]),
                batch_size=int(shared["perplexity_batch_size"]),
                delay_between_batches=float(shared["perplexity_delay"]),
            )
            metadata_extraction_config = MetadataExtractionConfig(
                include_institutional_entities=True,
                header_max_chars=DEFAULT_METADATA_HEADER_MAX_CHARS,
            )
            local_tema_punchline_config = LocalTemaPunchlineConfig(enabled=False)

            def worker() -> object:
                return run_batch(
                    files=selected_files,
                    out_dir=self.output_dir_var.get().strip(),
                    max_texto_chars=max_texto_chars,
                    combined_name=self.combined_name_var.get().strip(),
                    replace_newlines=self.replace_newlines_var.get(),
                    web_lookup_config=web_lookup_config,
                    local_tema_punchline_config=local_tema_punchline_config,
                    tema_punchline_config=tema_punchline_config,
                    assuntos_enrichment_config=assuntos_enrichment_config,
                    metadata_extraction_config=metadata_extraction_config,
                    logger=self.log,
                )

            def on_success(result: object) -> None:
                summaries, combined_path, compiled_rows = result  # type: ignore[misc]
                total_rows = sum(item.rows for item in summaries)
                total_trunc = sum(item.truncated_cells for item in summaries)
                messagebox.showinfo(
                    "Preparacao concluida",
                    (
                        f"Arquivos processados: {len(summaries)}\n"
                        f"Linhas processadas: {total_rows}\n"
                        f"Celulas truncadas: {total_trunc}\n"
                        f"Linhas no compilado: {compiled_rows}\n"
                        f"Compilado: {combined_path}"
                    ),
                )

            self._run_background(label="preparacao", worker=worker, on_success=on_success)

        def process_enrichment(self) -> None:
            import SJUR_csv_to_csv_APIenriching as api_enrich

            selected_paths = self._selected_paths()
            if not selected_paths:
                messagebox.showwarning("Aviso", "Selecione ao menos um arquivo CSV.")
                return

            try:
                shared = self._build_shared_settings()
            except ValueError as exc:
                messagebox.showerror("Erro", str(exc))
                return

            resolved_files = [str(self._resolve_enrichment_input(path)) for path in selected_paths]
            args = argparse.Namespace(
                files=resolved_files,
                out_dir=self.enrich_out_dir_var.get().strip(),
                suffix=self.enrich_suffix_var.get().strip() or "_APIenriched",
                in_place=self.enrich_in_place_var.get(),
                no_backup=self.enrich_no_backup_var.get(),
                skip_openai_tema_punchline=not self.enrich_fill_tema_punchline_var.get(),
                enriquecer_assuntos_openai=self.enrich_fill_assuntos_var.get(),
                skip_perplexity_urls=not self.enrich_fill_urls_var.get(),
                openai_api_key=self.openai_api_key_var.get().strip(),
                openai_key_file=self.openai_key_file_var.get().strip() or DEFAULT_OPENAI_KEY_FILE,
                openai_model=self.openai_model_var.get().strip() or "gpt-5.1",
                openai_max_workers=int(shared["openai_workers"]),
                openai_timeout=int(shared["openai_timeout"]),
                openai_batch_size=int(shared["openai_batch_size"]),
                openai_delay=float(shared["openai_delay"]),
                openai_retries=int(shared["openai_retries"]),
                openai_target_rpm=int(shared["openai_target_rpm"]),
                assuntos_max_itens=int(shared["assuntos_max_itens"]),
                assuntos_taxonomy_mode=str(shared["assuntos_taxonomy_mode"]),
                perplexity_api_key=self.perplexity_api_key_var.get().strip(),
                perplexity_key_file=self.perplexity_key_file_var.get().strip() or DEFAULT_PERPLEXITY_KEY_FILE,
                perplexity_model=self.perplexity_model_var.get().strip() or "sonar",
                perplexity_max_workers=int(shared["perplexity_workers"]),
                perplexity_timeout=int(shared["perplexity_timeout"]),
                perplexity_batch_size=int(shared["perplexity_batch_size"]),
                perplexity_delay=float(shared["perplexity_delay"]),
                verbose=self.verbose_terminal_var.get(),
                quiet=False,
                debug=False,
                log_file="",
                no_gui=True,
            )

            def worker() -> object:
                return api_enrich.run_enrichment(args, logger=self.log)

            def on_success(result: object) -> None:
                results = result  # type: ignore[assignment]
                summary_lines = api_enrich._summarize_results(results)  # pylint: disable=protected-access
                messagebox.showinfo("Enriquecimento concluido", "\n".join(summary_lines))

            self._run_background(label="enriquecimento", worker=worker, on_success=on_success)

    root = tk.Tk()
    App(root)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Converte CSVs para versao notion-friendly e gera compilado final."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Arquivos CSV para processar (se informado, roda em modo CLI).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Pasta de saida do CSV final consolidado. Intermediarios e artefatos vao para Artefatos/.",
    )
    parser.add_argument(
        "--max-texto-chars",
        type=int,
        default=9000,
        help="Limite de caracteres para textoDecisao e textoEmenta (0 = sem truncamento).",
    )
    parser.add_argument(
        "--max-ementa-chars",
        dest="max_texto_chars",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-chars",
        dest="max_texto_chars",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--combined-name",
        default="",
        help=(
            "Nome do arquivo compilado final. Se vazio: 1 arquivo => <entrada>_consolidado.csv; "
            f"varios arquivos => {DEFAULT_COMBINED_MULTI}."
        ),
    )
    parser.add_argument(
        "--keep-newlines",
        action="store_true",
        help="Preserva quebras de linha e tabs nas celulas.",
    )
    parser.add_argument(
        "--buscar-urls-perplexity",
        action="store_true",
        help="Ativa busca de URLs (TSE/TRE/Gerais) por linha via API da Perplexity.",
    )
    parser.add_argument(
        "--buscar-urls-gemini",
        action="store_true",
        help="Ativa busca economica de URLs (TSE/TRE/Gerais) por linha via Gemini com Google Search.",
    )
    parser.add_argument(
        "--gerar-tema-punchline-chatgpt",
        action="store_true",
        help="Ativa geracao de tema e punchline por linha via OpenAI (padrao: gpt-5.1).",
    )
    parser.add_argument(
        "--gerar-tema-punchline-local",
        action="store_true",
        help="Preenche tema e punchline por heuristica local, sem chamadas OpenAI.",
    )
    parser.add_argument(
        "--enriquecer-assuntos-openai",
        action="store_true",
        help="Ativa fallback OpenAI para preencher assuntos apenas em linhas ainda vazias.",
    )
    parser.add_argument(
        "--assuntos-max-itens",
        type=int,
        default=DEFAULT_ASSUNTOS_MAX_ITEMS,
        help=f"Numero maximo de assuntos por linha (padrao: {DEFAULT_ASSUNTOS_MAX_ITEMS}).",
    )
    parser.add_argument(
        "--assuntos-taxonomy-mode",
        default="mixed",
        choices=list(ASSUNTOS_TAXONOMY_CHOICES),
        help="Modo de taxonomia para assuntos: controlled, mixed ou free.",
    )
    parser.add_argument(
        "--perplexity-api-key",
        default="",
        help="API key da Perplexity (precedencia maior; evita depender de arquivo/env).",
    )
    parser.add_argument(
        "--perplexity-key-file",
        default=DEFAULT_PERPLEXITY_KEY_FILE,
        help=(
            "Arquivo local com a chave da Perplexity (padrao: "
            f"{DEFAULT_PERPLEXITY_KEY_FILE}; usado quando --perplexity-api-key/env nao forem informados)."
        ),
    )
    parser.add_argument(
        "--perplexity-model",
        default="sonar",
        help="Modelo da Perplexity (padrao: sonar).",
    )
    parser.add_argument(
        "--perplexity-max-workers",
        type=int,
        default=4,
        help="Numero maximo de workers para chamadas HTTP da Perplexity.",
    )
    parser.add_argument(
        "--perplexity-timeout",
        type=int,
        default=15,
        help="Timeout (segundos) de cada chamada Perplexity.",
    )
    parser.add_argument(
        "--perplexity-batch-size",
        type=int,
        default=20,
        help="Quantidade de linhas por lote de consultas Perplexity.",
    )
    parser.add_argument(
        "--perplexity-delay",
        type=float,
        default=0.3,
        help="Pausa (segundos) entre lotes de consultas Perplexity.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default="",
        help="API key do Gemini/Google AI Studio (precedencia maior; evita depender de arquivo/env).",
    )
    parser.add_argument(
        "--gemini-key-file",
        default=DEFAULT_GEMINI_KEY_FILE,
        help=(
            "Arquivo local com a chave do Gemini (padrao: "
            f"{DEFAULT_GEMINI_KEY_FILE}; usado quando --gemini-api-key/env nao forem informados)."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_NEWS_MODEL,
        help=f"Modelo Gemini para busca de noticias (padrao: {DEFAULT_GEMINI_NEWS_MODEL}).",
    )
    parser.add_argument(
        "--gemini-max-workers",
        type=int,
        default=GEMINI_DEFAULT_MAX_WORKERS,
        help="Numero maximo de workers para chamadas HTTP do Gemini.",
    )
    parser.add_argument(
        "--gemini-timeout",
        type=int,
        default=GEMINI_DEFAULT_TIMEOUT,
        help="Timeout (segundos) de cada chamada Gemini.",
    )
    parser.add_argument(
        "--gemini-batch-size",
        type=int,
        default=GEMINI_DEFAULT_BATCH_SIZE,
        help="Quantidade de linhas por lote de consultas Gemini.",
    )
    parser.add_argument(
        "--gemini-delay",
        type=float,
        default=GEMINI_DEFAULT_DELAY,
        help="Pausa (segundos) entre lotes de consultas Gemini.",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="API key da OpenAI (precedencia maior; evita depender de arquivo/env).",
    )
    parser.add_argument(
        "--openai-key-file",
        default=DEFAULT_OPENAI_KEY_FILE,
        help=(
            "Arquivo local com a chave da OpenAI (padrao: "
            f"{DEFAULT_OPENAI_KEY_FILE}; usado quando --openai-api-key/env nao forem informados)."
        ),
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5.1",
        help="Modelo OpenAI para gerar tema/punchline (padrao: gpt-5.1).",
    )
    parser.add_argument(
        "--openai-max-workers",
        type=int,
        default=OPENAI_DEFAULT_MAX_WORKERS,
        help="Numero maximo de workers para chamadas HTTP da OpenAI.",
    )
    parser.add_argument(
        "--openai-timeout",
        type=int,
        default=OPENAI_DEFAULT_TIMEOUT,
        help="Timeout (segundos) de cada chamada OpenAI.",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=OPENAI_DEFAULT_BATCH_SIZE,
        help="Quantidade de linhas por lote de consultas OpenAI.",
    )
    parser.add_argument(
        "--openai-delay",
        type=float,
        default=OPENAI_DEFAULT_DELAY,
        help="Pausa (segundos) entre lotes de consultas OpenAI.",
    )
    parser.add_argument(
        "--openai-retries",
        type=int,
        default=OPENAI_DEFAULT_RETRIES,
        help="Numero maximo de tentativas por chamada OpenAI.",
    )
    parser.add_argument(
        "--openai-target-rpm",
        type=int,
        default=OPENAI_DEFAULT_TARGET_RPM,
        help="Limite alvo de requests por minuto para OpenAI (0 desativa pacing).",
    )
    parser.add_argument(
        "--preserve-from-csv",
        default="",
        help=(
            "CSV de referencia para preservar colunas ja preenchidas "
            "(ex.: saida anterior com dados de API)."
        ),
    )
    parser.add_argument(
        "--preserve-columns",
        default="",
        help=(
            "Lista de colunas separadas por virgula/; para preservar do CSV de referencia. "
            f"Padrao: {', '.join(DEFAULT_PRESERVE_COLUMNS)}."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe logs detalhados.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Exibe apenas avisos/erros.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa modo debug técnico.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Arquivo opcional para salvar logs.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Forca modo CLI mesmo sem arquivos posicionais.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_standard_logging(
        "SJUR_csv_to_csv_NOTIONfriendly",
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
    )
    install_print_logger_bridge(globals(), logger)

    run_cli = bool(args.files) or args.no_gui
    if run_cli:
        if not args.files:
            parser.error("No modo --no-gui, informe ao menos um arquivo CSV.")
        if args.buscar_urls_gemini and args.buscar_urls_perplexity:
            parser.error("Use apenas um mecanismo de busca: --buscar-urls-gemini ou --buscar-urls-perplexity.")
        if args.perplexity_max_workers <= 0:
            parser.error("--perplexity-max-workers deve ser maior que zero.")
        if args.perplexity_timeout <= 0:
            parser.error("--perplexity-timeout deve ser maior que zero.")
        if args.perplexity_batch_size <= 0:
            parser.error("--perplexity-batch-size deve ser maior que zero.")
        if args.perplexity_delay < 0:
            parser.error("--perplexity-delay nao pode ser negativo.")
        if args.gemini_max_workers <= 0:
            parser.error("--gemini-max-workers deve ser maior que zero.")
        if args.gemini_timeout <= 0:
            parser.error("--gemini-timeout deve ser maior que zero.")
        if args.gemini_batch_size <= 0:
            parser.error("--gemini-batch-size deve ser maior que zero.")
        if args.gemini_delay < 0:
            parser.error("--gemini-delay nao pode ser negativo.")
        if args.openai_max_workers <= 0:
            parser.error("--openai-max-workers deve ser maior que zero.")
        if args.openai_timeout <= 0:
            parser.error("--openai-timeout deve ser maior que zero.")
        if args.openai_batch_size <= 0:
            parser.error("--openai-batch-size deve ser maior que zero.")
        if args.openai_retries <= 0:
            parser.error("--openai-retries deve ser maior que zero.")
        if args.openai_target_rpm < 0:
            parser.error("--openai-target-rpm nao pode ser negativo.")
        if args.openai_delay < 0:
            parser.error("--openai-delay nao pode ser negativo.")
        if args.assuntos_max_itens <= 0:
            parser.error("--assuntos-max-itens deve ser maior que zero.")

        def cli_logger(message: str) -> None:
            logger.info(message)

        resolved_openai_api_key = resolve_openai_api_key(args.openai_api_key.strip(), args.openai_key_file)
        tema_punchline_config = TemaPunchlineConfig(
            enabled=args.gerar_tema_punchline_chatgpt,
            api_key=resolved_openai_api_key,
            model=args.openai_model.strip() or "gpt-5.1",
            timeout_seconds=args.openai_timeout,
            max_workers=args.openai_max_workers,
            batch_size=args.openai_batch_size,
            delay_between_batches=args.openai_delay,
            retries=args.openai_retries,
            target_rpm=args.openai_target_rpm,
        )
        assuntos_enrichment_config = AssuntosEnrichmentConfig(
            enabled=args.enriquecer_assuntos_openai,
            api_key=resolved_openai_api_key,
            model=args.openai_model.strip() or "gpt-5.1",
            timeout_seconds=args.openai_timeout,
            max_workers=args.openai_max_workers,
            batch_size=args.openai_batch_size,
            delay_between_batches=args.openai_delay,
            retries=args.openai_retries,
            target_rpm=args.openai_target_rpm,
            max_items=args.assuntos_max_itens,
            taxonomy_mode=args.assuntos_taxonomy_mode.strip().lower(),
        )
        metadata_extraction_config = MetadataExtractionConfig(
            include_institutional_entities=True,
            header_max_chars=DEFAULT_METADATA_HEADER_MAX_CHARS,
        )
        web_lookup_provider = "gemini" if args.buscar_urls_gemini else "perplexity"
        if web_lookup_provider == "gemini":
            web_lookup_config = WebLookupConfig(
                enabled=args.buscar_urls_gemini,
                provider="gemini",
                api_key=resolve_gemini_api_key(args.gemini_api_key.strip(), args.gemini_key_file),
                model=args.gemini_model.strip() or DEFAULT_GEMINI_NEWS_MODEL,
                timeout_seconds=args.gemini_timeout,
                max_workers=args.gemini_max_workers,
                batch_size=args.gemini_batch_size,
                delay_between_batches=args.gemini_delay,
                max_tokens=GEMINI_DEFAULT_MAX_OUTPUT_TOKENS,
            )
        else:
            web_lookup_config = WebLookupConfig(
                enabled=args.buscar_urls_perplexity,
                provider="perplexity",
                api_key=resolve_perplexity_api_key(args.perplexity_api_key.strip(), args.perplexity_key_file),
                model=args.perplexity_model.strip() or "sonar",
                timeout_seconds=args.perplexity_timeout,
                max_workers=args.perplexity_max_workers,
                batch_size=args.perplexity_batch_size,
                delay_between_batches=args.perplexity_delay,
                max_tokens=PERPLEXITY_DEFAULT_MAX_TOKENS,
            )
        preserve_columns = parse_columns_arg(args.preserve_columns, DEFAULT_PRESERVE_COLUMNS)
        local_tema_punchline_config = LocalTemaPunchlineConfig(enabled=args.gerar_tema_punchline_local)

        try:
            summaries, combined_path, compiled_rows = run_batch(
                files=args.files,
                out_dir=args.out_dir,
                max_texto_chars=args.max_texto_chars,
                combined_name=args.combined_name,
                replace_newlines=not args.keep_newlines,
                web_lookup_config=web_lookup_config,
                local_tema_punchline_config=local_tema_punchline_config,
                tema_punchline_config=tema_punchline_config,
                assuntos_enrichment_config=assuntos_enrichment_config,
                metadata_extraction_config=metadata_extraction_config,
                logger=cli_logger,
                preserve_from_csv=args.preserve_from_csv,
                preserve_columns=preserve_columns,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Erro: {exc}", file=sys.stderr)
            return 1

        total_rows = sum(item.rows for item in summaries)
        total_trunc = sum(item.truncated_cells for item in summaries)
        print("\nResumo:")
        print(f"- Arquivos processados: {len(summaries)}")
        print(f"- Linhas processadas: {total_rows}")
        print(f"- Celulas truncadas: {total_trunc}")
        print(f"- Linhas no compilado: {compiled_rows}")
        print(f"- Compilado: {combined_path}")
        return 0

    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
