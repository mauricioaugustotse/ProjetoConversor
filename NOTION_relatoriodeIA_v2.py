#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera relatório estratégico do DJe/TSE em uma página do Notion.

Fluxo:
1. Abre GUI para coletar página alvo, base do Notion e período.
2. Consulta a base DJe por dataDecisao no período exato informado.
3. Analisa os processos em lotes com OpenAI, com fallback heurístico.
4. Publica o relatório em uma página do Notion com callouts, tabelas e comentários.
5. Salva um report JSON local para auditoria da execução.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import HTTPAdapter

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
from Artefatos.scripts.project_layout import (
    KEYS_DIR,
    default_checkpoint_path,
    default_report_path,
    named_log_path,
    notion_secret_candidates,
)
from Artefatos.scripts.openai_log_utils import configure_standard_logging, install_print_logger_bridge
from Artefatos.scripts.openai_progress_utils import read_json_dict, utc_now_iso, write_json_atomic

SCRIPT_FILE = Path(__file__).name
SCRIPT_STEM = Path(__file__).stem
REPORT_FILE = default_report_path(SCRIPT_STEM)
CHECKPOINT_FILE = default_checkpoint_path(SCRIPT_STEM)
DEFAULT_LOG_FILE = named_log_path("run_producao_relatorio_v2.log")
ANALYSIS_CACHE_FILE = SCRIPT_DIR / "Artefatos" / "cache" / "notion_relatorio_case_analysis_cache.json"

DEFAULT_SOURCE_DATABASE_URL = "https://www.notion.so/328721955c648020988af0654c3554a2?v=cfbaf74128e14ad1a210db1be8fc68c1"
DEFAULT_NOTION_VERSION = "2025-09-03"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
DEFAULT_MAX_CASES_PER_BATCH = 1
DEFAULT_OPENAI_MAX_WORKERS = 12
DEFAULT_OPENAI_TARGET_RPM = 360
DEFAULT_OPENAI_BATCH_CHAR_BUDGET = 14000
DEFAULT_MAX_OPENAI_CASES = 1000
DEFAULT_OPENAI_TRIAGE_THRESHOLD = 0
MAX_OPENAI_LIGHT_BATCH_WORKERS = 12
MAX_OPENAI_HEAVY_BATCH_WORKERS = 2
DEFAULT_HTTP_POOL_SIZE = 32
MAX_NOTION_APPEND_CHILDREN = 100
MAX_NOTION_TOGGLES_PER_APPEND = 10
MAX_NOTION_RICH_TEXT_CHARS = 2000
MAX_CASE_TEXT_CHARS = 1400
MAX_SUMMARY_CASES = 16
MAX_EXECUTIVE_HIGHLIGHTS = 3
MAX_EXECUTIVE_ALERT_ITEMS = 4
MAX_EXECUTIVE_BULLET_CHARS = 260
MAX_OVERVIEW_CALLOUT_CHARS = 320
MAX_PARTY_ALERTS = 2
MAX_LAWYER_SIGNALS = 1
MAX_WATCHPOINTS = 2
MAX_STRATEGIC_ALERT_SECTION_ITEMS = 3
MAX_ALERT_TEXT_CHARS = 260
MAX_PUBLISHED_CASES = 12
MAX_TABLE_WHAT_HAPPENED_CHARS = 240
MAX_TABLE_CONSEQUENCE_CHARS = 200
MAX_TOGGLE_WHY_RELEVANT_CHARS = 360
MAX_TOGGLE_FACTS_CHARS = 430
MAX_TOGGLE_LEGAL_CHARS = 420
MAX_TOGGLE_CONSEQUENCE_CHARS = 320
MAX_TOGGLE_STRATEGIC_CHARS = 360
MAX_RECURRING_LAWYERS_SECTION_ITEMS = 8
FEDERAL_INTEREST_MIN_SCORE = 35
MIN_RECURRING_ALERT_CASES = 2
HEADING_SUMMARY_MAX_CHARS = 130
HEADING_SUMMARY_TARGET_MIN = 90
HEADING_SUMMARY_TARGET_MAX = 120
HEADING_ACCEPTABLE_SCORE = 60
TOP_BAND_RECLASSIFY_BATCH_SIZE = 20
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
VERBOSE_HEARTBEAT_INTERVAL_S = 5.0

SECTION_EXECUTIVE_TITLE = "1. Leitura executiva e alertas"
SECTION_CASES_TITLE = "2. Casos prioritários"
SECTION_NOTES_TITLE = "3. Casos detalhados"
SECTION_INVENTORY_TITLE = "4. Demais casos do período"
SECTION_LAWYERS_TITLE = "5. Advogados recorrentes nos casos detalhados"
SECTION_METHODOLOGY_TITLE = "6. Metodologia"
SECTION_METHODOLOGY_WITHOUT_LAWYERS_TITLE = "5. Metodologia"
CASE_SECTION_TITLES = (
    SECTION_CASES_TITLE,
    "2. Casos selecionados",
    "4. Casos selecionados",
    "4. Tabela-síntese dos processos",
)
NOTES_SECTION_TITLES = (
    SECTION_NOTES_TITLE,
    "3. Notas por processo",
    "3. Comentários por processo",
    "5. Notas por processo",
    "5. Comentários por processo",
)
INVENTORY_SECTION_TITLES = (SECTION_INVENTORY_TITLE,)
LAWYERS_SECTION_TITLES = (
    SECTION_LAWYERS_TITLE,
    "4. Advogados recorrentes nos casos detalhados",
)
METHODOLOGY_SECTION_TITLES = (
    SECTION_METHODOLOGY_TITLE,
    SECTION_METHODOLOGY_WITHOUT_LAWYERS_TITLE,
    "4. Metodologia",
)

NOTION_BASE_URL = "https://api.notion.com"
OPENAI_BASE_URL = "https://api.openai.com"

INLINE_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)>\"]+)\)|(https?://[^\s)>\"]+)")
INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
INLINE_ITALIC_RE = re.compile(r"(?<!\*)\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)", flags=re.DOTALL)
LEADING_DATE_PREFIX_RE = re.compile(r"^(?:em|no|na)\s+\d{2}/\d{2}/\d{4},?\s*", flags=re.IGNORECASE)
LEADING_COURT_PREFIX_RE = re.compile(r"^(?:o|a)\s+TSE,?\s*", flags=re.IGNORECASE)
LEADING_MONOCRATIC_PREFIX_RE = re.compile(
    r"^(?:por\s+)?decis[aã]o\s+monocr[aá]tica(?:\s+do(?:a)?\s+min\.?\s+[^,]+)?\,?\s*",
    flags=re.IGNORECASE,
)
LEADING_RELATOR_PREFIX_RE = re.compile(
    r"^(?:o|a)\s+relator(?:a)?(?:\s+[^,]+)?\,?\s*",
    flags=re.IGNORECASE,
)
GENERIC_HEADING_OPENING_RE = re.compile(
    r"^(?:"
    r"em\s+decis[aã]o\s+monocr[aá]tica|"
    r"em\s+sede\s+de|"
    r"na\s+decis[aã]o\s+monocr[aá]tica|"
    r"interpost[oa]\s+contra|"
    r"contra\s+ac[oó]rd[aã]o|"
    r"o\s+ac[oó]rd[aã]o\s+regional|"
    r"ac[oó]rd[aã]o\s+regional|"
    r"o\s+relator(?:a)?|"
    r"relator(?:a)?"
    r")\b",
    flags=re.IGNORECASE,
)
NARRATIVE_HEADING_OPENING_RE = re.compile(
    r"^(?:"
    r"por\s+ac[oó]rd[aã]o|"
    r"por\s+unanimidade|"
    r"em\s+ac[oó]rd[aã]o(?:\s+un[aâ]nime)?|"
    r"tese\s+firmada|"
    r"deu\s+provimento|"
    r"negou\s+seguimento|"
    r"indeferiu|"
    r"manteve|"
    r"reformou|"
    r"julgou|"
    r"determinou|"
    r"reconheceu|"
    r"assentou|"
    r"firmou|"
    r"concluiu|"
    r"acolheu|"
    r"afastou|"
    r"cassou|"
    r"deferiu|"
    r"proveu"
    r")\b",
    flags=re.IGNORECASE,
)
ENGLISH_ITALIC_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(r"(?i)\bdark posts?\b"),
    re.compile(r"(?i)\bdeepfakes?\b"),
    re.compile(r"(?i)\bcompliance\b"),
    re.compile(r"(?i)\bwatchpoints?\b"),
    re.compile(r"(?i)\bdry run\b"),
]

PARTY_ABBREVIATIONS = [
    "AGIR",
    "AVANTE",
    "CIDADANIA",
    "DC",
    "MDB",
    "MOBILIZA",
    "NOVO",
    "PCDOB",
    "PDT",
    "PL",
    "PMB",
    "PODE",
    "PP",
    "PRD",
    "PRTB",
    "PROS",
    "PSB",
    "PSD",
    "PSDB",
    "PSOL",
    "PT",
    "PV",
    "REDE",
    "REPUBLICANOS",
    "SOLIDARIEDADE",
    "UNIAO BRASIL",
    "UP",
]
PARTY_REGEX_ALTERNATIVES = [re.escape(item).replace(r"\ ", r"\s+") for item in PARTY_ABBREVIATIONS]
PARTY_TOKEN_RE = re.compile(r"\b(" + "|".join(PARTY_REGEX_ALTERNATIVES) + r")\b", flags=re.IGNORECASE)
PARTY_PAREN_RE = re.compile(r"\(([A-Z]{2,20})\)")
PUBLIC_FIGURE_HINT_RE = re.compile(
    r"\b(prefeit[oa]|vice-prefeit[oa]|vereador(?:a)?|governador(?:a)?|senador(?:a)?|deputad[oa]|"
    r"presidente|candidato(?:a)?|prefeitura|governo|mandato|cass[a-z]+|inelegib[a-z]+)\b",
    flags=re.IGNORECASE,
)
ACCOUNTS_HINT_RE = re.compile(r"\b(contas|prestacao de contas|prestação de contas|fefc|fundo partid[aá]rio)\b", flags=re.IGNORECASE)
HIGH_IMPACT_HINT_RE = re.compile(r"\b(cass[a-z]+|inelegib[a-z]+|abuso de poder|capta[cç][aã]o il[ií]cita|conduta vedada)\b", flags=re.IGNORECASE)
RISK_DISPLAY_LABELS: Dict[str, str] = {
    "critico": "CRÍTICO",
    "alto": "ALTO",
    "medio": "MÉDIO",
    "baixo": "BAIXO",
}
RISK_SORT_ORDER: Dict[str, int] = {"critico": 0, "alto": 1, "medio": 2, "baixo": 3}
REPORTABLE_RISK_LEVELS = {"critico", "alto", "medio"}
PUBLISHED_RISK_LEVELS = {"critico", "alto"}
EXECUTIVE_MAJOR_ROLE_RE = re.compile(
    r"\b(presidente|governador(?:a)?|senador(?:a)?|prefeit[oa]|vice-prefeit[oa])\b",
    flags=re.IGNORECASE,
)
LEGISLATIVE_ROLE_RE = re.compile(r"\b(deputad[oa]|vereador(?:a)?)\b", flags=re.IGNORECASE)
HIGH_IMPACT_PUBLICATION_RE = re.compile(
    r"\b(cassa[cç][aã]o|cassou|inelegibil(?:idade|idade)|ineleg[ií]vel|perda de mandato|nulidade dos votos)\b",
    flags=re.IGNORECASE,
)
HIGH_IMPACT_ELECTORAL_RE = re.compile(
    r"\b(AIJE|AIME|RCED|abuso de poder|capta[cç][aã]o il[ií]cita de sufr[aá]gio|art\.?\s*30-?A|art\.?\s*41-?A)\b",
    flags=re.IGNORECASE,
)
HIGH_IMPACT_ACCOUNTS_RE = re.compile(r"\b(fundo partid[aá]rio|FEFC|presta[cç][aã]o de contas)\b", flags=re.IGNORECASE)
FEDERAL_HIGH_OFFICE_RE = re.compile(
    r"\b("
    r"presidente(?:\s+da\s+rep[uú]blica)?|vice-presidente(?:\s+da\s+rep[uú]blica)?|"
    r"senador(?:a)?(?:\s+da\s+rep[uú]blica)?|governador(?:a)?|deputad[oa]\s+federal|"
    r"ministro(?:a)?\s+de\s+estado"
    r")\b",
    flags=re.IGNORECASE,
)
LOWER_LOCAL_OFFICE_RE = re.compile(r"\b(prefeit[oa]|vice-prefeit[oa]|vereador(?:a)?)\b", flags=re.IGNORECASE)
NATIONAL_PARTY_SCOPE_RE = re.compile(
    r"\b(nacional|dire[cç][aã]o\s+nacional|diret[oó]rio\s+nacional|executiva\s+nacional|"
    r"[oó]rg[aã]o\s+nacional|funda[cç][aã]o\s+partid[aá]ria|registro\s+de\s+partido|"
    r"incorpora[cç][aã]o|fus[aã]o|federa[cç][aã]o\s+partid[aá]ria)\b",
    flags=re.IGNORECASE,
)
PARTY_INSTITUTIONAL_RE = re.compile(
    r"\b(partido\s+pol[ií]tico|diret[oó]rio|comiss[aã]o\s+provis[oó]ria|"
    r"presta[cç][aã]o\s+de\s+contas\s+anual|contas\s+partid[aá]rias|"
    r"fidelidade\s+partid[aá]ria|janela\s+partid[aá]ria|desfilia[cç][aã]o|"
    r"propaganda\s+partid[aá]ria|estatuto\s+partid[aá]rio)\b",
    flags=re.IGNORECASE,
)
PARTY_FINANCE_SYSTEMIC_RE = re.compile(
    r"\b(fundo\s+partid[aá]rio|FEFC|cota\s+de\s+g[eê]nero|cota\s+racial|"
    r"participa[cç][aã]o\s+pol[ií]tica\s+das\s+mulheres|ADPF\s*738|EC\s*n?[ºo]?\s*117|"
    r"EC\s*n?[ºo]?\s*133|devolu[cç][aã]o\s+ao\s+tesouro|suspens[aã]o\s+de\s+cotas)\b",
    flags=re.IGNORECASE,
)
GENERIC_ALERT_START_RE = re.compile(r"^(?:sem|nenhum|nenhuma|n/a|não há|nao ha)\b", flags=re.IGNORECASE)
GENERIC_COUNT_ONLY_ALERT_RE = re.compile(
    r"^[^:]{1,90}:\s*(?:citado(?:a|s)?|atua(?:m)?|aparece(?:m)?|reaparece(?:m)?|recorre(?:m)?|figura(?:m)?)\s+em\s+\d+\s+processo",
    flags=re.IGNORECASE,
)
GENERIC_PRIORITY_ONLY_ALERT_RE = re.compile(r":\s*prioridade\s*\d+\s*/\s*10\.?$", flags=re.IGNORECASE)
ALERT_MATERIALITY_RE = re.compile(
    r"\b(risco|impacto|mandato|cass[a-zçãõ]+|ineleg[a-zçãõ]+|vac[aâ]ncia|suplementar|contas|fefc|devolu[cç][aã]o|"
    r"abuso|san[cç][aã]o|nulidade|controle|precedente|tese|exposi[cç][aã]o|er[aá]rio|mandato)\b",
    flags=re.IGNORECASE,
)
ELLIPSIS_TRAILING_RE = re.compile(r"(?:\.{3,}|…)\s*$")
TEMPORAL_HEADING_CLAUSE_RE = re.compile(
    r"^(?:"
    r"exerc[ií]cio(?:\s+financeiro)?\s+de\s+(?P<year1>\d{4})|"
    r"elei[cç][oõ]es?\s+de\s+(?P<year2>\d{4})|"
    r"nas?\s+elei[cç][oõ]es?\s+de\s+(?P<year3>\d{4})"
    r")$",
    flags=re.IGNORECASE,
)
HEADING_TRAILING_STOPWORDS = {
    "a",
    "as",
    "à",
    "às",
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
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "sem",
    "sob",
}
SUMMARY_TRAILING_STOPWORDS = HEADING_TRAILING_STOPWORDS | {"ao", "aos", "como", "o", "os", "que", "um", "uma", "uns", "umas"}
INCOMPLETE_WORD_FRAGMENT_RE = re.compile(
    r"\b(?:"
    r"determi|determina[cç]|"
    r"repercu|"
    r"desaprova[cç]|"
    r"impugna[cç]|"
    r"retotaliza[cç]|"
    r"cassa[cç]|"
    r"anula[cç]|"
    r"aplica[cç]|"
    r"participa[cç]|"
    r"obriga[cç]|"
    r"comprova[cç]|"
    r"fiscaliza[cç]|"
    r"inelegibilid|"
    r"plausibilid|"
    r"admissibilid|"
    r"regularid|"
    r"irregularid|"
    r"confiabilid|"
    r"continuid|"
    r"conformid"
    r")\s*$",
    flags=re.IGNORECASE,
)
UNFINISHED_PREDICATE_RE = re.compile(
    r"(?:"
    r"\b(?:com\s+isso|com\s+efeitos?|para\s+reduzir)\s*$|"
    r"\b(?:permaneceram|permaneceu|ficaram|ficou|mantiveram|manteve|determinou|reconheceu|gerou|gera|geram)\s*$|"
    r"\b(?:gerou|gera|geram)\s+perda\s*$|"
    r"\b(?:perda|risco|impacto|obrigação|dever)\s+de\s+[\wÀ-ÿ-]+\s*$"
    r")",
    flags=re.IGNORECASE,
)
HEADING_SUSPICIOUS_END_RE = re.compile(
    r"(?:\b(?:a|as|à|às|com|da|das|de|do|dos|e|em|na|nas|no|nos|ou|para|pela|pelas|pelo|pelos|por|sem|sob)\s*$|[,;:/\-–(]\s*$)",
    flags=re.IGNORECASE,
)
SUMMARY_SUSPICIOUS_END_RE = re.compile(
    r"(?:\b(?:a|as|à|às|ao|aos|com|como|da|das|de|do|dos|e|em|na|nas|no|nos|ou|para|pela|pelas|pelo|pelos|por|que|sem|sob)\s*$|[,;:/\-–(]\s*$)",
    flags=re.IGNORECASE,
)
HEADING_INCOMPLETE_ART_RE = re.compile(
    r"\bart\.\s*\d+[º°]?(?:\s*,\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X))?\s*$",
    flags=re.IGNORECASE,
)
HEADING_INCOMPLETE_LEI_RE = re.compile(r"\bLei\s*n[ºo\.\-]?\s*\d+[\./]?\s*$", flags=re.IGNORECASE)
HEADING_INCOMPLETE_RES_RE = re.compile(r"\bRes\.?[-–]?TSE\s*\d+[\./]?\s*$", flags=re.IGNORECASE)
HEADING_OPEN_PAREN_RE = re.compile(r"\([^)]*$")
HEADING_TEMPORAL_PAREN_RE = re.compile(r"\((?:exerc[ií]cio|Elei[cç][oõ]es)\s+\d{4}\)$", flags=re.IGNORECASE)
HEADING_ESSENTIAL_LEGAL_RE = re.compile(
    r"\b(?:AIJE|AIME|RCED|art\.?\s*30-?A|art\.?\s*41-?A)\b",
    flags=re.IGNORECASE,
)
HEADING_NONESSENTIAL_LEGAL_TAIL_RE = re.compile(
    r"^(?:art\.?\s*\d+.*|Res\.?[-–]?TSE.*|S[úu]mulas?.*|Enunciados?.*)$",
    flags=re.IGNORECASE,
)
HEADING_WEAK_CLAUSE_RE = re.compile(
    r"^(?:falhas?|limites?|interposi[cç][aã]o|s[úu]mulas?|enunciados?|interposi[cç][aã]o|v[íi]cios?|outros?\s+v[íi]cios?)$",
    flags=re.IGNORECASE,
)
GENERIC_HEADING_EXACT_RE = re.compile(
    r"^(?:"
    r"obriga(?:toriedade)?|"
    r"admissibilidade\s+recursal\s+eleitoral|"
    r"elei[cç][oõ]es?\s+\d{4}|"
    r"aplica[cç][aã]o\s+do\s+art\.?\s*\d+[º°]?(?:\s*,\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X))?|"
    r"presta[cç][aã]o\s+de\s+contas(?:"
    r"\s+eleitorais?|"
    r"\s+partid[aá]rias(?:\s+eletr[oô]nicas)?|"
    r"\s+de\s+campanha(?:\s+eleitoral)?|"
    r"\s+de\s+campanha\s*,\s*vereador|"
    r"\s+eleitorais\s*,\s*vereador|"
    r"\s+de\s+partido\s+pol[ií]tico|"
    r"\s+partid[aá]rias\s+de\s+dire[cç][aã]o\s+nacional|"
    r"\s+de\s+candidato\s+a\s+vereador"
    r")?|"
    r"presta[cç][aã]o\s+de\s+contas.*,\s*exerc[ií]cio\s+financeiro"
    r")$",
    flags=re.IGNORECASE,
)
ELLIPSIS_ANALYSIS_FIELDS = (
    "title",
    "what_happened",
    "legal_grounds",
    "consequence",
    "strategic_comment",
    "why_relevant",
)

LOGGER = logging.getLogger(SCRIPT_STEM)


@dataclass
class NotionConfig:
    token: str
    notion_version: str = DEFAULT_NOTION_VERSION
    timeout_s: int = 30
    max_retries: int = 4


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = DEFAULT_OPENAI_MODEL
    timeout_s: int = 90
    retries: int = 3
    target_rpm: int = DEFAULT_OPENAI_TARGET_RPM


@dataclass
class RunInputs:
    page_url: str
    source_database_url: str
    start_date_iso: str
    end_date_iso: str
    force_clear: bool = False
    dry_run: bool = False
    delayed_cases_file: str = ""


@dataclass
class CaseRecord:
    case_id: str
    page_id: str
    source_url: str
    data_decisao: str
    numero_unico: str
    numero_processo: str
    ano_eleicao: str
    sigla_classe: str
    descricao_classe: str
    sigla_uf: str
    nome_municipio: str
    descricao_tipo_decisao: str
    assuntos: List[str] = field(default_factory=list)
    partes: List[str] = field(default_factory=list)
    partidos: List[str] = field(default_factory=list)
    relator: str = ""
    advogados: List[str] = field(default_factory=list)
    resultado: List[str] = field(default_factory=list)
    tema: str = ""
    punchline: str = ""
    texto_decisao: str = ""
    noticias: List[str] = field(default_factory=list)
    # Curadoria manual na base: checkbox "Dep. Federal" marca caso de interesse
    # direto do consultor; entra nos destaques independentemente da analise.
    dep_federal: bool = False

    def process_label(self) -> str:
        return self.numero_unico or self.numero_processo or self.case_id

    def local_label(self) -> str:
        if self.nome_municipio and self.sigla_uf:
            return f"{self.nome_municipio}/{self.sigla_uf}"
        return self.sigla_uf or self.nome_municipio or "-"

    def compact_ai_payload(self, recurring_party_counts: Counter[str], recurring_lawyer_counts: Counter[str]) -> Dict[str, Any]:
        recurring_parties = [party for party in self.partidos if recurring_party_counts.get(party, 0) > 1]
        recurring_lawyers = [name for name in self.advogados if recurring_lawyer_counts.get(name, 0) > 1]
        return {
            "case_id": self.page_id or self.case_id,
            "processo": self.process_label(),
            "data_decisao": self.data_decisao,
            "classe": self.sigla_classe or self.descricao_classe,
            "descricao_classe": self.descricao_classe,
            "tipo_decisao": self.descricao_tipo_decisao,
            "ano_eleicao": self.ano_eleicao,
            "uf": self.sigla_uf,
            "municipio": self.nome_municipio,
            "relator": self.relator,
            "assuntos": self.assuntos,
            "partes": self.partes,
            "partidos_identificados": self.partidos,
            "partidos_recorrentes_no_periodo": recurring_parties,
            "advogados": self.advogados,
            "advogados_recorrentes_no_periodo": recurring_lawyers,
            "resultado": self.resultado,
            "tema": self.tema,
            "punchline": self.punchline,
            "texto_decisao_trecho": truncate_text(self.texto_decisao, MAX_CASE_TEXT_CHARS),
            "urls_referencia": self.noticias + [self.source_url],
        }


@dataclass
class CaseAnalysis:
    case_id: str
    title: str
    relevance_score: int
    risk_level: str
    includes_public_figure: bool
    includes_party: bool
    public_figures: List[str]
    parties: List[str]
    lawyers_signal: str
    what_happened: str
    legal_grounds: str
    consequence: str
    strategic_comment: str
    why_relevant: str
    source_notes: List[str]
    page_id: str = ""
    display_score: int = 0
    ranking_reason: str = ""


@dataclass
class ReportSummary:
    overview_callout: str
    executive_highlights: List[str]
    party_alerts: List[str]
    lawyer_signals: List[str]
    watchpoints: List[str]
    closing_note: str


NOTION_CFG: Optional[NotionConfig] = None
OPENAI_CFG: Optional[OpenAIConfig] = None
NOTION_SESSION: Optional[requests.Session] = None
OPENAI_SESSION: Optional[requests.Session] = None
OPENAI_PACER: Optional["RequestPacer"] = None


CASE_BATCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "case_id": {"type": "string"},
                    "title": {"type": "string"},
                    "relevance_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "risk_level": {"type": "string", "enum": ["baixo", "medio", "alto", "critico"]},
                    "includes_public_figure": {"type": "boolean"},
                    "includes_party": {"type": "boolean"},
                    "public_figures": {"type": "array", "items": {"type": "string"}},
                    "parties": {"type": "array", "items": {"type": "string"}},
                    "lawyers_signal": {"type": "string"},
                    "what_happened": {"type": "string"},
                    "legal_grounds": {"type": "string"},
                    "consequence": {"type": "string"},
                    "strategic_comment": {"type": "string"},
                    "why_relevant": {"type": "string"},
                    "source_notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "case_id",
                    "title",
                    "relevance_score",
                    "risk_level",
                    "includes_public_figure",
                    "includes_party",
                    "public_figures",
                    "parties",
                    "lawyers_signal",
                    "what_happened",
                    "legal_grounds",
                    "consequence",
                    "strategic_comment",
                    "why_relevant",
                    "source_notes",
                ],
            },
        }
    },
    "required": ["items"],
}

TOP_BAND_RECLASSIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "case_id": {"type": "string"},
                    "display_score": {"type": "integer", "enum": [9, 10]},
                    "ranking_reason": {"type": "string"},
                },
                "required": ["case_id", "display_score", "ranking_reason"],
            },
        }
    },
    "required": ["items"],
}

SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overview_callout": {"type": "string"},
        "executive_highlights": {"type": "array", "items": {"type": "string"}},
        "party_alerts": {"type": "array", "items": {"type": "string"}},
        "lawyer_signals": {"type": "array", "items": {"type": "string"}},
        "watchpoints": {"type": "array", "items": {"type": "string"}},
        "closing_note": {"type": "string"},
    },
    "required": [
        "overview_callout",
        "executive_highlights",
        "party_alerts",
        "lawyer_signals",
        "watchpoints",
        "closing_note",
    ],
}


def _normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_join(values: Iterable[str], sep: str = ", ") -> str:
    out = [item for item in (_normalize_ws(value) for value in values) if item]
    return sep.join(out)


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in values:
        value = _normalize_ws(raw)
        if not value:
            continue
        marker = unicodedata.normalize("NFKD", value.casefold())
        marker = "".join(ch for ch in marker if not unicodedata.combining(ch))
        if marker in seen:
            continue
        seen.add(marker)
        out.append(value)
    return out


def extract_http_urls(values: Iterable[str]) -> List[str]:
    urls: List[str] = []
    for raw in values:
        text = _normalize_ws(raw)
        if not text:
            continue
        for match in INLINE_LINK_RE.finditer(text):
            url = _normalize_ws(match.group(2) or match.group(3))
            if url:
                urls.append(url)
    return _unique_preserve_order(urls)


def truncate_text(text: Any, max_chars: int, *, suffix: str = "...") -> str:
    source = _normalize_ws(text)
    if max_chars <= 0 or len(source) <= max_chars:
        return source
    cut = source.rfind(" ", 0, max_chars)
    if cut < int(max_chars * 0.65):
        cut = max_chars
    return source[:cut].rstrip() + _normalize_ws(suffix)


def _split_outside_parentheses(text: str, separators: Sequence[str]) -> List[str]:
    source = str(text or "")
    if not source:
        return []
    ordered_separators = tuple(sorted(separators, key=len, reverse=True))
    depth = 0
    cursor = 0
    chunk: List[str] = []
    parts: List[str] = []
    while cursor < len(source):
        char = source[cursor]
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1

        matched = ""
        if depth == 0:
            for separator in ordered_separators:
                if source.startswith(separator, cursor):
                    matched = separator
                    break
        if matched:
            part = _normalize_ws("".join(chunk)).strip(" .;,:")
            if part:
                parts.append(part)
            chunk = []
            cursor += len(matched)
            continue

        chunk.append(char)
        cursor += 1

    tail = _normalize_ws("".join(chunk)).strip(" .;,:")
    if tail:
        parts.append(tail)
    return parts


def _simplify_heading_clause(text: str) -> str:
    clause = _normalize_ws(text).strip(" .;,:")
    if not clause:
        return ""
    temporal_match = TEMPORAL_HEADING_CLAUSE_RE.match(clause)
    if temporal_match:
        year = temporal_match.group("year1") or temporal_match.group("year2") or temporal_match.group("year3")
        if year:
            if "elei" in clause.casefold():
                return f"Eleições {year}"
            return f"exercício {year}"
    replacements: Sequence[tuple[str, str]] = (
        (r"^recebimento de recursos de fonte vedada$", "fonte vedada"),
        (r"^recebimento de recursos do fundo partid[aá]rio$", "recursos do Fundo Partidário"),
        (r"^recebimento de recursos do fefc$", "recursos do FEFC"),
        (r"^recebimento de doa[cç][aã]o em esp[eé]cie acima do limite previsto no art\.?\s*21$", "doação em espécie acima do limite legal"),
        (r"^uso irregular do fundo partid[aá]rio$", "uso irregular do Fundo Partidário"),
        (r"^uso irregular do fefc$", "uso irregular do FEFC"),
        (r"^extrapola[cç][aã]o do limite global de gastos(?: nas elei[cç][oõ]es de \d{4})? e aplica[cç][aã]o de multa vinculada de 100% sobre o valor excedente$", "extrapolação do limite de gastos, multa de 100% sobre o excedente"),
        (r"^valora[cç][aã]o documental das despesas com combust[ií]veis e veda[cç][aã]o ao repasse de recursos do fefc a candidatos filiados a partidos diversos$", "despesas com combustíveis, vedação ao repasse de FEFC a partido diverso"),
        (r"^demiss[aã]o sem justa causa de servidores tempor[aá]rios no per[ií]odo entre as elei[cç][oõ]es e a posse e valora[cç][aã]o da san[cç][aã]o pecuni[aá]ria$", "demissão sem justa causa de temporários entre eleições e posse, sanção pecuniária"),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, clause, flags=re.IGNORECASE)
        if updated != clause:
            return _normalize_ws(updated)
    return clause


def _split_heading_semantic_units(text: str) -> List[str]:
    source = _normalize_ws(text)
    if not source:
        return []
    primary_parts = _split_outside_parentheses(source, (" — ", " – ", " - ", "; ", ";", ": ", ":"))
    units: List[str] = []
    for idx, part in enumerate(primary_parts):
        if idx == 0:
            units.append(part)
            continue

        comma_units: List[str] = []
        buffer: List[str] = []
        depth = 0
        cursor = 0
        while cursor < len(part):
            char = part[cursor]
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1

            if char == "," and depth == 0:
                lookahead = part[cursor + 1 :].lstrip()
                prefix = _normalize_ws("".join(buffer))
                if (
                    lookahead
                    and lookahead[0].islower()
                    and not re.search(r"(?:art\.?\s*\d+|§\s*\d+|Lei\s*n[ºo\.\-]?\s*\d+)$", prefix, flags=re.IGNORECASE)
                ):
                    comma_units.append(_normalize_ws("".join(buffer)).strip(" .;,:"))
                    buffer = []
                    cursor += 1
                    continue

            buffer.append(char)
            cursor += 1

        tail = _normalize_ws("".join(buffer)).strip(" .;,:")
        if tail:
            comma_units.append(tail)
        units.extend(comma_units)
    return _unique_preserve_order(_simplify_heading_clause(item) for item in units)


def _is_temporal_heading_unit(text: str) -> bool:
    clause = _normalize_ws(text)
    return bool(
        TEMPORAL_HEADING_CLAUSE_RE.match(clause)
        or re.match(r"^(?:exercício|Eleições)\s+\d{4}$", clause, flags=re.IGNORECASE)
    )


def _has_essential_heading_legal_reference(text: str) -> bool:
    return bool(HEADING_ESSENTIAL_LEGAL_RE.search(_normalize_ws(text)))


def _is_discardable_heading_unit(text: str) -> bool:
    clause = _normalize_ws(text).strip(" .;,:")
    if not clause:
        return True
    if _is_temporal_heading_unit(clause):
        return False
    if _has_essential_heading_legal_reference(clause):
        return False
    if HEADING_WEAK_CLAUSE_RE.match(clause):
        return True
    if heading_summary_looks_incomplete(clause):
        return True
    if looks_generic_case_heading_summary(clause) or looks_narrative_case_heading_summary(clause):
        return True
    if HEADING_NONESSENTIAL_LEGAL_TAIL_RE.match(clause) and not _has_essential_heading_legal_reference(clause):
        return True
    return False


def _normalize_temporal_heading_unit(text: str) -> str:
    clause = _simplify_heading_clause(text)
    if not _is_temporal_heading_unit(clause):
        return ""
    if "elei" in clause.casefold():
        year_match = re.search(r"(\d{4})", clause)
        return f"Eleições {year_match.group(1)}" if year_match else "Eleições"
    year_match = re.search(r"(\d{4})", clause)
    return f"exercício {year_match.group(1)}" if year_match else clause


def _strip_nonessential_legal_tail(text: str) -> str:
    clause = _normalize_ws(text)
    if not clause:
        return ""
    if _has_essential_heading_legal_reference(clause):
        return clause
    clause = re.sub(r"\s*\((?:Res\.?[-–]?TSE|S[úu]mulas?|Enunciados?|art\.).*?\)\s*$", "", clause, flags=re.IGNORECASE)
    clause = re.sub(r"[\s,;:-]+(?:Res\.?[-–]?TSE.*|S[úu]mulas?.*|Enunciados?.*)$", "", clause, flags=re.IGNORECASE)
    return _normalize_ws(clause).strip(" .;,:")


def _compact_heading_primary_clause(text: str) -> str:
    clause = _strip_nonessential_legal_tail(_simplify_heading_clause(text))
    if len(clause) <= HEADING_SUMMARY_MAX_CHARS:
        return clause
    subclauses = _split_outside_parentheses(clause, (" — ", " – ", " - ", "; ", ";", ": ", ":"))
    if subclauses:
        clause = _normalize_ws(subclauses[0]).strip(" .;,:")
    if len(clause) <= HEADING_SUMMARY_MAX_CHARS:
        return clause
    comma_chunks = _split_outside_parentheses(clause, (", ", ","))
    if comma_chunks:
        clause = _normalize_ws(comma_chunks[0]).strip(" .;,:")
    return clause


def _compact_heading_qualifier_clause(text: str) -> str:
    clause = _strip_nonessential_legal_tail(_simplify_heading_clause(text))
    if _is_discardable_heading_unit(clause):
        return ""
    return clause


def _split_heading_clauses(text: str) -> List[str]:
    return _split_heading_semantic_units(text)


def _pack_heading_clauses(text: str, max_chars: int) -> str:
    source_clauses = _split_heading_semantic_units(text)
    clauses = [item for item in source_clauses if item]
    if not clauses or max_chars <= 0:
        return ""
    primary = _compact_heading_primary_clause(clauses[0])
    primary_index = 0
    promote_primary = bool(
        primary
        and (
            heading_summary_looks_incomplete(primary)
            or re.match(
                r"^(?:aplica[cç][aã]o\s+do\s+art\.?\s*\d+[º°]?(?:\s*,\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X))?|"
                r"obriga(?:toriedade)?|"
                r"admissibilidade\s+recursal\s+eleitoral|"
                r"elei[cç][oõ]es?\s+\d{4})$",
                primary,
                flags=re.IGNORECASE,
            )
        )
    )
    if promote_primary:
        better_primary = next(
            (
                (idx, _compact_heading_primary_clause(item))
                for idx, item in enumerate(clauses[1:], start=1)
                if item and not _is_discardable_heading_unit(item)
            ),
            (-1, ""),
        )
        if better_primary[1] and not looks_generic_case_heading_summary(better_primary[1]):
            primary_index, primary = better_primary
    if not primary:
        fallback_primary = next((item for item in clauses[1:] if not _is_discardable_heading_unit(item)), "")
        primary = _compact_heading_primary_clause(fallback_primary)
    if not primary:
        return ""

    temporal = ""
    qualifiers: List[str] = []
    for idx, clause in enumerate(clauses[1:], start=1):
        if idx == primary_index:
            continue
        if _is_temporal_heading_unit(clause) and not temporal:
            temporal = _normalize_temporal_heading_unit(clause)
            continue
        compact = _compact_heading_qualifier_clause(clause)
        if compact:
            qualifiers.append(compact)

    headline = primary
    if temporal:
        temporal_candidate = f"{headline} ({temporal})"
        if len(temporal_candidate) <= max_chars:
            headline = temporal_candidate

    max_qualifiers = 2 if temporal else 1
    used_qualifiers = 0
    for qualifier in qualifiers:
        if used_qualifiers >= max_qualifiers:
            break
        candidate = f"{headline}, {qualifier}"
        if len(candidate) <= max_chars:
            headline = candidate
            used_qualifiers += 1

    return _normalize_ws(headline).strip(" .;,:")


def heading_summary_looks_incomplete(text: Any) -> bool:
    summary = _normalize_ws(text)
    if not summary:
        return False
    return bool(
        HEADING_SUSPICIOUS_END_RE.search(summary)
        or INCOMPLETE_WORD_FRAGMENT_RE.search(summary)
        or UNFINISHED_PREDICATE_RE.search(summary)
        or HEADING_INCOMPLETE_ART_RE.search(summary)
        or HEADING_INCOMPLETE_LEI_RE.search(summary)
        or HEADING_INCOMPLETE_RES_RE.search(summary)
        or HEADING_OPEN_PAREN_RE.search(summary)
    )


def _trim_suspicious_heading_tail(text: str) -> str:
    summary = _normalize_ws(text)
    if not summary:
        return ""
    while heading_summary_looks_incomplete(summary):
        words = summary.split()
        tail = ""
        if words:
            tail = re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", words[-1], flags=re.UNICODE).casefold()
        if len(words) > 4 and (tail in HEADING_TRAILING_STOPWORDS or INCOMPLETE_WORD_FRAGMENT_RE.search(tail)):
            summary = " ".join(words[:-1]).rstrip(" ,;:/-–(")
            continue

        cut_positions = [
            summary.rfind(". "),
            summary.rfind("! "),
            summary.rfind("? "),
            summary.rfind(", "),
            summary.rfind(","),
            summary.rfind("; "),
            summary.rfind(";"),
            summary.rfind(": "),
            summary.rfind(":"),
            summary.rfind(" - "),
            summary.rfind(" – "),
            summary.rfind(" ("),
            summary.rfind("("),
        ]
        cut = max(cut_positions)
        if cut < int(len(summary) * 0.5):
            break
        summary = summary[:cut].rstrip(" ,;:/-–(")
    return summary


def summary_text_looks_incomplete(text: Any) -> bool:
    summary = _normalize_ws(text)
    if not summary:
        return False
    return bool(
        SUMMARY_SUSPICIOUS_END_RE.search(summary)
        or INCOMPLETE_WORD_FRAGMENT_RE.search(summary)
        or UNFINISHED_PREDICATE_RE.search(summary)
        or HEADING_INCOMPLETE_ART_RE.search(summary)
        or HEADING_INCOMPLETE_LEI_RE.search(summary)
        or HEADING_INCOMPLETE_RES_RE.search(summary)
        or HEADING_OPEN_PAREN_RE.search(summary)
    )


def _trim_suspicious_summary_tail(text: str) -> str:
    summary = _normalize_ws(text).strip(" ;,:")
    if not summary:
        return ""
    while summary_text_looks_incomplete(summary):
        words = summary.split()
        tail = ""
        if words:
            tail = re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", words[-1], flags=re.UNICODE).casefold()
        if len(words) > 4 and (
            tail in SUMMARY_TRAILING_STOPWORDS
            or INCOMPLETE_WORD_FRAGMENT_RE.search(tail)
        ):
            summary = " ".join(words[:-1]).rstrip(" ,;:/-–(")
            continue

        cut_positions = [
            summary.rfind(". "),
            summary.rfind("! "),
            summary.rfind("? "),
            summary.rfind("; "),
            summary.rfind(";"),
            summary.rfind(": "),
            summary.rfind(":"),
            summary.rfind(", "),
            summary.rfind(","),
            summary.rfind(" - "),
            summary.rfind(" – "),
            summary.rfind(" ("),
            summary.rfind("("),
        ]
        cut = max(cut_positions)
        if cut >= int(len(summary) * 0.6):
            summary = summary[:cut].rstrip(" .,;:/-–(")
            continue
        break
    return summary


def _truncate_summary_snippet(text: Any, max_chars: int) -> str:
    source = _normalize_ws(text)
    if max_chars <= 0 or len(source) <= max_chars:
        return source
    strong_boundaries: List[int] = []
    for pattern in (r"\.\s+", r"!\s+", r"\?\s+", r";\s+"):
        for match in re.finditer(pattern, source):
            if match.end() <= max_chars:
                strong_boundaries.append(match.start())
    if strong_boundaries:
        cut = max(strong_boundaries)
        if cut >= int(max_chars * 0.55):
            return source[:cut].rstrip(" ;,:") + "."
    return truncate_text(source, max_chars, suffix="")


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|;\s+")
BOUNDARY_ABBREVIATION_RE = re.compile(r"\b(?:art|arts|inc|incs|n|no|nº|min|rel|res|dr|dra)\.$", flags=re.IGNORECASE)


def _sentence_boundary_offsets(text: str) -> List[int]:
    source = str(text or "")
    offsets: List[int] = []
    for match in SENTENCE_BOUNDARY_RE.finditer(source):
        cut = match.start()
        prefix = source[max(0, cut - 24) : cut]
        if BOUNDARY_ABBREVIATION_RE.search(prefix):
            continue
        offsets.append(cut)
    if source.endswith((".", "!", "?", ";")):
        offsets.append(len(source))
    return sorted(set(offsets))


def _finish_report_fragment(text: str) -> str:
    cleaned = _normalize_ws(text).rstrip(" ;,:")
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return cleaned


def _drop_leading_lowercase_fragment(text: str) -> str:
    value = _normalize_ws(text)
    if not value or not value[:1].islower():
        return value
    match = re.match(r"^[^.!?]{1,80}[.!?]\s+(?=[A-ZÀ-Ý])", value)
    if not match:
        return value
    prefix = value[: match.end()].rstrip()
    if BOUNDARY_ABBREVIATION_RE.search(prefix):
        return value
    remainder = value[match.end() :].strip()
    return remainder if len(remainder) >= 40 else value


def _compact_report_field_text(
    text: Any,
    max_chars: int,
    *,
    overflow_chars: int = 180,
) -> str:
    source = _drop_leading_lowercase_fragment(_normalize_ws(text))
    if max_chars <= 0 or len(source) <= max_chars:
        cleaned = _trim_suspicious_summary_tail(source) or source
        return _finish_report_fragment(cleaned)

    min_safe = max(48, int(max_chars * 0.45))
    offsets = _sentence_boundary_offsets(source)
    before = [offset for offset in offsets if min_safe <= offset <= max_chars]
    if before:
        return _finish_report_fragment(source[: max(before)])

    after_limit = max_chars + max(80, int(overflow_chars))
    after = [offset for offset in offsets if max_chars < offset <= after_limit]
    if after:
        return _finish_report_fragment(source[: min(after)])

    fallback = truncate_text(source, max_chars, suffix="")
    fallback = _trim_suspicious_summary_tail(fallback) or fallback
    fallback = fallback.rstrip(" .;,:")
    if " " in fallback:
        fallback = fallback[: fallback.rfind(" ")].rstrip(" .;,:")
    return _finish_report_fragment(fallback) if fallback else ""


def normalize_summary_snippet(text: Any, *, max_chars: Optional[int] = None) -> str:
    source = _normalize_ws(text)
    if max_chars is not None and max_chars > 0:
        source = _truncate_summary_snippet(source, max_chars)
    cleaned = _trim_suspicious_summary_tail(source)
    return _normalize_ws(cleaned or source).strip(" ;,:")


def truncate_heading_text(text: Any, max_chars: int) -> str:
    source = _normalize_ws(text)
    compact = _pack_heading_clauses(source, max_chars)
    if not compact:
        return ""
    words = compact.split()
    while len(words) > 3:
        tail = re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", words[-1], flags=re.UNICODE).casefold()
        if tail not in HEADING_TRAILING_STOPWORDS:
            break
        words.pop()
    trimmed = " ".join(words).rstrip(" ,;:/-–(")
    cleaned = _trim_suspicious_heading_tail(trimmed or compact)
    return cleaned or trimmed or compact


def has_terminal_ellipsis(text: Any) -> bool:
    return bool(ELLIPSIS_TRAILING_RE.search(_normalize_ws(text)))


def case_resume_key(case: CaseRecord) -> str:
    return _normalize_ws(case.page_id or case.case_id)


def analysis_resume_key(item: CaseAnalysis, fallback_key: str = "") -> str:
    return _normalize_ws(item.page_id or fallback_key or item.case_id)


def _read_secret_from_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def resolve_notion_key() -> str:
    for env_name in ("NOTION_API_KEY", "NOTION_TOKEN", "NOTION_SECRET"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    for candidate in notion_secret_candidates("Chave_Notion.txt"):
        value = _read_secret_from_file(candidate)
        if value:
            return value
    return ""


def resolve_openai_key() -> str:
    value = os.getenv("OPENAI_API_KEY", "").strip()
    if value:
        return value
    candidates = (
        KEYS_DIR / "CHAVE_SECRETA_API_Mauricio_local.txt",
        KEYS_DIR / "Chave Secreta API_Mauricio_local.txt",
        SCRIPT_DIR / "CHAVE_SECRETA_API_Mauricio_local.txt",
        SCRIPT_DIR / "Chave Secreta API_Mauricio_local.txt",
        Path.cwd() / "CHAVE_SECRETA_API_Mauricio_local.txt",
        Path.cwd() / "Chave Secreta API_Mauricio_local.txt",
    )
    for candidate in candidates:
        value = _read_secret_from_file(candidate)
        if value:
            return value
    return ""


def _normalize_notion_id(raw: str) -> str:
    text = re.sub(r"[^0-9A-Fa-f]", "", str(raw or ""))
    if len(text) != 32:
        raise ValueError(f"ID Notion invalido: {raw}")
    return str(uuid.UUID(text))


def extract_notion_id_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme and not parsed.netloc:
        return _normalize_notion_id(url)

    path_candidates = re.findall(r"[0-9A-Fa-f]{32}", parsed.path or "")
    if path_candidates:
        return _normalize_notion_id(path_candidates[-1])

    for _, values in parse_qs(parsed.query or "").items():
        for value in values:
            found = re.findall(r"[0-9A-Fa-f]{32}", value or "")
            if found:
                return _normalize_notion_id(found[-1])

    found_any = re.findall(r"[0-9A-Fa-f]{32}", url or "")
    if found_any:
        return _normalize_notion_id(found_any[-1])
    raise ValueError(f"Não foi possível extrair ID Notion da URL: {url}")


def _notion_page_url_from_id(page_id: str) -> str:
    return f"https://www.notion.so/{re.sub(r'-', '', page_id)}"


def _extract_http_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("error") or ""
            return _normalize_ws(message)
    except Exception:
        pass
    return _normalize_ws(response.text)


def _retry_delay_s(attempt: int, retry_after: Optional[str] = None) -> float:
    if retry_after:
        try:
            value = float(retry_after)
            if value > 0:
                return min(20.0, value)
        except Exception:
            pass
    return min(20.0, 0.8 * (2 ** max(0, attempt - 1)))


def _heartbeat_interval_s() -> float:
    return VERBOSE_HEARTBEAT_INTERVAL_S if LOGGER.isEnabledFor(logging.DEBUG) else DEFAULT_HEARTBEAT_INTERVAL_S


def _start_wait_heartbeat(label: str) -> tuple[threading.Event, threading.Thread, float]:
    stop_event = threading.Event()
    started_at = time.monotonic()
    interval = _heartbeat_interval_s()

    def _run() -> None:
        while not stop_event.wait(interval):
            elapsed = time.monotonic() - started_at
            LOGGER.info("%s em andamento... %.1fs decorrido.", label, elapsed)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return stop_event, thread, started_at


class RequestPacer:
    def __init__(self, target_rpm: int = 0) -> None:
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


def notion_request(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if NOTION_CFG is None or NOTION_SESSION is None:
        raise RuntimeError("Cliente Notion não inicializado.")

    url = path if path.startswith("http") else f"{NOTION_BASE_URL}{path}"
    max_attempts = max(1, int(NOTION_CFG.max_retries) + 1)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        request_label = f"Notion {method.upper()} {path}"
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("%s | tentativa %d/%d", request_label, attempt, max_attempts)
        stop_event, hb_thread, started_at = _start_wait_heartbeat(request_label)
        try:
            response = NOTION_SESSION.request(
                method=method.upper(),
                url=url,
                json=json_body,
                params=params,
                timeout=max(5, int(NOTION_CFG.timeout_s)),
            )
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "%s | tentativa %d/%d concluida em %.2fs | status=%s",
                    request_label,
                    attempt,
                    max_attempts,
                    time.monotonic() - started_at,
                    response.status_code,
                )
        except requests.RequestException as exc:
            last_error = str(exc)
            LOGGER.warning("Notion %s %s falhou na tentativa %d/%d: %s", method.upper(), path, attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(_retry_delay_s(attempt))
                continue
            raise RuntimeError(f"Falha de rede no Notion: {last_error}") from exc
        finally:
            stop_event.set()
            hb_thread.join(timeout=0.2)

        if 200 <= response.status_code < 300:
            if not response.text.strip():
                return {}
            try:
                payload = response.json()
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}

        message = _extract_http_message(response)
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < max_attempts:
            delay = _retry_delay_s(attempt, response.headers.get("Retry-After"))
            LOGGER.warning("Notion retornou %s. Retry em %.2fs. msg=%s", response.status_code, delay, message)
            time.sleep(delay)
            continue
        raise RuntimeError(f"Erro Notion {response.status_code} em {method.upper()} {path}: {message or '<sem mensagem>'}")
    raise RuntimeError(f"Falha no Notion: {last_error or 'erro desconhecido'}")


def _parse_openai_content(content_obj: Any) -> str:
    if isinstance(content_obj, str):
        return content_obj.strip()
    if isinstance(content_obj, list):
        parts: List[str] = []
        for item in content_obj:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return _normalize_ws(content_obj)


def _extract_openai_choice_text(raw: Any) -> tuple[str, str]:
    choice = (raw.get("choices") or [{}])[0] if isinstance(raw, dict) else {}
    if not isinstance(choice, dict):
        return "", ""
    message_obj = choice.get("message", {}) or {}
    finish_reason = _normalize_ws(choice.get("finish_reason", ""))
    content_obj = (message_obj if isinstance(message_obj, dict) else {}).get("content", "")
    return finish_reason, _parse_openai_content(content_obj)


def openai_chat_request(
    messages: Sequence[Dict[str, Any]],
    *,
    schema: Optional[Dict[str, Any]] = None,
    request_label: str = "OpenAI chat.completions",
    max_completion_tokens: int = 0,
) -> Dict[str, Any]:
    if OPENAI_CFG is None or OPENAI_SESSION is None or OPENAI_PACER is None:
        raise RuntimeError("Cliente OpenAI não inicializado.")

    payload: Dict[str, Any] = {
        "model": OPENAI_CFG.model,
        "messages": list(messages),
    }
    if max_completion_tokens > 0:
        payload["max_completion_tokens"] = int(max_completion_tokens)
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "relatorio_estrategico_dje",
                "strict": True,
                "schema": schema,
            },
        }

    max_attempts = max(1, int(OPENAI_CFG.retries) + 1)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        OPENAI_PACER.wait_turn()
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("%s | tentativa %d/%d | max_completion_tokens=%s", request_label, attempt, max_attempts, max_completion_tokens or "-")
        stop_event, hb_thread, started_at = _start_wait_heartbeat(request_label)
        try:
            response = OPENAI_SESSION.post(
                f"{OPENAI_BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=max(10, int(OPENAI_CFG.timeout_s)),
            )
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "%s | tentativa %d/%d concluida em %.2fs | status=%s",
                    request_label,
                    attempt,
                    max_attempts,
                    time.monotonic() - started_at,
                    response.status_code,
                )
        except requests.RequestException as exc:
            last_error = str(exc)
            LOGGER.warning("%s falhou na tentativa %d/%d: %s", request_label, attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(_retry_delay_s(attempt))
                continue
            raise RuntimeError(f"Falha de rede na OpenAI: {last_error}") from exc
        finally:
            stop_event.set()
            hb_thread.join(timeout=0.2)

        if 200 <= response.status_code < 300:
            payload_out = response.json()
            return payload_out if isinstance(payload_out, dict) else {}

        message = _extract_http_message(response)
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < max_attempts:
            delay = _retry_delay_s(attempt, response.headers.get("Retry-After"))
            LOGGER.warning("OpenAI retornou %s. Retry em %.2fs. msg=%s", response.status_code, delay, message)
            time.sleep(delay)
            continue
        raise RuntimeError(f"Erro OpenAI {response.status_code}: {message or '<sem mensagem>'}")
    raise RuntimeError(f"Falha na OpenAI: {last_error or 'erro desconhecido'}")


def openai_json_call(
    system_prompt: str,
    user_payload: Dict[str, Any],
    *,
    schema: Dict[str, Any],
    request_label: str,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    response = openai_chat_request(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ],
        schema=schema,
        request_label=request_label,
        max_completion_tokens=max_completion_tokens,
    )
    finish_reason, content = _extract_openai_choice_text(response)
    if not content:
        raise RuntimeError(f"{request_label} retornou resposta vazia. finish_reason={finish_reason or '<vazio>'}")
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        raise RuntimeError(f"{request_label} retornou JSON invalido: {exc}") from exc
    raise RuntimeError(f"{request_label} retornou payload inesperado.")


def parse_user_date(raw: str) -> date:
    text = _normalize_ws(raw)
    if not text:
        raise ValueError("Informe uma data.")
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Data invalida: {raw}. Use DD/MM/AAAA ou AAAA-MM-DD.")


def format_date_br(value: str | date) -> str:
    if isinstance(value, str):
        return parse_user_date(value).strftime("%d/%m/%Y")
    return value.strftime("%d/%m/%Y")


def format_period_br(start_iso: str, end_iso: str) -> str:
    start_br = format_date_br(start_iso)
    end_br = format_date_br(end_iso)
    return start_br if start_iso == end_iso else f"{start_br} a {end_br}"


def build_run_key(
    inputs: RunInputs,
    *,
    model: str,
    max_cases_per_batch: int,
    max_openai_cases: int = DEFAULT_MAX_OPENAI_CASES,
    openai_triage_threshold: int = DEFAULT_OPENAI_TRIAGE_THRESHOLD,
) -> str:
    material = {
        "source_database_url": _normalize_ws(inputs.source_database_url),
        "start_date_iso": _normalize_ws(inputs.start_date_iso),
        "end_date_iso": _normalize_ws(inputs.end_date_iso),
        "model": _normalize_ws(model),
        "max_cases_per_batch": int(max_cases_per_batch),
        "max_openai_cases": int(max_openai_cases),
        "openai_triage_threshold": int(openai_triage_threshold),
    }
    return json.dumps(material, ensure_ascii=False, sort_keys=True)


def load_last_inputs() -> Dict[str, Any]:
    payload = read_json_dict(REPORT_FILE)
    return payload.get("last_inputs", {}) if isinstance(payload.get("last_inputs"), dict) else {}


def load_matching_checkpoint(run_key: str, *, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {}
    payload = read_json_dict(CHECKPOINT_FILE)
    if payload.get("run_key") != run_key:
        return {}
    return payload


def _saved_inputs_match(saved_inputs: Dict[str, Any], inputs: RunInputs) -> bool:
    if not isinstance(saved_inputs, dict):
        return False
    return all(
        [
            _normalize_ws(saved_inputs.get("source_database_url")) == _normalize_ws(inputs.source_database_url),
            _normalize_ws(saved_inputs.get("start_date_iso")) == _normalize_ws(inputs.start_date_iso),
            _normalize_ws(saved_inputs.get("end_date_iso")) == _normalize_ws(inputs.end_date_iso),
        ]
    )


def load_matching_report_payload(inputs: RunInputs, *, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {}
    payload = read_json_dict(REPORT_FILE)
    if not _saved_inputs_match(payload.get("last_inputs", {}), inputs):
        return {}
    return payload


def save_checkpoint(run_key: str, payload: Dict[str, Any]) -> None:
    data = dict(payload)
    data["run_key"] = run_key
    data["updated_at_utc"] = utc_now_iso()
    write_json_atomic(CHECKPOINT_FILE, data)


def case_record_from_dict(payload: Dict[str, Any]) -> CaseRecord:
    allowed_fields = set(CaseRecord.__dataclass_fields__.keys())
    return CaseRecord(**{key: value for key, value in dict(payload).items() if key in allowed_fields})


def case_analysis_from_dict(payload: Dict[str, Any]) -> CaseAnalysis:
    allowed_fields = set(CaseAnalysis.__dataclass_fields__.keys())
    data = {key: value for key, value in dict(payload).items() if key in allowed_fields}
    data.setdefault("case_id", "")
    data.setdefault("title", "")
    data.setdefault("relevance_score", 1)
    data.setdefault("risk_level", "baixo")
    data.setdefault("includes_public_figure", False)
    data.setdefault("includes_party", False)
    data.setdefault("public_figures", [])
    data.setdefault("parties", [])
    data.setdefault("lawyers_signal", "")
    data.setdefault("what_happened", "")
    data.setdefault("legal_grounds", "")
    data.setdefault("consequence", "")
    data.setdefault("strategic_comment", "")
    data.setdefault("why_relevant", "")
    data.setdefault("source_notes", [])
    data.setdefault("page_id", "")
    if not data.get("display_score"):
        data["display_score"] = data.get("relevance_score", 0)
    data.setdefault("ranking_reason", "")
    for field_name in ("public_figures", "parties", "source_notes"):
        value = data.get(field_name)
        if isinstance(value, str):
            data[field_name] = _unique_preserve_order(re.split(r"\s*[,;]\s*", value))
        elif isinstance(value, Iterable):
            data[field_name] = _unique_preserve_order(str(item or "") for item in value)
        else:
            data[field_name] = []
    for field_name in ("relevance_score", "display_score"):
        try:
            data[field_name] = max(1, min(10, int(data.get(field_name) or 1)))
        except Exception:
            data[field_name] = 1
    data["includes_public_figure"] = bool(data.get("includes_public_figure"))
    data["includes_party"] = bool(data.get("includes_party"))
    data["title"] = truncate_heading_text(data.get("title") or "", HEADING_SUMMARY_MAX_CHARS)
    for field_name in ("what_happened", "legal_grounds", "consequence", "strategic_comment", "why_relevant", "lawyers_signal"):
        data[field_name] = _normalize_ws(data.get(field_name))
    return CaseAnalysis(**data)


def case_analysis_content_hash(case: CaseRecord) -> str:
    material = {
        "processo": _normalize_ws(case.numero_unico or case.numero_processo),
        "data_decisao": _normalize_ws(case.data_decisao),
        "classe": _normalize_ws(case.sigla_classe or case.descricao_classe),
        "uf": _normalize_ws(case.sigla_uf),
        "municipio": _normalize_ws(case.nome_municipio),
        "assuntos": [_normalize_ws(item) for item in case.assuntos],
        "partes": [_normalize_ws(item) for item in case.partes],
        "partidos": [_normalize_ws(item) for item in case.partidos],
        "resultado": [_normalize_ws(item) for item in case.resultado],
        "tema": _normalize_ws(case.tema),
        "punchline": _normalize_ws(case.punchline),
        "texto_decisao": _normalize_ws(case.texto_decisao)[:12000],
    }
    raw = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def case_analysis_cache_keys(case: CaseRecord) -> List[str]:
    keys: List[str] = []
    for prefix, value in (
        ("numero_unico", case.numero_unico),
        ("numero_processo", case.numero_processo),
        ("page_id", case.page_id),
        ("case_id", case.case_id),
    ):
        text = _normalize_ws(value)
        if text:
            keys.append(f"{prefix}:{text.casefold()}")
    keys.append(f"content_hash:{case_analysis_content_hash(case)}")
    return _unique_preserve_order(keys)


def load_case_analysis_cache(*, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {"version": 1, "items": {}}
    payload = read_json_dict(ANALYSIS_CACHE_FILE)
    if not isinstance(payload.get("items"), dict):
        payload["items"] = {}
    payload.setdefault("version", 1)
    return payload


def save_case_analysis_cache(cache_payload: Dict[str, Any]) -> None:
    cache_payload["updated_at_utc"] = utc_now_iso()
    write_json_atomic(ANALYSIS_CACHE_FILE, cache_payload)


def cached_case_analysis_for_case(
    case: CaseRecord,
    cache_payload: Dict[str, Any],
    *,
    model: str,
) -> Optional[CaseAnalysis]:
    items = cache_payload.get("items")
    if not isinstance(items, dict):
        return None
    current_hash = case_analysis_content_hash(case)
    current_model = _normalize_ws(model)
    for key in case_analysis_cache_keys(case):
        entry = items.get(key)
        if not isinstance(entry, dict):
            continue
        cached_hash = _normalize_ws(entry.get("content_hash"))
        if cached_hash and cached_hash != current_hash:
            continue
        cached_model = _normalize_ws(entry.get("model"))
        if cached_model and current_model and cached_model != current_model:
            continue
        analysis_payload = entry.get("analysis") if isinstance(entry.get("analysis"), dict) else entry
        if not isinstance(analysis_payload, dict):
            continue
        try:
            analysis = case_analysis_from_dict(analysis_payload)
        except Exception as exc:
            LOGGER.debug("Ignorando cache de análise inválido para %s: %s", case.process_label(), exc)
            continue
        analysis.case_id = case.case_id
        analysis.page_id = case.page_id
        return analysis
    return None


def store_case_analysis_cache_entries(
    cache_payload: Dict[str, Any],
    analyses: Sequence[CaseAnalysis],
    case_by_analysis_key: Mapping[str, CaseRecord],
    *,
    model: str,
) -> int:
    items = cache_payload.setdefault("items", {})
    if not isinstance(items, dict):
        items = {}
        cache_payload["items"] = items
    stored = 0
    for analysis in analyses:
        case = case_by_analysis_key.get(analysis_resume_key(analysis)) or case_by_analysis_key.get(_normalize_ws(analysis.case_id))
        if case is None:
            continue
        entry = {
            "model": _normalize_ws(model),
            "case_id": case.case_id,
            "page_id": case.page_id,
            "processo": case.process_label(),
            "content_hash": case_analysis_content_hash(case),
            "cached_at_utc": utc_now_iso(),
            "analysis": asdict(analysis),
        }
        for key in case_analysis_cache_keys(case):
            items[key] = entry
        stored += 1
    return stored


def report_summary_from_dict(payload: Dict[str, Any]) -> ReportSummary:
    return ReportSummary(**payload)


def build_resume_analysis_map(
    cases: Sequence[CaseRecord],
    checkpoint: Dict[str, Any],
    report_payload: Dict[str, Any],
    *,
    refresh_analysis: bool,
    refresh_ellipsis_only: bool,
) -> tuple[Dict[str, CaseAnalysis], int]:
    if refresh_analysis:
        return {}, 0

    checkpoint_list = checkpoint.get("analyses")
    report_list = report_payload.get("analyses")
    checkpoint_map = checkpoint.get("analyses_by_case_id", {})
    out: Dict[str, CaseAnalysis] = {}
    skipped_for_ellipsis = 0

    for idx, case in enumerate(cases):
        raw_payload: Optional[Dict[str, Any]] = None

        if isinstance(checkpoint_list, list) and len(checkpoint_list) == len(cases) and isinstance(checkpoint_list[idx], dict):
            raw_payload = dict(checkpoint_list[idx])
        elif isinstance(report_list, list) and len(report_list) == len(cases) and isinstance(report_list[idx], dict):
            raw_payload = dict(report_list[idx])
        elif isinstance(checkpoint_map, dict):
            for lookup_key in _unique_preserve_order([case_resume_key(case), case.page_id, case.case_id]):
                candidate = checkpoint_map.get(lookup_key)
                if isinstance(candidate, dict):
                    raw_payload = dict(candidate)
                    break

        if not raw_payload:
            continue

        raw_payload.setdefault("case_id", case.case_id)
        raw_payload.setdefault("page_id", case.page_id)
        try:
            analysis = case_analysis_from_dict(raw_payload)
        except Exception:
            continue

        if refresh_ellipsis_only and analysis_has_visible_ellipsis(analysis):
            skipped_for_ellipsis += 1
            continue

        out[case_resume_key(case)] = analysis

    return out, skipped_for_ellipsis


def run_gui(initials: Dict[str, Any]) -> Optional[RunInputs]:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception:
        LOGGER.warning("Tkinter indisponivel neste ambiente.")
        return None

    result: Dict[str, Any] = {}
    try:
        root = tk.Tk()
    except Exception as exc:
        LOGGER.warning("Falha ao abrir GUI Tkinter: %s", exc)
        return None

    root.title("Relatório Estratégico DJe - Notion")
    root.geometry("880x330")
    root.minsize(820, 300)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    ttk.Label(frame, text="Página-alvo do Notion:").grid(row=0, column=0, sticky="w", pady=(0, 8))
    page_var = tk.StringVar(value="")
    ttk.Entry(frame, textvariable=page_var).grid(row=0, column=1, sticky="ew", pady=(0, 8))

    ttk.Label(frame, text="Base do Notion:").grid(row=1, column=0, sticky="w", pady=(0, 8))
    db_var = tk.StringVar(value=str(initials.get("source_database_url", DEFAULT_SOURCE_DATABASE_URL)))
    ttk.Entry(frame, textvariable=db_var).grid(row=1, column=1, sticky="ew", pady=(0, 8))

    ttk.Label(frame, text="Data inicial:").grid(row=2, column=0, sticky="w", pady=(0, 8))
    start_var = tk.StringVar(value=str(initials.get("start_date_iso", "")))
    ttk.Entry(frame, textvariable=start_var).grid(row=2, column=1, sticky="ew", pady=(0, 8))

    ttk.Label(frame, text="Data final:").grid(row=3, column=0, sticky="w", pady=(0, 8))
    end_var = tk.StringVar(value=str(initials.get("end_date_iso", "")))
    ttk.Entry(frame, textvariable=end_var).grid(row=3, column=1, sticky="ew", pady=(0, 8))

    help_text = (
        "Use DD/MM/AAAA ou AAAA-MM-DD. A data final pode repetir a inicial para gerar o relatório de um único dia. "
        "A integração do Notion precisa ter acesso à página-alvo e à base."
    )
    ttk.Label(frame, text=help_text, wraplength=820, justify="left").grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 12))

    force_clear_var = tk.BooleanVar(value=bool(initials.get("force_clear", False)))
    ttk.Checkbutton(
        frame,
        text="Limpar a página caso já exista conteúdo (destrutivo)",
        variable=force_clear_var,
    ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 12))

    def _on_run() -> None:
        try:
            page_url = _normalize_ws(page_var.get())
            db_url = _normalize_ws(db_var.get())
            start_date = parse_user_date(start_var.get()).isoformat()
            end_raw = _normalize_ws(end_var.get()) or _normalize_ws(start_var.get())
            end_date = parse_user_date(end_raw).isoformat()
            if not page_url:
                raise ValueError("Informe a página-alvo do Notion.")
            extract_notion_id_from_url(page_url)
            extract_notion_id_from_url(db_url)
            if start_date > end_date:
                raise ValueError("A data inicial não pode ser posterior à data final.")
        except Exception as exc:
            messagebox.showerror("Dados inválidos", str(exc))
            return
        result["value"] = RunInputs(
            page_url=page_url,
            source_database_url=db_url,
            start_date_iso=start_date,
            end_date_iso=end_date,
            force_clear=bool(force_clear_var.get()),
            dry_run=False,
        )
        root.destroy()

    def _on_cancel() -> None:
        root.destroy()

    button_row = ttk.Frame(frame)
    button_row.grid(row=6, column=0, columnspan=2, sticky="ew")
    ttk.Button(button_row, text="Cancelar", command=_on_cancel).pack(side="right")
    ttk.Button(button_row, text="Executar", command=_on_run).pack(side="right", padx=(0, 8))

    root.bind("<Return>", lambda _event: _on_run())
    root.bind("<Escape>", lambda _event: _on_cancel())
    root.mainloop()
    return result.get("value")


def _rich_text_plain_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    out: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        plain = item.get("plain_text")
        if isinstance(plain, str):
            out.append(plain)
            continue
        text_obj = item.get("text")
        if isinstance(text_obj, dict):
            content = text_obj.get("content")
            if isinstance(content, str):
                out.append(content)
    return "".join(out).strip()


def _property_title(prop: Dict[str, Any]) -> str:
    return _rich_text_plain_text(prop.get("title"))


def _property_rich_text(prop: Dict[str, Any]) -> str:
    return _rich_text_plain_text(prop.get("rich_text"))


def _property_select(prop: Dict[str, Any]) -> str:
    obj = prop.get("select")
    return _normalize_ws(obj.get("name")) if isinstance(obj, dict) else ""


def _property_multi_select(prop: Dict[str, Any]) -> List[str]:
    raw = prop.get("multi_select")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, dict):
            value = _normalize_ws(item.get("name"))
            if value:
                out.append(value)
    return out


def _split_text_list_property(value: str) -> List[str]:
    raw = _normalize_ws(value)
    if not raw:
        return []
    try:
        import SJUR_csv_to_csv_NOTIONfriendly_v2 as sjur

        return [_normalize_ws(item) for item in sjur.split_multiselect_values(raw) if _normalize_ws(item)]
    except Exception:
        return [_normalize_ws(item) for item in re.split(r"\s*[,;]\s*", raw) if _normalize_ws(item)]


def _property_multi_select_or_text_list(prop: Dict[str, Any]) -> List[str]:
    return _unique_preserve_order(
        [
            *_property_multi_select(prop),
            *_split_text_list_property(_property_rich_text(prop)),
        ]
    )


def _property_url(prop: Dict[str, Any]) -> str:
    return _normalize_ws(prop.get("url"))


NEWS_URL_COLUMNS = ("noticia_TSE", "noticia_TRE", "noticia_geral_1", "noticia_geral_2")


def _header_metadata_from_case_texts(*texts: str) -> Dict[str, List[str]]:
    source_texts = [text for text in texts if _normalize_ws(text)]
    if not source_texts:
        return {"partes": [], "advogados": []}
    try:
        import SJUR_csv_to_csv_NOTIONfriendly_v2 as sjur

        config = sjur.MetadataExtractionConfig(
            include_institutional_entities=True,
            header_max_chars=getattr(sjur, "DEFAULT_METADATA_HEADER_MAX_CHARS", 1200),
        )
        metadata = sjur.extract_header_metadata(*source_texts, config=config)
    except Exception:
        return {"partes": [], "advogados": []}
    if not isinstance(metadata, dict):
        return {"partes": [], "advogados": []}
    return {
        "partes": [_normalize_ws(item) for item in metadata.get("partes", []) if _normalize_ws(item)],
        "advogados": [_normalize_ws(item) for item in metadata.get("advogados", []) if _normalize_ws(item)],
    }


def _property_date_start(prop: Dict[str, Any]) -> str:
    value = prop.get("date")
    return _normalize_ws(value.get("start")) if isinstance(value, dict) else ""


def _property_number_text(prop: Dict[str, Any]) -> str:
    value = prop.get("number")
    if value is None:
        return ""
    try:
        return str(int(value))
    except Exception:
        return _normalize_ws(value)


def _infer_party_scope(entry_upper: str, case_uf: str) -> str:
    if "NACIONAL" in entry_upper:
        return "Nacional"
    return case_uf or ""


def infer_parties_from_entries(entries: Sequence[str], case_uf: str) -> List[str]:
    parties: List[str] = []
    for raw_entry in entries:
        entry = _normalize_ws(raw_entry)
        if not entry:
            continue
        entry_upper = entry.upper()
        sigla = ""
        match = PARTY_PAREN_RE.search(entry_upper)
        if match:
            sigla = _normalize_ws(match.group(1))
        if not sigla:
            token_match = PARTY_TOKEN_RE.search(entry_upper)
            if token_match:
                sigla = _normalize_ws(token_match.group(1).upper())
        if not sigla and entry_upper.startswith("PARTIDO "):
            tail = entry_upper.replace("PARTIDO ", "", 1).strip()
            sigla = tail if len(tail) <= 20 else ""
        if not sigla:
            continue
        scope = _infer_party_scope(entry_upper, case_uf)
        label = sigla if not scope else f"{sigla}/{scope}"
        parties.append(label)
    return _unique_preserve_order(parties)


def build_case_record(page_obj: Dict[str, Any]) -> CaseRecord:
    props = page_obj.get("properties") or {}
    page_id = _normalize_notion_id(str(page_obj.get("id", "")))
    partes = _unique_preserve_order(
        [
            *_property_multi_select_or_text_list(props.get("partes", {})),
            *_split_text_list_property(_property_rich_text(props.get("partes_texto", {}))),
        ]
    )
    sigla_uf = _property_select(props.get("siglaUF", {}))
    texto_decisao = _property_rich_text(props.get("textoDecisao", {}))
    texto_ementa = _property_rich_text(props.get("textoEmenta", {}))
    parsed_header_metadata = _header_metadata_from_case_texts(texto_decisao, texto_ementa)
    partes = _unique_preserve_order([*partes, *parsed_header_metadata.get("partes", [])])
    advogados = _unique_preserve_order(
        [
            *_property_multi_select_or_text_list(props.get("advogados", {})),
            *_split_text_list_property(_property_rich_text(props.get("advogados_texto", {}))),
            *parsed_header_metadata.get("advogados", []),
        ]
    )
    noticias = _unique_preserve_order(
        [
            _property_url(props.get("noticia_TSE", {})),
            _property_url(props.get("noticia_TRE", {})),
            _property_url(props.get("noticia_geral_1", {})),
            _property_url(props.get("noticia_geral_2", {})),
        ]
    )
    numero_unico = _property_rich_text(props.get("numeroUnico", {}))
    numero_processo = _property_number_text(props.get("numeroProcesso", {}))
    case_id = numero_unico or numero_processo or page_id
    return CaseRecord(
        case_id=case_id,
        page_id=page_id,
        source_url=_notion_page_url_from_id(page_id),
        data_decisao=_property_date_start(props.get("dataDecisao", {})),
        numero_unico=numero_unico,
        numero_processo=numero_processo,
        ano_eleicao=_property_select(props.get("anoEleicao", {})),
        sigla_classe=_property_select(props.get("siglaClasse", {})),
        descricao_classe=_property_select(props.get("descricaoClasse", {})),
        sigla_uf=sigla_uf,
        nome_municipio=_property_select(props.get("nomeMunicipio", {})),
        descricao_tipo_decisao=_property_select(props.get("descricaoTipoDecisao", {})),
        assuntos=_property_multi_select(props.get("assuntos", {})),
        partes=partes,
        partidos=infer_parties_from_entries(partes, sigla_uf),
        relator=_property_select(props.get("relator", {})),
        advogados=advogados,
        resultado=_property_multi_select(props.get("resultado", {})),
        tema=_property_title(props.get("tema", {})),
        punchline=_property_rich_text(props.get("punchline", {})),
        texto_decisao=texto_decisao,
        noticias=noticias,
        dep_federal=bool(((props.get("Dep. Federal") or {}).get("checkbox"))),
    )


def retrieve_database_and_datasource_id(database_id: str) -> str:
    payload = notion_request("GET", f"/v1/databases/{database_id}")
    raw_data_sources = payload.get("data_sources")
    candidates: List[str] = []

    if isinstance(raw_data_sources, list):
        for item in raw_data_sources:
            if isinstance(item, dict) and item.get("id"):
                try:
                    candidates.append(_normalize_notion_id(str(item["id"])))
                except Exception:
                    continue
    elif isinstance(raw_data_sources, dict):
        for key in ("results", "items", "data_sources"):
            value = raw_data_sources.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("id"):
                        try:
                            candidates.append(_normalize_notion_id(str(item["id"])))
                        except Exception:
                            continue

    if candidates:
        return candidates[0]

    try:
        fallback = notion_request("GET", f"/v1/data_sources/{database_id}")
        candidate = fallback.get("id")
        if isinstance(candidate, str) and candidate:
            return _normalize_notion_id(candidate)
    except Exception:
        pass

    raise RuntimeError(
        "Não foi possível identificar data_source_id no database informado. "
        "Confirme a versão da API e o compartilhamento da integração."
    )


def query_cases_by_period(data_source_id: str, start_iso: str, end_iso: str) -> List[CaseRecord]:
    results: List[CaseRecord] = []
    cursor = ""
    page_loops = 0

    while True:
        page_loops += 1
        body: Dict[str, Any] = {
            "page_size": 100,
            "filter": {
                "and": [
                    {"property": "dataDecisao", "date": {"on_or_after": start_iso}},
                    {"property": "dataDecisao", "date": {"on_or_before": end_iso}},
                ]
            },
            "sorts": [
                {"property": "dataDecisao", "direction": "ascending"},
                {"property": "tema", "direction": "ascending"},
            ],
        }
        if cursor:
            body["start_cursor"] = cursor
        payload = notion_request("POST", f"/v1/data_sources/{data_source_id}/query", json_body=body)
        items = payload.get("results", [])
        if not isinstance(items, list):
            items = []

        LOGGER.info("Query página %d recebida | itens=%d | has_more=%s", page_loops, len(items), bool(payload.get("has_more")))
        for item in items:
            if not isinstance(item, dict) or str(item.get("object")) != "page":
                continue
            try:
                results.append(build_case_record(item))
            except Exception as exc:
                LOGGER.warning("Falha ao normalizar página %s: %s", item.get("id"), exc)

        if not bool(payload.get("has_more")):
            break
        cursor = _normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return results


def risk_from_score(score: int) -> str:
    if score >= 9:
        return "critico"
    if score >= 7:
        return "alto"
    if score >= 5:
        return "medio"
    return "baixo"


def risk_display_label(risk_level: str) -> str:
    return RISK_DISPLAY_LABELS.get(_normalize_ws(risk_level).casefold(), _normalize_ws(risk_level).upper() or "BAIXO")


RISK_EMOJIS = {"critico": "🔴", "alto": "🟠", "medio": "🟡", "baixo": "🟢"}


def risk_emoji(risk_level: str) -> str:
    return RISK_EMOJIS.get(_normalize_ws(risk_level).casefold(), "⚪")


def analysis_display_score(item: CaseAnalysis) -> int:
    raw = item.display_score or item.relevance_score
    return max(1, min(10, int(raw)))


def analysis_sort_key(item: CaseAnalysis) -> tuple[int, int, str]:
    risk = _normalize_ws(item.risk_level).casefold()
    return (
        RISK_SORT_ORDER.get(risk, 99),
        -max(1, min(10, int(item.relevance_score))),
        item.title.casefold(),
    )


def is_reportable_analysis(item: CaseAnalysis) -> bool:
    return _normalize_ws(item.risk_level).casefold() in REPORTABLE_RISK_LEVELS


def is_publishable_analysis(item: CaseAnalysis) -> bool:
    return _normalize_ws(item.risk_level).casefold() in PUBLISHED_RISK_LEVELS


def _publication_signal_text(case: CaseRecord, analysis: CaseAnalysis) -> str:
    return _safe_join(
        [
            _safe_join(analysis.public_figures),
            _safe_join(analysis.parties),
            _safe_join(case.partidos),
            _safe_join(case.partes),
            case.tema,
            case.punchline,
            analysis.what_happened,
            analysis.consequence,
        ],
        sep=" | ",
    )


def _strategic_focus_signal_text(case: CaseRecord, analysis: CaseAnalysis) -> str:
    return _safe_join(
        [
            _publication_signal_text(case, analysis),
            case.ano_eleicao,
            case.sigla_classe,
            case.descricao_classe,
            case.descricao_tipo_decisao,
            _safe_join(case.assuntos),
            analysis.legal_grounds,
            analysis.strategic_comment,
            analysis.why_relevant,
        ],
        sep=" | ",
    )


def compute_political_prominence_score(case: CaseRecord, analysis: CaseAnalysis) -> int:
    score = 0
    signal_text = _publication_signal_text(case, analysis)
    if analysis.includes_public_figure or bool(analysis.public_figures):
        score += 40
    if EXECUTIVE_MAJOR_ROLE_RE.search(signal_text):
        score += 25
    if LEGISLATIVE_ROLE_RE.search(signal_text):
        score += 15
    parties = _unique_preserve_order(list(analysis.parties) + list(case.partidos))
    if analysis.includes_party or bool(parties):
        score += 20
    if len(_unique_preserve_order(list(analysis.public_figures))) > 1 or len(parties) > 1:
        score += 10
    if (analysis.includes_public_figure or bool(analysis.public_figures)) and (analysis.includes_party or bool(parties)):
        score += 8
    return score


def compute_federal_interest_score(case: CaseRecord, analysis: CaseAnalysis) -> int:
    signal_text = _strategic_focus_signal_text(case, analysis)
    parties = _unique_preserve_order(list(analysis.parties) + list(case.partidos))
    score = 0
    if FEDERAL_HIGH_OFFICE_RE.search(signal_text):
        score += 45
    if NATIONAL_PARTY_SCOPE_RE.search(signal_text) or any(_normalize_ws(party).casefold().endswith("/nacional") for party in parties):
        score += 45
    if PARTY_INSTITUTIONAL_RE.search(signal_text):
        score += 28
    if PARTY_FINANCE_SYSTEMIC_RE.search(signal_text):
        score += 24
    if parties:
        score += 12
    if _normalize_ws(case.ano_eleicao) in {"2018", "2022", "2026"}:
        score += 8
    if HIGH_IMPACT_ELECTORAL_RE.search(signal_text) and (parties or FEDERAL_HIGH_OFFICE_RE.search(signal_text)):
        score += 8
    if LOWER_LOCAL_OFFICE_RE.search(signal_text) and score < FEDERAL_INTEREST_MIN_SCORE:
        score -= 12
    return max(0, score)


def is_federal_strategic_interest(case: CaseRecord, analysis: CaseAnalysis) -> bool:
    if is_target_office_case(case, analysis):
        return True
    return compute_federal_interest_score(case, analysis) >= FEDERAL_INTEREST_MIN_SCORE


RISK_FLOOR_ORDER = {"baixo": 0, "medio": 1, "alto": 2, "critico": 3}
TARGET_OFFICE_MIN_SCORE = 8


def is_target_office_case(case: CaseRecord, analysis: CaseAnalysis) -> bool:
    # Curadoria manual tem precedencia absoluta.
    if case.dep_federal:
        return True
    # Inclui assuntos e classe: o cargo-alvo muitas vezes so aparece no assunto
    # taxonomico da base (ex.: "Cargo - Governador").
    signal = _safe_join(
        [
            _publication_signal_text(case, analysis),
            _safe_join(case.assuntos),
            case.descricao_classe,
        ],
        sep=" | ",
    )
    return bool(FEDERAL_HIGH_OFFICE_RE.search(signal))


def apply_target_office_floor(case: CaseRecord, analysis: CaseAnalysis) -> CaseAnalysis:
    # Rede determinística: processo com alto cargo da República (Presidente, Vice,
    # governadores, senadores, deputados federais, ministros de Estado) nunca fica
    # abaixo de 8/alto, mesmo quando o modelo subavalia (ex.: improcedência mantida).
    if analysis.public_figures and not analysis.includes_public_figure:
        analysis = replace(analysis, includes_public_figure=True)
    if not is_target_office_case(case, analysis):
        return analysis
    new_relevance = max(int(analysis.relevance_score or 0), TARGET_OFFICE_MIN_SCORE)
    new_display = max(int(analysis.display_score or analysis.relevance_score or 0), TARGET_OFFICE_MIN_SCORE)
    risk = _normalize_ws(analysis.risk_level).casefold()
    new_risk = analysis.risk_level if RISK_FLOOR_ORDER.get(risk, 0) >= RISK_FLOOR_ORDER["alto"] else "alto"
    if (new_relevance, new_display, new_risk) == (analysis.relevance_score, analysis.display_score, analysis.risk_level):
        return analysis
    return replace(analysis, relevance_score=new_relevance, display_score=new_display, risk_level=new_risk)


def sanitize_analysis_parties(case: CaseRecord, analysis: CaseAnalysis) -> CaseAnalysis:
    # O modelo as vezes coloca nomes de PESSOAS em 'parties'. Mantem apenas o que
    # o validador deterministico reconhece como legenda partidaria; na falta,
    # recua para os partidos extraidos da base (case.partidos).
    if not analysis.parties:
        return analysis
    valid = infer_parties_from_entries(analysis.parties, case.sigla_uf)
    if not valid:
        valid = list(case.partidos)
    if list(valid) == list(analysis.parties):
        return analysis
    return replace(
        analysis,
        parties=_unique_preserve_order(valid),
        includes_party=bool(valid) or bool(case.partidos),
    )


def finalize_case_analyses(
    cases: Sequence[CaseRecord],
    analysis_by_case_id: Mapping[str, CaseAnalysis],
) -> List[CaseAnalysis]:
    out: List[CaseAnalysis] = []
    for case in cases:
        key = case_resume_key(case)
        if key not in analysis_by_case_id:
            continue
        analysis = sanitize_analysis_parties(case, analysis_by_case_id[key])
        out.append(apply_target_office_floor(case, analysis))
    return out


def compute_institutional_party_scope_score(case: CaseRecord, analysis: CaseAnalysis) -> int:
    parties = _unique_preserve_order(list(analysis.parties) + list(case.partidos))
    if not parties:
        return 0
    best = 0
    for party in parties:
        value = _normalize_ws(party)
        lowered = value.casefold()
        if lowered.endswith("/nacional"):
            best = max(best, 12)
        elif re.search(r"/[A-Z]{2}$", value):
            best = max(best, 8)
        else:
            best = max(best, 4)
    return best


def compute_high_impact_signal_score(case: CaseRecord, analysis: CaseAnalysis) -> int:
    signal_text = _publication_signal_text(case, analysis)
    score = 0
    if HIGH_IMPACT_PUBLICATION_RE.search(signal_text):
        score += 20
    if HIGH_IMPACT_ELECTORAL_RE.search(signal_text):
        score += 12
    if HIGH_IMPACT_ACCOUNTS_RE.search(signal_text):
        score += 8
    return score


def publication_sort_key(case: CaseRecord, analysis: CaseAnalysis) -> tuple[int, int, int, int, int, str, str]:
    risk = _normalize_ws(analysis.risk_level).casefold()
    return (
        RISK_SORT_ORDER.get(risk, 99),
        -analysis_display_score(analysis),
        -compute_federal_interest_score(case, analysis),
        -compute_political_prominence_score(case, analysis),
        -compute_institutional_party_scope_score(case, analysis),
        -compute_high_impact_signal_score(case, analysis),
        _normalize_ws(analysis.title).casefold(),
        case.process_label().casefold(),
    )


def build_publishable_case_pairs(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[tuple[CaseRecord, CaseAnalysis]]:
    case_map = {case_resume_key(case): case for case in cases}
    pairs: List[tuple[CaseRecord, CaseAnalysis]] = []
    for analysis in analyses:
        if not is_publishable_analysis(analysis):
            continue
        case = case_map.get(analysis_resume_key(analysis))
        if case is None:
            continue
        if not is_federal_strategic_interest(case, analysis):
            continue
        pairs.append((case, analysis))
    pairs.sort(key=lambda item: publication_sort_key(item[0], item[1]))
    # Casos-alvo (altos cargos da Republica) nunca ficam fora dos destaques:
    # entram todos, e o teto so limita os demais.
    target_pairs = [pair for pair in pairs if is_target_office_case(pair[0], pair[1])]
    other_pairs = [pair for pair in pairs if pair not in target_pairs]
    limit = max(MAX_PUBLISHED_CASES, len(target_pairs))
    combined = (target_pairs + other_pairs)[:limit]
    combined.sort(key=lambda item: publication_sort_key(item[0], item[1]))
    return combined


def build_focused_reportable_case_pairs(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[tuple[CaseRecord, CaseAnalysis]]:
    case_map = {case_resume_key(case): case for case in cases}
    pairs: List[tuple[CaseRecord, CaseAnalysis]] = []
    for analysis in analyses:
        if not is_reportable_analysis(analysis):
            continue
        case = case_map.get(analysis_resume_key(analysis))
        if case is None:
            continue
        if not is_federal_strategic_interest(case, analysis):
            continue
        pairs.append((case, analysis))
    pairs.sort(key=lambda item: publication_sort_key(item[0], item[1]))
    return pairs


def load_delayed_cases(path: str, *, exclude_keys: Optional[set] = None) -> List[CaseRecord]:
    """Materializa, por page_id, as decisões de semanas anteriores recém-importadas.

    O arquivo é gerado pelo orquestrador a partir do relatório do import
    (lista de objetos {page_id, dataDecisao, numeroUnico} sob a chave "cases").
    Busca cada página pelo id (mesmo padrão de load_cases_from_published_summary_table)
    e tolera páginas apagadas/arquivadas. ``exclude_keys`` descarta casos que já
    entrariam pela query normal da semana (defesa contra dupla contagem)."""
    raw_path = _normalize_ws(path)
    if not raw_path:
        return []
    file_path = Path(raw_path)
    if not file_path.is_file():
        LOGGER.warning("Arquivo de casos atrasados não encontrado: %s", file_path)
        return []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Falha ao ler casos atrasados de %s: %s", file_path, exc)
        return []
    entries = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    exclude = set(exclude_keys or set())
    seen_ids: set = set()
    out: List[CaseRecord] = []
    for entry in entries:
        if isinstance(entry, dict):
            page_id = _normalize_notion_id(str(entry.get("page_id", "")))
        else:
            page_id = _normalize_notion_id(str(entry))
        if not page_id or page_id in seen_ids:
            continue
        seen_ids.add(page_id)
        try:
            case = build_case_record(notion_request("GET", f"/v1/pages/{page_id}"))
        except Exception as exc:
            LOGGER.debug("Caso atrasado %s indisponível (apagado/arquivado?): %s", page_id, exc)
            continue
        if case_resume_key(case) in exclude:
            continue
        out.append(case)
    return out


def build_delayed_target_pairs(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[tuple[CaseRecord, CaseAnalysis]]:
    """Pares (caso, análise) dos atrasados que merecem destaque: casos-alvo
    (altos cargos da República / checkbox Dep. Federal) OU risco crítico/alto.
    As análises já vêm com o piso de alvo aplicado (finalize_case_analyses,
    chamado dentro de analyze_cases)."""
    case_map = {case_resume_key(case): case for case in cases}
    pairs: List[tuple[CaseRecord, CaseAnalysis]] = []
    for analysis in analyses:
        case = case_map.get(analysis_resume_key(analysis))
        if case is None:
            continue
        risk = _normalize_ws(analysis.risk_level).casefold()
        if is_target_office_case(case, analysis) or risk in {"critico", "alto"}:
            pairs.append((case, analysis))
    pairs.sort(key=lambda item: publication_sort_key(item[0], item[1]))
    return pairs


def _news_url_row_from_case(case: CaseRecord) -> Dict[str, str]:
    try:
        import SJUR_csv_to_csv_NOTIONfriendly_v2 as sjur
    except Exception:
        sjur = None

    row = {column: "" for column in NEWS_URL_COLUMNS}
    general_idx = 1
    for raw_url in extract_http_urls(case.noticias):
        url = _normalize_ws(raw_url)
        if not url:
            continue
        if sjur is not None and sjur.is_editorially_weak_news_url(url):
            continue
        if sjur is not None:
            tse_url = sjur._normalize_tse_news_url(url)
            tre_url = sjur._normalize_tre_news_url(url)
        else:
            tse_url = url if "tse.jus.br" in url.casefold() else ""
            tre_url = url if "tre-" in url.casefold() and ".jus.br" in url.casefold() else ""
        if tse_url and not row["noticia_TSE"]:
            row["noticia_TSE"] = tse_url
            continue
        if tre_url and not row["noticia_TRE"]:
            row["noticia_TRE"] = tre_url
            continue
        if general_idx <= 2:
            row[f"noticia_geral_{general_idx}"] = url
            general_idx += 1
    return row


def _news_lookup_payload_from_case(case: CaseRecord) -> Dict[str, str]:
    return {
        "numero_unico": case.numero_unico or case.numero_processo or case.case_id,
        "data_decisao": case.data_decisao,
        "assuntos": _safe_join(case.assuntos, sep=", "),
        "partes": _safe_join(case.partes, sep=", "),
        "advogados": _safe_join(case.advogados, sep=", "),
        "relator": case.relator,
        "descricao_classe": case.descricao_classe or case.sigla_classe,
        "nome_tipo_processo": case.descricao_classe or case.sigla_classe,
        "texto_decisao": truncate_text(case.texto_decisao, 900, suffix=""),
        "texto_ementa": "",
        "sigla_uf": case.sigla_uf,
        "nome_municipio": case.nome_municipio,
        "tribunal": "TSE",
        "origem": _safe_join([case.sigla_uf, case.nome_municipio], sep=" "),
        "tema": case.tema,
        "punchline": case.punchline,
    }


def _patch_case_news_urls(case: CaseRecord, row: Mapping[str, str]) -> int:
    if not case.page_id:
        return 0
    properties: Dict[str, Any] = {}
    for column in NEWS_URL_COLUMNS:
        url = _normalize_ws(row.get(column))
        if url:
            properties[column] = {"url": url}
    if not properties:
        return 0
    notion_request("PATCH", f"/v1/pages/{case.page_id}", json_body={"properties": properties})
    return len(properties)


def enrich_report_cases_with_gemini_news(
    cases: Sequence[CaseRecord],
    args: argparse.Namespace,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    selected_cases = [case for case in cases if case.page_id]
    if not selected_cases:
        return {"enabled": False, "reason": "no_cases"}
    if dry_run:
        return {"enabled": False, "reason": "dry_run"}

    try:
        import SJUR_csv_to_csv_NOTIONfriendly_v2 as sjur
    except Exception as exc:
        LOGGER.warning("Busca Gemini de notícias ignorada: falha ao carregar conversor SJUR (%s).", exc)
        return {"enabled": False, "reason": "sjur_import_error"}

    api_key = sjur.resolve_gemini_api_key(
        _normalize_ws(getattr(args, "gemini_api_key", "")),
        _normalize_ws(getattr(args, "gemini_key_file", "")) or sjur.DEFAULT_GEMINI_KEY_FILE,
    )
    if not api_key:
        LOGGER.warning("Busca Gemini de notícias ignorada: chave Gemini não encontrada.")
        return {"enabled": False, "reason": "missing_api_key", "cases": len(selected_cases)}

    rows = [_news_url_row_from_case(case) for case in selected_cases]
    original_rows = [dict(row) for row in rows]
    lookup_payloads = [_news_lookup_payload_from_case(case) for case in selected_cases]
    config = sjur.WebLookupConfig(
        enabled=True,
        provider="gemini",
        api_key=api_key,
        model=_normalize_ws(getattr(args, "gemini_model", "")) or sjur.DEFAULT_GEMINI_NEWS_MODEL,
        timeout_seconds=max(1, int(getattr(args, "gemini_timeout", sjur.GEMINI_DEFAULT_TIMEOUT) or sjur.GEMINI_DEFAULT_TIMEOUT)),
        max_workers=max(1, int(getattr(args, "gemini_max_workers", sjur.GEMINI_DEFAULT_MAX_WORKERS) or sjur.GEMINI_DEFAULT_MAX_WORKERS)),
        batch_size=max(1, int(getattr(args, "gemini_batch_size", sjur.GEMINI_DEFAULT_BATCH_SIZE) or sjur.GEMINI_DEFAULT_BATCH_SIZE)),
        delay_between_batches=max(0.0, float(getattr(args, "gemini_delay", sjur.GEMINI_DEFAULT_DELAY) or sjur.GEMINI_DEFAULT_DELAY)),
        max_tokens=sjur.GEMINI_DEFAULT_MAX_OUTPUT_TOKENS,
    )
    LOGGER.info(
        "[Gemini] Busca seletiva de notícias: apenas %d caso(s) publicado(s) no relatório.",
        len(selected_cases),
    )
    metrics = asyncio.run(
        sjur.enriquecer_rows_com_urls_async(
            rows,
            lambda message: LOGGER.info(message),
            config,
            lookup_payloads=lookup_payloads,
            cache_path=sjur.resolve_web_lookup_cache_path(),
        )
    )
    patched_pages = 0
    patched_properties = 0
    for case, original_row, row in zip(selected_cases, original_rows, rows):
        urls = _unique_preserve_order(
            [
                *case.noticias,
                *[_normalize_ws(row.get(column)) for column in NEWS_URL_COLUMNS if _normalize_ws(row.get(column))],
            ]
        )
        case.noticias = urls
        row_changed = any(_normalize_ws(row.get(column)) != _normalize_ws(original_row.get(column)) for column in NEWS_URL_COLUMNS)
        if row_changed and any(_normalize_ws(row.get(column)) for column in NEWS_URL_COLUMNS):
            patched = _patch_case_news_urls(case, row)
            if patched:
                patched_pages += 1
                patched_properties += patched
    return {
        "enabled": True,
        "cases": len(selected_cases),
        "patched_pages": patched_pages,
        "patched_properties": patched_properties,
        **metrics,
    }


def _normalize_summary_bullets(
    values: Iterable[str],
    *,
    max_items: int,
    max_chars: Optional[int] = None,
) -> List[str]:
    out: List[str] = []
    for raw in values:
        value = normalize_summary_snippet(raw, max_chars=max_chars)
        if value:
            out.append(value)
    return _unique_preserve_order(out)[:max_items]


def _first_meaningful_clause(text: Any, *, max_chars: int = 120) -> str:
    source = _normalize_ws(text).strip(" .;,:")
    if not source:
        return ""
    for part in re.split(r"(?<=[.!?])\s+|\s*\|\s*|;\s+", source):
        candidate = _normalize_ws(part).strip(" .;,:")
        if len(candidate) >= 20:
            return truncate_text(candidate, max_chars, suffix="")
    return truncate_text(source, max_chars, suffix="")


def _lowercase_initial_fragment(text: Any) -> str:
    value = _normalize_ws(text)
    if len(value) >= 2 and value[:1].isupper() and value[1:2].islower():
        return value[:1].lower() + value[1:]
    return value


def _is_material_alert_text(text: Any) -> bool:
    source = _normalize_ws(text)
    if not source or len(source) < 30:
        return False
    if GENERIC_ALERT_START_RE.match(source):
        return False
    if GENERIC_PRIORITY_ONLY_ALERT_RE.search(source):
        return False
    if GENERIC_COUNT_ONLY_ALERT_RE.match(source) and not ALERT_MATERIALITY_RE.search(source):
        return False
    if len(source) < 55 and not ALERT_MATERIALITY_RE.search(source):
        return False
    return True


def _reportable_case_analysis_pairs(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[tuple[CaseRecord, CaseAnalysis]]:
    case_map = {case_resume_key(case): case for case in cases}
    pairs: List[tuple[CaseRecord, CaseAnalysis]] = []
    for analysis in analyses:
        if not is_reportable_analysis(analysis):
            continue
        case = case_map.get(analysis_resume_key(analysis))
        if case is None:
            continue
        pairs.append((case, analysis))
    pairs.sort(key=lambda item: publication_sort_key(item[0], item[1]))
    return pairs


def _is_material_alert_pair(case: CaseRecord, analysis: CaseAnalysis) -> bool:
    return bool(
        is_publishable_analysis(analysis)
        or analysis_display_score(analysis) >= 8
        or compute_high_impact_signal_score(case, analysis) >= 20
        or compute_political_prominence_score(case, analysis) >= 45
    )


def _build_material_watchpoints(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    *,
    limit: int = MAX_WATCHPOINTS,
) -> List[str]:
    alerts: List[str] = []
    pairs = _reportable_case_analysis_pairs(cases, analyses)
    if not pairs:
        return alerts
    for idx, (case, analysis) in enumerate(pairs):
        if idx > 0 and not _is_material_alert_pair(case, analysis):
            continue
        title = truncate_heading_text(analysis.title or case.tema or case.process_label(), 96)
        consequence = _first_meaningful_clause(analysis.consequence or analysis.what_happened, max_chars=120).rstrip(".")
        strategic = _lowercase_initial_fragment(
            _first_meaningful_clause(analysis.strategic_comment or analysis.why_relevant, max_chars=120)
        ).rstrip(".")
        details = _unique_preserve_order([consequence, strategic])
        if not details:
            fallback_detail = _first_meaningful_clause(case.punchline or analysis.why_relevant or case.tema, max_chars=140).rstrip(".")
            if fallback_detail:
                details = [fallback_detail]
        candidate = truncate_text(f"{title}: {'; '.join(details)}.", MAX_ALERT_TEXT_CHARS, suffix="")
        if not _is_material_alert_text(candidate):
            continue
        alerts.append(candidate)
        if len(alerts) >= limit:
            break
    return _unique_preserve_order(alerts)[:limit]


def _build_material_party_alerts(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    party_counter: Counter[str],
    *,
    limit: int = MAX_PARTY_ALERTS,
) -> List[str]:
    pairs = _reportable_case_analysis_pairs(cases, analyses)
    grouped: Dict[str, Dict[str, Any]] = {}
    for case, analysis in pairs:
        for party in _unique_preserve_order(list(analysis.parties) + list(case.partidos)):
            key = _normalize_ws(party).casefold()
            if not key:
                continue
            entry = grouped.setdefault(key, {"label": _normalize_ws(party), "pairs": []})
            entry["pairs"].append((case, analysis))

    alerts: List[str] = []
    for party, _count in party_counter.most_common(10):
        key = _normalize_ws(party).casefold()
        entry = grouped.get(key)
        if not entry:
            continue
        related_pairs = list(entry["pairs"])
        if len(related_pairs) < MIN_RECURRING_ALERT_CASES:
            continue
        material_pairs = [item for item in related_pairs if _is_material_alert_pair(item[0], item[1])]
        if not material_pairs:
            continue
        best_case, best_analysis = material_pairs[0]
        detail = _lowercase_initial_fragment(
            _first_meaningful_clause(
                best_analysis.consequence or best_analysis.strategic_comment or best_analysis.why_relevant,
                max_chars=110,
            )
        ).rstrip(".")
        title = truncate_heading_text(best_analysis.title or best_case.tema or best_case.process_label(), 74)
        candidate = f"{entry['label']} reaparece em {len(related_pairs)} casos reportáveis, com destaque para {title}"
        if detail:
            candidate += f"; {detail}"
        candidate = truncate_text(candidate.rstrip(".") + ".", MAX_ALERT_TEXT_CHARS, suffix="")
        if not _is_material_alert_text(candidate):
            continue
        alerts.append(candidate)
        if len(alerts) >= limit:
            break
    return _unique_preserve_order(alerts)[:limit]


def _build_material_lawyer_signals(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    lawyer_counter: Counter[str],
    *,
    limit: int = MAX_LAWYER_SIGNALS,
) -> List[str]:
    pairs = _reportable_case_analysis_pairs(cases, analyses)
    grouped: Dict[str, Dict[str, Any]] = {}
    for case, analysis in pairs:
        for lawyer in _unique_preserve_order(case.advogados):
            key = _normalize_ws(lawyer).casefold()
            if not key:
                continue
            entry = grouped.setdefault(key, {"label": _normalize_ws(lawyer), "pairs": []})
            entry["pairs"].append((case, analysis))

    alerts: List[str] = []
    for lawyer, _count in lawyer_counter.most_common(10):
        key = _normalize_ws(lawyer).casefold()
        entry = grouped.get(key)
        if not entry:
            continue
        related_pairs = list(entry["pairs"])
        if len(related_pairs) < MIN_RECURRING_ALERT_CASES:
            continue
        material_pairs = [item for item in related_pairs if _is_material_alert_pair(item[0], item[1])]
        if not material_pairs:
            continue
        best_case, best_analysis = material_pairs[0]
        detail = _lowercase_initial_fragment(
            _first_meaningful_clause(
                best_analysis.strategic_comment or best_analysis.consequence or best_analysis.why_relevant,
                max_chars=108,
            )
        ).rstrip(".")
        title = truncate_heading_text(best_analysis.title or best_case.tema or best_case.process_label(), 70)
        candidate = f"{entry['label']} atua em {len(related_pairs)} casos reportáveis, incluindo {title}"
        if detail:
            candidate += f"; {detail}"
        candidate = truncate_text(candidate.rstrip(".") + ".", MAX_ALERT_TEXT_CHARS, suffix="")
        if not _is_material_alert_text(candidate):
            continue
        alerts.append(candidate)
        if len(alerts) >= limit:
            break
    return _unique_preserve_order(alerts)[:limit]


def _select_material_alert_items(
    primary: Iterable[str],
    *,
    fallback: Iterable[str],
    limit: int,
) -> List[str]:
    selected: List[str] = []
    seen: set[str] = set()
    for raw in list(primary) + list(fallback):
        value = normalize_summary_snippet(raw)
        if not _is_material_alert_text(value):
            continue
        marker = value.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        selected.append(value)
        if len(selected) >= limit:
            break
    return selected


def finalize_report_summary(
    summary: ReportSummary,
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    *,
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
) -> ReportSummary:
    fallback_party_alerts = _build_material_party_alerts(cases, analyses, party_counter)
    fallback_lawyer_signals = _build_material_lawyer_signals(cases, analyses, lawyer_counter)
    fallback_watchpoints = _build_material_watchpoints(cases, analyses)
    executive_highlights = _normalize_summary_bullets(
        summary.executive_highlights,
        max_items=MAX_EXECUTIVE_HIGHLIGHTS,
        max_chars=MAX_EXECUTIVE_BULLET_CHARS,
    )
    if not executive_highlights and fallback_watchpoints:
        executive_highlights = fallback_watchpoints[:1]
    return ReportSummary(
        overview_callout=normalize_summary_snippet(
            summary.overview_callout,
            max_chars=MAX_OVERVIEW_CALLOUT_CHARS,
        ),
        executive_highlights=executive_highlights,
        party_alerts=_select_material_alert_items(
            summary.party_alerts,
            fallback=fallback_party_alerts,
            limit=MAX_PARTY_ALERTS,
        ),
        lawyer_signals=_select_material_alert_items(
            summary.lawyer_signals,
            fallback=fallback_lawyer_signals,
            limit=MAX_LAWYER_SIGNALS,
        ),
        watchpoints=_select_material_alert_items(
            summary.watchpoints,
            fallback=fallback_watchpoints,
            limit=MAX_WATCHPOINTS,
        ),
        closing_note=truncate_text(summary.closing_note, 220, suffix=""),
    )


def report_summary_has_incomplete_items(summary: ReportSummary) -> bool:
    groups = (
        summary.executive_highlights,
        summary.party_alerts,
        summary.lawyer_signals,
        summary.watchpoints,
    )
    return any(
        summary_text_looks_incomplete(item) or has_terminal_ellipsis(item)
        for group in groups
        for item in group
    )


def build_strategic_alert_section_items(summary: ReportSummary) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for group in (summary.watchpoints, summary.party_alerts, summary.lawyer_signals):
        for raw in group:
            value = normalize_summary_snippet(raw)
            if not _is_material_alert_text(value):
                continue
            marker = value.casefold()
            if marker in seen:
                continue
            seen.add(marker)
            items.append(value)
            if len(items) >= MAX_STRATEGIC_ALERT_SECTION_ITEMS:
                return items
    return items


def build_executive_alert_items(summary: ReportSummary) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for raw in list(summary.executive_highlights) + build_strategic_alert_section_items(summary):
        value = normalize_summary_snippet(raw, max_chars=MAX_EXECUTIVE_BULLET_CHARS)
        if not value:
            continue
        marker = value.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        items.append(value)
        if len(items) >= MAX_EXECUTIVE_ALERT_ITEMS:
            break
    if not items and _normalize_ws(summary.closing_note):
        items.append(normalize_summary_snippet(summary.closing_note, max_chars=MAX_EXECUTIVE_BULLET_CHARS))
    return items


def _clean_lawyer_display_name(value: Any) -> str:
    text = _normalize_ws(value)
    if not text:
        return ""
    text = re.sub(r"\s*\([^)]*\bOAB\b[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[-–—]\s*\bOAB\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOAB\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;,.-")


def build_recurring_lawyer_lines(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[str]:
    freq: Counter[str] = Counter()
    labels: Dict[str, str] = {}
    item_refs: Dict[str, List[str]] = {}
    for idx, (case, _analysis) in enumerate(build_publishable_case_pairs(cases, analyses), start=1):
        seen_in_case: set[str] = set()
        for raw_name in case.advogados:
            name = _clean_lawyer_display_name(raw_name)
            if not name:
                continue
            key = name.casefold()
            if key in seen_in_case:
                continue
            seen_in_case.add(key)
            labels.setdefault(key, name)
            freq[key] += 1
            item_refs.setdefault(key, []).append(str(idx))

    ranked = [
        (labels[key], count, item_refs.get(key, []))
        for key, count in freq.items()
        if count >= MIN_RECURRING_ALERT_CASES
    ]
    ranked.sort(key=lambda item: (-item[1], item[0].casefold()))
    lines: List[str] = []
    for name, count, refs in ranked[:MAX_RECURRING_LAWYERS_SECTION_ITEMS]:
        suffix = "caso" if count == 1 else "casos"
        refs_label = ", ".join(refs[:6])
        if len(refs) > 6:
            refs_label += ", ..."
        lines.append(f"**{name}:** {count} {suffix} nos itens {refs_label}.")
    return lines


def build_recurring_lawyer_section_blocks(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[Dict[str, Any]]:
    lines = build_recurring_lawyer_lines(cases, analyses)
    if not lines:
        return []
    blocks: List[Dict[str, Any]] = [build_heading_2_block(SECTION_LAWYERS_TITLE)]
    blocks.extend(build_bulleted_block(line) for line in lines)
    return blocks


def analysis_has_visible_ellipsis(item: CaseAnalysis) -> bool:
    if not is_reportable_analysis(item):
        return False
    return any(has_terminal_ellipsis(getattr(item, field, "")) for field in ELLIPSIS_ANALYSIS_FIELDS)


def case_triage_signal_text(case: CaseRecord) -> str:
    return _safe_join(
        [
            case.tema,
            case.punchline,
            case.texto_decisao,
            case.descricao_classe,
            case.descricao_tipo_decisao,
            case.ano_eleicao,
            _safe_join(case.assuntos),
            _safe_join(case.partes),
            _safe_join(case.partidos),
            _safe_join(case.resultado),
        ],
        sep=" | ",
    )


def local_openai_triage_score(
    case: CaseRecord,
    *,
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
) -> int:
    signal_text = case_triage_signal_text(case)
    score = 0
    if case.partidos:
        score += 22
    if any(party_counter.get(party, 0) > 1 for party in case.partidos):
        score += 8
    if any(lawyer_counter.get(name, 0) > 1 for name in case.advogados):
        score += 6
    if FEDERAL_HIGH_OFFICE_RE.search(signal_text):
        score += 32
    elif PUBLIC_FIGURE_HINT_RE.search(signal_text):
        score += 20
    if NATIONAL_PARTY_SCOPE_RE.search(signal_text):
        score += 26
    if PARTY_INSTITUTIONAL_RE.search(signal_text):
        score += 20
    if PARTY_FINANCE_SYSTEMIC_RE.search(signal_text) or ACCOUNTS_HINT_RE.search(signal_text):
        score += 18
    if HIGH_IMPACT_ELECTORAL_RE.search(signal_text):
        score += 18
    elif HIGH_IMPACT_HINT_RE.search(signal_text):
        score += 14
    if _normalize_ws(case.ano_eleicao) in {"2018", "2022", "2026"}:
        score += 5
    if LOWER_LOCAL_OFFICE_RE.search(signal_text) and score < DEFAULT_OPENAI_TRIAGE_THRESHOLD:
        score -= 10
    return max(0, min(100, score))


def should_skip_openai_for_low_priority(
    case: CaseRecord,
    *,
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
    triage_threshold: int = DEFAULT_OPENAI_TRIAGE_THRESHOLD,
) -> bool:
    return local_openai_triage_score(case, party_counter=party_counter, lawyer_counter=lawyer_counter) < int(triage_threshold)


def fallback_case_analysis(case: CaseRecord, lawyer_counter: Counter[str]) -> CaseAnalysis:
    base_text = " ".join([case.tema, case.punchline, _safe_join(case.assuntos), _safe_join(case.partes)])
    score = 4
    if case.partidos:
        score += 1
    if ACCOUNTS_HINT_RE.search(base_text):
        score += 1
    if HIGH_IMPACT_HINT_RE.search(base_text):
        score += 2
    if PUBLIC_FIGURE_HINT_RE.search(base_text):
        score += 1
    score = max(1, min(10, score))

    recurring_lawyers = [name for name in case.advogados if lawyer_counter.get(name, 0) > 1]
    consequence = _safe_join(case.resultado, sep=", ")
    if case.punchline:
        consequence = f"{consequence}. {case.punchline}" if consequence else case.punchline
    return CaseAnalysis(
        case_id=case.case_id,
        title=truncate_heading_text(case.tema or case.punchline or case.process_label(), HEADING_SUMMARY_MAX_CHARS),
        relevance_score=score,
        display_score=score,
        risk_level=risk_from_score(score),
        includes_public_figure=bool(PUBLIC_FIGURE_HINT_RE.search(base_text)),
        includes_party=bool(case.partidos),
        public_figures=[],
        parties=list(case.partidos),
        lawyers_signal=(
            f"Advogados recorrentes no período: {_safe_join(recurring_lawyers, sep='; ')}."
            if recurring_lawyers
            else "Sem sinal claro de banca recorrente apenas com os dados do período."
        ),
        what_happened=_compact_report_field_text(case.punchline or case.tema or case.texto_decisao, MAX_TOGGLE_FACTS_CHARS),
        legal_grounds=_compact_report_field_text(
            _safe_join([_safe_join(case.assuntos, sep=", "), case.tema], sep=" | "),
            MAX_TOGGLE_LEGAL_CHARS,
        ),
        consequence=_compact_report_field_text(consequence or case.tema or "Resultado não detalhado na base.", MAX_TOGGLE_CONSEQUENCE_CHARS),
        strategic_comment=_compact_report_field_text(
            f"Caso em {case.local_label()} relatado por {case.relator or 'relator não informado'}. "
            f"Partidos citados: {_safe_join(case.partidos, sep='; ') or 'nenhum explícito'}.",
            MAX_TOGGLE_STRATEGIC_CHARS,
        ),
        why_relevant=_compact_report_field_text(
            "Relevante para monitoramento político-eleitoral pela combinação de tema, partes envolvidas e efeito prático indicado na base.",
            MAX_TOGGLE_WHY_RELEVANT_CHARS,
        ),
        source_notes=_unique_preserve_order(case.noticias + [case.source_url]),
        page_id=case.page_id,
    )


def build_case_analysis_prompt(start_iso: str, end_iso: str) -> str:
    period_text = format_period_br(start_iso, end_iso)
    return (
        "Você é um analista jurídico-eleitoral metódico. "
        "O destinatário é consultor legislativo da Câmara dos Deputados, com foco em Direito Eleitoral e impacto federal. "
        "Priorize precisão, raciocínio lógico, voz ativa e economia verbal. "
        f"Analise processos do período exato {period_text}. "
        "Use apenas os dados fornecidos. Não invente fatos, partidos, cargos ou consequências. "
        "Identifique partidos de modo conservador. Se não estiver explícito, diga menos. "
        "No campo 'parties', liste APENAS legendas partidárias (sigla ou nome oficial de partido, federação ou coligação) — "
        "NUNCA nomes de pessoas físicas; pessoas vão em public_figures. Sem partido explícito, devolva lista vazia. "
        "A unidade de análise é o processo judicial. Para cada processo, explique o que ocorreu, "
        "os fundamentos jurídicos adotados, a consequência aplicada e a relevância político-eleitoral. "
        "Considere como sinais máximos de relevância: Presidente da República, Vice-Presidente, governadores, senadores, "
        "deputados federais, candidatos a esses cargos, diretórios nacionais/estaduais, fundações partidárias, "
        "registro/incorporação/fidelidade partidária e temas de FEFC, Fundo Partidário, cotas e contas partidárias. "
        "Casos municipais rotineiros de prefeito, vice-prefeito ou vereador só devem receber nota alta se trouxerem tese replicável "
        "para partidos políticos ou efeito institucional claro além do município. "
        "Reserve notas 8 a 10 para casos com interesse federal/partidário material; temas locais sem esse vetor devem ficar em baixo ou médio. "
        "REGRA INDEROGÁVEL: processo que envolva Presidente da República, Vice-Presidente, governador, vice-governador, "
        "senador, deputado federal ou ministro de Estado (como parte, investigado ou beneficiário direto) recebe relevance_score "
        "mínimo 8 e risk_level mínimo 'alto', MESMO quando a decisão mantém improcedência, nega seguimento ou é processualmente "
        "desfavorável ao autor — o desfecho não reduz a relevância para a consultoria legislativa. "
        "Defina includes_public_figure=true sempre que public_figures não estiver vazio. "
        "No campo 'title', escreva um microtítulo temático curto e intuitivo do caso, em estilo nominal, idealmente entre 90 e 120 caracteres e no máximo com 130 caracteres. "
        "Evite narrativa processual. Não comece com expressões como 'em decisão monocrática', 'o pedido de', "
        "'interposto contra acórdão', 'deu provimento', 'negou seguimento' ou fórmulas semelhantes. "
        "Use como foco o tema jurídico-eleitoral do caso. Quando couber, preserve apenas referências essenciais para identificar o instituto, como AIJE, AIME, art. 30-A ou art. 41-A. "
        "Não deixe o título terminar com caudas abertas, artigos incompletos, súmulas soltas, parênteses abertos ou fragmentos sem conclusão. "
        "Quando usar expressão indispensável em inglês, marque-a em itálico Markdown com asteriscos simples. "
        "Não use reticências em nenhum campo. Feche cada frase com conclusão completa, sem '...'. "
        "Quando a decisão firmar tese nova, promover virada jurisprudencial ou superar entendimento consolidado do TSE, "
        "sinalize isso explicitamente no início de strategic_comment com o prefixo 'Tese nova:' ou 'Virada:'. "
        "Quando o caso tocar matéria objeto de legislação eleitoral em tramitação ou debate no Congresso Nacional "
        "(reforma eleitoral, FEFC/Fundo Partidário, cotas, fidelidade partidária, inelegibilidades, federações), "
        "aponte em strategic_comment o impacto potencial para o trabalho legislativo da Câmara dos Deputados."
    )


def build_top_band_reclassify_prompt(start_iso: str, end_iso: str) -> str:
    period_text = format_period_br(start_iso, end_iso)
    return (
        "Você é um revisor de priorização político-eleitoral. "
        f"Recalibre apenas a diferença entre notas 9/10 e 10/10 para casos do período {period_text}. "
        "Use só os dados fornecidos. Não invente fatos. "
        "Critério principal: proeminência política do caso. "
        "Considere figuras públicas, cargos eletivos ou pré-candidaturas, partidos com alcance nacional/estadual "
        "e sensibilidade institucional do processo. "
        "Retorne apenas display_score 9 ou 10 e um ranking_reason curto. "
        "Se houver dúvida, seja conservador e mantenha 9."
    )


def build_top_band_cache_key(case: CaseRecord, analysis: CaseAnalysis, *, model: str, start_iso: str, end_iso: str) -> str:
    material = {
        "page_id": _normalize_ws(case.page_id or case.case_id),
        "current_score": max(1, min(10, int(analysis.relevance_score))),
        "model": _normalize_ws(model),
        "period": f"{start_iso}:{end_iso}",
    }
    return json.dumps(material, ensure_ascii=False, sort_keys=True)


def reclassify_top_band_scores(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    *,
    start_iso: str,
    end_iso: str,
    cached_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[List[CaseAnalysis], Dict[str, Dict[str, Any]]]:
    cache_map: Dict[str, Dict[str, Any]] = dict(cached_map or {})
    if OPENAI_CFG is None or OPENAI_SESSION is None:
        return list(analyses), cache_map

    case_map = {case_resume_key(case): case for case in cases}
    candidates: List[tuple[CaseRecord, CaseAnalysis, str]] = []
    for analysis in analyses:
        if not is_publishable_analysis(analysis):
            continue
        if max(1, min(10, int(analysis.relevance_score))) not in {9, 10}:
            continue
        case = case_map.get(analysis_resume_key(analysis))
        if case is None:
            continue
        cache_key = build_top_band_cache_key(
            case,
            analysis,
            model=OPENAI_CFG.model,
            start_iso=start_iso,
            end_iso=end_iso,
        )
        cached = cache_map.get(cache_key)
        if isinstance(cached, dict) and cached.get("display_score") in {9, 10}:
            analysis.display_score = int(cached["display_score"])
            analysis.ranking_reason = _normalize_ws(cached.get("ranking_reason"))
            continue
        candidates.append((case, analysis, cache_key))

    if not candidates:
        return list(analyses), cache_map

    system_prompt = build_top_band_reclassify_prompt(start_iso, end_iso)
    for start in range(0, len(candidates), TOP_BAND_RECLASSIFY_BATCH_SIZE):
        chunk = candidates[start : start + TOP_BAND_RECLASSIFY_BATCH_SIZE]
        payload = {
            "periodo": {"inicio": start_iso, "fim": end_iso},
            "items": [
                {
                    "case_id": case.page_id or case.case_id,
                    "tema": case.tema,
                    "punchline": case.punchline,
                    "what_happened": analysis.what_happened,
                    "consequence": analysis.consequence,
                    "parties": analysis.parties or case.partidos,
                    "public_figures": analysis.public_figures,
                    "current_score": max(1, min(10, int(analysis.relevance_score))),
                    "current_risk": _normalize_ws(analysis.risk_level).casefold(),
                }
                for case, analysis, _cache_key in chunk
            ],
        }
        try:
            response = openai_json_call(
                system_prompt,
                payload,
                schema=TOP_BAND_RECLASSIFY_SCHEMA,
                request_label=f"OpenAI reclassificação 9/10-10/10 ({start + 1}-{start + len(chunk)})",
                max_completion_tokens=2500,
            )
        except Exception as exc:
            LOGGER.warning("Falha na reclassificação opcional do topo: %s", exc)
            continue
        raw_items = response.get("items")
        if not isinstance(raw_items, list):
            continue
        by_case_id = {
            _normalize_ws(item.get("case_id")): item
            for item in raw_items
            if isinstance(item, dict) and _normalize_ws(item.get("case_id"))
        }
        for case, analysis, cache_key in chunk:
            raw = by_case_id.get(_normalize_ws(case.page_id or case.case_id))
            if not isinstance(raw, dict):
                continue
            display_score = int(raw.get("display_score", analysis.relevance_score))
            if display_score not in {9, 10}:
                continue
            analysis.display_score = display_score
            analysis.ranking_reason = _normalize_ws(raw.get("ranking_reason"))
            cache_map[cache_key] = {
                "display_score": display_score,
                "ranking_reason": analysis.ranking_reason,
            }
    return list(analyses), cache_map


def estimate_case_payload_chars(
    case: CaseRecord,
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
) -> int:
    payload = case.compact_ai_payload(party_counter, lawyer_counter)
    return len(json.dumps(payload, ensure_ascii=False))


def chunk_cases_for_openai(
    cases: Sequence[CaseRecord],
    *,
    max_cases: int,
    char_budget: int,
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
) -> List[List[CaseRecord]]:
    chunks: List[List[CaseRecord]] = []
    current: List[CaseRecord] = []
    current_chars = 0
    limit_cases = max(1, int(max_cases))
    limit_chars = max(2000, int(char_budget))

    for case in cases:
        item_chars = estimate_case_payload_chars(case, party_counter, lawyer_counter)
        would_exceed_items = len(current) >= limit_cases
        would_exceed_chars = bool(current) and (current_chars + item_chars) > limit_chars
        if would_exceed_items or would_exceed_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(case)
        current_chars += item_chars
    if current:
        chunks.append(current)
    return chunks


def analyze_cases(
    cases: Sequence[CaseRecord],
    *,
    start_iso: str,
    end_iso: str,
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
    max_cases_per_batch: int = DEFAULT_MAX_CASES_PER_BATCH,
    openai_max_workers: int = DEFAULT_OPENAI_MAX_WORKERS,
    max_openai_cases: int = DEFAULT_MAX_OPENAI_CASES,
    openai_triage_threshold: int = DEFAULT_OPENAI_TRIAGE_THRESHOLD,
    analysis_cache_enabled: bool = True,
    resume_analysis_map: Optional[Dict[str, CaseAnalysis]] = None,
    progress_callback: Optional[Any] = None,
) -> List[CaseAnalysis]:
    analysis_by_case_id: Dict[str, CaseAnalysis] = dict(resume_analysis_map or {})
    pending_cases = [case for case in cases if case_resume_key(case) not in analysis_by_case_id]
    model_name = OPENAI_CFG.model if OPENAI_CFG is not None else DEFAULT_OPENAI_MODEL
    cache_payload = load_case_analysis_cache(enabled=analysis_cache_enabled)

    cached_cases = 0
    locally_skipped_cases: List[CaseRecord] = []
    openai_candidates: List[tuple[int, CaseRecord]] = []
    for case in pending_cases:
        cached_analysis = cached_case_analysis_for_case(case, cache_payload, model=model_name)
        if cached_analysis is not None:
            analysis_by_case_id[case_resume_key(case)] = cached_analysis
            cached_cases += 1
            continue
        triage_score = local_openai_triage_score(case, party_counter=party_counter, lawyer_counter=lawyer_counter)
        if triage_score < int(openai_triage_threshold):
            locally_skipped_cases.append(case)
            analysis_by_case_id[case_resume_key(case)] = fallback_case_analysis(case, lawyer_counter)
        else:
            openai_candidates.append((triage_score, case))

    openai_candidates.sort(
        key=lambda item: (
            -item[0],
            -(1 if item[1].partidos else 0),
            item[1].data_decisao,
            item[1].process_label(),
        )
    )
    max_openai = max(0, int(max_openai_cases))
    if max_openai:
        selected_candidates = openai_candidates[:max_openai]
        capped_cases = [case for _score, case in openai_candidates[max_openai:]]
    else:
        selected_candidates = []
        capped_cases = [case for _score, case in openai_candidates]
    openai_pending_cases = [case for _score, case in selected_candidates]
    for case in capped_cases:
        analysis_by_case_id[case_resume_key(case)] = fallback_case_analysis(case, lawyer_counter)

    if analysis_by_case_id:
        LOGGER.info(
            "[OpenAI] pré-processamento | análises_prontas=%d/%d | candidatos_gpt=%d",
            len(analysis_by_case_id),
            len(cases),
            len(openai_pending_cases),
        )
    if cached_cases:
        LOGGER.info("[OpenAI] cache local reaproveitou %d análise(s) sem nova chamada.", cached_cases)
    if locally_skipped_cases:
        LOGGER.info(
            "[OpenAI] triagem local evitou %d chamada(s) por score abaixo de %d.",
            len(locally_skipped_cases),
            int(openai_triage_threshold),
        )
    if capped_cases:
        LOGGER.info(
            "[OpenAI] teto econômico evitou %d chamada(s); limite=%d caso(s) enviados ao GPT.",
            len(capped_cases),
            max_openai,
        )

    if OPENAI_CFG is None or OPENAI_SESSION is None:
        LOGGER.warning("Chave OpenAI ausente ou cliente não inicializado. Usando fallback heurístico para todos os casos.")
        for case in openai_pending_cases:
            analysis_by_case_id[case_resume_key(case)] = fallback_case_analysis(case, lawyer_counter)
        if progress_callback is not None:
            progress_callback(analysis_by_case_id, 0, 0)
        return finalize_case_analyses(cases, analysis_by_case_id)

    system_prompt = build_case_analysis_prompt(start_iso, end_iso)

    def _coerce_analysis(case: CaseRecord, raw_item: Dict[str, Any]) -> CaseAnalysis:
        return CaseAnalysis(
            case_id=case.case_id,
            title=truncate_heading_text(raw_item.get("title") or case.tema or case.process_label(), HEADING_SUMMARY_MAX_CHARS),
            relevance_score=max(1, min(10, int(raw_item.get("relevance_score", 5)))),
            display_score=max(1, min(10, int(raw_item.get("display_score") or raw_item.get("relevance_score", 5)))),
            risk_level=_normalize_ws(raw_item.get("risk_level")) or "medio",
            includes_public_figure=bool(raw_item.get("includes_public_figure")),
            includes_party=bool(raw_item.get("includes_party")),
            public_figures=_unique_preserve_order(raw_item.get("public_figures") or []),
            parties=_unique_preserve_order(raw_item.get("parties") or case.partidos),
            lawyers_signal=_normalize_ws(raw_item.get("lawyers_signal")),
            what_happened=_normalize_ws(raw_item.get("what_happened")),
            legal_grounds=_normalize_ws(raw_item.get("legal_grounds")),
            consequence=_normalize_ws(raw_item.get("consequence")),
            strategic_comment=_normalize_ws(raw_item.get("strategic_comment")),
            why_relevant=_normalize_ws(raw_item.get("why_relevant")),
            source_notes=_unique_preserve_order(raw_item.get("source_notes") or case.noticias + [case.source_url]),
            page_id=case.page_id,
            ranking_reason=_normalize_ws(raw_item.get("ranking_reason")),
        )

    def _analyze_batch(batch: Sequence[CaseRecord], *, label: str) -> tuple[List[CaseAnalysis], bool]:
        payload = {
            "periodo": {"inicio": start_iso, "fim": end_iso},
            "estatisticas_periodo": {
                "processos_no_periodo": len(cases),
                "partidos_mais_citados": party_counter.most_common(10),
                "advogados_recorrentes": lawyer_counter.most_common(10),
            },
            "casos": [case.compact_ai_payload(party_counter, lawyer_counter) for case in batch],
        }
        try:
            response = openai_json_call(
                system_prompt,
                payload,
                schema=CASE_BATCH_SCHEMA,
                request_label=label,
                max_completion_tokens=min(8000, max(3000, 1400 * len(batch))),
            )
            raw_items = response.get("items", [])
            item_map = {
                _normalize_ws(item.get("case_id")): item
                for item in raw_items
                if isinstance(item, dict) and _normalize_ws(item.get("case_id"))
            }
            out: List[CaseAnalysis] = []
            cacheable = True
            for case in batch:
                raw_item = item_map.get(case.page_id or case.case_id)
                if not raw_item:
                    out.append(fallback_case_analysis(case, lawyer_counter))
                    cacheable = False
                    continue
                try:
                    out.append(_coerce_analysis(case, raw_item))
                except Exception as exc:
                    LOGGER.warning("Falha ao normalizar output da OpenAI para %s: %s", case.case_id, exc)
                    out.append(fallback_case_analysis(case, lawyer_counter))
                    cacheable = False
            return out, cacheable
        except Exception as exc:
            message = _normalize_ws(exc).casefold()
            if len(batch) > 1 and ("finish_reason=length" in message or " length" in message or "resposta vazia" in message):
                split_at = max(1, len(batch) // 2)
                LOGGER.warning(
                    "%s excedeu o limite da resposta. Dividindo o lote em %d + %d caso(s).",
                    label,
                    split_at,
                    len(batch) - split_at,
                )
                left = _analyze_batch(batch[:split_at], label=f"{label} | parte 1")
                right = _analyze_batch(batch[split_at:], label=f"{label} | parte 2")
                return left[0] + right[0], bool(left[1] and right[1])
            LOGGER.warning("%s falhou: %s. Usando fallback heurístico no lote.", label, exc)
            return [fallback_case_analysis(case, lawyer_counter) for case in batch], False

    batches = chunk_cases_for_openai(
        openai_pending_cases,
        max_cases=max_cases_per_batch,
        char_budget=DEFAULT_OPENAI_BATCH_CHAR_BUDGET,
        party_counter=party_counter,
        lawyer_counter=lawyer_counter,
    )
    total_batches = len(batches)
    if total_batches == 0:
        return finalize_case_analyses(cases, analysis_by_case_id)

    max_chars_batch = max(
        (
            sum(estimate_case_payload_chars(case, party_counter, lawyer_counter) for case in batch)
            for batch in batches
        ),
        default=0,
    )
    requested_workers = max(1, int(openai_max_workers))
    effective_workers = min(requested_workers, max(1, total_batches))
    if max_chars_batch >= int(DEFAULT_OPENAI_BATCH_CHAR_BUDGET * 0.9) and effective_workers > 1:
        LOGGER.info(
            "[OpenAI] lotes pesados (max_chars=%d). Limitando workers de %d para 1.",
            max_chars_batch,
            effective_workers,
        )
        effective_workers = 1
    low_risk_small_batches = int(max_cases_per_batch) <= 1 and max_chars_batch <= 5000
    if total_batches >= 4 and low_risk_small_batches and effective_workers > MAX_OPENAI_LIGHT_BATCH_WORKERS:
        LOGGER.info(
            "[OpenAI] lotes pequenos e leves. Limitando workers de %d para %d.",
            effective_workers,
            MAX_OPENAI_LIGHT_BATCH_WORKERS,
        )
        effective_workers = MAX_OPENAI_LIGHT_BATCH_WORKERS
    elif total_batches >= 4 and (not low_risk_small_batches) and effective_workers > MAX_OPENAI_HEAVY_BATCH_WORKERS:
        LOGGER.info(
            "[OpenAI] muitos lotes (%d). Limitando workers de %d para %d por estabilidade.",
            total_batches,
            effective_workers,
            MAX_OPENAI_HEAVY_BATCH_WORKERS,
        )
        effective_workers = MAX_OPENAI_HEAVY_BATCH_WORKERS

    LOGGER.info(
        "[OpenAI] lotes preparados | total=%d | max_chars_lote=%d | limite_chars=%d | max_itens_lote=%d | workers=%d",
        total_batches,
        max_chars_batch,
        DEFAULT_OPENAI_BATCH_CHAR_BUDGET,
        int(max_cases_per_batch),
        effective_workers,
    )

    def _submit_payload(batch_idx_zero: int, batch: Sequence[CaseRecord]) -> tuple[int, List[CaseAnalysis], bool]:
        first_case = batch[0] if batch else None
        last_case = batch[-1] if batch else None
        batch_label = (
            f"OpenAI análise de casos {first_case.process_label() if first_case else '?'}"
            f" .. {last_case.process_label() if last_case else '?'}"
        )
        LOGGER.info(
            "[OpenAI] lote %d/%d | casos_no_lote=%d | iniciado em worker",
            batch_idx_zero + 1,
            total_batches,
            len(batch),
        )
        batch_results, cacheable = _analyze_batch(batch, label=batch_label)
        return batch_idx_zero, batch_results, cacheable

    case_by_analysis_key: Dict[str, CaseRecord] = {}
    for case in openai_pending_cases:
        case_by_analysis_key[case_resume_key(case)] = case
        case_by_analysis_key[_normalize_ws(case.case_id)] = case

    def _store_cache_if_safe(batch_results: Sequence[CaseAnalysis], *, cacheable: bool) -> None:
        if not cacheable or not analysis_cache_enabled:
            return
        stored = store_case_analysis_cache_entries(cache_payload, batch_results, case_by_analysis_key, model=model_name)
        if stored:
            save_case_analysis_cache(cache_payload)
            LOGGER.info("[OpenAI] cache local atualizado com %d análise(s).", stored)

    if effective_workers <= 1:
        for batch_idx, batch in enumerate(batches, start=1):
            _, batch_results, cacheable = _submit_payload(batch_idx - 1, batch)
            for item in batch_results:
                analysis_by_case_id[analysis_resume_key(item)] = item
            _store_cache_if_safe(batch_results, cacheable=cacheable)
            LOGGER.info(
                "[OpenAI] lote %d/%d concluido | concluidos_agora=%d/%d",
                batch_idx,
                total_batches,
                len(analysis_by_case_id),
                len(cases),
            )
            if progress_callback is not None:
                progress_callback(analysis_by_case_id, batch_idx, total_batches)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_map = {
                executor.submit(_submit_payload, idx, list(batch)): idx
                for idx, batch in enumerate(batches)
            }
            completed_batches = 0
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    _, batch_results, cacheable = future.result()
                except Exception as exc:
                    LOGGER.warning("[OpenAI] lote %d/%d falhou no worker: %s", idx + 1, total_batches, exc)
                    batch_results = [fallback_case_analysis(case, lawyer_counter) for case in batches[idx]]
                    cacheable = False
                for item in batch_results:
                    analysis_by_case_id[analysis_resume_key(item)] = item
                _store_cache_if_safe(batch_results, cacheable=cacheable)
                completed_batches += 1
                LOGGER.info(
                    "[OpenAI] lote %d/%d concluido | concluidos_agora=%d/%d",
                    idx + 1,
                    total_batches,
                    len(analysis_by_case_id),
                    len(cases),
                )
                if progress_callback is not None:
                    progress_callback(analysis_by_case_id, completed_batches, total_batches)
    return finalize_case_analyses(cases, analysis_by_case_id)


def fallback_report_summary(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
    start_iso: str,
    end_iso: str,
) -> ReportSummary:
    high_priority = [item for item in analyses if item.relevance_score >= 7]
    uf_counter = Counter(case.sigla_uf for case in cases if case.sigla_uf)
    overview = (
        f"Foram mapeados {len(cases)} processos com foco federal/partidário no período {format_period_br(start_iso, end_iso)}. "
        f"{len(high_priority)} deles concentram risco político-eleitoral imediato."
    )
    highlights = [
        f"Partidos mais citados: {_safe_join([f'{party} ({count})' for party, count in party_counter.most_common(5)], sep='; ') or 'nenhum partido explícito na base'}.",
        f"UFs mais recorrentes dentro do filtro: {_safe_join([f'{uf} ({count})' for uf, count in uf_counter.most_common(5)], sep='; ') or 'sem concentração relevante'}.",
    ]
    party_alerts = _build_material_party_alerts(cases, analyses, party_counter)
    lawyer_signals = _build_material_lawyer_signals(cases, analyses, lawyer_counter)
    watchpoints = _build_material_watchpoints(cases, analyses)
    return ReportSummary(
        overview_callout=overview,
        executive_highlights=highlights,
        party_alerts=party_alerts,
        lawyer_signals=lawyer_signals,
        watchpoints=watchpoints,
        closing_note="Síntese gerada em fallback heurístico por indisponibilidade ou erro da OpenAI.",
    )


def summarize_report(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    *,
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
    start_iso: str,
    end_iso: str,
) -> ReportSummary:
    if not analyses:
        return ReportSummary(
            overview_callout=(
                f"No período {format_period_br(start_iso, end_iso)}, não houve caso com interesse federal/partidário material no limiar mínimo de publicação."
            ),
            executive_highlights=["Os processos consultados ficaram fora do filtro seletivo de altos cargos e partidos políticos."],
            party_alerts=[],
            lawyer_signals=[],
            watchpoints=[],
            closing_note="Casos locais sem efeito federal ou partidário claro foram excluídos para reduzir ruído analítico.",
        )

    if OPENAI_CFG is None or OPENAI_SESSION is None:
        return finalize_report_summary(
            fallback_report_summary(cases, analyses, party_counter, lawyer_counter, start_iso, end_iso),
            cases,
            analyses,
            party_counter=party_counter,
            lawyer_counter=lawyer_counter,
        )

    ranked = sorted(analyses, key=analysis_sort_key)
    payload = {
        "periodo": {"inicio": start_iso, "fim": end_iso, "descricao": format_period_br(start_iso, end_iso)},
        "estatisticas": {
            "processos_total": len(cases),
            "processos_prioridade_alta": sum(1 for item in analyses if item.relevance_score >= 7),
            "partidos_mais_citados": party_counter.most_common(10),
            "advogados_mais_recorrentes": lawyer_counter.most_common(10),
        },
        "analises_resumidas": [
            {
                "case_id": item.case_id,
                "title": item.title,
                "relevance_score": item.relevance_score,
                "risk_level": item.risk_level,
                "parties": item.parties,
                "public_figures": item.public_figures,
                "lawyers_signal": truncate_text(item.lawyers_signal, 140),
                "what_happened": truncate_text(item.what_happened, 220),
                "legal_grounds": truncate_text(item.legal_grounds, 160),
                "consequence": truncate_text(item.consequence, 180),
                "strategic_comment": truncate_text(item.strategic_comment, 180),
                "why_relevant": truncate_text(item.why_relevant, 160),
            }
            for item in ranked[:MAX_SUMMARY_CASES]
        ],
    }
    system_prompt = (
        "Você é um analista jurídico-eleitoral metódico e conciso. "
        "Produza uma leitura estratégica seletiva do período exato informado para consultoria legislativa federal. "
        "Priorize impactos para altos cargos da República, partidos políticos, direção partidária, fundos públicos de campanha, contas partidárias e teses com efeito nacional ou estadual. "
        "Não trate caso municipal rotineiro como destaque se ele não trouxer consequência replicável para partidos ou cargos de alta relevância. "
        "Reduza ruído: prefira poucos alertas materiais a listas longas. "
        "Só gere party_alerts ou lawyer_signals quando a recorrência vier acompanhada de consequência concreta, tese sensível ou exposição institucional. "
        "Não transforme frequência bruta em alerta. "
        "Cada bullet deve trazer fato + risco/efeito + implicação estratégica em uma única frase curta. "
        "A leitura executiva sera fundida aos alertas, portanto evite repeticao entre executive_highlights, party_alerts, lawyer_signals e watchpoints. "
        "overview_callout deve ter no maximo duas frases curtas e nao pode resumir caso a caso. "
        "Cada bullet deve ter ate 260 caracteres, com verbo concreto e consequencia operacional clara. "
        "Limites rígidos: até 3 executive_highlights, 2 party_alerts, 1 lawyer_signal e 2 watchpoints. "
        "Se não houver alerta material em uma categoria, devolva lista vazia. "
        "Não invente fatos. Use linguagem executiva, densa e objetiva. "
        "Quando usar expressão indispensável em inglês, marque-a em itálico Markdown com asteriscos simples. "
        "Dê prioridade, em executive_highlights e watchpoints, a teses novas, viradas jurisprudenciais e decisões "
        "com impacto em matéria eleitoral em tramitação ou debate no Congresso Nacional; quando existirem, "
        "esses itens vêm antes dos demais."
    )
    try:
        response = openai_json_call(
            system_prompt,
            payload,
            schema=SUMMARY_SCHEMA,
            request_label="OpenAI síntese executiva",
            max_completion_tokens=2200,
        )
        return finalize_report_summary(
            ReportSummary(
                overview_callout=_normalize_ws(response.get("overview_callout")),
                executive_highlights=_unique_preserve_order(response.get("executive_highlights") or []),
                party_alerts=_unique_preserve_order(response.get("party_alerts") or []),
                lawyer_signals=_unique_preserve_order(response.get("lawyer_signals") or []),
                watchpoints=_unique_preserve_order(response.get("watchpoints") or []),
                closing_note=truncate_text(response.get("closing_note"), 220, suffix=""),
            ),
            cases,
            analyses,
            party_counter=party_counter,
            lawyer_counter=lawyer_counter,
        )
    except Exception as exc:
        LOGGER.warning("Falha na síntese executiva via OpenAI: %s. Usando fallback heurístico.", exc)
        return finalize_report_summary(
            fallback_report_summary(cases, analyses, party_counter, lawyer_counter, start_iso, end_iso),
            cases,
            analyses,
            party_counter=party_counter,
            lawyer_counter=lawyer_counter,
        )


def _chunk_text_for_notion(text: str, max_chars: int = MAX_NOTION_RICH_TEXT_CHARS) -> List[str]:
    if max_chars <= 0:
        return [text]
    source = str(text or "")
    if not source:
        return [""]
    chunks: List[str] = []
    remaining = source
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        cut = remaining.rfind(" ", 0, max_chars)
        if cut < int(max_chars * 0.6):
            cut = max_chars
        chunks.append(remaining[:cut])
        # Preserve the boundary whitespace in the next chunk so adjacent
        # rich_text items do not collapse words when Notion renders them.
        remaining = remaining[cut:]
    return chunks or [""]


def _italicize_known_english_expressions(text: str) -> str:
    formatted = str(text or "")
    for pattern in ENGLISH_ITALIC_PATTERNS:
        def _wrap(match: re.Match[str]) -> str:
            start, end = match.span()
            before = formatted[start - 1] if start > 0 else ""
            after = formatted[end] if end < len(formatted) else ""
            if before == "*" and after == "*":
                return match.group(0)
            return f"*{match.group(0)}*"

        formatted = pattern.sub(_wrap, formatted)
    return formatted


def _prepare_display_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    return _italicize_known_english_expressions(normalized)


def _find_next_inline_match(segment: str) -> tuple[str, re.Match[str]] | tuple[None, None]:
    candidates: List[tuple[str, re.Match[str]]] = []
    for token_type, pattern in (
        ("link", INLINE_LINK_RE),
        ("bold", INLINE_BOLD_RE),
        ("italic", INLINE_ITALIC_RE),
    ):
        match = pattern.search(segment)
        if match is not None:
            candidates.append((token_type, match))
    if not candidates:
        return None, None
    token_type, match = min(
        candidates,
        key=lambda item: (
            item[1].start(),
            {"link": 0, "bold": 1, "italic": 2}[item[0]],
        ),
    )
    return token_type, match


def _make_rich_text_item(
    content: str,
    *,
    bold: bool = False,
    italic: bool = False,
    link_url: str = "",
) -> Dict[str, Any]:
    text_obj: Dict[str, Any] = {"content": content}
    if link_url:
        text_obj["link"] = {"url": link_url}
    return {
        "type": "text",
        "text": text_obj,
        "annotations": {
            "bold": bool(bold),
            "italic": bool(italic),
            "strikethrough": False,
            "underline": False,
            "code": False,
            "color": "default",
        },
    }


def markdown_bold_to_notion_rich_text(text: str) -> List[Dict[str, Any]]:
    source = _prepare_display_text(text)
    if not source:
        return []

    rich: List[Dict[str, Any]] = []

    def _append_formatted_segment(segment: str, *, bold: bool = False, italic: bool = False) -> None:
        if not segment:
            return
        cursor = 0
        while cursor < len(segment):
            token_type, match = _find_next_inline_match(segment[cursor:])
            if match is None or token_type is None:
                tail = segment[cursor:]
                for piece in _chunk_text_for_notion(tail):
                    if piece:
                        rich.append(_make_rich_text_item(piece, bold=bold, italic=italic))
                return

            start = cursor + match.start()
            end = cursor + match.end()
            if start > cursor:
                plain = segment[cursor:start]
                for piece in _chunk_text_for_notion(plain):
                    if piece:
                        rich.append(_make_rich_text_item(piece, bold=bold, italic=italic))

            if token_type == "link":
                label = ""
                url = ""
                if match.group(1) is not None and match.group(2) is not None:
                    label = match.group(1)
                    url = match.group(2)
                elif match.group(3) is not None:
                    label = match.group(3)
                    url = match.group(3)
                for piece in _chunk_text_for_notion(label):
                    if piece:
                        rich.append(_make_rich_text_item(piece, bold=bold, italic=italic, link_url=url))
            elif token_type == "bold":
                _append_formatted_segment(match.group(1), bold=True or bold, italic=italic)
            elif token_type == "italic":
                _append_formatted_segment(match.group(1), bold=bold, italic=True or italic)
            cursor = end

    _append_formatted_segment(source)
    if not rich:
        _append_formatted_segment(source)
    return rich


def build_heading_1_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "heading_1",
        "heading_1": {"rich_text": markdown_bold_to_notion_rich_text(text), "color": "default"},
    }


def build_heading_2_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": markdown_bold_to_notion_rich_text(text), "color": "default"},
    }


def build_paragraph_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": markdown_bold_to_notion_rich_text(text), "color": "default"},
    }


def build_bulleted_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": markdown_bold_to_notion_rich_text(text), "color": "default"},
    }


def build_divider_block() -> Dict[str, Any]:
    return {"object": "block", "type": "divider", "divider": {}}


def build_callout_block(text: str, *, icon: str = "💡", color: str = "blue_background") -> Dict[str, Any]:
    icon_payload = {"type": "emoji", "emoji": icon or "💡"}
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": markdown_bold_to_notion_rich_text(text),
            "icon": icon_payload,
            "color": color,
        },
    }


def build_toggle_block(text: str, *, children: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    toggle_block = {
        "object": "block",
        "type": "toggle",
        "toggle": {"rich_text": markdown_bold_to_notion_rich_text(text), "color": "gray_background"},
    }
    nested_children = [dict(child) for child in (children or []) if isinstance(child, dict)]
    if nested_children:
        toggle_block["toggle"]["children"] = nested_children
    return toggle_block


def build_table_row_block(cells: Sequence[str]) -> Dict[str, Any]:
    rich_cells: List[List[Dict[str, Any]]] = []
    for cell in cells:
        rich = markdown_bold_to_notion_rich_text(str(cell or ""))
        rich_cells.append(rich if rich else [_make_rich_text_item("-")])
    return {"object": "block", "type": "table_row", "table_row": {"cells": rich_cells}}


def build_table_block(headers: Sequence[str]) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(headers),
            "has_column_header": True,
            "has_row_header": False,
            "children": [build_table_row_block(headers)],
        },
    }


def append_block_children(
    parent_id: str,
    children: Sequence[Dict[str, Any]],
    *,
    after_block_id: str = "",
    max_children_per_request: int = MAX_NOTION_APPEND_CHILDREN,
) -> List[Dict[str, Any]]:
    created: List[Dict[str, Any]] = []
    if not children:
        return created

    batch_size = max(1, min(int(max_children_per_request or 0), MAX_NOTION_APPEND_CHILDREN))
    for start in range(0, len(children), batch_size):
        chunk = list(children[start : start + batch_size])
        body: Dict[str, Any] = {"children": chunk}
        if after_block_id:
            body["position"] = {"type": "after_block", "after_block": {"id": after_block_id}}
        payload = notion_request("PATCH", f"/v1/blocks/{parent_id}/children", json_body=body)
        results = payload.get("results", [])
        if isinstance(results, list):
            created.extend([item for item in results if isinstance(item, dict)])
        after_block_id = ""
    return created


def list_block_children(block_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cursor = ""
    while True:
        params: Dict[str, Any] = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        payload = notion_request("GET", f"/v1/blocks/{block_id}/children", params=params)
        items = payload.get("results", [])
        if isinstance(items, list):
            out.extend([item for item in items if isinstance(item, dict)])
        if not bool(payload.get("has_more")):
            break
        cursor = _normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return out


def delete_block(block_id: str) -> None:
    notion_request("DELETE", f"/v1/blocks/{block_id}")


def ensure_target_page_ready(page_id: str, *, force_clear: bool) -> Dict[str, Any]:
    page_obj = notion_request("GET", f"/v1/pages/{page_id}")
    children = list_block_children(page_id)
    if children and not force_clear:
        raise RuntimeError(
            f"A página alvo já possui {len(children)} bloco(s). "
            "Use uma página em branco ou habilite a limpeza forçada."
        )
    if children and force_clear:
        LOGGER.warning("Limpando %d bloco(s) existentes na página alvo.", len(children))
        for item in children:
            block_id = _normalize_ws(item.get("id"))
            if block_id:
                delete_block(block_id)
    return page_obj


def create_table(
    parent_id: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    after_block_id: str = "",
) -> str:
    created = append_block_children(parent_id, [build_table_block(headers)], after_block_id=after_block_id)
    if not created:
        raise RuntimeError("Falha ao criar tabela no Notion.")
    table_id = _normalize_ws(created[0].get("id"))
    if not table_id:
        raise RuntimeError("Resposta do Notion sem ID ao criar tabela.")
    table_rows = [build_table_row_block(row) for row in rows]
    append_block_children(table_id, table_rows)
    return table_id


def score_label(score: int, risk_level: str) -> str:
    return f"{risk_emoji(risk_level)} {max(1, min(10, int(score)))}/10 | {risk_display_label(risk_level)}"


def callout_color_for_risk(risk_level: str) -> str:
    value = _normalize_ws(risk_level).casefold()
    if value == "critico":
        return "red_background"
    if value == "alto":
        return "orange_background"
    if value == "medio":
        return "blue_background"
    return "gray_background"


def _strip_leading_marker_once(text: str, marker: str) -> str:
    token = _normalize_ws(marker)
    if not token:
        return text
    pattern = rf"^\s*{re.escape(token)}(?=$|[\s,;:|/\-])(?:\s*[,;:|/\-]\s*|\s+)?"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()


def clean_case_heading_summary(
    text: str,
    *,
    process_label: str = "",
    class_label: str = "",
    local_label: str = "",
    extra_markers: Sequence[str] = (),
) -> str:
    summary = _normalize_ws(text)
    if not summary:
        return ""

    for pattern in (
        LEADING_DATE_PREFIX_RE,
        LEADING_COURT_PREFIX_RE,
        LEADING_MONOCRATIC_PREFIX_RE,
        LEADING_RELATOR_PREFIX_RE,
    ):
        summary = pattern.sub("", summary).strip()

    city_label = ""
    uf_label = ""
    if "/" in local_label:
        city_label, uf_label = [part.strip() for part in local_label.rsplit("/", 1)]

    markers = [class_label, process_label, local_label, city_label, uf_label, *list(extra_markers)]
    changed = True
    while changed and summary:
        changed = False
        before = summary
        for marker in markers:
            summary = _strip_leading_marker_once(summary, marker)
        changed = summary != before

    summary = re.sub(r"^[,;:|/\\-]+\s*", "", summary).strip()
    summary = re.sub(r"\s*[,;:|/\\-]+\s*$", "", summary).strip()
    summary = re.sub(r"\s+", " ", summary)
    return summary


def collect_case_heading_entities(case: CaseRecord, analysis: CaseAnalysis) -> List[str]:
    return _unique_preserve_order(list(analysis.public_figures) + list(analysis.parties) + list(case.partidos))


def extract_heading_candidates(case: CaseRecord, analysis: CaseAnalysis) -> List[tuple[str, str]]:
    first_sentence = re.split(r"(?<=[.!?])\s+", _normalize_ws(analysis.what_happened), maxsplit=1)[0]
    return [
        ("case.tema", case.tema),
        ("analysis.title", analysis.title),
        ("case.punchline", case.punchline),
        ("analysis.what_happened.first_sentence", first_sentence),
    ]


def normalize_heading_candidate(
    text: str,
    *,
    process_label: str = "",
    class_label: str = "",
    local_label: str = "",
    extra_markers: Sequence[str] = (),
) -> str:
    return clean_case_heading_summary(
        text,
        process_label=process_label,
        class_label=class_label,
        local_label=local_label,
        extra_markers=extra_markers,
    )


def looks_generic_case_heading_summary(text: str) -> bool:
    summary = _normalize_ws(text)
    if not summary:
        return False
    if GENERIC_HEADING_OPENING_RE.search(summary):
        return True
    lowered = summary.casefold()
    if GENERIC_HEADING_EXACT_RE.match(summary):
        return True

    tokens = re.findall(r"[0-9A-Za-zÀ-ÿ]+", summary, flags=re.UNICODE)
    if len(tokens) <= 1:
        return True

    generic_fragments = (
        "formulado em agravo",
        "interposto contra acórdão",
        "o acórdão regional havia",
        "acórdão regional havia",
        "efeito suspensivo formulado",
        "pedido de efeito suspensivo formulado",
        "nos termos da resolução tse",
        "nos termos da resolução tse nº",
        "exercício financeiro",
    )
    if any(fragment in lowered for fragment in generic_fragments):
        if "," not in summary and "(" not in summary:
            return True

    if lowered.startswith("prestação de contas") and "," not in summary and "(" not in summary:
        generic_endings = (
            "prestação de contas",
            "prestação de contas eleitorais",
            "prestação de contas eleitorais, vereador",
            "prestação de contas de campanha",
            "prestação de contas de campanha eleitoral",
            "prestação de contas de campanha, vereador",
            "prestação de contas partidárias",
            "prestação de contas partidárias eletrônicas",
            "prestação de contas de partido político",
            "prestação de contas partidárias de direção nacional",
            "prestação de contas de candidato a vereador",
            "prestação de contas de campanha eleitoral",
        )
        if lowered in generic_endings:
            return True

    return False


def looks_narrative_case_heading_summary(text: str) -> bool:
    return bool(NARRATIVE_HEADING_OPENING_RE.search(_normalize_ws(text)))


def build_semantic_microheadline(text: str, max_chars: int = HEADING_SUMMARY_MAX_CHARS) -> str:
    return truncate_heading_text(text, max_chars)


def score_heading_candidate(text: str) -> int:
    summary = _normalize_ws(text)
    if not summary:
        return -999

    score = 0
    if not looks_generic_case_heading_summary(summary):
        score += 25
    else:
        score -= 40
    if not looks_narrative_case_heading_summary(summary):
        score += 20
    else:
        score -= 30
    if not heading_summary_looks_incomplete(summary):
        score += 20
    else:
        score -= 35

    length = len(summary)
    if HEADING_SUMMARY_TARGET_MIN <= length <= HEADING_SUMMARY_TARGET_MAX:
        score += 12
    elif length <= HEADING_SUMMARY_MAX_CHARS:
        score += 6
    else:
        score -= min(20, length - HEADING_SUMMARY_MAX_CHARS)

    if _has_essential_heading_legal_reference(summary):
        score += 6
    elif HEADING_NONESSENTIAL_LEGAL_TAIL_RE.search(summary):
        score -= 8

    if HEADING_TEMPORAL_PAREN_RE.search(summary):
        score += 4
    if ";" in summary:
        score -= 4
    if summary.count(",") > 2:
        score -= 3

    return score


def select_best_heading_summary(
    case: CaseRecord,
    analysis: CaseAnalysis,
    *,
    extra_markers: Sequence[str] = (),
) -> str:
    class_label = _normalize_ws(case.sigla_classe or case.descricao_classe)
    local_label = _normalize_ws(case.local_label())
    best_summary = ""
    best_score = -999

    for source_name, raw_candidate in extract_heading_candidates(case, analysis):
        summary = normalize_heading_candidate(
            raw_candidate,
            process_label=case.process_label(),
            class_label=class_label,
            local_label=local_label,
            extra_markers=extra_markers,
        )
        if not summary:
            continue
        compact = build_semantic_microheadline(summary, max_chars=HEADING_SUMMARY_MAX_CHARS)
        score = score_heading_candidate(compact)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Heading candidate | source=%s | score=%d | text=%s", source_name, score, compact)
        if score > best_score:
            best_summary = compact
            best_score = score
        reasons = audit_heading_summary(compact)
        if source_name in {"case.tema", "analysis.title"} and not reasons:
            return compact
        if score >= HEADING_ACCEPTABLE_SCORE and not reasons:
            return compact

    return best_summary


def summarize_case_heading_from_analysis(
    case: CaseRecord,
    analysis: CaseAnalysis,
    *,
    extra_markers: Sequence[str] = (),
) -> str:
    return select_best_heading_summary(case, analysis, extra_markers=extra_markers)


def build_heading_entities_label(public_figures: Sequence[str], parties: Sequence[str]) -> str:
    entities = _unique_preserve_order(list(public_figures) + list(parties))
    if not entities:
        return ""
    return truncate_text(_safe_join(entities[:3], sep="; "), 90, suffix="")


def build_case_heading_entities(case: CaseRecord, analysis: CaseAnalysis) -> str:
    return build_heading_entities_label(analysis.public_figures, _unique_preserve_order(list(analysis.parties) + list(case.partidos)))


def _plain_heading_segment(text: str) -> str:
    return _normalize_ws(re.sub(r"\*+", "", str(text or "")))


def _dedupe_heading_segments(parts: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for part in parts:
        value = _normalize_ws(part)
        if not value:
            continue
        marker = _plain_heading_segment(value).casefold()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(value)
    return deduped


def compose_case_toggle_heading(
    *,
    index_label: str,
    score_display: str,
    process_label: str,
    class_label: str,
    local_label: str,
    entities_label: str,
    summary_text: str,
) -> str:
    parts = _dedupe_heading_segments([
        f"{index_label}. **{score_display}**",
        f"**{process_label}**" if process_label else "",
        f"**{class_label}**" if class_label else "",
        f"**{local_label}**" if local_label else "",
        entities_label,
        summary_text,
    ])
    return " | ".join(part for part in parts if _normalize_ws(part))


def build_case_toggle_heading(index: int, case: CaseRecord, analysis: CaseAnalysis) -> str:
    heading_entities = collect_case_heading_entities(case, analysis)
    return compose_case_toggle_heading(
        index_label=str(index),
        score_display=score_label(analysis_display_score(analysis), analysis.risk_level),
        process_label=case.process_label(),
        class_label=_normalize_ws(case.sigla_classe or case.descricao_classe),
        local_label=_normalize_ws(case.local_label()),
        entities_label=build_heading_entities_label(analysis.public_figures, _unique_preserve_order(list(analysis.parties) + list(case.partidos))),
        summary_text=summarize_case_heading_from_analysis(case, analysis, extra_markers=heading_entities),
    )


def build_case_toggle_children(case: CaseRecord, analysis: CaseAnalysis) -> List[Dict[str, Any]]:
    meta_line = (
        f"**Processo:** {case.process_label()} | **Classe:** {case.sigla_classe or case.descricao_classe or '-'} | "
        f"**Local:** {case.local_label()} | **Relator:** {case.relator or '-'} | **Resultado:** {_safe_join(case.resultado, sep=', ') or '-'}"
    )
    parties_line = _safe_join(analysis.parties or case.partidos, sep="; ") or "Sem partido explícito na base."
    parts_line = _safe_join(case.partes, sep="; ") or "-"
    notion_case_url = _normalize_ws(case.source_url)
    external_references = [
        url
        for url in extract_http_urls(analysis.source_notes + case.noticias)
        if not notion_case_url or url.casefold() != notion_case_url.casefold()
    ]
    # Apenas links reais (noticias confirmadas pela busca Gemini gravadas na base
    # e a pagina do caso). Link sintetico de busca no TSE foi removido: a SPA de
    # jurisprudencia ignora o fragmento e abria sempre a home.
    refs_segments: List[str] = []
    for url in external_references:
        domain = _normalize_ws(urlparse(url).netloc).replace("www.", "") or "link"
        refs_segments.append(f"[Notícia: {domain}]({url})")
    if notion_case_url:
        refs_segments.append(f"[Página do caso (Notion)]({notion_case_url})")
    refs_line = " | ".join(refs_segments)

    blocks: List[Dict[str, Any]] = [
        build_callout_block(
            _compact_report_field_text(
                analysis.why_relevant or "Sem observação adicional.",
                MAX_TOGGLE_WHY_RELEVANT_CHARS,
            ),
            icon="⚠️",
            color=callout_color_for_risk(analysis.risk_level),
        ),
        build_paragraph_block(meta_line),
        build_paragraph_block(
            f"**Partes/partidos:** {truncate_text(parts_line, 150, suffix='').rstrip(' ;,')} | "
            f"**Partidos identificados:** {truncate_text(parties_line, 90, suffix='').rstrip(' ;,')}"
        ),
        build_paragraph_block(
            f"**Decisão:** {_compact_report_field_text(analysis.what_happened, MAX_TOGGLE_FACTS_CHARS)}"
        ),
        build_paragraph_block(
            f"**Fundamento:** {_compact_report_field_text(analysis.legal_grounds, MAX_TOGGLE_LEGAL_CHARS)}"
        ),
        build_paragraph_block(
            f"**Efeito:** {_compact_report_field_text(analysis.consequence, MAX_TOGGLE_CONSEQUENCE_CHARS)}"
        ),
    ]
    if _is_material_alert_text(analysis.strategic_comment):
        blocks.append(
            build_paragraph_block(
                f"**Leitura estratégica:** {_compact_report_field_text(analysis.strategic_comment, MAX_TOGGLE_STRATEGIC_CHARS)}"
            )
        )
    if refs_line:
        blocks.append(build_paragraph_block(f"**Referências:** {refs_line}"))
    return blocks


def build_published_summary_rows(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[List[str]]:
    rows: List[List[str]] = []
    for idx, (case, analysis) in enumerate(build_publishable_case_pairs(cases, analyses), start=1):
        subjects = _safe_join(analysis.parties + analysis.public_figures, sep="; ") or _safe_join(case.partidos, sep="; ") or "-"
        rows.append(
            [
                str(idx),
                score_label(analysis_display_score(analysis), analysis.risk_level),
                f"{case.process_label()} | {case.local_label()}",
                f"[Abrir]({case.source_url})" if case.source_url else "-",
                _compact_report_field_text(analysis.what_happened or case.tema, MAX_TABLE_WHAT_HAPPENED_CHARS),
                _compact_report_field_text(analysis.consequence or subjects, MAX_TABLE_CONSEQUENCE_CHARS),
            ]
        )
    return rows


def build_published_toggle_blocks(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[Dict[str, Any]]:
    toggle_blocks: List[Dict[str, Any]] = []
    for idx, (case, analysis) in enumerate(build_publishable_case_pairs(cases, analyses), start=1):
        title = build_case_toggle_heading(idx, case, analysis)
        toggle_blocks.append(build_toggle_block(title, children=build_case_toggle_children(case, analysis)))
    return toggle_blocks


def build_inventory_rows(
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> List[List[str]]:
    published_keys = {case_resume_key(case) for case, _analysis in build_publishable_case_pairs(cases, analyses)}
    analysis_map = {analysis_resume_key(analysis): analysis for analysis in analyses}
    leftovers = [case for case in cases if case_resume_key(case) not in published_keys]

    def _sort_key(case: CaseRecord) -> tuple:
        analysis = analysis_map.get(case_resume_key(case))
        if analysis is not None:
            return analysis_sort_key(analysis)
        return (99, 0, case.process_label().casefold())

    leftovers.sort(key=_sort_key)
    rows: List[List[str]] = []
    for case in leftovers:
        analysis = analysis_map.get(case_resume_key(case))
        if analysis is not None:
            risk_cell = f"{risk_emoji(analysis.risk_level)} {analysis_display_score(analysis)}/10"
            theme_source = analysis.title or case.tema or case.punchline
        else:
            risk_cell = "⚪ -"
            theme_source = case.tema or case.punchline
        process_cell = (
            f"[{case.process_label()}]({case.source_url})" if _normalize_ws(case.source_url) else case.process_label()
        )
        rows.append(
            [
                process_cell,
                _normalize_ws(case.sigla_classe or case.descricao_classe) or "-",
                _normalize_ws(case.local_label()) or "-",
                _compact_report_field_text(theme_source, 130),
                risk_cell,
            ]
        )
    return rows


def append_inventory_section(
    page_id: str,
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
) -> int:
    rows = build_inventory_rows(cases, analyses)
    if not rows:
        return 0
    append_block_children(
        page_id,
        [
            build_heading_2_block(SECTION_INVENTORY_TITLE),
            build_paragraph_block(
                f"Inventário dos {len(rows)} demais processos do período, em ordem de risco. "
                "Garante que nenhum caso da base fique fora do relatório."
            ),
        ],
    )
    create_table(page_id, ["Processo", "Classe", "Local", "Tema", "Risco"], rows)
    return len(rows)


def build_period_stats_blocks(
    cases: Sequence[CaseRecord],
    published_count: int,
) -> List[Dict[str, Any]]:
    total = len(cases)
    if not total:
        return []
    with_party = sum(1 for case in cases if case.partidos)
    class_counter = Counter(
        _normalize_ws(case.sigla_classe or case.descricao_classe)
        for case in cases
        if _normalize_ws(case.sigla_classe or case.descricao_classe)
    )
    uf_counter = Counter(case.sigla_uf for case in cases if case.sigla_uf)
    text = f"**{total} processo(s) no período** | {published_count} em destaque | {with_party} com partido identificado."
    top_classes = ", ".join(f"{name} ({count})" for name, count in class_counter.most_common(6))
    if top_classes:
        text += f"\nClasses mais frequentes: {top_classes}."
    top_ufs = ", ".join(f"{name} ({count})" for name, count in uf_counter.most_common(6))
    if top_ufs:
        text += f"\nUFs: {top_ufs}."
    return [build_callout_block(text, icon="📊", color="gray_background")]


def append_published_sections(
    page_id: str,
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    *,
    after_block_id: str = "",
) -> Dict[str, Any]:
    intro_blocks = [
        build_heading_2_block(SECTION_CASES_TITLE),
        build_paragraph_block(
            f"Recorte dos {MAX_PUBLISHED_CASES} casos com maior interesse federal ou partidário material, ordenados por prioridade."
        ),
    ]
    created_intro = append_block_children(page_id, intro_blocks, after_block_id=after_block_id)
    last_after_id = _normalize_ws(created_intro[-1].get("id")) if created_intro else _normalize_ws(after_block_id)

    summary_rows = build_published_summary_rows(cases, analyses)
    table_id = create_table(
        page_id,
        ["#", "Prioridade", "Processo / local", "Link", "Ponto crítico", "Efeito"],
        summary_rows,
        after_block_id=last_after_id,
    )

    heading5_blocks = append_block_children(page_id, [build_heading_2_block(SECTION_NOTES_TITLE)], after_block_id=table_id)
    heading5_id = _normalize_ws(heading5_blocks[-1].get("id")) if heading5_blocks else table_id
    toggle_blocks = build_published_toggle_blocks(cases, analyses)
    append_block_children(
        page_id,
        toggle_blocks,
        after_block_id=heading5_id,
        max_children_per_request=MAX_NOTION_TOGGLES_PER_APPEND,
    )
    return {
        "table_id": table_id,
        "toggle_blocks_created": len(toggle_blocks),
        "case_rows_created": len(summary_rows),
    }


def append_delayed_cases_section(
    page_id: str,
    pairs: Sequence[tuple[CaseRecord, CaseAnalysis]],
    *,
    after_block_id: str = "",
) -> Dict[str, int]:
    """Seção 'Casos atrasados': decisões de semanas anteriores que só entraram na
    base nesta atualização, restritas a casos-alvo ou de risco crítico/alto.
    Heading SEM número para não colidir com os matchers das seções fixas 1..6.
    Sem pares, não escreve nada (defesa dupla com o orquestrador)."""
    if not pairs:
        return {"delayed_rows_created": 0, "delayed_toggles_created": 0}
    intro_blocks = [
        build_heading_2_block("⏳ Casos atrasados (decisões de semanas anteriores)"),
        build_paragraph_block(
            f"{len(pairs)} caso(s)-alvo ou de risco crítico/alto cuja decisão é de semana anterior, "
            "mas que só entrou na base nesta atualização. Listados aqui para não exigir a regeneração "
            "dos relatórios das semanas de origem."
        ),
    ]
    created_intro = append_block_children(page_id, intro_blocks, after_block_id=after_block_id)
    last_after_id = _normalize_ws(created_intro[-1].get("id")) if created_intro else _normalize_ws(after_block_id)

    rows: List[List[str]] = []
    for idx, (case, analysis) in enumerate(pairs, start=1):
        subjects = _safe_join(analysis.parties + analysis.public_figures, sep="; ") or _safe_join(case.partidos, sep="; ") or "-"
        rows.append(
            [
                str(idx),
                f"{_normalize_ws(case.data_decisao) or '-'} · {score_label(analysis_display_score(analysis), analysis.risk_level)}",
                f"{case.process_label()} | {case.local_label()}",
                f"[Abrir]({case.source_url})" if case.source_url else "-",
                _compact_report_field_text(analysis.what_happened or case.tema, MAX_TABLE_WHAT_HAPPENED_CHARS),
                _compact_report_field_text(analysis.consequence or subjects, MAX_TABLE_CONSEQUENCE_CHARS),
            ]
        )
    table_id = create_table(
        page_id,
        ["#", "Decisão / prioridade", "Processo / local", "Link", "Ponto crítico", "Efeito"],
        rows,
        after_block_id=last_after_id,
    )
    toggle_blocks = [
        build_toggle_block(
            build_case_toggle_heading(idx, case, analysis),
            children=build_case_toggle_children(case, analysis),
        )
        for idx, (case, analysis) in enumerate(pairs, start=1)
    ]
    append_block_children(
        page_id,
        toggle_blocks,
        after_block_id=table_id,
        max_children_per_request=MAX_NOTION_TOGGLES_PER_APPEND,
    )
    return {"delayed_rows_created": len(rows), "delayed_toggles_created": len(toggle_blocks)}


def build_methodology_blocks(title: str = SECTION_METHODOLOGY_TITLE) -> List[Dict[str, Any]]:
    return [
        build_heading_2_block(title),
        build_callout_block(
            "Relatório gerado a partir da base DJe consolidada no Notion, filtrada por dataDecisao no período exato solicitado. "
            "A unidade de análise é o processo judicial. A IA produziu a leitura estratégica com filtro editorial de interesse federal: "
            "altos cargos da República, partidos políticos, direção partidária, financiamento público, contas partidárias e teses com efeito replicável. "
            "Regra determinística de cobertura: processo que envolva Presidente, Vice-Presidente, governador, senador, deputado federal "
            "ou ministro de Estado recebe prioridade mínima 8/alto e entra nos destaques, independentemente do desfecho. "
            "Casos municipais rotineiros foram rebaixados ou excluídos quando não havia esse vetor material. Quando a base não trazia elemento explícito, "
            "o texto manteve linguagem conservadora.",
            icon="💡",
            color="yellow_background",
        ),
    ]


def _plain_rich_text(items: Any) -> str:
    return "".join(((item.get("text") or {}).get("content") or "") for item in (items or []) if isinstance(item, dict))


def parse_toggle_heading_text(title_text: str) -> Dict[str, str]:
    parts = [_normalize_ws(part) for part in str(title_text or "").split(" | ")]
    if len(parts) < 5:
        return {}
    first = parts[0]
    first_match = re.match(r"^(?P<index>\d+)\.\s*(?:[^\d\s]+\s+)?(?P<score_num>\d+/\d+)$", first)
    if not first_match:
        return {}

    score_num = _normalize_ws(first_match.group("score_num"))
    risk_label = _normalize_ws(parts[1] if len(parts) > 1 else "")
    process_label = _normalize_ws(parts[2] if len(parts) > 2 else "")
    class_label = _normalize_ws(parts[3] if len(parts) > 3 else "")
    local_label = _normalize_ws(parts[4] if len(parts) > 4 else "")
    entities_label = ""
    summary_text = ""
    if len(parts) >= 7:
        entities_label = _normalize_ws(parts[5])
        summary_text = _normalize_ws(" | ".join(parts[6:]))
    elif len(parts) == 6:
        summary_text = _normalize_ws(parts[5])

    return {
        "index_label": _normalize_ws(first_match.group("index")),
        "score_num": score_num,
        "risk_label": risk_label,
        "score_display": f"{score_num} | {risk_label}" if score_num and risk_label else "",
        "process_label": process_label,
        "class_label": class_label,
        "local_label": local_label,
        "entities_label": entities_label,
        "summary_text": summary_text,
    }


def audit_heading_summary(summary_text: str) -> List[str]:
    reasons: List[str] = []
    summary = _normalize_ws(summary_text)
    if not summary:
        return ["empty"]
    if looks_generic_case_heading_summary(summary):
        reasons.append("generic")
    if looks_narrative_case_heading_summary(summary):
        reasons.append("narrative")
    if heading_summary_looks_incomplete(summary):
        reasons.append("incomplete")
    return reasons


def _parse_toggle_detail_blocks(toggle_block_id: str) -> Dict[str, Any]:
    meta_text = ""
    parts_text = ""
    what_happened = ""
    why_relevant = ""
    legal_grounds = ""
    consequence = ""
    strategic_comment = ""
    lawyers_signal = ""
    reference_urls: List[str] = []
    for child in list_block_children(toggle_block_id):
        block_type = child.get("type")
        if block_type == "callout":
            why_relevant = _normalize_ws(_plain_rich_text(((child.get("callout") or {}).get("rich_text")) or []))
            continue
        if block_type != "paragraph":
            continue
        rich_text = (child.get("paragraph") or {}).get("rich_text") or []
        text = _plain_rich_text(rich_text)
        if text.startswith("Processo:"):
            meta_text = text
        elif text.startswith("Partes e partidos:"):
            parts_text = text
        elif text.startswith("O que ocorreu:"):
            what_happened = _normalize_ws(text.split(":", 1)[1] if ":" in text else text)
        elif text.startswith("Fundamentos jurídicos:"):
            legal_grounds = _normalize_ws(text.split(":", 1)[1] if ":" in text else text)
        elif text.startswith("Consequência aplicada:"):
            consequence = _normalize_ws(text.split(":", 1)[1] if ":" in text else text)
        elif text.startswith("Leitura estratégica:"):
            strategic_comment = _normalize_ws(text.split(":", 1)[1] if ":" in text else text)
        elif text.startswith("Banca e sinalização:"):
            lawyers_signal = _normalize_ws(text.split(":", 1)[1] if ":" in text else text)
        elif text.startswith("Referências:"):
            reference_urls = extract_http_urls([text]) + [
                _normalize_ws(((item.get("text") or {}).get("link") or {}).get("url"))
                for item in rich_text
                if isinstance(item, dict) and ((item.get("text") or {}).get("link") or {}).get("url")
            ]
            reference_urls = _unique_preserve_order(reference_urls)
    return {
        "meta_text": meta_text,
        "parts_text": parts_text,
        "what_happened": what_happened,
        "why_relevant": why_relevant,
        "legal_grounds": legal_grounds,
        "consequence": consequence,
        "strategic_comment": strategic_comment,
        "lawyers_signal": lawyers_signal,
        "reference_urls": reference_urls,
    }


def _parse_labeled_inline_fields(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in str(text or "").split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        out[_normalize_ws(key)] = _normalize_ws(value)
    return out


def _extract_public_figures_from_parts_text(parts_text: str) -> List[str]:
    parsed = _parse_labeled_inline_fields(parts_text)
    entries = _normalize_ws(parsed.get("Partes e partidos"))
    figures: List[str] = []
    institution_hints = (
        "ministério público",
        "partido",
        "federação",
        "coligação",
        "tribunal",
        "procuradoria",
        "município",
        "estado",
        "união ",
        "justiça eleitoral",
        "diretório",
        "prefeitura",
        "câmara",
    )
    for raw in entries.split(";"):
        item = _normalize_ws(raw)
        if not item:
            continue
        lowered = item.casefold()
        if any(hint in lowered for hint in institution_hints):
            continue
        if "/" in item and len(item) <= 20:
            continue
        if len(item.split()) > 8:
            continue
        figures.append(item)
    return _unique_preserve_order(figures[:3])


def _extract_parties_from_parts_text(parts_text: str) -> List[str]:
    parsed = _parse_labeled_inline_fields(parts_text)
    parties_value = _normalize_ws(parsed.get("Partidos identificados"))
    if parties_value.casefold().startswith("sem partido"):
        return []
    return [item for item in _unique_preserve_order(parties_value.split(";")) if item != "-"]


def _build_repair_analysis_stub(
    *,
    current_summary: str,
    what_happened: str,
    score_num: str,
    risk_label: str,
    public_figures: Sequence[str],
    parties: Sequence[str],
    page_id: str = "",
) -> CaseAnalysis:
    try:
        relevance_score = max(1, min(10, int((_normalize_ws(score_num).split("/", 1)[0] or "5"))))
    except Exception:
        relevance_score = 5
    return CaseAnalysis(
        case_id=page_id or "repair",
        title=current_summary,
        relevance_score=relevance_score,
        display_score=relevance_score,
        risk_level=_normalize_ws(risk_label).casefold() or "medio",
        includes_public_figure=bool(public_figures),
        includes_party=bool(parties),
        public_figures=list(public_figures),
        parties=list(parties),
        lawyers_signal="",
        what_happened=what_happened,
        legal_grounds="",
        consequence="",
        strategic_comment="",
        why_relevant="",
        source_notes=[],
        page_id=page_id,
    )


def _extract_source_case_from_reference_urls(reference_urls: Sequence[str]) -> Optional[CaseRecord]:
    for url in reference_urls:
        normalized_url = _normalize_ws(url)
        if "notion.so" not in normalized_url.casefold():
            continue
        try:
            page_id = extract_notion_id_from_url(normalized_url)
        except Exception:
            continue
        try:
            return build_case_record(notion_request("GET", f"/v1/pages/{page_id}"))
        except Exception as exc:
            LOGGER.debug("Falha ao carregar página-fonte %s: %s", normalized_url, exc)
            continue
    return None


def _extract_notion_reference_url(reference_urls: Sequence[str]) -> str:
    for url in reference_urls:
        normalized_url = _normalize_ws(url)
        if "notion.so" in normalized_url.casefold():
            return normalized_url
    return ""


def _table_cell_plain_text(cell: Any) -> str:
    if not isinstance(cell, list):
        return ""
    return _plain_rich_text(cell)


def _table_cell_links(cell: Any) -> List[str]:
    links: List[str] = []
    if not isinstance(cell, list):
        return links
    for item in cell:
        if not isinstance(item, dict):
            continue
        link = _normalize_ws((((item.get("text") or {}).get("link")) or {}).get("url"))
        if link:
            links.append(link)
    return _unique_preserve_order(links)


def load_cases_from_published_summary_table(page_id: str) -> List[CaseRecord]:
    blocks = list_block_children(page_id)
    idx_cases = _find_heading_2_index_any(blocks, CASE_SECTION_TITLES)
    idx_notes = _find_heading_2_index_any(blocks, NOTES_SECTION_TITLES)
    if idx_cases < 0:
        return []
    end_idx = idx_notes if idx_notes > idx_cases else len(blocks)
    table_blocks = [block for block in blocks[idx_cases + 1 : end_idx] if _normalize_ws(block.get("type")) == "table"]
    if not table_blocks:
        return []

    cases: List[CaseRecord] = []
    for row in list_block_children(_normalize_ws(table_blocks[0].get("id")))[1:]:
        if _normalize_ws(row.get("type")) != "table_row":
            continue
        cells = ((row.get("table_row") or {}).get("cells")) or []
        if len(cells) < 4:
            continue
        link_urls = _table_cell_links(cells[3])
        notion_url = next((url for url in link_urls if "notion.so" in url.casefold()), "")
        if not notion_url:
            continue
        try:
            source_page_id = extract_notion_id_from_url(notion_url)
            cases.append(build_case_record(notion_request("GET", f"/v1/pages/{source_page_id}")))
        except Exception as exc:
            LOGGER.debug("Falha ao reaproveitar caso da tabela publicada %s: %s", notion_url, exc)
            continue
    return cases


def _build_case_record_from_heading_and_details(
    parsed: Dict[str, str],
    details: Dict[str, Any],
    *,
    source_url: str = "",
) -> CaseRecord:
    local_label = _normalize_ws(parsed.get("local_label"))
    city = ""
    uf = ""
    if "/" in local_label:
        city, uf = [part.strip() for part in local_label.rsplit("/", 1)]
    else:
        uf = local_label
    parts_entries = _parse_labeled_inline_fields(details.get("parts_text", ""))
    parts_value = _normalize_ws(parts_entries.get("Partes e partidos"))
    partes = [item for item in _unique_preserve_order(parts_value.split(";")) if item and item != "-"]
    parties = _extract_parties_from_parts_text(details.get("parts_text", ""))
    page_id = ""
    notion_reference = _extract_notion_reference_url(details.get("reference_urls", []))
    if notion_reference:
        try:
            page_id = extract_notion_id_from_url(notion_reference)
        except Exception:
            page_id = ""
    meta_entries = _parse_labeled_inline_fields(details.get("meta_text", ""))
    relator = _normalize_ws(meta_entries.get("Relator"))
    resultado = _normalize_ws(meta_entries.get("Resultado"))
    process_label = _normalize_ws(parsed.get("process_label")) or page_id or "repair"
    external_urls = [
        url
        for url in details.get("reference_urls", [])
        if _normalize_ws(url) and "notion.so" not in _normalize_ws(url).casefold()
    ]
    return CaseRecord(
        case_id=page_id or process_label,
        page_id=page_id,
        source_url=source_url or notion_reference,
        data_decisao="",
        numero_unico=process_label,
        numero_processo="",
        ano_eleicao="",
        sigla_classe=_normalize_ws(parsed.get("class_label")),
        descricao_classe="",
        sigla_uf=_normalize_ws(uf),
        nome_municipio=_normalize_ws(city),
        descricao_tipo_decisao="",
        assuntos=[],
        partes=partes,
        partidos=parties,
        relator=relator,
        advogados=[],
        resultado=[resultado] if resultado else [],
        tema=_normalize_ws(parsed.get("summary_text")) or _normalize_ws(details.get("what_happened")),
        punchline=_normalize_ws(details.get("what_happened")),
        texto_decisao="",
        noticias=external_urls,
    )


def load_cases_and_analyses_from_published_page(page_id: str) -> Dict[str, Any]:
    blocks = list_block_children(page_id)
    idx_notes = _find_heading_2_index_any(blocks, NOTES_SECTION_TITLES)
    idx_methodology = _find_heading_2_index_any(blocks, METHODOLOGY_SECTION_TITLES)
    if idx_notes < 0:
        return {"cases": load_cases_from_published_summary_table(page_id), "analyses": []}

    end_idx = idx_methodology if idx_methodology > idx_notes else len(blocks)
    toggle_blocks = [block for block in blocks[idx_notes + 1 : end_idx] if _normalize_ws(block.get("type")) == "toggle"]
    if not toggle_blocks:
        return {"cases": load_cases_from_published_summary_table(page_id), "analyses": []}
    cases: List[CaseRecord] = []
    analyses: List[CaseAnalysis] = []
    for idx, block in enumerate(toggle_blocks, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(toggle_blocks):
            LOGGER.info(
                "Reaproveitando análises da página publicada | toggle=%d/%d",
                idx,
                len(toggle_blocks),
            )
        toggle_id = _normalize_ws(block.get("id"))
        title_text = _plain_rich_text((block.get("toggle") or {}).get("rich_text") or [])
        parsed = parse_toggle_heading_text(title_text)
        if not parsed:
            continue
        details = _parse_toggle_detail_blocks(toggle_id)
        source_case = _build_case_record_from_heading_and_details(
            parsed,
            details,
            source_url=_extract_notion_reference_url(details.get("reference_urls", [])),
        )
        figures = _extract_public_figures_from_parts_text(details.get("parts_text", ""))
        parties = _extract_parties_from_parts_text(details.get("parts_text", ""))
        analysis = _build_repair_analysis_stub(
            current_summary=_normalize_ws(parsed.get("summary_text")),
            what_happened=_normalize_ws(details.get("what_happened")),
            score_num=parsed.get("score_num", ""),
            risk_label=parsed.get("risk_label", ""),
            public_figures=figures,
            parties=parties,
            page_id=_normalize_ws(source_case.page_id),
        )
        analysis.why_relevant = _normalize_ws(details.get("why_relevant"))
        analysis.legal_grounds = _normalize_ws(details.get("legal_grounds"))
        analysis.consequence = _normalize_ws(details.get("consequence"))
        analysis.strategic_comment = _normalize_ws(details.get("strategic_comment"))
        analysis.lawyers_signal = _normalize_ws(details.get("lawyers_signal"))
        analysis.source_notes = list(details.get("reference_urls", []))
        cases.append(source_case)
        analyses.append(analysis)

    return {"cases": cases, "analyses": analyses}


def audit_page_case_headings(page_id: str) -> Dict[str, Any]:
    toggles = [block for block in list_block_children(page_id) if isinstance(block, dict) and block.get("type") == "toggle"]
    suspicious: List[Dict[str, Any]] = []
    for toggle in toggles:
        title_text = _plain_rich_text((toggle.get("toggle") or {}).get("rich_text") or [])
        parsed = parse_toggle_heading_text(title_text)
        if not parsed:
            continue
        reasons = audit_heading_summary(parsed.get("summary_text", ""))
        if not reasons:
            continue
        suspicious.append(
            {
                "block_id": _normalize_ws(toggle.get("id")),
                "index_label": parsed.get("index_label", ""),
                "process_label": parsed.get("process_label", ""),
                "reasons": reasons,
                "title": title_text,
            }
        )
    return {
        "toggle_count": len(toggles),
        "suspicious_count": len(suspicious),
        "suspicious": suspicious,
    }


def repair_page_case_headings(page_id: str, *, dry_run: bool = False) -> Dict[str, Any]:
    toggles = [block for block in list_block_children(page_id) if isinstance(block, dict) and block.get("type") == "toggle"]
    updated = 0
    skipped = 0
    inspected = 0
    suspicious_before = 0
    suspicious_after = 0
    for toggle in toggles:
        title_text = _plain_rich_text((toggle.get("toggle") or {}).get("rich_text") or [])
        parsed = parse_toggle_heading_text(title_text)
        if not parsed:
            continue
        inspected += 1
        current_summary = _normalize_ws(parsed.get("summary_text"))
        current_reasons = audit_heading_summary(current_summary)
        if current_reasons:
            suspicious_before += 1

        details = _parse_toggle_detail_blocks(_normalize_ws(toggle.get("id")))
        source_case = _extract_source_case_from_reference_urls(details.get("reference_urls", []))
        if source_case is None:
            skipped += 1
            continue

        figures = _extract_public_figures_from_parts_text(details.get("parts_text", ""))
        parties = _extract_parties_from_parts_text(details.get("parts_text", ""))
        analysis_stub = _build_repair_analysis_stub(
            current_summary=current_summary,
            what_happened=_normalize_ws(details.get("what_happened")),
            score_num=parsed.get("score_num", ""),
            risk_label=parsed.get("risk_label", ""),
            public_figures=figures,
            parties=parties,
            page_id=_normalize_ws(source_case.page_id),
        )

        extra_markers = _unique_preserve_order(figures + parties + [parsed.get("entities_label", "")])
        new_summary = select_best_heading_summary(source_case, analysis_stub, extra_markers=extra_markers)
        if not new_summary:
            skipped += 1
            continue

        new_reasons = audit_heading_summary(new_summary)
        current_score = score_heading_candidate(current_summary)
        new_score = score_heading_candidate(new_summary)
        should_replace = (
            not new_reasons
            and _normalize_ws(new_summary) != current_summary
            and (bool(current_reasons) or new_score > current_score)
        )
        if current_reasons and not new_reasons:
            suspicious_after += 0
        elif current_reasons:
            suspicious_after += 1

        if not should_replace:
            skipped += 1
            continue

        new_title = compose_case_toggle_heading(
            index_label=parsed.get("index_label", ""),
            score_display=parsed.get("score_display", ""),
            process_label=parsed.get("process_label", ""),
            class_label=parsed.get("class_label", ""),
            local_label=parsed.get("local_label", ""),
            entities_label=parsed.get("entities_label", ""),
            summary_text=new_summary,
        )
        if dry_run:
            LOGGER.info("[repair-headings] dry-run | processo=%s | novo=%s", parsed.get("process_label", ""), new_title)
        else:
            notion_request(
                "PATCH",
                f"/v1/blocks/{_normalize_ws(toggle.get('id'))}",
                json_body={
                    "toggle": {
                        "rich_text": markdown_bold_to_notion_rich_text(new_title),
                        "color": (toggle.get("toggle") or {}).get("color") or "gray_background",
                    }
                },
            )
        updated += 1

    audit_after = audit_page_case_headings(page_id)
    return {
        "toggle_count": len(toggles),
        "inspected": inspected,
        "updated": updated,
        "skipped": skipped,
        "suspicious_after": audit_after.get("suspicious_count", 0),
        "suspicious_before_estimate": suspicious_before,
        "dry_run": bool(dry_run),
    }


def _block_plain_text(block: Dict[str, Any]) -> str:
    block_type = _normalize_ws(block.get("type"))
    if not block_type:
        return ""
    payload = block.get(block_type) or {}
    if not isinstance(payload, dict):
        return ""
    return _plain_rich_text(payload.get("rich_text") or [])


def _find_heading_2_index(blocks: Sequence[Dict[str, Any]], title: str) -> int:
    target = _normalize_ws(title).casefold()
    for idx, block in enumerate(blocks):
        if _normalize_ws(block.get("type")) != "heading_2":
            continue
        if _block_plain_text(block).casefold() == target:
            return idx
    return -1


def _find_heading_2_index_any(blocks: Sequence[Dict[str, Any]], titles: Sequence[str]) -> int:
    indexes = [_find_heading_2_index(blocks, title) for title in titles]
    found = [idx for idx in indexes if idx >= 0]
    return min(found) if found else -1


def repair_published_sections(
    page_id: str,
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    blocks = list_block_children(page_id)
    idx_cases = _find_heading_2_index_any(blocks, CASE_SECTION_TITLES)
    idx_methodology = _find_heading_2_index_any(blocks, METHODOLOGY_SECTION_TITLES)
    if idx_cases < 0 or idx_methodology < 0:
        raise RuntimeError("Não foi possível localizar as seções publicadas na página para reparo focal.")

    to_delete = list(blocks[idx_cases:])
    previous_block_id = _normalize_ws(blocks[idx_cases - 1].get("id")) if idx_cases > 0 else ""
    summary_rows = build_published_summary_rows(cases, analyses)
    toggle_blocks = build_published_toggle_blocks(cases, analyses)
    lawyer_blocks = build_recurring_lawyer_section_blocks(cases, analyses)
    methodology_title = SECTION_METHODOLOGY_TITLE if lawyer_blocks else SECTION_METHODOLOGY_WITHOUT_LAWYERS_TITLE
    methodology_blocks = build_methodology_blocks(methodology_title)

    if dry_run:
        return {
            "deleted_blocks": len(to_delete),
            "case_rows_created": len(summary_rows),
            "toggle_blocks_created": len(toggle_blocks),
            "lawyer_blocks_created": len(lawyer_blocks),
            "dry_run": True,
        }

    for block in to_delete:
        block_id = _normalize_ws(block.get("id"))
        if block_id:
            delete_block(block_id)

    created_stats = append_published_sections(page_id, cases, analyses, after_block_id=previous_block_id)
    inventory_rows_created = append_inventory_section(page_id, cases, analyses)
    if lawyer_blocks:
        append_block_children(page_id, lawyer_blocks)
    append_block_children(page_id, methodology_blocks)
    return {
        "deleted_blocks": len(to_delete),
        "case_rows_created": int(created_stats.get("case_rows_created", 0)),
        "toggle_blocks_created": int(created_stats.get("toggle_blocks_created", 0)),
        "inventory_rows_created": inventory_rows_created,
        "lawyer_blocks_created": len(lawyer_blocks),
        "dry_run": False,
    }


def render_report_page(
    page_id: str,
    cases: Sequence[CaseRecord],
    analyses: Sequence[CaseAnalysis],
    summary: ReportSummary,
    *,
    start_iso: str,
    end_iso: str,
    source_database_url: str,
    party_counter: Counter[str],
    lawyer_counter: Counter[str],
    delayed_pairs: Sequence[tuple[CaseRecord, CaseAnalysis]] = (),
) -> Dict[str, Any]:
    focused_pairs = build_focused_reportable_case_pairs(cases, analyses)
    ranked = [analysis for _case, analysis in focused_pairs]
    reported_cases = [case for case, _analysis in focused_pairs]
    published_pairs = build_publishable_case_pairs(cases, analyses)
    high_priority_count = sum(1 for item in ranked if item.relevance_score >= 7)
    public_figure_count = sum(1 for item in ranked if item.includes_public_figure)
    with_party_count = sum(1 for item in reported_cases if item.partidos)
    accounts_count = sum(
        1
        for item in reported_cases
        if ACCOUNTS_HINT_RE.search(" ".join([item.tema, item.punchline, _safe_join(item.assuntos)]))
        or PARTY_FINANCE_SYSTEMIC_RE.search(" ".join([item.tema, item.punchline, _safe_join(item.assuntos)]))
    )
    uf_counter = Counter(case.sigla_uf for case in reported_cases if case.sigla_uf)

    blocks: List[Dict[str, Any]] = [
        build_heading_1_block(f"Relatório seletivo do DJe/TSE | {format_period_br(start_iso, end_iso)}"),
        build_paragraph_block(
            f"Base consultada: [DJe consolidado no Notion]({source_database_url}). "
            f"Período exato analisado: **{format_period_br(start_iso, end_iso)}**. "
            "Filtro editorial: interesse federal, altos cargos e partidos políticos."
        ),
        build_callout_block(summary.overview_callout, icon="💡", color="blue_background"),
        *build_period_stats_blocks(cases, len(published_pairs)),
        build_divider_block(),
        build_heading_2_block(SECTION_EXECUTIVE_TITLE),
    ]
    executive_alerts = build_executive_alert_items(summary)
    blocks.extend(build_bulleted_block(text) for text in executive_alerts)
    append_block_children(page_id, blocks)

    # Seção opcional de "casos atrasados" (decisões de semanas anteriores que só
    # chegaram neste import). Renderizada entre a leitura executiva e os casos
    # prioritários; ausente quando não há atrasados a destacar.
    delayed_stats = append_delayed_cases_section(page_id, list(delayed_pairs))

    published_stats = append_published_sections(page_id, cases, analyses)
    created_toggles = int(published_stats.get("toggle_blocks_created", 0))

    inventory_rows_created = append_inventory_section(page_id, cases, analyses)

    lawyer_blocks = build_recurring_lawyer_section_blocks(cases, analyses)
    if lawyer_blocks:
        append_block_children(page_id, lawyer_blocks)
    methodology_title = SECTION_METHODOLOGY_TITLE if lawyer_blocks else SECTION_METHODOLOGY_WITHOUT_LAWYERS_TITLE
    append_block_children(page_id, build_methodology_blocks(methodology_title))
    return {
        "toggle_blocks_created": created_toggles,
        "case_rows_created": int(published_stats.get("case_rows_created", 0)),
        "inventory_rows_created": inventory_rows_created,
        "delayed_rows_created": int(delayed_stats.get("delayed_rows_created", 0)),
        "delayed_toggles_created": int(delayed_stats.get("delayed_toggles_created", 0)),
        "executive_alert_count": len(executive_alerts),
        "lawyer_blocks_created": len(lawyer_blocks),
        "metrics": {
            "processos_total": len(cases),
            "processos_alta_prioridade": high_priority_count,
            "casos_com_partido": with_party_count,
            "casos_contas": accounts_count,
            "casos_com_figura_publica": public_figure_count,
            "casos_publicados": len(published_pairs),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera relatório estratégico do DJe/TSE em uma página do Notion.")
    parser.add_argument("--page-url", default="", help="URL da página alvo do Notion.")
    parser.add_argument("--source-database-url", default=DEFAULT_SOURCE_DATABASE_URL, help="URL da base do Notion.")
    parser.add_argument("--start-date", default="", help="Data inicial (DD/MM/AAAA ou AAAA-MM-DD).")
    parser.add_argument("--end-date", default="", help="Data final (DD/MM/AAAA ou AAAA-MM-DD).")
    parser.add_argument("--force-clear", action="store_true", help="Remove os blocos existentes da página alvo antes de publicar.")
    parser.add_argument("--dry-run", action="store_true", help="Executa consulta e análise, mas não escreve no Notion.")
    parser.add_argument("--no-gui", action="store_true", help="Não abre GUI; usa apenas argumentos CLI.")
    parser.add_argument(
        "--repair-headings-only",
        action="store_true",
        help="Recalcula e corrige apenas os cabeçalhos dos comentários por processo na página informada.",
    )
    parser.add_argument(
        "--audit-headings",
        action="store_true",
        help="Audita os cabeçalhos atuais da página informada sem escrever no Notion.",
    )
    parser.add_argument(
        "--repair-published-sections",
        choices=["table_comments"],
        default="",
        help="Reescreve apenas as seções publicadas de casos, advogados e metodologia da página informada.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignora checkpoint anterior e recomputa tudo.")
    parser.add_argument(
        "--refresh-analysis",
        action="store_true",
        help="Reprocessa análises OpenAI e síntese, mas reaproveita a query da base quando possível.",
    )
    parser.add_argument(
        "--refresh-ellipsis-only",
        action="store_true",
        help="Reprocessa apenas análises publicáveis ainda afetadas por reticências, reaproveitando o restante.",
    )
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help=f"Modelo OpenAI (padrao: {DEFAULT_OPENAI_MODEL}).")
    parser.add_argument(
        "--force-openai-fallback",
        action="store_true",
        help="Gera o relatório com heurística local, sem chamar a OpenAI.",
    )
    parser.add_argument(
        "--max-cases-per-batch",
        "--openai-batch-size",
        dest="max_cases_per_batch",
        type=int,
        default=DEFAULT_MAX_CASES_PER_BATCH,
        help=f"Quantidade maxima de processos por lote na OpenAI (padrao: {DEFAULT_MAX_CASES_PER_BATCH}).",
    )
    parser.add_argument(
        "--openai-max-workers",
        type=int,
        default=DEFAULT_OPENAI_MAX_WORKERS,
        help=f"Workers paralelos para lotes OpenAI (padrao: {DEFAULT_OPENAI_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--openai-target-rpm",
        type=int,
        default=DEFAULT_OPENAI_TARGET_RPM,
        help=f"Limite alvo de requests/min para OpenAI (padrao: {DEFAULT_OPENAI_TARGET_RPM}).",
    )
    parser.add_argument(
        "--max-openai-cases",
        type=int,
        default=DEFAULT_MAX_OPENAI_CASES,
        help=f"Maximo de processos analisados pela OpenAI por relatorio (padrao: {DEFAULT_MAX_OPENAI_CASES}; 0 desativa chamadas OpenAI no relatorio).",
    )
    parser.add_argument(
        "--openai-triage-threshold",
        type=int,
        default=DEFAULT_OPENAI_TRIAGE_THRESHOLD,
        help=f"Score minimo da triagem local para enviar processo ao GPT (padrao: {DEFAULT_OPENAI_TRIAGE_THRESHOLD}).",
    )
    parser.add_argument(
        "--disable-analysis-cache",
        action="store_true",
        help="Ignora o cache local de analises por processo.",
    )
    parser.add_argument(
        "--reclassify-top-band-with-openai",
        action="store_true",
        help="Opcional: recalibra apenas a diferença entre 9/10 e 10/10 para casos publicados, usando payload mínimo.",
    )
    parser.add_argument(
        "--enrich-news-gemini",
        action="store_true",
        help="Busca notícias via Gemini apenas para os casos publicados no relatório semanal.",
    )
    parser.add_argument(
        "--delayed-cases-file",
        default="",
        help=(
            "Opcional: caminho de um JSON com decisões de semanas anteriores recém-importadas "
            "(lista de {page_id, dataDecisao}). Quando informado, abre no relatório a seção "
            "'Casos atrasados' com os casos-alvo ou de risco crítico/alto desse delta, "
            "evitando regenerar as semanas antigas."
        ),
    )
    parser.add_argument("--gemini-api-key", default="", help="API key do Gemini/Google AI Studio para busca seletiva de notícias.")
    parser.add_argument("--gemini-key-file", default="Chave_Gemini.txt", help="Arquivo local com a chave do Gemini.")
    parser.add_argument("--gemini-model", default="", help="Modelo Gemini para busca seletiva de notícias.")
    parser.add_argument("--gemini-max-workers", type=int, default=2, help="Workers paralelos para busca Gemini seletiva.")
    parser.add_argument("--gemini-timeout", type=int, default=25, help="Timeout por chamada Gemini na busca seletiva.")
    parser.add_argument("--gemini-batch-size", type=int, default=8, help="Quantidade de casos por lote Gemini na busca seletiva.")
    parser.add_argument("--gemini-delay", type=float, default=0.2, help="Pausa entre lotes Gemini na busca seletiva.")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Arquivo de log.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs.")
    parser.add_argument("--verbose", action="store_true", help="Aumenta o detalhamento dos logs.")
    parser.add_argument("--debug", action="store_true", help="Ativa DEBUG.")
    return parser


def build_inputs_from_args_or_gui(args: argparse.Namespace) -> RunInputs:
    start_date_iso = ""
    end_date_iso = ""
    if _normalize_ws(args.start_date):
        start_date_iso = parse_user_date(args.start_date).isoformat()
    if _normalize_ws(args.end_date):
        end_date_iso = parse_user_date(args.end_date).isoformat()
    elif start_date_iso:
        end_date_iso = start_date_iso

    initials = load_last_inputs()
    merged_initials = {
        "page_url": _normalize_ws(args.page_url) or initials.get("page_url", ""),
        "source_database_url": _normalize_ws(args.source_database_url) or initials.get("source_database_url", DEFAULT_SOURCE_DATABASE_URL),
        "start_date_iso": start_date_iso or initials.get("start_date_iso", ""),
        "end_date_iso": end_date_iso or initials.get("end_date_iso", ""),
        "force_clear": bool(args.force_clear or initials.get("force_clear", False)),
    }

    if not args.no_gui:
        gui_value = run_gui(merged_initials)
        if gui_value is None:
            raise SystemExit(1)
        gui_value.dry_run = bool(args.dry_run)
        return gui_value

    page_url = _normalize_ws(args.page_url)
    database_url = _normalize_ws(args.source_database_url)
    if not start_date_iso:
        raise ValueError("Informe --start-date quando usar --no-gui.")
    if not end_date_iso:
        end_date_iso = start_date_iso
    if page_url:
        extract_notion_id_from_url(page_url)
    elif not args.dry_run:
        raise ValueError("Informe --page-url ou use a GUI.")
    extract_notion_id_from_url(database_url)
    if start_date_iso > end_date_iso:
        raise ValueError("A data inicial não pode ser posterior à data final.")
    return RunInputs(
        page_url=page_url,
        source_database_url=database_url,
        start_date_iso=start_date_iso,
        end_date_iso=end_date_iso,
        force_clear=bool(args.force_clear),
        dry_run=bool(args.dry_run),
        delayed_cases_file=_normalize_ws(getattr(args, "delayed_cases_file", "")),
    )


def initialize_clients(args: argparse.Namespace) -> Dict[str, Any]:
    global NOTION_CFG, OPENAI_CFG, NOTION_SESSION, OPENAI_SESSION, OPENAI_PACER

    notion_key = resolve_notion_key()
    if not notion_key:
        raise RuntimeError("Chave do Notion não encontrada.")
    pool_size = max(DEFAULT_HTTP_POOL_SIZE, int(getattr(args, "openai_max_workers", DEFAULT_OPENAI_MAX_WORKERS) or DEFAULT_OPENAI_MAX_WORKERS) + 4)
    NOTION_CFG = NotionConfig(token=notion_key)
    NOTION_SESSION = requests.Session()
    notion_adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    NOTION_SESSION.mount("https://", notion_adapter)
    NOTION_SESSION.mount("http://", notion_adapter)
    NOTION_SESSION.headers.update(
        {
            "Authorization": f"Bearer {NOTION_CFG.token}",
            "Notion-Version": NOTION_CFG.notion_version,
            "Content-Type": "application/json",
        }
    )

    force_openai_fallback = bool(getattr(args, "force_openai_fallback", False))
    if force_openai_fallback:
        openai_key = ""
        LOGGER.warning("Modo fallback OpenAI forçado. O relatório será gerado sem chamadas à OpenAI.")
    else:
        openai_key = resolve_openai_key()
    if openai_key:
        OPENAI_CFG = OpenAIConfig(
            api_key=openai_key,
            model=_normalize_ws(getattr(args, "model", "")) or DEFAULT_OPENAI_MODEL,
            target_rpm=max(0, int(getattr(args, "openai_target_rpm", DEFAULT_OPENAI_TARGET_RPM) or DEFAULT_OPENAI_TARGET_RPM)),
        )
        OPENAI_SESSION = requests.Session()
        openai_adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        OPENAI_SESSION.mount("https://", openai_adapter)
        OPENAI_SESSION.mount("http://", openai_adapter)
        OPENAI_SESSION.headers.update(
            {
                "Authorization": f"Bearer {OPENAI_CFG.api_key}",
                "Content-Type": "application/json",
            }
        )
        OPENAI_PACER = RequestPacer(target_rpm=OPENAI_CFG.target_rpm)
    else:
        OPENAI_CFG = None
        OPENAI_SESSION = None
        OPENAI_PACER = None
        LOGGER.warning("Chave OpenAI não encontrada. O script usará fallback heurístico.")
    LOGGER.info(
        "Perf config: openai_batch=%d | openai_workers=%d | openai_rpm=%d | max_openai_cases=%d | triage_threshold=%d | analysis_cache=%s | openai_timeout=%ss",
        int(getattr(args, "max_cases_per_batch", DEFAULT_MAX_CASES_PER_BATCH) or DEFAULT_MAX_CASES_PER_BATCH),
        int(getattr(args, "openai_max_workers", DEFAULT_OPENAI_MAX_WORKERS) or DEFAULT_OPENAI_MAX_WORKERS),
        int(getattr(args, "openai_target_rpm", DEFAULT_OPENAI_TARGET_RPM) or DEFAULT_OPENAI_TARGET_RPM),
        max(0, int(getattr(args, "max_openai_cases", DEFAULT_MAX_OPENAI_CASES))),
        int(getattr(args, "openai_triage_threshold", DEFAULT_OPENAI_TRIAGE_THRESHOLD)),
        not bool(getattr(args, "disable_analysis_cache", False)),
        int(OPENAI_CFG.timeout_s) if OPENAI_CFG is not None else 0,
    )
    return {"openai_enabled": OPENAI_CFG is not None}


def load_cases_and_analyses_for_inputs(
    inputs: RunInputs,
    args: argparse.Namespace,
    *,
    allow_reclassify_top_band: bool,
    prefer_published_page_fallback: bool = False,
) -> Dict[str, Any]:
    run_key = build_run_key(
        inputs,
        model=_normalize_ws(args.model) or DEFAULT_OPENAI_MODEL,
        max_cases_per_batch=max(1, int(args.max_cases_per_batch or DEFAULT_MAX_CASES_PER_BATCH)),
        max_openai_cases=max(0, int(getattr(args, "max_openai_cases", DEFAULT_MAX_OPENAI_CASES))),
        openai_triage_threshold=int(getattr(args, "openai_triage_threshold", DEFAULT_OPENAI_TRIAGE_THRESHOLD)),
    )
    checkpoint = load_matching_checkpoint(run_key, enabled=not bool(args.no_resume))
    report_payload_resume = load_matching_report_payload(inputs, enabled=not bool(args.no_resume))

    database_id = extract_notion_id_from_url(inputs.source_database_url)
    data_source_id = _normalize_ws(checkpoint.get("data_source_id"))
    if checkpoint.get("cases"):
        cases = [case_record_from_dict(item) for item in checkpoint.get("cases", []) if isinstance(item, dict)]
    elif prefer_published_page_fallback and inputs.page_url:
        page_loaded = load_cases_and_analyses_from_published_page(extract_notion_id_from_url(inputs.page_url))
        cases = list(page_loaded.get("cases") or [])
        analyses = list(page_loaded.get("analyses") or [])
        party_counter: Counter[str] = Counter()
        lawyer_counter: Counter[str] = Counter()
        for case in cases:
            party_counter.update(case.partidos)
            lawyer_counter.update(case.advogados)
        top_band_cache = {}
        if isinstance(checkpoint.get("top_band_reclassifications_by_cache_key"), dict):
            top_band_cache.update(checkpoint.get("top_band_reclassifications_by_cache_key") or {})
        if isinstance(report_payload_resume.get("top_band_reclassifications_by_cache_key"), dict):
            top_band_cache.update(report_payload_resume.get("top_band_reclassifications_by_cache_key") or {})
        if allow_reclassify_top_band and analyses:
            analyses, top_band_cache = reclassify_top_band_scores(
                cases,
                analyses,
                start_iso=inputs.start_date_iso,
                end_iso=inputs.end_date_iso,
                cached_map=top_band_cache,
            )
        if cases and analyses:
            LOGGER.info(
                "Reaproveitando %d caso(s) e %d análise(s) a partir da própria página publicada, sem nova query/análise.",
                len(cases),
                len(analyses),
            )
            return {
                "run_key": run_key,
                "checkpoint": checkpoint,
                "report_payload_resume": report_payload_resume,
                "database_id": database_id,
                "data_source_id": data_source_id,
                "cases": cases,
                "analyses": analyses,
                "party_counter": party_counter,
                "lawyer_counter": lawyer_counter,
                "top_band_reclassifications_by_cache_key": top_band_cache,
            }
        if cases and not analyses:
            LOGGER.info(
                "Reaproveitando %d caso(s) da página/tabela publicada; reanalisando apenas esse subconjunto.",
                len(cases),
            )
            analyses = analyze_cases(
                cases,
                start_iso=inputs.start_date_iso,
                end_iso=inputs.end_date_iso,
                party_counter=party_counter,
                lawyer_counter=lawyer_counter,
                max_cases_per_batch=max(1, int(args.max_cases_per_batch or DEFAULT_MAX_CASES_PER_BATCH)),
                openai_max_workers=max(1, int(args.openai_max_workers or DEFAULT_OPENAI_MAX_WORKERS)),
                max_openai_cases=max(0, int(getattr(args, "max_openai_cases", DEFAULT_MAX_OPENAI_CASES))),
                openai_triage_threshold=int(getattr(args, "openai_triage_threshold", DEFAULT_OPENAI_TRIAGE_THRESHOLD)),
                analysis_cache_enabled=not bool(getattr(args, "disable_analysis_cache", False) or getattr(args, "refresh_analysis", False)),
                resume_analysis_map={},
            )
            if allow_reclassify_top_band and analyses:
                analyses, top_band_cache = reclassify_top_band_scores(
                    cases,
                    analyses,
                    start_iso=inputs.start_date_iso,
                    end_iso=inputs.end_date_iso,
                    cached_map=top_band_cache,
                )
            return {
                "run_key": run_key,
                "checkpoint": checkpoint,
                "report_payload_resume": report_payload_resume,
                "database_id": database_id,
                "data_source_id": data_source_id,
                "cases": cases,
                "analyses": analyses,
                "party_counter": party_counter,
                "lawyer_counter": lawyer_counter,
                "top_band_reclassifications_by_cache_key": top_band_cache,
            }
    if not checkpoint.get("cases") and not cases:
        if not data_source_id:
            data_source_id = retrieve_database_and_datasource_id(database_id)
        cases = query_cases_by_period(data_source_id, inputs.start_date_iso, inputs.end_date_iso)

    party_counter: Counter[str] = Counter()
    lawyer_counter: Counter[str] = Counter()
    for case in cases:
        party_counter.update(case.partidos)
        lawyer_counter.update(case.advogados)

    resume_analysis_map, _ellipsis_refresh_count = build_resume_analysis_map(
        cases,
        checkpoint,
        report_payload_resume,
        refresh_analysis=bool(args.refresh_analysis),
        refresh_ellipsis_only=bool(args.refresh_ellipsis_only),
    )
    if len(resume_analysis_map) >= len(cases):
        analyses = [resume_analysis_map[case_resume_key(case)] for case in cases if case_resume_key(case) in resume_analysis_map]
    else:
        analyses = analyze_cases(
            cases,
            start_iso=inputs.start_date_iso,
            end_iso=inputs.end_date_iso,
            party_counter=party_counter,
            lawyer_counter=lawyer_counter,
            max_cases_per_batch=max(1, int(args.max_cases_per_batch or DEFAULT_MAX_CASES_PER_BATCH)),
            openai_max_workers=max(1, int(args.openai_max_workers or DEFAULT_OPENAI_MAX_WORKERS)),
            max_openai_cases=max(0, int(getattr(args, "max_openai_cases", DEFAULT_MAX_OPENAI_CASES))),
            openai_triage_threshold=int(getattr(args, "openai_triage_threshold", DEFAULT_OPENAI_TRIAGE_THRESHOLD)),
            analysis_cache_enabled=not bool(getattr(args, "disable_analysis_cache", False) or getattr(args, "refresh_analysis", False)),
            resume_analysis_map=resume_analysis_map,
        )

    top_band_cache = {}
    if isinstance(checkpoint.get("top_band_reclassifications_by_cache_key"), dict):
        top_band_cache.update(checkpoint.get("top_band_reclassifications_by_cache_key") or {})
    if isinstance(report_payload_resume.get("top_band_reclassifications_by_cache_key"), dict):
        top_band_cache.update(report_payload_resume.get("top_band_reclassifications_by_cache_key") or {})

    if allow_reclassify_top_band:
        analyses, top_band_cache = reclassify_top_band_scores(
            cases,
            analyses,
            start_iso=inputs.start_date_iso,
            end_iso=inputs.end_date_iso,
            cached_map=top_band_cache,
        )

    return {
        "run_key": run_key,
        "checkpoint": checkpoint,
        "report_payload_resume": report_payload_resume,
        "database_id": database_id,
        "data_source_id": data_source_id,
        "cases": cases,
        "analyses": analyses,
        "party_counter": party_counter,
        "lawyer_counter": lawyer_counter,
        "top_band_reclassifications_by_cache_key": top_band_cache,
    }


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if int(args.max_cases_per_batch) <= 0:
        parser.error("--max-cases-per-batch/--openai-batch-size deve ser maior que zero.")
    if int(args.openai_max_workers) <= 0:
        parser.error("--openai-max-workers deve ser maior que zero.")
    if int(args.openai_target_rpm) < 0:
        parser.error("--openai-target-rpm não pode ser negativo.")
    if int(args.max_openai_cases) < 0:
        parser.error("--max-openai-cases não pode ser negativo.")
    if not 0 <= int(args.openai_triage_threshold) <= 100:
        parser.error("--openai-triage-threshold deve ficar entre 0 e 100.")
    if int(args.gemini_max_workers) <= 0:
        parser.error("--gemini-max-workers deve ser maior que zero.")
    if int(args.gemini_timeout) <= 0:
        parser.error("--gemini-timeout deve ser maior que zero.")
    if int(args.gemini_batch_size) <= 0:
        parser.error("--gemini-batch-size deve ser maior que zero.")
    if float(args.gemini_delay) < 0:
        parser.error("--gemini-delay não pode ser negativo.")

    logger = configure_standard_logging(
        SCRIPT_STEM,
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or DEFAULT_LOG_FILE),
    )
    install_print_logger_bridge(globals(), logger)
    if int(args.max_openai_cases) == 0:
        args.force_openai_fallback = True
        LOGGER.info("--max-openai-cases=0: chamadas OpenAI desativadas para este relatório.")

    special_mode_count = sum(
        [
            1 if bool(args.repair_headings_only) else 0,
            1 if bool(args.audit_headings) else 0,
            1 if bool(_normalize_ws(args.repair_published_sections)) else 0,
        ]
    )
    if special_mode_count > 1:
        parser.error("--repair-headings-only, --audit-headings e --repair-published-sections são mutuamente exclusivos.")

    special_heading_mode = bool(args.repair_headings_only or args.audit_headings)
    if special_heading_mode:
        page_url = _normalize_ws(args.page_url)
        if not page_url:
            parser.error("--page-url é obrigatório em --repair-headings-only/--audit-headings.")
        page_id = extract_notion_id_from_url(page_url)
        initialize_clients(args)
        if bool(args.audit_headings):
            audit = audit_page_case_headings(page_id)
            LOGGER.info(
                "[audit-headings] página=%s | toggles=%d | suspeitos=%d",
                page_url,
                int(audit.get("toggle_count", 0)),
                int(audit.get("suspicious_count", 0)),
            )
            for item in (audit.get("suspicious") or [])[:20]:
                LOGGER.info(
                    "[audit-headings] #%s | processo=%s | razões=%s | título=%s",
                    item.get("index_label", "?"),
                    item.get("process_label", "?"),
                    ",".join(item.get("reasons") or []),
                    item.get("title", ""),
                )
            return 0

        repair = repair_page_case_headings(page_id, dry_run=bool(args.dry_run))
        LOGGER.info(
            "[repair-headings] página=%s | toggles=%d | inspecionados=%d | atualizados=%d | ignorados=%d | suspeitos_restantes=%d | dry_run=%s",
            page_url,
            int(repair.get("toggle_count", 0)),
            int(repair.get("inspected", 0)),
            int(repair.get("updated", 0)),
            int(repair.get("skipped", 0)),
            int(repair.get("suspicious_after", 0)),
            bool(repair.get("dry_run", False)),
        )
        return 0

    if _normalize_ws(args.repair_published_sections):
        inputs = build_inputs_from_args_or_gui(args)
        if not inputs.page_url:
            parser.error("--page-url é obrigatório em --repair-published-sections.")
        page_id = extract_notion_id_from_url(inputs.page_url)
        initialize_clients(args)
        loaded = load_cases_and_analyses_for_inputs(
            inputs,
            args,
            allow_reclassify_top_band=bool(args.reclassify_top_band_with_openai),
            prefer_published_page_fallback=True,
        )
        repair = repair_published_sections(
            page_id,
            loaded["cases"],
            loaded["analyses"],
            dry_run=bool(inputs.dry_run),
        )
        LOGGER.info(
            "[repair-published-sections] página=%s | removidos=%d | linhas_tabela=%d | comentários=%d | dry_run=%s",
            inputs.page_url,
            int(repair.get("deleted_blocks", 0)),
            int(repair.get("case_rows_created", 0)),
            int(repair.get("toggle_blocks_created", 0)),
            bool(repair.get("dry_run", False)),
        )
        return 0

    inputs = build_inputs_from_args_or_gui(args)
    initialize_clients(args)
    run_key = build_run_key(
        inputs,
        model=_normalize_ws(args.model) or DEFAULT_OPENAI_MODEL,
        max_cases_per_batch=max(1, int(args.max_cases_per_batch or DEFAULT_MAX_CASES_PER_BATCH)),
        max_openai_cases=max(0, int(args.max_openai_cases)),
        openai_triage_threshold=int(args.openai_triage_threshold),
    )
    checkpoint = load_matching_checkpoint(run_key, enabled=not bool(args.no_resume))
    if checkpoint:
        LOGGER.info(
            "Checkpoint compatível encontrado | stage=%s | atualizado_em=%s",
            checkpoint.get("stage", "?"),
            checkpoint.get("updated_at_utc", "?"),
        )
    report_payload_resume = load_matching_report_payload(inputs, enabled=not bool(args.no_resume))
    if report_payload_resume:
        LOGGER.info(
            "Artefato de relatório compatível encontrado | concluído_em=%s",
            report_payload_resume.get("finished_at_utc", "?"),
        )

    payload_report: Dict[str, Any] = {
        "script": SCRIPT_FILE,
        "started_at_utc": utc_now_iso(),
        "last_inputs": asdict(inputs),
        "checkpoint_file": str(CHECKPOINT_FILE),
        "openai_model": _normalize_ws(args.model) or DEFAULT_OPENAI_MODEL,
    }

    page_id = ""
    if inputs.page_url:
        page_id = extract_notion_id_from_url(inputs.page_url)
        payload_report["page_id"] = page_id
    database_id = extract_notion_id_from_url(inputs.source_database_url)
    payload_report["database_id"] = database_id

    if page_id and not inputs.dry_run:
        LOGGER.info("[Stage 1/5] Validando página alvo...")
        ensure_target_page_ready(page_id, force_clear=bool(inputs.force_clear))

    LOGGER.info("[Stage 2/5] Resolvendo data source e consultando base...")
    data_source_id = _normalize_ws(checkpoint.get("data_source_id"))
    if data_source_id:
        LOGGER.info("Reaproveitando data_source_id do checkpoint.")
    else:
        data_source_id = retrieve_database_and_datasource_id(database_id)
    payload_report["data_source_id"] = data_source_id
    if checkpoint.get("cases"):
        cases = [case_record_from_dict(item) for item in checkpoint.get("cases", []) if isinstance(item, dict)]
        LOGGER.info("Reaproveitando %d processo(s) do checkpoint, sem nova query na base.", len(cases))
    else:
        cases = query_cases_by_period(data_source_id, inputs.start_date_iso, inputs.end_date_iso)
        save_checkpoint(
            run_key,
            {
                "stage": "queried",
                "last_inputs": asdict(inputs),
                "database_id": database_id,
                "data_source_id": data_source_id,
                "cases": [asdict(case) for case in cases],
                "analyses_by_case_id": {},
            },
        )
    payload_report["case_count"] = len(cases)

    if not cases:
        LOGGER.info("Nenhum processo encontrado no período %s.", format_period_br(inputs.start_date_iso, inputs.end_date_iso))
        payload_report["finished_at_utc"] = utc_now_iso()
        write_json_atomic(REPORT_FILE, payload_report)
        if page_id and not inputs.dry_run:
            blocks = [
                build_heading_1_block(f"Relatório estratégico do DJe | {format_period_br(inputs.start_date_iso, inputs.end_date_iso)}"),
                build_callout_block(
                    f"Nenhum processo foi encontrado na base para o período exato {format_period_br(inputs.start_date_iso, inputs.end_date_iso)}.",
                    icon="💡",
                    color="gray_background",
                ),
            ]
            append_block_children(page_id, blocks)
        return 0

    party_counter: Counter[str] = Counter()
    lawyer_counter: Counter[str] = Counter()
    for case in cases:
        party_counter.update(case.partidos)
        lawyer_counter.update(case.advogados)

    resume_analysis_map, ellipsis_refresh_count = build_resume_analysis_map(
        cases,
        checkpoint,
        report_payload_resume,
        refresh_analysis=bool(args.refresh_analysis),
        refresh_ellipsis_only=bool(args.refresh_ellipsis_only),
    )
    if bool(args.refresh_analysis) and checkpoint.get("analyses_by_case_id"):
        LOGGER.info("Ignorando análises salvas no checkpoint por causa de --refresh-analysis.")
    elif bool(args.refresh_ellipsis_only):
        LOGGER.info(
            "Refresh seletivo de reticências | reaproveitadas=%d | reprocessar=%d",
            len(resume_analysis_map),
            ellipsis_refresh_count,
        )

    def _serialize_analysis_map(current_map: Dict[str, CaseAnalysis]) -> Dict[str, Any]:
        return {key: asdict(item) for key, item in current_map.items()}

    def _serialize_analysis_list(items: Sequence[CaseAnalysis]) -> List[Dict[str, Any]]:
        return [asdict(item) for item in items]

    def _save_analysis_checkpoint(current_map: Dict[str, CaseAnalysis], batch_idx: int, total_batches: int) -> None:
        save_checkpoint(
            run_key,
            {
                "stage": "analysing",
                "last_inputs": asdict(inputs),
                "database_id": database_id,
                "data_source_id": data_source_id,
                "cases": [asdict(case) for case in cases],
                "analyses_by_case_id": _serialize_analysis_map(current_map),
                "analysis_progress": {
                    "completed_cases": len(current_map),
                    "total_cases": len(cases),
                    "batch_idx": batch_idx,
                    "total_batches": total_batches,
                },
            },
        )

    LOGGER.info("[Stage 3/5] Analisando %d processo(s)...", len(cases))
    analyses = analyze_cases(
        cases,
        start_iso=inputs.start_date_iso,
        end_iso=inputs.end_date_iso,
        party_counter=party_counter,
        lawyer_counter=lawyer_counter,
        max_cases_per_batch=max(1, int(args.max_cases_per_batch or DEFAULT_MAX_CASES_PER_BATCH)),
        openai_max_workers=max(1, int(args.openai_max_workers or DEFAULT_OPENAI_MAX_WORKERS)),
        max_openai_cases=max(0, int(args.max_openai_cases)),
        openai_triage_threshold=int(args.openai_triage_threshold),
        analysis_cache_enabled=not bool(args.disable_analysis_cache or args.refresh_analysis),
        resume_analysis_map=resume_analysis_map,
        progress_callback=_save_analysis_checkpoint,
    )
    top_band_cache = {}
    if isinstance(checkpoint.get("top_band_reclassifications_by_cache_key"), dict):
        top_band_cache.update(checkpoint.get("top_band_reclassifications_by_cache_key") or {})
    if isinstance(report_payload_resume.get("top_band_reclassifications_by_cache_key"), dict):
        top_band_cache.update(report_payload_resume.get("top_band_reclassifications_by_cache_key") or {})
    if bool(args.reclassify_top_band_with_openai):
        analyses, top_band_cache = reclassify_top_band_scores(
            cases,
            analyses,
            start_iso=inputs.start_date_iso,
            end_iso=inputs.end_date_iso,
            cached_map=top_band_cache,
        )
    payload_report["analyses"] = _serialize_analysis_list(analyses)
    focused_reportable_pairs = build_focused_reportable_case_pairs(cases, analyses)
    reportable_cases = [case for case, _analysis in focused_reportable_pairs]
    reportable_analyses = [analysis for _case, analysis in focused_reportable_pairs]
    publishable_pairs = build_publishable_case_pairs(cases, analyses)
    publishable_analyses = [analysis for _case, analysis in publishable_pairs]
    if bool(getattr(args, "enrich_news_gemini", False)):
        payload_report["news_enrichment"] = enrich_report_cases_with_gemini_news(
            [case for case, _analysis in publishable_pairs],
            args,
            dry_run=inputs.dry_run,
        )
    reportable_party_counter: Counter[str] = Counter()
    reportable_lawyer_counter: Counter[str] = Counter()
    for case in reportable_cases:
        reportable_party_counter.update(case.partidos)
        reportable_lawyer_counter.update(case.advogados)
    payload_report["reportable_case_count"] = len(reportable_analyses)
    payload_report["published_case_count"] = len(publishable_analyses)
    payload_report["filtered_out_of_focus_case_count"] = max(0, len([item for item in analyses if is_reportable_analysis(item)]) - len(reportable_analyses))
    payload_report["filtered_low_case_count"] = sum(1 for item in analyses if _normalize_ws(item.risk_level).casefold() == "baixo")
    payload_report["filtered_medium_case_count"] = sum(1 for item in analyses if _normalize_ws(item.risk_level).casefold() == "medio")
    payload_report["top_band_reclassifications_by_cache_key"] = top_band_cache
    save_checkpoint(
        run_key,
        {
            "stage": "analysed",
            "last_inputs": asdict(inputs),
            "database_id": database_id,
            "data_source_id": data_source_id,
            "cases": [asdict(case) for case in cases],
            "analyses_by_case_id": {analysis_resume_key(item): asdict(item) for item in analyses},
            "analyses": _serialize_analysis_list(analyses),
            "top_band_reclassifications_by_cache_key": top_band_cache,
        },
    )

    LOGGER.info("[Stage 4/5] Gerando síntese executiva...")
    if (
        checkpoint.get("summary")
        and len(resume_analysis_map) >= len(cases)
        and not bool(args.refresh_analysis)
        and not bool(args.refresh_ellipsis_only and ellipsis_refresh_count > 0)
    ):
        cached_summary = report_summary_from_dict(checkpoint.get("summary", {}))
        if report_summary_has_incomplete_items(cached_summary):
            LOGGER.info("Síntese executiva do checkpoint contém itens incompletos. Regerando via API.")
            summary = summarize_report(
                reportable_cases,
                reportable_analyses,
                party_counter=reportable_party_counter,
                lawyer_counter=reportable_lawyer_counter,
                start_iso=inputs.start_date_iso,
                end_iso=inputs.end_date_iso,
            )
        else:
            summary = cached_summary
            LOGGER.info("Reaproveitando síntese executiva do checkpoint.")
    else:
        summary = summarize_report(
            reportable_cases,
            reportable_analyses,
            party_counter=reportable_party_counter,
            lawyer_counter=reportable_lawyer_counter,
            start_iso=inputs.start_date_iso,
            end_iso=inputs.end_date_iso,
        )
    summary = finalize_report_summary(
        summary,
        reportable_cases,
        reportable_analyses,
        party_counter=reportable_party_counter,
        lawyer_counter=reportable_lawyer_counter,
    )
    payload_report["summary"] = asdict(summary)
    save_checkpoint(
        run_key,
        {
            "stage": "summarised",
            "last_inputs": asdict(inputs),
            "database_id": database_id,
            "data_source_id": data_source_id,
            "cases": [asdict(case) for case in cases],
            "analyses_by_case_id": {analysis_resume_key(item): asdict(item) for item in analyses},
            "analyses": _serialize_analysis_list(analyses),
            "summary": asdict(summary),
            "top_band_reclassifications_by_cache_key": top_band_cache,
        },
    )

    # Casos atrasados: decisões de semanas anteriores recém-importadas (delta),
    # restritas a alvo ou risco crítico/alto. Processa SÓ o delta (poucos casos),
    # evitando regenerar as semanas antigas inteiras.
    delayed_pairs: List[tuple[CaseRecord, CaseAnalysis]] = []
    if _normalize_ws(inputs.delayed_cases_file):
        existing_keys = {case_resume_key(case) for case in cases}
        delayed_cases = load_delayed_cases(inputs.delayed_cases_file, exclude_keys=existing_keys)
        if delayed_cases:
            LOGGER.info("[Atrasados] %d decisão(ões) de semanas anteriores recebidas para triagem.", len(delayed_cases))
            delayed_party_counter: Counter[str] = Counter()
            delayed_lawyer_counter: Counter[str] = Counter()
            for case in delayed_cases:
                delayed_party_counter.update(case.partidos)
                delayed_lawyer_counter.update(case.advogados)
            delayed_analyses = analyze_cases(
                delayed_cases,
                start_iso=inputs.start_date_iso,
                end_iso=inputs.end_date_iso,
                party_counter=delayed_party_counter,
                lawyer_counter=delayed_lawyer_counter,
                max_cases_per_batch=max(1, int(args.max_cases_per_batch or DEFAULT_MAX_CASES_PER_BATCH)),
                openai_max_workers=max(1, int(args.openai_max_workers or DEFAULT_OPENAI_MAX_WORKERS)),
                max_openai_cases=max(0, int(args.max_openai_cases)),
                openai_triage_threshold=int(args.openai_triage_threshold),
                analysis_cache_enabled=not bool(args.disable_analysis_cache or args.refresh_analysis),
            )
            delayed_pairs = build_delayed_target_pairs(delayed_cases, delayed_analyses)
            payload_report["delayed_target_cases"] = [
                {
                    "page_id": case.page_id,
                    "processo": case.process_label(),
                    "data_decisao": case.data_decisao,
                    "risk_level": analysis.risk_level,
                    "is_target_office": is_target_office_case(case, analysis),
                }
                for case, analysis in delayed_pairs
            ]
            LOGGER.info(
                "[Atrasados] %d caso(s) na seção (alvo ou risco crítico/alto) de %d recebido(s).",
                len(delayed_pairs),
                len(delayed_cases),
            )

    if page_id and not inputs.dry_run:
        LOGGER.info("[Stage 5/5] Publicando relatório no Notion...")
        publish_stats = render_report_page(
            page_id,
            cases,
            reportable_analyses,
            summary,
            start_iso=inputs.start_date_iso,
            end_iso=inputs.end_date_iso,
            source_database_url=inputs.source_database_url,
            party_counter=reportable_party_counter,
            lawyer_counter=reportable_lawyer_counter,
            delayed_pairs=delayed_pairs,
        )
        payload_report["publish_stats"] = publish_stats
    else:
        LOGGER.info("[Stage 5/5] Dry run ativo; nenhuma escrita foi feita no Notion.")

    payload_report["finished_at_utc"] = utc_now_iso()
    write_json_atomic(REPORT_FILE, payload_report)
    save_checkpoint(
        run_key,
        {
            "stage": "finished",
            "last_inputs": asdict(inputs),
            "database_id": database_id,
            "data_source_id": data_source_id,
            "cases": [asdict(case) for case in cases],
            "analyses_by_case_id": {analysis_resume_key(item): asdict(item) for item in analyses},
            "analyses": _serialize_analysis_list(analyses),
            "summary": asdict(summary),
            "top_band_reclassifications_by_cache_key": top_band_cache,
            "payload_report_file": str(REPORT_FILE),
            "payload_report_finished_at_utc": payload_report["finished_at_utc"],
        },
    )
    LOGGER.info(
        "Concluído | processos=%d | período=%s | dry_run=%s",
        len(cases),
        format_period_br(inputs.start_date_iso, inputs.end_date_iso),
        bool(inputs.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
