#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical TRF1 Notion agent.

Scope:
- classe
- relator(a)
- ramo_do_direito
- informativo
- subramo_do_direito

Strategy:
- process only problematic rows
- deterministic PDF extraction first
- low-cost OpenAI fallback
- direct Notion updates (default apply=True)
- checkpoint + reports
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

try:
    from openai import OpenAI
    import openai
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    openai = None  # type: ignore[assignment]

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from TRF1_pdf_to_csv_viaAPI_openai import extract_rows_from_pdf


DEFAULT_DATABASE_ID = "30d721955c6480679e7ae359ff1c19e7"
DEFAULT_NOTION_VERSION = "2022-06-28"
NOTION_API_BASE = "https://api.notion.com/v1"
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_CHECKPOINT_FILE = Path(".checkpoint/trf1_canonical_checkpoint.json")
DEFAULT_CONFIG_DIR = Path("canon_config")

CRITICAL_PROPERTIES = (
    "classe",
    "relator(a)",
    "ramo_do_direito",
    "informativo",
    "subramo_do_direito",
)

NUMERIC_INFORMATIVO_RE = re.compile(r"^\d{2,4}$")
RELATOR_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:"
    r"des\.?\s*(?:fed(?:eral)?\.?)?\s+|"
    r"desembargador(?:a)?\s+|"
    r"min\.?\s+|ministro(?:a)?\s+|"
    r"ju[ií]z(?:a)?\s+federal\s+|"
    r"ju[ií]z(?:a)?\s+"
    r")+"
)


@dataclass
class PageRecord:
    page_id: str
    page_url: str
    numero_processo: str
    tema: str
    texto_do_boletim: str
    classe: str
    relator: str
    ramo: str
    informativo: str
    subramos: List[str]


@dataclass
class PdfCandidate:
    source_pdf: str
    informativo: str
    numero_processo: str
    classe: str
    relator: str
    texto_do_boletim: str


@dataclass
class ResolutionResult:
    page: PageRecord
    problematic_columns: List[str]
    resolved: Dict[str, Any]
    unresolved_columns: List[str]
    notes: List[str]
    llm_used: bool


@dataclass
class RuntimeStats:
    pages_total: int = 0
    pages_processed: int = 0
    pages_problematic: int = 0
    pages_updated: int = 0
    pages_update_failed: int = 0
    unresolved_pages: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    schema_cleanup_attempted: bool = False
    schema_cleanup_ok: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages_total": self.pages_total,
            "pages_processed": self.pages_processed,
            "pages_problematic": self.pages_problematic,
            "pages_updated": self.pages_updated,
            "pages_update_failed": self.pages_update_failed,
            "unresolved_pages": self.unresolved_pages,
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "schema_cleanup_attempted": self.schema_cleanup_attempted,
            "schema_cleanup_ok": self.schema_cleanup_ok,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def fold_text(value: str) -> str:
    text = normalize_ws(value).casefold()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return text


def slug_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", fold_text(value))


def tokenize_for_similarity(value: str) -> Set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]{2,}", fold_text(value))}


def jaccard_similarity(a: str, b: str) -> float:
    ta = tokenize_for_similarity(a)
    tb = tokenize_for_similarity(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    if not union:
        return 0.0
    return float(len(inter)) / float(len(union))


def split_notion_multiselect(value: str) -> List[str]:
    if not value:
        return []
    return [normalize_ws(x) for x in value.split(",") if normalize_ws(x)]


def read_secret_from_file(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return ""
    first = raw.splitlines()[0].strip()
    if "=" in first:
        first = first.split("=", 1)[1].strip()
    return first.strip("\"' ")


def load_text_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def ensure_csv_header(path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()


def append_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> int:
    rows_list = list(rows)
    if not rows_list:
        return 0
    ensure_csv_header(path, fieldnames)
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writerows(rows_list)
    return len(rows_list)


def notion_plain_text_from_property(prop: Dict[str, Any]) -> str:
    if not isinstance(prop, dict):
        return ""
    p_type = prop.get("type")
    if p_type == "title":
        items = prop.get("title", [])
        return "".join(str(it.get("plain_text", "")) for it in items if isinstance(it, dict)).strip()
    if p_type == "rich_text":
        items = prop.get("rich_text", [])
        return "".join(str(it.get("plain_text", "")) for it in items if isinstance(it, dict)).strip()
    if p_type == "select":
        select_obj = prop.get("select")
        if isinstance(select_obj, dict):
            return normalize_ws(select_obj.get("name", ""))
        return ""
    if p_type == "multi_select":
        arr = prop.get("multi_select", [])
        vals: List[str] = []
        for item in arr:
            if isinstance(item, dict):
                name = normalize_ws(item.get("name", ""))
                if name:
                    vals.append(name)
        return ", ".join(vals)
    if p_type == "date":
        date_obj = prop.get("date")
        if isinstance(date_obj, dict):
            return normalize_ws(date_obj.get("start", ""))
        return ""
    if p_type == "url":
        return normalize_ws(prop.get("url", ""))
    return ""


def notion_option_set_from_schema(schema: Dict[str, Any], prop_name: str, option_type: str) -> Set[str]:
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(props, dict):
        return set()
    meta = props.get(prop_name, {})
    if not isinstance(meta, dict):
        return set()
    subtype = meta.get(option_type, {})
    if not isinstance(subtype, dict):
        return set()
    out: Set[str] = set()
    for option in subtype.get("options", []):
        if isinstance(option, dict):
            name = normalize_ws(option.get("name", ""))
            if name:
                out.add(name)
    return out


class NotionClient:
    def __init__(self, token: str, notion_version: str) -> None:
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": notion_version,
            "Content-Type": "application/json",
        }

    def request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        retries: int = 4,
        sleep_s: float = 0.4,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
        url = f"{NOTION_API_BASE}{path}"
        last_err = ""
        last_status: Optional[int] = None
        for attempt in range(max(1, retries) + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                last_status = resp.status_code
            except requests.RequestException as exc:
                last_err = f"request_exception: {exc}"
                if attempt >= retries:
                    return None, last_err, last_status
                wait = (2 ** attempt) * sleep_s + random.uniform(0.0, 0.3)
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                snippet = (resp.text or "")[:800]
                last_err = f"http_{resp.status_code}: {snippet}"
                if resp.status_code in TRANSIENT_STATUS and attempt < retries:
                    wait = (2 ** attempt) * sleep_s + random.uniform(0.0, 0.3)
                    time.sleep(wait)
                    continue
                return None, last_err, resp.status_code

            try:
                return resp.json(), None, resp.status_code
            except ValueError:
                return None, "invalid_json_response", resp.status_code

        return None, last_err or "unknown_error", last_status


class OpenAIFallback:
    def __init__(self, model: str, logger: logging.Logger, max_retries: int = 3, timeout: int = 70) -> None:
        self.logger = logger
        self.model = model
        self.max_retries = max(1, int(max_retries))
        self.timeout = max(10, int(timeout))
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            api_key = read_secret_from_file("CHAVE_SECRETA_API_Mauricio_local.txt")
        self.enabled = bool(api_key and OpenAI is not None and openai is not None)
        self.client: Optional[OpenAI] = OpenAI(api_key=api_key, max_retries=0) if self.enabled else None

    def call_json_schema(self, system: str, user: str, schema_name: str, schema: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        if not self.enabled or self.client is None:
            return False, {}, "llm_disabled"

        fatal_error_classes = (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
        )
        recoverable_error_classes = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )

        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    reasoning_effort="minimal",
                    verbosity="low",
                    timeout=self.timeout,
                )
                choice = resp.choices[0]
                content_obj = choice.message.content
                text = ""
                if isinstance(content_obj, str):
                    text = content_obj.strip()
                elif isinstance(content_obj, list):
                    chunks: List[str] = []
                    for item in content_obj:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            chunks.append(str(item["text"]))
                    text = "".join(chunks).strip()
                if not text:
                    raise ValueError("empty_llm_json")
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError("llm_json_not_object")
                return True, payload, ""
            except fatal_error_classes as exc:
                return False, {}, str(exc)
            except recoverable_error_classes as exc:
                last_err = str(exc)
                if attempt < self.max_retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.3)
                    self.logger.debug("LLM retry %d/%d in %.2fs: %s", attempt, self.max_retries, wait, exc)
                    time.sleep(wait)
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                if attempt < self.max_retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.3)
                    self.logger.debug("LLM retry %d/%d in %.2fs: %s", attempt, self.max_retries, wait, exc)
                    time.sleep(wait)
        return False, {}, last_err or "llm_unknown_error"


class PdfResolver:
    def __init__(self, pdf_root: Path, logger: logging.Logger) -> None:
        self.pdf_root = pdf_root
        self.logger = logger
        self.info_to_pdfs: Dict[str, List[Path]] = defaultdict(list)
        self.rows_by_info: Dict[str, List[PdfCandidate]] = {}
        self.process_index: Optional[Dict[str, List[PdfCandidate]]] = None
        self._build_pdf_index()

    def _build_pdf_index(self) -> None:
        for path in self.pdf_root.rglob("*.pdf"):
            m = re.search(r"Bij[_-]?(\d+)$", path.stem, flags=re.IGNORECASE)
            if not m:
                continue
            info = m.group(1)
            self.info_to_pdfs[info].append(path)
        for info in list(self.info_to_pdfs.keys()):
            self.info_to_pdfs[info] = sorted(self.info_to_pdfs[info])
        self.logger.info("PDF index built: informativos=%d", len(self.info_to_pdfs))

    @staticmethod
    def _remove_relator_prefixes(value: str) -> str:
        text = normalize_ws(value)
        if not text:
            return ""
        conv = ""
        if re.search(r"(?i)\(convocad[oa]\)", text):
            conv = "convocada" if re.search(r"(?i)\(convocada\)", text) else "convocado"
        text = re.sub(r"(?i)\(convocad[oa]\)", "", text)
        text = RELATOR_PREFIX_RE.sub("", text)
        text = normalize_ws(text).strip(" ,.;")
        if not text:
            return ""
        if conv:
            return f"{text} ({conv})"
        return text

    def _parse_info_rows(self, info: str) -> List[PdfCandidate]:
        if info in self.rows_by_info:
            return self.rows_by_info[info]
        rows: List[PdfCandidate] = []
        for pdf_path in self.info_to_pdfs.get(info, []):
            extracted = extract_rows_from_pdf(pdf_path, self.logger)
            for row in extracted:
                proc = normalize_ws(row.get("numero_processo", ""))
                if not proc:
                    continue
                rows.append(
                    PdfCandidate(
                        source_pdf=str(pdf_path),
                        informativo=normalize_ws(row.get("informativo", "")),
                        numero_processo=proc,
                        classe=normalize_ws(row.get("classe", "")),
                        relator=self._remove_relator_prefixes(normalize_ws(row.get("relator(a)", ""))),
                        texto_do_boletim=normalize_ws(row.get("texto_do_boletim", "")),
                    )
                )
        self.rows_by_info[info] = rows
        return rows

    def _build_process_index(self) -> None:
        if self.process_index is not None:
            return
        idx: Dict[str, List[PdfCandidate]] = defaultdict(list)
        infos = sorted(self.info_to_pdfs.keys(), key=lambda x: int(x))
        for info in infos:
            for row in self._parse_info_rows(info):
                idx[row.numero_processo].append(row)
        self.process_index = idx
        self.logger.info("PDF process index built: processos=%d", len(idx))

    def find_candidates(self, numero_processo: str, informativo_hint: str = "") -> List[PdfCandidate]:
        proc = normalize_ws(numero_processo)
        if not proc:
            return []
        hint = normalize_ws(informativo_hint)
        if NUMERIC_INFORMATIVO_RE.fullmatch(hint):
            candidates = [c for c in self._parse_info_rows(hint) if c.numero_processo == proc]
            if candidates:
                return candidates
        self._build_process_index()
        assert self.process_index is not None
        return list(self.process_index.get(proc, []))

    def pick_best_candidate(self, page: PageRecord, candidates: List[PdfCandidate]) -> Tuple[Optional[PdfCandidate], bool]:
        if not candidates:
            return None, False
        if len(candidates) == 1:
            return candidates[0], False
        scored: List[Tuple[float, PdfCandidate]] = []
        page_ctx = f"{page.tema} {page.texto_do_boletim[:700]}"
        for cand in candidates:
            score = jaccard_similarity(page_ctx, cand.texto_do_boletim)
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score = scored[0][0]
        if len(scored) == 1:
            return scored[0][1], False
        second = scored[1][0]
        is_tie = abs(top_score - second) < 0.04
        return scored[0][1], is_tie


class CanonicalRules:
    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        ensure_default_config_files(config_dir)

        classe_cfg = load_text_or_json(config_dir / "classe_canonical.yaml")
        ramo_cfg = load_text_or_json(config_dir / "ramo_canonical.yaml")
        sub_cfg = load_text_or_json(config_dir / "subramo_overrides.yaml")
        rel_cfg = load_text_or_json(config_dir / "relator_overrides.yaml")

        self.classe_canonical: List[str] = [normalize_ws(x) for x in classe_cfg.get("canonical", []) if normalize_ws(x)]
        self.classe_aliases: Dict[str, str] = {
            slug_alnum(k): normalize_ws(v)
            for k, v in dict(classe_cfg.get("aliases", {})).items()
            if slug_alnum(str(k)) and normalize_ws(v)
        }

        self.ramo_canonical: List[str] = [normalize_ws(x) for x in ramo_cfg.get("canonical", []) if normalize_ws(x)]
        self.ramo_aliases: Dict[str, str] = {
            slug_alnum(k): normalize_ws(v)
            for k, v in dict(ramo_cfg.get("aliases", {})).items()
            if slug_alnum(str(k)) and normalize_ws(v)
        }
        self.ramo_drop: Set[str] = {slug_alnum(x) for x in ramo_cfg.get("drop_exact", []) if slug_alnum(str(x))}

        self.sub_aliases: Dict[str, str] = {
            slug_alnum(k): normalize_ws(v)
            for k, v in dict(sub_cfg.get("aliases", {})).items()
            if slug_alnum(str(k)) and normalize_ws(v)
        }
        self.sub_drop: Set[str] = {slug_alnum(x) for x in sub_cfg.get("drop_exact", []) if slug_alnum(str(x))}
        self.sub_defaults_by_ramo: Dict[str, List[str]] = {
            normalize_ws(k): [normalize_ws(x) for x in v if normalize_ws(x)]
            for k, v in dict(sub_cfg.get("forced_two_by_ramo", {})).items()
            if normalize_ws(k) and isinstance(v, list)
        }

        self.relator_aliases: Dict[str, str] = {
            fold_text(k): normalize_ws(v)
            for k, v in dict(rel_cfg.get("aliases", {})).items()
            if fold_text(str(k)) and normalize_ws(v)
        }

        self.classe_set: Set[str] = set(self.classe_canonical)
        self.ramo_set: Set[str] = set(self.ramo_canonical)

    @staticmethod
    def remove_relator_prefixes(value: str) -> str:
        text = normalize_ws(value)
        if not text:
            return ""
        conv = ""
        if re.search(r"(?i)\(convocad[oa]\)", text):
            conv = "convocada" if re.search(r"(?i)\(convocada\)", text) else "convocado"
        text = re.sub(r"(?i)\(convocad[oa]\)", "", text)
        text = RELATOR_PREFIX_RE.sub("", text)
        text = normalize_ws(text).strip(" ,.;")
        if not text:
            return ""
        if conv:
            return f"{text} ({conv})"
        return text

    def normalize_classe(self, value: str) -> str:
        text = normalize_ws(value)
        if not text:
            return ""
        if text in self.classe_set:
            return text
        key = slug_alnum(text)
        if key in self.classe_aliases:
            return self.classe_aliases[key]
        for cand in self.classe_canonical:
            if slug_alnum(cand) == key:
                return cand
        return ""

    def normalize_ramo(self, value: str) -> str:
        text = normalize_ws(value)
        if not text:
            return ""
        key = slug_alnum(text)
        if key in self.ramo_drop:
            return ""
        if text in self.ramo_set:
            return text
        if key in self.ramo_aliases:
            return self.ramo_aliases[key]
        for cand in self.ramo_canonical:
            if slug_alnum(cand) == key:
                return cand
        return ""

    def normalize_relator(self, value: str) -> str:
        text = normalize_ws(value)
        if not text:
            return ""
        folded = fold_text(text)
        if folded in self.relator_aliases:
            text = self.relator_aliases[folded]
        text = self.remove_relator_prefixes(text)
        folded = fold_text(text)
        if folded in self.relator_aliases:
            text = self.relator_aliases[folded]
        return normalize_ws(text)

    def normalize_subramo_item(self, value: str) -> str:
        text = normalize_ws(value)
        if not text:
            return ""
        key = slug_alnum(text)
        if key in self.sub_drop:
            return ""
        if key in self.sub_aliases:
            return self.sub_aliases[key]
        return text


class CanonicalAgent:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger

        token = read_secret_from_file(args.notion_key_file)
        if not token:
            raise RuntimeError(f"Cannot read Notion token from {args.notion_key_file}")

        self.notion = NotionClient(token=token, notion_version=args.notion_version)
        self.rules = CanonicalRules(Path(args.config_dir))
        self.pdf = PdfResolver(Path(args.pdf_root), logger)
        self.llm = OpenAIFallback(model=args.openai_model, logger=logger, max_retries=3, timeout=70)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path(args.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        prefix = normalize_ws(args.report_prefix)
        if not prefix:
            prefix = f"trf1_canonical_{self.timestamp}"
        self.detected_csv = self.report_dir / f"{prefix}_detected.csv"
        self.applied_csv = self.report_dir / f"{prefix}_applied.csv"
        self.unresolved_csv = self.report_dir / f"{prefix}_unresolved.csv"
        self.summary_json = self.report_dir / f"{prefix}_summary.json"

        cp_arg = normalize_ws(args.checkpoint_file)
        self.checkpoint_file = Path(cp_arg) if cp_arg else DEFAULT_CHECKPOINT_FILE
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        self.stats = RuntimeStats()
        self.processed_page_ids: Set[str] = set()

        self.schema: Dict[str, Any] = {}
        self.ramo_options: Set[str] = set()
        self.classe_options: Set[str] = set()
        self.relator_options: Set[str] = set()
        self.subramo_options: Set[str] = set()

        self.subramo_frequency: Counter[str] = Counter()
        self.subramo_by_ramo_frequency: Dict[str, Counter[str]] = defaultdict(Counter)

    def _load_checkpoint_if_needed(self) -> None:
        if not self.args.resume:
            return
        if not self.checkpoint_file.exists():
            return
        data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        if str(data.get("database_id", "")) != self.args.database_id:
            return
        self.processed_page_ids = set(str(x) for x in data.get("processed_page_ids", []))
        stats_raw = data.get("stats", {})
        if isinstance(stats_raw, dict):
            self.stats = RuntimeStats(
                pages_total=int(stats_raw.get("pages_total", 0) or 0),
                pages_processed=int(stats_raw.get("pages_processed", 0) or 0),
                pages_problematic=int(stats_raw.get("pages_problematic", 0) or 0),
                pages_updated=int(stats_raw.get("pages_updated", 0) or 0),
                pages_update_failed=int(stats_raw.get("pages_update_failed", 0) or 0),
                unresolved_pages=int(stats_raw.get("unresolved_pages", 0) or 0),
                llm_calls=int(stats_raw.get("llm_calls", 0) or 0),
                llm_failures=int(stats_raw.get("llm_failures", 0) or 0),
                schema_cleanup_attempted=bool(stats_raw.get("schema_cleanup_attempted", False)),
                schema_cleanup_ok=bool(stats_raw.get("schema_cleanup_ok", False)),
            )
        self.logger.info("Checkpoint loaded: processed=%d", len(self.processed_page_ids))

    def _save_checkpoint(self) -> None:
        payload = {
            "database_id": self.args.database_id,
            "updated_at": utc_now_iso(),
            "processed_page_ids": sorted(self.processed_page_ids),
            "stats": self.stats.to_dict(),
            "report_files": {
                "detected": str(self.detected_csv),
                "applied": str(self.applied_csv),
                "unresolved": str(self.unresolved_csv),
                "summary": str(self.summary_json),
            },
        }
        write_json_atomic(self.checkpoint_file, payload)

    def _fetch_schema(self) -> None:
        data, err, _ = self.notion.request_json("GET", f"/databases/{self.args.database_id}")
        if err or data is None:
            raise RuntimeError(f"Cannot load database schema: {err}")
        self.schema = data
        props = data.get("properties", {}) if isinstance(data, dict) else {}
        if not isinstance(props, dict):
            raise RuntimeError("Invalid Notion schema properties")

        required_types = {
            "classe": "select",
            "relator(a)": "select",
            "ramo_do_direito": "select",
            "informativo": "select",
            "subramo_do_direito": "multi_select",
            "numero_processo": "rich_text",
            "tema": "title",
            "texto_do_boletim": "rich_text",
        }

        for prop, expected_type in required_types.items():
            meta = props.get(prop)
            if not isinstance(meta, dict) or normalize_ws(meta.get("type", "")) != expected_type:
                got = "" if not isinstance(meta, dict) else str(meta.get("type", ""))
                raise RuntimeError(f"Schema mismatch for '{prop}': expected={expected_type}, got={got}")

        def extract_options(prop_name: str, p_type: str) -> Set[str]:
            meta = props.get(prop_name, {})
            if not isinstance(meta, dict):
                return set()
            subtype = meta.get(p_type, {})
            if not isinstance(subtype, dict):
                return set()
            out: Set[str] = set()
            for option in subtype.get("options", []):
                if isinstance(option, dict):
                    name = normalize_ws(option.get("name", ""))
                    if name:
                        out.add(name)
            return out

        self.classe_options = extract_options("classe", "select")
        self.relator_options = extract_options("relator(a)", "select")
        self.ramo_options = extract_options("ramo_do_direito", "select")
        self.subramo_options = extract_options("subramo_do_direito", "multi_select")

        self.logger.info(
            "Schema loaded: classe=%d relator=%d ramo=%d subramo=%d",
            len(self.classe_options),
            len(self.relator_options),
            len(self.ramo_options),
            len(self.subramo_options),
        )

    def _iter_pages(self) -> List[PageRecord]:
        pages: List[PageRecord] = []
        start_cursor: Optional[str] = None
        while True:
            payload: Dict[str, Any] = {"page_size": 100}
            if start_cursor:
                payload["start_cursor"] = start_cursor
            data, err, _ = self.notion.request_json("POST", f"/databases/{self.args.database_id}/query", payload=payload)
            if err or data is None:
                raise RuntimeError(f"Database query failed: {err}")

            results = data.get("results", [])
            for obj in results:
                if not isinstance(obj, dict):
                    continue
                props = obj.get("properties", {})
                if not isinstance(props, dict):
                    continue
                page = PageRecord(
                    page_id=normalize_ws(obj.get("id", "")),
                    page_url=normalize_ws(obj.get("url", "")),
                    numero_processo=notion_plain_text_from_property(props.get("numero_processo", {})),
                    tema=notion_plain_text_from_property(props.get("tema", {})),
                    texto_do_boletim=notion_plain_text_from_property(props.get("texto_do_boletim", {})),
                    classe=notion_plain_text_from_property(props.get("classe", {})),
                    relator=notion_plain_text_from_property(props.get("relator(a)", {})),
                    ramo=notion_plain_text_from_property(props.get("ramo_do_direito", {})),
                    informativo=notion_plain_text_from_property(props.get("informativo", {})),
                    subramos=split_notion_multiselect(notion_plain_text_from_property(props.get("subramo_do_direito", {}))),
                )
                pages.append(page)
                ramo_norm = self.rules.normalize_ramo(page.ramo)
                for sub in page.subramos:
                    s_norm = self.rules.normalize_subramo_item(sub)
                    if s_norm:
                        self.subramo_frequency[s_norm] += 1
                        if ramo_norm:
                            self.subramo_by_ramo_frequency[ramo_norm][s_norm] += 1

            if not data.get("has_more"):
                break
            start_cursor = normalize_ws(data.get("next_cursor", "")) or None

        return pages

    def _is_problematic(self, page: PageRecord) -> List[str]:
        problems: List[str] = []

        classe_norm = self.rules.normalize_classe(page.classe)
        if not classe_norm:
            problems.append("classe")

        relator_norm = self.rules.normalize_relator(page.relator)
        relator_has_prefix = bool(RELATOR_PREFIX_RE.search(page.relator or ""))
        if not relator_norm or relator_has_prefix:
            problems.append("relator(a)")

        ramo_norm = self.rules.normalize_ramo(page.ramo)
        if not ramo_norm:
            problems.append("ramo_do_direito")

        if not NUMERIC_INFORMATIVO_RE.fullmatch(normalize_ws(page.informativo)):
            problems.append("informativo")

        normalized_sub = [self.rules.normalize_subramo_item(x) for x in page.subramos]
        normalized_sub = [x for x in normalized_sub if x]
        normalized_sub = dedupe_keep_order(normalized_sub)
        if len(normalized_sub) != 2:
            problems.append("subramo_do_direito")
        else:
            for item in normalized_sub:
                if item not in self.subramo_options:
                    problems.append("subramo_do_direito")
                    break

        return dedupe_keep_order(problems)

    def _llm_pick_candidate(self, page: PageRecord, candidates: List[PdfCandidate]) -> Optional[PdfCandidate]:
        if not candidates:
            return None
        if not self.llm.enabled:
            return None
        schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "integer", "minimum": 1, "maximum": len(candidates)},
                "justification": {"type": "string"},
            },
            "required": ["choice", "justification"],
            "additionalProperties": False,
        }
        lines = []
        for idx, cand in enumerate(candidates, start=1):
            snippet = normalize_ws(cand.texto_do_boletim)[:260]
            lines.append(
                f"[{idx}] info={cand.informativo} classe={cand.classe} relator={cand.relator} texto={snippet}"
            )
        user = (
            "Choose the best PDF match for this Notion row.\n"
            f"numero_processo={page.numero_processo}\n"
            f"tema={page.tema}\n"
            f"texto={normalize_ws(page.texto_do_boletim)[:320]}\n"
            "Candidates:\n"
            + "\n".join(lines)
        )
        ok, payload, err = self.llm.call_json_schema(
            system="Return strict JSON only.",
            user=user,
            schema_name="pdf_candidate_choice",
            schema=schema,
        )
        self.stats.llm_calls += 1
        if not ok:
            self.stats.llm_failures += 1
            self.logger.debug("LLM candidate selection failed: %s", err)
            return None
        choice = int(payload.get("choice", 0) or 0)
        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]
        return None

    def _llm_pick_ramo(self, page: PageRecord, allowed: Sequence[str]) -> str:
        if not self.llm.enabled:
            return ""
        schema = {
            "type": "object",
            "properties": {
                "ramo": {"type": "string"},
            },
            "required": ["ramo"],
            "additionalProperties": False,
        }
        allowed_line = "; ".join(allowed)
        user = (
            "Classify ramo_do_direito into one of allowed values only.\n"
            f"Allowed: {allowed_line}\n"
            f"tema={page.tema}\n"
            f"texto={normalize_ws(page.texto_do_boletim)[:1400]}\n"
            f"classe={page.classe} relator={page.relator}"
        )
        ok, payload, err = self.llm.call_json_schema(
            system="Return strict JSON only.",
            user=user,
            schema_name="ramo_choice",
            schema=schema,
        )
        self.stats.llm_calls += 1
        if not ok:
            self.stats.llm_failures += 1
            self.logger.debug("LLM ramo failed: %s", err)
            return ""
        ramo = normalize_ws(payload.get("ramo", ""))
        ramo = self.rules.normalize_ramo(ramo)
        if ramo in self.rules.ramo_set:
            return ramo
        return ""

    def _map_subramo_to_allowed(self, raw_item: str) -> str:
        norm_item = self.rules.normalize_subramo_item(raw_item)
        if not norm_item:
            return ""
        if norm_item in self.subramo_options:
            return norm_item
        key_target = slug_alnum(norm_item)
        if not key_target:
            return ""
        # exact folded match first
        for cand in self.subramo_options:
            if slug_alnum(cand) == key_target:
                return cand
        # fuzzy fallback
        best = ""
        best_score = 0.0
        for cand in self.subramo_options:
            score = similarity_key(key_target, slug_alnum(cand))
            if score > best_score:
                best = cand
                best_score = score
        if best and best_score >= 0.84:
            return best
        return ""

    def _suggest_subramo_candidates(self, ramo: str) -> List[str]:
        pool: List[str] = []
        if ramo and ramo in self.subramo_by_ramo_frequency:
            pool.extend([x for x, _ in self.subramo_by_ramo_frequency[ramo].most_common(50)])
        pool.extend([x for x, _ in self.subramo_frequency.most_common(80)])
        # add config defaults for ramo
        pool.extend(self.rules.sub_defaults_by_ramo.get(ramo, []))
        # include any direct ramo-like entries if present
        if ramo and ramo in self.subramo_options:
            pool.append(ramo)
        out = dedupe_keep_order([x for x in pool if x in self.subramo_options])
        return out[:60]

    def _llm_pick_subramos(self, page: PageRecord, ramo: str, candidates: Sequence[str]) -> List[str]:
        if not self.llm.enabled or not candidates:
            return []
        schema = {
            "type": "object",
            "properties": {
                "subramos": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string"},
                }
            },
            "required": ["subramos"],
            "additionalProperties": False,
        }
        allowed_line = "; ".join(candidates)
        user = (
            "Select exactly 2 subramo_do_direito values from allowed list.\n"
            f"Allowed: {allowed_line}\n"
            f"ramo={ramo}\n"
            f"tema={page.tema}\n"
            f"texto={normalize_ws(page.texto_do_boletim)[:1500]}\n"
            f"subramo_atual={', '.join(page.subramos)}"
        )
        ok, payload, err = self.llm.call_json_schema(
            system="Return strict JSON only.",
            user=user,
            schema_name="subramo_choice",
            schema=schema,
        )
        self.stats.llm_calls += 1
        if not ok:
            self.stats.llm_failures += 1
            self.logger.debug("LLM subramo failed: %s", err)
            return []
        raw = payload.get("subramos", [])
        if not isinstance(raw, list):
            return []
        mapped: List[str] = []
        for item in raw:
            val = self._map_subramo_to_allowed(str(item))
            if val:
                mapped.append(val)
        mapped = dedupe_keep_order(mapped)
        if len(mapped) >= 2:
            return mapped[:2]
        return []

    def _resolve_row(self, page: PageRecord) -> ResolutionResult:
        problems = self._is_problematic(page)
        resolved: Dict[str, Any] = {}
        unresolved: List[str] = []
        notes: List[str] = []
        llm_used = False

        if not problems:
            return ResolutionResult(page, [], {}, [], [], False)

        candidates = self.pdf.find_candidates(page.numero_processo, page.informativo)
        picked, tie = self.pdf.pick_best_candidate(page, candidates)
        if tie and len(candidates) > 1:
            chosen = self._llm_pick_candidate(page, candidates)
            if chosen is not None:
                llm_used = True
                picked = chosen
                notes.append("pdf_candidate_selected_by_llm")
            else:
                notes.append("pdf_candidate_tie_unresolved")

        # classe
        if "classe" in problems:
            classe_new = ""
            if picked is not None:
                classe_new = self.rules.normalize_classe(picked.classe)
            if not classe_new:
                classe_new = self.rules.normalize_classe(page.classe)
            if classe_new:
                resolved["classe"] = classe_new
            else:
                unresolved.append("classe")

        # relator(a)
        if "relator(a)" in problems:
            relator_new = ""
            if picked is not None:
                relator_new = self.rules.normalize_relator(picked.relator)
            if not relator_new:
                relator_new = self.rules.normalize_relator(page.relator)
            if relator_new:
                resolved["relator(a)"] = relator_new
            else:
                unresolved.append("relator(a)")

        # informativo
        if "informativo" in problems:
            info_new = ""
            if picked is not None and NUMERIC_INFORMATIVO_RE.fullmatch(picked.informativo):
                info_new = picked.informativo
            else:
                if NUMERIC_INFORMATIVO_RE.fullmatch(page.informativo):
                    info_new = page.informativo
            if info_new:
                resolved["informativo"] = info_new
            else:
                unresolved.append("informativo")

        # ramo_do_direito
        ramo_new = self.rules.normalize_ramo(page.ramo)
        if "ramo_do_direito" in problems:
            if not ramo_new:
                ramo_new = ""
                ramo_candidates = list(self.rules.ramo_canonical)
                llm_guess = self._llm_pick_ramo(page, ramo_candidates)
                if llm_guess:
                    llm_used = True
                    ramo_new = llm_guess
            if ramo_new:
                resolved["ramo_do_direito"] = ramo_new
            else:
                unresolved.append("ramo_do_direito")

        # subramo_do_direito
        if "subramo_do_direito" in problems:
            sub_norm = [self._map_subramo_to_allowed(x) for x in page.subramos]
            sub_norm = dedupe_keep_order([x for x in sub_norm if x])

            if len(sub_norm) > 2:
                # keep top by frequency for target ramo first
                target_ramo = ramo_new or self.rules.normalize_ramo(page.ramo)
                local_freq = self.subramo_by_ramo_frequency.get(target_ramo, Counter())
                sub_norm = sorted(
                    sub_norm,
                    key=lambda x: (local_freq.get(x, 0), self.subramo_frequency.get(x, 0)),
                    reverse=True,
                )[:2]

            if len(sub_norm) == 1:
                target_ramo = ramo_new or self.rules.normalize_ramo(page.ramo)
                defaults = self.rules.sub_defaults_by_ramo.get(target_ramo, [])
                for cand in defaults:
                    mapped = self._map_subramo_to_allowed(cand)
                    if mapped and mapped not in sub_norm:
                        sub_norm.append(mapped)
                    if len(sub_norm) >= 2:
                        break

            if len(sub_norm) < 2:
                target_ramo = ramo_new or self.rules.normalize_ramo(page.ramo)
                cand_pool = self._suggest_subramo_candidates(target_ramo)
                llm_sub = self._llm_pick_subramos(page, target_ramo, cand_pool)
                if llm_sub:
                    llm_used = True
                    sub_norm = llm_sub

            sub_norm = dedupe_keep_order(sub_norm)
            if len(sub_norm) >= 2:
                resolved["subramo_do_direito"] = sub_norm[:2]
            else:
                unresolved.append("subramo_do_direito")

        return ResolutionResult(
            page=page,
            problematic_columns=problems,
            resolved=resolved,
            unresolved_columns=dedupe_keep_order(unresolved),
            notes=notes,
            llm_used=llm_used,
        )

    def _build_notion_patch(self, resolution: ResolutionResult) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        resolved = resolution.resolved

        if "classe" in resolved and normalize_ws(resolved["classe"]) != normalize_ws(resolution.page.classe):
            properties["classe"] = {"select": {"name": str(resolved["classe"])}}

        if "relator(a)" in resolved and normalize_ws(resolved["relator(a)"]) != normalize_ws(resolution.page.relator):
            properties["relator(a)"] = {"select": {"name": str(resolved["relator(a)"])}}

        if "ramo_do_direito" in resolved and normalize_ws(resolved["ramo_do_direito"]) != normalize_ws(resolution.page.ramo):
            properties["ramo_do_direito"] = {"select": {"name": str(resolved["ramo_do_direito"])}}

        if "informativo" in resolved and normalize_ws(resolved["informativo"]) != normalize_ws(resolution.page.informativo):
            properties["informativo"] = {"select": {"name": str(resolved["informativo"])}}

        if "subramo_do_direito" in resolved:
            new_items = [normalize_ws(x) for x in resolved["subramo_do_direito"] if normalize_ws(x)]
            old_items = dedupe_keep_order([normalize_ws(x) for x in resolution.page.subramos if normalize_ws(x)])
            if new_items != old_items:
                properties["subramo_do_direito"] = {
                    "multi_select": [{"name": x} for x in new_items]
                }

        return {"properties": properties}

    def _attempt_schema_cleanup(self, used_values: Dict[str, Set[str]]) -> str:
        self.stats.schema_cleanup_attempted = True

        def make_opts(names: Iterable[str]) -> List[Dict[str, str]]:
            return [{"name": x} for x in sorted({normalize_ws(n) for n in names if normalize_ws(n)})]

        payload = {
            "properties": {
                "classe": {"select": {"options": make_opts(used_values.get("classe", set()))}},
                "relator(a)": {"select": {"options": make_opts(used_values.get("relator(a)", set()))}},
                "ramo_do_direito": {"select": {"options": make_opts(used_values.get("ramo_do_direito", set()))}},
                "informativo": {"select": {"options": make_opts(used_values.get("informativo", set()))}},
                "subramo_do_direito": {
                    "multi_select": {"options": make_opts(used_values.get("subramo_do_direito", set()))}
                },
            }
        }

        _, err, _ = self.notion.request_json("PATCH", f"/databases/{self.args.database_id}", payload=payload, retries=2)
        if err:
            self.stats.schema_cleanup_ok = False
            return f"schema_cleanup_failed: {err}. fallback=ignored"

        self.stats.schema_cleanup_ok = True
        return "schema_cleanup_ok"

    def run(self) -> int:
        self._fetch_schema()
        self._load_checkpoint_if_needed()

        pages = self._iter_pages()
        self.stats.pages_total = len(pages)
        self.logger.info("Loaded pages: %d", len(pages))

        detected_fields = [
            "page_id",
            "page_url",
            "numero_processo",
            "problematic_columns",
            "classe_before",
            "classe_after",
            "relator_before",
            "relator_after",
            "ramo_before",
            "ramo_after",
            "informativo_before",
            "informativo_after",
            "subramo_before",
            "subramo_after",
            "notes",
            "llm_used",
        ]
        applied_fields = [
            "page_id",
            "page_url",
            "numero_processo",
            "updated_properties",
            "status",
            "error",
        ]
        unresolved_fields = [
            "page_id",
            "page_url",
            "numero_processo",
            "unresolved_columns",
            "problematic_columns",
            "notes",
        ]

        ensure_csv_header(self.detected_csv, detected_fields)
        ensure_csv_header(self.applied_csv, applied_fields)
        ensure_csv_header(self.unresolved_csv, unresolved_fields)

        detected_buffer: List[Dict[str, Any]] = []
        applied_buffer: List[Dict[str, Any]] = []
        unresolved_buffer: List[Dict[str, Any]] = []

        used_values: Dict[str, Set[str]] = {
            "classe": set(self.classe_options),
            "relator(a)": set(self.relator_options),
            "ramo_do_direito": set(self.ramo_options),
            "informativo": {
                x for x in notion_option_set_from_schema(self.schema, "informativo", "select") if normalize_ws(x)
            },
            "subramo_do_direito": set(self.subramo_options),
        }

        batch_save = 50
        processed_this_run = 0

        for page in pages:
            if self.args.max_pages > 0 and processed_this_run >= self.args.max_pages:
                break
            if page.page_id in self.processed_page_ids:
                continue

            resolution = self._resolve_row(page)
            self.stats.pages_processed += 1
            processed_this_run += 1

            if resolution.problematic_columns:
                self.stats.pages_problematic += 1
                detected_buffer.append(
                    {
                        "page_id": page.page_id,
                        "page_url": page.page_url,
                        "numero_processo": page.numero_processo,
                        "problematic_columns": ", ".join(resolution.problematic_columns),
                        "classe_before": page.classe,
                        "classe_after": normalize_ws(resolution.resolved.get("classe", "")),
                        "relator_before": page.relator,
                        "relator_after": normalize_ws(resolution.resolved.get("relator(a)", "")),
                        "ramo_before": page.ramo,
                        "ramo_after": normalize_ws(resolution.resolved.get("ramo_do_direito", "")),
                        "informativo_before": page.informativo,
                        "informativo_after": normalize_ws(resolution.resolved.get("informativo", "")),
                        "subramo_before": ", ".join(page.subramos),
                        "subramo_after": ", ".join(resolution.resolved.get("subramo_do_direito", [])),
                        "notes": "; ".join(resolution.notes),
                        "llm_used": "1" if resolution.llm_used else "0",
                    }
                )

            patch_payload = self._build_notion_patch(resolution)
            should_update = bool(patch_payload.get("properties"))

            if resolution.unresolved_columns:
                self.stats.unresolved_pages += 1
                unresolved_buffer.append(
                    {
                        "page_id": page.page_id,
                        "page_url": page.page_url,
                        "numero_processo": page.numero_processo,
                        "unresolved_columns": ", ".join(resolution.unresolved_columns),
                        "problematic_columns": ", ".join(resolution.problematic_columns),
                        "notes": "; ".join(resolution.notes),
                    }
                )

            if should_update:
                for key, value in resolution.resolved.items():
                    if key == "subramo_do_direito":
                        for item in value:
                            used_values["subramo_do_direito"].add(item)
                    else:
                        used_values[key].add(str(value))

                if self.args.apply:
                    _, err, _ = self.notion.request_json(
                        "PATCH",
                        f"/pages/{page.page_id}",
                        payload=patch_payload,
                    )
                    if err:
                        self.stats.pages_update_failed += 1
                        applied_buffer.append(
                            {
                                "page_id": page.page_id,
                                "page_url": page.page_url,
                                "numero_processo": page.numero_processo,
                                "updated_properties": ", ".join(sorted(patch_payload.get("properties", {}).keys())),
                                "status": "update_failed",
                                "error": err,
                            }
                        )
                    else:
                        self.stats.pages_updated += 1
                        applied_buffer.append(
                            {
                                "page_id": page.page_id,
                                "page_url": page.page_url,
                                "numero_processo": page.numero_processo,
                                "updated_properties": ", ".join(sorted(patch_payload.get("properties", {}).keys())),
                                "status": "updated",
                                "error": "",
                            }
                        )
                else:
                    applied_buffer.append(
                        {
                            "page_id": page.page_id,
                            "page_url": page.page_url,
                            "numero_processo": page.numero_processo,
                            "updated_properties": ", ".join(sorted(patch_payload.get("properties", {}).keys())),
                            "status": "would_update",
                            "error": "",
                        }
                    )

            self.processed_page_ids.add(page.page_id)

            if self.stats.pages_processed % batch_save == 0:
                append_csv_rows(self.detected_csv, detected_fields, detected_buffer)
                append_csv_rows(self.applied_csv, applied_fields, applied_buffer)
                append_csv_rows(self.unresolved_csv, unresolved_fields, unresolved_buffer)
                detected_buffer.clear()
                applied_buffer.clear()
                unresolved_buffer.clear()
                self._save_checkpoint()
                self.logger.info(
                    "Progress processed=%d problematic=%d updated=%d unresolved=%d",
                    self.stats.pages_processed,
                    self.stats.pages_problematic,
                    self.stats.pages_updated,
                    self.stats.unresolved_pages,
                )

        append_csv_rows(self.detected_csv, detected_fields, detected_buffer)
        append_csv_rows(self.applied_csv, applied_fields, applied_buffer)
        append_csv_rows(self.unresolved_csv, unresolved_fields, unresolved_buffer)

        cleanup_note = "schema_cleanup_skipped"
        if self.args.apply:
            cleanup_note = self._attempt_schema_cleanup(used_values)

        summary = {
            "timestamp": utc_now_iso(),
            "database_id": self.args.database_id,
            "apply": bool(self.args.apply),
            "coverage": self.args.coverage,
            "stats": self.stats.to_dict(),
            "files": {
                "detected": str(self.detected_csv),
                "applied": str(self.applied_csv),
                "unresolved": str(self.unresolved_csv),
                "checkpoint": str(self.checkpoint_file),
            },
            "schema_cleanup": cleanup_note,
        }

        write_json_atomic(self.summary_json, summary)
        self._save_checkpoint()

        self.logger.info("Completed. summary=%s", self.summary_json)
        return 0


def similarity_key(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # ratio approximation without importing difflib heavy loops repeatedly
    shared = len(set(a) & set(b))
    return (2.0 * shared) / float(len(set(a)) + len(set(b)) or 1)


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        val = normalize_ws(item)
        key = fold_text(val)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def subramo_is_valid_two(items: Sequence[str], rules: CanonicalRules, allowed_options: Set[str]) -> bool:
    normalized = [rules.normalize_subramo_item(x) for x in items]
    normalized = dedupe_keep_order([x for x in normalized if x])
    if len(normalized) != 2:
        return False
    return all(x in allowed_options for x in normalized)


def ensure_default_config_files(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)

    classe_file = config_dir / "classe_canonical.yaml"
    if not classe_file.exists():
        classe_default = {
            "canonical": [
                "AC", "ACR", "AGRE", "AGSS", "AGTAR", "AI", "APN", "APOrd", "AR", "AgExPe", "AgInCiv",
                "AgIntCiv", "AgIntCrim", "AgIntREspAp", "Ap", "ApReeNec", "ArgIncCiv", "CC", "Cjur", "CumSen",
                "EDAp", "EDCiv", "EDCrim", "EI", "EINul", "EIfNu", "EInfNu", "EInfNul", "ETciv", "ExSusp",
                "ExcSusp", "HC", "IAC", "IP", "IRDR", "IncImp", "IncSus", "IncSusp", "MS", "PET", "PedConESus",
                "PetCiv", "REO", "REsp", "RSE", "Rcl", "ReclCrim", "ReeNec", "RevCrim", "RvC", "SLAT", "SLS",
                "TutCautAnt",
                "AG", "AgExp", "AgExPenal", "AgInt", "AgIntAI", "AgIntApReeNec", "AgIntCr", "AgIntMS",
                "AgIntREsp", "AgRRvCr", "AgRegHC", "ApRemNec", "ED", "EDAI", "EDAR", "EDHC", "EDReeNec",
                "ETCrim", "IncImpCiv", "MSCrim", "PBACrim", "PICMP", "PetCrim", "ReSE", "ReeNecCrim",
                "RvCr", "SuspApel",
            ],
            "aliases": {
                "ReeNeec": "ReeNec",
                "AgExpe": "AgExPe",
                "CJUR": "Cjur",
                "CJur": "Cjur",
                "Pet": "PET",
                "Reo": "REO",
                "AP": "Ap",
                "MsCrim": "MSCrim",
            },
        }
        classe_file.write_text(json.dumps(classe_default, ensure_ascii=False, indent=2), encoding="utf-8")

    ramo_file = config_dir / "ramo_canonical.yaml"
    if not ramo_file.exists():
        ramo_default = {
            "canonical": [
                "Administração Pública",
                "Direito Administrativo",
                "Direito Aduaneiro",
                "Direito Agrário",
                "Direito Ambiental",
                "Direito Civil",
                "Direito Constitucional",
                "Direito Educacional",
                "Direito Empresarial",
                "Direito Internacional",
                "Direito Internacional Público",
                "Direito Militar",
                "Direito Penal",
                "Direito Previdenciário",
                "Direito Processual",
                "Direito Processual Civil",
                "Direito Processual Penal",
                "Direito Público",
                "Direito Tributário",
                "Direito do Consumidor",
                "Direito do Trabalho",
                "Direitos Humanos",
                "Direitos Humanos e Assistência Social",
            ],
            "aliases": {
                "Administrativo": "Direito Administrativo",
                "Civil": "Direito Civil",
                "Penal": "Direito Penal",
                "Processual": "Direito Processual",
                "Processual Civil": "Direito Processual Civil",
                "Processual Penal": "Direito Processual Penal",
                "Direito tributario": "Direito Tributário",
                "Direito tribute1rio": "Direito Tributário",
                "Direito tribut\u001ario": "Direito Tributário",
                "Direitos fundamentais": "Direitos Humanos",
                "Direito Administrativista": "Direito Administrativo",
            },
            "drop_exact": [
                "TRF1",
                "Processo",
                "iniciando-se o prazo prescricional em 30/06/2017",
                "viola a boa-fé objetiva e a proibição do comportamento contraditório (venire contra factum proprium). A decisão judicial que supre a lacuna normativa para solucionar o caso concreto não viola o princípio da separação dos poderes",
            ],
        }
        ramo_file.write_text(json.dumps(ramo_default, ensure_ascii=False, indent=2), encoding="utf-8")

    sub_file = config_dir / "subramo_overrides.yaml"
    if not sub_file.exists():
        sub_default = {
            "aliases": {
                "Direito Previdenciario": "Direito Previdenciário",
                "Processo Tributário": "Processual Tributário",
            },
            "drop_exact": ["TRF1"],
            "forced_two_by_ramo": {
                "Direito Administrativo": ["Direito Administrativo", "Processo Administrativo"],
                "Direito Penal": ["Processo Penal", "Direito Penal Comum"],
                "Direito Previdenciário": ["Benefícios Previdenciários", "Processo Previdenciário"],
                "Direito Tributário": ["Processual Tributário", "Tributação Federal"],
                "Direito Civil": ["Responsabilidade Civil", "Processo Civil"],
                "Direito Público": ["Direito Administrativo", "Processo Civil"],
            },
        }
        sub_file.write_text(json.dumps(sub_default, ensure_ascii=False, indent=2), encoding="utf-8")

    rel_file = config_dir / "relator_overrides.yaml"
    if not rel_file.exists():
        rel_default = {
            "aliases": {
                "des. fed. cândido ribeiro": "Cândido Ribeiro",
                "des. fed. ney bello": "Ney Bello",
                "des. fed. olindo menezes": "Olindo Menezes",
                "des. fed. néviton guedes": "Néviton Guedes",
                "des. des.federal wilson alves de souza": "Wilson Alves de Souza",
            }
        }
        rel_file.write_text(json.dumps(rel_default, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical TRF1 Notion agent.")
    parser.add_argument("--database-id", default=DEFAULT_DATABASE_ID)
    parser.add_argument("--notion-key-file", default="Chave_Notion.txt")
    parser.add_argument("--pdf-root", default=".")
    parser.add_argument("--coverage", default="problematic", choices=["problematic"])
    parser.add_argument("--openai-model", default="gpt-5-mini")
    parser.add_argument("--notion-version", default=DEFAULT_NOTION_VERSION)
    parser.add_argument("--checkpoint-file", default=str(DEFAULT_CHECKPOINT_FILE))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--report-prefix", default="")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--apply", dest="apply", action="store_true")
    parser.add_argument("--no-apply", dest="apply", action="store_false")
    parser.set_defaults(apply=True)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("trf1_canonical_agent")

    agent = CanonicalAgent(args=args, logger=logger)
    return agent.run()


if __name__ == "__main__":
    raise SystemExit(main())
