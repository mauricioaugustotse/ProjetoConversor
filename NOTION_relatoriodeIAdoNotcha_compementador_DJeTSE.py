#!/usr/bin/env python3
"""
Agente IA para atualizar relatorios no Notion com GUI simples.

Fluxo:
1. Sempre abre uma GUI para confirmar a URL da pagina alvo.
2. Le blocos da pagina no Notion e identifica blocos textuais.
3. Monta contexto usando links Notion do proprio relatorio e fallback via data source.
4. Reescreve blocos textuais com OpenAI (gpt-5.1), incluindo destaque de partidos.
5. Atualiza os blocos no Notion preservando blocos nao textuais.
6. Gera relatorio local em .<nome_do_script>.report.json.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from openai_log_utils import configure_standard_logging, install_print_logger_bridge
from openai_progress_utils import read_json_dict, utc_now_iso, write_json_atomic


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_FILE = Path(__file__).name
SCRIPT_STEM = Path(__file__).stem
REPORT_FILE = SCRIPT_DIR / f".{SCRIPT_STEM}.report.json"
DEFAULT_SOURCE_DATABASE_URL = (
    "https://www.notion.so/317721955c6480d3b642cc296d6074c7"
)

NOTION_BASE_URL = "https://api.notion.com"
OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_NOTION_VERSION = "2025-09-03"

SUPPORTED_TEXT_BLOCK_TYPES = {
    "paragraph",
    "heading_1",
    "heading_2",
    "heading_3",
    "bulleted_list_item",
    "numbered_list_item",
    "quote",
    "callout",
    "toggle",
    "to_do",
}

CNJ_RE = re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b")
PROC_SHORT_RE = re.compile(r"\b\d{6,7}-\d{2}\b")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
URL_RE = re.compile(r"https?://[^\s)>\"]+")
CITY_UF_RE = re.compile(r"\b([A-Z][A-Za-zÀ-ÿ' .-]{2,})\s*/\s*([A-Z]{2})\b")
PARTY_FULL_NAME_RE = re.compile(
    r"\b((?:PARTIDO\s+)?[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ' .-]{2,}?)\s*\(([A-Z]{2,20})\)\s*(?:-\s*(NACIONAL|ESTADUAL|MUNICIPAL))?",
    flags=re.IGNORECASE,
)

PARTY_ABBREVIATIONS = {
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
}
PARTY_REGEX_ALTERNATIVES = [
    re.escape(party).replace(r"\ ", r"\s+")
    for party in sorted(PARTY_ABBREVIATIONS, key=len, reverse=True)
]
PARTY_TOKEN_RE = re.compile(
    r"\b("
    + "|".join(PARTY_REGEX_ALTERNATIVES)
    + r")(?:/[A-Z]{2})?\b",
    flags=re.IGNORECASE,
)
PARTY_UF_RE = re.compile(
    r"\b("
    + "|".join(PARTY_REGEX_ALTERNATIVES)
    + r"/[A-Z]{2})\b",
    flags=re.IGNORECASE,
)
GENERIC_PARTY_SCOPED_RE = re.compile(r"\b([A-Z]{2,20}/(?:[A-Z]{2}|Nacional))\b")

ADV_BLOCK_RE = re.compile(r"(?is)\badvogad(?:o|a|os|as)\s*:\s*(.*?)(?=(?:\n\n|\r\n\r\n|$))")
ADV_OAB_RE = re.compile(
    r"([A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]+(?:\s+[A-ZÀ-ÖØ-Ýa-zà-öø-ÿ'`´^~.-]+){1,}?)\s*(?:[-–—]\s*)?OAB\b"
)

MAX_CONTEXT_CASES = 36
MAX_FALLBACK_CONTEXTS = 28
MAX_BLOCK_BATCH = 12
MAX_NOTION_RICH_TEXT_CHARS = 2000
OPENAI_BATCH_CHAR_BUDGET = 5600
DEFAULT_HEARTBEAT_INTERVAL_S = 20.0
VERBOSE_HEARTBEAT_INTERVAL_S = 10.0
LINK_CONTEXT_PROGRESS_EVERY = 5
FALLBACK_PROGRESS_EVERY = 25
DEFAULT_OPENAI_BATCH_SIZE = 10
DEFAULT_OPENAI_TARGET_RPM = 180
DEFAULT_OPENAI_MAX_WORKERS = 2
DEFAULT_NOTION_MIN_INTERVAL_S = 0.25
MAX_NOTION_APPEND_CHILDREN = 100
AUTO_TABLE_MARKER_PREFIX = "[[AUTO_CASES::"
AUTO_LAWYERS_MARKER = "[[AUTO_ADVOGADOS]]"
AUTO_MARKER_PREFIXES = (AUTO_TABLE_MARKER_PREFIX, AUTO_LAWYERS_MARKER)
LAWYER_FREQ_MIN_CASES = 2
LAWYER_FREQ_MAX_ITEMS = 25


LOGGER = logging.getLogger(SCRIPT_STEM)

NOTION_CFG: Optional["NotionConfig"] = None
OPENAI_CFG: Optional["OpenAIConfig"] = None
NOTION_SESSION: Optional[requests.Session] = None
OPENAI_SESSION: Optional[requests.Session] = None
NOTION_PACER: Optional["RequestPacer"] = None
OPENAI_PACER: Optional["RequestPacer"] = None
BLOCK_INDEX: Dict[str, Dict[str, Any]] = {}
OPENAI_DISABLE_REASONING_PARAMS = True
OPENAI_SCHEMA_TEMP_DISABLED = False
OPENAI_TIMEOUT_STREAK = 0
OPENAI_RUNTIME_STATE_LOCK = threading.Lock()


@dataclass
class NotionConfig:
    token: str
    notion_version: str = DEFAULT_NOTION_VERSION
    timeout_s: int = 30
    max_retries: int = 4
    min_interval_s: float = DEFAULT_NOTION_MIN_INTERVAL_S


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-5.1"
    timeout_s: int = 75
    retries: int = 3
    target_rpm: int = DEFAULT_OPENAI_TARGET_RPM


@dataclass
class RunConfig:
    page_url: str
    source_database_url: str
    verbose: bool = False
    quiet: bool = False
    debug: bool = False
    log_file: str = ""
    openai_batch_size: int = DEFAULT_OPENAI_BATCH_SIZE
    openai_max_workers: int = DEFAULT_OPENAI_MAX_WORKERS
    openai_target_rpm: int = DEFAULT_OPENAI_TARGET_RPM
    openai_timeout_s: int = 75
    openai_retries: int = 3
    notion_min_interval_s: float = DEFAULT_NOTION_MIN_INTERVAL_S
    notion_timeout_s: int = 30
    notion_retries: int = 4


@dataclass
class CaseContext:
    processo_cnj: str = ""
    numero_unico: str = ""
    data_decisao: str = ""
    sigla_classe: str = ""
    nome_municipio: str = ""
    sigla_uf: str = ""
    relator: str = ""
    tema: str = ""
    eleicao_ano: str = ""
    cidade_uf: str = ""
    partes: str = ""
    punchline: str = ""
    texto_decisao: str = ""
    partidos: str = ""
    alegacoes: str = ""
    advogados: str = ""
    fundamentos: str = ""
    resultado: str = ""
    fonte_url: str = ""


@dataclass
class BlockPatchPlan:
    block_id: str
    block_type: str
    original_text: str
    improved_text: str
    rich_text_payload: List[Dict[str, Any]]


@dataclass
class TableCompanionTarget:
    table_id: str
    parent_id: str
    source_linked_page_ids: List[str]
    row_text_by_page_id: Dict[str, str]
    marker: str


@dataclass
class TableCompanionPlan:
    table_id: str
    parent_id: str
    marker: str
    title: str
    process_texts: List[str]


class RequestPacer:
    def __init__(self, min_interval_s: float = 0.0, target_rpm: int = 0) -> None:
        if target_rpm > 0:
            self._min_interval = 60.0 / float(target_rpm)
        else:
            self._min_interval = max(0.0, float(min_interval_s))
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


def _normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_multiline_ws(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    compact = [line for line in lines if line]
    return "\n".join(compact).strip()


def _normalize_lower(value: Any) -> str:
    return _normalize_ws(value).lower()


def _safe_join(values: Iterable[str], *, sep: str = ", ") -> str:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_ws(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return sep.join(out)


def _read_secret_from_file(path: Path) -> str:
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


def resolve_notion_key() -> str:
    env_keys = [
        "NOTION_API_KEY",
        "NOTION_TOKEN",
        "NOTION_SECRET",
    ]
    for key_name in env_keys:
        value = os.getenv(key_name, "").strip()
        if value:
            return value

    candidates = [
        SCRIPT_DIR / "Chave_Notion.txt",
        Path.cwd() / "Chave_Notion.txt",
    ]
    for candidate in candidates:
        value = _read_secret_from_file(candidate)
        if value:
            return value
    return ""


def resolve_openai_key() -> str:
    env_value = os.getenv("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value

    candidates = [
        SCRIPT_DIR / "CHAVE_SECRETA_API_Mauricio_local.txt",
        SCRIPT_DIR / "Chave Secreta API_Mauricio_local.txt",
        Path.cwd() / "CHAVE_SECRETA_API_Mauricio_local.txt",
        Path.cwd() / "Chave Secreta API_Mauricio_local.txt",
    ]
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

    raise ValueError(f"Nao foi possivel extrair ID Notion da URL: {url}")


def _extract_http_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            msg = payload.get("message") or payload.get("error") or ""
            return _normalize_ws(msg)
    except Exception:
        pass
    return _normalize_ws(response.text)


def _retry_delay_s(attempt: int, retry_after: str | None = None) -> float:
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


def notion_request(
    method: str,
    path: str,
    json_body: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if NOTION_CFG is None or NOTION_SESSION is None or NOTION_PACER is None:
        raise RuntimeError("Cliente Notion nao inicializado.")

    url = path if path.startswith("http") else f"{NOTION_BASE_URL}{path}"
    timeout_s = max(5, int(NOTION_CFG.timeout_s))
    max_attempts = max(1, int(NOTION_CFG.max_retries) + 1)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        NOTION_PACER.wait_turn()
        request_label = f"Notion {method.upper()} {path}"
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "%s | tentativa %d/%d | timeout=%ss",
                request_label,
                attempt,
                max_attempts,
                timeout_s,
            )
        stop_event, hb_thread, started_at = _start_wait_heartbeat(request_label)
        try:
            response = NOTION_SESSION.request(
                method=method.upper(),
                url=url,
                json=json_body,
                params=params,
                timeout=timeout_s,
            )
            elapsed = time.monotonic() - started_at
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "%s | tentativa %d/%d concluida em %.2fs | status=%s",
                    request_label,
                    attempt,
                    max_attempts,
                    elapsed,
                    response.status_code,
                )
        except requests.RequestException as exc:
            last_error = str(exc)
            elapsed = time.monotonic() - started_at
            LOGGER.warning(
                "%s falhou na tentativa %d/%d apos %.2fs: %s",
                request_label,
                attempt,
                max_attempts,
                elapsed,
                exc,
            )
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
            LOGGER.warning(
                "Notion %s %s retornou %s. Retry em %.2fs. msg=%s",
                method.upper(),
                path,
                response.status_code,
                delay,
                message,
            )
            time.sleep(delay)
            continue

        raise RuntimeError(
            f"Erro Notion {response.status_code} em {method.upper()} {path}: {message or '<sem mensagem>'}"
        )

    raise RuntimeError(f"Falha no Notion: {last_error or 'erro desconhecido'}")


def _openai_chat_request(
    messages: Sequence[Dict[str, Any]],
    schema: Optional[dict[str, Any]] = None,
    *,
    request_label: str = "OpenAI chat.completions",
    max_completion_tokens: int = 0,
    timeout_s_override: int = 0,
    max_attempts_override: int = 0,
) -> dict[str, Any]:
    if OPENAI_CFG is None or OPENAI_SESSION is None or OPENAI_PACER is None:
        raise RuntimeError("Cliente OpenAI nao inicializado.")

    global OPENAI_DISABLE_REASONING_PARAMS, OPENAI_SCHEMA_TEMP_DISABLED, OPENAI_TIMEOUT_STREAK

    payload: Dict[str, Any] = {
        "model": OPENAI_CFG.model,
        "messages": list(messages),
    }
    # Reduz consumo de tokens internos de raciocinio para priorizar saida JSON.
    with OPENAI_RUNTIME_STATE_LOCK:
        disable_reasoning_params = bool(OPENAI_DISABLE_REASONING_PARAMS)
    if not disable_reasoning_params:
        payload["reasoning_effort"] = "minimal"
        payload["verbosity"] = "low"
    if max_completion_tokens > 0:
        payload["max_completion_tokens"] = int(max_completion_tokens)
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "notion_relatorio_blocks",
                "strict": True,
                "schema": schema,
            },
        }

    timeout_s = (
        max(5, int(timeout_s_override))
        if int(timeout_s_override or 0) > 0
        else max(5, int(OPENAI_CFG.timeout_s))
    )
    max_attempts = max(1, int(OPENAI_CFG.retries) + 1)
    if int(max_attempts_override or 0) > 0:
        max_attempts = min(max_attempts, max(1, int(max_attempts_override)))
    compat_dropped_reasoning = False
    for attempt in range(1, max_attempts + 1):
        OPENAI_PACER.wait_turn()
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "%s | tentativa %d/%d | timeout=%ss | schema=%s",
                request_label,
                attempt,
                max_attempts,
                timeout_s,
                "sim" if schema is not None else "nao",
            )
        stop_event, hb_thread, started_at = _start_wait_heartbeat(request_label)
        try:
            response = OPENAI_SESSION.post(
                f"{OPENAI_BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=timeout_s,
            )
            elapsed = time.monotonic() - started_at
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "%s | tentativa %d/%d concluida em %.2fs | status=%s",
                    request_label,
                    attempt,
                    max_attempts,
                    elapsed,
                    response.status_code,
                )
        except requests.RequestException as exc:
            elapsed = time.monotonic() - started_at
            LOGGER.warning(
                "%s falhou na tentativa %d/%d apos %.2fs: %s",
                request_label,
                attempt,
                max_attempts,
                elapsed,
                exc,
            )
            if isinstance(exc, requests.ReadTimeout):
                should_log_disable = False
                with OPENAI_RUNTIME_STATE_LOCK:
                    OPENAI_TIMEOUT_STREAK += 1
                    if (
                        schema is not None
                        and OPENAI_TIMEOUT_STREAK >= 2
                        and not OPENAI_SCHEMA_TEMP_DISABLED
                    ):
                        OPENAI_SCHEMA_TEMP_DISABLED = True
                        should_log_disable = True
                    timeout_streak = OPENAI_TIMEOUT_STREAK
                if should_log_disable:
                    LOGGER.warning(
                        "[OpenAI] Instabilidade detectada (%d timeout(s) consecutivos). "
                        "Schema sera temporariamente despriorizado; lotes seguintes irao direto ao fallback compacto.",
                        timeout_streak,
                    )
            if attempt < max_attempts:
                time.sleep(_retry_delay_s(attempt))
                continue
            raise RuntimeError(f"Falha de rede na OpenAI: {exc}") from exc
        finally:
            stop_event.set()
            hb_thread.join(timeout=0.2)

        if 200 <= response.status_code < 300:
            with OPENAI_RUNTIME_STATE_LOCK:
                OPENAI_TIMEOUT_STREAK = 0
            obj = response.json()
            return obj if isinstance(obj, dict) else {}

        msg = _extract_http_message(response)
        if (
            response.status_code == 400
            and not compat_dropped_reasoning
            and ("reasoning_effort" in msg.lower() or "verbosity" in msg.lower() or "unknown parameter" in msg.lower())
        ):
            payload.pop("reasoning_effort", None)
            payload.pop("verbosity", None)
            compat_dropped_reasoning = True
            with OPENAI_RUNTIME_STATE_LOCK:
                OPENAI_DISABLE_REASONING_PARAMS = True
            LOGGER.warning(
                "%s recebeu 400 por parametro de compatibilidade; repetindo sem reasoning/verbosity. "
                "Parametros foram desativados globalmente para as proximas chamadas.",
                request_label,
            )
            if attempt < max_attempts:
                time.sleep(0.5)
                continue
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < max_attempts:
            delay = _retry_delay_s(attempt, response.headers.get("Retry-After"))
            LOGGER.warning(
                "OpenAI retornou %s. Retry em %.2fs. msg=%s",
                response.status_code,
                delay,
                msg,
            )
            time.sleep(delay)
            continue

        raise RuntimeError(f"Erro OpenAI {response.status_code}: {msg or '<sem mensagem>'}")

    raise RuntimeError("Falha na chamada OpenAI.")


def _parse_openai_content(content_obj: Any) -> str:
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
    return _normalize_ws(content_obj)


def _strip_markdown_fence(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _parse_json_loose(text: str) -> Optional[Any]:
    if not text:
        return None
    clean = _strip_markdown_fence(text)
    try:
        obj = json.loads(clean)
        if isinstance(obj, (dict, list)):
            return obj
        return None
    except Exception:
        pass
    list_start = clean.find("[")
    list_end = clean.rfind("]")
    if list_start >= 0 and list_end > list_start:
        snippet = clean[list_start : list_end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        snippet = clean[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, (dict, list)):
                return obj
            return None
        except Exception:
            return None
    return None


def _extract_openai_choice_text(raw: Any) -> tuple[str, str]:
    choice = (raw.get("choices") or [{}])[0] if isinstance(raw, dict) else {}
    if not isinstance(choice, dict):
        return "", ""
    message_obj = choice.get("message", {}) or {}
    finish_reason = _normalize_ws(choice.get("finish_reason", ""))
    content_obj = (message_obj if isinstance(message_obj, dict) else {}).get("content", "")
    parsed_text = _parse_openai_content(content_obj)
    return finish_reason, parsed_text


def _notion_page_url_from_id(page_id: str) -> str:
    no_dash = re.sub(r"-", "", page_id)
    return f"https://www.notion.so/{no_dash}"


def _rich_text_plain_text(rich_text: Any) -> str:
    if not isinstance(rich_text, list):
        return ""
    out: List[str] = []
    for item in rich_text:
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


def _block_rich_text(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    block_type = str(block.get("type", "") or "")
    body = block.get(block_type)
    if isinstance(body, dict):
        rich_text = body.get("rich_text")
        if isinstance(rich_text, list):
            return [item for item in rich_text if isinstance(item, dict)]
    return []


def _table_row_rich_text(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    if str(block.get("type", "") or "") != "table_row":
        return []
    body = block.get("table_row")
    if not isinstance(body, dict):
        return []
    cells = body.get("cells")
    if not isinstance(cells, list):
        return []
    out: List[Dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, list):
            continue
        for item in cell:
            if isinstance(item, dict):
                out.append(item)
    return out


def _iter_block_rich_text_items(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = _block_rich_text(block)
    if items:
        return items
    return _table_row_rich_text(block)


def _block_plain_text(block: Dict[str, Any]) -> str:
    return _rich_text_plain_text(_block_rich_text(block))


def _table_row_plain_text(block: Dict[str, Any]) -> str:
    if str(block.get("type", "") or "") != "table_row":
        return ""
    body = block.get("table_row")
    if not isinstance(body, dict):
        return ""
    cells = body.get("cells")
    if not isinstance(cells, list):
        return ""
    cell_texts: List[str] = []
    for cell in cells:
        text = _rich_text_plain_text(cell if isinstance(cell, list) else [])
        if text:
            cell_texts.append(text)
    return " | ".join(cell_texts).strip()


def _block_plain_text_any(block: Dict[str, Any]) -> str:
    text = _block_plain_text(block)
    if text:
        return text
    return _table_row_plain_text(block)


def _contains_auto_marker(text: Any) -> bool:
    value = str(text or "")
    if not value:
        return False
    return any(marker in value for marker in AUTO_MARKER_PREFIXES)


def _block_contains_auto_marker(block: Dict[str, Any]) -> bool:
    return _contains_auto_marker(_block_plain_text_any(block))


def retrieve_all_block_children_recursive(block_id: str) -> List[Dict[str, Any]]:
    all_blocks: List[Dict[str, Any]] = []

    def _walk(parent_id: str, depth: int) -> None:
        cursor = None
        while True:
            params: Dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            payload = notion_request("GET", f"/v1/blocks/{parent_id}/children", params=params)
            results = payload.get("results", [])
            if not isinstance(results, list):
                results = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                item["_depth"] = depth
                item["_parent_id"] = parent_id
                all_blocks.append(item)
                child_id = str(item.get("id", "") or "")
                if child_id and bool(item.get("has_children")):
                    _walk(child_id, depth + 1)
            if not bool(payload.get("has_more")):
                break
            cursor = payload.get("next_cursor")
            if not cursor:
                break

    _walk(block_id, 0)
    return all_blocks


def extract_textual_blocks(page_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in page_blocks:
        block_id = str(block.get("id", "") or "")
        block_type = str(block.get("type", "") or "")
        if not block_id or block_type not in SUPPORTED_TEXT_BLOCK_TYPES:
            continue
        text = _block_plain_text(block)
        out.append(
            {
                "id": block_id,
                "type": block_type,
                "text": text,
                "depth": int(block.get("_depth", 0) or 0),
                "block": block,
            }
        )
    return out


def _collect_auto_root_block_ids(page_blocks: Sequence[Dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for block in page_blocks:
        block_id = _normalize_ws(block.get("id", ""))
        if not block_id:
            continue
        if _block_contains_auto_marker(block):
            out.add(block_id)
    return out


def _block_has_ancestor(block: Dict[str, Any], ancestor_ids: set[str]) -> bool:
    parent_id = _normalize_ws(block.get("_parent_id", ""))
    visited: set[str] = set()
    while parent_id and parent_id not in visited:
        if parent_id in ancestor_ids:
            return True
        visited.add(parent_id)
        parent = BLOCK_INDEX.get(parent_id)
        if not isinstance(parent, dict):
            break
        parent_id = _normalize_ws(parent.get("_parent_id", ""))
    return False


def _collect_ignored_block_ids(page_blocks: Sequence[Dict[str, Any]], auto_root_ids: set[str]) -> set[str]:
    ignored = set(auto_root_ids)
    if not auto_root_ids:
        return ignored
    for block in page_blocks:
        block_id = _normalize_ws(block.get("id", ""))
        if not block_id or block_id in ignored:
            continue
        if _block_has_ancestor(block, auto_root_ids):
            ignored.add(block_id)
    return ignored


def _extract_notion_ids_from_text(text: str) -> List[str]:
    out: List[str] = []
    for raw_url in URL_RE.findall(str(text or "")):
        if "notion.so" not in raw_url.lower():
            continue
        try:
            out.append(extract_notion_id_from_url(raw_url))
        except Exception:
            continue
    return out


def _normalize_notion_reference_url(raw_url: Any) -> str:
    value = _normalize_ws(raw_url)
    if not value:
        return ""
    value = value.strip("<>")
    value = value.lstrip("(")
    value = value.rstrip(".,;:!?)")
    if "notion.so" not in value.casefold():
        return ""
    if not value.startswith("http://") and not value.startswith("https://"):
        return ""
    return value


def _extract_notion_urls_from_text(text: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw_url in URL_RE.findall(str(text or "")):
        cleaned = _normalize_notion_reference_url(raw_url)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def extract_notion_page_links_from_blocks(page_blocks: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for block in page_blocks:
        block_type = str(block.get("type", "") or "")
        if block_type == "link_to_page":
            obj = block.get("link_to_page")
            if isinstance(obj, dict):
                page_id = obj.get("page_id")
                if page_id:
                    try:
                        ids.append(_normalize_notion_id(str(page_id)))
                    except Exception:
                        pass
        rich_text = _block_rich_text(block)
        for item in rich_text:
            href = item.get("href")
            if isinstance(href, str) and "notion.so" in href.lower():
                try:
                    ids.append(extract_notion_id_from_url(href))
                except Exception:
                    pass
            mention = item.get("mention")
            if isinstance(mention, dict):
                mention_type = mention.get("type")
                if mention_type == "page":
                    page = mention.get("page")
                    if isinstance(page, dict):
                        page_id = page.get("id")
                        if page_id:
                            try:
                                ids.append(_normalize_notion_id(str(page_id)))
                            except Exception:
                                pass
            plain = item.get("plain_text")
            if isinstance(plain, str):
                ids.extend(_extract_notion_ids_from_text(plain))
    deduped = _safe_join((_notion_page_url_from_id(page_id) for page_id in ids), sep="\n").splitlines()
    return deduped


def _extract_notion_page_ids_from_block(block: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    block_type = str(block.get("type", "") or "")
    if block_type == "link_to_page":
        obj = block.get("link_to_page")
        if isinstance(obj, dict):
            page_id = obj.get("page_id")
            if page_id:
                try:
                    ids.append(_normalize_notion_id(str(page_id)))
                except Exception:
                    pass

    rich_text = _iter_block_rich_text_items(block)
    for item in rich_text:
        href = item.get("href")
        if isinstance(href, str) and "notion.so" in href.lower():
            try:
                ids.append(extract_notion_id_from_url(href))
            except Exception:
                pass

        mention = item.get("mention")
        if isinstance(mention, dict) and mention.get("type") == "page":
            page = mention.get("page")
            if isinstance(page, dict):
                page_id = page.get("id")
                if page_id:
                    try:
                        ids.append(_normalize_notion_id(str(page_id)))
                    except Exception:
                        pass

        plain = item.get("plain_text")
        if isinstance(plain, str):
            for raw_id in _extract_notion_ids_from_text(plain):
                try:
                    ids.append(_normalize_notion_id(raw_id))
                except Exception:
                    pass

    out: List[str] = []
    seen: set[str] = set()
    for page_id in ids:
        if page_id in seen:
            continue
        seen.add(page_id)
        out.append(page_id)
    return out


def _extract_notion_urls_from_block(block: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        cleaned = _normalize_notion_reference_url(candidate)
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        urls.append(cleaned)

    block_type = str(block.get("type", "") or "")
    if block_type == "link_to_page":
        obj = block.get("link_to_page")
        if isinstance(obj, dict):
            page_id = obj.get("page_id")
            if isinstance(page_id, str) and page_id:
                try:
                    _add(_notion_page_url_from_id(_normalize_notion_id(page_id)))
                except Exception:
                    pass

    rich_text = _iter_block_rich_text_items(block)
    for item in rich_text:
        href = item.get("href")
        if isinstance(href, str):
            _add(href)

        text_obj = item.get("text")
        if isinstance(text_obj, dict):
            link_obj = text_obj.get("link")
            if isinstance(link_obj, dict):
                _add(link_obj.get("url"))

        mention = item.get("mention")
        if isinstance(mention, dict) and mention.get("type") == "page":
            page_obj = mention.get("page")
            if isinstance(page_obj, dict):
                page_id = page_obj.get("id")
                if isinstance(page_id, str) and page_id:
                    try:
                        _add(_notion_page_url_from_id(_normalize_notion_id(page_id)))
                    except Exception:
                        pass

        plain = item.get("plain_text")
        if isinstance(plain, str):
            for found_url in _extract_notion_urls_from_text(plain):
                _add(found_url)

    return urls


def _rich_text_item_plain_text(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    plain = item.get("plain_text")
    if isinstance(plain, str):
        return plain
    text_obj = item.get("text")
    if isinstance(text_obj, dict):
        content = text_obj.get("content")
        if isinstance(content, str):
            return content
    return ""


def _rich_text_item_link_url(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    href = item.get("href")
    if isinstance(href, str) and href:
        return href
    text_obj = item.get("text")
    if isinstance(text_obj, dict):
        link_obj = text_obj.get("link")
        if isinstance(link_obj, dict):
            url = link_obj.get("url")
            if isinstance(url, str):
                return url
    mention = item.get("mention")
    if isinstance(mention, dict) and mention.get("type") == "page":
        page_obj = mention.get("page")
        if isinstance(page_obj, dict):
            page_id = page_obj.get("id")
            if isinstance(page_id, str) and page_id:
                try:
                    return _notion_page_url_from_id(_normalize_notion_id(page_id))
                except Exception:
                    return ""
    return ""


def _wrap_markdown_bold(text: str, *, bold: bool) -> str:
    if not text:
        return ""
    if not bold:
        return text
    return f"**{text}**"


def _replace_notion_urls_in_text_with_refs(
    text: str,
    reference_numbers: Dict[str, int],
) -> tuple[str, bool]:
    raw_text = str(text or "")
    if not raw_text:
        return "", False
    replaced = False

    def _repl(match: re.Match[str]) -> str:
        nonlocal replaced
        raw_url = match.group(0)
        cleaned = _normalize_notion_reference_url(raw_url)
        if not cleaned:
            return raw_url
        idx = reference_numbers.get(cleaned.casefold())
        if not idx:
            return raw_url
        replaced = True
        suffix = raw_url[len(cleaned) :] if raw_url.startswith(cleaned) else ""
        return f"[{idx}]({cleaned}){suffix}"

    return URL_RE.sub(_repl, raw_text), replaced


def _render_rich_text_items_with_global_refs(
    items: Sequence[Dict[str, Any]],
    reference_numbers: Dict[str, int],
) -> tuple[str, bool]:
    parts: List[str] = []
    touched_notion_ref = False

    for item in items:
        if not isinstance(item, dict):
            continue
        bold = bool((item.get("annotations") or {}).get("bold"))
        raw_link_url = _rich_text_item_link_url(item)
        notion_link_url = _normalize_notion_reference_url(raw_link_url)
        if notion_link_url:
            idx = reference_numbers.get(notion_link_url.casefold())
            if idx:
                parts.append(_wrap_markdown_bold(f"[{idx}]({notion_link_url})", bold=bold))
                touched_notion_ref = True
                continue

        text = _rich_text_item_plain_text(item)
        if not text:
            continue

        text, replaced_raw_url = _replace_notion_urls_in_text_with_refs(text, reference_numbers)
        touched_notion_ref = touched_notion_ref or replaced_raw_url

        external_link_url = _normalize_ws(raw_link_url)
        if external_link_url and not notion_link_url:
            if text == external_link_url:
                rendered = external_link_url
            else:
                rendered = f"[{text}]({external_link_url})"
        else:
            rendered = text
        parts.append(_wrap_markdown_bold(rendered, bold=bold))

    return "".join(parts), touched_notion_ref


def _build_global_notion_reference_numbers(page_blocks: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    reference_numbers: Dict[str, int] = {}
    next_index = 1
    for block in page_blocks:
        for url in _extract_notion_urls_from_block(block):
            key = url.casefold()
            if key in reference_numbers:
                continue
            reference_numbers[key] = next_index
            next_index += 1
    return reference_numbers


def _page_parent_matches_source(
    page_id: str,
    *,
    source_database_id: str,
    source_data_source_id: str,
    cache: Dict[str, bool],
) -> bool:
    if page_id in cache:
        return cache[page_id]

    matches = False
    try:
        page_obj = notion_request("GET", f"/v1/pages/{page_id}")
        parent = page_obj.get("parent")
        if isinstance(parent, dict):
            parent_type = str(parent.get("type", "") or "")
            raw_parent_id = parent.get(parent_type) if parent_type else ""
            parent_id = ""
            if isinstance(raw_parent_id, str) and raw_parent_id:
                try:
                    parent_id = _normalize_notion_id(raw_parent_id)
                except Exception:
                    parent_id = ""

            if parent_type == "data_source_id" and parent_id and parent_id == source_data_source_id:
                matches = True
            elif parent_type == "database_id" and parent_id and parent_id == source_database_id:
                matches = True
            elif parent_id and (parent_id == source_database_id or parent_id == source_data_source_id):
                matches = True
    except Exception as exc:
        LOGGER.warning("Falha ao validar parent da pagina vinculada %s: %s", page_id, exc)
        matches = False

    cache[page_id] = matches
    return matches


def select_target_text_blocks_by_source_links(
    textual_blocks: Sequence[Dict[str, Any]],
    *,
    source_database_id: str,
    source_data_source_id: str,
) -> tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    target_blocks: List[Dict[str, Any]] = []
    source_linked_page_ids: List[str] = []
    seen_source_page_ids: set[str] = set()
    parent_cache: Dict[str, bool] = {}
    blocks_with_any_notion_link = 0

    total = len(textual_blocks)
    LOGGER.info(
        "[Filtro-alvo] Validando blocos com links explicitos para a base fonte... total=%d",
        total,
    )
    for idx, block in enumerate(textual_blocks, start=1):
        raw_block = block.get("block")
        if not isinstance(raw_block, dict):
            continue
        linked_page_ids = _extract_notion_page_ids_from_block(raw_block)
        if not linked_page_ids:
            continue
        blocks_with_any_notion_link += 1

        matched_ids: List[str] = []
        for linked_page_id in linked_page_ids:
            if _page_parent_matches_source(
                linked_page_id,
                source_database_id=source_database_id,
                source_data_source_id=source_data_source_id,
                cache=parent_cache,
            ):
                matched_ids.append(linked_page_id)

        if matched_ids:
            target_item = dict(block)
            target_item["source_linked_page_ids"] = list(matched_ids)
            target_blocks.append(target_item)
            for linked_page_id in matched_ids:
                if linked_page_id in seen_source_page_ids:
                    continue
                seen_source_page_ids.add(linked_page_id)
                source_linked_page_ids.append(linked_page_id)

        if idx % 40 == 0 or idx == total:
            LOGGER.info(
                "[Filtro-alvo] progresso %d/%d | com_link_notion=%d | alvos=%d",
                idx,
                total,
                blocks_with_any_notion_link,
                len(target_blocks),
            )

    stats = {
        "text_blocks_total": total,
        "text_blocks_with_any_notion_link": blocks_with_any_notion_link,
        "text_blocks_targeted": len(target_blocks),
        "source_linked_pages": len(source_linked_page_ids),
        "text_blocks_not_targeted": max(0, total - len(target_blocks)),
    }
    LOGGER.info(
        "[Filtro-alvo] concluido | total=%d | com_link=%d | alvo=%d | fora_criterio=%d | paginas_fonte=%d",
        stats["text_blocks_total"],
        stats["text_blocks_with_any_notion_link"],
        stats["text_blocks_targeted"],
        stats["text_blocks_not_targeted"],
        stats["source_linked_pages"],
    )
    return target_blocks, source_linked_page_ids, stats


def select_table_companion_targets_by_source_links(
    page_blocks: Sequence[Dict[str, Any]],
    *,
    source_database_id: str,
    source_data_source_id: str,
    ignored_block_ids: Optional[set[str]] = None,
) -> tuple[List[TableCompanionTarget], List[str], Dict[str, int]]:
    ignored_ids = ignored_block_ids or set()
    parent_cache: Dict[str, bool] = {}
    source_linked_page_ids: List[str] = []
    seen_global_page_ids: set[str] = set()
    target_order: List[str] = []
    target_map: Dict[str, TableCompanionTarget] = {}
    table_rows_with_any_notion_link = 0

    table_rows = [
        block
        for block in page_blocks
        if str(block.get("type", "") or "") == "table_row"
        and _normalize_ws(block.get("id", "")) not in ignored_ids
    ]
    total = len(table_rows)
    LOGGER.info(
        "[Filtro-tabela] Validando linhas de tabela com links explicitos para a base fonte... total=%d",
        total,
    )

    for idx, block in enumerate(table_rows, start=1):
        linked_page_ids = _extract_notion_page_ids_from_block(block)
        if not linked_page_ids:
            continue
        table_rows_with_any_notion_link += 1

        matched_ids: List[str] = []
        for linked_page_id in linked_page_ids:
            if _page_parent_matches_source(
                linked_page_id,
                source_database_id=source_database_id,
                source_data_source_id=source_data_source_id,
                cache=parent_cache,
            ):
                matched_ids.append(linked_page_id)
        if not matched_ids:
            continue

        table_id = _normalize_ws(block.get("_parent_id", ""))
        table_block = BLOCK_INDEX.get(table_id)
        if not table_id or not isinstance(table_block, dict):
            continue
        if str(table_block.get("type", "") or "") != "table":
            continue
        parent_id = _normalize_ws(table_block.get("_parent_id", ""))
        if not parent_id:
            continue

        row_text = _table_row_plain_text(block)
        if table_id not in target_map:
            target_map[table_id] = TableCompanionTarget(
                table_id=table_id,
                parent_id=parent_id,
                source_linked_page_ids=[],
                row_text_by_page_id={},
                marker=f"{AUTO_TABLE_MARKER_PREFIX}{table_id}]]",
            )
            target_order.append(table_id)
        target = target_map[table_id]

        for matched_id in matched_ids:
            if matched_id not in target.row_text_by_page_id:
                target.source_linked_page_ids.append(matched_id)
                target.row_text_by_page_id[matched_id] = row_text
            if matched_id not in seen_global_page_ids:
                seen_global_page_ids.add(matched_id)
                source_linked_page_ids.append(matched_id)

        if idx % 40 == 0 or idx == total:
            LOGGER.info(
                "[Filtro-tabela] progresso %d/%d | linhas_com_link=%d | tabelas_alvo=%d",
                idx,
                total,
                table_rows_with_any_notion_link,
                len(target_order),
            )

    targets = [target_map[table_id] for table_id in target_order]
    stats = {
        "table_rows_total": total,
        "table_rows_with_any_notion_link": table_rows_with_any_notion_link,
        "tables_targeted": len(targets),
        "table_companion_cases": sum(len(item.source_linked_page_ids) for item in targets),
    }
    LOGGER.info(
        "[Filtro-tabela] concluido | linhas=%d | linhas_com_link=%d | tabelas_alvo=%d | casos=%d | paginas_fonte=%d",
        stats["table_rows_total"],
        stats["table_rows_with_any_notion_link"],
        stats["tables_targeted"],
        stats["table_companion_cases"],
        len(source_linked_page_ids),
    )
    return targets, source_linked_page_ids, stats


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
        "Nao foi possivel identificar data_source_id no database informado. "
        "Confirme a versao da API e o compartilhamento da integracao."
    )


def _property_to_text(prop: Dict[str, Any]) -> str:
    prop_type = str(prop.get("type", "") or "")
    if prop_type == "title":
        return _rich_text_plain_text(prop.get("title"))
    if prop_type == "rich_text":
        return _rich_text_plain_text(prop.get("rich_text"))
    if prop_type == "number":
        value = prop.get("number")
        return "" if value is None else str(value)
    if prop_type == "url":
        return _normalize_ws(prop.get("url"))
    if prop_type == "email":
        return _normalize_ws(prop.get("email"))
    if prop_type == "phone_number":
        return _normalize_ws(prop.get("phone_number"))
    if prop_type == "select":
        obj = prop.get("select")
        if isinstance(obj, dict):
            return _normalize_ws(obj.get("name"))
    if prop_type == "status":
        obj = prop.get("status")
        if isinstance(obj, dict):
            return _normalize_ws(obj.get("name"))
    if prop_type == "multi_select":
        items = prop.get("multi_select")
        if isinstance(items, list):
            return _safe_join((str(item.get("name", "")) for item in items if isinstance(item, dict)))
    if prop_type == "date":
        obj = prop.get("date")
        if isinstance(obj, dict):
            start = _normalize_ws(obj.get("start"))
            end = _normalize_ws(obj.get("end"))
            if start and end:
                return f"{start} a {end}"
            return start
    if prop_type == "checkbox":
        return "true" if bool(prop.get("checkbox")) else "false"
    if prop_type == "formula":
        obj = prop.get("formula")
        if isinstance(obj, dict):
            sub_type = obj.get("type")
            value = obj.get(sub_type) if sub_type else ""
            if isinstance(value, dict):
                return _normalize_ws(value.get("start") or value.get("name") or value.get("id"))
            if isinstance(value, list):
                return _safe_join((str(v) for v in value))
            return _normalize_ws(value)
    if prop_type == "relation":
        items = prop.get("relation")
        if isinstance(items, list):
            return _safe_join((str(item.get("id", "")) for item in items if isinstance(item, dict)))
    if prop_type == "people":
        items = prop.get("people")
        if isinstance(items, list):
            return _safe_join((str(item.get("name", "")) for item in items if isinstance(item, dict)))
    if prop_type == "created_time":
        return _normalize_ws(prop.get("created_time"))
    if prop_type == "last_edited_time":
        return _normalize_ws(prop.get("last_edited_time"))
    return ""


def _page_properties_text(page_obj: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    props = page_obj.get("properties")
    if not isinstance(props, dict):
        return out
    for key, prop in props.items():
        if not isinstance(prop, dict):
            continue
        out[str(key)] = _property_to_text(prop)
    return out


def _normalize_prop_name(value: Any) -> str:
    raw = _normalize_ws(value).casefold()
    if not raw:
        return ""
    nfkd = unicodedata.normalize("NFKD", raw)
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", stripped)


def _pick_property_text(properties_text: Dict[str, str], aliases: Sequence[str]) -> str:
    alias_norm = [_normalize_prop_name(alias) for alias in aliases]
    alias_norm = [alias for alias in alias_norm if alias]
    if not alias_norm:
        return ""

    # Passo 1: match exato do nome da coluna normalizada.
    for prop_name, prop_text in properties_text.items():
        value = _normalize_ws(prop_text)
        if not value:
            continue
        prop_norm = _normalize_prop_name(prop_name)
        if prop_norm in alias_norm:
            return value

    # Passo 2: match por contem apenas para aliases minimamente informativos.
    for prop_name, prop_text in properties_text.items():
        value = _normalize_ws(prop_text)
        if not value:
            continue
        prop_norm = _normalize_prop_name(prop_name)
        if any(len(alias) >= 5 and alias in prop_norm for alias in alias_norm):
            return value

    return ""


def _text_sentences(text: str) -> List[str]:
    raw = str(text or "").replace("\r", "\n")
    parts = re.split(r"(?<=[\.\!\?;])\s+|\n+", raw)
    sentences = [_normalize_ws(part) for part in parts if _normalize_ws(part)]
    return sentences


def _pick_sentences(text: str, keywords: Sequence[str], max_items: int = 3) -> str:
    sentences = _text_sentences(text)
    out: List[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        low = sentence.casefold()
        if not any(keyword in low for keyword in keywords):
            continue
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
        if len(out) >= max_items:
            break
    return _safe_join(out, sep=" | ")


def _extract_cnj(text: str) -> str:
    match = CNJ_RE.search(text or "")
    if match:
        return match.group(0)
    digits = re.sub(r"\D", "", str(text or ""))
    if len(digits) >= 20:
        first = digits[:20]
        return (
            f"{first[0:7]}-{first[7:9]}.{first[9:13]}."
            f"{first[13]}.{first[14:16]}.{first[16:20]}"
        )
    return ""


def _extract_election_year(text: str) -> str:
    low = _normalize_lower(text)
    scoped = re.findall(r"elei(?:cao|coes|caoes|ção|ções)[^\n\r]{0,60}", low)
    for segment in scoped:
        for year in YEAR_RE.findall(segment):
            if 1990 <= int(year) <= 2100:
                return year
    for year in YEAR_RE.findall(text or ""):
        if 1990 <= int(year) <= 2100:
            return year
    return ""


def _extract_city_uf(text: str) -> str:
    match = CITY_UF_RE.search(text or "")
    if not match:
        return ""
    city = _normalize_ws(match.group(1))
    uf = _normalize_ws(match.group(2)).upper()
    return f"{city}/{uf}"


def _looks_national_party_context(text: str) -> bool:
    low = _normalize_lower(text)
    if not low:
        return False
    return bool(
        re.search(
            r"\((?:[a-z]{2,20})\)\s*-\s*nacional|"
            r"\bdiret[oó]rio\s+nacional\b|"
            r"\bexecutiva\s+nacional\b|"
            r"\bcomiss[aã]o\s+provis[oó]ria\s+nacional\b|"
            r"\bpartido\b.{0,40}\bnacional\b",
            low,
        )
    )


def _format_party_token(token: str, *, sigla_uf: str = "", is_national: bool = False) -> str:
    clean = _normalize_ws(token).upper()
    if not clean:
        return ""
    if "/" in clean:
        return clean
    if is_national:
        return f"{clean}/Nacional"
    uf = _normalize_ws(sigla_uf).upper()
    if re.fullmatch(r"[A-Z]{2}", uf or ""):
        return f"{clean}/{uf}"
    return clean


def _extract_partidos(text: str, sigla_uf: str = "") -> str:
    is_national = _looks_national_party_context(text)
    out: List[str] = []
    for match in PARTY_UF_RE.finditer(text or ""):
        token = _normalize_ws(match.group(1)).upper()
        base = token.split("/", 1)[0]
        if base in PARTY_ABBREVIATIONS:
            out.append(token)
    for match in PARTY_FULL_NAME_RE.finditer(text or ""):
        full_name = _normalize_ws(match.group(1))
        sigla = _normalize_ws(match.group(2)).upper()
        scope = _normalize_ws(match.group(3)).upper()
        full_name_low = _normalize_lower(full_name)
        looks_like_party_name = (
            sigla in PARTY_ABBREVIATIONS
            or "partido" in full_name_low
            or "coliga" in full_name_low
            or "diret" in full_name_low
            or "federa" in full_name_low
            or "comiss" in full_name_low
            or "executiva" in full_name_low
        )
        if not looks_like_party_name:
            continue
        out.append(
            _format_party_token(
                sigla,
                sigla_uf=sigla_uf,
                is_national=(scope == "NACIONAL") or is_national,
            )
        )
    for match in PARTY_TOKEN_RE.finditer(text or ""):
        token = _normalize_ws(match.group(0)).upper()
        if token:
            out.append(_format_party_token(token, sigla_uf=sigla_uf, is_national=is_national))

    deduped = _safe_join(out).split(", ") if out else []
    scoped_bases = {
        token.split("/", 1)[0].casefold()
        for token in deduped
        if "/" in token
    }
    filtered = [
        token
        for token in deduped
        if "/" in token or token.casefold() not in scoped_bases
    ]
    return _safe_join(filtered)


def _extract_advogados(text: str) -> str:
    names: List[str] = []
    for block_match in ADV_BLOCK_RE.finditer(text or ""):
        block = block_match.group(1) or ""
        for name_match in ADV_OAB_RE.finditer(block):
            names.append(_normalize_ws(name_match.group(1)))
    return _safe_join(names)


def _best_partes(properties_text: Dict[str, str], full_text: str) -> str:
    prioritized_keys = [
        "partes",
        "parte",
        "autor",
        "autora",
        "reu",
        "réu",
        "recorrente",
        "recorrido",
        "agravante",
        "agravado",
    ]
    for key in prioritized_keys:
        for prop_name, prop_text in properties_text.items():
            if key in _normalize_lower(prop_name) and _normalize_ws(prop_text):
                return _normalize_ws(prop_text)
    extracted = _pick_sentences(
        full_text,
        keywords=[
            "autor",
            "autora",
            "reu",
            "réu",
            "recorrente",
            "recorrido",
            "agravante",
            "agravado",
            "parte",
            "partes",
            "coligacao",
            "coligação",
            "partido",
        ],
        max_items=2,
    )
    return extracted


def _aggregate_text_from_blocks(blocks: Sequence[Dict[str, Any]]) -> str:
    texts: List[str] = []
    for block in blocks:
        txt = _block_plain_text(block)
        if txt:
            texts.append(txt)
    return "\n".join(texts)


def _context_has_signal(ctx: CaseContext) -> bool:
    joined = " ".join(
        [
            ctx.processo_cnj,
            ctx.numero_unico,
            ctx.data_decisao,
            ctx.sigla_classe,
            ctx.nome_municipio,
            ctx.sigla_uf,
            ctx.relator,
            ctx.tema,
            ctx.partes,
            ctx.punchline,
            ctx.texto_decisao,
            ctx.partidos,
            ctx.alegacoes,
            ctx.fundamentos,
            ctx.resultado,
            ctx.advogados,
        ]
    )
    return bool(_normalize_ws(joined))


def _build_context_from_page(page_obj: Dict[str, Any], page_text: str, source_url: str) -> CaseContext:
    prop_map = _page_properties_text(page_obj)
    prop_blob = "\n".join(f"{k}: {v}" for k, v in prop_map.items() if _normalize_ws(v))
    full_text = f"{prop_blob}\n{page_text}".strip()

    numero_unico = _pick_property_text(
        prop_map,
        aliases=[
            "numeroUnico",
            "numero unico",
            "número único",
            "numero do processo",
            "número do processo",
        ],
    )
    data_decisao = _pick_property_text(
        prop_map,
        aliases=["dataDecisao", "data decisao", "data de decisao", "data do julgamento"],
    )
    sigla_classe = _pick_property_text(
        prop_map,
        aliases=["siglaClasse", "sigla classe", "classe"],
    )
    nome_municipio = _pick_property_text(
        prop_map,
        aliases=["nomeMunicipio", "nome municipio", "municipio", "cidade"],
    )
    sigla_uf = _pick_property_text(
        prop_map,
        aliases=["siglaUF", "sigla uf", "uf", "estado"],
    )
    relator = _pick_property_text(
        prop_map,
        aliases=["relator", "relatora", "ministro relator", "juiz relator"],
    )
    advogados_coluna = _pick_property_text(
        prop_map,
        aliases=["advogados", "advogado", "banca de advogados", "banca"],
    )
    tema = _pick_property_text(
        prop_map,
        aliases=["tema", "tema principal", "assunto", "topico"],
    )
    punchline = _pick_property_text(
        prop_map,
        aliases=["punchline", "ementa curta", "resumo", "sintese"],
    )
    texto_decisao = _pick_property_text(
        prop_map,
        aliases=["textoDecisao", "texto decisao", "inteiro teor", "texto da decisao"],
    )
    analysis_text = "\n".join(
        [
            _normalize_multiline_ws(punchline),
            _normalize_multiline_ws(texto_decisao),
            _normalize_multiline_ws(page_text),
            full_text,
        ]
    ).strip()

    sigla_uf_clean = _normalize_ws(sigla_uf).upper()
    if not re.fullmatch(r"[A-Z]{2}", sigla_uf_clean or ""):
        sigla_uf_clean = ""
    nome_municipio_clean = _normalize_ws(nome_municipio)
    cidade_uf = ""
    if nome_municipio_clean and sigla_uf_clean:
        cidade_uf = f"{nome_municipio_clean}/{sigla_uf_clean}"
    if not cidade_uf:
        cidade_uf = _extract_city_uf(full_text)

    return CaseContext(
        processo_cnj=_extract_cnj(f"{numero_unico}\n{full_text}"),
        numero_unico=numero_unico,
        data_decisao=data_decisao,
        sigla_classe=sigla_classe,
        nome_municipio=nome_municipio_clean,
        sigla_uf=sigla_uf_clean,
        relator=relator,
        tema=tema,
        eleicao_ano=_extract_election_year(full_text),
        cidade_uf=cidade_uf,
        partes=_best_partes(prop_map, analysis_text or full_text),
        punchline=punchline,
        texto_decisao=texto_decisao,
        partidos=_extract_partidos(full_text, sigla_uf=sigla_uf_clean),
        alegacoes=_pick_sentences(
            analysis_text or full_text,
            keywords=["alega", "sustenta", "argumenta", "afirma", "defende", "tese"],
            max_items=3,
        ),
        advogados=advogados_coluna or _extract_advogados(full_text),
        fundamentos=_pick_sentences(
            analysis_text or full_text,
            keywords=["fundamento", "art.", "lei", "resolucao", "resolução", "sumula", "súmula"],
            max_items=3,
        ),
        resultado=_pick_sentences(
            analysis_text or full_text,
            keywords=[
                "resultado",
                "julg",
                "decid",
                "defer",
                "indefer",
                "conhec",
                "provido",
                "desprovido",
                "negou",
                "acolheu",
                "rejeitou",
            ],
            max_items=3,
        ),
        fonte_url=source_url,
    )


def _dedupe_contexts(items: Sequence[CaseContext], *, limit: int = MAX_CONTEXT_CASES) -> List[CaseContext]:
    out: List[CaseContext] = []
    seen: set[str] = set()
    for item in items:
        key = (
            _normalize_ws(item.processo_cnj)
            or _normalize_ws(item.fonte_url)
            or _normalize_ws(item.partes)[:120]
            or _normalize_ws(item.resultado)[:120]
        )
        if not key:
            continue
        key_low = key.casefold()
        if key_low in seen:
            continue
        seen.add(key_low)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def build_case_context_map_from_linked_pages(linked_page_ids: List[str]) -> Dict[str, CaseContext]:
    contexts_by_page_id: Dict[str, CaseContext] = {}
    seen_ids: set[str] = set()
    total_input = len(linked_page_ids)
    if total_input:
        LOGGER.info("[Contexto-links] Iniciando leitura de %d pagina(s) vinculada(s)...", total_input)
    for idx, raw_id in enumerate(linked_page_ids, start=1):
        try:
            page_id = _normalize_notion_id(raw_id)
        except Exception:
            continue
        if page_id in seen_ids:
            continue
        seen_ids.add(page_id)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("[Contexto-links] pagina %d/%d | id=%s", idx, total_input, page_id)
        try:
            page_obj = notion_request("GET", f"/v1/pages/{page_id}")
        except Exception as exc:
            LOGGER.warning("Falha ao ler pagina vinculada %s: %s", page_id, exc)
            continue
        try:
            blocks = retrieve_all_block_children_recursive(page_id)
            page_text = _aggregate_text_from_blocks(blocks)
        except Exception as exc:
            LOGGER.warning("Falha ao ler blocos da pagina vinculada %s: %s", page_id, exc)
            page_text = ""
        ctx = _build_context_from_page(page_obj, page_text, _notion_page_url_from_id(page_id))
        if _context_has_signal(ctx):
            contexts_by_page_id[page_id] = ctx
        if idx % LINK_CONTEXT_PROGRESS_EVERY == 0 or idx == total_input:
            LOGGER.info(
                "[Contexto-links] progresso %d/%d | contextos_validos=%d",
                idx,
                total_input,
                len(contexts_by_page_id),
            )
    LOGGER.info(
        "[Contexto-links] concluido | paginas_lidas=%d | contextos_validos=%d",
        len(seen_ids),
        len(contexts_by_page_id),
    )
    return contexts_by_page_id


def build_case_context_from_linked_pages(linked_page_ids: List[str]) -> List[CaseContext]:
    contexts_by_page_id = build_case_context_map_from_linked_pages(linked_page_ids)
    return _dedupe_contexts(list(contexts_by_page_id.values()))


def _extract_terms_from_report(textual_blocks: Sequence[Dict[str, Any]]) -> List[str]:
    joined = "\n".join(_normalize_ws(item.get("text", "")) for item in textual_blocks if item.get("text"))
    terms: List[str] = []
    terms.extend(CNJ_RE.findall(joined))
    terms.extend(PROC_SHORT_RE.findall(joined))
    for match in PARTY_TOKEN_RE.findall(joined):
        terms.append(_normalize_ws(match).upper())
    dedup: List[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = _normalize_ws(term)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(clean)
    return dedup[:24]


def query_fallback_cases_from_datasource(data_source_id: str, terms: List[str]) -> List[CaseContext]:
    normalized_terms = [t.casefold() for t in terms if len(_normalize_ws(t)) >= 4]
    results: List[CaseContext] = []
    cursor = None
    page_loops = 0
    seen_page_ids: set[str] = set()
    scanned_items = 0
    matched_items = 0

    LOGGER.info(
        "[Fallback] Iniciando consulta no data_source %s | termos=%d",
        data_source_id,
        len(normalized_terms),
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("[Fallback] termos=%s", normalized_terms)

    while page_loops < 10:
        page_loops += 1
        body: Dict[str, Any] = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        payload = notion_request("POST", f"/v1/data_sources/{data_source_id}/query", json_body=body)
        items = payload.get("results", [])
        if not isinstance(items, list):
            items = []
        LOGGER.info(
            "[Fallback] pagina %d recebida | itens=%d | has_more=%s",
            page_loops,
            len(items),
            bool(payload.get("has_more")),
        )

        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("object", "")) != "page":
                continue
            scanned_items += 1
            page_id_raw = item.get("id")
            if not page_id_raw:
                continue
            try:
                page_id = _normalize_notion_id(str(page_id_raw))
            except Exception:
                continue
            if page_id in seen_page_ids:
                continue
            seen_page_ids.add(page_id)

            prop_map = _page_properties_text(item)
            searchable = " ".join(prop_map.values()).casefold()
            if normalized_terms and not any(term in searchable for term in normalized_terms):
                continue
            matched_items += 1

            try:
                page_blocks = retrieve_all_block_children_recursive(page_id)
                page_text = _aggregate_text_from_blocks(page_blocks)
            except Exception as exc:
                LOGGER.warning("Falha ao ler blocos do fallback %s: %s", page_id, exc)
                page_text = ""

            context = _build_context_from_page(item, page_text, _notion_page_url_from_id(page_id))
            if _context_has_signal(context):
                results.append(context)
                if len(results) >= MAX_FALLBACK_CONTEXTS:
                    deduped_early = _dedupe_contexts(results, limit=MAX_FALLBACK_CONTEXTS)
                    LOGGER.info(
                        "[Fallback] limite atingido | analisados=%d | matches=%d | contextos=%d",
                        scanned_items,
                        matched_items,
                        len(deduped_early),
                    )
                    return deduped_early

            if scanned_items % FALLBACK_PROGRESS_EVERY == 0:
                LOGGER.info(
                    "[Fallback] progresso | analisados=%d | matches=%d | contextos=%d",
                    scanned_items,
                    matched_items,
                    len(results),
                )

        if not bool(payload.get("has_more")):
            break
        cursor = payload.get("next_cursor")
        if not cursor:
            break

    deduped = _dedupe_contexts(results, limit=MAX_FALLBACK_CONTEXTS)
    LOGGER.info(
        "[Fallback] concluido | paginas_query=%d | analisados=%d | matches=%d | contextos=%d",
        page_loops,
        scanned_items,
        matched_items,
        len(deduped),
    )
    return deduped


def _truncate_text(text: str, max_chars: int) -> str:
    clean = _normalize_ws(text)
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip()


def _context_for_prompt(contexts: Sequence[CaseContext]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in contexts[:MAX_CONTEXT_CASES]:
        out.append(
            {
                "processo_cnj": _truncate_text(item.processo_cnj, 40),
                "numero_unico": _truncate_text(item.numero_unico, 80),
                "data_decisao": _truncate_text(item.data_decisao, 32),
                "sigla_classe": _truncate_text(item.sigla_classe, 32),
                "nome_municipio": _truncate_text(item.nome_municipio, 120),
                "sigla_uf": _truncate_text(item.sigla_uf, 8),
                "relator": _truncate_text(item.relator, 140),
                "tema": _truncate_text(item.tema, 220),
                "eleicao_ano": _truncate_text(item.eleicao_ano, 8),
                "cidade_uf": _truncate_text(item.cidade_uf, 80),
                "partes": _truncate_text(item.partes, 220),
                "punchline": _truncate_text(item.punchline, 380),
                "texto_decisao": _truncate_text(item.texto_decisao, 1200),
                "partidos": _truncate_text(item.partidos, 120),
                "alegacoes": _truncate_text(item.alegacoes, 380),
                "advogados": _truncate_text(item.advogados, 220),
                "fundamentos": _truncate_text(item.fundamentos, 380),
                "resultado": _truncate_text(item.resultado, 320),
                "fonte_url": _truncate_text(item.fonte_url, 200),
            }
        )
    return out


def _ensure_party_markdown_bold(text: str) -> str:
    if not text:
        return ""

    pieces = re.split(r"(\*\*.+?\*\*)", text, flags=re.DOTALL)
    out: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) >= 4:
            out.append(piece)
            continue
        piece = GENERIC_PARTY_SCOPED_RE.sub(lambda m: f"**{m.group(1)}**", piece)
        piece = PARTY_UF_RE.sub(lambda m: f"**{m.group(1)}**", piece)
        subparts = re.split(r"(\*\*.+?\*\*)", piece, flags=re.DOTALL)
        rebuilt: List[str] = []
        for sub in subparts:
            if not sub:
                continue
            if sub.startswith("**") and sub.endswith("**") and len(sub) >= 4:
                rebuilt.append(sub)
                continue
            rebuilt.append(PARTY_TOKEN_RE.sub(lambda m: f"**{m.group(0)}**", sub))
        out.append("".join(rebuilt))
    return "".join(out)


def _build_openai_prompt_payload(
    block_batch: Sequence[Dict[str, Any]],
    context: Sequence[CaseContext],
    *,
    compact: bool,
) -> Dict[str, Any]:
    _ = context  # Contexto global desativado: cada bloco usa apenas seu proprio escopo.
    blocks_payload: List[Dict[str, Any]] = []
    scoped_context_limit = 2 if compact else 4
    for idx, block in enumerate(block_batch):
        block_type = str(block.get("type", "") or "")
        item_kind = _normalize_ws(block.get("item_kind", "")) or "patch_text_block"
        if compact:
            max_len = 120 if block_type.startswith("heading_") else 900
        else:
            max_len = 180 if block_type.startswith("heading_") else 1700
        refs = _block_reference_urls(block)

        scoped_contexts_raw = block.get("context_cases")
        scoped_contexts: List[CaseContext] = []
        if isinstance(scoped_contexts_raw, list):
            for ctx in scoped_contexts_raw:
                if isinstance(ctx, CaseContext):
                    scoped_contexts.append(ctx)
        scoped_contexts = _dedupe_contexts(scoped_contexts, limit=scoped_context_limit)

        blocks_payload.append(
            {
                "index": idx,
                "item_kind": item_kind,
                "block_type": block_type,
                "headline_semente": _truncate_text(str(block.get("headline_seed", "")), 220),
                "texto_original": _truncate_text(str(block.get("text", "")), max_len),
                "links_referencia": refs,
                "contexto_casos_bloco": _context_for_prompt(scoped_contexts),
            }
        )

    return {
        "objetivo": (
            "Melhorar textos de relatorio juridico eleitoral no Notion. "
            "Reescrever apenas os itens abaixo, sem inventar fatos e mantendo rastreabilidade para a base Notion."
        ),
        "regras_obrigatorias": [
            "Retorne um item por bloco com os campos padronizados exigidos.",
            "Use somente o contexto_casos_bloco do item correspondente ao mesmo index.",
            "Nao use contexto de um bloco para escrever outro bloco.",
            "Se contexto_casos_bloco vier vazio, reescreva com cautela usando apenas texto_original e links_referencia.",
            "Priorize os campos estruturados do contexto_casos_bloco quando presentes (numero_unico, data_decisao, sigla_classe, nome_municipio, sigla_uf, relator, partes, advogados, punchline, texto_decisao, tema, partidos, fundamentos, resultado).",
            "Para item_kind=table_companion, trate a unidade de analise como um processo judicial e use o contexto estruturado como fonte principal dos fatos.",
            "Para item_kind=patch_text_block, reescreva o proprio bloco preservando o foco tematico da secao onde ele esta inserido.",
            "Preencha obrigatoriamente headline, o_que_ocorreu, fundamentos_juridicos e consequencia_aplicada.",
            "No campo o_que_ocorreu, explique com 2-4 frases o contexto do processo, o que foi discutido, os fundamentos mais relevantes e o desfecho aplicado.",
            "No campo fundamentos_juridicos, resuma a ratio decidendi e as teses normativas/jurisprudenciais invocadas.",
            "No campo consequencia_aplicada, descreva o efeito juridico e pratico concreto da decisao.",
            "No campo impacto_2026, escreva somente quando houver lastro suficiente na propria base; caso contrario, retorne string vazia.",
            "No campo partidos, normalize em sigla/UF quando houver siglaUF do caso e o contexto nao for nacional; quando o proprio caso for nacional, normalize em sigla/Nacional.",
            "Sempre negrite partidos no texto final sera responsabilidade do renderer, mas o campo partidos deve vir limpo e consistente.",
            "Use advogados_relevantes priorizando os nomes estruturados do contexto; nao invente notoriedade externa.",
            "Nao inventar dados; quando faltar evidencia para um campo complementar, retorne string vazia nesse campo.",
            "Nunca escreva placeholders de ausencia de dados.",
            "Considere links_referencia para preservar rastreabilidade da base.",
            "Priorize cidade_uf e processo_cnj quando houver evidencia no caso.",
            "Evite repeticao literal entre o_que_ocorreu, fundamentos_juridicos e consequencia_aplicada.",
            "Usar frases diretas, voz ativa, sem redundancia.",
        ],
        "contexto_casos": [],
        "blocos": blocks_payload,
        "saida_esperada": {
            "items": [
                {
                    "index": 0,
                    "headline": "titulo curto do caso",
                    "o_que_ocorreu": "descricao analitica do caso",
                    "fundamentos_juridicos": "fundamentos centrais",
                    "consequencia_aplicada": "efeitos concretos da decisao",
                    "impacto_2026": "impacto para o pleito de 2026, se houver base",
                    "partes_relevantes": "nomes das partes relevantes",
                    "partidos": "partidos envolvidos",
                    "advogados_relevantes": "advogados relevantes",
                    "cidade_uf": "cidade/UF",
                    "processo_cnj": "numero CNJ",
                }
            ]
        },
    }


def _estimate_openai_input_chars(block_batch: Sequence[Dict[str, Any]], *, compact: bool) -> int:
    total = 0
    for block in block_batch:
        total += _estimate_openai_block_chars(block, compact=compact)
    return total


def _estimate_openai_block_chars(block: Dict[str, Any], *, compact: bool) -> int:
    block_type = str(block.get("type", "") or "")
    if compact:
        max_len = 120 if block_type.startswith("heading_") else 900
    else:
        max_len = 180 if block_type.startswith("heading_") else 1700
    return len(_truncate_text(str(block.get("text", "")), max_len))


def _estimate_openai_max_tokens(total_input_chars: int, *, compact: bool) -> int:
    if compact:
        return min(9000, max(1400, int(total_input_chars / 2.6) + 1200))
    return min(12000, max(2200, int(total_input_chars / 1.6) + 1800))


def _normalize_openai_items_payload(parsed: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(parsed, dict):
        items = parsed.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        if "index" in parsed and "improved_text" in parsed:
            return [parsed]
        return None
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return None


def _block_reference_urls(block: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()

    raw_block = block.get("block")
    if isinstance(raw_block, dict):
        for url in _extract_notion_urls_from_block(raw_block):
            key = url.casefold()
            if key in seen:
                continue
            seen.add(key)
            urls.append(url)

    for raw_page_id in (block.get("source_linked_page_ids") or []):
        try:
            page_id = _normalize_notion_id(str(raw_page_id))
        except Exception:
            continue
        url = _notion_page_url_from_id(page_id)
        key = url.casefold()
        if key in seen:
            continue
        seen.add(key)
        urls.append(url)
    return urls


def _normalize_structured_value(value: Any) -> str:
    text = _normalize_ws(value)
    if not text:
        return ""
    lowered = text.casefold()
    if lowered in {
        "nao identificado na base",
        "não identificado na base",
        "nao identificado",
        "não identificado",
        "n/a",
        "na",
        "none",
        "null",
        "desconhecido",
        "-",
    }:
        return ""
    return text


def _remove_placeholder_lines(text: Any) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    kept: List[str] = []
    for line in lines:
        lowered = _normalize_ws(line).casefold()
        if "nao identificado" in lowered or "não identificado" in lowered:
            continue
        kept.append(line)
    if kept:
        return "\n".join(kept)
    cleaned = re.sub(r"(?i)\bn[aã]o identificado(?: na base)?\b", "", raw)
    normalized = _normalize_ws(cleaned)
    if re.fullmatch(r"[A-Za-zÀ-ÿ0-9()/_\-\s]+:\s*", normalized):
        return ""
    return normalized


def _strip_reference_lines(text: Any) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    cleaned = [
        line
        for line in lines
        if not line.casefold().startswith("referencias notion:")
        and not line.casefold().startswith("referências notion:")
        and not line.casefold().startswith("referência notion:")
        and not line.casefold().startswith("referencia notion:")
        and not line.casefold().startswith("referencias:")
        and not line.casefold().startswith("referências:")
    ]
    return "\n".join(cleaned)


KNOWN_OUTPUT_LABEL_RE = re.compile(
    r"(?i)^(?:-\s*)?(?:\*\*)?(?:"
    r"caso|"
    r"o\s+que\s+ocorreu|"
    r"fundamentos\s+jur[ií]dicos|"
    r"consequ[eê]ncia\s+aplicada|"
    r"impacto\s+para\s+2026|"
    r"partes\s+relevantes|"
    r"partidos\s+envolvidos|"
    r"advogados\s+relevantes|"
    r"cidade/uf|"
    r"processo\s+\(cnj\)|"
    r"refer[eê]ncia(?:s)?\s+(?:notion)?"
    r")(?:\*\*)?\s*:\s*"
)


def _strip_known_output_labels(text: Any) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    cleaned_lines: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- ") and ":" in line:
            line = line.split(":", 1)[1].strip()
        line = KNOWN_OUTPUT_LABEL_RE.sub("", line).strip()
        if line:
            cleaned_lines.append(line)
    return _normalize_ws(" ".join(cleaned_lines))


def _markdown_bold(value: Any) -> str:
    text = _normalize_structured_value(value)
    if not text:
        return ""
    if text.startswith("**") and text.endswith("**") and len(text) >= 4:
        return text
    return f"**{text}**"


def _apply_semantic_bold(text: str, values: Sequence[str]) -> str:
    out = str(text or "")
    tokens: List[str] = []
    seen: set[str] = set()
    for value in values:
        token = _normalize_structured_value(value)
        if not token:
            continue
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    if not tokens:
        return out

    tokens.sort(key=len, reverse=True)
    pieces = re.split(r"(\*\*.+?\*\*)", out, flags=re.DOTALL)
    updated: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) >= 4:
            updated.append(piece)
            continue
        edited = piece
        for token in tokens:
            escaped = re.escape(token)
            if re.match(r"^\w", token) and re.match(r".*\w$", token):
                pattern = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
            else:
                pattern = re.compile(escaped, flags=re.IGNORECASE)
            edited = pattern.sub(lambda m: f"**{m.group(0)}**", edited)
        updated.append(edited)
    return "".join(updated)


def _infer_consequencias(text: str) -> str:
    base = _normalize_ws(text)
    if not base:
        return ""
    sentences = [
        _normalize_ws(piece)
        for piece in re.split(r"(?<=[\.\!\?;])\s+|\n+", base)
        if _normalize_ws(piece)
    ]
    if not sentences:
        return ""

    keywords = (
        "cass",
        "multa",
        "indefer",
        "defer",
        "conden",
        "absolv",
        "improv",
        "provimento",
        "procedente",
        "improcedente",
        "ineleg",
        "registro",
        "determin",
        "suspens",
        "resultado",
        "acord",
        "sentenca",
    )
    picked = [s for s in sentences if any(k in s.casefold() for k in keywords)]
    if not picked:
        return ""
    return _truncate_text(" ".join(picked[:2]), 500)


def _normalize_compare_key(value: Any) -> str:
    base = _normalize_ws(value).casefold()
    if not base:
        return ""
    base = re.sub(r"[^\w\s]", "", base, flags=re.UNICODE)
    return re.sub(r"\s+", " ", base).strip()


def _dedupe_non_redundant_values(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_structured_value(value)
        if not text:
            continue
        key = _normalize_compare_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _ensure_sentence_end(text: Any) -> str:
    out = _normalize_ws(text)
    if not out:
        return ""
    if out.endswith((".", "!", "?", ";", ":")):
        return out
    return f"{out}."


def _infer_impact_2026(text: str) -> str:
    cleaned = _normalize_ws(text)
    if "2026" not in cleaned:
        return ""
    picked = _pick_sentences(cleaned, keywords=["2026"], max_items=2)
    return picked


def _resolve_first_non_empty(*values: Any) -> str:
    for value in values:
        text = _normalize_structured_value(value)
        if text:
            return text
    return ""


def _render_standardized_block_text(
    item: Dict[str, Any],
    reference_urls: Sequence[str],
    *,
    original_text: str = "",
    compact: bool = False,
) -> str:
    cleaned_original = _strip_reference_lines(_remove_placeholder_lines(original_text))
    normalized_original_summary = _strip_known_output_labels(cleaned_original)
    headline = _resolve_first_non_empty(item.get("headline"), item.get("headline_seed"))
    o_que_ocorreu = _resolve_first_non_empty(item.get("o_que_ocorreu"), normalized_original_summary)
    fundamentos = _resolve_first_non_empty(item.get("fundamentos_juridicos"), item.get("fundamentos"))
    consequencia = _resolve_first_non_empty(
        item.get("consequencia_aplicada"),
        item.get("consequencias"),
    )
    impacto_2026 = _resolve_first_non_empty(item.get("impacto_2026"))
    partes = _resolve_first_non_empty(item.get("partes_relevantes"), item.get("partes"))
    partidos = _resolve_first_non_empty(item.get("partidos"))
    advogados = _resolve_first_non_empty(item.get("advogados_relevantes"), item.get("advogados_famosos"))
    processo = _resolve_first_non_empty(item.get("processo_cnj"))
    if not o_que_ocorreu:
        o_que_ocorreu = "Trecho sem informacao suficiente para sintese objetiva."
    if not consequencia:
        consequence_source = ". ".join(
            [
                part
                for part in [
                    item.get("resultado"),
                    fundamentos,
                    item.get("punchline"),
                    o_que_ocorreu,
                    normalized_original_summary,
                ]
                if _normalize_structured_value(part)
            ]
        )
        consequencia = _infer_consequencias(consequence_source)
    if not impacto_2026:
        impacto_2026 = _infer_impact_2026(
            ". ".join(
                part
                for part in [
                    item.get("impacto_2026"),
                    item.get("tema"),
                    item.get("punchline"),
                    item.get("texto_decisao"),
                    cleaned_original,
                ]
                if _normalize_structured_value(part)
            )
        )

    lines: List[str] = []
    if headline and not compact:
        lines.append(f"**Caso:** {headline}")
    lines.append(f"**O que ocorreu:** {_ensure_sentence_end(o_que_ocorreu)}")
    if fundamentos:
        lines.append(f"**Fundamentos jurídicos:** {_ensure_sentence_end(fundamentos)}")
    if consequencia:
        lines.append(f"**Consequência aplicada:** {_ensure_sentence_end(consequencia)}")
    else:
        lines.append(
            "**Consequência aplicada:** Sem desfecho explicito no trecho; recomenda-se monitoramento do andamento processual."
        )
    if impacto_2026:
        lines.append(f"**Impacto para 2026:** {_ensure_sentence_end(impacto_2026)}")
    if partes:
        lines.append(f"**Partes relevantes:** {partes}")
    if partidos:
        lines.append(f"**Partidos envolvidos:** {partidos}")
    if advogados:
        lines.append(f"**Advogados relevantes:** {advogados}")
    if processo:
        lines.append(f"**Processo (CNJ):** {processo}")
    out = "\n".join(lines)
    out = _apply_semantic_bold(out, [partes, partidos, advogados])
    out = _ensure_party_markdown_bold(out)
    return _append_reference_line(out, reference_urls)


def _append_reference_line(text: str, reference_urls: Sequence[str]) -> str:
    base_raw = str(text or "").strip()
    merged_refs: List[str] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        cleaned = _normalize_notion_reference_url(candidate)
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        merged_refs.append(cleaned)

    for url in reference_urls:
        _add(url)
    for url in _extract_notion_urls_from_text(base_raw):
        _add(url)

    lines = [line.strip() for line in base_raw.splitlines() if line.strip()]
    if not lines and base_raw:
        lines = [_normalize_ws(base_raw)]
    base_without_ref = [
        line
        for line in lines
        if not line.casefold().startswith("referencias notion:")
        and not line.casefold().startswith("referências notion:")
        and not line.casefold().startswith("referência notion:")
        and not line.casefold().startswith("referencia notion:")
        and not line.casefold().startswith("referencias:")
        and not line.casefold().startswith("referências:")
    ]
    cleaned_base = "\n".join(base_without_ref).strip()

    if not merged_refs:
        return cleaned_base

    numbered_refs = " ".join(f"[{idx}]({url})" for idx, url in enumerate(merged_refs, start=1))
    ref_line = f"Referência Notion: {numbered_refs}"
    if not cleaned_base:
        return ref_line
    return f"{cleaned_base}\n{ref_line}"


def _render_standardized_fallback_text(
    base_text: str,
    reference_urls: Sequence[str],
    *,
    compact: bool = False,
) -> str:
    base_raw = _strip_reference_lines(_remove_placeholder_lines(base_text)).strip()
    if not base_raw:
        minimal = _render_standardized_block_text(
            {
                "o_que_ocorreu": "Trecho original sem informacao util para sintese objetiva.",
                "consequencia_aplicada": "Necessario monitorar o andamento do processo na pagina de referencia.",
            },
            reference_urls,
            original_text="",
            compact=compact,
        )
        return minimal

    fallback_summary = _strip_known_output_labels(base_raw) or _normalize_ws(base_raw)
    consequence = _infer_consequencias(fallback_summary)
    return _render_standardized_block_text(
        {
            "o_que_ocorreu": _truncate_text(fallback_summary, 900),
            "consequencia_aplicada": consequence,
        },
        reference_urls,
        original_text=base_raw,
        compact=compact,
    )


def improve_text_blocks_with_openai(
    block_batch: List[Dict[str, Any]],
    context: List[CaseContext],
    *,
    batch_label: str = "",
    _allow_micro_fallback: bool = True,
) -> List[str]:
    if not block_batch:
        return []
    if OPENAI_CFG is None:
        return [str(block.get("text", "")) for block in block_batch]

    label = batch_label or f"lote_{len(block_batch)}blocos"
    total_input_chars = _estimate_openai_input_chars(block_batch, compact=False)
    scoped_context_refs = 0
    blocks_with_scoped_context = 0
    for block in block_batch:
        scoped = block.get("context_cases")
        if isinstance(scoped, list) and scoped:
            blocks_with_scoped_context += 1
            scoped_context_refs += len(scoped)
    LOGGER.info(
        "[OpenAI] %s | iniciando geracao para %d bloco(s) | contexto_escopo_refs=%d | blocos_com_contexto=%d | chars=%d.",
        label,
        len(block_batch),
        scoped_context_refs,
        blocks_with_scoped_context,
        total_input_chars,
    )

    is_single_block = len(block_batch) == 1
    with OPENAI_RUNTIME_STATE_LOCK:
        schema_temporarily_disabled = bool(OPENAI_SCHEMA_TEMP_DISABLED)

    # Em instabilidade persistente, evitar requests longas por lote para nao ficar preso em timeout.
    if _allow_micro_fallback and not is_single_block and schema_temporarily_disabled:
        LOGGER.warning(
            "[OpenAI] %s | modo de contingencia ativo; processando em micro-fallback por bloco.",
            label,
        )
        micro_out: List[str] = []
        for idx, block in enumerate(block_batch, start=1):
            micro_label = f"{label} | micro {idx}/{len(block_batch)}"
            micro_res = improve_text_blocks_with_openai(
                [block],
                context[:12],
                batch_label=micro_label,
                _allow_micro_fallback=False,
            )
            micro_out.extend(micro_res)
        return micro_out

    if _allow_micro_fallback and len(block_batch) > 1 and total_input_chars > OPENAI_BATCH_CHAR_BUDGET:
        mid = max(1, len(block_batch) // 2)
        LOGGER.info(
            "[OpenAI] %s | lote acima do limite (%d chars). Dividindo em sublotes (%d + %d).",
            label,
            total_input_chars,
            mid,
            len(block_batch) - mid,
        )
        left = improve_text_blocks_with_openai(
            block_batch[:mid],
            context,
            batch_label=f"{label} | sublote A",
            _allow_micro_fallback=True,
        )
        right = improve_text_blocks_with_openai(
            block_batch[mid:],
            context,
            batch_label=f"{label} | sublote B",
            _allow_micro_fallback=True,
        )
        return left + right

    prompt_payload = _build_openai_prompt_payload(block_batch, context, compact=False)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "headline": {"type": "string"},
                        "o_que_ocorreu": {"type": "string"},
                        "fundamentos_juridicos": {"type": "string"},
                        "consequencia_aplicada": {"type": "string"},
                        "impacto_2026": {"type": "string"},
                        "partes_relevantes": {"type": "string"},
                        "partidos": {"type": "string"},
                        "advogados_relevantes": {"type": "string"},
                        "cidade_uf": {"type": "string"},
                        "processo_cnj": {"type": "string"},
                    },
                    "required": [
                        "index",
                        "headline",
                        "o_que_ocorreu",
                        "fundamentos_juridicos",
                        "consequencia_aplicada",
                        "impacto_2026",
                        "partes_relevantes",
                        "partidos",
                        "advogados_relevantes",
                        "cidade_uf",
                        "processo_cnj",
                    ],
                },
            }
        },
        "required": ["items"],
    }

    messages = [
        {
            "role": "system",
            "content": (
                "Voce e um assistente juridico eleitoral metodico, orientado por dados e precisao. "
                "Responda sempre em JSON valido."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(prompt_payload, ensure_ascii=False),
        },
    ]

    parsed: Optional[Any] = None
    regular_max_tokens = _estimate_openai_max_tokens(total_input_chars, compact=False)
    compact_input_chars = _estimate_openai_input_chars(block_batch, compact=True)
    compact_max_tokens = _estimate_openai_max_tokens(compact_input_chars, compact=True)
    is_large_batch = total_input_chars >= int(OPENAI_BATCH_CHAR_BUDGET * 0.8) or len(block_batch) >= 8
    schema_timeout_s = 45 if not is_large_batch else 35
    compact_timeout_s = 40 if is_single_block else 30
    schema_attempts = 1
    compact_attempts = 2 if is_single_block else 1
    allow_schema_boost = is_single_block or len(block_batch) <= 3
    allow_compact_boost = is_single_block or len(block_batch) <= 3
    schema_finish_reason = ""
    schema_content_chars = 0
    compact_finish_reason = ""
    compact_content_chars = 0
    if schema_temporarily_disabled:
        LOGGER.info(
            "[OpenAI] %s | schema temporariamente despriorizado por instabilidade recente; iniciando fallback compacto.",
            label,
        )
    else:
        try:
            raw = _openai_chat_request(
                messages,
                schema=schema,
                request_label=f"OpenAI {label} (schema)",
                max_completion_tokens=regular_max_tokens,
                timeout_s_override=min(max(5, int(OPENAI_CFG.timeout_s)), schema_timeout_s),
                max_attempts_override=schema_attempts,
            )
            schema_finish_reason, parsed_text = _extract_openai_choice_text(raw)
            schema_content_chars = len(parsed_text)
            parsed = _parse_json_loose(parsed_text)
            if parsed is None:
                LOGGER.warning(
                    "[OpenAI] %s | schema retornou JSON invalido | finish_reason=%s | content_chars=%d",
                    label,
                    schema_finish_reason or "desconhecido",
                    schema_content_chars,
                )
        except Exception as exc:
            LOGGER.warning("[OpenAI] %s | schema falhou: %s. Tentando fallback compacto.", label, exc)

    if parsed is None and allow_schema_boost and schema_finish_reason == "length" and schema_content_chars == 0:
        boosted_tokens = min(16000, max(4200, regular_max_tokens * 2))
        LOGGER.warning(
            "[OpenAI] %s | length sem conteudo no schema. Repetindo schema com max_completion_tokens=%d.",
            label,
            boosted_tokens,
        )
        try:
            raw = _openai_chat_request(
                messages,
                schema=schema,
                request_label=f"OpenAI {label} (schema_boost)",
                max_completion_tokens=boosted_tokens,
                timeout_s_override=min(max(5, int(OPENAI_CFG.timeout_s)), 45),
                max_attempts_override=1,
            )
            schema_finish_reason, parsed_text = _extract_openai_choice_text(raw)
            schema_content_chars = len(parsed_text)
            parsed = _parse_json_loose(parsed_text)
            if parsed is None:
                LOGGER.warning(
                    "[OpenAI] %s | schema_boost ainda invalido | finish_reason=%s | content_chars=%d",
                    label,
                    schema_finish_reason or "desconhecido",
                    schema_content_chars,
                )
        except Exception as exc:
            LOGGER.warning("[OpenAI] %s | schema_boost falhou: %s", label, exc)

    if parsed is None:
        try:
            compact_payload = _build_openai_prompt_payload(block_batch, context, compact=True)
            compact_messages = [
                {
                    "role": "system",
                    "content": (
                        "Voce e um assistente juridico eleitoral metodico, orientado por dados e precisao. "
                        "Responda sempre em JSON valido."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(compact_payload, ensure_ascii=False),
                },
            ]
            raw = _openai_chat_request(
                compact_messages,
                schema=None,
                request_label=f"OpenAI {label} (compacto)",
                max_completion_tokens=compact_max_tokens,
                timeout_s_override=min(max(5, int(OPENAI_CFG.timeout_s)), compact_timeout_s),
                max_attempts_override=compact_attempts,
            )
            compact_finish_reason, parsed_text = _extract_openai_choice_text(raw)
            compact_content_chars = len(parsed_text)
            parsed = _parse_json_loose(parsed_text)
            if parsed is None:
                LOGGER.warning(
                    "[OpenAI] %s | fallback compacto retornou JSON invalido | finish_reason=%s | content_chars=%d",
                    label,
                    compact_finish_reason or "desconhecido",
                    compact_content_chars,
                )
        except Exception as exc:
            LOGGER.error("[OpenAI] %s | fallback compacto falhou: %s", label, exc)

    if parsed is None and allow_compact_boost and compact_finish_reason == "length" and compact_content_chars == 0:
        boosted_tokens = min(16000, max(3200, compact_max_tokens * 2))
        LOGGER.warning(
            "[OpenAI] %s | length sem conteudo no compacto. Repetindo compacto com tokens=%d e contexto reduzido.",
            label,
            boosted_tokens,
        )
        try:
            compact_payload_retry = _build_openai_prompt_payload(block_batch, context[:8], compact=True)
            compact_messages_retry = [
                {
                    "role": "system",
                    "content": (
                        "Voce e um assistente juridico eleitoral metodico, orientado por dados e precisao. "
                        "Responda sempre em JSON valido."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(compact_payload_retry, ensure_ascii=False),
                },
            ]
            raw = _openai_chat_request(
                compact_messages_retry,
                schema=None,
                request_label=f"OpenAI {label} (compacto_boost)",
                max_completion_tokens=boosted_tokens,
                timeout_s_override=min(max(5, int(OPENAI_CFG.timeout_s)), 35),
                max_attempts_override=1,
            )
            compact_finish_reason, parsed_text = _extract_openai_choice_text(raw)
            compact_content_chars = len(parsed_text)
            parsed = _parse_json_loose(parsed_text)
            if parsed is None:
                LOGGER.warning(
                    "[OpenAI] %s | compacto_boost ainda invalido | finish_reason=%s | content_chars=%d",
                    label,
                    compact_finish_reason or "desconhecido",
                    compact_content_chars,
                )
        except Exception as exc:
            LOGGER.warning("[OpenAI] %s | compacto_boost falhou: %s", label, exc)

    if parsed is None and _allow_micro_fallback and len(block_batch) > 1:
        LOGGER.warning(
            "[OpenAI] %s | iniciando micro-fallback por bloco para reduzir timeout.",
            label,
        )
        micro_out: List[str] = []
        for idx, block in enumerate(block_batch, start=1):
            micro_label = f"{label} | micro {idx}/{len(block_batch)}"
            micro_res = improve_text_blocks_with_openai(
                [block],
                context[:12],
                batch_label=micro_label,
                _allow_micro_fallback=False,
            )
            micro_out.extend(micro_res)
        return micro_out

    if parsed is None:
        LOGGER.error("[OpenAI] %s | sem retorno valido apos retries/fallback.", label)
        fallback_out: List[str] = []
        for block in block_batch:
            original = _normalize_ws(block.get("text", ""))
            refs = _block_reference_urls(block)
            fallback_out.append(_render_standardized_fallback_text(original, refs))
        return fallback_out

    items = _normalize_openai_items_payload(parsed)
    mapped: Dict[int, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            idx_int: Optional[int] = None
            if isinstance(idx, int):
                idx_int = idx
            elif isinstance(idx, str) and idx.strip().isdigit():
                idx_int = int(idx.strip())
            if idx_int is not None:
                mapped[idx_int] = dict(item)
    else:
        LOGGER.warning("[OpenAI] %s | JSON sem lista 'items'; usando textos originais.", label)

    out: List[str] = []
    for idx, block in enumerate(block_batch):
        original = _normalize_ws(block.get("text", ""))
        refs = _block_reference_urls(block)
        item = mapped.get(idx, {})
        merged_item = dict(item)
        scoped_contexts_raw = block.get("context_cases")
        primary_ctx: Optional[CaseContext] = None
        if isinstance(scoped_contexts_raw, list):
            for ctx in scoped_contexts_raw:
                if isinstance(ctx, CaseContext):
                    primary_ctx = ctx
                    break
        if primary_ctx is not None:
            fallback_fields = {
                "headline_seed": primary_ctx.tema,
                "partes_relevantes": primary_ctx.partes,
                "partidos": primary_ctx.partidos,
                "advogados_relevantes": primary_ctx.advogados,
                "cidade_uf": primary_ctx.cidade_uf,
                "processo_cnj": primary_ctx.processo_cnj or primary_ctx.numero_unico,
                "fundamentos_juridicos": primary_ctx.fundamentos or primary_ctx.punchline,
                "consequencia_aplicada": primary_ctx.resultado,
                "tema": primary_ctx.tema,
                "punchline": primary_ctx.punchline,
                "texto_decisao": primary_ctx.texto_decisao,
            }
            for key, value in fallback_fields.items():
                if not _normalize_structured_value(merged_item.get(key, "")):
                    merged_item[key] = value
        improved = ""
        if merged_item:
            has_structured = any(
                _normalize_structured_value(merged_item.get(key, "")) for key in (
                    "headline",
                    "o_que_ocorreu",
                    "fundamentos_juridicos",
                    "consequencia_aplicada",
                    "impacto_2026",
                    "partes_relevantes",
                    "partidos",
                    "advogados_relevantes",
                    "cidade_uf",
                    "processo_cnj",
                )
            )
            if has_structured:
                improved = _render_standardized_block_text(
                    merged_item,
                    refs,
                    original_text=original,
                    compact=bool(block.get("render_compact", False)),
                )
            else:
                free_text = _normalize_ws(merged_item.get("improved_text", ""))
                if free_text:
                    improved = _render_standardized_fallback_text(
                        _ensure_party_markdown_bold(free_text),
                        refs,
                        compact=bool(block.get("render_compact", False)),
                    )

        if not improved:
            improved = _render_standardized_fallback_text(
                original,
                refs,
                compact=bool(block.get("render_compact", False)),
            )
        out.append(improved)
    LOGGER.info(
        "[OpenAI] %s | concluido | blocos=%d | melhorados=%d",
        label,
        len(block_batch),
        sum(1 for idx, block in enumerate(block_batch) if _normalize_ws(out[idx]) != _normalize_ws(block.get("text", ""))),
    )
    return out


def _chunk_text_for_notion(text: str, max_chars: int = MAX_NOTION_RICH_TEXT_CHARS) -> List[str]:
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        cut = remaining.rfind(" ", 0, max_chars)
        if cut < int(max_chars * 0.6):
            cut = max_chars
        chunk = remaining[:cut]
        chunks.append(chunk)
        remaining = remaining[cut:].lstrip()
    if not chunks:
        return [""]
    return chunks


def _make_rich_text_item(
    content: str,
    *,
    bold: bool = False,
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
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": False,
            "color": "default",
        },
    }


def markdown_bold_to_notion_rich_text(text: str) -> List[Dict[str, Any]]:
    source = str(text or "")
    if not source:
        return []

    rich: List[Dict[str, Any]] = []

    def _append_plain_segment(segment: str, *, bold: bool) -> None:
        if not segment:
            return
        cursor_inner = 0
        for link_match in re.finditer(r"\[([^\]\n]+)\]\((https?://[^\s)>\"]+)\)|(https?://[^\s)>\"]+)", segment):
            start, end = link_match.span()
            if start > cursor_inner:
                plain = segment[cursor_inner:start]
                for piece in _chunk_text_for_notion(plain):
                    if piece:
                        rich.append(_make_rich_text_item(piece, bold=bold))

            label = ""
            url = ""
            if link_match.group(1) is not None and link_match.group(2) is not None:
                label = link_match.group(1)
                url = link_match.group(2)
            elif link_match.group(3) is not None:
                label = link_match.group(3)
                url = link_match.group(3)

            if label and url:
                display_label = f"[{label}]" if re.fullmatch(r"\d+", label) else label
                for piece in _chunk_text_for_notion(display_label):
                    if piece:
                        rich.append(_make_rich_text_item(piece, bold=bold, link_url=url))
            cursor_inner = end

        if cursor_inner < len(segment):
            plain_tail = segment[cursor_inner:]
            for piece in _chunk_text_for_notion(plain_tail):
                if piece:
                    rich.append(_make_rich_text_item(piece, bold=bold))

    cursor = 0
    for match in re.finditer(r"\*\*(.+?)\*\*", source, flags=re.DOTALL):
        start, end = match.span()
        if start > cursor:
            _append_plain_segment(source[cursor:start], bold=False)
        bold_text = match.group(1)
        if bold_text:
            _append_plain_segment(bold_text, bold=True)
        cursor = end
    if cursor < len(source):
        _append_plain_segment(source[cursor:], bold=False)
    if not rich:
        _append_plain_segment(source, bold=False)
    return rich


def _build_patch_payload(block: Dict[str, Any], block_type: str, rich_text: List[Dict[str, Any]]) -> Dict[str, Any]:
    body: Dict[str, Any] = {"rich_text": rich_text}
    original = block.get(block_type)
    if not isinstance(original, dict):
        original = {}

    if block_type in {"paragraph", "bulleted_list_item", "numbered_list_item", "quote", "toggle"}:
        body["color"] = original.get("color", "default")
    elif block_type in {"heading_1", "heading_2", "heading_3"}:
        body["color"] = original.get("color", "default")
        if "is_toggleable" in original:
            body["is_toggleable"] = bool(original.get("is_toggleable"))
    elif block_type == "to_do":
        body["checked"] = bool(original.get("checked", False))
        body["color"] = original.get("color", "default")
    elif block_type == "callout":
        body["color"] = original.get("color", "default")
        if "icon" in original:
            body["icon"] = original.get("icon")

    return {block_type: body}


def patch_text_block(block_id: str, block_type: str, rich_text: List[Dict[str, Any]]) -> None:
    block = BLOCK_INDEX.get(block_id)
    if block is None:
        raise RuntimeError(f"Bloco nao encontrado para patch: {block_id}")
    payload = _build_patch_payload(block, block_type, rich_text)
    notion_request("PATCH", f"/v1/blocks/{block_id}", json_body=payload)


def patch_table_row_block(block_id: str, cells: Sequence[Sequence[Dict[str, Any]]]) -> None:
    block = BLOCK_INDEX.get(block_id)
    if block is None:
        raise RuntimeError(f"Linha de tabela nao encontrada para patch: {block_id}")
    payload = {"table_row": {"cells": [list(cell) for cell in cells]}}
    notion_request("PATCH", f"/v1/blocks/{block_id}", json_body=payload)


def delete_block(block_id: str) -> None:
    notion_request("DELETE", f"/v1/blocks/{block_id}")


def append_block_children(
    parent_id: str,
    children: Sequence[Dict[str, Any]],
    *,
    after_block_id: str = "",
) -> List[Dict[str, Any]]:
    created: List[Dict[str, Any]] = []
    if not children:
        return created

    for start in range(0, len(children), MAX_NOTION_APPEND_CHILDREN):
        chunk = list(children[start : start + MAX_NOTION_APPEND_CHILDREN])
        body: Dict[str, Any] = {"children": chunk}
        if after_block_id:
            body["position"] = {
                "type": "after_block",
                "after_block": {"id": after_block_id},
            }
        payload = notion_request("PATCH", f"/v1/blocks/{parent_id}/children", json_body=body)
        results = payload.get("results", [])
        if isinstance(results, list):
            created.extend([item for item in results if isinstance(item, dict)])
        after_block_id = ""
    return created


def _normalize_text_block_notion_references(
    block: Dict[str, Any],
    reference_numbers: Dict[str, int],
) -> bool:
    block_id = _normalize_ws(block.get("id", ""))
    block_type = str(block.get("type", "") or "")
    if not block_id or block_type not in SUPPORTED_TEXT_BLOCK_TYPES:
        return False
    current_items = _block_rich_text(block)
    if not current_items:
        return False
    rendered_markdown, touched = _render_rich_text_items_with_global_refs(current_items, reference_numbers)
    if not touched:
        return False
    new_rich_text = markdown_bold_to_notion_rich_text(rendered_markdown)
    current_plain = _normalize_ws(_rich_text_plain_text(current_items))
    new_plain = _normalize_ws(_rich_text_plain_text(new_rich_text))
    if current_plain == new_plain:
        return False
    patch_text_block(block_id, block_type, new_rich_text)
    return True


def _normalize_table_row_notion_references(
    block: Dict[str, Any],
    reference_numbers: Dict[str, int],
) -> bool:
    if str(block.get("type", "") or "") != "table_row":
        return False
    block_id = _normalize_ws(block.get("id", ""))
    if not block_id:
        return False
    body = block.get("table_row")
    if not isinstance(body, dict):
        return False
    cells = body.get("cells")
    if not isinstance(cells, list):
        return False

    any_touched = False
    any_changed = False
    new_cells: List[List[Dict[str, Any]]] = []
    for cell in cells:
        cell_items = [item for item in cell if isinstance(item, dict)] if isinstance(cell, list) else []
        rendered_markdown, touched = _render_rich_text_items_with_global_refs(cell_items, reference_numbers)
        if not touched:
            new_cells.append(cell_items)
            continue
        any_touched = True
        new_rich_text = markdown_bold_to_notion_rich_text(rendered_markdown)
        current_plain = _normalize_ws(_rich_text_plain_text(cell_items))
        new_plain = _normalize_ws(_rich_text_plain_text(new_rich_text))
        if current_plain != new_plain:
            any_changed = True
        new_cells.append(new_rich_text)

    if not any_touched or not any_changed:
        return False
    patch_table_row_block(block_id, new_cells)
    return True


def _convert_link_to_page_block_to_reference(
    block: Dict[str, Any],
    reference_numbers: Dict[str, int],
) -> bool:
    if str(block.get("type", "") or "") != "link_to_page":
        return False
    block_id = _normalize_ws(block.get("id", ""))
    parent_id = _normalize_ws(block.get("_parent_id", ""))
    if not block_id or not parent_id:
        return False
    obj = block.get("link_to_page")
    if not isinstance(obj, dict):
        return False
    page_id = obj.get("page_id")
    if not isinstance(page_id, str) or not page_id:
        return False
    try:
        notion_url = _notion_page_url_from_id(_normalize_notion_id(page_id))
    except Exception:
        return False
    idx = reference_numbers.get(notion_url.casefold())
    if not idx:
        return False

    created = append_block_children(
        parent_id,
        [_build_paragraph_block(f"[{idx}]({notion_url})")],
        after_block_id=block_id,
    )
    if not created:
        raise RuntimeError(f"Falha ao converter link_to_page {block_id}")
    delete_block(block_id)
    return True


def normalize_global_notion_references(page_id: str) -> Dict[str, Any]:
    page_blocks = retrieve_all_block_children_recursive(page_id)
    global BLOCK_INDEX
    BLOCK_INDEX = {str(block.get("id", "")): block for block in page_blocks if block.get("id")}

    reference_numbers = _build_global_notion_reference_numbers(page_blocks)
    stats: Dict[str, Any] = {
        "reference_count": len(reference_numbers),
        "text_blocks_normalized": 0,
        "table_rows_normalized": 0,
        "link_to_page_converted": 0,
        "failures": [],
    }
    if not reference_numbers:
        return stats

    for block in page_blocks:
        block_id = _normalize_ws(block.get("id", ""))
        block_type = str(block.get("type", "") or "")
        try:
            if block_type in SUPPORTED_TEXT_BLOCK_TYPES:
                if _normalize_text_block_notion_references(block, reference_numbers):
                    stats["text_blocks_normalized"] += 1
            elif block_type == "table_row":
                if _normalize_table_row_notion_references(block, reference_numbers):
                    stats["table_rows_normalized"] += 1
            elif block_type == "link_to_page":
                if _convert_link_to_page_block_to_reference(block, reference_numbers):
                    stats["link_to_page_converted"] += 1
        except Exception as exc:
            stats["failures"].append(
                {
                    "block_id": block_id,
                    "block_type": block_type,
                    "erro": str(exc),
                }
            )
            LOGGER.error(
                "[Refs-globais] Falha ao normalizar bloco %s (%s): %s",
                block_id,
                block_type,
                exc,
            )

    return stats


def _build_paragraph_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": markdown_bold_to_notion_rich_text(text),
            "color": "default",
        },
    }


def _build_bulleted_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": markdown_bold_to_notion_rich_text(text),
            "color": "default",
        },
    }


def _build_heading_2_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": markdown_bold_to_notion_rich_text(text),
            "color": "default",
        },
    }


def _build_toggle_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": markdown_bold_to_notion_rich_text(text),
            "color": "gray_background",
        },
    }


def _load_last_page_url() -> str:
    payload = read_json_dict(REPORT_FILE)
    return _normalize_ws(payload.get("page_url", ""))


def _run_gui_for_page_url(initial_value: str = "") -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception:
        LOGGER.warning("Tkinter indisponivel. GUI nao pode ser aberta neste ambiente.")
        return None

    result = {"url": ""}

    try:
        root = tk.Tk()
    except Exception as exc:
        LOGGER.warning("Falha ao abrir GUI Tkinter: %s", exc)
        return None

    root.title("Agente IA - Notion Relatorio")
    root.geometry("840x220")
    root.minsize(760, 200)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Cole o link da pagina do Notion que sera processada pela API:",
    ).pack(anchor="w")
    ttk.Label(
        frame,
        text=(
            "A pagina informada abaixo sera a pagina lida e atualizada. "
            "Voce pode colar qualquer URL de pagina do seu workspace com acesso da integracao."
        ),
        wraplength=790,
        justify="left",
    ).pack(anchor="w", pady=(4, 0))

    url_var = tk.StringVar(value=initial_value)
    entry = ttk.Entry(frame, textvariable=url_var)
    entry.pack(fill="x", pady=(10, 12))
    entry.focus_set()
    entry.icursor("end")

    row = ttk.Frame(frame)
    row.pack(fill="x")

    def _on_run() -> None:
        value = _normalize_ws(url_var.get())
        if not value:
            messagebox.showwarning("URL obrigatoria", "Informe o link da pagina do Notion.")
            return
        try:
            extract_notion_id_from_url(value)
        except Exception as exc:
            messagebox.showerror("URL invalida", str(exc))
            return
        result["url"] = value
        root.destroy()

    def _on_cancel() -> None:
        result["url"] = ""
        root.destroy()

    root.bind("<Return>", lambda _event: _on_run())
    root.bind("<Escape>", lambda _event: _on_cancel())

    ttk.Button(row, text="Cancelar", command=_on_cancel).pack(side="right")
    ttk.Button(row, text="Executar", command=_on_run).pack(side="right", padx=(0, 8))

    root.mainloop()
    return result["url"]


def _extract_linked_page_ids(links: Sequence[str], exclude_page_id: str = "") -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for link in links:
        try:
            page_id = extract_notion_id_from_url(link)
        except Exception:
            continue
        if page_id == exclude_page_id:
            continue
        if page_id in seen:
            continue
        seen.add(page_id)
        out.append(page_id)
    return out


def _attach_scoped_contexts_to_blocks(
    target_blocks: Sequence[Dict[str, Any]],
    *,
    context_by_page_id: Dict[str, CaseContext],
    per_block_limit: int = 3,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    blocks_with_context = 0
    blocks_without_context = 0
    total_context_refs = 0
    for block in target_blocks:
        item = dict(block)
        contexts: List[CaseContext] = []
        for raw_page_id in (block.get("source_linked_page_ids") or []):
            try:
                page_id = _normalize_notion_id(str(raw_page_id))
            except Exception:
                continue
            ctx = context_by_page_id.get(page_id)
            if ctx is None:
                continue
            contexts.append(ctx)
        scoped = _dedupe_contexts(contexts, limit=max(1, int(per_block_limit)))
        item["context_cases"] = scoped
        total_context_refs += len(scoped)
        if scoped:
            blocks_with_context += 1
        else:
            blocks_without_context += 1
        out.append(item)

    stats = {
        "blocks_total": len(target_blocks),
        "blocks_with_context": blocks_with_context,
        "blocks_without_context": blocks_without_context,
        "total_context_refs": total_context_refs,
    }
    return out, stats


def _build_table_companion_items(
    targets: Sequence[TableCompanionTarget],
    *,
    context_by_page_id: Dict[str, CaseContext],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    items: List[Dict[str, Any]] = []
    tables_with_context = 0
    tables_without_context = 0
    cases_with_context = 0
    cases_without_context = 0

    for target in targets:
        local_count = 0
        for page_id in target.source_linked_page_ids:
            ctx = context_by_page_id.get(page_id)
            if ctx is None:
                cases_without_context += 1
                continue
            local_count += 1
            cases_with_context += 1
            row_text = _normalize_ws(target.row_text_by_page_id.get(page_id, ""))
            seed_text = row_text or _normalize_ws(ctx.tema) or _normalize_ws(ctx.punchline)
            items.append(
                {
                    "id": f"{target.table_id}:{page_id}",
                    "table_id": target.table_id,
                    "table_marker": target.marker,
                    "parent_id": target.parent_id,
                    "type": "table_companion",
                    "item_kind": "table_companion",
                    "text": seed_text,
                    "headline_seed": _normalize_ws(ctx.tema) or row_text,
                    "source_linked_page_ids": [page_id],
                    "context_cases": [ctx],
                    "render_compact": False,
                }
            )
        if local_count > 0:
            tables_with_context += 1
        else:
            tables_without_context += 1

    stats = {
        "tables_total": len(targets),
        "tables_with_context": tables_with_context,
        "tables_without_context": tables_without_context,
        "cases_with_context": cases_with_context,
        "cases_without_context": cases_without_context,
    }
    return items, stats


def _split_structured_names(value: Any) -> List[str]:
    raw = str(value or "")
    if not raw:
        return []
    parts = re.split(r"\s*[;\n]\s*|\s*,\s*", raw)
    out: List[str] = []
    seen: set[str] = set()
    for part in parts:
        text = _normalize_ws(part)
        if not text:
            continue
        text = re.sub(r"\s*-\s*OAB.*$", "", text, flags=re.IGNORECASE)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _compute_lawyer_frequency_lines(
    used_page_ids: Sequence[str],
    *,
    context_by_page_id: Dict[str, CaseContext],
) -> List[str]:
    freq: Counter[str] = Counter()
    for raw_page_id in used_page_ids:
        try:
            page_id = _normalize_notion_id(str(raw_page_id))
        except Exception:
            continue
        ctx = context_by_page_id.get(page_id)
        if ctx is None:
            continue
        for name in _split_structured_names(ctx.advogados):
            freq[name] += 1

    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0].casefold()))
    filtered = [(name, count) for name, count in ranked if count >= LAWYER_FREQ_MIN_CASES]
    if not filtered:
        filtered = ranked[:LAWYER_FREQ_MAX_ITEMS]
    else:
        filtered = filtered[:LAWYER_FREQ_MAX_ITEMS]

    out: List[str] = []
    for name, count in filtered:
        suffix = "caso" if count == 1 else "casos"
        out.append(f"{name} — {count} {suffix}")
    return out


def _log_block_link_audit(target_blocks: Sequence[Dict[str, Any]]) -> None:
    total = len(target_blocks)
    LOGGER.info("[Auditoria-link] Inicio | blocos_alvo=%d", total)
    for idx, block in enumerate(target_blocks, start=1):
        block_id = _normalize_ws(block.get("id", ""))
        block_type = _normalize_ws(block.get("type", ""))

        source_page_ids: List[str] = []
        seen_page_ids: set[str] = set()
        for raw_page_id in (block.get("source_linked_page_ids") or []):
            try:
                page_id = _normalize_notion_id(str(raw_page_id))
            except Exception:
                continue
            if page_id in seen_page_ids:
                continue
            seen_page_ids.add(page_id)
            source_page_ids.append(page_id)

        context_cases_raw = block.get("context_cases")
        context_urls: List[str] = []
        seen_urls: set[str] = set()
        if isinstance(context_cases_raw, list):
            for ctx in context_cases_raw:
                if not isinstance(ctx, CaseContext):
                    continue
                url = _normalize_ws(ctx.fonte_url)
                if not url:
                    continue
                key = url.casefold()
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                context_urls.append(url)

        source_pages_text = ", ".join(source_page_ids) if source_page_ids else "-"
        context_urls_text = " | ".join(context_urls) if context_urls else "-"
        LOGGER.info(
            "[Auditoria-link] bloco %d/%d | id=%s | tipo=%s | source_page_ids=%s | context_count=%d | context_urls=%s",
            idx,
            total,
            block_id or "-",
            block_type or "-",
            source_pages_text,
            len(context_urls),
            context_urls_text,
        )
    LOGGER.info("[Auditoria-link] Fim | blocos_auditados=%d", total)


def _merge_contexts(primary: Sequence[CaseContext], fallback: Sequence[CaseContext]) -> List[CaseContext]:
    return _dedupe_contexts(list(primary) + list(fallback), limit=MAX_CONTEXT_CASES)


def _chunk_items(items: Sequence[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    size = max(1, int(chunk_size))
    out: List[List[Dict[str, Any]]] = []
    for start in range(0, len(items), size):
        out.append(list(items[start : start + size]))
    return out


def _chunk_items_for_openai(
    items: Sequence[Dict[str, Any]],
    *,
    max_items: int,
    char_budget: int,
) -> List[List[Dict[str, Any]]]:
    item_limit = max(1, int(max_items))
    budget = max(1200, int(char_budget))
    out: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0

    for item in items:
        item_chars = _estimate_openai_block_chars(item, compact=False)
        would_exceed_items = len(current) >= item_limit
        would_exceed_chars = (current_chars + item_chars) > budget if current else False

        if current and (would_exceed_items or would_exceed_chars):
            out.append(current)
            current = []
            current_chars = 0

        current.append(item)
        current_chars += item_chars

    if current:
        out.append(current)
    return out


def _build_table_companion_plans(
    targets: Sequence[TableCompanionTarget],
    item_texts_by_id: Dict[str, str],
) -> List[TableCompanionPlan]:
    plans: List[TableCompanionPlan] = []
    for target in targets:
        process_texts: List[str] = []
        for page_id in target.source_linked_page_ids:
            item_id = f"{target.table_id}:{page_id}"
            text = _normalize_multiline_ws(item_texts_by_id.get(item_id, ""))
            if text:
                process_texts.append(text)
        if not process_texts:
            continue
        plans.append(
            TableCompanionPlan(
                table_id=target.table_id,
                parent_id=target.parent_id,
                marker=target.marker,
                title=f"{target.marker} Casos detalhados por processo",
                process_texts=process_texts,
            )
        )
    return plans


def _apply_table_companion_plan(plan: TableCompanionPlan) -> str:
    created = append_block_children(
        plan.parent_id,
        [_build_toggle_block(plan.title)],
        after_block_id=plan.table_id,
    )
    if not created:
        raise RuntimeError(f"Falha ao criar companion da tabela {plan.table_id}")
    root_id = _normalize_ws(created[0].get("id", ""))
    if not root_id:
        raise RuntimeError(f"Resposta sem ID ao criar companion da tabela {plan.table_id}")

    children = [_build_paragraph_block(text) for text in plan.process_texts]
    append_block_children(root_id, children)
    return root_id


def _apply_lawyer_section(page_id: str, lines: Sequence[str]) -> List[str]:
    blocks: List[Dict[str, Any]] = [
        _build_heading_2_block(f"13. Advogados recorrentes nos casos detalhados {AUTO_LAWYERS_MARKER}"),
    ]
    if lines:
        blocks.append(_build_toggle_block(f"{AUTO_LAWYERS_MARKER} Lista por frequencia"))
    else:
        blocks.append(_build_paragraph_block(f"{AUTO_LAWYERS_MARKER} Nenhum advogado recorrente foi identificado."))

    created = append_block_children(page_id, blocks)
    created_ids = [_normalize_ws(item.get("id", "")) for item in created if _normalize_ws(item.get("id", ""))]
    if lines and len(created_ids) >= 2:
        append_block_children(created_ids[1], [_build_bulleted_block(line) for line in lines])
    return created_ids


def _delete_auto_generated_roots(auto_root_ids: Sequence[str]) -> int:
    deleted = 0
    for block_id in auto_root_ids:
        try:
            delete_block(block_id)
            deleted += 1
        except Exception as exc:
            LOGGER.warning("Falha ao remover bloco auto-gerado %s: %s", block_id, exc)
    return deleted


def _process_items_via_openai(
    items: Sequence[Dict[str, Any]],
    *,
    config: RunConfig,
    final_context: Sequence[CaseContext],
    batch_prefix: str,
) -> tuple[Dict[str, str], Dict[str, int]]:
    outputs_by_id: Dict[str, str] = {}
    effective_batch_size = max(1, int(config.openai_batch_size or MAX_BLOCK_BATCH))
    batches = _chunk_items_for_openai(
        items,
        max_items=effective_batch_size,
        char_budget=OPENAI_BATCH_CHAR_BUDGET,
    )
    total_batches = len(batches)
    max_chars_batch = max((_estimate_openai_input_chars(batch, compact=False) for batch in batches), default=0)
    LOGGER.info(
        "[OpenAI] %s | lotes preparados | total=%d | max_chars_lote=%d | limite_chars=%d | max_itens_lote=%d",
        batch_prefix,
        total_batches,
        max_chars_batch,
        OPENAI_BATCH_CHAR_BUDGET,
        effective_batch_size,
    )
    openai_workers = min(max(1, int(config.openai_max_workers or 1)), max(1, total_batches))
    if max_chars_batch >= int(OPENAI_BATCH_CHAR_BUDGET * 0.9) and openai_workers > 1:
        LOGGER.info(
            "[OpenAI] %s | ajuste de estabilidade: lotes pesados (max_chars=%d). Limitando workers de %d para 1.",
            batch_prefix,
            max_chars_batch,
            openai_workers,
        )
        openai_workers = 1
    if total_batches >= 4 and openai_workers > 2:
        LOGGER.info(
            "[OpenAI] %s | ajuste de estabilidade: limitando workers de %d para 2 (lotes=%d).",
            batch_prefix,
            openai_workers,
            total_batches,
        )
        openai_workers = 2

    improved_batches: Dict[int, List[str]] = {}
    if total_batches > 0 and openai_workers <= 1:
        LOGGER.info("[OpenAI] %s | execucao sequencial de %d lote(s).", batch_prefix, total_batches)
        for batch_idx, batch in enumerate(batches, start=1):
            batch_label = f"{batch_prefix} | lote {batch_idx}/{total_batches}"
            LOGGER.info("[OpenAI] %s | itens_no_lote=%d", batch_label, len(batch))
            improved_batches[batch_idx - 1] = improve_text_blocks_with_openai(
                list(batch),
                list(final_context),
                batch_label=batch_label,
            )
    elif total_batches > 0:
        LOGGER.info(
            "[OpenAI] %s | execucao paralela de %d lote(s) com %d worker(s).",
            batch_prefix,
            total_batches,
            openai_workers,
        )

        def _batch_job(payload: tuple[int, List[Dict[str, Any]]]) -> tuple[int, List[str]]:
            zero_idx, batch_data = payload
            one_idx = zero_idx + 1
            batch_label = f"{batch_prefix} | lote {one_idx}/{total_batches}"
            LOGGER.info("[OpenAI] %s | iniciado em worker | itens=%d", batch_label, len(batch_data))
            improved_local = improve_text_blocks_with_openai(
                batch_data,
                list(final_context),
                batch_label=batch_label,
            )
            return zero_idx, improved_local

        with ThreadPoolExecutor(max_workers=openai_workers) as executor:
            future_map = {
                executor.submit(_batch_job, (idx, list(batch))): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    done_idx, improved = future.result()
                    improved_batches[done_idx] = improved
                    LOGGER.info("[OpenAI] %s | lote %d/%d concluido.", batch_prefix, done_idx + 1, total_batches)
                except Exception as exc:
                    LOGGER.error(
                        "[OpenAI] %s | lote %d/%d falhou no worker: %s",
                        batch_prefix,
                        idx + 1,
                        total_batches,
                        exc,
                    )
                    improved_batches[idx] = [str(block.get("text", "")) for block in batches[idx]]

    for batch_idx, batch in enumerate(batches):
        improved = improved_batches.get(batch_idx, [str(block.get("text", "")) for block in batch])
        for item, new_text in zip(batch, improved):
            item_id = _normalize_ws(item.get("id", ""))
            if not item_id:
                continue
            outputs_by_id[item_id] = _normalize_multiline_ws(new_text)

    stats = {
        "items_total": len(items),
        "batches_total": total_batches,
        "max_chars_batch": max_chars_batch,
    }
    return outputs_by_id, stats


def _build_clients(config: RunConfig) -> None:
    global NOTION_CFG, OPENAI_CFG, NOTION_SESSION, OPENAI_SESSION, NOTION_PACER, OPENAI_PACER
    global OPENAI_DISABLE_REASONING_PARAMS, OPENAI_SCHEMA_TEMP_DISABLED, OPENAI_TIMEOUT_STREAK

    notion_key = resolve_notion_key()
    if not notion_key:
        raise RuntimeError(
            "Chave do Notion ausente. Crie o arquivo Chave_Notion.txt "
            "ou defina NOTION_API_KEY/NOTION_TOKEN."
        )
    openai_key = resolve_openai_key()
    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY ausente. Defina no .env ou em variavel de ambiente."
        )

    NOTION_CFG = NotionConfig(
        token=notion_key,
        timeout_s=max(5, int(config.notion_timeout_s)),
        max_retries=max(1, int(config.notion_retries)),
        min_interval_s=max(0.0, float(config.notion_min_interval_s)),
    )
    OPENAI_CFG = OpenAIConfig(
        api_key=openai_key,
        timeout_s=max(10, int(config.openai_timeout_s)),
        retries=max(1, int(config.openai_retries)),
        target_rpm=max(0, int(config.openai_target_rpm)),
    )

    NOTION_SESSION = requests.Session()
    NOTION_SESSION.headers.update(
        {
            "Authorization": f"Bearer {NOTION_CFG.token}",
            "Notion-Version": NOTION_CFG.notion_version,
            "Content-Type": "application/json",
        }
    )
    OPENAI_SESSION = requests.Session()
    OPENAI_SESSION.headers.update(
        {
            "Authorization": f"Bearer {OPENAI_CFG.api_key}",
            "Content-Type": "application/json",
        }
    )

    NOTION_PACER = RequestPacer(min_interval_s=NOTION_CFG.min_interval_s)
    OPENAI_PACER = RequestPacer(target_rpm=OPENAI_CFG.target_rpm)
    with OPENAI_RUNTIME_STATE_LOCK:
        OPENAI_DISABLE_REASONING_PARAMS = True
        OPENAI_SCHEMA_TEMP_DISABLED = False
        OPENAI_TIMEOUT_STREAK = 0
    LOGGER.info(
        "Perf config: openai_batch=%d | openai_workers=%d | openai_rpm=%d | openai_timeout=%ss | notion_interval=%.2fs",
        int(config.openai_batch_size),
        int(config.openai_max_workers),
        int(OPENAI_CFG.target_rpm),
        int(OPENAI_CFG.timeout_s),
        float(NOTION_CFG.min_interval_s),
    )


def _close_clients() -> None:
    global NOTION_SESSION, OPENAI_SESSION
    if NOTION_SESSION is not None:
        NOTION_SESSION.close()
    if OPENAI_SESSION is not None:
        OPENAI_SESSION.close()
    NOTION_SESSION = None
    OPENAI_SESSION = None


def run_agent(config: RunConfig) -> int:
    started_at = time.time()
    report: Dict[str, Any] = {
        "script": SCRIPT_FILE,
        "started_at": utc_now_iso(),
        "status": "running",
        "page_url": config.page_url,
        "source_database_url": config.source_database_url,
        "stats": {},
    }

    try:
        LOGGER.info("[Stage 1/6] Preparando IDs e validando acesso...")
        page_id = extract_notion_id_from_url(config.page_url)
        source_database_id = extract_notion_id_from_url(config.source_database_url)
        report["page_id"] = page_id
        report["source_database_id"] = source_database_id

        notion_request("GET", f"/v1/pages/{page_id}")
        data_source_id = retrieve_database_and_datasource_id(source_database_id)
        report["data_source_id"] = data_source_id

        LOGGER.info("[Stage 2/6] Lendo blocos da pagina de relatorio...")
        page_blocks = retrieve_all_block_children_recursive(page_id)
        global BLOCK_INDEX
        BLOCK_INDEX = {str(block.get("id", "")): block for block in page_blocks if block.get("id")}
        auto_root_ids = _collect_auto_root_block_ids(page_blocks)
        ignored_block_ids = _collect_ignored_block_ids(page_blocks, auto_root_ids)
        filtered_page_blocks = [
            block
            for block in page_blocks
            if _normalize_ws(block.get("id", "")) not in ignored_block_ids
        ]
        textual_blocks = extract_textual_blocks(filtered_page_blocks)
        LOGGER.info(
            "Blocos totais: %d | textuais_filtrados: %d | auto_roots_detectados: %d | blocos_ignorados: %d",
            len(page_blocks),
            len(textual_blocks),
            len(auto_root_ids),
            len(ignored_block_ids),
        )

        LOGGER.info("[Stage 3/6] Selecionando alvos textuais e companions de tabela com link explicito da base fonte...")
        target_text_blocks, linked_page_ids, target_stats = select_target_text_blocks_by_source_links(
            textual_blocks,
            source_database_id=source_database_id,
            source_data_source_id=data_source_id,
        )
        table_targets, table_linked_page_ids, table_stats = select_table_companion_targets_by_source_links(
            filtered_page_blocks,
            source_database_id=source_database_id,
            source_data_source_id=data_source_id,
            ignored_block_ids=ignored_block_ids,
        )

        all_linked_page_ids: List[str] = []
        seen_linked_page_ids: set[str] = set()
        for raw_page_id in list(linked_page_ids) + list(table_linked_page_ids):
            try:
                page_id_norm = _normalize_notion_id(str(raw_page_id))
            except Exception:
                continue
            if page_id_norm in seen_linked_page_ids:
                continue
            seen_linked_page_ids.add(page_id_norm)
            all_linked_page_ids.append(page_id_norm)

        if not target_text_blocks and not table_targets:
            deleted_auto = _delete_auto_generated_roots(sorted(auto_root_ids))
            elapsed_no_target = round(max(0.0, time.time() - started_at), 2)
            report.update(
                {
                    "status": "completed",
                    "elapsed_seconds": elapsed_no_target,
                    "updated_at": utc_now_iso(),
                    "stats": {
                        "blocks_total": len(page_blocks),
                        "text_blocks_total": len(textual_blocks),
                        "text_blocks_targeted": 0,
                        "text_blocks_not_targeted": len(textual_blocks),
                        "tables_targeted": 0,
                        "table_companion_cases": 0,
                        "patch_candidates": 0,
                        "updated": 0,
                        "failed": 0,
                        "unchanged_targeted": 0,
                        "untouched_non_target": len(textual_blocks),
                        "linked_pages": 0,
                        "auto_deleted": deleted_auto,
                        "context_primary": 0,
                        "context_fallback": 0,
                        "context_final": 0,
                    },
                    "note": "Nenhum alvo elegivel: somente blocos textuais e tabelas com links explicitos para a base fonte sao processados.",
                }
            )
            write_json_atomic(REPORT_FILE, report)
            LOGGER.info(
                "Nenhum alvo elegivel para alteracao. "
                "A pagina foi mantida intacta (criterio: link explicito para base fonte). auto_removidos=%d",
                deleted_auto,
            )
            LOGGER.info("Relatorio salvo em: %s", REPORT_FILE)
            return 0

        LOGGER.info("[Stage 4/6] Construindo contexto estrito por link de cada bloco...")
        context_by_page_id = build_case_context_map_from_linked_pages(all_linked_page_ids)
        context_primary = _dedupe_contexts(list(context_by_page_id.values()), limit=MAX_CONTEXT_CASES)
        context_fallback: List[CaseContext] = []
        final_context = context_primary
        target_text_blocks, scope_stats = _attach_scoped_contexts_to_blocks(
            target_text_blocks,
            context_by_page_id=context_by_page_id,
            per_block_limit=3,
        )
        prepared_text_targets: List[Dict[str, Any]] = []
        for block in target_text_blocks:
            item = dict(block)
            item["item_kind"] = "patch_text_block"
            item["render_compact"] = True
            prepared_text_targets.append(item)
        target_text_blocks = prepared_text_targets

        table_case_items, table_scope_stats = _build_table_companion_items(
            table_targets,
            context_by_page_id=context_by_page_id,
        )
        LOGGER.info(
            "[Fallback] desativado no modo estrito por bloco: cada bloco usa apenas contexto do(s) seu(s) link(s)."
        )

        LOGGER.info(
            "Contexto: links_fonte=%d | contexto_links=%d | blocos_com_contexto=%d | blocos_sem_contexto=%d | refs_contexto=%d | tabelas_com_contexto=%d | casos_tabela=%d",
            len(all_linked_page_ids),
            len(context_primary),
            int(scope_stats.get("blocks_with_context", 0)),
            int(scope_stats.get("blocks_without_context", 0)),
            int(scope_stats.get("total_context_refs", 0)),
            int(table_scope_stats.get("tables_with_context", 0)),
            int(table_scope_stats.get("cases_with_context", 0)),
        )
        _log_block_link_audit(target_text_blocks)

        LOGGER.info("[Stage 5/6] Gerando melhorias via OpenAI...")
        patch_plans: List[BlockPatchPlan] = []
        text_outputs_by_id, text_ai_stats = _process_items_via_openai(
            target_text_blocks,
            config=config,
            final_context=final_context,
            batch_prefix="patch_text_block",
        )
        table_outputs_by_id, table_ai_stats = _process_items_via_openai(
            table_case_items,
            config=config,
            final_context=final_context,
            batch_prefix="table_companion",
        )

        for item in target_text_blocks:
            original = _normalize_ws(item.get("text", ""))
            improved_text = _normalize_multiline_ws(text_outputs_by_id.get(str(item.get("id", "")), ""))
            if not improved_text:
                improved_text = original
            if _normalize_ws(improved_text) == original:
                continue
            rich_text = markdown_bold_to_notion_rich_text(improved_text)
            patch_plans.append(
                BlockPatchPlan(
                    block_id=str(item.get("id", "")),
                    block_type=str(item.get("type", "")),
                    original_text=original,
                    improved_text=improved_text,
                    rich_text_payload=rich_text,
                )
            )

        companion_plans = _build_table_companion_plans(table_targets, table_outputs_by_id)
        lawyer_lines = _compute_lawyer_frequency_lines(
            all_linked_page_ids,
            context_by_page_id=context_by_page_id,
        )

        LOGGER.info(
            "[Stage 6/6] Aplicando patches, companions e secao de advogados no Notion... patches=%d | companions=%d | advogados=%d",
            len(patch_plans),
            len(companion_plans),
            len(lawyer_lines),
        )
        deleted_auto = _delete_auto_generated_roots(sorted(auto_root_ids))
        updated = 0
        failed = 0
        unchanged_targeted = len(target_text_blocks) - len(patch_plans)
        untouched_non_target = len(textual_blocks) - len(target_text_blocks)
        table_companions_created = 0
        lawyer_section_created = 0
        failures: List[Dict[str, str]] = []

        for idx, plan in enumerate(patch_plans, start=1):
            try:
                patch_text_block(plan.block_id, plan.block_type, plan.rich_text_payload)
                updated += 1
                LOGGER.info(
                    "[%d/%d] Bloco atualizado: %s (%s)",
                    idx,
                    len(patch_plans),
                    plan.block_id,
                    plan.block_type,
                )
            except Exception as exc:
                failed += 1
                failures.append({"block_id": plan.block_id, "erro": str(exc)})
                LOGGER.error(
                    "[%d/%d] Falha no bloco %s (%s): %s",
                    idx,
                    len(patch_plans),
                    plan.block_id,
                    plan.block_type,
                    exc,
                )

        for idx, plan in enumerate(companion_plans, start=1):
            try:
                root_id = _apply_table_companion_plan(plan)
                table_companions_created += 1
                LOGGER.info(
                    "[Companion %d/%d] Tabela %s recebeu cluster %s com %d processo(s).",
                    idx,
                    len(companion_plans),
                    plan.table_id,
                    root_id,
                    len(plan.process_texts),
                )
            except Exception as exc:
                failed += 1
                failures.append({"table_id": plan.table_id, "erro": str(exc)})
                LOGGER.error(
                    "[Companion %d/%d] Falha ao criar cluster da tabela %s: %s",
                    idx,
                    len(companion_plans),
                    plan.table_id,
                    exc,
                )

        try:
            lawyer_section_ids = _apply_lawyer_section(page_id, lawyer_lines)
            lawyer_section_created = len(lawyer_section_ids)
            LOGGER.info(
                "[Advogados] Secao atualizada | blocos_criados=%d | itens=%d",
                lawyer_section_created,
                len(lawyer_lines),
            )
        except Exception as exc:
            failed += 1
            failures.append({"section": "advogados_recorrentes", "erro": str(exc)})
            LOGGER.error("[Advogados] Falha ao atualizar secao final: %s", exc)

        reference_norm_stats: Dict[str, Any] = {
            "reference_count": 0,
            "text_blocks_normalized": 0,
            "table_rows_normalized": 0,
            "link_to_page_converted": 0,
            "failures": [],
        }
        try:
            LOGGER.info("[Refs-globais] Normalizando referencias do Notion na ordem do relatorio...")
            reference_norm_stats = normalize_global_notion_references(page_id)
            reference_norm_failures = [
                {
                    "section": "normalizacao_referencias",
                    "block_id": item.get("block_id", ""),
                    "block_type": item.get("block_type", ""),
                    "erro": item.get("erro", ""),
                }
                for item in (reference_norm_stats.get("failures") or [])
                if isinstance(item, dict)
            ]
            if reference_norm_failures:
                failed += len(reference_norm_failures)
                failures.extend(reference_norm_failures)
            LOGGER.info(
                "[Refs-globais] referencias=%d | blocos_texto=%d | linhas_tabela=%d | link_to_page=%d | falhas=%d",
                int(reference_norm_stats.get("reference_count", 0)),
                int(reference_norm_stats.get("text_blocks_normalized", 0)),
                int(reference_norm_stats.get("table_rows_normalized", 0)),
                int(reference_norm_stats.get("link_to_page_converted", 0)),
                len(reference_norm_failures),
            )
        except Exception as exc:
            failed += 1
            failures.append({"section": "normalizacao_referencias", "erro": str(exc)})
            LOGGER.error("[Refs-globais] Falha na normalizacao final: %s", exc)

        elapsed = round(max(0.0, time.time() - started_at), 2)
        report.update(
            {
                "status": "completed_with_errors" if failed else "completed",
                "elapsed_seconds": elapsed,
                "updated_at": utc_now_iso(),
                "stats": {
                    "blocks_total": len(page_blocks),
                    "text_blocks_total": len(textual_blocks),
                    "text_blocks_targeted": len(target_text_blocks),
                    "text_blocks_not_targeted": untouched_non_target,
                    "text_blocks_with_any_notion_link": int(target_stats.get("text_blocks_with_any_notion_link", 0)),
                    "tables_targeted": len(table_targets),
                    "table_rows_with_any_notion_link": int(table_stats.get("table_rows_with_any_notion_link", 0)),
                    "table_companion_cases": int(table_stats.get("table_companion_cases", 0)),
                    "table_companions_created": table_companions_created,
                    "lawyer_frequency_items": len(lawyer_lines),
                    "lawyer_section_blocks_created": lawyer_section_created,
                    "auto_deleted": deleted_auto,
                    "global_reference_count": int(reference_norm_stats.get("reference_count", 0)),
                    "text_blocks_reference_normalized": int(reference_norm_stats.get("text_blocks_normalized", 0)),
                    "table_rows_reference_normalized": int(reference_norm_stats.get("table_rows_normalized", 0)),
                    "link_to_page_reference_converted": int(reference_norm_stats.get("link_to_page_converted", 0)),
                    "patch_candidates": len(patch_plans),
                    "updated": updated,
                    "failed": failed,
                    "unchanged_targeted": unchanged_targeted,
                    "untouched_non_target": untouched_non_target,
                    "linked_pages": len(all_linked_page_ids),
                    "context_primary": len(context_primary),
                    "context_fallback": len(context_fallback),
                    "context_final": len(final_context),
                    "table_cases_with_context": int(table_scope_stats.get("cases_with_context", 0)),
                    "table_cases_without_context": int(table_scope_stats.get("cases_without_context", 0)),
                    "openai_text_batches": int(text_ai_stats.get("batches_total", 0)),
                    "openai_table_batches": int(table_ai_stats.get("batches_total", 0)),
                },
                "failures": failures,
            }
        )
        write_json_atomic(REPORT_FILE, report)
        LOGGER.info("Relatorio salvo em: %s", REPORT_FILE)
        LOGGER.info(
            "Resumo: atualizados=%d | companions=%d | advogados=%d | inalterados_alvo=%d | intactos_fora_criterio=%d | auto_removidos=%d | falhas=%d",
            updated,
            table_companions_created,
            len(lawyer_lines),
            unchanged_targeted,
            untouched_non_target,
            deleted_auto,
            failed,
        )
        return 0 if failed == 0 else 2
    except Exception as exc:
        elapsed = round(max(0.0, time.time() - started_at), 2)
        report.update(
            {
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": elapsed,
                "updated_at": utc_now_iso(),
            }
        )
        try:
            write_json_atomic(REPORT_FILE, report)
        except Exception:
            pass
        LOGGER.error("Falha na execucao: %s", exc)
        return 1


def _resolve_page_url(args: argparse.Namespace) -> str:
    initial_value = _normalize_ws(args.page_url) or _load_last_page_url()
    picked = _run_gui_for_page_url(initial_value=initial_value)
    if picked is None:
        return initial_value
    return _normalize_ws(picked)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atualiza blocos textuais de relatorio no Notion com apoio de IA."
    )
    parser.add_argument(
        "--page-url",
        default="",
        help="URL inicial da pagina no Notion para preencher a GUI.",
    )
    parser.add_argument(
        "--source-database-url",
        default=DEFAULT_SOURCE_DATABASE_URL,
        help="URL do database/base fonte no Notion.",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=DEFAULT_OPENAI_BATCH_SIZE,
        help=f"Tamanho do lote de blocos para OpenAI (padrao: {DEFAULT_OPENAI_BATCH_SIZE}).",
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
        "--openai-timeout",
        type=int,
        default=75,
        help="Timeout por chamada OpenAI em segundos (padrao: 75).",
    )
    parser.add_argument(
        "--openai-retries",
        type=int,
        default=3,
        help="Tentativas de chamada OpenAI (padrao: 3).",
    )
    parser.add_argument(
        "--notion-min-interval",
        type=float,
        default=DEFAULT_NOTION_MIN_INTERVAL_S,
        help=f"Intervalo minimo entre requests Notion em segundos (padrao: {DEFAULT_NOTION_MIN_INTERVAL_S:.2f}).",
    )
    parser.add_argument(
        "--notion-timeout",
        type=int,
        default=30,
        help="Timeout por chamada Notion em segundos (padrao: 30).",
    )
    parser.add_argument(
        "--notion-retries",
        type=int,
        default=4,
        help="Tentativas de chamada Notion (padrao: 4).",
    )
    parser.add_argument("--verbose", action="store_true", help="Logs detalhados.")
    parser.add_argument("--quiet", action="store_true", help="Exibe apenas avisos/erros.")
    parser.add_argument("--debug", action="store_true", help="Ativa logs de debug tecnico.")
    parser.add_argument("--log-file", default="", help="Arquivo opcional de log.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logger = configure_standard_logging(
        SCRIPT_STEM,
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
    )
    install_print_logger_bridge(globals(), logger)

    page_url = _resolve_page_url(args)
    if not page_url:
        logger.error("Nenhuma URL informada.")
        return 1
    try:
        extract_notion_id_from_url(page_url)
    except Exception as exc:
        logger.error("URL da pagina invalida: %s", exc)
        return 1

    run_config = RunConfig(
        page_url=page_url,
        source_database_url=str(args.source_database_url or DEFAULT_SOURCE_DATABASE_URL),
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
        openai_batch_size=max(1, int(args.openai_batch_size)),
        openai_max_workers=max(1, int(args.openai_max_workers)),
        openai_target_rpm=max(0, int(args.openai_target_rpm)),
        openai_timeout_s=max(10, int(args.openai_timeout)),
        openai_retries=max(1, int(args.openai_retries)),
        notion_min_interval_s=max(0.0, float(args.notion_min_interval)),
        notion_timeout_s=max(5, int(args.notion_timeout)),
        notion_retries=max(1, int(args.notion_retries)),
    )

    try:
        _build_clients(run_config)
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    try:
        return run_agent(run_config)
    finally:
        _close_clients()


if __name__ == "__main__":
    raise SystemExit(main())
