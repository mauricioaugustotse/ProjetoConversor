#!/usr/bin/env python3
"""
Agente IA para atualizar relatorios no Notion com GUI simples.

Fluxo:
1. Resolve URL da pagina alvo via CLI/GUI/input.
2. Le blocos da pagina no Notion e identifica blocos textuais.
3. Monta contexto usando links Notion do proprio relatorio e fallback via data source.
4. Reescreve blocos textuais com OpenAI (gpt-5.1), incluindo destaque de partidos.
5. Atualiza os blocos no Notion preservando blocos nao textuais.
6. Gera relatorio local em .notion_relatorio_updater.report.json.
"""

from __future__ import annotations

import argparse
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
from openai_progress_utils import utc_now_iso, write_json_atomic


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_FILE = SCRIPT_DIR / ".notion_relatorio_updater.report.json"
DEFAULT_SOURCE_DATABASE_URL = (
    "https://www.notion.so/301721955c6480afaa2eedbdc7cd2aba?v=301721955c6481eb9d07000cfb23cbe5"
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


LOGGER = logging.getLogger("NOTION_relatorio_updater_agent")

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


def _block_plain_text(block: Dict[str, Any]) -> str:
    return _rich_text_plain_text(_block_rich_text(block))


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

    rich_text = _block_rich_text(block)
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

    rich_text = _block_rich_text(block)
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


def _extract_partidos(text: str) -> str:
    out: List[str] = []
    for match in PARTY_UF_RE.finditer(text or ""):
        token = _normalize_ws(match.group(1)).upper()
        base = token.split("/", 1)[0]
        if base in PARTY_ABBREVIATIONS:
            out.append(token)
    for match in PARTY_TOKEN_RE.finditer(text or ""):
        token = _normalize_ws(match.group(0)).upper()
        if token:
            out.append(token)
    return _safe_join(out)


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
        partes=_best_partes(prop_map, full_text),
        punchline=punchline,
        texto_decisao=texto_decisao,
        partidos=_extract_partidos(full_text),
        alegacoes=_pick_sentences(
            full_text,
            keywords=["alega", "sustenta", "argumenta", "afirma", "defende", "tese"],
            max_items=3,
        ),
        advogados=advogados_coluna or _extract_advogados(full_text),
        fundamentos=_pick_sentences(
            full_text,
            keywords=["fundamento", "art.", "lei", "resolucao", "resolução", "sumula", "súmula"],
            max_items=3,
        ),
        resultado=_pick_sentences(
            full_text,
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
                "block_type": block_type,
                "texto_original": _truncate_text(str(block.get("text", "")), max_len),
                "links_referencia": refs,
                "contexto_casos_bloco": _context_for_prompt(scoped_contexts),
            }
        )

    return {
        "objetivo": (
            "Melhorar textos de relatorio juridico eleitoral no Notion. "
            "Reescrever apenas os blocos abaixo, sem inventar fatos."
        ),
        "regras_obrigatorias": [
            "Retorne um item por bloco com os campos padronizados exigidos.",
            "Use somente o contexto_casos_bloco do item correspondente ao mesmo index.",
            "Nao use contexto de um bloco para escrever outro bloco.",
            "Se contexto_casos_bloco vier vazio, reescreva com cautela usando apenas texto_original e links_referencia.",
            "Priorize os campos estruturados do contexto_casos_bloco quando presentes (numero_unico, data_decisao, sigla_classe, nome_municipio, sigla_uf, relator, partes, advogados, punchline, texto_decisao, tema).",
            "Sempre esclarecer o que ocorreu no caso (campo o_que_ocorreu).",
            "Sempre preencher consequencias com o impacto/desdobramento efetivo do caso.",
            "Evite estilo telegrafado; escreva com fluidez, conectivos e frases completas.",
            "No campo o_que_ocorreu, escreva um paragrafo analitico de 2-4 frases com contexto, prova e desfecho.",
            "No campo consequencias, descreva efeitos juridicos e praticos em linguagem corrida, sem repeticao literal.",
            "Campos centrais por item: o_que_ocorreu e consequencias.",
            "No JSON final, inclua tambem analise_estrategica, partes, partidos, advogados_famosos, cidade_uf, processo_cnj, eleicao_ano, alegacoes, fundamentos e resultado.",
            "Considere links_referencia para preservar rastreabilidade da base.",
            "Nao inventar dados; quando faltar evidencia para um campo complementar, retorne string vazia nesse campo.",
            "Nunca escreva placeholders de ausencia de dados.",
            "Priorize cidade_uf quando houver evidencia no caso.",
            "Evite repeticao literal entre o_que_ocorreu, consequencias e resultado.",
            "Quando possivel, forneca analise_estrategica curta (1-2 frases) com impacto pratico para o analista.",
            "Usar frases diretas, voz ativa, sem redundancia.",
        ],
        "contexto_casos": [],
        "blocos": blocks_payload,
        "saida_esperada": {
            "items": [
                {
                    "index": 0,
                    "o_que_ocorreu": "descricao objetiva do caso",
                    "consequencias": "efeitos concretos da decisao",
                    "analise_estrategica": "implicacao pratica para o analista",
                    "partes": "nomes das partes",
                    "partidos": "partidos envolvidos",
                    "advogados_famosos": "advogados relevantes",
                    "cidade_uf": "cidade/UF",
                    "processo_cnj": "numero CNJ",
                    "eleicao_ano": "ano da eleicao",
                    "alegacoes": "alegacoes das partes",
                    "fundamentos": "fundamentos juridicos",
                    "resultado": "resultado do julgamento",
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
        and not line.casefold().startswith("referencias:")
        and not line.casefold().startswith("referências:")
    ]
    return "\n".join(cleaned)


KNOWN_OUTPUT_LABEL_RE = re.compile(
    r"(?i)^(?:-\s*)?(?:\*\*)?(?:"
    r"o\s+que\s+ocorreu|"
    r"consequ[eê]ncias?|"
    r"partes\s+envolvidas|"
    r"partidos\s+pol[ií]ticos\s+envolvidos|"
    r"advogados\s+em\s+destaque\s+no\s+direito\s+eleitoral|"
    r"cidade/uf(?:\s+de\s+origem)?|"
    r"processo\s+\(cnj\)|"
    r"an[oa]\s+da\s+elei[cç][aã]o|"
    r"alega[cç][oõ]es\s+suscitadas|"
    r"fundamentos\s+jur[ií]dicos|"
    r"resultado\s+do\s+julgamento|"
    r"refer[eê]ncias(?:\s+notion)?"
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


def _resolve_output_origin(cidade_uf: str, original_text: str) -> str:
    origem = _normalize_structured_value(cidade_uf)
    if not origem:
        origem = _extract_city_uf(original_text)
    if not origem:
        origem = "Origem do caso"
    return origem


def _compose_primary_narrative(
    o_que_ocorreu: str,
    analise_estrategica: str,
    fundamentos: str,
    consequencias: str,
    resultado: str,
) -> str:
    # Fundamentos juridicos deixam de ser linha separada e passam a integrar o paragrafo principal.
    parts = _dedupe_non_redundant_values([o_que_ocorreu, fundamentos, analise_estrategica])
    if not parts:
        parts = _dedupe_non_redundant_values([o_que_ocorreu, consequencias, resultado])
    if not parts:
        return "Trecho sem informacao suficiente para sintese objetiva."
    return " ".join(_ensure_sentence_end(part) for part in parts)


def _render_standardized_block_text(
    item: Dict[str, Any],
    reference_urls: Sequence[str],
    *,
    original_text: str = "",
) -> str:
    cleaned_original = _strip_reference_lines(_remove_placeholder_lines(original_text))
    normalized_original_summary = _strip_known_output_labels(cleaned_original)
    o_que_ocorreu = _normalize_structured_value(item.get("o_que_ocorreu"))
    consequencias = _normalize_structured_value(item.get("consequencias"))
    analise_estrategica = _normalize_structured_value(item.get("analise_estrategica"))

    partes = _normalize_structured_value(item.get("partes"))
    partidos = _normalize_structured_value(item.get("partidos"))
    advogados = _normalize_structured_value(item.get("advogados_famosos"))
    cidade = _normalize_structured_value(item.get("cidade_uf"))
    processo = _normalize_structured_value(item.get("processo_cnj"))
    eleicao = _normalize_structured_value(item.get("eleicao_ano"))
    alegacoes = _normalize_structured_value(item.get("alegacoes"))
    fundamentos = _normalize_structured_value(item.get("fundamentos"))
    resultado = _normalize_structured_value(item.get("resultado"))

    if not o_que_ocorreu:
        o_que_ocorreu = normalized_original_summary
    if not o_que_ocorreu:
        o_que_ocorreu = "Trecho sem informacao suficiente para sintese objetiva."
    if not consequencias:
        consequence_source = ". ".join(
            [part for part in [resultado, fundamentos, o_que_ocorreu, normalized_original_summary] if part]
        )
        consequencias = _infer_consequencias(consequence_source)
    if not consequencias:
        consequencias = "Sem desfecho explicito no trecho; recomenda-se monitoramento do andamento processual."

    origem = _resolve_output_origin(cidade, cleaned_original)
    primary_narrative = _compose_primary_narrative(
        o_que_ocorreu,
        analise_estrategica,
        fundamentos,
        consequencias,
        resultado,
    )
    lines: List[str] = [f"**{origem}**: {primary_narrative}"]

    seen_detail_keys: set[str] = set()

    def _append_detail(
        label: str,
        value: str,
        *,
        label_bold: bool = False,
        value_bold: bool = False,
        party: bool = False,
        dedupe_against: Sequence[str] = (),
    ) -> None:
        normalized_value = _normalize_structured_value(value)
        if not normalized_value:
            return
        value_key = _normalize_compare_key(normalized_value)
        if not value_key:
            return
        for other in dedupe_against:
            other_key = _normalize_compare_key(other)
            if not other_key:
                continue
            if value_key == other_key or value_key in other_key or other_key in value_key:
                return
        if value_key in seen_detail_keys:
            return
        seen_detail_keys.add(value_key)
        rendered_value = _markdown_bold(normalized_value) if value_bold else normalized_value
        if party:
            rendered_value = _ensure_party_markdown_bold(rendered_value)
        rendered_label = f"**{label}:**" if label_bold else f"{label}:"
        lines.append(f"{rendered_label} {rendered_value}")

    # Ordem fixa solicitada pelo usuario.
    _append_detail("Consequências", consequencias, label_bold=True)
    _append_detail("Alegações suscitadas", alegacoes, label_bold=True)
    _append_detail("Resultado do julgamento", resultado, label_bold=True)
    _append_detail("Partidos políticos envolvidos", partidos, party=True)
    _append_detail("Ano da eleição", eleicao, value_bold=True)
    _append_detail("Partes envolvidas", partes, value_bold=True)
    _append_detail("Processo (CNJ)", processo, label_bold=True, value_bold=True)
    _append_detail("Advogados em destaque no Direito Eleitoral", advogados, value_bold=True)

    out = "\n".join(lines)
    out = _apply_semantic_bold(out, [origem, cidade, partes, partidos, advogados])
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
        and not line.casefold().startswith("referencias:")
        and not line.casefold().startswith("referências:")
    ]
    cleaned_base = "\n".join(base_without_ref).strip()

    if not merged_refs:
        return cleaned_base

    numbered_refs = " ".join(f"[{idx}]({url})" for idx, url in enumerate(merged_refs, start=1))
    ref_line = f"Referências: {numbered_refs}"
    if not cleaned_base:
        return ref_line
    return f"{cleaned_base}\n{ref_line}"


def _render_standardized_fallback_text(base_text: str, reference_urls: Sequence[str]) -> str:
    base_raw = _strip_reference_lines(_remove_placeholder_lines(base_text)).strip()
    if not base_raw:
        minimal = (
            "**Origem do caso**: Trecho original sem informacao util para sintese objetiva.\n"
            "**Consequências:** Necessario monitorar o andamento do processo na pagina de referencia."
        )
        return _append_reference_line(minimal, reference_urls)

    origem = _resolve_output_origin("", base_raw)
    fallback_summary = _strip_known_output_labels(base_raw)
    if not fallback_summary:
        fallback_summary = _normalize_ws(base_raw)
    if not fallback_summary:
        fallback_summary = "Trecho original sem informacao util para sintese objetiva."
    if len(fallback_summary) > 900:
        fallback_summary = _truncate_text(fallback_summary, 900)

    lowered = fallback_summary.casefold()
    has_consequencias = "consequencias:" in lowered or "consequências:" in lowered

    lines: List[str] = [f"**{origem}**: {fallback_summary}"]
    if not has_consequencias:
        inferred = _infer_consequencias(fallback_summary)
        if inferred:
            lines.append(f"**Consequências:** {inferred}")
        else:
            lines.append(
                "**Consequências:** Sem desfecho explicito no trecho; recomenda-se monitoramento do andamento processual."
            )

    fallback = "\n".join(lines)
    fallback = _ensure_party_markdown_bold(fallback)
    return _append_reference_line(fallback, reference_urls)


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
                        "o_que_ocorreu": {"type": "string"},
                        "consequencias": {"type": "string"},
                        "analise_estrategica": {"type": "string"},
                        "partes": {"type": "string"},
                        "partidos": {"type": "string"},
                        "advogados_famosos": {"type": "string"},
                        "cidade_uf": {"type": "string"},
                        "processo_cnj": {"type": "string"},
                        "eleicao_ano": {"type": "string"},
                        "alegacoes": {"type": "string"},
                        "fundamentos": {"type": "string"},
                        "resultado": {"type": "string"},
                    },
                    "required": [
                        "index",
                        "o_que_ocorreu",
                        "consequencias",
                        "analise_estrategica",
                        "partes",
                        "partidos",
                        "advogados_famosos",
                        "cidade_uf",
                        "processo_cnj",
                        "eleicao_ano",
                        "alegacoes",
                        "fundamentos",
                        "resultado",
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
        improved = ""
        if item:
            has_structured = any(
                _normalize_structured_value(item.get(key, "")) for key in (
                    "o_que_ocorreu",
                    "consequencias",
                    "analise_estrategica",
                    "partes",
                    "partidos",
                    "advogados_famosos",
                    "cidade_uf",
                    "processo_cnj",
                    "eleicao_ano",
                    "alegacoes",
                    "fundamentos",
                    "resultado",
                )
            )
            if has_structured:
                improved = _render_standardized_block_text(item, refs, original_text=original)
            else:
                free_text = _normalize_ws(item.get("improved_text", ""))
                if free_text:
                    improved = _render_standardized_fallback_text(
                        _ensure_party_markdown_bold(free_text),
                        refs,
                    )

        if not improved:
            improved = _render_standardized_fallback_text(original, refs)
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


def _run_gui_for_page_url(initial_value: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception:
        return ""

    result = {"url": ""}

    root = tk.Tk()
    root.title("Agente IA - Notion Relatorio")
    root.geometry("760x160")
    root.minsize(680, 140)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Cole o link da pagina do relatorio no Notion para atualizar:",
    ).pack(anchor="w")

    url_var = tk.StringVar(value=initial_value)
    entry = ttk.Entry(frame, textvariable=url_var)
    entry.pack(fill="x", pady=(8, 10))
    entry.focus_set()

    row = ttk.Frame(frame)
    row.pack(fill="x")

    def _on_run() -> None:
        value = _normalize_ws(url_var.get())
        if not value:
            messagebox.showwarning("URL obrigatoria", "Informe o link da pagina do Notion.")
            return
        result["url"] = value
        root.destroy()

    def _on_cancel() -> None:
        result["url"] = ""
        root.destroy()

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
        "script": "NOTION_relatorio_updater_agent.py",
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
        textual_blocks = extract_textual_blocks(page_blocks)
        LOGGER.info("Blocos totais: %d | textuais: %d", len(page_blocks), len(textual_blocks))

        LOGGER.info("[Stage 3/6] Selecionando apenas blocos-alvo com link explicito da base fonte...")
        target_text_blocks, linked_page_ids, target_stats = select_target_text_blocks_by_source_links(
            textual_blocks,
            source_database_id=source_database_id,
            source_data_source_id=data_source_id,
        )
        if not target_text_blocks:
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
                        "patch_candidates": 0,
                        "updated": 0,
                        "failed": 0,
                        "unchanged_targeted": 0,
                        "untouched_non_target": len(textual_blocks),
                        "linked_pages": 0,
                        "context_primary": 0,
                        "context_fallback": 0,
                        "context_final": 0,
                    },
                    "note": "Nenhum bloco elegivel: somente blocos com links explicitos para a base fonte sao alterados.",
                }
            )
            write_json_atomic(REPORT_FILE, report)
            LOGGER.info(
                "Nenhum bloco elegivel para alteracao. "
                "A pagina foi mantida intacta (criterio: link explicito para base fonte)."
            )
            LOGGER.info("Relatorio salvo em: %s", REPORT_FILE)
            return 0

        LOGGER.info("[Stage 4/6] Construindo contexto estrito por link de cada bloco...")
        context_by_page_id = build_case_context_map_from_linked_pages(linked_page_ids)
        context_primary = _dedupe_contexts(list(context_by_page_id.values()), limit=MAX_CONTEXT_CASES)
        context_fallback: List[CaseContext] = []
        final_context = context_primary
        target_text_blocks, scope_stats = _attach_scoped_contexts_to_blocks(
            target_text_blocks,
            context_by_page_id=context_by_page_id,
            per_block_limit=3,
        )
        LOGGER.info(
            "[Fallback] desativado no modo estrito por bloco: cada bloco usa apenas contexto do(s) seu(s) link(s)."
        )

        LOGGER.info(
            "Contexto: links_fonte=%d | contexto_links=%d | blocos_com_contexto=%d | blocos_sem_contexto=%d | refs_contexto=%d",
            len(linked_page_ids),
            len(context_primary),
            int(scope_stats.get("blocks_with_context", 0)),
            int(scope_stats.get("blocks_without_context", 0)),
            int(scope_stats.get("total_context_refs", 0)),
        )
        _log_block_link_audit(target_text_blocks)

        LOGGER.info("[Stage 5/6] Gerando melhorias via OpenAI...")
        patch_plans: List[BlockPatchPlan] = []
        effective_batch_size = max(1, int(config.openai_batch_size or MAX_BLOCK_BATCH))
        batches = _chunk_items_for_openai(
            target_text_blocks,
            max_items=effective_batch_size,
            char_budget=OPENAI_BATCH_CHAR_BUDGET,
        )
        total_batches = len(batches)
        max_chars_batch = max((_estimate_openai_input_chars(batch, compact=False) for batch in batches), default=0)
        LOGGER.info(
            "[OpenAI] Lotes preparados | total=%d | max_chars_lote=%d | limite_chars=%d | max_blocos_lote=%d",
            total_batches,
            max_chars_batch,
            OPENAI_BATCH_CHAR_BUDGET,
            effective_batch_size,
        )
        openai_workers = min(max(1, int(config.openai_max_workers or 1)), max(1, total_batches))
        if max_chars_batch >= int(OPENAI_BATCH_CHAR_BUDGET * 0.9) and openai_workers > 1:
            LOGGER.info(
                "[OpenAI] Ajuste de estabilidade: lotes pesados (max_chars=%d). Limitando workers de %d para 1.",
                max_chars_batch,
                openai_workers,
            )
            openai_workers = 1
        if total_batches >= 4 and openai_workers > 2:
            LOGGER.info(
                "[OpenAI] Ajuste de estabilidade: limitando workers de %d para 2 (lotes=%d).",
                openai_workers,
                total_batches,
            )
            openai_workers = 2
        if total_batches == 0:
            LOGGER.info("[OpenAI] Nenhum bloco textual para processar.")

        improved_batches: Dict[int, List[str]] = {}
        if total_batches > 0 and openai_workers <= 1:
            LOGGER.info("[OpenAI] Execucao sequencial de %d lote(s).", total_batches)
            for batch_idx, batch in enumerate(batches, start=1):
                batch_label = f"lote {batch_idx}/{total_batches}"
                LOGGER.info(
                    "[OpenAI] %s | blocos_no_lote=%d",
                    batch_label,
                    len(batch),
                )
                improved_batches[batch_idx - 1] = improve_text_blocks_with_openai(
                    batch,
                    final_context,
                    batch_label=batch_label,
                )
        elif total_batches > 0:
            LOGGER.info(
                "[OpenAI] Execucao paralela de %d lote(s) com %d worker(s).",
                total_batches,
                openai_workers,
            )

            def _batch_job(payload: tuple[int, List[Dict[str, Any]]]) -> tuple[int, List[str]]:
                zero_idx, batch_data = payload
                one_idx = zero_idx + 1
                batch_label = f"lote {one_idx}/{total_batches}"
                LOGGER.info(
                    "[OpenAI] %s | iniciado em worker | blocos=%d",
                    batch_label,
                    len(batch_data),
                )
                improved_local = improve_text_blocks_with_openai(
                    batch_data,
                    final_context,
                    batch_label=batch_label,
                )
                return zero_idx, improved_local

            with ThreadPoolExecutor(max_workers=openai_workers) as executor:
                future_map = {
                    executor.submit(_batch_job, (idx, batch)): idx
                    for idx, batch in enumerate(batches)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        done_idx, improved = future.result()
                        improved_batches[done_idx] = improved
                        LOGGER.info("[OpenAI] lote %d/%d concluido.", done_idx + 1, total_batches)
                    except Exception as exc:
                        LOGGER.error(
                            "[OpenAI] lote %d/%d falhou no worker: %s",
                            idx + 1,
                            total_batches,
                            exc,
                        )
                        improved_batches[idx] = [str(block.get("text", "")) for block in batches[idx]]

        for batch_idx, batch in enumerate(batches):
            improved = improved_batches.get(batch_idx, [str(block.get("text", "")) for block in batch])
            for item, new_text in zip(batch, improved):
                original = _normalize_ws(item.get("text", ""))
                improved_text = _normalize_multiline_ws(new_text)
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

        LOGGER.info(
            "[Stage 6/6] Aplicando patches no Notion... total=%d",
            len(patch_plans),
        )
        updated = 0
        failed = 0
        unchanged_targeted = len(target_text_blocks) - len(patch_plans)
        untouched_non_target = len(textual_blocks) - len(target_text_blocks)
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
                    "patch_candidates": len(patch_plans),
                    "updated": updated,
                    "failed": failed,
                    "unchanged_targeted": unchanged_targeted,
                    "untouched_non_target": untouched_non_target,
                    "linked_pages": len(linked_page_ids),
                    "context_primary": len(context_primary),
                    "context_fallback": len(context_fallback),
                    "context_final": len(final_context),
                },
                "failures": failures,
            }
        )
        write_json_atomic(REPORT_FILE, report)
        LOGGER.info("Relatorio salvo em: %s", REPORT_FILE)
        LOGGER.info(
            "Resumo: atualizados=%d | inalterados_alvo=%d | intactos_fora_criterio=%d | falhas=%d",
            updated,
            unchanged_targeted,
            untouched_non_target,
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
    page_url = _normalize_ws(args.page_url)
    if page_url:
        return page_url
    if not bool(args.no_gui):
        picked = _run_gui_for_page_url(initial_value="")
        if picked:
            return _normalize_ws(picked)
    try:
        typed = input("Cole o link da pagina do relatorio no Notion: ").strip()
    except EOFError:
        typed = ""
    return _normalize_ws(typed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atualiza blocos textuais de relatorio no Notion com apoio de IA."
    )
    parser.add_argument("--page-url", default="", help="URL da pagina de relatorio no Notion.")
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
    parser.add_argument("--no-gui", action="store_true", help="Desativa GUI e usa CLI/input.")
    parser.add_argument("--verbose", action="store_true", help="Logs detalhados.")
    parser.add_argument("--quiet", action="store_true", help="Exibe apenas avisos/erros.")
    parser.add_argument("--debug", action="store_true", help="Ativa logs de debug tecnico.")
    parser.add_argument("--log-file", default="", help="Arquivo opcional de log.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logger = configure_standard_logging(
        "NOTION_relatorio_updater_agent",
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
