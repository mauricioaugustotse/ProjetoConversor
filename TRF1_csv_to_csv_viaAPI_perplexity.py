import asyncio
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp

# === CONFIGURACAO ===
INPUT_FILE = "boletins_de_jurisprudencia_TRF1.csv"
OUTPUT_FILE = "boletins_de_jurisprudencia_TRF1_with_news.csv"
CACHE_FILE = "perplexity_cache.json"
CHECKPOINT_FILE = ".boletins_de_jurisprudencia_TRF1_with_news.checkpoint.json"
CHECKPOINT_VERSION = 2

ENDPOINT = "https://api.perplexity.ai/search"

MAX_BATCH_SIZE = 3
MAX_CONCURRENT = 3
DELAY_BETWEEN_BATCHES_SEC = 0.5

MAX_LINKS_PER_ROW = 2
LEGACY_LINK_SEPARATOR = "; "
LEGACY_NOTICIA_COL = "noticia"
NOTICIA_COL_1 = "noticia_1"
NOTICIA_COL_2 = "noticia_2"

# === Cache em disco ===
CACHE: Dict[str, List[str]] = {}


class APIFatalStopError(RuntimeError):
    """Erro fatal nao-retryable da API (creditos/quota/autorizacao)."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_links(value: Any, *, max_links: Optional[int] = MAX_LINKS_PER_ROW) -> List[str]:
    raw_links: List[str] = []
    if isinstance(value, list):
        raw_links = [normalize_text(item) for item in value]
    elif isinstance(value, str):
        text = normalize_text(value)
        if text:
            # Delimitador legado oficial: "; " (nao dividir por ";" puro).
            raw_links = [normalize_text(part) for part in text.split(LEGACY_LINK_SEPARATOR)]
    elif value is not None:
        text = normalize_text(value)
        if text:
            raw_links = [text]

    deduped: List[str] = []
    seen: Set[str] = set()
    for link in raw_links:
        if not link:
            continue
        key = link.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)

    if max_links is None or max_links < 0:
        return deduped
    return deduped[:max_links]


def get_row_news_links(row: Dict[str, str]) -> List[str]:
    direct = normalize_links(
        [
            normalize_text(row.get(NOTICIA_COL_1, "")),
            normalize_text(row.get(NOTICIA_COL_2, "")),
        ],
        max_links=MAX_LINKS_PER_ROW,
    )
    if direct:
        return direct
    return normalize_links(row.get(LEGACY_NOTICIA_COL, ""), max_links=MAX_LINKS_PER_ROW)


def set_row_news_links(row: Dict[str, str], links: Any) -> None:
    normalized = normalize_links(links, max_links=MAX_LINKS_PER_ROW)
    row[NOTICIA_COL_1] = normalized[0] if len(normalized) > 0 else ""
    row[NOTICIA_COL_2] = normalized[1] if len(normalized) > 1 else ""
    row.pop(LEGACY_NOTICIA_COL, None)


def row_has_news(row: Dict[str, str]) -> bool:
    return bool(normalize_text(row.get(NOTICIA_COL_1, "")) or normalize_text(row.get(NOTICIA_COL_2, "")))


def build_output_fieldnames(input_fieldnames: List[str]) -> List[str]:
    if LEGACY_NOTICIA_COL in input_fieldnames:
        out: List[str] = []
        inserted = False
        for name in input_fieldnames:
            if name == LEGACY_NOTICIA_COL:
                if not inserted:
                    out.extend([NOTICIA_COL_1, NOTICIA_COL_2])
                    inserted = True
                continue
            if name in (NOTICIA_COL_1, NOTICIA_COL_2):
                continue
            out.append(name)
        return out

    out = list(input_fieldnames)
    if NOTICIA_COL_1 not in out:
        out.append(NOTICIA_COL_1)
    if NOTICIA_COL_2 not in out:
        out.append(NOTICIA_COL_2)
    return out


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


def atomic_write_json(path: str, payload: Dict[str, Any], indent: Optional[int] = None) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    os.replace(tmp_path, path)


def load_cache() -> None:
    if not os.path.exists(CACHE_FILE):
        return
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if not isinstance(cached, dict):
            print(f"Aviso: cache invalido em {CACHE_FILE}; ignorando.")
            return
        for key, value in cached.items():
            if not isinstance(key, str):
                continue
            CACHE[key] = normalize_links(value, max_links=MAX_LINKS_PER_ROW)
    except Exception as exc:  # noqa: BLE001
        print(f"Aviso: nao foi possivel carregar cache ({exc}).")


def save_cache() -> None:
    atomic_write_json(CACHE_FILE, CACHE, indent=2)


def file_signature(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    return {
        "path": os.path.normcase(os.path.abspath(path)),
        "size": int(st.st_size),
        "mtime_ns": int(mtime_ns),
    }


def same_signature(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        normalize_text(a.get("path")) == normalize_text(b.get("path"))
        and int(a.get("size", -1)) == int(b.get("size", -2))
        and int(a.get("mtime_ns", -1)) == int(b.get("mtime_ns", -2))
    )


def read_checkpoint() -> Dict[str, Any]:
    if not os.path.exists(CHECKPOINT_FILE):
        return {}
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {}
        return obj
    except Exception as exc:  # noqa: BLE001
        print(f"Aviso: checkpoint invalido ({exc}).")
        return {}


def checkpoint_payload(
    *,
    input_sig: Dict[str, Any],
    total_rows: int,
    total_candidates: int,
    processed_rows: Dict[str, List[str]],
) -> Dict[str, Any]:
    processed_count = len(processed_rows)
    with_url = sum(1 for links in processed_rows.values() if links)
    return {
        "version": CHECKPOINT_VERSION,
        "input_signature": input_sig,
        "total_rows": total_rows,
        "total_candidates": total_candidates,
        "processed_rows": processed_rows,
        "summary": {
            "processed_count": processed_count,
            "with_url_count": with_url,
            "without_url_count": processed_count - with_url,
            "remaining_count": max(0, total_candidates - processed_count),
        },
        "updated_at": utc_now_iso(),
    }


def save_checkpoint(
    *,
    input_sig: Dict[str, Any],
    total_rows: int,
    total_candidates: int,
    processed_rows: Dict[str, List[str]],
) -> None:
    payload = checkpoint_payload(
        input_sig=input_sig,
        total_rows=total_rows,
        total_candidates=total_candidates,
        processed_rows=processed_rows,
    )
    atomic_write_json(CHECKPOINT_FILE, payload, indent=2)


def normalize_processed_rows(
    raw: Any,
    *,
    max_rows: int,
    allowed_indices: Set[int],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        try:
            idx = int(str(key))
        except Exception:
            continue
        if idx < 0 or idx >= max_rows or idx not in allowed_indices:
            continue
        out[str(idx)] = normalize_links(value, max_links=MAX_LINKS_PER_ROW)
    return out


def write_output_csv(rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    tmp_path = f"{OUTPUT_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp_path, OUTPUT_FILE)


def persist_snapshot(
    *,
    rows: List[Dict[str, str]],
    fieldnames: List[str],
    input_sig: Dict[str, Any],
    total_candidates: int,
    processed_rows: Dict[str, List[str]],
) -> None:
    save_cache()
    write_output_csv(rows, fieldnames)
    save_checkpoint(
        input_sig=input_sig,
        total_rows=len(rows),
        total_candidates=total_candidates,
        processed_rows=processed_rows,
    )


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 100.0
    return (part / whole) * 100.0


def extract_api_error_message(raw_body: str) -> str:
    text = normalize_text(raw_body)
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except Exception:
        return text
    if not isinstance(payload, dict):
        return text

    parts: List[str] = []
    for key in ("message", "detail", "code", "type"):
        value = payload.get(key)
        if isinstance(value, str) and normalize_text(value):
            parts.append(normalize_text(value))

    error_payload = payload.get("error")
    if isinstance(error_payload, str) and normalize_text(error_payload):
        parts.append(normalize_text(error_payload))
    elif isinstance(error_payload, dict):
        for key in ("message", "detail", "code", "type"):
            value = error_payload.get(key)
            if isinstance(value, str) and normalize_text(value):
                parts.append(normalize_text(value))

    if not parts:
        return text
    return " | ".join(dict.fromkeys(parts))


def classify_fatal_api_error(status_code: int, error_message: str) -> Optional[str]:
    haystack = normalize_text(error_message).lower()

    if status_code in (401, 403):
        return "autorizacao"
    if status_code == 402:
        return "creditos/quota"

    auth_tokens = (
        "authorization required",
        "unauthorized",
        "invalid api key",
        "forbidden",
        "authentication",
    )
    if any(token in haystack for token in auth_tokens):
        return "autorizacao"

    credit_tokens = (
        "insufficient credit",
        "insufficient credits",
        "out of credits",
        "credit exhausted",
        "credits exhausted",
        "insufficient quota",
        "quota exceeded",
        "quota exhausted",
        "billing",
        "payment required",
        "balance",
        "usage limit",
        "ran out",
    )
    if any(token in haystack for token in credit_tokens):
        return "creditos/quota"

    if status_code == 429 and any(
        token in haystack for token in ("credit", "quota", "billing", "balance")
    ):
        return "creditos/quota"

    return None


# === Filtro de links juridicos brasileiros ===
def is_valid_juridical_link(url: str) -> bool:
    if not url:
        return False

    url_lower = url.lower()

    forbidden_domains = [
        "mega.nz",
        "dropbox.com",
        "drive.google.com",
        "mediafire.com",
        "4shared.com",
        "box.com",
        "onedrive.live.com",
    ]
    forbidden_patterns = [
        ".pdf",
        ".zip",
        ".rar",
        ".doc",
        ".docx",
        ".xlsx",
        ".ppt",
        "download",
        "downloads",
        "anonymous.hosting",
    ]
    if any(domain in url_lower for domain in forbidden_domains):
        return False
    if any(pat in url_lower for pat in forbidden_patterns):
        return False

    allowed_patterns = [
        "trf1.jus.br",
        "tse.jus.br",
        ".jus.br",
        "cnj.jus.br",
        "stf.jus.br",
        "jurisprudencia",
        "jurisprudência",
        "jurisdição",
        "jurisdicao",
        "juridica",
        "jurídica",
        "juridico",
        "jurídico",
        "jusbrasil",
        "jurisprudnoonline",
        "jurisprud",
        "juris",
        "jus",
        "conjur.com.br",
        "migalhas.com.br",
        "poder360.com.br",
    ]
    if any(pat in url_lower for pat in allowed_patterns):
        return True
    if ".br" in url_lower:
        return True

    return False


def truncate_text(text: str, max_chars: int = 120) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].strip() + " ..."


def make_query(row: Dict[str, str]) -> Optional[str]:
    punchline = normalize_text(row.get("punchline", ""))
    texto_do_boletim = normalize_text(row.get("texto_do_boletim", ""))
    tribunal = normalize_text(row.get("tribunal", ""))
    numero_processo = normalize_text(row.get("numero_processo", ""))
    tema = normalize_text(row.get("tema", ""))

    if not tribunal and not numero_processo and not punchline:
        return None

    parts: List[str] = []
    if tribunal:
        parts.append(tribunal)
    if numero_processo:
        parts.append(numero_processo)
    if punchline:
        parts.append(punchline)

    if texto_do_boletim:
        truncated = truncate_text(texto_do_boletim, 120)
        if "..." in truncated or len(truncated.strip()) > 20:
            parts.append(truncated)
    elif tema:
        parts.append(tema)

    parts.extend(
        [
            "notícia jurídica",
            "matéria jurídica",
            "imprensa jurídica",
            "site jurídico",
        ]
    )

    query = " ".join(parts).strip()
    return query or None


async def perplexity_lookup(
    session: aiohttp.ClientSession,
    query: str,
    api_key: str,
    max_results: int = MAX_LINKS_PER_ROW,
) -> Optional[List[str]]:
    if not query:
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    domains_order = [
        "trf1.jus.br",
        "jota.info",
        "conjur.com.br",
        "migalhas.com.br",
        "gov.br",
        "poder360.com.br",
        "ambitojuridico.com.br",
        "jurisnews.com.br",
        "direitonews.com.br",
        "olharjuridico.com.br",
    ]

    payload = {
        "query": query,
        "max_results": max_results,
        "search_domain_order": domains_order,
    }

    try:
        async with session.post(ENDPOINT, headers=headers, json=payload) as resp:
            if resp.status != 200:
                raw_body = await resp.text()
                error_message = extract_api_error_message(raw_body)
                fatal_error_kind = classify_fatal_api_error(
                    resp.status,
                    error_message or raw_body,
                )
                if fatal_error_kind:
                    detail = error_message or "sem detalhe da API"
                    raise APIFatalStopError(
                        f"HTTP {resp.status} | tipo={fatal_error_kind} | "
                        f"{detail} | query='{query[:120]}'"
                    )

                detail_preview = normalize_text(error_message or raw_body)[:220]
                print(
                    f"Erro HTTP {resp.status} para query: {query[:120]} "
                    f"| detalhe: {detail_preview or 'sem detalhe'}"
                )
                return None
            data: Dict[str, Any] = await resp.json()
            links: List[str] = []
            results = data.get("results", [])
            if not isinstance(results, list):
                return []
            for result in results:
                if not isinstance(result, dict):
                    continue
                url = normalize_text(result.get("url"))
                if url and is_valid_juridical_link(url):
                    links.append(url)
            return normalize_links(links, max_links=MAX_LINKS_PER_ROW)
    except APIFatalStopError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao chamar Perplexity: {exc}")
        return None


async def fill_noticia_column() -> None:
    api_key = resolve_perplexity_key("")
    if not api_key:
        raise SystemExit(
            "ERRO: chave da Perplexity ausente. Defina PERPLEXITY_API_KEY "
            "ou preencha Chave_secreta_Perplexity.txt."
        )
    load_cache()

    rows: List[Dict[str, str]] = []
    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        input_fieldnames = list(reader.fieldnames or [])
        if not input_fieldnames:
            raise SystemExit(f"ERRO: CSV sem cabecalho: {INPUT_FILE}")
        fieldnames = build_output_fieldnames(input_fieldnames)
        for raw_row in reader:
            row: Dict[str, str] = {}
            for key in input_fieldnames:
                row[key] = normalize_text(raw_row.get(key, ""))
            set_row_news_links(row, get_row_news_links(row))
            for key in fieldnames:
                row.setdefault(key, "")
            rows.append(row)

    queries: List[Optional[str]] = [None] * len(rows)
    missing_indices: List[int] = []
    candidate_indices: List[int] = []
    for idx, row in enumerate(rows):
        if row_has_news(row):
            continue
        missing_indices.append(idx)
        query = make_query(row)
        queries[idx] = query
        if query:
            candidate_indices.append(idx)

    total_rows = len(rows)
    total_missing = len(missing_indices)
    total_candidates = len(candidate_indices)
    no_query_count = total_missing - total_candidates
    already_with_news = total_rows - total_missing

    input_sig = file_signature(INPUT_FILE)
    checkpoint = read_checkpoint()
    processed_rows: Dict[str, List[str]] = {}
    resumed_with_url = 0

    if checkpoint:
        cp_sig = checkpoint.get("input_signature", {})
        cp_version = int(checkpoint.get("version", 0) or 0)
        cp_total_rows = int(checkpoint.get("total_rows", -1) or -1)
        cp_total_candidates = int(checkpoint.get("total_candidates", -1) or -1)
        if (
            cp_version in (1, CHECKPOINT_VERSION)
            and cp_total_rows == total_rows
            and cp_total_candidates == total_candidates
            and isinstance(cp_sig, dict)
            and same_signature(cp_sig, input_sig)
        ):
            processed_rows = normalize_processed_rows(
                checkpoint.get("processed_rows", {}),
                max_rows=total_rows,
                allowed_indices=set(candidate_indices),
            )
            for idx_str, links in processed_rows.items():
                idx = int(idx_str)
                set_row_news_links(rows[idx], links)
                if links:
                    resumed_with_url += 1
        else:
            print("Checkpoint ignorado por incompatibilidade com o CSV atual.")

    pending_indices = [idx for idx in candidate_indices if str(idx) not in processed_rows]

    print("=" * 72)
    print("TRF1 CSV -> CSV via Perplexity (com checkpoint/resume)")
    print(f"Linhas totais: {total_rows}")
    print(f"Ja com noticia no CSV de entrada: {already_with_news}")
    print(f"Sem noticia no CSV de entrada: {total_missing}")
    print(f"Sem dados para query (nao chama API): {no_query_count}")
    print(f"Elegiveis para chamada de API: {total_candidates}")
    if processed_rows:
        print(
            f"Checkpoint retomado: {len(processed_rows)}/{total_candidates} "
            f"({percent(len(processed_rows), total_candidates):.2f}%)"
        )
        print(f"Links restaurados do checkpoint: {resumed_with_url}")
    else:
        print("Checkpoint: sem progresso reaproveitavel.")
    print(f"Restante para este run: {len(pending_indices)}")
    print("=" * 72)

    if total_candidates == 0:
        persist_snapshot(
            rows=rows,
            fieldnames=fieldnames,
            input_sig=input_sig,
            total_candidates=total_candidates,
            processed_rows=processed_rows,
        )
        print(f"Nenhuma chamada de API necessaria. Arquivo atualizado: {OUTPUT_FILE}")
        print(f"Checkpoint atualizado: {CHECKPOINT_FILE}")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def bounded_perplexity_lookup(
        session: aiohttp.ClientSession,
        query: str,
    ) -> Optional[List[str]]:
        cached = CACHE.get(query)
        if cached is not None:
            return cached
        async with semaphore:
            result = await perplexity_lookup(
                session,
                query,
                api_key=api_key,
                max_results=MAX_LINKS_PER_ROW,
            )
        if result is not None:
            CACHE[query] = normalize_links(result, max_links=MAX_LINKS_PER_ROW)
        return result

    start_run = time.time()
    completed_this_run = 0
    cache_hits_total = 0
    api_calls_total = 0
    with_url_total = 0
    without_url_total = 0
    retryable_errors_total = 0

    total_batches = (len(pending_indices) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

    try:
        async with aiohttp.ClientSession() as session:
            for start in range(0, len(pending_indices), MAX_BATCH_SIZE):
                batch_number = (start // MAX_BATCH_SIZE) + 1
                batch_indices = pending_indices[start : start + MAX_BATCH_SIZE]

                print(
                    f"Lote {batch_number}/{total_batches} | linhas {start + 1}-"
                    f"{start + len(batch_indices)} de {len(pending_indices)} pendentes"
                )

                batch_results: Dict[int, Any] = {}
                jobs = []
                task_by_query: Dict[str, asyncio.Task] = {}
                batch_cache_hits = 0

                for idx in batch_indices:
                    query = queries[idx]
                    if not query:
                        batch_results[idx] = []
                        continue
                    if query in CACHE:
                        batch_results[idx] = CACHE.get(query, [])
                        batch_cache_hits += 1
                        continue
                    task = task_by_query.get(query)
                    if task is None:
                        task = asyncio.create_task(bounded_perplexity_lookup(session, query))
                        task_by_query[query] = task
                    jobs.append((idx, query, task))

                api_calls_batch = len(task_by_query)
                if task_by_query:
                    gathered = await asyncio.gather(
                        *task_by_query.values(),
                        return_exceptions=True,
                    )
                    result_by_task = {
                        task: result for task, result in zip(task_by_query.values(), gathered)
                    }
                    for idx, _, task in jobs:
                        batch_results[idx] = result_by_task.get(task)

                batch_ok_with_url = 0
                batch_ok_without_url = 0
                batch_errors = 0
                fatal_stop_error: Optional[APIFatalStopError] = None

                for idx in batch_indices:
                    result = batch_results.get(idx)
                    if isinstance(result, APIFatalStopError):
                        if fatal_stop_error is None:
                            fatal_stop_error = result
                            print(f"  - ERRO FATAL DA API detectado na linha {idx + 1}: {result}")
                        continue
                    if isinstance(result, Exception):
                        batch_errors += 1
                        print(f"  - Falha linha {idx + 1}: {result}")
                        continue
                    if result is None:
                        batch_errors += 1
                        continue

                    links = normalize_links(result, max_links=MAX_LINKS_PER_ROW)
                    if links:
                        set_row_news_links(rows[idx], links)
                        batch_ok_with_url += 1
                    else:
                        set_row_news_links(rows[idx], [])
                        batch_ok_without_url += 1

                    processed_rows[str(idx)] = links

                completed_batch = batch_ok_with_url + batch_ok_without_url
                completed_this_run += completed_batch
                cache_hits_total += batch_cache_hits
                api_calls_total += api_calls_batch
                with_url_total += batch_ok_with_url
                without_url_total += batch_ok_without_url
                retryable_errors_total += batch_errors

                persist_snapshot(
                    rows=rows,
                    fieldnames=fieldnames,
                    input_sig=input_sig,
                    total_candidates=total_candidates,
                    processed_rows=processed_rows,
                )

                processed_total = len(processed_rows)
                remaining = max(0, total_candidates - processed_total)
                elapsed = max(0.001, time.time() - start_run)
                rate = completed_this_run / elapsed
                eta = format_duration(remaining / rate) if rate > 0 else "N/D"

                print(
                    "Lote concluido | "
                    f"api={api_calls_batch} cache={batch_cache_hits} "
                    f"ok_url={batch_ok_with_url} ok_sem_url={batch_ok_without_url} "
                    f"erros_retry={batch_errors}"
                )
                print(
                    f"Progresso: {processed_total}/{total_candidates} "
                    f"({percent(processed_total, total_candidates):.2f}%) | "
                    f"Restante: {remaining} | ETA: {eta}"
                )

                if fatal_stop_error is not None:
                    print(
                        "Interrompendo processamento: API reportou erro fatal "
                        "(autorizacao/creditos/quota)."
                    )
                    raise fatal_stop_error

                if start + MAX_BATCH_SIZE < len(pending_indices) and DELAY_BETWEEN_BATCHES_SEC > 0:
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES_SEC)

    except APIFatalStopError as exc:
        print(
            "\nPROCESSAMENTO INTERROMPIDO: erro fatal da API Perplexity "
            "(autorizacao/creditos/quota)."
        )
        print("Salvando snapshot/checkpoint antes de encerrar...")
        persist_snapshot(
            rows=rows,
            fieldnames=fieldnames,
            input_sig=input_sig,
            total_candidates=total_candidates,
            processed_rows=processed_rows,
        )
        elapsed_total = time.time() - start_run
        processed_total = len(processed_rows)
        remaining = max(0, total_candidates - processed_total)
        remaining_batches = (remaining + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        output_with_news = sum(1 for row in rows if row_has_news(row))
        missing_output = total_rows - output_with_news
        print(f"Motivo da API: {exc}")
        print(
            f"Resumo da interrupcao: processado={processed_total}/{total_candidates} "
            f"({percent(processed_total, total_candidates):.2f}%) "
            f"| faltam={remaining} elegiveis para API "
            f"| lotes_restantes~{remaining_batches}"
        )
        print(
            f"CSV parcial: com_noticia={output_with_news}/{total_rows} "
            f"| sem_noticia={missing_output}/{total_rows}"
        )
        print(f"Tempo ate interrupcao: {format_duration(elapsed_total)}")
        print(f"Checkpoint preservado em: {CHECKPOINT_FILE}")
        print("Recarregue creditos e execute novamente para retomar do checkpoint.")
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nInterrupcao detectada. Salvando snapshot para retomada...")
        persist_snapshot(
            rows=rows,
            fieldnames=fieldnames,
            input_sig=input_sig,
            total_candidates=total_candidates,
            processed_rows=processed_rows,
        )
        processed_total = len(processed_rows)
        remaining = max(0, total_candidates - processed_total)
        print(
            f"Checkpoint salvo. Progresso preservado: {processed_total}/{total_candidates} "
            f"| restante={remaining}"
        )
        return
    except Exception:
        print("\nFalha inesperada. Salvando snapshot antes de encerrar...")
        persist_snapshot(
            rows=rows,
            fieldnames=fieldnames,
            input_sig=input_sig,
            total_candidates=total_candidates,
            processed_rows=processed_rows,
        )
        raise

    persist_snapshot(
        rows=rows,
        fieldnames=fieldnames,
        input_sig=input_sig,
        total_candidates=total_candidates,
        processed_rows=processed_rows,
    )

    elapsed_total = time.time() - start_run
    processed_total = len(processed_rows)
    remaining = max(0, total_candidates - processed_total)
    output_with_news = sum(1 for row in rows if row_has_news(row))

    print("=" * 72)
    print(f"Tempo total: {format_duration(elapsed_total)}")
    print(
        "Resumo do run: "
        f"api={api_calls_total} cache={cache_hits_total} "
        f"ok_url={with_url_total} ok_sem_url={without_url_total} "
        f"erros_retry={retryable_errors_total}"
    )
    print(
        f"Checkpoint atual: {processed_total}/{total_candidates} "
        f"({percent(processed_total, total_candidates):.2f}%) | restante={remaining}"
    )
    print(f"CSV final com noticia preenchida: {output_with_news}/{total_rows}")
    print(f"Arquivo gerado/atualizado: {OUTPUT_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    if remaining > 0:
        print("Ainda ha pendencias retryable. Rode o script novamente para retomar.")
    else:
        print("Processamento concluido sem pendencias.")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(fill_noticia_column())
