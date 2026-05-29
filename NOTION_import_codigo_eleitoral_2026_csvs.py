#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import requests


NOTION_BASE_URL = "https://api.notion.com"
DEFAULT_NOTION_VERSION = "2026-03-11"
DEFAULT_TARGET_PAGE_URL = "https://www.notion.so/leis-eleitorais-36f721955c64802fa202d847daba84b3"
DEFAULT_CSV_DIR = Path("Codigo_Eleitoral_2026_csv_normas")
DEFAULT_CHECKPOINT = DEFAULT_CSV_DIR / ".notion_import_checkpoint.json"
DEFAULT_DATABASE_TITLE = "Codigo Eleitoral 2026 anotado - dispositivos"

CSV_COLUMNS = [
    "id",
    "ordem_doc",
    "arquivo_fonte",
    "norma_id",
    "norma_titulo",
    "norma_nome_popular",
    "macrogrupo",
    "grupo_sumario",
    "pagina_inicial",
    "pagina_final",
    "hierarquia_normativa",
    "tipo_dispositivo",
    "dispositivo",
    "dispositivo_pai",
    "dispositivo_pai_id",
    "dispositivo_pai_row_key",
    "artigo",
    "paragrafo",
    "inciso",
    "alinea",
    "item",
    "texto_dispositivo",
    "notas_texto",
    "notas_json",
    "qtd_notas",
    "referencias_normativas",
    "referencias_jurisprudenciais",
    "referencias_sumulas",
    "texto_rag",
    "resumo_curto",
    "palavras_chave",
    "incluir_no_rag",
    "chars",
    "tokens_estimados",
]

NUMBER_COLUMNS = {
    "id",
    "ordem_doc",
    "pagina_inicial",
    "pagina_final",
    "qtd_notas",
    "dispositivo_pai_id",
    "chars",
    "tokens_estimados",
}
CHECKBOX_COLUMNS = {"incluir_no_rag"}
SELECT_COLUMNS = {"tipo_dispositivo"}
EXTRA_COLUMNS = ["row_key", "csv_arquivo"]
TITLE_PROPERTY = "Nome"

LOGGER = logging.getLogger("notion_codigo_eleitoral_import")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\ufeff", " ")).strip()


def normalize_notion_id(value: str) -> str:
    raw = re.sub(r"[^0-9A-Fa-f]", "", str(value or ""))
    if len(raw) != 32:
        raise ValueError(f"ID Notion invalido: {value}")
    raw = raw.lower()
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def extract_notion_id_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme and not parsed.netloc:
        return normalize_notion_id(url)
    candidates = re.findall(r"[0-9A-Fa-f]{32}", parsed.path or "")
    if candidates:
        return normalize_notion_id(candidates[-1])
    for values in parse_qs(parsed.query or "").values():
        for value in values:
            found = re.findall(r"[0-9A-Fa-f]{32}", value or "")
            if found:
                return normalize_notion_id(found[-1])
    found_any = re.findall(r"[0-9A-Fa-f]{32}", url or "")
    if found_any:
        return normalize_notion_id(found_any[-1])
    raise ValueError(f"Nao foi possivel extrair ID Notion da URL: {url}")


def notion_url_from_id(page_id: str) -> str:
    return f"https://www.notion.so/{page_id.replace('-', '')}"


def read_secret_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def resolve_notion_token(explicit_token: str = "") -> str:
    if explicit_token:
        return explicit_token.strip()
    for env_name in ("NOTION_API_KEY", "NOTION_TOKEN", "NOTION_SECRET"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    for candidate in (
        Path("Chave_Notion.txt"),
        Path.home() / "Chave_Notion.txt",
        Path.home() / ".notion_token",
    ):
        value = read_secret_file(candidate)
        if value:
            return value
    return ""


def text_object(content: str) -> Dict[str, Any]:
    return {"type": "text", "text": {"content": content}}


def chunk_text(text: Any, *, max_chars: int) -> List[Dict[str, Any]]:
    clean = str(text or "").replace("\x00", "")
    if max_chars > 0 and len(clean) > max_chars:
        marker = "\n[TRUNCADO NO IMPORTADOR NOTION]"
        clean = clean[: max(0, max_chars - len(marker))] + marker
    if not clean:
        return []
    return [text_object(clean[i : i + 2000]) for i in range(0, len(clean), 2000)]


def title_property(text: str) -> Dict[str, Any]:
    title = normalize_ws(text)[:1900] or "Dispositivo sem titulo"
    return {"title": [text_object(title)]}


def rich_text_property(text: Any, *, max_chars: int) -> Dict[str, Any]:
    return {"rich_text": chunk_text(text, max_chars=max_chars)}


def number_property(text: Any) -> Dict[str, Any]:
    raw = normalize_ws(text)
    if not raw:
        return {"number": None}
    normalized = raw.replace(".", "").replace(",", ".") if re.search(r"\d,\d", raw) else raw
    try:
        value = float(normalized)
    except ValueError:
        return {"number": None}
    if value.is_integer():
        value = int(value)
    return {"number": value}


def checkbox_property(text: Any) -> Dict[str, Any]:
    raw = normalize_ws(text).lower()
    return {"checkbox": raw in {"1", "true", "t", "yes", "y", "sim", "s"}}


def select_property(text: Any) -> Dict[str, Any]:
    raw = normalize_ws(text).replace(",", " ")
    return {"select": {"name": raw[:100]} if raw else None}


def schema_for_import() -> Dict[str, Any]:
    properties: Dict[str, Any] = {TITLE_PROPERTY: {"title": {}}}
    for name in EXTRA_COLUMNS + CSV_COLUMNS:
        if name in NUMBER_COLUMNS:
            properties[name] = {"number": {"format": "number"}}
        elif name in CHECKBOX_COLUMNS:
            properties[name] = {"checkbox": {}}
        elif name in SELECT_COLUMNS:
            properties[name] = {"select": {"options": []}}
        else:
            properties[name] = {"rich_text": {}}
    return properties


class NotionClient:
    def __init__(
        self,
        token: str,
        *,
        notion_version: str,
        timeout_s: int,
        max_retries: int,
        rate_rps: float,
    ) -> None:
        self.timeout_s = max(5, int(timeout_s))
        self.max_retries = max(0, int(max_retries))
        self.min_interval_s = 1.0 / rate_rps if rate_rps > 0 else 0.0
        self.next_request_at = 0.0
        self.pace_lock = threading.Lock()
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": notion_version,
            "Content-Type": "application/json",
        }
        self.thread_local = threading.local()

    def session_for_thread(self) -> requests.Session:
        session = getattr(self.thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self.headers)
            self.thread_local.session = session
        return session

    def pace(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self.pace_lock:
            now = time.monotonic()
            sleep_for = self.next_request_at - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self.next_request_at = max(now, time.monotonic()) + self.min_interval_s

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = path if path.startswith("http") else f"{NOTION_BASE_URL}{path}"
        attempts = self.max_retries + 1
        last_message = ""
        for attempt in range(1, attempts + 1):
            self.pace()
            try:
                response = self.session_for_thread().request(
                    method.upper(),
                    url,
                    json=dict(json_body) if json_body is not None else None,
                    params=dict(params) if params is not None else None,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_message = str(exc)
                if attempt < attempts:
                    time.sleep(min(20.0, 0.8 * 2 ** (attempt - 1)))
                    continue
                raise RuntimeError(f"Falha de rede no Notion: {last_message}") from exc

            if 200 <= response.status_code < 300:
                if not response.text.strip():
                    return {}
                payload = response.json()
                return payload if isinstance(payload, dict) else {}

            last_message = extract_error_message(response)
            retry_after = response.headers.get("Retry-After")
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt < attempts:
                delay = retry_delay(attempt, retry_after)
                LOGGER.warning("Notion %s %s retornou %s. Retry em %.2fs: %s", method, path, response.status_code, delay, last_message)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Erro Notion {response.status_code} em {method.upper()} {path}: {last_message}")
        raise RuntimeError(f"Falha no Notion: {last_message or 'erro desconhecido'}")


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return normalize_ws(payload.get("message") or payload.get("error") or response.text)
    except Exception:
        pass
    return normalize_ws(response.text)


def retry_delay(attempt: int, retry_after: Optional[str]) -> float:
    if retry_after:
        try:
            parsed = float(retry_after)
            if parsed > 0:
                return min(30.0, parsed)
        except ValueError:
            pass
    return min(30.0, 0.8 * 2 ** max(0, attempt - 1))


def read_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        raise RuntimeError(f"Checkpoint invalido em {path}: {exc}") from exc


def write_checkpoint(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def iter_csv_rows(csv_dir: Path) -> Iterable[Dict[str, str]]:
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {csv_dir}")
    for csv_path in files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            missing = [name for name in CSV_COLUMNS if name not in fieldnames]
            if missing:
                raise RuntimeError(f"{csv_path} nao tem colunas esperadas: {', '.join(missing)}")
            for row in reader:
                normalized = {name: row.get(name, "") for name in CSV_COLUMNS}
                normalized["csv_arquivo"] = csv_path.name
                normalized["row_key"] = f"{csv_path.stem}:{normalize_ws(row.get('id')) or 'sem-id'}"
                yield normalized


def load_rows(csv_dir: Path, *, limit: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in iter_csv_rows(csv_dir):
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def build_row_title(row: Mapping[str, str]) -> str:
    norma = normalize_ws(row.get("norma_nome_popular")) or normalize_ws(row.get("norma_titulo")) or normalize_ws(row.get("norma_id"))
    dispositivo = normalize_ws(row.get("dispositivo")) or normalize_ws(row.get("tipo_dispositivo")) or f"linha {normalize_ws(row.get('id'))}"
    return " - ".join(part for part in (norma, dispositivo) if part) or normalize_ws(row.get("row_key")) or "Dispositivo"


def build_page_properties(row: Mapping[str, str], *, max_rich_text_chars: int) -> Dict[str, Any]:
    properties: Dict[str, Any] = {TITLE_PROPERTY: title_property(build_row_title(row))}
    for name in EXTRA_COLUMNS + CSV_COLUMNS:
        raw = row.get(name, "")
        if name in NUMBER_COLUMNS:
            properties[name] = number_property(raw)
        elif name in CHECKBOX_COLUMNS:
            properties[name] = checkbox_property(raw)
        elif name in SELECT_COLUMNS:
            properties[name] = select_property(raw)
        else:
            properties[name] = rich_text_property(raw, max_chars=max_rich_text_chars)
    return properties


def create_database(client: NotionClient, parent_page_id: str, title: str) -> Tuple[str, str]:
    schema = schema_for_import()
    title_rich_text = [text_object(title)]
    body = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": title_rich_text,
        "is_inline": True,
        "initial_data_source": {
            "title": title_rich_text,
            "properties": schema,
        },
    }
    payload = client.request("POST", "/v1/databases", json_body=body)
    database_id = normalize_notion_id(str(payload.get("id", "")))
    data_source_id = extract_data_source_id(client, database_id, payload)
    return database_id, data_source_id


def extract_data_source_id(client: NotionClient, database_id: str, database_payload: Optional[Mapping[str, Any]] = None) -> str:
    payload = dict(database_payload or {})
    if not payload:
        payload = client.request("GET", f"/v1/databases/{database_id}")
    candidates: List[str] = []
    raw_data_sources = payload.get("data_sources")
    if isinstance(raw_data_sources, list):
        for item in raw_data_sources:
            if isinstance(item, dict) and item.get("id"):
                candidates.append(str(item["id"]))
    elif isinstance(raw_data_sources, dict):
        for key in ("results", "items", "data_sources"):
            value = raw_data_sources.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("id"):
                        candidates.append(str(item["id"]))
    if candidates:
        return normalize_notion_id(candidates[0])

    refreshed = client.request("GET", f"/v1/databases/{database_id}")
    raw_data_sources = refreshed.get("data_sources")
    if isinstance(raw_data_sources, list):
        for item in raw_data_sources:
            if isinstance(item, dict) and item.get("id"):
                return normalize_notion_id(str(item["id"]))
    raise RuntimeError("Nao foi possivel identificar data_source_id da base criada.")


def create_page(client: NotionClient, data_source_id: str, properties: Mapping[str, Any]) -> str:
    payload = client.request(
        "POST",
        "/v1/pages",
        json_body={
            "parent": {"type": "data_source_id", "data_source_id": data_source_id},
            "properties": dict(properties),
        },
    )
    return normalize_notion_id(str(payload.get("id", "")))


def property_plain_text(prop: Mapping[str, Any]) -> str:
    prop_type = normalize_ws(prop.get("type"))
    values = prop.get(prop_type)
    if prop_type in {"rich_text", "title"} and isinstance(values, list):
        parts: List[str] = []
        for item in values:
            if isinstance(item, dict):
                parts.append(str(item.get("plain_text") or ((item.get("text") or {}).get("content")) or ""))
        return "".join(parts)
    return ""


def sync_existing_pages_by_row_key(client: NotionClient, data_source_id: str) -> Dict[str, str]:
    existing: Dict[str, str] = {}
    cursor = ""
    while True:
        body: Dict[str, Any] = {"page_size": 100, "result_type": "page"}
        if cursor:
            body["start_cursor"] = cursor
        payload = client.request("POST", f"/v1/data_sources/{data_source_id}/query", json_body=body)
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict) or item.get("object") != "page":
                continue
            props = item.get("properties") or {}
            if not isinstance(props, dict):
                continue
            row_key = normalize_ws(property_plain_text(props.get("row_key", {}) or {}))
            page_id = normalize_ws(item.get("id"))
            if row_key and page_id:
                existing[row_key] = normalize_notion_id(page_id)
        if not payload.get("has_more"):
            break
        cursor = normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return existing


def summarize_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    csv_files = {row.get("csv_arquivo", "") for row in rows}
    normas = {row.get("norma_id", "") for row in rows}
    return {"rows": len(rows), "csv_files": len(csv_files), "normas": len(normas)}


def import_pending_rows_concurrently(
    client: NotionClient,
    rows: Sequence[Mapping[str, str]],
    *,
    data_source_id: str,
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    created_pages: Dict[str, str],
    max_rich_text_chars: int,
    checkpoint_every: int,
    workers: int,
) -> Tuple[int, int]:
    pending = [
        (index, row)
        for index, row in enumerate(rows, start=1)
        if normalize_ws(row.get("row_key")) not in created_pages
    ]
    skipped = len(rows) - len(pending)
    imported = 0
    submitted = 0
    in_flight: Dict[Future[str], Tuple[int, str]] = {}
    max_in_flight = max(1, workers * 3)

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal submitted
        while submitted < len(pending) and len(in_flight) < max_in_flight:
            index, row = pending[submitted]
            submitted += 1
            row_key = normalize_ws(row.get("row_key"))
            props = build_page_properties(row, max_rich_text_chars=max_rich_text_chars)
            future = executor.submit(create_page, client, data_source_id, props)
            in_flight[future] = (index, row_key)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_next(executor)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                index, row_key = in_flight.pop(future)
                try:
                    page_id = future.result()
                except Exception:
                    checkpoint["created_count"] = len(created_pages)
                    checkpoint["updated_at"] = now_iso()
                    write_checkpoint(checkpoint_path, checkpoint)
                    raise
                created_pages[row_key] = page_id
                imported += 1
                checkpoint["last_index"] = max(int(checkpoint.get("last_index") or 0), index)
                checkpoint["created_count"] = len(created_pages)
                checkpoint["updated_at"] = now_iso()
                if imported % checkpoint_every == 0 or imported == len(pending):
                    write_checkpoint(checkpoint_path, checkpoint)
                    LOGGER.info(
                        "Progresso: %s/%s linhas registradas, %s criadas nesta execucao, %s ignoradas por checkpoint/sync.",
                        len(created_pages),
                        len(rows),
                        imported,
                        skipped,
                    )
            submit_next(executor)
    return imported, skipped


def import_rows(
    client: NotionClient,
    rows: Sequence[Mapping[str, str]],
    *,
    checkpoint_path: Path,
    target_page_id: str,
    database_title: str,
    max_rich_text_chars: int,
    checkpoint_every: int,
    workers: int,
    sync_existing: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    checkpoint = read_checkpoint(checkpoint_path)
    if not checkpoint:
        checkpoint = {
            "started_at": now_iso(),
            "target_page_id": target_page_id,
            "database_title": database_title,
            "created_pages": {},
            "errors": [],
        }

    created_pages = checkpoint.setdefault("created_pages", {})
    if not isinstance(created_pages, dict):
        raise RuntimeError("Campo created_pages do checkpoint precisa ser um objeto.")

    if "database_id" not in checkpoint or "data_source_id" not in checkpoint:
        if dry_run:
            LOGGER.info("[dry-run] Criaria database '%s' sob pagina %s", database_title, target_page_id)
            checkpoint["database_id"] = ""
            checkpoint["data_source_id"] = ""
        else:
            LOGGER.info("Criando database '%s' sob pagina %s", database_title, target_page_id)
            database_id, data_source_id = create_database(client, target_page_id, database_title)
            checkpoint["database_id"] = database_id
            checkpoint["data_source_id"] = data_source_id
            checkpoint["database_url"] = notion_url_from_id(database_id)
            checkpoint["data_source_url"] = notion_url_from_id(data_source_id)
            checkpoint["updated_at"] = now_iso()
            write_checkpoint(checkpoint_path, checkpoint)
            LOGGER.info("Database criado: %s", checkpoint["database_url"])

    data_source_id = normalize_ws(checkpoint.get("data_source_id"))
    if sync_existing and data_source_id and not dry_run:
        LOGGER.info("Sincronizando paginas ja existentes pelo row_key antes de retomar...")
        existing = sync_existing_pages_by_row_key(client, data_source_id)
        new_keys = {key: value for key, value in existing.items() if key not in created_pages}
        if new_keys:
            created_pages.update(new_keys)
            checkpoint["created_count"] = len(created_pages)
            checkpoint["updated_at"] = now_iso()
            write_checkpoint(checkpoint_path, checkpoint)
        LOGGER.info("Sync concluido: %s row_keys encontrados na database (%s novos no checkpoint).", len(existing), len(new_keys))

    if workers > 1 and not dry_run:
        imported, skipped = import_pending_rows_concurrently(
            client,
            rows,
            data_source_id=data_source_id,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            created_pages=created_pages,
            max_rich_text_chars=max_rich_text_chars,
            checkpoint_every=checkpoint_every,
            workers=workers,
        )
        checkpoint["rows_total"] = len(rows)
        checkpoint["created_count"] = len(created_pages)
        checkpoint["completed_at"] = now_iso()
        write_checkpoint(checkpoint_path, checkpoint)
        LOGGER.info("Importacao concorrente finalizada: %s criadas, %s ignoradas.", imported, skipped)
        return checkpoint

    imported = 0
    skipped = 0
    total = len(rows)

    for index, row in enumerate(rows, start=1):
        row_key = normalize_ws(row.get("row_key"))
        if row_key in created_pages:
            skipped += 1
            continue

        props = build_page_properties(row, max_rich_text_chars=max_rich_text_chars)
        if dry_run:
            page_id = f"dry-run-{index}"
        else:
            page_id = create_page(client, data_source_id, props)
        created_pages[row_key] = page_id
        imported += 1
        checkpoint["last_index"] = index
        checkpoint["created_count"] = len(created_pages)
        checkpoint["updated_at"] = now_iso()

        if imported % checkpoint_every == 0 or index == total:
            write_checkpoint(checkpoint_path, checkpoint)
            LOGGER.info(
                "Progresso: %s/%s linhas processadas, %s criadas nesta execucao, %s ignoradas por checkpoint.",
                index,
                total,
                imported,
                skipped,
            )

    checkpoint["rows_total"] = total
    checkpoint["created_count"] = len(created_pages)
    checkpoint["completed_at"] = now_iso()
    write_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa CSVs do Codigo Eleitoral 2026 anotado para uma database Notion.")
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--target-page-url", default=DEFAULT_TARGET_PAGE_URL)
    parser.add_argument("--database-title", default=DEFAULT_DATABASE_TITLE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--notion-token", default="")
    parser.add_argument("--notion-version", default=DEFAULT_NOTION_VERSION)
    parser.add_argument("--rate-rps", type=float, default=2.5)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--max-rich-text-chars", type=int, default=30000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sync-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    target_page_id = extract_notion_id_from_url(args.target_page_url)
    rows = load_rows(args.csv_dir, limit=max(0, int(args.limit or 0)))
    summary = summarize_rows(rows)
    LOGGER.info("CSV dir: %s", args.csv_dir)
    LOGGER.info("Linhas: %s | CSVs: %s | normas: %s", summary["rows"], summary["csv_files"], summary["normas"])
    if not rows:
        raise RuntimeError("Nenhuma linha carregada dos CSVs.")

    token = resolve_notion_token(args.notion_token)
    if not token and not args.dry_run:
        raise RuntimeError("Token do Notion nao encontrado. Configure NOTION_TOKEN ou Chave_Notion.txt.")
    client = NotionClient(
        token or "dry-run",
        notion_version=args.notion_version,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        rate_rps=max(0.1, float(args.rate_rps or 0.0)),
    )
    checkpoint = import_rows(
        client,
        rows,
        checkpoint_path=args.checkpoint,
        target_page_id=target_page_id,
        database_title=args.database_title,
        max_rich_text_chars=max(2000, int(args.max_rich_text_chars or 0)),
        checkpoint_every=max(1, int(args.checkpoint_every or 1)),
        workers=max(1, int(args.workers or 1)),
        sync_existing=bool(args.sync_existing),
        dry_run=bool(args.dry_run),
    )
    LOGGER.info("Concluido. Database: %s", checkpoint.get("database_url") or checkpoint.get("database_id") or "<dry-run>")
    LOGGER.info("Checkpoint: %s", args.checkpoint)
    LOGGER.info("Paginas criadas/registradas: %s", checkpoint.get("created_count", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
