#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrige apenas a propriedade `bullet_points` em bancos do Notion.

Regra aplicada:
- substitui vírgulas internas por " - "
- preserva vírgula separadora entre bullets (ex.: fim de linha antes de próxima linha com "•")
- preserva vírgula inline antes de bullet (", •")

Modo padrão é dry-run. Para aplicar, use --apply.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


NOTION_API_BASE = "https://api.notion.com/v1"
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
TOKEN_BSEP = "__BSEP__"
DEFAULT_NOTION_VERSION = "2022-06-28"


DATABASES_FIELDS = ["database_id", "database_title", "database_url"]
ALTERADAS_FIELDS = [
    "database_id",
    "database_title",
    "page_id",
    "page_url",
    "old_len",
    "new_len",
    "status",
    "preview_before",
    "preview_after",
]
SEM_ALTERACAO_FIELDS = [
    "database_id",
    "database_title",
    "page_id",
    "page_url",
    "len_bullet_points",
]
ERROS_FIELDS = [
    "stage",
    "database_id",
    "database_title",
    "page_id",
    "page_url",
    "error_type",
    "http_status",
    "detail",
]


@dataclass
class RuntimeStats:
    databases_found: int = 0
    databases_processed: int = 0
    pages_lidas: int = 0
    pages_com_bullet_points: int = 0
    pages_alteradas: int = 0
    pages_sem_alteracao: int = 0
    pages_ignoradas_sem_bullet_points: int = 0
    pages_ignoradas_tipo_invalido: int = 0
    pages_update_ok: int = 0
    pages_update_fail: int = 0
    errors_total: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "databases_found": self.databases_found,
            "databases_processed": self.databases_processed,
            "pages_lidas": self.pages_lidas,
            "pages_com_bullet_points": self.pages_com_bullet_points,
            "pages_alteradas": self.pages_alteradas,
            "pages_sem_alteracao": self.pages_sem_alteracao,
            "pages_ignoradas_sem_bullet_points": self.pages_ignoradas_sem_bullet_points,
            "pages_ignoradas_tipo_invalido": self.pages_ignoradas_tipo_invalido,
            "pages_update_ok": self.pages_update_ok,
            "pages_update_fail": self.pages_update_fail,
            "errors_total": self.errors_total,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeStats":
        obj = cls()
        for key in obj.to_dict().keys():
            try:
                setattr(obj, key, int(data.get(key, 0)))
            except Exception:
                setattr(obj, key, 0)
        return obj


class NotionAPI:
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
        retries: int = 3,
        sleep_s: float = 0.34,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
        url = f"{NOTION_API_BASE}{path}"
        last_error = ""
        last_status: Optional[int] = None

        for attempt in range(max(1, retries) + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=payload,
                    timeout=45,
                )
                last_status = resp.status_code
            except requests.RequestException as exc:
                last_error = f"request_exception: {exc}"
                if attempt >= retries:
                    return None, last_error, last_status
                wait = (2 ** attempt) * sleep_s + random.uniform(0.0, 0.25)
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                snippet = (resp.text or "")[:600]
                last_error = f"http_{resp.status_code}: {snippet}"
                if resp.status_code in TRANSIENT_STATUS and attempt < retries:
                    wait = (2 ** attempt) * sleep_s + random.uniform(0.0, 0.25)
                    time.sleep(wait)
                    continue
                return None, last_error, resp.status_code

            try:
                return resp.json(), None, resp.status_code
            except ValueError:
                return None, "invalid_json_response", resp.status_code

        return None, last_error or "unknown_error", last_status


def today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def make_default_prefix() -> str:
    return f"auditoria_notion_bullets_{today_yyyymmdd()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrige vírgulas problemáticas de bullet_points em bancos acessíveis no Notion."
    )
    parser.add_argument(
        "--notion-key-file",
        default="Chave_Notion.txt",
        help="Arquivo com o token da integração do Notion (default: Chave_Notion.txt).",
    )
    parser.add_argument(
        "--notion-version",
        default=DEFAULT_NOTION_VERSION,
        help=f"Versão da API Notion (default: {DEFAULT_NOTION_VERSION}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplica atualização no Notion. Sem essa flag roda em dry-run.",
    )
    parser.add_argument(
        "--audit-prefix",
        default=make_default_prefix(),
        help="Prefixo dos arquivos de auditoria.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Arquivo de checkpoint (default: <audit-prefix>_checkpoint.json).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Tamanho de página na API (default: 100).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Salvar checkpoint e flush de auditoria a cada N páginas (default: 50).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Máximo global de páginas para processar (0 = sem limite).",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.34,
        help="Delay mínimo entre updates ao aplicar (default: 0.34).",
    )
    parser.add_argument(
        "--verbose-audit",
        action="store_true",
        help="Gera também o CSV de páginas sem alteração.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Executa testes locais da normalização e sai.",
    )
    return parser.parse_args()


def clean_preview(value: str, max_chars: int = 240) -> str:
    s = (value or "").replace("\r", " ").replace("\n", "\\n")
    return s[:max_chars]


def read_notion_token(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de chave não encontrado: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"Arquivo de chave vazio: {path}")
    line = raw.splitlines()[0].strip()
    if "=" in line:
        line = line.split("=", 1)[1].strip()
    token = line.strip("\"' ")
    if not token:
        raise RuntimeError(f"Não foi possível ler token em: {path}")
    return token


def next_non_empty_index(lines: Sequence[str], idx: int) -> int:
    for j in range(idx + 1, len(lines)):
        if lines[j].strip():
            return j
    return -1


def normalize_bullet_points(text: str) -> str:
    if text is None:
        return ""
    if not text:
        return ""

    lines = text.splitlines()
    if not lines:
        return text

    out: List[str] = []
    for i, line in enumerate(lines):
        work = line.rstrip()
        next_idx = next_non_empty_index(lines, i)
        next_is_bullet = next_idx >= 0 and lines[next_idx].lstrip().startswith("•")

        # Preserva separador inline ", •".
        work = re.sub(r",(?=\s*•)", TOKEN_BSEP, work)

        # Preserva separador final entre bullets.
        if next_is_bullet and work.strip():
            if re.search(r",\s*$", work):
                work = re.sub(r",\s*$", TOKEN_BSEP, work)
            elif re.search(r"\s-\s*$", work):
                # Se já passou por normalização antiga que trocou a vírgula final por " - ",
                # restaura a vírgula separadora.
                work = re.sub(r"\s-\s*$", TOKEN_BSEP, work)

        work = re.sub(r"\s*,\s*", " - ", work)
        work = work.replace(TOKEN_BSEP, ",")
        out.append(work)

    return "\n".join(out)


def rich_text_to_plain_text(prop: Dict[str, Any]) -> str:
    parts = prop.get("rich_text", [])
    if not isinstance(parts, list):
        return ""
    chunks: List[str] = []
    for piece in parts:
        if not isinstance(piece, dict):
            continue
        chunks.append(str(piece.get("plain_text", "")))
    return "".join(chunks)


def chunk_text_for_notion(text: str, max_len: int = 1900) -> List[str]:
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    remain = text
    while remain:
        if len(remain) <= max_len:
            chunks.append(remain)
            break

        cut = remain.rfind("\n", 0, max_len + 1)
        if cut <= 0:
            cut = max_len
            chunks.append(remain[:cut])
            remain = remain[cut:]
        else:
            # Inclui o \n no chunk para preservar o texto original.
            chunks.append(remain[: cut + 1])
            remain = remain[cut + 1 :]
    return chunks


def make_rich_text_payload(text: str) -> List[Dict[str, Any]]:
    chunks = chunk_text_for_notion(text, max_len=1900)
    out: List[Dict[str, Any]] = []
    for c in chunks:
        out.append(
            {
                "type": "text",
                "text": {
                    "content": c,
                },
            }
        )
    return out


def extract_database_title(database_obj: Dict[str, Any]) -> str:
    title = database_obj.get("title", [])
    if not isinstance(title, list):
        return ""
    parts: List[str] = []
    for piece in title:
        if not isinstance(piece, dict):
            continue
        plain = piece.get("plain_text")
        if plain:
            parts.append(str(plain))
    return "".join(parts).strip()


def ensure_csv_header(path: Path, fieldnames: Sequence[str]) -> None:
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


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> int:
    rows_list = list(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows_list)
    return len(rows_list)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_databases(
    api: NotionAPI,
    page_size: int,
    retries: int,
    sleep_s: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    results: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    seen: set[str] = set()
    start_cursor: Optional[str] = None

    while True:
        payload: Dict[str, Any] = {
            "filter": {"property": "object", "value": "database"},
            "page_size": max(1, min(int(page_size), 100)),
        }
        if start_cursor:
            payload["start_cursor"] = start_cursor

        data, err, status = api.request_json(
            method="POST",
            path="/search",
            payload=payload,
            retries=retries,
            sleep_s=sleep_s,
        )
        if err or data is None:
            errors.append(
                {
                    "stage": "search_databases",
                    "database_id": "",
                    "database_title": "",
                    "page_id": "",
                    "page_url": "",
                    "error_type": "search_failed",
                    "http_status": "" if status is None else str(status),
                    "detail": err or "erro desconhecido",
                }
            )
            break

        for obj in data.get("results", []):
            if not isinstance(obj, dict):
                continue
            if obj.get("object") != "database":
                continue
            db_id = str(obj.get("id", "")).strip()
            if not db_id or db_id in seen:
                continue
            seen.add(db_id)
            results.append(
                {
                    "database_id": db_id,
                    "database_title": extract_database_title(obj),
                    "database_url": str(obj.get("url", "")).strip(),
                }
            )

        if not data.get("has_more"):
            break
        start_cursor = data.get("next_cursor")

    results.sort(key=lambda x: ((x.get("database_title") or "").lower(), x.get("database_id") or ""))
    return results, errors


def fetch_database_schema(
    api: NotionAPI,
    database_id: str,
    retries: int,
    sleep_s: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
    return api.request_json(
        method="GET",
        path=f"/databases/{database_id}",
        payload=None,
        retries=retries,
        sleep_s=sleep_s,
    )


def run_normalization_tests() -> None:
    cases = [
        (
            "• art. 1º, inc. I,\n• segundo bullet",
            "• art. 1º - inc. I,\n• segundo bullet",
        ),
        (
            "• ponto A, • ponto B",
            "• ponto A, • ponto B",
        ),
        (
            "• sem virgulas",
            "• sem virgulas",
        ),
        (
            "• primeiro item,\n\n• segundo item",
            "• primeiro item,\n\n• segundo item",
        ),
        (
            "• já normalizado -\n• seguinte",
            "• já normalizado,\n• seguinte",
        ),
    ]
    for idx, (inp, exp) in enumerate(cases, 1):
        got = normalize_bullet_points(inp)
        if got != exp:
            raise AssertionError(
                f"Falha no teste #{idx}\nEntrada: {inp!r}\nEsperado: {exp!r}\nObtido: {got!r}"
            )
    print("Testes de normalização: OK")


def init_or_load_checkpoint(
    checkpoint_path: Path,
    is_apply: bool,
    notion_version: str,
) -> Dict[str, Any]:
    if checkpoint_path.exists():
        data = read_json(checkpoint_path)
        cp_apply = bool(data.get("apply", False))
        cp_version = str(data.get("notion_version", ""))
        if cp_apply != is_apply:
            raise RuntimeError(
                "Checkpoint incompatível com modo atual (--apply). "
                "Remova o checkpoint ou use o mesmo modo."
            )
        if cp_version and cp_version != notion_version:
            raise RuntimeError(
                "Checkpoint incompatível com notion-version atual. "
                "Remova o checkpoint ou use a mesma versão."
            )
        return data

    return {
        "apply": bool(is_apply),
        "notion_version": notion_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "processed_page_ids": [],
        "db_cursors": {},
        "db_completed": [],
        "stats": RuntimeStats().to_dict(),
    }


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    args = parse_args()

    if args.run_tests:
        run_normalization_tests()
        return

    page_size = max(1, min(int(args.page_size), 100))
    save_every = max(1, int(args.save_every))
    max_pages = max(0, int(args.max_pages))
    sleep_s = max(0.0, float(args.sleep_s))

    audit_prefix = str(args.audit_prefix).strip() or make_default_prefix()
    checkpoint_path = Path(args.checkpoint.strip() or f"{audit_prefix}_checkpoint.json")
    notion_key_path = Path(args.notion_key_file)

    databases_path = Path(f"{audit_prefix}_databases_encontrados.csv")
    alteradas_path = Path(f"{audit_prefix}_paginas_alteradas.csv")
    sem_alteracao_path = Path(f"{audit_prefix}_paginas_sem_alteracao.csv")
    erros_path = Path(f"{audit_prefix}_erros.csv")
    resumo_path = Path(f"{audit_prefix}_resumo.txt")

    token = read_notion_token(notion_key_path)
    api = NotionAPI(token=token, notion_version=args.notion_version)

    checkpoint = init_or_load_checkpoint(
        checkpoint_path=checkpoint_path,
        is_apply=bool(args.apply),
        notion_version=args.notion_version,
    )

    processed_page_ids = set(str(x) for x in checkpoint.get("processed_page_ids", []))
    db_cursors: Dict[str, Optional[str]] = {
        str(k): (None if v in ("", None) else str(v))
        for k, v in dict(checkpoint.get("db_cursors", {})).items()
    }
    db_completed = set(str(x) for x in checkpoint.get("db_completed", []))
    stats = RuntimeStats.from_dict(dict(checkpoint.get("stats", {})))

    # Buffers para append incremental em auditoria.
    buffer_alteradas: List[Dict[str, Any]] = []
    buffer_sem_alteracao: List[Dict[str, Any]] = []
    buffer_erros: List[Dict[str, Any]] = []

    # Descoberta de bancos.
    databases, search_errors = list_databases(
        api=api,
        page_size=page_size,
        retries=3,
        sleep_s=sleep_s,
    )
    stats.databases_found = len(databases)
    if search_errors:
        stats.errors_total += len(search_errors)
        buffer_erros.extend(search_errors)

    # Sempre reescreve bancos encontrados (snapshot atual).
    write_csv_rows(databases_path, DATABASES_FIELDS, databases)

    if not checkpoint_path.exists():
        # Inicializa arquivos de auditoria do run novo.
        ensure_csv_header(alteradas_path, ALTERADAS_FIELDS)
        ensure_csv_header(erros_path, ERROS_FIELDS)
        if args.verbose_audit:
            ensure_csv_header(sem_alteracao_path, SEM_ALTERACAO_FIELDS)

    def flush_audits() -> None:
        nonlocal buffer_alteradas, buffer_sem_alteracao, buffer_erros
        if buffer_alteradas:
            append_csv_rows(alteradas_path, ALTERADAS_FIELDS, buffer_alteradas)
            buffer_alteradas = []
        if args.verbose_audit and buffer_sem_alteracao:
            append_csv_rows(sem_alteracao_path, SEM_ALTERACAO_FIELDS, buffer_sem_alteracao)
            buffer_sem_alteracao = []
        if buffer_erros:
            append_csv_rows(erros_path, ERROS_FIELDS, buffer_erros)
            buffer_erros = []

    def save_checkpoint() -> None:
        payload = {
            "apply": bool(args.apply),
            "notion_version": args.notion_version,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "processed_page_ids": list(processed_page_ids),
            "db_cursors": db_cursors,
            "db_completed": list(db_completed),
            "stats": stats.to_dict(),
        }
        write_json_atomic(checkpoint_path, payload)

    stop_due_to_limit = False
    since_save = 0

    for db in databases:
        db_id = db.get("database_id", "")
        db_title = db.get("database_title", "")
        if not db_id:
            continue
        if db_id in db_completed:
            continue

        # Valida schema antes de paginar, para evitar varrer bancos não relacionados.
        schema, schema_err, schema_status = fetch_database_schema(
            api=api,
            database_id=db_id,
            retries=3,
            sleep_s=sleep_s,
        )
        if schema_err or schema is None:
            stats.errors_total += 1
            buffer_erros.append(
                {
                    "stage": "database_schema",
                    "database_id": db_id,
                    "database_title": db_title,
                    "page_id": "",
                    "page_url": "",
                    "error_type": "schema_fetch_failed",
                    "http_status": "" if schema_status is None else str(schema_status),
                    "detail": schema_err or "erro desconhecido",
                }
            )
            db_completed.add(db_id)
            stats.databases_processed = len(db_completed)
            continue

        props = schema.get("properties", {})
        if not isinstance(props, dict) or "bullet_points" not in props:
            stats.errors_total += 1
            buffer_erros.append(
                {
                    "stage": "database_schema",
                    "database_id": db_id,
                    "database_title": db_title,
                    "page_id": "",
                    "page_url": "",
                    "error_type": "database_without_bullet_points",
                    "http_status": "",
                    "detail": "Database sem propriedade 'bullet_points'; ignorado.",
                }
            )
            db_completed.add(db_id)
            stats.databases_processed = len(db_completed)
            continue

        if not isinstance(props.get("bullet_points"), dict) or props["bullet_points"].get("type") != "rich_text":
            stats.errors_total += 1
            buffer_erros.append(
                {
                    "stage": "database_schema",
                    "database_id": db_id,
                    "database_title": db_title,
                    "page_id": "",
                    "page_url": "",
                    "error_type": "database_bullet_points_type_invalid",
                    "http_status": "",
                    "detail": f"Tipo da propriedade no schema: {props.get('bullet_points', {}).get('type')}",
                }
            )
            db_completed.add(db_id)
            stats.databases_processed = len(db_completed)
            continue

        start_cursor = db_cursors.get(db_id)

        while True:
            if max_pages and stats.pages_lidas >= max_pages:
                stop_due_to_limit = True
                break

            payload: Dict[str, Any] = {"page_size": page_size}
            if start_cursor:
                payload["start_cursor"] = start_cursor

            data, err, status = api.request_json(
                method="POST",
                path=f"/databases/{db_id}/query",
                payload=payload,
                retries=3,
                sleep_s=sleep_s,
            )
            if err or data is None:
                stats.errors_total += 1
                buffer_erros.append(
                    {
                        "stage": "query_database",
                        "database_id": db_id,
                        "database_title": db_title,
                        "page_id": "",
                        "page_url": "",
                        "error_type": "database_query_failed",
                        "http_status": "" if status is None else str(status),
                        "detail": err or "erro desconhecido",
                    }
                )
                db_completed.add(db_id)
                break

            rows = data.get("results", [])
            for page in rows:
                if max_pages and stats.pages_lidas >= max_pages:
                    stop_due_to_limit = True
                    break
                if not isinstance(page, dict):
                    continue

                page_id = str(page.get("id", "")).strip()
                page_url = str(page.get("url", "")).strip()
                if not page_id:
                    continue
                if page_id in processed_page_ids:
                    continue

                stats.pages_lidas += 1
                since_save += 1

                props = page.get("properties", {})
                if not isinstance(props, dict):
                    stats.errors_total += 1
                    buffer_erros.append(
                        {
                            "stage": "page_parse",
                            "database_id": db_id,
                            "database_title": db_title,
                            "page_id": page_id,
                            "page_url": page_url,
                            "error_type": "invalid_properties",
                            "http_status": "",
                            "detail": "Objeto properties ausente ou inválido.",
                        }
                    )
                    processed_page_ids.add(page_id)
                    continue

                bp_prop = props.get("bullet_points")
                if bp_prop is None:
                    stats.pages_ignoradas_sem_bullet_points += 1
                    buffer_erros.append(
                        {
                            "stage": "page_parse",
                            "database_id": db_id,
                            "database_title": db_title,
                            "page_id": page_id,
                            "page_url": page_url,
                            "error_type": "missing_bullet_points_property",
                            "http_status": "",
                            "detail": "Propriedade bullet_points não encontrada na página.",
                        }
                    )
                    processed_page_ids.add(page_id)
                    stats.errors_total += 1
                    continue

                if not isinstance(bp_prop, dict) or bp_prop.get("type") != "rich_text":
                    stats.pages_ignoradas_tipo_invalido += 1
                    buffer_erros.append(
                        {
                            "stage": "page_parse",
                            "database_id": db_id,
                            "database_title": db_title,
                            "page_id": page_id,
                            "page_url": page_url,
                            "error_type": "invalid_bullet_points_type",
                            "http_status": "",
                            "detail": f"Tipo encontrado: {bp_prop.get('type') if isinstance(bp_prop, dict) else type(bp_prop).__name__}",
                        }
                    )
                    processed_page_ids.add(page_id)
                    stats.errors_total += 1
                    continue

                original = rich_text_to_plain_text(bp_prop)
                if not original.strip():
                    stats.pages_ignoradas_sem_bullet_points += 1
                    processed_page_ids.add(page_id)
                    continue

                stats.pages_com_bullet_points += 1
                normalized = normalize_bullet_points(original)

                if normalized != original:
                    stats.pages_alteradas += 1
                    record = {
                        "database_id": db_id,
                        "database_title": db_title,
                        "page_id": page_id,
                        "page_url": page_url,
                        "old_len": len(original),
                        "new_len": len(normalized),
                        "status": "would_update" if not args.apply else "updated",
                        "preview_before": clean_preview(original),
                        "preview_after": clean_preview(normalized),
                    }

                    if args.apply:
                        rich_text_payload = make_rich_text_payload(normalized)
                        patch_payload = {
                            "properties": {
                                "bullet_points": {
                                    "rich_text": rich_text_payload,
                                }
                            }
                        }
                        _, patch_err, patch_status = api.request_json(
                            method="PATCH",
                            path=f"/pages/{page_id}",
                            payload=patch_payload,
                            retries=3,
                            sleep_s=sleep_s,
                        )
                        time.sleep(sleep_s)
                        if patch_err:
                            stats.pages_update_fail += 1
                            stats.errors_total += 1
                            record["status"] = "update_failed"
                            buffer_erros.append(
                                {
                                    "stage": "page_update",
                                    "database_id": db_id,
                                    "database_title": db_title,
                                    "page_id": page_id,
                                    "page_url": page_url,
                                    "error_type": "update_failed",
                                    "http_status": "" if patch_status is None else str(patch_status),
                                    "detail": patch_err,
                                }
                            )
                        else:
                            stats.pages_update_ok += 1
                            record["status"] = "updated"
                    buffer_alteradas.append(record)
                else:
                    stats.pages_sem_alteracao += 1
                    if args.verbose_audit:
                        buffer_sem_alteracao.append(
                            {
                                "database_id": db_id,
                                "database_title": db_title,
                                "page_id": page_id,
                                "page_url": page_url,
                                "len_bullet_points": len(original),
                            }
                        )

                processed_page_ids.add(page_id)

                if since_save >= save_every:
                    flush_audits()
                    save_checkpoint()
                    since_save = 0
                    print(
                        f"[progress] pages_lidas={stats.pages_lidas} "
                        f"alteradas={stats.pages_alteradas} erros={stats.errors_total}"
                    )

            if stop_due_to_limit:
                break

            has_more = bool(data.get("has_more"))
            next_cursor = data.get("next_cursor")
            start_cursor = str(next_cursor) if next_cursor else None
            db_cursors[db_id] = start_cursor

            if not has_more:
                db_completed.add(db_id)
                stats.databases_processed = len(db_completed)
                break

        db_cursors[db_id] = start_cursor
        if stop_due_to_limit:
            break

    flush_audits()
    stats.databases_processed = len(db_completed)
    save_checkpoint()

    resumo_lines = [
        f"Modo: {'APPLY' if args.apply else 'DRY-RUN'}",
        f"Notion Version: {args.notion_version}",
        f"Databases encontrados: {stats.databases_found}",
        f"Databases concluídos: {stats.databases_processed}",
        f"Páginas lidas: {stats.pages_lidas}",
        f"Páginas com bullet_points: {stats.pages_com_bullet_points}",
        f"Páginas alteradas (detectadas): {stats.pages_alteradas}",
        f"Páginas sem alteração: {stats.pages_sem_alteracao}",
        f"Páginas ignoradas sem bullet_points: {stats.pages_ignoradas_sem_bullet_points}",
        f"Páginas ignoradas tipo inválido: {stats.pages_ignoradas_tipo_invalido}",
        f"Updates OK: {stats.pages_update_ok}",
        f"Updates falhos: {stats.pages_update_fail}",
        f"Erros totais: {stats.errors_total}",
        f"Limite max-pages atingido: {'sim' if stop_due_to_limit else 'não'}",
        f"Checkpoint: {checkpoint_path}",
        f"CSV bancos: {databases_path}",
        f"CSV alteradas: {alteradas_path}",
        f"CSV sem alteração: {sem_alteracao_path if args.verbose_audit else '(desligado)'}",
        f"CSV erros: {erros_path}",
    ]
    write_text(resumo_path, "\n".join(resumo_lines) + "\n")

    print("\n".join(resumo_lines))


if __name__ == "__main__":
    main()
