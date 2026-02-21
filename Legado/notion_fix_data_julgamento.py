#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrige a propriedade `data_julgamento` no Notion.

Fluxo:
1) Lê os valores textuais (ex.: "26 de janeiro de 2026").
2) Converte para ISO (YYYY-MM-DD), formato aceito pela API de Date.
3) Opcionalmente converte a propriedade para tipo `date`.
4) Atualiza apenas a coluna alvo nas páginas.

Por segurança, roda em dry-run por padrão. Use --apply para efetivar.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

NOTION_API_BASE = "https://api.notion.com/v1"
DEFAULT_DATABASE_ID = "30d721955c6480679e7ae359ff1c19e7"
DEFAULT_PROPERTY_NAME = "data_julgamento"
DEFAULT_NOTION_VERSION = "2022-06-28"
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

MONTHS_PT = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


@dataclass
class PageDateCandidate:
    page_id: str
    page_url: str
    raw_text: str
    iso_date: str


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def strip_accents(value: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", value or "")
        if unicodedata.category(ch) != "Mn"
    )


def read_secret_from_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return ""
    first = raw.splitlines()[0].strip()
    if "=" in first:
        first = first.split("=", 1)[1].strip()
    return first.strip("\"' ")


def property_plain_text(prop: Dict[str, Any]) -> str:
    if not isinstance(prop, dict):
        return ""

    p_type = prop.get("type")
    if p_type in {"rich_text", "title"}:
        arr = prop.get(p_type, [])
        if not isinstance(arr, list):
            return ""
        parts: List[str] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            plain = item.get("plain_text")
            if isinstance(plain, str):
                parts.append(plain)
                continue
            text_obj = item.get("text")
            if isinstance(text_obj, dict) and isinstance(text_obj.get("content"), str):
                parts.append(text_obj["content"])
        return normalize_ws("".join(parts))

    if p_type == "date":
        d = prop.get("date")
        if isinstance(d, dict):
            return normalize_ws(d.get("start", ""))
        return ""

    if p_type == "formula":
        f = prop.get("formula", {})
        if not isinstance(f, dict):
            return ""
        f_type = f.get("type")
        if f_type == "string":
            return normalize_ws(f.get("string", ""))
        if f_type == "date":
            d = f.get("date")
            if isinstance(d, dict):
                return normalize_ws(d.get("start", ""))
        return ""

    if p_type == "select":
        s = prop.get("select")
        if isinstance(s, dict):
            return normalize_ws(s.get("name", ""))
        return ""

    if p_type == "url":
        return normalize_ws(prop.get("url", ""))

    return ""


def parse_ptbr_date_to_iso(raw_value: str) -> Tuple[str, str]:
    """
    Retorna (iso, status) com status em:
    - parsed
    - empty
    - unparsed
    """
    raw = normalize_ws(raw_value).replace("º", "")
    if not raw:
        return "", "empty"

    # Já está em ISO.
    ymd = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", raw)
    if ymd:
        y, m, d = map(int, ymd.groups())
        try:
            return date(y, m, d).isoformat(), "parsed"
        except ValueError:
            return "", "unparsed"

    # Formato numérico com dia primeiro: DD/MM/AAAA ou DD-MM-AAAA.
    dmy = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", raw)
    if dmy:
        d, m, y = map(int, dmy.groups())
        if y < 100:
            y = 2000 + y if y <= 30 else 1900 + y
        try:
            return date(y, m, d).isoformat(), "parsed"
        except ValueError:
            return "", "unparsed"

    # Formato textual: 26 de janeiro de 2026
    folded = strip_accents(raw.casefold())
    textual = re.fullmatch(r"(\d{1,2})\s+de\s+([a-z]+)\s+de\s+(\d{4})", folded)
    if textual:
        d_s, month_name, y_s = textual.groups()
        month = MONTHS_PT.get(month_name, 0)
        if not month:
            return "", "unparsed"
        d = int(d_s)
        y = int(y_s)
        try:
            return date(y, month, d).isoformat(), "parsed"
        except ValueError:
            return "", "unparsed"

    return "", "unparsed"


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
        base_sleep_s: float = 0.45,
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
                wait = (2**attempt) * base_sleep_s + random.uniform(0.0, 0.3)
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                snippet = (resp.text or "")[:1000]
                last_err = f"http_{resp.status_code}: {snippet}"
                if resp.status_code in TRANSIENT_STATUS and attempt < retries:
                    wait = (2**attempt) * base_sleep_s + random.uniform(0.0, 0.3)
                    time.sleep(wait)
                    continue
                return None, last_err, resp.status_code

            try:
                return resp.json(), None, resp.status_code
            except ValueError:
                return None, "invalid_json_response", resp.status_code

        return None, last_err or "unknown_error", last_status


def iter_database_pages(
    client: NotionClient,
    database_id: str,
    page_size: int,
    max_pages: int,
) -> Iterable[Dict[str, Any]]:
    start_cursor: Optional[str] = None
    yielded = 0
    while True:
        payload: Dict[str, Any] = {"page_size": min(max(1, page_size), 100)}
        if start_cursor:
            payload["start_cursor"] = start_cursor
        data, err, _ = client.request_json("POST", f"/databases/{database_id}/query", payload=payload)
        if err or data is None:
            raise RuntimeError(f"Falha ao consultar database: {err}")

        for obj in data.get("results", []):
            if isinstance(obj, dict):
                yield obj
                yielded += 1
                if max_pages > 0 and yielded >= max_pages:
                    return

        if not data.get("has_more"):
            return
        start_cursor = normalize_ws(data.get("next_cursor", "")) or None


def write_report_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "page_id",
        "page_url",
        "status",
        "raw_text",
        "iso_date",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Corrige `data_julgamento` no Notion, convertendo texto pt-BR para Date."
    )
    p.add_argument("--database-id", default=DEFAULT_DATABASE_ID)
    p.add_argument("--property-name", default=DEFAULT_PROPERTY_NAME)
    p.add_argument("--notion-key-file", default="Chave_Notion.txt")
    p.add_argument("--notion-version", default=DEFAULT_NOTION_VERSION)
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=0)
    p.add_argument("--sleep-ms", type=int, default=120)
    p.add_argument(
        "--skip-property-type-conversion",
        action="store_true",
        help="Não tenta converter a propriedade para tipo date.",
    )
    p.add_argument(
        "--report-file",
        default="",
        help="Caminho do CSV de auditoria (default: reports/data_julgamento_fix_<timestamp>.csv).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Efetiva mudanças. Sem essa flag, roda somente dry-run.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    token = read_secret_from_file(args.notion_key_file)
    if not token:
        raise SystemExit(f"Não consegui ler token do Notion em {args.notion_key_file}")

    client = NotionClient(token=token, notion_version=args.notion_version)

    db, err, _ = client.request_json("GET", f"/databases/{args.database_id}")
    if err or db is None:
        raise SystemExit(f"Falha ao carregar schema do database: {err}")

    props = db.get("properties", {})
    if not isinstance(props, dict) or args.property_name not in props:
        raise SystemExit(f"Propriedade não encontrada no database: {args.property_name}")

    prop_meta = props.get(args.property_name, {})
    if not isinstance(prop_meta, dict):
        raise SystemExit(f"Schema inválido para propriedade: {args.property_name}")

    current_type = normalize_ws(prop_meta.get("type", ""))
    property_id = normalize_ws(prop_meta.get("id", ""))
    if not property_id:
        raise SystemExit(f"Não consegui ler o ID da propriedade: {args.property_name}")

    print(
        f"Database={args.database_id} | propriedade={args.property_name} | "
        f"tipo_atual={current_type} | apply={args.apply}"
    )

    candidates: List[PageDateCandidate] = []
    report_rows: List[Dict[str, str]] = []
    scanned = 0
    parsed = 0
    empty = 0
    unparsed = 0
    already_date = 0

    for page in iter_database_pages(
        client=client,
        database_id=args.database_id,
        page_size=args.page_size,
        max_pages=args.max_pages,
    ):
        scanned += 1
        page_id = normalize_ws(page.get("id", ""))
        page_url = normalize_ws(page.get("url", ""))
        props_page = page.get("properties", {})
        if not isinstance(props_page, dict):
            continue
        prop = props_page.get(args.property_name, {})
        if not isinstance(prop, dict):
            continue

        ptype = normalize_ws(prop.get("type", ""))

        # Se já for date preenchida, não precisa mexer.
        if ptype == "date":
            d = prop.get("date")
            if isinstance(d, dict) and normalize_ws(d.get("start", "")):
                already_date += 1
                report_rows.append(
                    {
                        "page_id": page_id,
                        "page_url": page_url,
                        "status": "already_date",
                        "raw_text": normalize_ws(d.get("start", "")),
                        "iso_date": normalize_ws(d.get("start", "")),
                        "error": "",
                    }
                )
                continue

        raw_text = property_plain_text(prop)
        iso_date, status = parse_ptbr_date_to_iso(raw_text)
        if status == "parsed":
            parsed += 1
            candidates.append(
                PageDateCandidate(
                    page_id=page_id,
                    page_url=page_url,
                    raw_text=raw_text,
                    iso_date=iso_date,
                )
            )
            report_rows.append(
                {
                    "page_id": page_id,
                    "page_url": page_url,
                    "status": "parsed",
                    "raw_text": raw_text,
                    "iso_date": iso_date,
                    "error": "",
                }
            )
        elif status == "empty":
            empty += 1
            report_rows.append(
                {
                    "page_id": page_id,
                    "page_url": page_url,
                    "status": "empty",
                    "raw_text": "",
                    "iso_date": "",
                    "error": "",
                }
            )
        else:
            unparsed += 1
            report_rows.append(
                {
                    "page_id": page_id,
                    "page_url": page_url,
                    "status": "unparsed",
                    "raw_text": raw_text,
                    "iso_date": "",
                    "error": "formato_nao_reconhecido",
                }
            )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_file) if args.report_file else Path("reports") / f"data_julgamento_fix_{timestamp}.csv"
    write_report_csv(report_path, report_rows)

    print(
        json.dumps(
            {
                "scanned": scanned,
                "parsed": parsed,
                "empty": empty,
                "unparsed": unparsed,
                "already_date": already_date,
                "to_update": len(candidates),
                "report_file": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not args.apply:
        print("Dry-run concluído. Use --apply para efetivar as mudanças.")
        return 0

    if current_type != "date":
        if args.skip_property_type_conversion:
            raise SystemExit(
                f"A propriedade está como '{current_type}', não date. "
                "Remova --skip-property-type-conversion para converter via API antes de atualizar."
            )

        # Converte a propriedade para date usando o ID estável.
        payload_by_id = {
            "properties": {
                property_id: {
                    "name": args.property_name,
                    "date": {},
                }
            }
        }
        _, err, _ = client.request_json("PATCH", f"/databases/{args.database_id}", payload=payload_by_id, retries=2)
        if err:
            # Fallback: algumas workspaces aceitam melhor por nome da propriedade.
            payload_by_name = {"properties": {args.property_name: {"date": {}}}}
            _, err, _ = client.request_json(
                "PATCH",
                f"/databases/{args.database_id}",
                payload=payload_by_name,
                retries=2,
            )
            if err:
                raise SystemExit(f"Falha ao converter propriedade para date: {err}")
        print(f"Propriedade '{args.property_name}' convertida para date.")
        time.sleep(1.5)

    updated = 0
    failed = 0
    for idx, cand in enumerate(candidates, start=1):
        patch = {
            "properties": {
                args.property_name: {
                    "date": {"start": cand.iso_date},
                }
            }
        }
        _, err, _ = client.request_json("PATCH", f"/pages/{cand.page_id}", payload=patch)
        if err:
            failed += 1
        else:
            updated += 1

        if idx % 100 == 0 or idx == len(candidates):
            print(f"Progresso updates: {idx}/{len(candidates)} | ok={updated} | fail={failed}")
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(
        json.dumps(
            {"updated": updated, "failed": failed, "total_candidates": len(candidates)},
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
