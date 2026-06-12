#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Importa CSVs DJe ja tratados para a base Notion usada pelos relatorios.

Este script complementa o conversor SJUR_csv_to_csv_NOTIONfriendly_v2.py:
1. le o CSV final tratado;
2. normaliza os tipos conforme o schema real da base Notion;
3. cria ou atualiza paginas pelo numeroUnico/numeroProcesso.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import NOTION_relatoriodeIA_v2 as report
import SJUR_csv_to_csv_NOTIONfriendly_v2 as sjur
from Artefatos.scripts.project_layout import default_report_path
from Artefatos.scripts.openai_log_utils import configure_standard_logging
from Artefatos.scripts.openai_progress_utils import utc_now_iso, write_json_atomic


DEFAULT_DATABASE_URL = report.DEFAULT_SOURCE_DATABASE_URL
SCRIPT_STEM = Path(__file__).stem
IMPORT_REPORT_FILE = default_report_path(SCRIPT_STEM)

PROPERTY_ALIASES = {
    "relatores": "relator",
}
MAX_SELECT_NAME_CHARS = 100
DEFAULT_MAX_RICH_TEXT_CHARS = 9000
HIGH_CARDINALITY_MULTI_SELECT_COLUMNS = {"partes", "advogados"}
LIST_DISPLAY_TEXT_COLUMNS = {"partes", "advogados", "partes_texto", "advogados_texto"}
NEWS_URL_COLUMNS = {"noticia_TSE", "noticia_TRE", "noticia_geral_1", "noticia_geral_2"}
TEXT_MIRROR_PROPERTY_SOURCES = {
    "partes_texto": "partes",
    "advogados_texto": "advogados",
}


@dataclass
class ImportResult:
    csv_path: str
    rows_seen: int
    rows_skipped: int
    created: int
    updated: int
    unchanged: int
    dry_run: bool
    start_date_iso: str
    end_date_iso: str
    report_path: str
    skipped_new_high_cardinality_options: Dict[str, int] = field(default_factory=dict)


def _normalize_ws(value: Any) -> str:
    return report._normalize_ws(value)


def _normalize_list_display_text(value: Any) -> str:
    raw = _normalize_ws(value)
    if not raw:
        return ""
    try:
        values = sjur.split_multiselect_values(raw)
    except Exception:
        values = re.split(r"\s*[,;]\s*", raw)
    return ", ".join(_unique(str(item or "") for item in values))


def _unique(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in items:
        item = _normalize_ws(raw)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _digits(value: Any) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def _csv_date_order_from_headers(headers: Sequence[str]) -> str:
    normalized = {PROPERTY_ALIASES.get(str(name or ""), str(name or "")) for name in headers}
    if {"tema", "punchline", "relator"} & normalized:
        return "mdy"
    if "numeroProtocolo" in normalized or "relatores" in normalized:
        return "dmy"
    return "auto"


def parse_csv_date(value: Any, *, date_order: str = "auto") -> Optional[date]:
    raw = _normalize_ws(value)
    if not raw:
        return None
    raw = raw[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass

    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if not match:
        return None
    first, second, year = [int(part) for part in match.groups()]
    order = date_order
    if order == "auto":
        if first > 12:
            order = "dmy"
        elif second > 12:
            order = "mdy"
        else:
            order = "mdy"
    day, month = (first, second) if order == "dmy" else (second, first)
    try:
        return date(year, month, day)
    except ValueError:
        return None


def read_csv_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    encoding, delimiter = sjur.detect_csv_format(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        headers = [str(name or "") for name in (reader.fieldnames or [])]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            row: Dict[str, str] = {}
            for key, value in (raw or {}).items():
                clean_key = PROPERTY_ALIASES.get(str(key or ""), str(key or ""))
                if clean_key in LIST_DISPLAY_TEXT_COLUMNS:
                    row[clean_key] = _normalize_list_display_text(value)
                else:
                    row[clean_key] = str(value or "")
            rows.append(row)
    if not headers:
        raise RuntimeError(f"CSV sem cabecalho valido: {path}")
    return headers, rows


def _rich_text_items(value: Any, *, max_chars: int) -> List[Dict[str, Any]]:
    text = _normalize_ws(value)
    if max_chars > 0:
        text = text[:max_chars].rstrip()
    return [{"type": "text", "text": {"content": part}} for part in report._chunk_text_for_notion(text)]


def _title_property(value: Any) -> Dict[str, Any]:
    text = _normalize_ws(value) or "DJe/TSE"
    return {"title": _rich_text_items(text, max_chars=0)}


def _rich_text_property(value: Any, *, max_chars: int) -> Optional[Dict[str, Any]]:
    text = _normalize_ws(value)
    if not text:
        return None
    return {"rich_text": _rich_text_items(text, max_chars=max_chars)}


def _select_property(value: Any) -> Optional[Dict[str, Any]]:
    text = _normalize_ws(value)
    if not text:
        return None
    text = re.sub(r"\s*,\s*", "; ", text)
    return {"select": {"name": text[:MAX_SELECT_NAME_CHARS]}}


def _multi_select_values(value: Any) -> List[str]:
    raw = _normalize_ws(value)
    if not raw:
        return []
    try:
        values = sjur.split_multiselect_values(raw)
    except Exception:
        values = re.split(r"\s*[,;]\s*", raw)
    return _unique(str(item or "")[:MAX_SELECT_NAME_CHARS] for item in values)


def _existing_multi_select_option_names(prop: Mapping[str, Any]) -> set[str]:
    raw_options = ((prop.get("multi_select") or {}) if isinstance(prop, Mapping) else {}).get("options") or []
    names: set[str] = set()
    for option in raw_options:
        if not isinstance(option, Mapping):
            continue
        name = _normalize_ws(option.get("name"))
        if name:
            names.add(name)
    return names


def _multi_select_property(
    value: Any,
    *,
    existing_options: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    values = _multi_select_values(value)
    if existing_options is not None:
        values = [item for item in values if item in existing_options]
    if not values:
        return None
    return {"multi_select": [{"name": item} for item in values]}


def _number_property(value: Any) -> Optional[Dict[str, Any]]:
    digits = _digits(value)
    if not digits:
        return None
    try:
        return {"number": int(digits)}
    except ValueError:
        return None


def _date_property(value: Any, *, date_order: str) -> Optional[Dict[str, Any]]:
    parsed = parse_csv_date(value, date_order=date_order)
    if parsed is None:
        return None
    return {"date": {"start": parsed.isoformat()}}


def _normalize_news_url_for_property(name: str, value: Any) -> str:
    text = _normalize_ws(value)
    if not re.match(r"^https?://", text, flags=re.IGNORECASE):
        return ""
    if sjur.is_editorially_weak_news_url(text):
        return ""
    if name == "noticia_TSE":
        normalized = sjur._normalize_tse_news_url(text) or ""
        if normalized and sjur.official_news_url_live_status(normalized) is False:
            return ""
        return normalized
    if name == "noticia_TRE":
        normalized = sjur._normalize_tre_news_url(text) or ""
        if normalized and sjur.official_news_url_live_status(normalized) is False:
            return ""
        return normalized
    if name.startswith("noticia_geral"):
        if (sjur._is_tse_news_url(text) or sjur._is_tre_news_url(text)) and sjur.official_news_url_live_status(text) is False:
            return ""
        return text
    return text


def _url_property(value: Any, *, property_name: str = "") -> Optional[Dict[str, Any]]:
    text = _normalize_ws(value)
    if not re.match(r"^https?://", text, flags=re.IGNORECASE):
        return None
    if property_name in NEWS_URL_COLUMNS:
        normalized = _normalize_news_url_for_property(property_name, text)
        return {"url": normalized or None}
    return {"url": text}


def ensure_high_cardinality_text_properties(
    data_source_id: str,
    schema: Mapping[str, Mapping[str, Any]],
    *,
    logger: Any,
    dry_run: bool,
) -> Mapping[str, Mapping[str, Any]]:
    to_convert = {
        prop_name: {"rich_text": {}}
        for prop_name in HIGH_CARDINALITY_MULTI_SELECT_COLUMNS
        if _normalize_ws((schema.get(prop_name) or {}).get("type")) == "multi_select"
    }
    if not to_convert:
        return schema
    logger.warning(
        "Convertendo propriedades de alta cardinalidade para rich_text no Notion: %s",
        ", ".join(sorted(to_convert)),
    )
    if dry_run:
        patched_schema = dict(schema)
        for prop_name in to_convert:
            patched_schema[prop_name] = {"type": "rich_text", "rich_text": {}}
        return patched_schema
    report.notion_request(
        "PATCH",
        f"/v1/data_sources/{data_source_id}",
        json_body={"properties": to_convert},
    )
    data_source = report.notion_request("GET", f"/v1/data_sources/{data_source_id}")
    refreshed_schema = data_source.get("properties") or {}
    if not isinstance(refreshed_schema, dict) or not refreshed_schema:
        raise RuntimeError("Schema da base Notion nao encontrado apos criacao das propriedades textuais.")
    return refreshed_schema


def _row_title(row: Mapping[str, str]) -> str:
    title = _normalize_ws(row.get("tema"))
    if title:
        return title
    parts = [
        _normalize_ws(row.get("siglaClasse") or row.get("descricaoClasse")),
        _normalize_ws(row.get("numeroUnico") or row.get("numeroProcesso")),
    ]
    return " - ".join(part for part in parts if part) or "DJe/TSE"


def build_page_properties(
    row: Mapping[str, str],
    schema: Mapping[str, Mapping[str, Any]],
    *,
    date_order: str,
    max_rich_text_chars: int,
) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    high_cardinality_options = {
        name: _existing_multi_select_option_names(prop)
        for name, prop in schema.items()
        if name in HIGH_CARDINALITY_MULTI_SELECT_COLUMNS and _normalize_ws(prop.get("type")) == "multi_select"
    }
    for name, prop in schema.items():
        prop_type = _normalize_ws(prop.get("type"))
        if name == "tema":
            properties[name] = _title_property(_row_title(row))
            continue

        source_name = TEXT_MIRROR_PROPERTY_SOURCES.get(name, name)
        raw = row.get(source_name, "")
        value: Optional[Dict[str, Any]] = None
        if prop_type == "rich_text":
            value = _rich_text_property(raw, max_chars=max_rich_text_chars)
        elif prop_type == "number":
            value = _number_property(raw)
        elif prop_type == "date":
            value = _date_property(raw, date_order=date_order)
        elif prop_type == "select":
            value = _select_property(raw)
        elif prop_type == "multi_select":
            value = _multi_select_property(
                raw,
                existing_options=high_cardinality_options.get(name)
                if name in HIGH_CARDINALITY_MULTI_SELECT_COLUMNS
                else None,
            )
        elif prop_type == "url":
            value = _url_property(raw, property_name=name)

        if value is not None:
            properties[name] = value
    return properties


def count_new_high_cardinality_options(
    rows: Sequence[Mapping[str, str]],
    schema: Mapping[str, Mapping[str, Any]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name in HIGH_CARDINALITY_MULTI_SELECT_COLUMNS:
        prop = schema.get(name, {})
        if _normalize_ws(prop.get("type")) != "multi_select":
            continue
        existing = _existing_multi_select_option_names(prop)
        new_values: set[str] = set()
        for row in rows:
            for value in _multi_select_values(row.get(name, "")):
                if value and value not in existing:
                    new_values.add(value)
        counts[name] = len(new_values)
    return counts


# A unidade real da base e a DECISAO: um processo pode ter varias decisoes em
# datas distintas e MAIS DE UMA decisao no mesmo dia (ex.: decisao + despacho).
# Chave composta numero+dataDecisao; o emparelhamento linha<->pagina e feito por
# LISTA (cada linha do CSV consome uma pagina da chave; esgotou, cria nova).
def row_unique_keys(row: Mapping[str, str], decision_date_iso: str = "") -> List[str]:
    suffix = f"|{decision_date_iso}" if decision_date_iso else ""
    keys: List[str] = []
    numero_unico = _digits(row.get("numeroUnico"))
    if numero_unico:
        keys.append(f"numeroUnico:{numero_unico}{suffix}")
    numero_processo = _digits(row.get("numeroProcesso"))
    if numero_processo:
        keys.append(f"numeroProcesso:{numero_processo}{suffix}")
    return keys


def _page_unique_keys(page_obj: Mapping[str, Any]) -> List[str]:
    props = page_obj.get("properties") or {}
    numero_unico = _digits(report._property_rich_text(props.get("numeroUnico", {})))
    numero_processo = _digits(report._property_number_text(props.get("numeroProcesso", {})))
    # [:10] normaliza datetimes eventuais ('YYYY-MM-DDT...') para casar com o
    # isoformat() de date usado na chave da linha.
    decision_date_iso = report._property_date_start(props.get("dataDecisao", {}))[:10]
    suffix = f"|{decision_date_iso}" if decision_date_iso else ""
    keys: List[str] = []
    if numero_unico:
        keys.append(f"numeroUnico:{numero_unico}{suffix}")
    if numero_processo:
        keys.append(f"numeroProcesso:{numero_processo}{suffix}")
    return keys


def _row_date_range(rows: Sequence[Mapping[str, str]], *, date_order: str) -> tuple[str, str]:
    dates = [
        parsed
        for row in rows
        for parsed in [parse_csv_date(row.get("dataDecisao"), date_order=date_order)]
        if parsed is not None
    ]
    if not dates:
        return "", ""
    return min(dates).isoformat(), max(dates).isoformat()


def query_existing_pages_by_period(data_source_id: str, start_iso: str, end_iso: str) -> Dict[str, str]:
    if not start_iso or not end_iso:
        return {}
    existing: Dict[str, List[str]] = {}
    cursor = ""
    while True:
        body: Dict[str, Any] = {
            "page_size": 100,
            "filter": {
                "and": [
                    {"property": "dataDecisao", "date": {"on_or_after": start_iso}},
                    {"property": "dataDecisao", "date": {"on_or_before": end_iso}},
                ]
            },
            # Ordem deterministica: o emparelhamento linha<->pagina e posicional,
            # entao reimports precisam ver as paginas na mesma ordem de criacao.
            "sorts": [{"timestamp": "created_time", "direction": "ascending"}],
        }
        if cursor:
            body["start_cursor"] = cursor
        payload = report.notion_request("POST", f"/v1/data_sources/{data_source_id}/query", json_body=body)
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict) or item.get("object") != "page":
                continue
            page_id = report._normalize_notion_id(str(item.get("id", "")))
            for key in _page_unique_keys(item):
                existing.setdefault(key, []).append(page_id)
        if not payload.get("has_more"):
            break
        cursor = _normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return existing


def create_page(data_source_id: str, properties: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        "parent": {"type": "data_source_id", "data_source_id": data_source_id},
        "properties": dict(properties),
    }
    return report.notion_request("POST", "/v1/pages", json_body=payload)


def update_page(page_id: str, properties: Mapping[str, Any]) -> Dict[str, Any]:
    return report.notion_request("PATCH", f"/v1/pages/{page_id}", json_body={"properties": dict(properties)})


def import_csv_to_notion(
    csv_path: Path,
    *,
    database_url: str,
    mode: str,
    date_order: str,
    max_rich_text_chars: int,
    dry_run: bool,
    limit: int,
    logger: Any,
) -> ImportResult:
    headers, rows = read_csv_rows(csv_path)
    effective_date_order = _csv_date_order_from_headers(headers) if date_order == "auto" else date_order
    if limit > 0:
        rows = rows[:limit]
    start_iso, end_iso = _row_date_range(rows, date_order=effective_date_order)

    database_id = report.extract_notion_id_from_url(database_url)
    data_source_id = report.retrieve_database_and_datasource_id(database_id)
    data_source = report.notion_request("GET", f"/v1/data_sources/{data_source_id}")
    schema = data_source.get("properties") or {}
    if not isinstance(schema, dict) or not schema:
        raise RuntimeError("Schema da base Notion nao encontrado.")
    schema = ensure_high_cardinality_text_properties(
        data_source_id,
        schema,
        logger=logger,
        dry_run=dry_run,
    )
    skipped_high_cardinality = count_new_high_cardinality_options(rows, schema)
    for name, skipped_count in sorted(skipped_high_cardinality.items()):
        if skipped_count <= 0:
            continue
        logger.warning(
            "Coluna %s e multi_select de alta cardinalidade; %d opcao(oes) nova(s) "
            "nao serao criadas para evitar estouro do schema do Notion.",
            name,
            skipped_count,
        )

    # Linhas com mesma chave processo+data E mesmo teor sao a MESMA decisao
    # repetida (exports do DJe com periodos sobrepostos) — mantem so a primeira.
    # Teores diferentes no mesmo dia (decisao + despacho) continuam separados.
    seen_fingerprints: set = set()
    deduped_rows: List[Dict[str, str]] = []
    rows_deduped = 0
    for row in rows:
        parsed = parse_csv_date(row.get("dataDecisao"), date_order=effective_date_order)
        teor = _normalize_ws(row.get("textoDecisao"))
        fingerprint = "|".join(
            [
                _digits(row.get("numeroUnico")) or _digits(row.get("numeroProcesso")),
                parsed.isoformat() if parsed else "",
                hashlib.sha256(teor.encode("utf-8")).hexdigest(),
            ]
        )
        if teor and fingerprint in seen_fingerprints:
            rows_deduped += 1
            continue
        seen_fingerprints.add(fingerprint)
        deduped_rows.append(row)
    rows = deduped_rows
    if rows_deduped:
        logger.info("Linhas identicas (processo+data+teor) descartadas do CSV: %d", rows_deduped)

    existing = query_existing_pages_by_period(data_source_id, start_iso, end_iso)
    logger.info(
        "Importacao CSV -> Notion | arquivo=%s | linhas=%d | periodo=%s a %s | existentes_no_periodo=%d",
        csv_path,
        len(rows),
        start_iso or "?",
        end_iso or "?",
        len({pid for ids in existing.values() for pid in ids}),
    )

    created = 0
    updated = 0
    unchanged = 0
    skipped = 0
    errors: List[Dict[str, Any]] = []
    # Cada linha do CSV consome no maximo UMA pagina da chave (processo+data);
    # linhas excedentes criam paginas novas — assim multiplas decisoes do mesmo
    # dia coexistem sem se sobrescreverem.
    consumed_ids: set = set()
    for index, row in enumerate(rows, start=1):
        parsed_date = parse_csv_date(row.get("dataDecisao"), date_order=effective_date_order)
        keys = row_unique_keys(row, decision_date_iso=(parsed_date.isoformat() if parsed_date else ""))
        if not keys or parsed_date is None:
            skipped += 1
            errors.append({"row": index, "reason": "sem chave unica ou dataDecisao valida", "key": keys[0] if keys else ""})
            continue
        properties = build_page_properties(
            row,
            schema,
            date_order=effective_date_order,
            max_rich_text_chars=max_rich_text_chars,
        )
        page_id = next(
            (pid for key in keys for pid in existing.get(key, []) if pid not in consumed_ids),
            "",
        )
        if page_id:
            consumed_ids.add(page_id)
            if mode == "create-only":
                unchanged += 1
                continue
            logger.info("[%d/%d] Atualizando %s (%s)", index, len(rows), row.get("numeroUnico") or row.get("numeroProcesso"), page_id)
            if not dry_run:
                update_page(page_id, properties)
            updated += 1
        else:
            if mode == "update-only":
                skipped += 1
                continue
            logger.info("[%d/%d] Criando %s", index, len(rows), row.get("numeroUnico") or row.get("numeroProcesso"))
            if not dry_run:
                created_page = create_page(data_source_id, properties)
                created_id = report._normalize_notion_id(str(created_page.get("id", "")))
                if created_id:
                    consumed_ids.add(created_id)
            created += 1

    payload = {
        "script": Path(__file__).name,
        "finished_at_utc": utc_now_iso(),
        "csv_path": str(csv_path.resolve()),
        "database_url": database_url,
        "data_source_id": data_source_id,
        "date_order": effective_date_order,
        "rows_seen": len(rows),
        "rows_skipped": skipped,
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "dry_run": dry_run,
        "start_date_iso": start_iso,
        "end_date_iso": end_iso,
        "errors": errors[:50],
        "skipped_new_high_cardinality_options": skipped_high_cardinality,
    }
    write_json_atomic(IMPORT_REPORT_FILE, payload)
    return ImportResult(
        csv_path=str(csv_path.resolve()),
        rows_seen=len(rows),
        rows_skipped=skipped,
        created=created,
        updated=updated,
        unchanged=unchanged,
        dry_run=dry_run,
        start_date_iso=start_iso,
        end_date_iso=end_iso,
        report_path=str(IMPORT_REPORT_FILE),
        skipped_new_high_cardinality_options=skipped_high_cardinality,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Importa CSV tratado do DJe/TSE para a base Notion.")
    parser.add_argument("--csv", required=True, help="CSV tratado para importar.")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL, help="URL da base Notion.")
    parser.add_argument("--mode", choices=("upsert", "create-only", "update-only"), default="upsert")
    parser.add_argument(
        "--date-order",
        choices=("auto", "dmy", "mdy"),
        default="auto",
        help="Ordem das datas com barra. CSV tratado pelo conversor costuma usar mdy.",
    )
    parser.add_argument("--max-rich-text-chars", type=int, default=DEFAULT_MAX_RICH_TEXT_CHARS)
    parser.add_argument("--limit", type=int, default=0, help="Limita linhas para teste.")
    parser.add_argument("--dry-run", action="store_true", help="Nao escreve no Notion.")
    parser.add_argument("--model", default=report.DEFAULT_OPENAI_MODEL, help="Modelo apenas para inicializacao compartilhada.")
    parser.add_argument("--max-cases-per-batch", type=int, default=report.DEFAULT_MAX_CASES_PER_BATCH, help=argparse.SUPPRESS)
    parser.add_argument("--openai-max-workers", type=int, default=report.DEFAULT_OPENAI_MAX_WORKERS, help=argparse.SUPPRESS)
    parser.add_argument("--openai-target-rpm", type=int, default=report.DEFAULT_OPENAI_TARGET_RPM, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-file", default="")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists() or not csv_path.is_file():
        parser.error(f"CSV nao encontrado: {csv_path}")
    if args.max_rich_text_chars < 0:
        parser.error("--max-rich-text-chars nao pode ser negativo.")
    if args.limit < 0:
        parser.error("--limit nao pode ser negativo.")

    logger = configure_standard_logging(
        SCRIPT_STEM,
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
    )
    args.force_openai_fallback = True
    report.initialize_clients(args)
    result = import_csv_to_notion(
        csv_path,
        database_url=args.database_url,
        mode=args.mode,
        date_order=args.date_order,
        max_rich_text_chars=int(args.max_rich_text_chars),
        dry_run=bool(args.dry_run),
        limit=int(args.limit),
        logger=logger,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
