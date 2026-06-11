#!/usr/bin/env python3
from __future__ import annotations

# Script movido da raiz do projeto para conversores/. O bloco abaixo mantem
# imports (Artefatos.*, gui_intuitiva e scripts irmaos) e caminhos relativos
# ancorados na raiz do projeto.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "conversores")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_PROJECT_ROOT)

import argparse
import csv
import json
import logging
import re
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import NOTION_import_codigo_eleitoral_2026_csvs as notion_import
import NOTION_import_resolucoes_consolidado as resolucoes


DEFAULT_CSV = Path("resolucoes_rag_consolidado.csv")
DEFAULT_IMPORT_CHECKPOINT = Path("Artefatos/checkpoints/resolucoes_rag_consolidado_checkpoint.json")
DEFAULT_REPAIR_CHECKPOINT = Path("Artefatos/checkpoints/resolucoes_rag_vigencia_repair_checkpoint.json")

LOGGER = logging.getLogger("notion_resolucoes_vigencia_repair")

RENAMED_TEXT_COLUMNS = {
    "texto_vigente": "texto_redacao_anterior",
    "texto_proposto": "texto_redacao_proposta",
    "texto_consolidado": "texto_redacao_consolidada_proposta",
    "texto_anterior": "texto_fragmento_anterior",
    "texto_novo": "texto_fragmento_novo",
}

NEW_COLUMNS = [
    "texto_em_vigor",
    "classe_vigencia_texto",
    "usar_como_texto_vigente",
    "alerta_vigencia",
    "prioridade_rag",
]

RICH_TEXT_UPDATE_COLUMNS = [
    "texto_dispositivo",
    "texto_em_vigor",
    "texto_redacao_anterior",
    "texto_redacao_proposta",
    "texto_redacao_consolidada_proposta",
    "texto_marcado",
    "texto_fragmento_anterior",
    "texto_fragmento_novo",
    "alteracoes_json",
    "alerta_vigencia",
    "referencias_normativas",
    "referencias_jurisprudenciais",
    "resumo_curto",
    "palavras_chave",
    "texto_rag",
]
SELECT_UPDATE_COLUMNS = ["status_alteracao", "classe_vigencia_texto", "prioridade_rag"]
CHECKBOX_UPDATE_COLUMNS = ["incluir_no_rag", "usar_como_texto_vigente"]
NUMBER_UPDATE_COLUMNS = ["chars", "tokens_estimados"]


def normalize_ws(text: Any) -> str:
    return resolucoes.normalize_ws(text)


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv_rows(csv_path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def value_from(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = normalize_ws(row.get(name))
        if value:
            return value
    return ""


def normalize_bool(value: Any) -> str:
    raw = normalize_ws(value).lower()
    return "true" if raw in {"1", "true", "t", "yes", "y", "sim", "s"} else "false"


def is_comparative(row: Mapping[str, str]) -> bool:
    return normalize_ws(row.get("tipo_linha")) in {"alteracao", "alteracao_granular"} or normalize_ws(row.get("tipo_base")).startswith("comparativo")


def build_summary(row: Mapping[str, str]) -> str:
    tipo_linha = normalize_ws(row.get("tipo_linha"))
    tipo_base = normalize_ws(row.get("tipo_base"))
    norma = normalize_ws(row.get("norma_nome_popular")) or normalize_ws(row.get("norma_titulo"))
    dispositivo = normalize_ws(row.get("dispositivo"))
    hierarchy = normalize_ws(row.get("hierarquia_normativa"))
    status = normalize_ws(row.get("status_alteracao"))
    operation = normalize_ws(row.get("tipo_operacao"))

    if tipo_base == "operacional" and tipo_linha == "dispositivo":
        return resolucoes.limit_text(f"Texto oficial extraido da fonte matriz: {norma}, {hierarchy or dispositivo}.")
    if tipo_linha == "alteracao":
        return resolucoes.limit_text(
            f"Comparativo de alteracao ({status or 'sem status'}) em {norma}, {dispositivo}; redacao anterior e redacao proposta/consolidada ficam separadas e nao sao tratadas como texto vigente autoritativo."
        )
    if tipo_linha == "alteracao_granular":
        return resolucoes.limit_text(
            f"Fragmento granular de auditoria ({operation or 'alteracao'}) em {norma}, {dispositivo}; usar apenas com o contexto do dispositivo comparativo pai."
        )
    if tipo_linha == "estrutura":
        return resolucoes.limit_text(f"Estrutura de navegacao em {norma}: {hierarchy or dispositivo}.")
    if tipo_linha == "norma":
        return resolucoes.limit_text(f"Raiz normativa para navegacao e agrupamento: {normalize_ws(row.get('norma_titulo')) or norma}.")
    return resolucoes.limit_text(f"{norma}, {dispositivo}.")


def build_safe_rag(row: Mapping[str, str]) -> str:
    labels = [
        ("Base", "Resolucoes TSE"),
        ("Norma", row.get("norma_titulo", "")),
        ("Nome popular", row.get("norma_nome_popular", "")),
        ("Grupo", row.get("grupo_resolucao", "")),
        ("Tipo da linha", row.get("tipo_linha", "")),
        ("Tipo do dispositivo", row.get("tipo_dispositivo", "")),
        ("Classe de vigencia", row.get("classe_vigencia_texto", "")),
        ("Usar como texto vigente", "sim" if normalize_bool(row.get("usar_como_texto_vigente")) == "true" else "nao"),
        ("Prioridade RAG", row.get("prioridade_rag", "")),
        ("Dispositivo", row.get("dispositivo", "")),
        ("Hierarquia", row.get("hierarquia_normativa", "")),
        ("Dispositivo pai", row.get("dispositivo_pai", "")),
        ("Row key do pai", row.get("dispositivo_pai_row_key", "")),
        ("Fonte CSV", row.get("fonte_csv", "")),
    ]
    parts = [f"{label}: {normalize_ws(value)}" for label, value in labels if normalize_ws(value)]

    alert = normalize_ws(row.get("alerta_vigencia"))
    if alert:
        parts.extend(["", "Alerta de vigencia:", alert])

    summary = normalize_ws(row.get("resumo_curto"))
    if summary:
        parts.extend(["", "Resumo:", summary])

    refs_norm = normalize_ws(row.get("referencias_normativas"))
    if refs_norm:
        parts.extend(["", "Referencias normativas:", refs_norm])
    refs_juris = normalize_ws(row.get("referencias_jurisprudenciais"))
    if refs_juris:
        parts.extend(["", "Referencias jurisprudenciais:", refs_juris])

    text_em_vigor = resolucoes.normalize_text_block(row.get("texto_em_vigor", ""))
    if text_em_vigor:
        parts.extend(["", "Texto em vigor autoritativo nesta base:", text_em_vigor])

    if is_comparative(row):
        for label, column in [
            ("Redacao anterior/pre-alteracao", "texto_redacao_anterior"),
            ("Redacao proposta", "texto_redacao_proposta"),
            ("Redacao consolidada no comparativo/proposta", "texto_redacao_consolidada_proposta"),
            ("Fragmento anterior", "texto_fragmento_anterior"),
            ("Fragmento novo", "texto_fragmento_novo"),
            ("Texto marcado", "texto_marcado"),
            ("Alteracoes JSON", "alteracoes_json"),
        ]:
            value = resolucoes.normalize_text_block(row.get(column, ""))
            if value:
                parts.extend(["", f"{label}:", value])
    else:
        text = resolucoes.normalize_text_block(row.get("texto_dispositivo", ""))
        if text and text != text_em_vigor:
            parts.extend(["", "Texto do dispositivo/estrutura:", text])

    notes = resolucoes.normalize_text_block(row.get("notas_texto", ""))
    if notes:
        parts.extend(["", "Notas:", notes])
    return "\n".join(parts).strip()


def repair_row(raw_row: Mapping[str, str]) -> Dict[str, str]:
    row = dict(raw_row)
    for old, new in RENAMED_TEXT_COLUMNS.items():
        if new not in row or not normalize_ws(row.get(new)):
            row[new] = row.get(old, "")
        row.pop(old, None)

    tipo_linha = normalize_ws(row.get("tipo_linha"))
    tipo_base = normalize_ws(row.get("tipo_base"))
    original_texto_dispositivo = normalize_ws(row.get("texto_dispositivo"))
    redacao_anterior = value_from(row, "texto_redacao_anterior")
    redacao_proposta = value_from(row, "texto_redacao_proposta")
    redacao_consolidada = value_from(row, "texto_redacao_consolidada_proposta")

    if tipo_base == "operacional" and tipo_linha == "dispositivo":
        row["texto_em_vigor"] = original_texto_dispositivo
        row["classe_vigencia_texto"] = "oficial_em_vigor"
        row["usar_como_texto_vigente"] = "true"
        row["prioridade_rag"] = "principal"
        row["alerta_vigencia"] = ""
        if not normalize_ws(row.get("status_alteracao")):
            row["status_alteracao"] = "vigente_extraido"
        row["incluir_no_rag"] = "true"
    elif tipo_linha == "alteracao":
        row["texto_em_vigor"] = ""
        row["classe_vigencia_texto"] = "comparativo_nao_autoritativo"
        row["usar_como_texto_vigente"] = "false"
        row["prioridade_rag"] = "comparativo"
        row["alerta_vigencia"] = (
            "Linha comparativa. A redacao proposta/consolidada desta linha nao deve ser citada como texto vigente "
            "sem conferencia externa da publicacao oficial e da resolucao alteradora."
        )
        row["texto_dispositivo"] = ""
        row["incluir_no_rag"] = "true"
    elif tipo_linha == "alteracao_granular":
        row["texto_em_vigor"] = ""
        row["classe_vigencia_texto"] = "fragmento_alteracao_nao_autoritativo"
        row["usar_como_texto_vigente"] = "false"
        row["prioridade_rag"] = "auditoria"
        row["alerta_vigencia"] = (
            "Fragmento granular de alteracao. Nao usar como resposta isolada nem como redacao vigente; "
            "usar apenas para auditoria da alteracao dentro do dispositivo pai."
        )
        row["texto_dispositivo"] = ""
        row["incluir_no_rag"] = "false"
    elif tipo_linha == "estrutura":
        row["texto_em_vigor"] = ""
        row["classe_vigencia_texto"] = "estrutura"
        row["usar_como_texto_vigente"] = "false"
        row["prioridade_rag"] = "contexto"
        row["alerta_vigencia"] = ""
        if not normalize_ws(row.get("status_alteracao")):
            row["status_alteracao"] = "estrutura"
        row["incluir_no_rag"] = "true"
    elif tipo_linha == "norma":
        row["texto_em_vigor"] = ""
        row["classe_vigencia_texto"] = "norma"
        row["usar_como_texto_vigente"] = "false"
        row["prioridade_rag"] = "contexto"
        row["alerta_vigencia"] = ""
        if not normalize_ws(row.get("status_alteracao")):
            row["status_alteracao"] = "norma"
        row["incluir_no_rag"] = "true"
    else:
        row["texto_em_vigor"] = ""
        row["classe_vigencia_texto"] = "indefinido"
        row["usar_como_texto_vigente"] = "false"
        row["prioridade_rag"] = "revisar"
        if not normalize_ws(row.get("status_alteracao")):
            row["status_alteracao"] = "indefinido"

    row["resumo_curto"] = build_summary(row)
    row["referencias_normativas"] = resolucoes.extract_normative_refs(
        row.get("texto_em_vigor", ""),
        redacao_anterior,
        redacao_proposta,
        redacao_consolidada,
        row.get("texto_dispositivo", ""),
        row.get("alteracoes_json", ""),
    )
    row["referencias_jurisprudenciais"] = resolucoes.extract_juris_refs(
        row.get("texto_em_vigor", ""),
        redacao_anterior,
        redacao_proposta,
        redacao_consolidada,
        row.get("notas_texto", ""),
    )
    row["texto_rag"] = build_safe_rag(row)
    row["chars"] = str(len(row["texto_rag"]))
    row["tokens_estimados"] = str(resolucoes.estimate_tokens(row["texto_rag"]))
    return row


def repaired_fieldnames(fieldnames: Sequence[str]) -> List[str]:
    result: List[str] = []
    inserted_new = False
    for name in fieldnames:
        if name in RENAMED_TEXT_COLUMNS:
            replacement = RENAMED_TEXT_COLUMNS[name]
            if replacement not in result:
                result.append(replacement)
            if replacement == "texto_redacao_consolidada_proposta" and not inserted_new:
                for column in NEW_COLUMNS:
                    if column not in result:
                        result.append(column)
                inserted_new = True
            continue
        if name not in result:
            result.append(name)
        if name == "texto_dispositivo" and not inserted_new:
            for column in NEW_COLUMNS:
                if column not in result:
                    result.append(column)
            inserted_new = True
    if not inserted_new:
        result.extend(column for column in NEW_COLUMNS if column not in result)
    for column in RICH_TEXT_UPDATE_COLUMNS + SELECT_UPDATE_COLUMNS + CHECKBOX_UPDATE_COLUMNS + NUMBER_UPDATE_COLUMNS:
        if column not in result:
            result.append(column)
    return result


def repair_csv(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]], Dict[str, int]]:
    fieldnames, rows = read_csv_rows(csv_path)
    repaired = [repair_row(row) for row in rows]
    new_fieldnames = repaired_fieldnames(fieldnames)
    write_csv_rows(csv_path, new_fieldnames, repaired)
    return new_fieldnames, repaired, validate_rows(repaired)


def validate_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    totals = {
        "rows": len(rows),
        "status_blank": 0,
        "texto_rag_blank": 0,
        "texto_em_vigor_operacional_blank": 0,
        "texto_em_vigor_comparativo_nonblank": 0,
        "granular_incluir_no_rag_true": 0,
        "usar_vigente_comparativo_true": 0,
    }
    for row in rows:
        if not normalize_ws(row.get("status_alteracao")):
            totals["status_blank"] += 1
        if not normalize_ws(row.get("texto_rag")):
            totals["texto_rag_blank"] += 1
        if normalize_ws(row.get("tipo_base")) == "operacional" and normalize_ws(row.get("tipo_linha")) == "dispositivo":
            if not normalize_ws(row.get("texto_em_vigor")):
                totals["texto_em_vigor_operacional_blank"] += 1
        if is_comparative(row) and normalize_ws(row.get("texto_em_vigor")):
            totals["texto_em_vigor_comparativo_nonblank"] += 1
        if normalize_ws(row.get("tipo_linha")) == "alteracao_granular" and normalize_bool(row.get("incluir_no_rag")) == "true":
            totals["granular_incluir_no_rag_true"] += 1
        if is_comparative(row) and normalize_bool(row.get("usar_como_texto_vigente")) == "true":
            totals["usar_vigente_comparativo_true"] += 1
    return totals


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def ensure_schema(client: notion_import.NotionClient, data_source_id: str) -> None:
    for old, new in RENAMED_TEXT_COLUMNS.items():
        schema = client.request("GET", f"/v1/data_sources/{data_source_id}").get("properties") or {}
        if new in schema:
            continue
        if old in schema:
            LOGGER.info("Renomeando propriedade Notion: %s -> %s", old, new)
            client.request("PATCH", f"/v1/data_sources/{data_source_id}", json_body={"properties": {old: {"name": new}}})
            continue
        LOGGER.info("Criando propriedade Notion ausente: %s", new)
        client.request("PATCH", f"/v1/data_sources/{data_source_id}", json_body={"properties": {new: {"rich_text": {}}}})

    schema = client.request("GET", f"/v1/data_sources/{data_source_id}").get("properties") or {}
    expected = {
        "texto_em_vigor": {"rich_text": {}},
        "classe_vigencia_texto": {"select": {"options": []}},
        "usar_como_texto_vigente": {"checkbox": {}},
        "alerta_vigencia": {"rich_text": {}},
        "prioridade_rag": {"select": {"options": []}},
    }
    patch: Dict[str, Any] = {}
    for name, config in expected.items():
        current = schema.get(name)
        expected_type = next(iter(config))
        if not isinstance(current, dict) or current.get("type") != expected_type:
            patch[name] = config
    if patch:
        LOGGER.info("Criando/ajustando propriedades Notion: %s", ", ".join(patch))
        client.request("PATCH", f"/v1/data_sources/{data_source_id}", json_body={"properties": patch})


def build_update_properties(row: Mapping[str, str], *, max_rich_text_chars: int) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for column in RICH_TEXT_UPDATE_COLUMNS:
        props[column] = notion_import.rich_text_property(row.get(column, ""), max_chars=max_rich_text_chars)
    for column in SELECT_UPDATE_COLUMNS:
        props[column] = notion_import.select_property(row.get(column, ""))
    for column in CHECKBOX_UPDATE_COLUMNS:
        props[column] = notion_import.checkbox_property(row.get(column, ""))
    for column in NUMBER_UPDATE_COLUMNS:
        props[column] = notion_import.number_property(row.get(column, ""))
    return props


def update_page(client: notion_import.NotionClient, page_id: str, props: Mapping[str, Any]) -> None:
    client.request("PATCH", f"/v1/pages/{page_id}", json_body={"properties": dict(props)})


def update_notion(
    rows: Sequence[Mapping[str, str]],
    *,
    import_checkpoint_path: Path,
    repair_checkpoint_path: Path,
    workers: int,
    max_rich_text_chars: int,
    checkpoint_every: int,
    rate_rps: float,
    timeout_s: int,
    max_retries: int,
) -> Dict[str, int]:
    import_checkpoint = read_json(import_checkpoint_path)
    data_source_id = normalize_ws(import_checkpoint.get("data_source_id"))
    page_ids_by_row_key = import_checkpoint.get("created_pages") or {}
    if not data_source_id or not isinstance(page_ids_by_row_key, dict):
        raise RuntimeError("Checkpoint de importacao sem data_source_id/created_pages.")

    client = notion_import.NotionClient(
        notion_import.resolve_notion_token(),
        notion_version=notion_import.DEFAULT_NOTION_VERSION,
        timeout_s=timeout_s,
        max_retries=max_retries,
        rate_rps=rate_rps,
    )
    ensure_schema(client, data_source_id)

    repair_checkpoint = read_json(repair_checkpoint_path) or {
        "started_at": notion_import.now_iso(),
        "data_source_id": data_source_id,
        "updated_pages": {},
    }
    updated_pages = repair_checkpoint.setdefault("updated_pages", {})
    if not isinstance(updated_pages, dict):
        raise RuntimeError("Campo updated_pages invalido no checkpoint de reparo.")

    version = "vigencia_v2"
    pending: List[Tuple[str, str, Dict[str, Any]]] = []
    missing_pages = 0
    for row in rows:
        row_key = normalize_ws(row.get("row_key"))
        if updated_pages.get(row_key) == version:
            continue
        page_id = page_ids_by_row_key.get(row_key)
        if not page_id:
            missing_pages += 1
            continue
        pending.append((row_key, page_id, build_update_properties(row, max_rich_text_chars=max_rich_text_chars)))

    LOGGER.info("Paginas pendentes para reparo de vigencia: %s; paginas ausentes: %s.", len(pending), missing_pages)
    submitted = 0
    updated = 0
    in_flight: Dict[Future[None], str] = {}
    max_in_flight = max(1, workers * 3)
    lock = threading.Lock()

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal submitted
        while submitted < len(pending) and len(in_flight) < max_in_flight:
            row_key, page_id, props = pending[submitted]
            submitted += 1
            future = executor.submit(update_page, client, page_id, props)
            in_flight[future] = row_key

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_next(executor)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                row_key = in_flight.pop(future)
                future.result()
                with lock:
                    updated_pages[row_key] = version
                    updated += 1
                    if updated % checkpoint_every == 0 or updated == len(pending):
                        repair_checkpoint["updated_count"] = len(updated_pages)
                        repair_checkpoint["updated_at"] = notion_import.now_iso()
                        write_json(repair_checkpoint_path, repair_checkpoint)
                        LOGGER.info("Reparo Notion: %s/%s paginas atualizadas.", updated, len(pending))
            submit_next(executor)

    repair_checkpoint["updated_count"] = len(updated_pages)
    repair_checkpoint["completed_at"] = notion_import.now_iso()
    write_json(repair_checkpoint_path, repair_checkpoint)
    return {"updated": updated, "pending": len(pending), "missing_pages": missing_pages, "total_marked": len(updated_pages)}


def property_checkbox(prop: Mapping[str, Any]) -> bool:
    if isinstance(prop, dict) and prop.get("type") == "checkbox":
        return bool(prop.get("checkbox"))
    return False


def property_select(prop: Mapping[str, Any]) -> str:
    if isinstance(prop, dict) and prop.get("type") == "select":
        value = prop.get("select")
        if isinstance(value, dict):
            return normalize_ws(value.get("name"))
    return ""


def validate_notion(client: notion_import.NotionClient, data_source_id: str) -> Dict[str, int]:
    totals = {
        "rows": 0,
        "status_blank": 0,
        "texto_rag_blank": 0,
        "operacional_dispositivo": 0,
        "operacional_texto_em_vigor_blank": 0,
        "comparativo": 0,
        "comparativo_texto_em_vigor_nonblank": 0,
        "comparativo_usar_vigente_true": 0,
        "granular": 0,
        "granular_incluir_no_rag_true": 0,
    }
    cursor = ""
    while True:
        body: Dict[str, Any] = {"page_size": 100, "result_type": "page"}
        if cursor:
            body["start_cursor"] = cursor
        payload = client.request("POST", f"/v1/data_sources/{data_source_id}/query", json_body=body)
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict) or item.get("object") != "page":
                continue
            totals["rows"] += 1
            props = item.get("properties") or {}
            tipo_linha = property_select(props.get("tipo_linha", {}) or {})
            tipo_base = property_select(props.get("tipo_base", {}) or {})
            status = property_select(props.get("status_alteracao", {}) or {})
            texto_rag = normalize_ws(notion_import.property_plain_text(props.get("texto_rag", {}) or {}))
            texto_em_vigor = normalize_ws(notion_import.property_plain_text(props.get("texto_em_vigor", {}) or {}))
            incluir = property_checkbox(props.get("incluir_no_rag", {}) or {})
            usar_vigente = property_checkbox(props.get("usar_como_texto_vigente", {}) or {})
            comparative = tipo_linha in {"alteracao", "alteracao_granular"} or tipo_base.startswith("comparativo")
            if not status:
                totals["status_blank"] += 1
            if not texto_rag:
                totals["texto_rag_blank"] += 1
            if tipo_base == "operacional" and tipo_linha == "dispositivo":
                totals["operacional_dispositivo"] += 1
                if not texto_em_vigor:
                    totals["operacional_texto_em_vigor_blank"] += 1
            if comparative:
                totals["comparativo"] += 1
                if texto_em_vigor:
                    totals["comparativo_texto_em_vigor_nonblank"] += 1
                if usar_vigente:
                    totals["comparativo_usar_vigente_true"] += 1
            if tipo_linha == "alteracao_granular":
                totals["granular"] += 1
                if incluir:
                    totals["granular_incluir_no_rag_true"] += 1
        if not payload.get("has_more"):
            break
        cursor = normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return totals


def update_import_checkpoint(import_checkpoint_path: Path, validation: Mapping[str, Any], repair_result: Mapping[str, Any]) -> None:
    payload = read_json(import_checkpoint_path)
    payload["vigencia_repair_validation"] = dict(validation)
    payload["vigencia_repair_result"] = dict(repair_result)
    payload["vigencia_repair_completed_at"] = notion_import.now_iso()
    write_json(import_checkpoint_path, payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrige semantica de vigencia da base Resolucoes TSE - RAG consolidado.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--import-checkpoint", type=Path, default=DEFAULT_IMPORT_CHECKPOINT)
    parser.add_argument("--repair-checkpoint", type=Path, default=DEFAULT_REPAIR_CHECKPOINT)
    parser.add_argument("--csv-only", action="store_true")
    parser.add_argument("--notion-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-rps", type=float, default=2.7)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-rich-text-chars", type=int, default=30000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.notion_only:
        _, rows = read_csv_rows(args.csv)
    else:
        _, rows, validation = repair_csv(args.csv)
        LOGGER.info("Validacao CSV: %s", json.dumps(validation, ensure_ascii=False, sort_keys=True))
        blocking = [
            "status_blank",
            "texto_rag_blank",
            "texto_em_vigor_operacional_blank",
            "texto_em_vigor_comparativo_nonblank",
            "granular_incluir_no_rag_true",
            "usar_vigente_comparativo_true",
        ]
        if any(validation[name] for name in blocking):
            raise RuntimeError("Validacao CSV encontrou pendencias de vigencia.")

    if args.csv_only:
        return 0

    result = update_notion(
        rows,
        import_checkpoint_path=args.import_checkpoint,
        repair_checkpoint_path=args.repair_checkpoint,
        workers=max(1, int(args.workers or 1)),
        max_rich_text_chars=max(2000, int(args.max_rich_text_chars or 0)),
        checkpoint_every=max(1, int(args.checkpoint_every or 1)),
        rate_rps=max(0.1, float(args.rate_rps or 0.0)),
        timeout_s=max(5, int(args.timeout_s or 0)),
        max_retries=max(0, int(args.max_retries or 0)),
    )
    LOGGER.info("Reparo Notion concluido: %s", json.dumps(result, ensure_ascii=False, sort_keys=True))

    import_checkpoint = read_json(args.import_checkpoint)
    data_source_id = normalize_ws(import_checkpoint.get("data_source_id"))
    client = notion_import.NotionClient(
        notion_import.resolve_notion_token(),
        notion_version=notion_import.DEFAULT_NOTION_VERSION,
        timeout_s=max(5, int(args.timeout_s or 0)),
        max_retries=max(0, int(args.max_retries or 0)),
        rate_rps=max(0.1, float(args.rate_rps or 0.0)),
    )
    validation = validate_notion(client, data_source_id)
    LOGGER.info("Validacao Notion: %s", json.dumps(validation, ensure_ascii=False, sort_keys=True))
    blocking = [
        "status_blank",
        "texto_rag_blank",
        "operacional_texto_em_vigor_blank",
        "comparativo_texto_em_vigor_nonblank",
        "comparativo_usar_vigente_true",
        "granular_incluir_no_rag_true",
    ]
    if any(validation[name] for name in blocking):
        raise RuntimeError("Validacao Notion encontrou pendencias de vigencia.")
    update_import_checkpoint(args.import_checkpoint, validation, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
