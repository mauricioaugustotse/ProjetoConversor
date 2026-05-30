#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import NOTION_import_codigo_eleitoral_2026_csvs as notion_import


DEFAULT_CSV_DIR = Path("Codigo_Eleitoral_2026_csv_normas")
DEFAULT_CHECKPOINT = DEFAULT_CSV_DIR / ".notion_import_checkpoint.json"

EXTRA_CSV_COLUMNS = [
    "row_key",
    "csv_arquivo",
    "dispositivo_pai_id",
    "dispositivo_pai_row_key",
]

RELATION_PROPERTY = "dispositivo_pai_relacao"
UPDATE_PROPERTIES = [
    "Nome",
    "norma_nome_popular",
    "referencias_jurisprudenciais",
    "resumo_curto",
    "texto_rag",
    "dispositivo_pai",
    "chars",
    "tokens_estimados",
    "dispositivo_pai_id",
    "dispositivo_pai_row_key",
    RELATION_PROPERTY,
]

LOGGER = logging.getLogger("codigo_eleitoral_2026_repair")


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\ufeff", " ")).strip()


def csv_row_key(csv_name: str, row: Mapping[str, str]) -> str:
    stem = Path(csv_name).stem
    row_id = normalize_ws(row.get("id")) or "sem-id"
    return f"{stem}:{row_id}"


def year_from_row(row: Mapping[str, str]) -> str:
    candidates = [
        normalize_ws(row.get("norma_titulo")),
        normalize_ws(row.get("norma_id")),
        normalize_ws(row.get("csv_arquivo")),
    ]
    found: List[str] = []
    for candidate in candidates:
        found.extend(re.findall(r"(?:19|20)\d{2}", candidate))
    return found[-1] if found else ""


def canonical_norm_label(row: Mapping[str, str]) -> str:
    title = normalize_ws(row.get("norma_titulo"))
    norm_id = normalize_ws(row.get("norma_id"))
    year = year_from_row(row)
    if not title:
        return norm_id

    if norm_id == "constituicao_federal_1988" or title.lower().startswith("constituição federal"):
        return "Constituição Federal"

    if title.startswith("Súmula"):
        return title

    match = re.match(r"^(?P<prefix>.+?\bn\.)\s*(?P<num>[\d.]+)(?:\s*,?\s+de\b.*)?$", title, flags=re.IGNORECASE)
    if match and year:
        prefix = normalize_ws(match.group("prefix"))
        number = normalize_ws(match.group("num"))
        prefix = re.sub(r"(?i)^Lei Complementar n\.$", "LC n.", prefix)
        prefix = re.sub(r"(?i)^Emenda Constitucional n\.$", "EC n.", prefix)
        return f"{prefix} {number}/{year}"

    return title


def repaired_popular_name(row: Mapping[str, str]) -> str:
    current = normalize_ws(row.get("norma_nome_popular"))
    return current or canonical_norm_label(row)


def clean_text_for_summary(text: str) -> str:
    text = normalize_ws(text)
    text = re.sub(r"^Art\.\s*[\wº°.-]+(?:-[A-Z])?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Parágrafo único\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^§\s*\d+[º°]?(?:-[A-Z])?\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[IVXLCDM]+\s*[-–—]\s*", "", text)
    text = re.sub(r"^[a-z]\)\s*", "", text)
    text = re.sub(r"^\d+[.)]\s*", "", text)
    return normalize_ws(text)


def first_legal_sentence(text: str) -> str:
    text = clean_text_for_summary(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])", text)
    first = normalize_ws(parts[0] if parts else text)
    return first or text


def limit_text(text: str, max_chars: int = 420) -> str:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}..."


def build_summary(row: Mapping[str, str]) -> str:
    kind = normalize_ws(row.get("tipo_dispositivo"))
    dispositivo = normalize_ws(row.get("dispositivo"))
    parent = normalize_ws(row.get("dispositivo_pai"))
    norm = normalize_ws(row.get("norma_nome_popular")) or canonical_norm_label(row)
    text = normalize_ws(row.get("texto_dispositivo"))
    refs_juris = normalize_ws(row.get("referencias_jurisprudenciais"))
    try:
        notes_count = int(float(normalize_ws(row.get("qtd_notas")) or "0"))
    except ValueError:
        notes_count = 0

    sentence = first_legal_sentence(text) or first_legal_sentence(row.get("notas_texto", ""))
    if kind == "norma":
        prefix = f"Norma {norm}"
    elif kind in {"estrutura", "artigo", "paragrafo", "inciso", "alinea", "item"} and parent:
        prefix = f"{dispositivo}, subordinado a {parent}"
    elif kind == "estrutura":
        prefix = f"Estrutura da {norm}"
    elif kind == "sumula":
        prefix = f"Enunciado {dispositivo}"
    elif kind == "artigo":
        prefix = f"{dispositivo}"
    elif kind == "paragrafo":
        prefix = f"{dispositivo}"
    else:
        prefix = dispositivo or norm

    if sentence and sentence.lower() != normalize_ws(dispositivo).lower():
        summary = f"{prefix}: {sentence}"
    else:
        summary = prefix

    flags: List[str] = []
    lower_text = f"{text} {row.get('notas_texto', '')}".lower()
    if re.search(r"\b(revogado|revogada|cancelada|cancelado|vetado|vetada)\b", lower_text):
        flags.append("há marcação de revogação/cancelamento/veto")
    if notes_count:
        flags.append(f"{notes_count} nota(s) vinculada(s)")
    if refs_juris:
        flags.append("jurisprudência extraída das notas")
    if flags:
        summary = f"{summary} | " + "; ".join(flags)
    return limit_text(summary)


def extract_jurisprudence_references(notes_text: str) -> str:
    text = normalize_ws(notes_text)
    if not text:
        return ""
    patterns = [
        r"Ac\.-(?:TSE|STF|STJ)\s+n\.\s*[\d./-]+",
        r"Ac\.-(?:TSE|STF|STJ)(?:,\s*de\s*\d{1,2}/\d{1,2}/\d{4})?,\s*(?:no|na|nos|nas)\s*[^:;.]{1,100}?\bn\.\s*[\d./-]+",
        r"\b(?:ADI|ADC|ADPF|ADO|ARE|HC|MS|MI|RE|REspe|AgR-REspe|REspEl|AgR-REspEl|RO|RMS|PA|CtaEl|Cta|Rp|Rcl|Pet|AI)\s+n\.\s*[\d./-]+",
        r"\bTema\s+n\.\s*\d+",
        r"\bTema\s+\d+\b",
        r"\bRepercussão Geral no RE\s+n\.\s*[\d./-]+",
    ]
    found: List[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = normalize_ws(match.group(0)).rstrip(" ,;:.")
            if not value:
                continue
            key = value.lower()
            if key not in seen:
                seen.add(key)
                found.append(value)
    filtered: List[str] = []
    for value in found:
        value_key = value.lower()
        if any(value_key != other.lower() and value_key in other.lower() for other in found):
            continue
        filtered.append(value)
    return "; ".join(filtered)


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def build_text_rag(row: Mapping[str, str]) -> str:
    parts = [
        "Documento: Codigo Eleitoral 2026 anotado",
        f"Norma: {normalize_ws(row.get('norma_titulo'))}",
    ]
    popular = normalize_ws(row.get("norma_nome_popular"))
    if popular:
        parts.append(f"Nome popular/citação curta: {popular}")
    for label, column in [
        ("Macrogrupo", "macrogrupo"),
        ("Grupo do sumário", "grupo_sumario"),
        ("Dispositivo", "dispositivo"),
        ("Tipo", "tipo_dispositivo"),
        ("Hierarquia", "hierarquia_normativa"),
        ("Dispositivo pai", "dispositivo_pai"),
        ("ID do dispositivo pai", "dispositivo_pai_row_key"),
    ]:
        value = normalize_ws(row.get(column))
        if value:
            parts.append(f"{label}: {value}")
    page_start = normalize_ws(row.get("pagina_inicial"))
    page_end = normalize_ws(row.get("pagina_final"))
    if page_start:
        parts.append(f"Páginas: {page_start}-{page_end or page_start}")
    summary = normalize_ws(row.get("resumo_curto"))
    if summary:
        parts.extend(["", f"Resumo: {summary}"])
    refs_juris = normalize_ws(row.get("referencias_jurisprudenciais"))
    if refs_juris:
        parts.extend(["", f"Referências jurisprudenciais: {refs_juris}"])
    text = normalize_ws(row.get("texto_dispositivo"))
    if text:
        parts.extend(["", text])
    notes = normalize_ws(row.get("notas_texto"))
    if notes:
        parts.extend(["", "Notas vinculadas:", notes])
    return normalize_ws("\n".join(parts))


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def hierarchy_parent(hierarchy: str) -> str:
    parts = [part.strip() for part in normalize_ws(hierarchy).split(" > ") if part.strip()]
    if len(parts) <= 1:
        return ""
    return " > ".join(parts[:-1])


def set_parent(row: Dict[str, str], parent_row: Optional[Mapping[str, str]], csv_name: str) -> bool:
    before = (
        normalize_ws(row.get("dispositivo_pai")),
        normalize_ws(row.get("dispositivo_pai_id")),
        normalize_ws(row.get("dispositivo_pai_row_key")),
    )
    if parent_row and parent_row is not row:
        row["dispositivo_pai"] = normalize_ws(parent_row.get("dispositivo"))
        row["dispositivo_pai_id"] = normalize_ws(parent_row.get("id"))
        row["dispositivo_pai_row_key"] = csv_row_key(csv_name, parent_row)
    else:
        row["dispositivo_pai"] = ""
        row["dispositivo_pai_id"] = ""
        row["dispositivo_pai_row_key"] = ""
    after = (
        normalize_ws(row.get("dispositivo_pai")),
        normalize_ws(row.get("dispositivo_pai_id")),
        normalize_ws(row.get("dispositivo_pai_row_key")),
    )
    return before != after


def write_csv_rows(csv_path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def repair_file(csv_path: Path) -> Tuple[int, int, int, int]:
    fieldnames, rows = read_csv_rows(csv_path)
    for column in EXTRA_CSV_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    for row in rows:
        row["csv_arquivo"] = csv_path.name
        row["row_key"] = csv_row_key(csv_path.name, row)

    display_to_row = {
        normalize_ws(row.get("dispositivo")).lower(): row
        for row in rows
        if normalize_ws(row.get("dispositivo"))
    }
    norm_row = next((row for row in rows if normalize_ws(row.get("tipo_dispositivo")) == "norma"), None)
    if norm_row is None:
        norm_row = next((row for row in rows if normalize_ws(row.get("tipo_dispositivo")) == "sumula"), None)
    structure_by_hierarchy: Dict[str, Dict[str, str]] = {}

    popular_filled = 0
    refs_changed = 0
    summaries_changed = 0
    parent_links = 0

    for row in rows:
        before_popular = normalize_ws(row.get("norma_nome_popular"))
        row["norma_nome_popular"] = repaired_popular_name(row)
        if not before_popular and row["norma_nome_popular"]:
            popular_filled += 1

        tipo = normalize_ws(row.get("tipo_dispositivo"))
        parent_row: Optional[Mapping[str, str]] = None
        if tipo == "estrutura":
            hierarchy = normalize_ws(row.get("hierarquia_normativa"))
            parent_row = structure_by_hierarchy.get(hierarchy_parent(hierarchy)) or norm_row
        elif tipo == "artigo":
            hierarchy = normalize_ws(row.get("hierarquia_normativa"))
            parent_row = structure_by_hierarchy.get(hierarchy) or norm_row
        elif tipo == "paragrafo" and not normalize_ws(row.get("dispositivo_pai")):
            parent_row = display_to_row.get(normalize_ws(row.get("artigo")).lower()) or norm_row
        elif tipo in {"paragrafo", "inciso", "alinea", "item"}:
            parent_text = normalize_ws(row.get("dispositivo_pai"))
            parent_row = display_to_row.get(parent_text.lower()) if parent_text else None

        if set_parent(row, parent_row, csv_path.name):
            parent_links += 1
        elif normalize_ws(row.get("dispositivo_pai_row_key")):
            parent_links += 1

        if tipo == "estrutura":
            hierarchy = normalize_ws(row.get("hierarquia_normativa"))
            if hierarchy:
                structure_by_hierarchy[hierarchy] = row

        new_refs = extract_jurisprudence_references(row.get("notas_texto", ""))
        if normalize_ws(row.get("referencias_jurisprudenciais")) != new_refs:
            refs_changed += 1
        row["referencias_jurisprudenciais"] = new_refs

        new_summary = build_summary(row)
        if normalize_ws(row.get("resumo_curto")) != new_summary:
            summaries_changed += 1
        row["resumo_curto"] = new_summary

        rag = build_text_rag(row)
        row["texto_rag"] = rag
        row["chars"] = str(len(rag))
        row["tokens_estimados"] = str(estimate_tokens(rag))

    write_csv_rows(csv_path, fieldnames, rows)
    return popular_filled, refs_changed, summaries_changed, parent_links


def repair_csv_dir(csv_dir: Path, *, limit_files: int = 0) -> Dict[str, int]:
    totals = {
        "files": 0,
        "popular_filled": 0,
        "refs_changed": 0,
        "summaries_changed": 0,
        "parent_links": 0,
    }
    files = sorted(csv_dir.glob("*.csv"))
    if limit_files > 0:
        files = files[:limit_files]
    for csv_path in files:
        popular, refs, summaries, parents = repair_file(csv_path)
        totals["files"] += 1
        totals["popular_filled"] += popular
        totals["refs_changed"] += refs
        totals["summaries_changed"] += summaries
        totals["parent_links"] += parents
    return totals


def iter_repaired_rows(csv_dir: Path, *, limit: int = 0) -> Iterable[Dict[str, str]]:
    count = 0
    for csv_path in sorted(csv_dir.glob("*.csv")):
        _, rows = read_csv_rows(csv_path)
        for row in rows:
            count += 1
            yield row
            if limit > 0 and count >= limit:
                return


def ensure_notion_schema(client: notion_import.NotionClient, data_source_id: str) -> None:
    data_source = client.request("GET", f"/v1/data_sources/{data_source_id}")
    schema = data_source.get("properties") or {}
    if not isinstance(schema, dict):
        raise RuntimeError("Schema da data source nao retornado pelo Notion.")
    patch: Dict[str, Any] = {}
    expected = {
        "dispositivo_pai_id": {"number": {"format": "number"}},
        "dispositivo_pai_row_key": {"rich_text": {}},
        RELATION_PROPERTY: {"relation": {"data_source_id": data_source_id, "single_property": {}}},
    }
    for name, config in expected.items():
        current = schema.get(name)
        if not isinstance(current, dict) or normalize_ws(current.get("type")) not in config:
            patch[name] = config
    if patch:
        LOGGER.info("Atualizando schema Notion: %s", ", ".join(patch))
        client.request("PATCH", f"/v1/data_sources/{data_source_id}", json_body={"properties": patch})


def build_page_update_properties(
    row: Mapping[str, str],
    *,
    page_ids_by_row_key: Mapping[str, str],
    max_rich_text_chars: int,
) -> Dict[str, Any]:
    parent_key = normalize_ws(row.get("dispositivo_pai_row_key"))
    parent_page_id = page_ids_by_row_key.get(parent_key, "") if parent_key else ""
    props: Dict[str, Any] = {
        "Nome": notion_import.title_property(notion_import.build_row_title(row)),
        "norma_nome_popular": notion_import.rich_text_property(row.get("norma_nome_popular", ""), max_chars=max_rich_text_chars),
        "referencias_jurisprudenciais": notion_import.rich_text_property(row.get("referencias_jurisprudenciais", ""), max_chars=max_rich_text_chars),
        "resumo_curto": notion_import.rich_text_property(row.get("resumo_curto", ""), max_chars=max_rich_text_chars),
        "texto_rag": notion_import.rich_text_property(row.get("texto_rag", ""), max_chars=max_rich_text_chars),
        "dispositivo_pai": notion_import.rich_text_property(row.get("dispositivo_pai", ""), max_chars=max_rich_text_chars),
        "chars": notion_import.number_property(row.get("chars", "")),
        "tokens_estimados": notion_import.number_property(row.get("tokens_estimados", "")),
        "dispositivo_pai_id": notion_import.number_property(row.get("dispositivo_pai_id", "")),
        "dispositivo_pai_row_key": notion_import.rich_text_property(row.get("dispositivo_pai_row_key", ""), max_chars=max_rich_text_chars),
        RELATION_PROPERTY: {"relation": [{"id": parent_page_id}] if parent_page_id else []},
    }
    return props


def update_page(client: notion_import.NotionClient, page_id: str, props: Mapping[str, Any]) -> None:
    client.request("PATCH", f"/v1/pages/{page_id}", json_body={"properties": dict(props)})


def update_notion_rows(
    csv_dir: Path,
    checkpoint_path: Path,
    *,
    workers: int,
    rate_rps: float,
    max_rich_text_chars: int,
    limit: int,
    only_types: Optional[set[str]] = None,
    only_row_keys: Optional[set[str]] = None,
) -> Dict[str, int]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    data_source_id = normalize_ws(checkpoint.get("data_source_id"))
    page_ids_by_row_key = checkpoint.get("created_pages") or {}
    if not data_source_id or not isinstance(page_ids_by_row_key, dict):
        raise RuntimeError("Checkpoint sem data_source_id/created_pages.")

    client = notion_import.NotionClient(
        notion_import.resolve_notion_token(),
        notion_version=notion_import.DEFAULT_NOTION_VERSION,
        timeout_s=45,
        max_retries=6,
        rate_rps=rate_rps,
    )
    ensure_notion_schema(client, data_source_id)

    rows: List[Dict[str, str]] = []
    for row in iter_repaired_rows(csv_dir):
        if only_types and normalize_ws(row.get("tipo_dispositivo")) not in only_types:
            continue
        if only_row_keys and normalize_ws(row.get("row_key")) not in only_row_keys:
            continue
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    total = len(rows)
    updated = 0
    missing_pages = 0
    lock = threading.Lock()
    submitted = 0
    in_flight: Dict[Future[None], str] = {}
    max_in_flight = max(1, workers * 3)

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal submitted, missing_pages
        while submitted < total and len(in_flight) < max_in_flight:
            row = rows[submitted]
            submitted += 1
            row_key = normalize_ws(row.get("row_key"))
            page_id = page_ids_by_row_key.get(row_key, "")
            if not page_id:
                missing_pages += 1
                continue
            props = build_page_update_properties(
                row,
                page_ids_by_row_key=page_ids_by_row_key,
                max_rich_text_chars=max_rich_text_chars,
            )
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
                    updated += 1
                    if updated % 100 == 0 or updated == total:
                        LOGGER.info("Notion atualizado: %s/%s paginas", updated, total)
            submit_next(executor)

    return {"rows": total, "updated": updated, "missing_pages": missing_pages}


def validate_csvs(csv_dir: Path) -> Dict[str, int]:
    totals = {
        "rows": 0,
        "missing_popular": 0,
        "missing_parent_key": 0,
        "parent_rows": 0,
        "article_rows": 0,
        "article_rows_with_parent": 0,
        "structure_rows": 0,
        "structure_rows_with_parent": 0,
        "juris_with_bare_n": 0,
    }
    for csv_path in sorted(csv_dir.glob("*.csv")):
        _, rows = read_csv_rows(csv_path)
        for row in rows:
            totals["rows"] += 1
            if not normalize_ws(row.get("norma_nome_popular")):
                totals["missing_popular"] += 1
            if normalize_ws(row.get("tipo_dispositivo")) == "artigo":
                totals["article_rows"] += 1
                if normalize_ws(row.get("dispositivo_pai_row_key")):
                    totals["article_rows_with_parent"] += 1
            if normalize_ws(row.get("tipo_dispositivo")) == "estrutura":
                totals["structure_rows"] += 1
                if normalize_ws(row.get("dispositivo_pai_row_key")):
                    totals["structure_rows_with_parent"] += 1
            if normalize_ws(row.get("dispositivo_pai")):
                totals["parent_rows"] += 1
                if not normalize_ws(row.get("dispositivo_pai_row_key")):
                    totals["missing_parent_key"] += 1
            if re.search(r"\bn\.?$", normalize_ws(row.get("referencias_jurisprudenciais")), flags=re.IGNORECASE):
                totals["juris_with_bare_n"] += 1
    return totals


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repara CSVs do Codigo Eleitoral 2026 e atualiza as linhas correspondentes no Notion.")
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--csv-only", action="store_true")
    parser.add_argument("--notion-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--types", default="", help="Atualiza apenas tipos separados por virgula, ex.: estrutura,artigo.")
    parser.add_argument("--row-keys-file", type=Path, default=None, help="Atualiza apenas row_keys listados no arquivo, um por linha.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-rps", type=float, default=2.8)
    parser.add_argument("--max-rich-text-chars", type=int, default=30000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.notion_only:
        totals = repair_csv_dir(args.csv_dir, limit_files=max(0, int(args.limit_files or 0)))
        LOGGER.info("CSVs reparados: %s", json.dumps(totals, ensure_ascii=False, sort_keys=True))

    validation = validate_csvs(args.csv_dir)
    LOGGER.info("Validacao CSV: %s", json.dumps(validation, ensure_ascii=False, sort_keys=True))
    if (
        validation["missing_popular"]
        or validation["missing_parent_key"]
        or validation["juris_with_bare_n"]
        or validation["article_rows"] != validation["article_rows_with_parent"]
        or validation["structure_rows"] != validation["structure_rows_with_parent"]
    ):
        raise RuntimeError("Validacao CSV encontrou pendencias.")

    if not args.csv_only:
        only_types = {normalize_ws(item) for item in str(args.types or "").split(",") if normalize_ws(item)}
        only_row_keys: Optional[set[str]] = None
        if args.row_keys_file:
            only_row_keys = {
                normalize_ws(line)
                for line in args.row_keys_file.read_text(encoding="utf-8").splitlines()
                if normalize_ws(line)
            }
        result = update_notion_rows(
            args.csv_dir,
            args.checkpoint,
            workers=max(1, int(args.workers or 1)),
            rate_rps=max(0.1, float(args.rate_rps or 0.0)),
            max_rich_text_chars=max(2000, int(args.max_rich_text_chars or 0)),
            limit=max(0, int(args.limit or 0)),
            only_types=only_types or None,
            only_row_keys=only_row_keys,
        )
        LOGGER.info("Notion reparado: %s", json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
