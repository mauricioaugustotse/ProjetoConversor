import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CSV_FILE = "boletins_de_jurisprudencia_TRF1_with_news.csv"
DEFAULT_CACHE_FILE = "perplexity_cache.json"
DEFAULT_CHECKPOINT_FILE = ".boletins_de_jurisprudencia_TRF1_with_news.checkpoint.json"

LEGACY_NOTICIA_COL = "noticia"
NOTICIA_COL_1 = "noticia_1"
NOTICIA_COL_2 = "noticia_2"
LEGACY_LINK_SEPARATOR = "; "
MAX_LINKS_PER_ROW = 2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_links(value: Any, *, max_links: Optional[int] = None) -> List[str]:
    raw: List[str] = []
    if isinstance(value, list):
        raw = [normalize_text(v) for v in value]
    elif isinstance(value, str):
        text = normalize_text(value)
        if text:
            raw = [normalize_text(part) for part in text.split(LEGACY_LINK_SEPARATOR)]
    elif value is not None:
        text = normalize_text(value)
        if text:
            raw = [text]

    deduped: List[str] = []
    seen = set()
    for link in raw:
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


def split_kept_extras(links: List[str], *, max_links: int = MAX_LINKS_PER_ROW) -> Tuple[List[str], List[str]]:
    kept = links[:max_links]
    extras = links[max_links:]
    return kept, extras


def atomic_write_json(path: str, payload: Dict[str, Any], indent: int = 2) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def write_csv_atomic(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def backup_file(path: str, timestamp: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    backup_path = f"{path}.backup_{timestamp}"
    shutil.copy2(path, backup_path)
    return backup_path


def build_csv_fieldnames(original_fieldnames: List[str]) -> List[str]:
    if LEGACY_NOTICIA_COL in original_fieldnames:
        out: List[str] = []
        inserted = False
        for col in original_fieldnames:
            if col == LEGACY_NOTICIA_COL:
                if not inserted:
                    out.extend([NOTICIA_COL_1, NOTICIA_COL_2])
                    inserted = True
                continue
            if col in (NOTICIA_COL_1, NOTICIA_COL_2):
                continue
            out.append(col)
        return out

    out = list(original_fieldnames)
    if NOTICIA_COL_1 not in out:
        out.append(NOTICIA_COL_1)
    if NOTICIA_COL_2 not in out:
        out.append(NOTICIA_COL_2)
    return out


def get_row_links_for_migration(row: Dict[str, str]) -> List[str]:
    new_links = normalize_links(
        [
            row.get(NOTICIA_COL_1, ""),
            row.get(NOTICIA_COL_2, ""),
        ],
        max_links=None,
    )
    legacy_links = normalize_links(row.get(LEGACY_NOTICIA_COL, ""), max_links=None)
    if new_links:
        return normalize_links(new_links + legacy_links, max_links=None)
    return legacy_links


def set_row_links(row: Dict[str, str], links: List[str]) -> None:
    row[NOTICIA_COL_1] = links[0] if len(links) > 0 else ""
    row[NOTICIA_COL_2] = links[1] if len(links) > 1 else ""
    row.pop(LEGACY_NOTICIA_COL, None)


def add_overflow_record(
    *,
    overflow_records: List[Dict[str, str]],
    source_file: str,
    record_id: str,
    query_or_row: str,
    kept: List[str],
    extras: List[str],
) -> None:
    if not extras:
        return
    overflow_records.append(
        {
            "source_file": source_file,
            "record_id": record_id,
            "query_or_row": query_or_row,
            "kept_1": kept[0] if len(kept) > 0 else "",
            "kept_2": kept[1] if len(kept) > 1 else "",
            "extras_json": json.dumps(extras, ensure_ascii=False),
        }
    )


def migrate_csv_file(
    csv_file: str,
    overflow_records: List[Dict[str, str]],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "path": csv_file,
        "exists": os.path.exists(csv_file),
        "status": "missing",
        "legacy_column_present": False,
        "rows_total": 0,
        "rows_with_any_link": 0,
        "rows_changed": 0,
        "overflow_rows": 0,
    }
    if not stats["exists"]:
        return stats

    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames or [])
        if not original_fieldnames:
            stats["status"] = "invalid_header"
            return stats

        stats["legacy_column_present"] = LEGACY_NOTICIA_COL in original_fieldnames
        output_fieldnames = build_csv_fieldnames(original_fieldnames)
        migrated_rows: List[Dict[str, str]] = []

        for idx, raw_row in enumerate(reader):
            stats["rows_total"] += 1
            row = {k: normalize_text(raw_row.get(k, "")) for k in original_fieldnames}

            before_links = get_row_links_for_migration(row)
            kept, extras = split_kept_extras(before_links, max_links=MAX_LINKS_PER_ROW)
            set_row_links(row, kept)

            if kept:
                stats["rows_with_any_link"] += 1
            if extras:
                stats["overflow_rows"] += 1
                row_label = normalize_text(row.get("numero_processo", "")) or f"row_{idx + 1}"
                add_overflow_record(
                    overflow_records=overflow_records,
                    source_file=os.path.basename(csv_file),
                    record_id=str(idx),
                    query_or_row=row_label,
                    kept=kept,
                    extras=extras,
                )

            if normalize_links(before_links, max_links=MAX_LINKS_PER_ROW) != kept:
                stats["rows_changed"] += 1
            if LEGACY_NOTICIA_COL in original_fieldnames:
                stats["rows_changed"] += 1

            for col in output_fieldnames:
                row.setdefault(col, "")
            migrated_rows.append(row)

    stats["status"] = "migrated"
    stats["original_fieldnames"] = original_fieldnames
    stats["output_fieldnames"] = output_fieldnames
    stats["migrated_rows"] = migrated_rows
    return stats


def migrate_checkpoint_file(
    checkpoint_file: str,
    overflow_records: List[Dict[str, str]],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "path": checkpoint_file,
        "exists": os.path.exists(checkpoint_file),
        "status": "missing",
        "version_before": None,
        "version_after": 2,
        "processed_count_before": 0,
        "processed_count_after": 0,
        "with_url_count_after": 0,
        "without_url_count_after": 0,
        "overflow_rows": 0,
    }
    if not stats["exists"]:
        return stats

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    if not isinstance(checkpoint, dict):
        stats["status"] = "invalid_json"
        return stats

    stats["version_before"] = checkpoint.get("version")
    raw_processed = checkpoint.get("processed_rows", {})
    processed_in = raw_processed if isinstance(raw_processed, dict) else {}
    stats["processed_count_before"] = len(processed_in)

    processed_out: Dict[str, List[str]] = {}
    for idx_key, raw_value in processed_in.items():
        idx = normalize_text(idx_key)
        if not idx:
            continue
        full_links = normalize_links(raw_value, max_links=None)
        kept, extras = split_kept_extras(full_links, max_links=MAX_LINKS_PER_ROW)
        processed_out[idx] = kept
        if extras:
            stats["overflow_rows"] += 1
            add_overflow_record(
                overflow_records=overflow_records,
                source_file=os.path.basename(checkpoint_file),
                record_id=idx,
                query_or_row=idx,
                kept=kept,
                extras=extras,
            )

    total_candidates = int(checkpoint.get("total_candidates", 0) or 0)
    processed_count = len(processed_out)
    with_url_count = sum(1 for links in processed_out.values() if links)
    without_url_count = processed_count - with_url_count
    remaining_count = max(0, total_candidates - processed_count)

    checkpoint_out = dict(checkpoint)
    checkpoint_out["version"] = 2
    checkpoint_out["processed_rows"] = processed_out
    checkpoint_out["summary"] = {
        "processed_count": processed_count,
        "with_url_count": with_url_count,
        "without_url_count": without_url_count,
        "remaining_count": remaining_count,
    }
    checkpoint_out["updated_at"] = utc_now_iso()

    stats["processed_count_after"] = processed_count
    stats["with_url_count_after"] = with_url_count
    stats["without_url_count_after"] = without_url_count
    stats["status"] = "migrated"
    stats["migrated_checkpoint"] = checkpoint_out
    return stats


def migrate_cache_file(
    cache_file: str,
    overflow_records: List[Dict[str, str]],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "path": cache_file,
        "exists": os.path.exists(cache_file),
        "status": "missing",
        "entries_before": 0,
        "entries_after": 0,
        "overflow_entries": 0,
    }
    if not stats["exists"]:
        return stats

    with open(cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if not isinstance(cache, dict):
        stats["status"] = "invalid_json"
        return stats

    migrated_cache: Dict[str, List[str]] = {}
    stats["entries_before"] = len(cache)
    for query, raw_value in cache.items():
        if not isinstance(query, str):
            continue
        full_links = normalize_links(raw_value, max_links=None)
        kept, extras = split_kept_extras(full_links, max_links=MAX_LINKS_PER_ROW)
        migrated_cache[query] = kept
        if extras:
            stats["overflow_entries"] += 1
            add_overflow_record(
                overflow_records=overflow_records,
                source_file=os.path.basename(cache_file),
                record_id=str(len(migrated_cache) - 1),
                query_or_row=query,
                kept=kept,
                extras=extras,
            )

    stats["entries_after"] = len(migrated_cache)
    stats["status"] = "migrated"
    stats["migrated_cache"] = migrated_cache
    return stats


def write_overflow_csv(path: str, overflow_records: List[Dict[str, str]]) -> None:
    fieldnames = ["source_file", "record_id", "query_or_row", "kept_1", "kept_2", "extras_json"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in overflow_records:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migra TRF1 news schema: noticia -> noticia_1/noticia_2 com auditoria."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Simula e gera relatorio sem gravar arquivos-alvo.")
    mode.add_argument("--apply", action="store_true", help="Aplica migracao e grava arquivos-alvo.")
    parser.add_argument("--timestamp", default="", help="Timestamp fixo para nomes de backup/relatorio.")
    parser.add_argument("--csv-file", default=DEFAULT_CSV_FILE)
    parser.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "apply" if args.apply else "dry-run"
    timestamp = normalize_text(args.timestamp) or datetime.now().strftime("%Y%m%d_%H%M%S")

    overflow_records: List[Dict[str, str]] = []
    backups: List[Dict[str, str]] = []

    csv_stats = migrate_csv_file(args.csv_file, overflow_records)
    checkpoint_stats = migrate_checkpoint_file(args.checkpoint_file, overflow_records)
    cache_stats = migrate_cache_file(args.cache_file, overflow_records)

    if args.apply:
        for path in (args.csv_file, args.cache_file, args.checkpoint_file):
            backup_path = backup_file(path, timestamp)
            if backup_path:
                backups.append({"source": path, "backup": backup_path})

        if csv_stats.get("status") == "migrated":
            write_csv_atomic(args.csv_file, csv_stats["output_fieldnames"], csv_stats["migrated_rows"])
        if checkpoint_stats.get("status") == "migrated":
            atomic_write_json(args.checkpoint_file, checkpoint_stats["migrated_checkpoint"], indent=2)
        if cache_stats.get("status") == "migrated":
            atomic_write_json(args.cache_file, cache_stats["migrated_cache"], indent=2)

    overflow_file = f"trf1_news_migration_overflow_{timestamp}.csv"
    report_file = f"trf1_news_migration_report_{timestamp}.json"
    write_overflow_csv(overflow_file, overflow_records)

    report = {
        "timestamp": timestamp,
        "mode": mode,
        "defaults": {
            "max_links_per_row": MAX_LINKS_PER_ROW,
            "legacy_separator": LEGACY_LINK_SEPARATOR,
        },
        "paths": {
            "csv_file": args.csv_file,
            "cache_file": args.cache_file,
            "checkpoint_file": args.checkpoint_file,
            "overflow_file": overflow_file,
            "report_file": report_file,
        },
        "csv": {k: v for k, v in csv_stats.items() if not k.startswith("migrated_")},
        "checkpoint": {k: v for k, v in checkpoint_stats.items() if not k.startswith("migrated_")},
        "cache": {k: v for k, v in cache_stats.items() if not k.startswith("migrated_")},
        "overflow_records_count": len(overflow_records),
        "backups": backups,
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(report_file, report, indent=2)

    print("=" * 72)
    print(f"Migracao TRF1 news schema v2 | modo={mode}")
    print(f"CSV: {csv_stats.get('status')} | linhas={csv_stats.get('rows_total', 0)}")
    print(
        "Checkpoint: "
        f"{checkpoint_stats.get('status')} | processados={checkpoint_stats.get('processed_count_after', 0)}"
    )
    print(f"Cache: {cache_stats.get('status')} | entradas={cache_stats.get('entries_after', 0)}")
    print(f"Overflow auditado: {len(overflow_records)}")
    print(f"Relatorio: {report_file}")
    print(f"Auditoria overflow: {overflow_file}")
    if backups:
        print(f"Backups gerados: {len(backups)}")
    else:
        print("Backups gerados: 0")
    print("=" * 72)


if __name__ == "__main__":
    main()
