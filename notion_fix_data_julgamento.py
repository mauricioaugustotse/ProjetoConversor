#!/usr/bin/env python3
"""
Replace `data_julgamento` in the current TRF1 CSV with exact values from OLD CSV.

Default behavior:
- Reads:
  - boletins_de_jurisprudencia_TRF1_with_news.csv
  - boletins_de_jurisprudencia_TRF1_with_news_OLD.csv
- Replaces `data_julgamento` row by row.
- Creates a backup of the target file.
- Writes the updated data back to the target file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd


DEFAULT_TARGET = "boletins_de_jurisprudencia_TRF1_with_news.csv"
DEFAULT_SOURCE = "boletins_de_jurisprudencia_TRF1_with_news_OLD.csv"
COLUMN_NAME = "data_julgamento"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Substitute `data_julgamento` values in target CSV with values "
            "from source CSV, row by row."
        )
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"CSV to be updated (default: {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"CSV source for exact dates (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional output CSV path. If omitted, updates the target file "
            "in place."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backup when output is the same as target.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_frames(target_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    if COLUMN_NAME not in target_df.columns:
        raise ValueError(f"Column `{COLUMN_NAME}` not found in target CSV.")
    if COLUMN_NAME not in source_df.columns:
        raise ValueError(f"Column `{COLUMN_NAME}` not found in source CSV.")

    if len(target_df) != len(source_df):
        raise ValueError(
            "CSV row count mismatch: "
            f"target={len(target_df)} source={len(source_df)}. "
            "Row-by-row replacement requires equal lengths."
        )


def replace_dates_row_by_row(target_df: pd.DataFrame, source_df: pd.DataFrame) -> int:
    changed = 0
    for idx in range(len(target_df)):
        old_value = target_df.at[idx, COLUMN_NAME]
        new_value = source_df.at[idx, COLUMN_NAME]
        if old_value != new_value:
            changed += 1
        target_df.at[idx, COLUMN_NAME] = new_value
    return changed


def create_backup(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.backup_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    args = parse_args()

    target_path = Path(args.target)
    source_path = Path(args.source)
    output_path = Path(args.output) if args.output else target_path

    target_df = read_csv(target_path)
    source_df = read_csv(source_path)
    validate_frames(target_df, source_df)

    backup_path = None
    same_output = output_path.resolve() == target_path.resolve()
    if same_output and not args.no_backup:
        backup_path = create_backup(target_path)

    changed = replace_dates_row_by_row(target_df, source_df)
    target_df.to_csv(output_path, index=False)

    print(f"Rows processed: {len(target_df)}")
    print(f"Values changed in `{COLUMN_NAME}`: {changed}")
    print(f"Output file: {output_path}")
    if backup_path is not None:
        print(f"Backup file: {backup_path}")


if __name__ == "__main__":
    main()
