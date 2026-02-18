#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divide temas_selec_TSE_all_4.csv em 3 partes."
    )
    parser.add_argument(
        "--csv-in",
        default="temas_selec_TSE_all_4.csv",
        help="CSV de entrada (default: temas_selec_TSE_all_4.csv)",
    )
    return parser.parse_args()


def chunk_sizes(total_rows: int, parts: int) -> List[int]:
    base = total_rows // parts
    extra = total_rows % parts
    return [base + (1 if i < extra else 0) for i in range(parts)]


def main() -> None:
    args = parse_args()
    input_path = Path(args.csv_in)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError("CSV sem cabeçalho.")

    sizes = chunk_sizes(len(rows), 3)
    stem = input_path.stem
    suffix = input_path.suffix or ".csv"

    start = 0
    for i, size in enumerate(sizes, start=1):
        end = start + size
        out_rows = rows[start:end]
        out_path = input_path.with_name(f"{stem}_part_{i}{suffix}")

        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

        print(f"{out_path.name}: {len(out_rows)} linhas")
        start = end


if __name__ == "__main__":
    main()

