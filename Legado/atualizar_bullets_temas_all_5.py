#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Atualiza apenas a coluna `bullet_points` dos arquivos:
- temas_selec_TSE_all_5.csv
- temas_selec_TSE_all_5_part_1.csv
- temas_selec_TSE_all_5_part_2.csv
- temas_selec_TSE_all_5_part_3.csv

Regra:
- vírgulas internas -> " - "
- preserva vírgula separadora entre bullets
  (no fim da linha imediatamente anterior ao próximo bullet)
- preserva vírgula inline antes de bullet (", •")
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


DEFAULT_FILES = [
    "temas_selec_TSE_all_5.csv",
    "temas_selec_TSE_all_5_part_1.csv",
    "temas_selec_TSE_all_5_part_2.csv",
    "temas_selec_TSE_all_5_part_3.csv",
]

TOKEN_BSEP = "__BSEP__"


def _next_non_empty(lines: Sequence[str], idx: int) -> int:
    for j in range(idx + 1, len(lines)):
        if lines[j].strip():
            return j
    return -1


def normalize_bullet_points(text: str) -> Tuple[str, int, int]:
    if not text:
        return text, 0, 0

    lines = text.splitlines()
    if not lines:
        return text, 0, 0

    total_subs = 0
    separators_preserved = 0
    out: List[str] = []

    for i, line in enumerate(lines):
        work = line.rstrip()
        next_idx = _next_non_empty(lines, i)
        next_is_bullet = next_idx >= 0 and lines[next_idx].lstrip().startswith("•")

        # Preserva separador inline ", •".
        work = re.sub(r",(?=\s*•)", TOKEN_BSEP, work)

        # Preserva separador final antes do próximo bullet.
        if next_is_bullet and work.strip():
            if re.search(r",\s*$", work):
                work = re.sub(r",\s*$", TOKEN_BSEP, work)
            elif re.search(r"\s-\s*$", work):
                # Restaura casos já alterados indevidamente em execuções antigas.
                work = re.sub(r"\s-\s*$", TOKEN_BSEP, work)

        new_line, n = re.subn(r"\s*,\s*", " - ", work)
        total_subs += n

        if TOKEN_BSEP in new_line:
            separators_preserved += new_line.count(TOKEN_BSEP)
            new_line = new_line.replace(TOKEN_BSEP, ",")

        out.append(new_line)

    return "\n".join(out), total_subs, separators_preserved


def process_file(path: Path, dry_run: bool) -> Tuple[int, int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "bullet_points" not in fieldnames:
        raise RuntimeError(f"Coluna 'bullet_points' não encontrada em {path.name}")

    changed_rows = 0
    replaced_commas = 0
    preserved_separators = 0

    for row in rows:
        original = row.get("bullet_points", "") or ""
        updated, subs, seps = normalize_bullet_points(original)
        replaced_commas += subs
        preserved_separators += seps
        if updated != original:
            row["bullet_points"] = updated
            changed_rows += 1

    if not dry_run:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)

    return changed_rows, replaced_commas, preserved_separators


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrige vírgulas internas de bullet_points em temas_selec_TSE_all_5 e partes."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Arquivos CSV alvo (default: all_5 + part_1 + part_2 + part_3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas simula e mostra contagens, sem reescrever arquivos.",
    )
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    args = parse_args()

    total_changed = 0
    total_subs = 0
    total_seps = 0

    for name in args.files:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        changed, subs, seps = process_file(path=path, dry_run=args.dry_run)
        total_changed += changed
        total_subs += subs
        total_seps += seps

        print(
            f"{path.name}: linhas_alteradas={changed} | "
            f"virgulas_substituidas={subs} | separadores_preservados={seps}"
        )

    mode = "DRY-RUN" if args.dry_run else "APLICADO"
    print(
        f"TOTAL [{mode}]: linhas_alteradas={total_changed} | "
        f"virgulas_substituidas={total_subs} | separadores_preservados={total_seps}"
    )


if __name__ == "__main__":
    main()
