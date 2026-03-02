#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = [
    BASE_DIR / "DJe - 1ª semana - FEV_26.csv",
    BASE_DIR / "DJe - 2ª semana - FEV_26.csv",
]
OUTPUT_FILE = BASE_DIR / "DJe - consolidado - FEV_26.csv"
COLUMNS_TO_EXCLUDE = {f"noticia_geral_{i}" for i in range(3, 10)}


def ensure_inputs_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        missing_list = "\n- ".join(missing)
        raise FileNotFoundError(f"Arquivos nao encontrados:\n- {missing_list}")


def collect_fieldnames(paths: list[Path]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                continue
            for name in reader.fieldnames:
                if name and name not in seen:
                    seen.add(name)
                    fieldnames.append(name)

    return [name for name in fieldnames if name not in COLUMNS_TO_EXCLUDE]


def consolidate_csv(paths: list[Path], output_path: Path) -> tuple[int, list[str]]:
    final_fieldnames = collect_fieldnames(paths)
    total_rows = 0

    with output_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=final_fieldnames, extrasaction="ignore")
        writer.writeheader()

        for path in paths:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    writer.writerow({field: row.get(field, "") for field in final_fieldnames})
                    total_rows += 1

    return total_rows, final_fieldnames


def main() -> None:
    ensure_inputs_exist(INPUT_FILES)
    total_rows, final_fieldnames = consolidate_csv(INPUT_FILES, OUTPUT_FILE)

    print(f"Arquivo consolidado criado: {OUTPUT_FILE.name}")
    print(f"Total de linhas consolidadas: {total_rows}")
    print(f"Total de colunas no consolidado: {len(final_fieldnames)}")
    print("Colunas removidas: noticia_geral_3 a noticia_geral_9")


if __name__ == "__main__":
    main()
