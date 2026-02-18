#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple


DEFAULT_FILES = [
    "temas_selec_TSE_all_4_part_1.csv",
    "temas_selec_TSE_all_4_part_2.csv",
    "temas_selec_TSE_all_4_part_3.csv",
]


def _proximo_nao_vazio(linhas: List[str], idx: int) -> int:
    for j in range(idx + 1, len(linhas)):
        if linhas[j].strip():
            return j
    return -1


def normalizar_bullet_points(texto: str) -> Tuple[str, int, int]:
    if not texto:
        return texto, 0, 0

    total_subs = 0
    separadores_preservados = 0
    novas_linhas: List[str] = []
    linhas = texto.splitlines()
    token_sep = "__BSEP__"

    for i, linha in enumerate(linhas):
        prox = _proximo_nao_vazio(linhas, i)
        proxima_e_bullet = prox >= 0 and linhas[prox].lstrip().startswith("•")

        work = linha.rstrip()
        # Preserva separador inline ", • ...".
        work = re.sub(r",(?=\s*•)", token_sep, work)

        # Preserva/restaura o separador final antes do próximo bullet:
        # - se já era "," no fim, mantém;
        # - se virou " -" por execução anterior, restaura para ",".
        if proxima_e_bullet and work.strip():
            if re.search(r",\s*$", work):
                work = re.sub(r",\s*$", token_sep, work)
            elif re.search(r"\s-\s*$", work):
                work = re.sub(r"\s-\s*$", token_sep, work)

        nova_linha, n = re.subn(r"\s*,\s*", " - ", work)
        total_subs += n

        if token_sep in nova_linha:
            separadores_preservados += nova_linha.count(token_sep)
            nova_linha = nova_linha.replace(token_sep, ",")

        novas_linhas.append(nova_linha)
    return "\n".join(novas_linhas), total_subs, separadores_preservados


def processar_csv(path: Path) -> Tuple[int, int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "bullet_points" not in fieldnames:
        raise RuntimeError(f"Coluna 'bullet_points' não encontrada em {path.name}")

    linhas_alteradas = 0
    virgulas_substituidas = 0
    separadores_preservados = 0

    for row in rows:
        original = row.get("bullet_points", "") or ""
        novo, subs, sep_count = normalizar_bullet_points(original)
        if novo != original:
            row["bullet_points"] = novo
            linhas_alteradas += 1
        virgulas_substituidas += subs
        separadores_preservados += sep_count

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)

    return linhas_alteradas, virgulas_substituidas, separadores_preservados


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Substitui vírgulas internas da coluna bullet_points por ' - ' em 3 partes CSV."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Lista de CSVs a processar (default: as 3 partes de temas_selec_TSE_all_4).",
    )
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    args = parse_args()

    total_linhas = 0
    total_subs = 0
    total_sep = 0

    for nome in args.files:
        path = Path(nome)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        linhas_alt, subs, sep = processar_csv(path)
        total_linhas += linhas_alt
        total_subs += subs
        total_sep += sep
        print(
            f"{path.name}: linhas_alteradas={linhas_alt} | "
            f"virgulas_substituidas={subs} | separadores_preservados={sep}"
        )

    print(
        f"TOTAL: linhas_alteradas={total_linhas} | "
        f"virgulas_substituidas={total_subs} | separadores_preservados={total_sep}"
    )


if __name__ == "__main__":
    main()
