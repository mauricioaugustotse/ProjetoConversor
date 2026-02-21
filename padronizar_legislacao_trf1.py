#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Padroniza a coluna `legislacao` do CSV TRF1.

Regras principais:
- separador entre dispositivos: virgula;
- remove dispositivos genericos sem artigo (ex.: "CF/88", "CPC/2015");
- excecao: manter EC e Lei/LC numeradas mesmo sem artigo;
- normaliza "artigo" -> "art.", "inciso" -> "inc.", "paragrafo" -> "§";
- adiciona ordinal em artigo < 10 (ex.: "art. 5º");
- evita virgula dentro do dispositivo padronizado.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


GENERIC_BASES = {
    "ADCT",
    "CC",
    "CDC",
    "CF/88",
    "CLT",
    "CP",
    "CPC",
    "CPC/1973",
    "CPC/2015",
    "CPP",
    "CTB",
    "CTN",
    "ECA",
    "LEP",
    "RITRF1",
}

REPLACEMENTS_PREFIX: Sequence[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"(?i)^\s*constitui[cç][aã]o federal(?: da rep[úu]blica)?(?: de)?(?: 1988)?\b"), "CF/88"),
    (re.compile(r"(?i)^\s*cf\s*/?\s*(?:1988|88)\b"), "CF/88"),
    (re.compile(r"(?i)^\s*cpc\s*/?\s*2015\b"), "CPC/2015"),
    (re.compile(r"(?i)^\s*cpc\s*/?\s*1973\b"), "CPC/1973"),
    (re.compile(r"(?i)^\s*c[oó]digo de processo civil\b"), "CPC"),
    (re.compile(r"(?i)^\s*c[oó]digo penal(?:\s+cp)?\b"), "CP"),
    (re.compile(r"(?i)^\s*c[oó]digo de processo penal(?:\s+cpp)?\b"), "CPP"),
    (re.compile(r"(?i)^\s*c[oó]digo tribut[aá]rio nacional\b"), "CTN"),
    (re.compile(r"(?i)^\s*c[oó]digo de defesa do consumidor\b"), "CDC"),
    (re.compile(r"(?i)^\s*c[oó]digo de tr[aâ]nsito brasileiro\b"), "CTB"),
    (re.compile(r"(?i)^\s*c[oó]digo civil\s*1916\b"), "CC/1916"),
    (re.compile(r"(?i)^\s*c[oó]digo civil\s*2002\b"), "CC/2002"),
    (re.compile(r"(?i)^\s*c[oó]digo civil\b"), "CC"),
    (re.compile(r"(?i)^\s*ritrf\s*[-/ ]?\s*1\b"), "RITRF1"),
)

RE_WHITESPACE = re.compile(r"\s+")
RE_LEI_HEAD = re.compile(
    r"(?i)^\s*(lei(?:\s+complementar)?)\s*(?:n(?:º|°|o|\.)?\s*)?"
    r"(\d{1,6}(?:\.\d{3})*|\d+)\s*(.*)$"
)
RE_EC_HEAD = re.compile(
    r"(?i)^\s*(?:ec|emenda\s+constitucional)\s*(?:n(?:º|°|o|\.)?\s*)?"
    r"(\d{1,3})\s*(.*)$"
)
RE_DL_HEAD = re.compile(
    r"(?i)^\s*(?:dl|decreto[- ]?lei)\s*(?:n(?:º|°|o|\.)?\s*)?"
    r"(\d{1,6}(?:\.\d{3})*|\d+)\s*(.*)$"
)
RE_DECRETO_HEAD = re.compile(
    r"(?i)^\s*(decreto)\s*(?:n(?:º|°|o|\.)?\s*)?"
    r"(\d{1,6}(?:\.\d{3})*|\d+)\s*(.*)$"
)
RE_MPV_HEAD = re.compile(
    r"(?i)^\s*(?:medida\s+provis[óo]ria|mp)\s*(?:n(?:º|°|o|\.)?\s*)?"
    r"(\d{1,6}(?:\.\d{3})*(?:-\d{1,3})?)\s*(.*)$"
)
RE_TAIL_YEAR_SEP = re.compile(r"^[/-]\s*(\d{2,4})\b\s*(.*)$")
RE_TAIL_YEAR_DIRECT = re.compile(r"^(\d{4})\b\s*(.*)$")

RE_ART_TOKEN = re.compile(r"(?i)\b(article|artigo|artigos|arts?|art\.?)\b")
RE_INC_TOKEN = re.compile(r"(?i)\bincisos?|inc\.?\b")
RE_AL_TOKEN = re.compile(r"(?i)\bal[ií]neas?|alnea|al\.?\b")
RE_PAR_UNICO = re.compile(r"(?i)\bpar[aá]grafo\s+u[nn][ií]co\b")
RE_PAR_TOKEN = re.compile(r"(?i)\bpar[aá]grafos?\b")
RE_PAR_ABBR = re.compile(r"(?i)\bpar\.?\b")
RE_CAPUT = re.compile(r"(?i)\bcaput\b")
RE_SYMBOL = re.compile(r"[–—]")
RE_COMMAS = re.compile(r",+")
RE_ART_DOUBLE_DOT = re.compile(r"(?i)\bart\.\.+")
RE_INC_DOUBLE_DOT = re.compile(r"(?i)\binc\.\.+")
RE_AL_DOUBLE_DOT = re.compile(r"(?i)\bal\.\.+")
RE_ORDINAL_DOT_SYMBOL = re.compile(r"(\d)\.\s*[º°]")
RE_ORDINAL_SYMBOL_DOT = re.compile(r"(\d)\s*[º°]\.")
RE_ORDINAL_LETTER_O = re.compile(r"(?i)(\d)\s*o\b")
RE_ART_SPACE_THOUSANDS = re.compile(r"(?i)\bart\.\s*(\d)\s+(\d{3})\b")
RE_ART_DECIMAL_SPACE = re.compile(r"(?i)\bart\.\s*(\d+)\s+(\d{3})\b")
RE_ART_ROMAN_SUFFIX_HIGH = re.compile(r"(?i)\bart\.\s*([4-9]\d{2,3})\s+([ivxlcdm])\b")
RE_ART_ROMAN = re.compile(r"(?i)(art\.\s*\d+(?:\.\d+)?(?:-[A-Za-z])?)\s+([ivxlcdm]{1,8})\b")
RE_PAR_ROMAN = re.compile(r"(?i)(§\s*\d+º?)\s+([ivxlcdm]{1,8})\b")
RE_ART_SUFFIX = re.compile(r"(?i)\bart\.\s*(\d+)\s+([A-HJ-UWYZ])\b")
RE_ART_ORDINAL = re.compile(r"(?i)\bart\.\s*([1-9])(?![\dº°.])\b")
RE_PAR_ORDINAL = re.compile(r"(?i)§\s*([1-9])(?![\dº°])\b")
RE_INC_ROMAN = re.compile(r"(?i)\binc\.\s*([ivxlcdm]{1,8})\b")
RE_AL_LETTER = re.compile(r"(?i)\bal\.\s*([a-z])\b")
RE_INC_WITH_AL = re.compile(r"(?i)\binc\.\s*([ivxlcdm]{1,8})\s+([a-z])\b")
RE_HAS_ART = re.compile(r"(?i)\bart\.\s*\d")
RE_HAS_PAR = re.compile(r"(?i)§\s*\d")
RE_HAS_INC = re.compile(r"(?i)\binc\.\s*[ivxlcdm\d]+")
RE_ONLY_JUNK = re.compile(r"^[\W_]*$")
RE_ONLY_NUMBER = re.compile(r"^\d+$")
RE_LEI_CANON = re.compile(r"^(Lei nº|LC nº)\s+\d")
RE_EC_CANON = re.compile(r"^EC\s+\d")
RE_BASE_TOKEN = re.compile(r"^([A-Z]{2,}(?:/[0-9]{2,4})?)\b")
RE_INLINE_LEI_SPACED_YEAR = re.compile(
    r"(?i)\b(Lei(?:\s+Complementar)?\s*(?:n(?:º|°|o|\.)?\s*)?)"
    r"(\d{1,6}(?:\.\d{3})*|\d+)\s+(\d{4})\b"
)
RE_INLINE_EC_SPACED_YEAR = re.compile(r"(?i)\bEC\s+(\d{1,3})\s+(\d{4})\b")
RE_PAR_O_UNICO = re.compile(r"(?i)§\s*(?:o\s+unico|o\s+nico|agro\s+unico|agro\s+nico)\b")
RE_PAR_S_NUMBER = re.compile(r"(?i)§\s*(?:o|s)\s+(\d+)\b")


@dataclass
class ItemResult:
    normalized: str
    action: str


def normalize_ws(value: Any) -> str:
    return RE_WHITESPACE.sub(" ", str(value or "")).strip()


def normalize_year(value: str) -> str:
    year = re.sub(r"\D", "", value or "")
    if not year:
        return ""
    if len(year) == 2:
        num = int(year)
        return str(1900 + num if num >= 30 else 2000 + num)
    if len(year) >= 4:
        return year[:4]
    return year


def format_norm_number(raw: str) -> str:
    num = normalize_ws(raw).replace(" ", "")
    num = re.sub(r"[^0-9.]", "", num)
    if not num:
        return ""
    if "." in num:
        return num
    if len(num) > 3:
        return f"{num[:-3]}.{num[-3:]}"
    return num


def format_mpv_number(raw: str) -> str:
    num = normalize_ws(raw).replace(" ", "")
    num = re.sub(r"[^0-9.\-]", "", num)
    return num


def split_year_and_tail(raw_tail: str) -> Tuple[str, str]:
    tail = normalize_ws(raw_tail)
    if not tail:
        return "", ""

    m_sep = RE_TAIL_YEAR_SEP.match(tail)
    if m_sep:
        year = normalize_year(m_sep.group(1))
        rest = normalize_ws(m_sep.group(2))
        return year, rest

    m_direct = RE_TAIL_YEAR_DIRECT.match(tail)
    if m_direct:
        year = normalize_year(m_direct.group(1))
        rest = normalize_ws(m_direct.group(2))
        return year, rest

    return "", tail


def normalize_prefix_aliases(item: str) -> str:
    text = normalize_ws(item)
    for regex, replacement in REPLACEMENTS_PREFIX:
        if regex.search(text):
            text = regex.sub(replacement, text, count=1)
            return normalize_ws(text)
    return text


def normalize_numbered_norm_prefix(item: str) -> str:
    text = normalize_ws(item)

    m_lei = RE_LEI_HEAD.match(text)
    if m_lei:
        lei_type = normalize_ws(m_lei.group(1)).lower()
        number = format_norm_number(m_lei.group(2))
        year, tail = split_year_and_tail(m_lei.group(3))
        prefix = "LC nº" if "complementar" in lei_type else "Lei nº"
        canon = f"{prefix} {number}"
        if year:
            canon += f"/{year}"
        return normalize_ws(f"{canon} {tail}")

    m_ec = RE_EC_HEAD.match(text)
    if m_ec:
        number = str(int(m_ec.group(1)))
        year, tail = split_year_and_tail(m_ec.group(2))
        canon = f"EC {number}"
        if year:
            canon += f"/{year}"
        return normalize_ws(f"{canon} {tail}")

    m_dl = RE_DL_HEAD.match(text)
    if m_dl:
        number = format_norm_number(m_dl.group(1))
        year, tail = split_year_and_tail(m_dl.group(2))
        canon = f"DL nº {number}"
        if year:
            canon += f"/{year}"
        return normalize_ws(f"{canon} {tail}")

    m_dec = RE_DECRETO_HEAD.match(text)
    if m_dec:
        number = format_norm_number(m_dec.group(2))
        year, tail = split_year_and_tail(m_dec.group(3))
        canon = f"Decreto nº {number}"
        if year:
            canon += f"/{year}"
        return normalize_ws(f"{canon} {tail}")

    m_mpv = RE_MPV_HEAD.match(text)
    if m_mpv:
        number = format_mpv_number(m_mpv.group(1))
        year, tail = split_year_and_tail(m_mpv.group(2))
        canon = f"MP nº {number}"
        if year:
            canon += f"/{year}"
        return normalize_ws(f"{canon} {tail}")

    return text


def normalize_inline_numbered_norms(item: str) -> str:
    def repl_lei(match: re.Match[str]) -> str:
        prefix_raw = normalize_ws(match.group(1)).lower()
        number = format_norm_number(match.group(2))
        year = normalize_year(match.group(3))
        prefix = "LC nº" if "complementar" in prefix_raw else "Lei nº"
        return f"{prefix} {number}/{year}"

    def repl_ec(match: re.Match[str]) -> str:
        number = str(int(match.group(1)))
        year = normalize_year(match.group(2))
        return f"EC {number}/{year}"

    text = RE_INLINE_LEI_SPACED_YEAR.sub(repl_lei, item)
    text = RE_INLINE_EC_SPACED_YEAR.sub(repl_ec, text)
    return text


def normalize_legal_tokens(item: str) -> str:
    text = normalize_ws(item)
    text = RE_SYMBOL.sub("-", text)
    text = RE_ORDINAL_DOT_SYMBOL.sub(r"\1º", text)
    text = RE_ORDINAL_SYMBOL_DOT.sub(r"\1º", text)
    text = RE_ORDINAL_LETTER_O.sub(r"\1º", text)
    text = RE_PAR_UNICO.sub("§ único", text)
    text = RE_PAR_TOKEN.sub("§", text)
    text = RE_PAR_ABBR.sub("§", text)
    text = RE_ART_TOKEN.sub("art.", text)
    text = RE_INC_TOKEN.sub("inc.", text)
    text = RE_AL_TOKEN.sub("al.", text)
    text = RE_CAPUT.sub("caput", text)
    text = RE_ART_DOUBLE_DOT.sub("art.", text)
    text = RE_INC_DOUBLE_DOT.sub("inc.", text)
    text = RE_AL_DOUBLE_DOT.sub("al.", text)

    # Pontuacao fixa dos marcadores.
    text = re.sub(r"(?i)\bart\.\s*", "art. ", text)
    text = re.sub(r"(?i)\binc\.\s*", "inc. ", text)
    text = re.sub(r"(?i)\bal\.\s*", "al. ", text)
    text = re.sub(r"\s*§\s*", " § ", text)

    # OCR comum: "art. 1 022" -> "art. 1.022".
    text = RE_ART_SPACE_THOUSANDS.sub(r"art. \1.\2", text)
    text = RE_ART_DECIMAL_SPACE.sub(r"art. \1.\2", text)

    # Casos como "art. 475 L" (sufixo de artigo, nao inciso).
    text = RE_ART_ROMAN_SUFFIX_HIGH.sub(lambda m: f"art. {m.group(1)}-{m.group(2).upper()}", text)

    # "art. 5 V" -> "art. 5 inc. V".
    text = RE_ART_ROMAN.sub(lambda m: f"{m.group(1)} inc. {m.group(2).upper()}", text)
    text = RE_PAR_ROMAN.sub(lambda m: f"{m.group(1)} inc. {m.group(2).upper()}", text)

    # "art. 334 A" -> "art. 334-A".
    text = RE_ART_SUFFIX.sub(lambda m: f"art. {m.group(1)}-{m.group(2).upper()}", text)

    # Ordinal para artigos < 10.
    text = RE_ART_ORDINAL.sub(r"art. \1º", text)
    text = RE_PAR_ORDINAL.sub(r"§ \1º", text)

    # Caixa de incisos/alineas.
    text = RE_INC_ROMAN.sub(lambda m: f"inc. {m.group(1).upper()}", text)
    text = RE_INC_WITH_AL.sub(lambda m: f"inc. {m.group(1).upper()} al. {m.group(2).lower()}", text)
    text = RE_AL_LETTER.sub(lambda m: f"al. {m.group(1).lower()}", text)
    text = RE_PAR_O_UNICO.sub("§ único", text)
    text = RE_PAR_S_NUMBER.sub(r"§ \1", text)
    text = RE_PAR_ORDINAL.sub(r"§ \1º", text)
    text = re.sub(r"(?i)art\.\s*\.\s*", "art. ", text)
    text = re.sub(r"(?i)inc\.\s*\.\s*", "inc. ", text)
    text = re.sub(r"(?i)al\.\s*\.\s*", "al. ", text)

    # Limpeza final.
    text = RE_COMMAS.sub(" ", text)
    text = normalize_ws(text)
    text = text.strip(" ;.-")
    text = normalize_ws(text)
    return text


def has_article_reference(item: str) -> bool:
    text = normalize_ws(item)
    return bool(RE_HAS_ART.search(text) or RE_HAS_PAR.search(text) or RE_HAS_INC.search(text))


def is_numbered_law_or_ec(item: str) -> bool:
    text = normalize_ws(item)
    return bool(RE_LEI_CANON.search(text) or RE_EC_CANON.search(text))


def get_base_token(item: str) -> str:
    m = RE_BASE_TOKEN.search(normalize_ws(item))
    if not m:
        return ""
    return m.group(1).strip()


def should_drop_item(item: str) -> bool:
    text = normalize_ws(item)
    if not text:
        return True
    if RE_ONLY_JUNK.search(text):
        return True
    if RE_ONLY_NUMBER.search(text):
        return True
    if text.startswith(("§", "inc.", "al.")):
        return True
    if is_numbered_law_or_ec(text):
        return False
    base = get_base_token(text)
    if base in GENERIC_BASES and not has_article_reference(text):
        return True
    return False


def normalize_item(raw_item: str) -> ItemResult:
    original = normalize_ws(raw_item)
    if not original:
        return ItemResult(normalized="", action="drop_empty")

    text = normalize_prefix_aliases(original)
    text = normalize_numbered_norm_prefix(text)
    text = normalize_inline_numbered_norms(text)
    text = normalize_legal_tokens(text)
    text = normalize_ws(text)

    if should_drop_item(text):
        return ItemResult(normalized="", action="drop_generic_or_invalid")
    if text != original:
        return ItemResult(normalized=text, action="normalized")
    return ItemResult(normalized=text, action="unchanged")


def normalize_cell(raw_cell: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    original = normalize_ws(raw_cell)
    if not original:
        return "", []

    raw_items = [normalize_ws(part) for part in original.split(",")]
    raw_items = [item for item in raw_items if item]

    normalized_items: List[str] = []
    seen: set[str] = set()
    details: List[Tuple[str, str, str]] = []

    for raw_item in raw_items:
        result = normalize_item(raw_item)
        out = result.normalized
        action = result.action
        details.append((raw_item, out, action))
        if not out:
            continue
        if out not in seen:
            normalized_items.append(out)
            seen.add(out)

    return ", ".join(normalized_items), details


def read_csv_rows(path: Path, encoding: str = "utf-8-sig") -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]], encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Padroniza a coluna legislacao do CSV TRF1.")
    parser.add_argument("--input", default="boletins_de_jurisprudencia_TRF1_with_news.csv", help="CSV de entrada.")
    parser.add_argument("--output", default="", help="CSV de saida. Sem --inplace, usa sufixo _legislacao_padronizada.")
    parser.add_argument("--column", default="legislacao", help='Nome da coluna alvo. Padrao: "legislacao".')
    parser.add_argument("--inplace", action="store_true", help="Sobrescreve arquivo de entrada (cria backup automatico).")
    parser.add_argument("--dry-run", action="store_true", help="Nao grava CSV de saida; apenas gera relatorios.")
    parser.add_argument("--report-dir", default="reports", help="Diretorio de relatorios.")
    parser.add_argument("--report-prefix", default="legislacao_padronizacao", help="Prefixo dos relatorios.")
    return parser.parse_args()


def determine_output_path(input_path: Path, output_arg: str, inplace: bool) -> Path:
    if inplace:
        return input_path
    if normalize_ws(output_arg):
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_legislacao_padronizada{input_path.suffix}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    output_path = determine_output_path(input_path, args.output, args.inplace)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = normalize_ws(args.report_prefix) or "legislacao_padronizacao"
    report_applied = report_dir / f"{prefix}_{timestamp}_aplicadas.csv"
    report_items = report_dir / f"{prefix}_{timestamp}_itens.csv"
    report_summary = report_dir / f"{prefix}_{timestamp}_summary.json"

    fieldnames, rows = read_csv_rows(input_path)
    if args.column not in fieldnames:
        raise KeyError(f'Coluna "{args.column}" nao encontrada. Colunas: {", ".join(fieldnames)}')

    rows_changed = 0
    cells_before_nonempty = 0
    cells_after_nonempty = 0
    pairs_counter: Counter[Tuple[str, str]] = Counter()
    item_counter: Counter[Tuple[str, str, str]] = Counter()
    drop_counter = 0

    unique_before: Counter[str] = Counter()
    unique_after: Counter[str] = Counter()

    for row in rows:
        before = normalize_ws(row.get(args.column, ""))
        if before:
            cells_before_nonempty += 1
            unique_before[before] += 1

        after, details = normalize_cell(before)
        row[args.column] = after

        if after:
            cells_after_nonempty += 1
            unique_after[after] += 1

        for raw_item, norm_item, action in details:
            item_counter[(raw_item, norm_item, action)] += 1
            if action.startswith("drop_"):
                drop_counter += 1

        if before != after:
            rows_changed += 1
            pairs_counter[(before, after)] += 1

    if not args.dry_run:
        if args.inplace:
            backup_path = input_path.with_name(f"{input_path.name}.backup_legislacao_{timestamp}")
            shutil.copy2(input_path, backup_path)
            print(f"Backup criado: {backup_path}")
        write_csv_rows(output_path, fieldnames, rows)

    applied_rows = [
        {"before": before, "after": after, "rows": count}
        for (before, after), count in sorted(pairs_counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ]
    write_csv_rows(report_applied, fieldnames=("before", "after", "rows"), rows=applied_rows)

    item_rows = [
        {"item_before": b, "item_after": a, "action": act, "occurrences": c}
        for (b, a, act), c in sorted(item_counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1], x[0][2]))
    ]
    write_csv_rows(report_items, fieldnames=("item_before", "item_after", "action", "occurrences"), rows=item_rows)

    summary = {
        "timestamp": timestamp,
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "output_written": not bool(args.dry_run),
        "column": args.column,
        "rows_total": len(rows),
        "rows_changed": rows_changed,
        "cells_nonempty_before": cells_before_nonempty,
        "cells_nonempty_after": cells_after_nonempty,
        "unique_cells_before": len(unique_before),
        "unique_cells_after": len(unique_after),
        "unique_pairs_changed": len(pairs_counter),
        "items_total_seen": int(sum(item_counter.values())),
        "items_dropped": int(drop_counter),
        "report_applied": str(report_applied),
        "report_items": str(report_items),
    }
    report_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("Resumo da padronizacao de legislacao:")
    print(f"- Linhas totais: {summary['rows_total']}")
    print(f"- Linhas alteradas: {summary['rows_changed']}")
    print(f"- Celulas nao vazias antes: {summary['cells_nonempty_before']}")
    print(f"- Celulas nao vazias depois: {summary['cells_nonempty_after']}")
    print(f"- Valores unicos antes: {summary['unique_cells_before']}")
    print(f"- Valores unicos depois: {summary['unique_cells_after']}")
    print(f"- Itens descartados: {summary['items_dropped']}")
    print(f"- CSV de saida gravado: {'sim' if summary['output_written'] else 'nao (dry-run)'}")
    print(f"- Relatorio de celulas: {report_applied}")
    print(f"- Relatorio de itens: {report_items}")
    print(f"- Resumo JSON: {report_summary}")


if __name__ == "__main__":
    main()
