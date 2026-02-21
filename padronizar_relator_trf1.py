#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Padroniza nomes da coluna `relator(a)` em CSV do TRF1.

Objetivos:
- limpar prefixos/ruidos (ex.: "des.", "juiz federal", "p/ acordao");
- agrupar variantes da mesma pessoa e escolher forma por extenso;
- preservar sufixo "(convocado)" / "(convocada)";
- gerar relatorios de mudancas e ambiguidades.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore[assignment]


RELATOR_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:"
    r"(?:p\s*/\s*o\s+|p\s*/\s*|para\s+o\s+)ac[oó]rd[aã]o\s+|"
    r"rel\.?\s*(?:ator(?:\(a\)|a|o)?)?\s*[:.\-]?\s*|"
    r"des\.?\s*(?:fed(?:eral)?\.?)?\s+|"
    r"desembargador(?:a)?\s+|"
    r"min\.?\s+|ministro(?:a)?\s+|"
    r"ju[ií]z(?:a)?\.?\s+federal\s+|"
    r"ju[ií]z(?:a)?\.?\s+|"
    r"fed\.?\s+"
    r")+"
)
CONVOCADO_TAIL_RE = re.compile(r"(?i)\s*\(([^)]*convocad[oa][^)]*)\)\s*$")
CONVOCADO_WORD_TAIL_RE = re.compile(r"(?i)\s+(convocad[oa])\s*$")
DATE_TRAIL_RE = re.compile(r"(?i)\s+em\s+\d{1,2}/\d{1,2}/\d{2,4}.*$")
SESSION_TRAIL_RE = re.compile(r"(?i)\s+em\s+sess[aã]o\b.*$")
REGIME_TRAIL_RE = re.compile(r"(?i)\s+em\s+regime\b.*$")
PJE_TRAIL_RE = re.compile(r"(?i)\s*[-–]\s*pje\s*$")

WORD_RE = re.compile(r"[a-z0-9]{2,}")
STOPWORDS = {"de", "da", "do", "dos", "das", "e"}
NOISE_TOKENS = {
    "sessao",
    "virtual",
    "realizada",
    "regime",
    "auxilio",
    "distancia",
    "acordao",
    "pje",
    "telepresencial",
}

MAX_AMBIGUOUS_CANDIDATES = 5


@dataclass(frozen=True)
class NameNode:
    key: str
    display: str
    count: int
    tokens: Tuple[str, ...]


@dataclass
class CanonicalModel:
    nodes: Dict[str, NameNode]
    alias_by_key: Dict[str, str]
    reason_by_key: Dict[str, str]
    ambiguous_rows: List[Dict[str, Any]]


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def fold_text(value: str) -> str:
    text = normalize_ws(value).casefold()
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def sanitize_display(value: str) -> str:
    text = normalize_ws(value)
    text = text.replace("’", "'").replace("`", "'")
    return text


def tokenize_for_match(value: str) -> List[str]:
    text = fold_text(value)
    text = re.sub(r"[’'`]", "", text)
    tokens = WORD_RE.findall(text)
    return [tok for tok in tokens if tok not in STOPWORDS]


def is_noisy_name(name: str) -> bool:
    text = fold_text(name)
    if re.search(r"\d", text):
        return True
    tokens = tokenize_for_match(name)
    if not tokens:
        return True
    if len(tokens) > 8:
        return True
    return any(tok in NOISE_TOKENS for tok in tokens)


def clean_relator_value(raw_value: str) -> Tuple[str, str]:
    text = normalize_ws(raw_value)
    if not text:
        return "", ""

    conv = ""
    m = CONVOCADO_TAIL_RE.search(text)
    if m:
        marker = fold_text(m.group(1))
        conv = "convocada" if "convocada" in marker else "convocado"
        text = normalize_ws(text[: m.start()])
    else:
        m2 = CONVOCADO_WORD_TAIL_RE.search(text)
        if m2:
            marker = fold_text(m2.group(1))
            conv = "convocada" if "convocada" in marker else "convocado"
            text = normalize_ws(text[: m2.start()])

    text = PJE_TRAIL_RE.sub("", text)
    text = DATE_TRAIL_RE.sub("", text)
    text = SESSION_TRAIL_RE.sub("", text)
    text = REGIME_TRAIL_RE.sub("", text)

    prev = None
    while prev != text:
        prev = text
        text = RELATOR_PREFIX_RE.sub("", text)
        text = normalize_ws(text)

    text = normalize_ws(text).strip(" ,.;:-")
    text = sanitize_display(text)
    return text, conv


def choose_display_variant(variants: Mapping[str, int]) -> str:
    def score(name: str, count: int) -> Tuple[int, int, int, int, int]:
        tokens = tokenize_for_match(name)
        diacritics = sum(1 for ch in name if ord(ch) > 127)
        noisy = is_noisy_name(name)
        return (0 if noisy else 1, len(tokens), diacritics, count, len(name))

    best_name, _ = max(variants.items(), key=lambda kv: score(kv[0], int(kv[1])))
    return sanitize_display(best_name)


def load_text_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def load_manual_aliases(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload = load_text_or_json(path)
    if not isinstance(payload, dict):
        return {}
    raw_aliases = payload.get("aliases", payload)
    if not isinstance(raw_aliases, dict):
        return {}

    aliases: Dict[str, str] = {}
    for raw_key, raw_value in raw_aliases.items():
        src_base, _ = clean_relator_value(str(raw_key))
        dst_base, _ = clean_relator_value(str(raw_value))
        if not src_base or not dst_base:
            continue
        aliases[fold_text(src_base)] = sanitize_display(dst_base)
    return aliases


def build_nodes_from_counts(raw_counts: Mapping[str, int]) -> Dict[str, NameNode]:
    variants_by_key: Dict[str, Counter[str]] = defaultdict(Counter)
    for raw_value, count in raw_counts.items():
        base, _ = clean_relator_value(raw_value)
        if not base:
            continue
        variants_by_key[fold_text(base)][base] += int(count)

    nodes: Dict[str, NameNode] = {}
    for key, variants in variants_by_key.items():
        display = choose_display_variant(variants)
        total_count = int(sum(variants.values()))
        nodes[key] = NameNode(
            key=key,
            display=display,
            count=total_count,
            tokens=tuple(tokenize_for_match(display)),
        )
    return nodes


def build_canonical_model(nodes: Dict[str, NameNode], manual_aliases: Mapping[str, str]) -> CanonicalModel:
    working_nodes: Dict[str, NameNode] = dict(nodes)

    # Ensure manual targets exist as canonical nodes.
    for target_base in manual_aliases.values():
        target_key = fold_text(target_base)
        if target_key in working_nodes:
            continue
        display = sanitize_display(target_base)
        working_nodes[target_key] = NameNode(
            key=target_key,
            display=display,
            count=0,
            tokens=tuple(tokenize_for_match(display)),
        )

    by_first_token: Dict[str, List[str]] = defaultdict(list)
    for key, node in working_nodes.items():
        if node.tokens:
            by_first_token[node.tokens[0]].append(key)

    alias_by_key: Dict[str, str] = {}
    reason_by_key: Dict[str, str] = {}
    ambiguous_rows: List[Dict[str, Any]] = []

    # Manual aliases have priority.
    for src_key, target_base in manual_aliases.items():
        target_key = fold_text(target_base)
        alias_by_key[src_key] = target_key
        reason_by_key[src_key] = "manual"

    # Auto subset mapping.
    for key, node in working_nodes.items():
        if key in reason_by_key:
            continue

        source_set = set(node.tokens)
        if len(source_set) <= 1:
            alias_by_key[key] = key
            reason_by_key[key] = "self"
            continue

        first_token = node.tokens[0]
        candidates: List[str] = []
        for cand_key in by_first_token.get(first_token, []):
            cand_node = working_nodes[cand_key]
            target_set = set(cand_node.tokens)
            if source_set.issubset(target_set) and not is_noisy_name(cand_node.display):
                candidates.append(cand_key)

        if not candidates:
            alias_by_key[key] = key
            reason_by_key[key] = "self"
            continue

        candidates.sort(
            key=lambda cand: (
                len(set(working_nodes[cand].tokens)),
                working_nodes[cand].count,
                len(working_nodes[cand].display),
            ),
            reverse=True,
        )
        top_key = candidates[0]
        top_set = set(working_nodes[top_key].tokens)

        branched = False
        for alt_key in candidates[1:]:
            alt_set = set(working_nodes[alt_key].tokens)
            if not (alt_set.issubset(top_set) or top_set.issubset(alt_set)):
                branched = True
                break

        if branched:
            alias_by_key[key] = key
            reason_by_key[key] = "ambiguous"
            row: Dict[str, Any] = {
                "source_name": node.display,
                "source_count": node.count,
            }
            for idx, cand_key in enumerate(candidates[:MAX_AMBIGUOUS_CANDIDATES], start=1):
                row[f"candidate_{idx}"] = working_nodes[cand_key].display
            ambiguous_rows.append(row)
            continue

        alias_by_key[key] = top_key
        reason_by_key[key] = "auto_superset" if top_key != key else "self"

    # Typo-like fallback for remaining self mappings.
    for key, node in working_nodes.items():
        if reason_by_key.get(key) not in {"self", "ambiguous"}:
            continue
        if len(node.tokens) < 2:
            continue

        first_token = node.tokens[0]
        best_key = ""
        best_ratio = 0.0
        source_set = set(node.tokens)

        for cand_key in by_first_token.get(first_token, []):
            if cand_key == key:
                continue
            cand_node = working_nodes[cand_key]
            if is_noisy_name(cand_node.display):
                continue
            if cand_node.count < max(4, node.count * 3):
                continue
            if len(cand_node.tokens) < len(node.tokens):
                continue

            common_count = len(source_set & set(cand_node.tokens))
            ratio = SequenceMatcher(None, key, cand_key).ratio()
            if common_count >= len(node.tokens) - 1 and ratio >= 0.90 and ratio > best_ratio:
                best_key = cand_key
                best_ratio = ratio

        if best_key:
            alias_by_key[key] = best_key
            reason_by_key[key] = "auto_typo"

    return CanonicalModel(
        nodes=working_nodes,
        alias_by_key=alias_by_key,
        reason_by_key=reason_by_key,
        ambiguous_rows=ambiguous_rows,
    )


def normalize_relator_with_model(raw_value: str, model: CanonicalModel, convocado_mode: str) -> Tuple[str, str]:
    original = normalize_ws(raw_value)
    if not original:
        return "", "empty"

    base, conv = clean_relator_value(original)
    if not base:
        return original, "unresolved"

    src_key = fold_text(base)
    target_key = model.alias_by_key.get(src_key, src_key)
    target_node = model.nodes.get(target_key)
    canonical_base = target_node.display if target_node else sanitize_display(base)

    if conv:
        label = conv if convocado_mode == "preserve" else "convocado"
        normalized = f"{canonical_base} ({label})"
    else:
        normalized = canonical_base

    normalized = normalize_ws(normalized)
    reason = model.reason_by_key.get(src_key, "self")
    return normalized, reason


def read_csv_rows(path: Path, encoding: str = "utf-8-sig") -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]], encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def determine_output_path(input_path: Path, output_arg: str, inplace: bool) -> Path:
    if inplace:
        return input_path
    if normalize_ws(output_arg):
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_relator_padronizado{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Padroniza nomes da coluna relator(a) em CSV.")
    parser.add_argument(
        "--input",
        default="boletins_de_jurisprudencia_TRF1_with_news.csv",
        help="CSV de entrada.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="CSV de saida. Se omitido e sem --inplace, usa <input>_relator_padronizado.csv.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Sobrescreve o arquivo de entrada (com backup automatico).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nao grava CSV de saida; apenas gera relatorios e resumo.",
    )
    parser.add_argument(
        "--column",
        default="relator(a)",
        help='Nome da coluna alvo. Padrao: "relator(a)".',
    )
    parser.add_argument(
        "--alias-file",
        default="canon_config/relator_overrides.yaml",
        help="Arquivo YAML/JSON com aliases manuais (opcional).",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Diretorio para salvar relatorios.",
    )
    parser.add_argument(
        "--report-prefix",
        default="relator_padronizacao",
        help="Prefixo dos arquivos de relatorio.",
    )
    parser.add_argument(
        "--convocado-mode",
        choices=("preserve", "generic"),
        default="preserve",
        help='preserve = mantem convocado/convocada; generic = sempre "(convocado)".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    output_path = determine_output_path(input_path, args.output, args.inplace)
    alias_file_path = Path(args.alias_file)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_base = f"{normalize_ws(args.report_prefix) or 'relator_padronizacao'}_{timestamp}"
    applied_report_path = report_dir / f"{report_base}_aplicadas.csv"
    ambiguous_report_path = report_dir / f"{report_base}_ambiguas.csv"
    summary_json_path = report_dir / f"{report_base}_summary.json"

    fieldnames, rows = read_csv_rows(input_path)
    if args.column not in fieldnames:
        available = ", ".join(fieldnames)
        raise KeyError(f'Coluna "{args.column}" nao encontrada. Colunas disponiveis: {available}')

    raw_counts: Counter[str] = Counter()
    for row in rows:
        raw_counts[normalize_ws(row.get(args.column, ""))] += 1
    if "" in raw_counts:
        del raw_counts[""]

    manual_aliases = load_manual_aliases(alias_file_path)
    nodes = build_nodes_from_counts(raw_counts)
    model = build_canonical_model(nodes, manual_aliases)

    before_unique = len(raw_counts)
    after_counter: Counter[str] = Counter()
    pair_counter: Counter[Tuple[str, str, str]] = Counter()
    rows_changed = 0

    for row in rows:
        raw_original = row.get(args.column, "")
        original = normalize_ws(raw_original)
        normalized, reason = normalize_relator_with_model(raw_original, model, args.convocado_mode)
        row[args.column] = normalized
        after_counter[normalized] += 1
        if original != normalized:
            rows_changed += 1
            pair_counter[(original, normalized, reason)] += 1

    changed_rows_report: List[Dict[str, Any]] = []
    for (before, after, reason), count in sorted(pair_counter.items(), key=lambda item: (-item[1], item[0][0])):
        changed_rows_report.append(
            {
                "before": before,
                "after": after,
                "reason": reason,
                "rows": int(count),
            }
        )

    if not args.dry_run:
        if args.inplace:
            backup_path = input_path.with_name(f"{input_path.name}.backup_{timestamp}")
            shutil.copy2(input_path, backup_path)
            print(f"Backup criado: {backup_path}")
        write_csv_rows(output_path, fieldnames, rows)

    write_csv_rows(
        applied_report_path,
        fieldnames=("before", "after", "reason", "rows"),
        rows=changed_rows_report,
    )

    ambiguous_fieldnames = ["source_name", "source_count"] + [f"candidate_{idx}" for idx in range(1, MAX_AMBIGUOUS_CANDIDATES + 1)]
    ambiguous_rows_sorted = sorted(model.ambiguous_rows, key=lambda x: (-int(x.get("source_count", 0)), str(x.get("source_name", ""))))
    write_csv_rows(
        ambiguous_report_path,
        fieldnames=ambiguous_fieldnames,
        rows=ambiguous_rows_sorted,
    )

    summary = {
        "timestamp": timestamp,
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "output_written": not bool(args.dry_run),
        "rows_total": len(rows),
        "rows_changed": rows_changed,
        "unique_relator_before": before_unique,
        "unique_relator_after": len(after_counter),
        "unique_value_pairs_changed": len(changed_rows_report),
        "manual_aliases_loaded": len(manual_aliases),
        "ambiguous_groups_detected": len(model.ambiguous_rows),
        "applied_report": str(applied_report_path),
        "ambiguous_report": str(ambiguous_report_path),
        "convocado_mode": args.convocado_mode,
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("Resumo da padronizacao:")
    print(f"- Linhas totais: {summary['rows_total']}")
    print(f"- Linhas alteradas: {summary['rows_changed']}")
    print(f"- Valores unicos antes: {summary['unique_relator_before']}")
    print(f"- Valores unicos depois: {summary['unique_relator_after']}")
    print(f"- Pares de mudanca: {summary['unique_value_pairs_changed']}")
    print(f"- Aliases manuais carregados: {summary['manual_aliases_loaded']}")
    print(f"- Grupos ambiguos detectados: {summary['ambiguous_groups_detected']}")
    print(f"- CSV de saida gravado: {'sim' if summary['output_written'] else 'nao (dry-run)'}")
    print(f"- Relatorio de aplicadas: {applied_report_path}")
    print(f"- Relatorio de ambiguas: {ambiguous_report_path}")
    print(f"- Resumo JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
