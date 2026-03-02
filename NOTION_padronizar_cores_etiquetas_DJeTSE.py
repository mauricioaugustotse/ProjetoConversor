#!/usr/bin/env python3
"""Padroniza cores de etiquetas no Notion para a base DJe.

Regras atuais:
1) Nao altera a coluna `composicao`.
2) Replica cores de `composicao` para `relator`.
3) Padroniza `siglaUF` e `nomeMunicipio` por regiao do Brasil.
   - Sudeste => blue
   - Sul => green
   - Centro-Oeste => yellow
   - Nordeste => orange
   - Norte => red
4) Padroniza `partes` e `advogados` por ordem alfabetica (faixas de letras).
5) Padroniza `siglaClasse` e `descricaoClasse` por familias de classes processuais.
6) Sanea etiquetas: remove opcoes sem uso real nos registros da coluna.

Padrao de execucao: dry-run. Use --apply para efetivar alteracoes.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import re
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import requests


NOTION_BASE_URL = "https://api.notion.com"
DEFAULT_NOTION_VERSION = "2025-09-03"
DEFAULT_DATABASE_URL = "https://www.notion.so/316721955c6480b4af2cf19fa557a5dd?v=316721955c64816e8f6f000c06433647"
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT_S = 30
DEFAULT_RETRIES = 4
DEFAULT_MIN_INTERVAL_S = 0.20
NOTION_OPTIONS_PATCH_LIMIT = 100

LOGGER = logging.getLogger("NOTION_padronizar_cores_etiquetas_DJeTSE")

PHASE_1_PROPERTY_NAMES = ("siglaUF", "relator", "siglaClasse", "descricaoClasse")
PHASE_2_PROPERTY_NAMES = ("nomeMunicipio", "partes", "advogados")

SUPPORTED_COLORS = {
    "default",
    "gray",
    "brown",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "red",
}

REGION_TO_COLOR = {
    "norte": "red",
    "nordeste": "orange",
    "centro-oeste": "yellow",
    "sudeste": "blue",
    "sul": "green",
}

ALPHABETICAL_COLORS = [
    "blue",
    "green",
    "yellow",
    "orange",
    "red",
    "pink",
    "purple",
    "brown",
    "gray",
]

UF_TO_REGION = {
    "AC": "norte",
    "AP": "norte",
    "AM": "norte",
    "PA": "norte",
    "RO": "norte",
    "RR": "norte",
    "TO": "norte",
    "AL": "nordeste",
    "BA": "nordeste",
    "CE": "nordeste",
    "MA": "nordeste",
    "PB": "nordeste",
    "PE": "nordeste",
    "PI": "nordeste",
    "RN": "nordeste",
    "SE": "nordeste",
    "DF": "centro-oeste",
    "GO": "centro-oeste",
    "MT": "centro-oeste",
    "MS": "centro-oeste",
    "ES": "sudeste",
    "MG": "sudeste",
    "RJ": "sudeste",
    "SP": "sudeste",
    "PR": "sul",
    "RS": "sul",
    "SC": "sul",
}

RELATOR_LABEL_RE = re.compile(r"(?i)relator\(a\)\s*([^,;]+)")
MIN_TOKEN_RE = re.compile(r"(?i)\bmin\.?\s*[^,;]+")

# Familias para padronizacao de classe processual.
CLASS_FAMILY_TO_COLOR = {
    "contas": "yellow",
    "recursos": "blue",
    "tutelas": "orange",
    "contencioso_eleitoral": "red",
    "procedimental": "purple",
    "administrativo": "green",
    "outros": "gray",
}

SIGLA_CLASS_FAMILY_KEYS = {
    "contas": {"pc", "pce", "pc pp"},
    "recursos": {"arespel", "respel", "ro el", "rms"},
    "tutelas": {"tutantant", "tutcautant"},
    "contencioso_eleitoral": {"aije", "rp", "ajdescargele", "proppart"},
    "procedimental": {"confjurisd", "rcl", "petciv", "cumsen", "msciv", "hccrim"},
    "administrativo": {"ctael", "lt", "czer", "dp", "dpi"},
}


@dataclass
class TagProperty:
    name: str
    property_id: str
    prop_type: str
    options: List[Dict[str, Any]]


@dataclass
class PropertyPlan:
    property_name: str
    property_type: str
    desired_color_by_key: Dict[str, str]
    reason: str


@dataclass
class ScanData:
    city_to_uf: Dict[str, str]
    used_keys_by_property: Dict[str, set[str]]
    descricao_to_sigla_counts: Dict[str, Counter[str]]
    pages_scanned: int


@dataclass
class ResumePlanData:
    target_colors_by_property: Dict[str, Dict[str, str]]
    remove_keys_by_property: Dict[str, set[str]]
    rows_by_property: Dict[str, List[Dict[str, str]]]
    has_block_columns: bool


def _parse_only_properties(raw: str) -> set[str]:
    if not raw:
        return set()
    items = [_normalize_key(part) for part in str(raw).split(",")]
    return {item for item in items if item}


def _phase_property_keys(phase: str) -> set[str]:
    phase_1 = {
        _normalize_key("siglaUF"),
        _normalize_key("relator"),
        _normalize_key("siglaClasse"),
        _normalize_key("descricaoClasse"),
    }
    phase_2 = {
        _normalize_key("nomeMunicipio"),
        _normalize_key("partes"),
        _normalize_key("advogados"),
    }
    if phase == "1":
        return phase_1
    if phase == "2":
        return phase_2
    return phase_1 | phase_2


def _resolve_manual_plan_path(raw_path: str) -> Path:
    candidate = _normalize_ws(raw_path)
    if candidate:
        out = Path(candidate)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path.cwd() / f"notion_etiquetas_plano_manual_{stamp}.csv"
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _manual_plan_rows_for_property(
    *,
    prop: TagProperty,
    plan: PropertyPlan,
    used_keys: Optional[set[str]],
    prune_unused: bool,
    force_remove_keys: Optional[set[str]] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in prop.options:
        name = _normalize_ws(item.get("name"))
        if not name:
            continue
        key = _normalize_key(name)
        in_use = bool(used_keys and key in used_keys)
        current = _normalize_ws(item.get("color")) or "default"
        if current not in SUPPORTED_COLORS:
            current = "default"
        target = plan.desired_color_by_key.get(key) or current
        if target not in SUPPORTED_COLORS:
            target = current
        forced_remove = bool(force_remove_keys is not None and key in force_remove_keys)
        would_remove = forced_remove or bool(prune_unused and used_keys is not None and key not in used_keys)
        rows.append(
            {
                "coluna": plan.property_name,
                "etiqueta": name,
                "cor_atual": current,
                "cor_alvo": target,
                "em_uso": "1" if in_use else "0",
                "remover_se_apply": "1" if would_remove else "0",
            }
        )
    return rows


def _write_manual_plan_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = ["coluna", "etiqueta", "cor_atual", "cor_alvo", "em_uso", "remover_se_apply"]
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_phase2_blocks_dir(raw_path: str, fallback_parent: Path) -> Path:
    candidate = _normalize_ws(raw_path)
    if candidate:
        out = Path(candidate)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = fallback_parent / f"notion_etiquetas_fase2_blocos_{stamp}"
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def _requested_property_keys(args: argparse.Namespace) -> set[str]:
    phase_keys = _phase_property_keys(_normalize_ws(getattr(args, "phase", "all")) or "all")
    only_keys = _parse_only_properties(_normalize_ws(getattr(args, "only_properties", "")))
    if only_keys:
        return phase_keys & only_keys
    return set(phase_keys)


def _canonical_phase2_name_by_key() -> Dict[str, str]:
    return {_normalize_key(name): name for name in PHASE_2_PROPERTY_NAMES}


def _resolve_phase2_chat_output_dir(raw_path: str, fallback_parent: Path) -> Path:
    candidate = _normalize_ws(raw_path)
    if candidate:
        out = Path(candidate)
    else:
        # Fluxo simplificado: por padrao, grava no diretorio base (sem pasta extra por timestamp).
        out = fallback_parent
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_phase2_queue_file(raw_path: str, out_dir: Path) -> Path:
    candidate = _normalize_ws(raw_path)
    if candidate:
        out = Path(candidate)
    else:
        out = out_dir / "phase2_chat_queue.csv"
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _phase2_block_sort_key(path: Path) -> Tuple[int, str]:
    stem = _normalize_ws(path.stem)
    match = re.search(r"_bloco_(\d+)_", stem)
    if match:
        try:
            return (int(match.group(1)), _normalize_key(stem))
        except Exception:
            pass
    return (999999, _normalize_key(stem))


def _collect_phase2_block_csv_files(input_dir: Path) -> List[Path]:
    files = [
        path
        for path in input_dir.glob("phase2_*_bloco_*.csv")
        if path.is_file()
    ]
    files.sort(key=lambda path: (_normalize_key(path.name), _phase2_block_sort_key(path)))
    return files


def _infer_phase2_block_metadata_from_filename(path: Path) -> Tuple[str, str, str]:
    canonical = _canonical_phase2_name_by_key()
    stem = _normalize_ws(path.stem)
    match = re.match(r"phase2_(.+?)_bloco_(\d+)_(.+)$", stem)
    if not match:
        return ("", "", "")

    token = _normalize_key(match.group(1).replace("_", " "))
    prop_name = ""
    for key, name in canonical.items():
        if _normalize_key(name) == token or _sanitize_filename_token(name) == _sanitize_filename_token(match.group(1)):
            prop_name = name
            break
    block_id = _normalize_ws(match.group(2))
    range_label = _normalize_ws(match.group(3).replace("_", "-"))
    return (prop_name, block_id, range_label)


def _load_phase2_block_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"coluna", "etiqueta", "cor_atual", "cor_alvo", "remover_se_apply"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"CSV de bloco invalido ({path.name}): faltam colunas {', '.join(sorted(missing))}"
            )
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "coluna": _normalize_ws(row.get("coluna")),
                    "etiqueta": _normalize_ws(row.get("etiqueta")),
                    "cor_atual": _normalize_ws(row.get("cor_atual")) or "default",
                    "cor_alvo": _normalize_ws(row.get("cor_alvo")) or "default",
                    "em_uso": _normalize_ws(row.get("em_uso")),
                    "remover_se_apply": "1" if _to_bool_flag(row.get("remover_se_apply")) else "0",
                    "bloco": _normalize_ws(row.get("bloco")),
                    "faixa_alfabetica": _normalize_ws(row.get("faixa_alfabetica")),
                }
            )
    return rows


def _dedupe_labels(labels: List[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for label in labels:
        name = _normalize_ws(label)
        key = _normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(name)
    return result


def _build_phase2_default_payload(
    *,
    database_url: str,
    property_name: str,
    block_id: str,
    range_label: str,
    rows: List[Dict[str, str]],
    force_default: bool,
    include_removals: bool,
) -> Dict[str, Any]:
    if force_default:
        set_default_raw = [row.get("etiqueta", "") for row in rows]
    else:
        set_default_raw = [
            row.get("etiqueta", "")
            for row in rows
            if _normalize_ws(row.get("cor_atual")) != _normalize_ws(row.get("cor_alvo"))
        ]

    remove_unused_raw: List[str] = []
    if include_removals:
        remove_unused_raw = [
            row.get("etiqueta", "")
            for row in rows
            if _to_bool_flag(row.get("remover_se_apply"))
        ]

    set_default = _dedupe_labels(set_default_raw)
    remove_unused = _dedupe_labels(remove_unused_raw)
    return {
        "database_url": _normalize_ws(database_url),
        "property": property_name,
        "block": _normalize_ws(block_id),
        "range": _normalize_ws(range_label),
        "set_default": set_default,
        "remove_unused": remove_unused,
    }


def _build_phase2_prompt_text(payload: Dict[str, Any]) -> str:
    return (
        "Atualize APENAS a propriedade indicada abaixo na base do Notion.\n\n"
        "Regras obrigatorias:\n"
        "1) Para cada etiqueta em `set_default`, defina a cor para `default`.\n"
        "2) Para cada etiqueta em `remove_unused`, remova a opcao da propriedade.\n"
        "3) Nao altere outras colunas/propriedades.\n"
        "4) No final, informe quantas opcoes foram recoloridas e quantas removidas.\n\n"
        f"Base: {payload.get('database_url', '')}\n"
        f"Propriedade: {payload.get('property', '')}\n"
        f"Bloco: {payload.get('block', '')} ({payload.get('range', '')})\n\n"
        "JSON do bloco:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _build_phase2_property_payload(
    *,
    property_name: str,
    rows: List[Dict[str, str]],
    option_url_lookup_by_property: Dict[str, Dict[str, Dict[str, str]]],
    force_default: bool,
    include_removals: bool,
) -> Tuple[Dict[str, Any], int, int, List[str]]:
    remove_unused_raw: List[str] = []
    if include_removals:
        remove_unused_raw = [
            row.get("etiqueta", "")
            for row in rows
            if _to_bool_flag(row.get("remover_se_apply"))
        ]

    remove_unused = _dedupe_labels(remove_unused_raw)
    remove_keys = {_normalize_key(name) for name in remove_unused}

    prop_lookup = option_url_lookup_by_property.get(_normalize_key(property_name), {})
    url_by_exact = prop_lookup.get("exact", {})
    url_by_key = prop_lookup.get("key", {})

    options: List[Dict[str, str]] = []
    missing_url_labels: List[str] = []
    seen_option_keys: set[str] = set()
    for row in rows:
        name = _normalize_ws(row.get("etiqueta"))
        key = _normalize_key(name)
        if not key or key in seen_option_keys:
            continue
        seen_option_keys.add(key)
        if key in remove_keys:
            continue

        if force_default:
            color = "default"
        else:
            color = _normalize_ws(row.get("cor_alvo")) or _normalize_ws(row.get("cor_atual")) or "default"
            if color not in SUPPORTED_COLORS:
                color = "default"

        option_url = _normalize_ws(url_by_exact.get(name) or url_by_key.get(key))
        if not option_url:
            missing_url_labels.append(name)
            continue

        options.append(
            {
                "name": name,
                "url": option_url,
                "color": color,
            }
        )

    payload = {
        "property": property_name,
        "options": options,
    }
    set_default_count = sum(1 for option in options if _normalize_ws(option.get("color")) == "default")
    remove_unused_count = len(remove_unused)
    return payload, set_default_count, remove_unused_count, missing_url_labels


def _build_phase2_property_prompt_text(
    *,
    database_url: str,
    payload: Dict[str, Any],
    set_default_count: int,
    remove_unused_count: int,
) -> str:
    return (
        "Aplique este JSON completo na configuracao da propriedade no Notion.\n\n"
        "Regras obrigatorias:\n"
        "1) Use EXATAMENTE o array `options` enviado abaixo.\n"
        "2) Nao altere outras propriedades/colunas.\n"
        "3) Confirme ao final as contagens aplicadas.\n\n"
        f"Base: {database_url}\n"
        f"Propriedade: {payload.get('property', '')}\n\n"
        f"Contagem esperada: set_default={set_default_count} | removidos={remove_unused_count}\n\n"
        "Formato solicitado pelo Notion Chat:\n"
        "{\n"
        '  "property": "...",\n'
        '  "options": [\n'
        '    {"name": "...", "url": "...", "color": "..."}\n'
        "  ]\n"
        "}\n\n"
        "JSON da propriedade:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _generate_phase2_notion_chat_compact_queue(
    *,
    args: argparse.Namespace,
    selected_phase2_keys: set[str],
    manual_rows: List[Dict[str, str]],
    option_url_lookup_by_property: Dict[str, Dict[str, Dict[str, str]]],
    fallback_parent: Path,
) -> Tuple[Path, List[Dict[str, str]]]:
    out_dir = _resolve_phase2_chat_output_dir(args.phase2_chat_output_dir, fallback_parent)
    queue_path = _resolve_phase2_queue_file(args.phase2_queue_file, out_dir)
    canonical = _canonical_phase2_name_by_key()

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in manual_rows:
        prop_name_raw = _normalize_ws(row.get("coluna"))
        prop_key = _normalize_key(prop_name_raw)
        if prop_key not in selected_phase2_keys:
            continue
        prop_name = canonical.get(prop_key, prop_name_raw)
        if not prop_name:
            continue
        grouped[prop_name].append(row)

    if not grouped:
        raise RuntimeError("Sem dados de fase 2 para gerar pacotes JSON do Notion Chat.")

    fieldnames = [
        "coluna",
        "bloco",
        "faixa_alfabetica",
        "arquivo_csv",
        "arquivo_payload",
        "arquivo_prompt",
        "status",
        "data_execucao",
        "observacao",
        "total_set_default",
        "total_remove_unused",
    ]
    queue_rows: List[Dict[str, str]] = []
    for prop_name in sorted(grouped, key=_normalize_key):
        payload, set_default_count, remove_unused_count, missing_url_labels = _build_phase2_property_payload(
            property_name=prop_name,
            rows=grouped[prop_name],
            option_url_lookup_by_property=option_url_lookup_by_property,
            force_default=bool(args.phase2_force_default),
            include_removals=bool(args.phase2_include_removals),
        )
        if missing_url_labels:
            preview = ", ".join(missing_url_labels[:8])
            if len(missing_url_labels) > 8:
                preview += f" ... +{len(missing_url_labels) - 8}"
            raise RuntimeError(
                f"[{prop_name}] Nao foi possivel preencher URL interna de {len(missing_url_labels)} etiquetas. "
                f"Exemplos: {preview}"
            )

        # Se houver remocoes previstas, o payload pode ter `options` vazio (limpeza total da propriedade).
        if not payload.get("options") and remove_unused_count <= 0:
            continue

        prop_token = _sanitize_filename_token(prop_name)
        payload_path = out_dir / f"phase2_{prop_token}.json"
        prompt_path = out_dir / f"phase2_{prop_token}.prompt.txt"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prompt_path.write_text(
            _build_phase2_property_prompt_text(
                database_url=args.database_url,
                payload=payload,
                set_default_count=set_default_count,
                remove_unused_count=remove_unused_count,
            ),
            encoding="utf-8",
        )

        queue_rows.append(
            {
                "coluna": prop_name,
                "bloco": "ALL",
                "faixa_alfabetica": "ALL",
                "arquivo_csv": "",
                "arquivo_payload": str(payload_path),
                "arquivo_prompt": str(prompt_path),
                "status": "GERADO",
                "data_execucao": "",
                "observacao": "Pacote unico por coluna (fase 2)",
                "total_set_default": str(set_default_count),
                "total_remove_unused": str(remove_unused_count),
            }
        )

    if not queue_rows:
        raise RuntimeError("Nenhum pacote gerado para fase 2. Verifique filtros e colunas selecionadas.")

    with queue_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queue_rows)

    LOGGER.warning(
        "Pacotes Notion Chat (fase 2) gerados: %s | itens=%d",
        queue_path,
        len(queue_rows),
    )
    for row in queue_rows:
        LOGGER.warning(
            "  - %s | set_default=%s | remove_unused=%s | json=%s",
            row.get("coluna", ""),
            row.get("total_set_default", "0"),
            row.get("total_remove_unused", "0"),
            row.get("arquivo_payload", ""),
        )
    return queue_path, queue_rows


def _generate_phase2_notion_chat_queue(
    *,
    args: argparse.Namespace,
    selected_phase2_keys: set[str],
) -> Tuple[Path, List[Dict[str, str]]]:
    input_dir_raw = _normalize_ws(args.phase2_blocks_input_dir)
    if not input_dir_raw:
        raise RuntimeError(
            "Modo phase2=notion-chat exige --phase2-blocks-input-dir apontando para pasta com CSVs phase2_*_bloco_*.csv."
        )

    input_dir = Path(input_dir_raw)
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Pasta de blocos da fase 2 nao encontrada: {input_dir}")

    out_dir = _resolve_phase2_chat_output_dir(args.phase2_chat_output_dir, input_dir)
    queue_path = _resolve_phase2_queue_file(args.phase2_queue_file, out_dir)
    canonical_phase2 = _canonical_phase2_name_by_key()
    files = _collect_phase2_block_csv_files(input_dir)
    if not files:
        raise RuntimeError(f"Nenhum CSV de bloco encontrado em {input_dir} (esperado: phase2_*_bloco_*.csv)")

    queue_rows: List[Dict[str, str]] = []
    fieldnames = [
        "coluna",
        "bloco",
        "faixa_alfabetica",
        "arquivo_csv",
        "arquivo_payload",
        "arquivo_prompt",
        "status",
        "data_execucao",
        "observacao",
        "total_set_default",
        "total_remove_unused",
    ]

    for csv_path in files:
        fallback_prop, fallback_block, fallback_range = _infer_phase2_block_metadata_from_filename(csv_path)
        try:
            rows = _load_phase2_block_rows(csv_path)
        except Exception as exc:
            prop_key_guess = _normalize_key(fallback_prop)
            if prop_key_guess and prop_key_guess in selected_phase2_keys:
                queue_rows.append(
                    {
                        "coluna": fallback_prop or "desconhecida",
                        "bloco": fallback_block,
                        "faixa_alfabetica": fallback_range,
                        "arquivo_csv": str(csv_path),
                        "arquivo_payload": "",
                        "arquivo_prompt": "",
                        "status": "ERRO",
                        "data_execucao": "",
                        "observacao": _normalize_ws(str(exc)),
                        "total_set_default": "0",
                        "total_remove_unused": "0",
                    }
                )
            LOGGER.error("Bloco invalido ignorado (%s): %s", csv_path.name, exc)
            continue

        if not rows:
            prop_key_guess = _normalize_key(fallback_prop)
            if prop_key_guess and prop_key_guess in selected_phase2_keys:
                queue_rows.append(
                    {
                        "coluna": fallback_prop or "desconhecida",
                        "bloco": fallback_block,
                        "faixa_alfabetica": fallback_range,
                        "arquivo_csv": str(csv_path),
                        "arquivo_payload": "",
                        "arquivo_prompt": "",
                        "status": "ERRO",
                        "data_execucao": "",
                        "observacao": "CSV de bloco vazio",
                        "total_set_default": "0",
                        "total_remove_unused": "0",
                    }
                )
            continue

        prop_name_raw = _normalize_ws(rows[0].get("coluna"))
        prop_key = _normalize_key(prop_name_raw)
        prop_name = canonical_phase2.get(prop_key, prop_name_raw) or fallback_prop
        if _normalize_key(prop_name) not in selected_phase2_keys:
            continue

        block_id = _normalize_ws(rows[0].get("bloco"))
        if not block_id:
            block_id = fallback_block
        range_label = _normalize_ws(rows[0].get("faixa_alfabetica"))
        if not range_label:
            range_label = fallback_range

        payload = _build_phase2_default_payload(
            database_url=args.database_url,
            property_name=prop_name,
            block_id=block_id,
            range_label=range_label,
            rows=rows,
            force_default=bool(args.phase2_force_default),
            include_removals=bool(args.phase2_include_removals),
        )

        if not payload["set_default"] and not payload["remove_unused"]:
            continue

        payload_path = out_dir / f"{csv_path.stem}.payload.json"
        prompt_path = out_dir / f"{csv_path.stem}.prompt.txt"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        prompt_path.write_text(_build_phase2_prompt_text(payload), encoding="utf-8")

        queue_rows.append(
            {
                "coluna": prop_name,
                "bloco": block_id,
                "faixa_alfabetica": range_label,
                "arquivo_csv": str(csv_path),
                "arquivo_payload": str(payload_path),
                "arquivo_prompt": str(prompt_path),
                "status": "PENDENTE",
                "data_execucao": "",
                "observacao": "",
                "total_set_default": str(len(payload["set_default"])),
                "total_remove_unused": str(len(payload["remove_unused"])),
            }
        )

    if not queue_rows:
        raise RuntimeError(
            "Nenhum bloco elegivel para fila do Notion Chat. Verifique colunas selecionadas e conteudo dos CSVs."
        )

    queue_rows.sort(
        key=lambda row: (
            _normalize_key(row.get("coluna")),
            int(_normalize_ws(row.get("bloco")) or "0"),
            _normalize_key(row.get("faixa_alfabetica")),
        )
    )
    with queue_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queue_rows)

    total = len(queue_rows)
    pending_rows = [row for row in queue_rows if _normalize_ws(row.get("status")).upper() == "PENDENTE"]
    error_rows = [row for row in queue_rows if _normalize_ws(row.get("status")).upper() == "ERRO"]
    by_col_pending = Counter(_normalize_ws(row.get("coluna")) for row in pending_rows)
    by_col_error = Counter(_normalize_ws(row.get("coluna")) for row in error_rows)
    LOGGER.warning(
        "Fila phase2/notion-chat gerada: %s | itens=%d | pendentes=%d | erro=%d",
        queue_path,
        total,
        len(pending_rows),
        len(error_rows),
    )
    for col in sorted(set(by_col_pending) | set(by_col_error), key=_normalize_key):
        LOGGER.warning(
            "  - %s: pendentes=%d | erro=%d",
            col,
            by_col_pending.get(col, 0),
            by_col_error.get(col, 0),
        )
    LOGGER.warning("Pasta de artefatos JSON/PROMPT: %s", out_dir)
    return queue_path, queue_rows


def _log_phase2_queue_alert(queue_rows: List[Dict[str, str]], selected_phase2_keys: set[str]) -> None:
    status_values = {
        _normalize_ws(row.get("status")).upper()
        for row in queue_rows
        if _normalize_ws(row.get("status"))
    }
    if status_values and status_values <= {"GERADO"}:
        by_prop_total: Dict[str, int] = {}
        for row in queue_rows:
            prop = _normalize_ws(row.get("coluna"))
            key = _normalize_key(prop)
            if key not in selected_phase2_keys:
                continue
            by_prop_total[prop] = by_prop_total.get(prop, 0) + 1
        LOGGER.warning("ALERTA FASE 2 (NOTION CHAT) | Pacotes gerados:")
        for prop_name in sorted(by_prop_total, key=_normalize_key):
            LOGGER.warning("  - %s: %d arquivo(s).", prop_name, by_prop_total[prop_name])
        if by_prop_total:
            LOGGER.warning("Use os arquivos JSON/PROMPT para executar no Notion Chat (um por coluna).")
        return

    by_prop_pending: Dict[str, int] = {}
    by_prop_total: Dict[str, int] = {}
    for row in queue_rows:
        prop = _normalize_ws(row.get("coluna"))
        key = _normalize_key(prop)
        if key not in selected_phase2_keys:
            continue
        by_prop_total[prop] = by_prop_total.get(prop, 0) + 1
        if _normalize_ws(row.get("status")).upper() == "PENDENTE":
            by_prop_pending[prop] = by_prop_pending.get(prop, 0) + 1

    LOGGER.warning("ALERTA FASE 2 (NOTION CHAT) | Status da fila:")
    for prop_name in sorted(by_prop_total, key=_normalize_key):
        pending = by_prop_pending.get(prop_name, 0)
        if pending <= 0:
            LOGGER.warning("  - %s: SEM PENDENCIAS.", prop_name)
        else:
            LOGGER.warning("  - %s: %d pendencias.", prop_name, pending)

    selected_keys_lower = {key for key in selected_phase2_keys}
    expected_full = {_normalize_key("nomeMunicipio"), _normalize_key("partes"), _normalize_key("advogados")}
    all_clean = all(
        by_prop_pending.get(_canonical_phase2_name_by_key().get(key, key), 0) <= 0
        for key in selected_keys_lower
    )
    if selected_keys_lower >= expected_full and all_clean:
        LOGGER.warning("ALERTA FINAL: SEM PENDENCIAS em advogados, partes e nomeMunicipio.")
    elif all_clean:
        LOGGER.warning("ALERTA FINAL: SEM PENDENCIAS nas colunas de fase 2 selecionadas.")


def _label_bucket_letter(value: Any) -> str:
    text = _normalize_key(value)
    for ch in text:
        if "a" <= ch <= "z":
            return ch.upper()
    return "#"


def _split_rows_in_blocks(
    rows: List[Dict[str, str]],
    *,
    max_block_size: int,
) -> List[Tuple[str, List[Dict[str, str]]]]:
    if not rows:
        return []
    block_size = max(1, int(max_block_size))
    ordered = sorted(
        rows,
        key=lambda row: (
            _label_bucket_letter(row.get("etiqueta")),
            _normalize_key(row.get("etiqueta")),
        ),
    )

    blocks: List[Tuple[str, List[Dict[str, str]]]] = []
    current: List[Dict[str, str]] = []
    letters: List[str] = []
    for row in ordered:
        letter = _label_bucket_letter(row.get("etiqueta"))
        if current and len(current) >= block_size:
            start = letters[0]
            end = letters[-1]
            blocks.append((f"{start}-{end}" if start != end else start, current))
            current = []
            letters = []
        current.append(row)
        letters.append(letter)

    if current:
        start = letters[0]
        end = letters[-1]
        blocks.append((f"{start}-{end}" if start != end else start, current))
    return blocks


def _sanitize_filename_token(value: str) -> str:
    token = _normalize_key(value).replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]+", "", token)
    return token or "coluna"


def _write_phase2_block_files(
    *,
    manual_rows: List[Dict[str, str]],
    out_dir: Path,
    block_size: int,
    include_all_phase2_labels: bool = False,
) -> Tuple[Path, List[Dict[str, Any]]]:
    phase2_keys = _phase_property_keys("2")
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in manual_rows:
        prop_name = _normalize_ws(row.get("coluna"))
        prop_key = _normalize_key(prop_name)
        if prop_key not in phase2_keys:
            continue
        needs_color = include_all_phase2_labels or (
            _normalize_ws(row.get("cor_atual")) != _normalize_ws(row.get("cor_alvo"))
        )
        needs_remove = _to_bool_flag(row.get("remover_se_apply"))
        if not (needs_color or needs_remove):
            continue
        grouped[prop_name].append(row)

    summary_rows: List[Dict[str, Any]] = []
    fieldnames = [
        "coluna",
        "etiqueta",
        "cor_atual",
        "cor_alvo",
        "em_uso",
        "remover_se_apply",
        "bloco",
        "faixa_alfabetica",
    ]

    for prop_name, rows in sorted(grouped.items(), key=lambda item: _normalize_key(item[0])):
        blocks = _split_rows_in_blocks(rows, max_block_size=block_size)
        prop_token = _sanitize_filename_token(prop_name)
        for idx, (range_label, block_rows) in enumerate(blocks, start=1):
            range_token = range_label.replace("#", "NUM").replace("-", "_")
            file_name = f"phase2_{prop_token}_bloco_{idx:02d}_{range_token}.csv"
            out_path = out_dir / file_name
            with out_path.open("w", newline="", encoding="utf-8-sig") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                for row in block_rows:
                    payload = dict(row)
                    payload["bloco"] = str(idx)
                    payload["faixa_alfabetica"] = range_label
                    writer.writerow(payload)

            summary_rows.append(
                {
                    "coluna": prop_name,
                    "bloco": idx,
                    "faixa_alfabetica": range_label,
                    "linhas": len(block_rows),
                    "mudancas_cor": sum(
                        1 for row in block_rows if _normalize_ws(row.get("cor_atual")) != _normalize_ws(row.get("cor_alvo"))
                    ),
                    "remocoes": sum(1 for row in block_rows if _to_bool_flag(row.get("remover_se_apply"))),
                    "arquivo_csv": str(out_path),
                }
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"phase2_blocos_resumo_{stamp}.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["coluna", "bloco", "faixa_alfabetica", "linhas", "mudancas_cor", "remocoes", "arquivo_csv"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path, summary_rows


def _to_bool_flag(raw: Any) -> bool:
    text = _normalize_ws(raw).lower()
    return text in {"1", "true", "t", "yes", "y", "sim", "s"}


def _load_resume_plan_csv(path: Path) -> ResumePlanData:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Arquivo de retomada nao encontrado: {path}")

    target_colors_by_property: Dict[str, Dict[str, str]] = defaultdict(dict)
    remove_keys_by_property: Dict[str, set[str]] = defaultdict(set)
    rows_by_property: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    has_block_columns = False

    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"coluna", "etiqueta", "cor_atual", "cor_alvo", "remover_se_apply"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"CSV de retomada invalido (faltam colunas): {', '.join(sorted(missing))}"
            )

        fieldnames = {name.strip() for name in (reader.fieldnames or []) if name}
        has_block_columns = "bloco" in fieldnames or "faixa_alfabetica" in fieldnames

        rows = 0
        for row in reader:
            rows += 1
            prop_name = _normalize_ws(row.get("coluna"))
            label_name = _normalize_ws(row.get("etiqueta"))
            prop_key = _normalize_key(prop_name)
            label_key = _normalize_key(label_name)
            if not prop_key or not label_key:
                continue

            current = _normalize_ws(row.get("cor_atual")) or "default"
            target = _normalize_ws(row.get("cor_alvo")) or current
            if target not in SUPPORTED_COLORS:
                target = current
            remove_flag = _to_bool_flag(row.get("remover_se_apply"))

            rows_by_property[prop_key].append(
                {
                    "coluna": prop_name,
                    "etiqueta": label_name,
                    "cor_atual": current,
                    "cor_alvo": target,
                    "em_uso": _normalize_ws(row.get("em_uso")),
                    "remover_se_apply": "1" if remove_flag else "0",
                    "bloco": _normalize_ws(row.get("bloco")),
                    "faixa_alfabetica": _normalize_ws(row.get("faixa_alfabetica")),
                }
            )

            if remove_flag:
                remove_keys_by_property[prop_key].add(label_key)
                continue

            if target != current:
                target_colors_by_property[prop_key][label_key] = target

    LOGGER.info(
        "CSV de retomada carregado | arquivo=%s | linhas=%d | colunas_com_cor=%d | colunas_com_remocao=%d | colunas=%d | possui_blocos=%s",
        path,
        rows,
        len(target_colors_by_property),
        len(remove_keys_by_property),
        len(rows_by_property),
        "sim" if has_block_columns else "nao",
    )
    return ResumePlanData(
        target_colors_by_property=dict(target_colors_by_property),
        remove_keys_by_property=dict(remove_keys_by_property),
        rows_by_property=dict(rows_by_property),
        has_block_columns=has_block_columns,
    )


def _normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _format_duration_s(total_s: float) -> str:
    secs = max(0, int(round(float(total_s))))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _normalize_key(value: Any) -> str:
    text = _normalize_ws(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _normalize_ws(text)


def _normalize_person_key(value: Any) -> str:
    text = _normalize_key(value)
    if not text:
        return ""
    text = re.sub(r"\bdesignado a\b", " ", text)
    text = re.sub(r"\brelator a\b", " ", text)
    text = re.sub(r"\brelator\b", " ", text)
    text = re.sub(r"\bdesignado\b", " ", text)
    text = re.sub(r"\bministra\b", " ", text)
    text = re.sub(r"\bministro\b", " ", text)
    text = re.sub(r"\bmin\b", " ", text)
    return _normalize_ws(text)


def _top_counter_key(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def _first_alpha_char(value: Any) -> str:
    text = _normalize_key(value)
    for ch in text:
        if "a" <= ch <= "z":
            return ch
    return ""


def alphabetical_color_for_name(value: Any) -> Optional[str]:
    first = _first_alpha_char(value)
    if not first:
        return None
    idx = ord(first) - ord("a")
    if idx < 0 or idx > 25:
        return None
    bucket = (idx * len(ALPHABETICAL_COLORS)) // 26
    bucket = max(0, min(len(ALPHABETICAL_COLORS) - 1, bucket))
    color = ALPHABETICAL_COLORS[bucket]
    return color if color in SUPPORTED_COLORS else None


def _read_secret_from_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not raw:
        return ""
    first = raw.splitlines()[0].strip()
    if "=" in first:
        first = first.split("=", 1)[1].strip()
    return first.strip("\"' ")


def resolve_notion_key() -> str:
    for env_name in ("NOTION_API_KEY", "NOTION_TOKEN", "NOTION_SECRET"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    for candidate in (Path.cwd() / "Chave_Notion.txt", Path(__file__).resolve().parent / "Chave_Notion.txt"):
        value = _read_secret_from_file(candidate)
        if value:
            return value
    return ""


def _normalize_notion_id(raw: str) -> str:
    text = re.sub(r"[^0-9A-Fa-f]", "", str(raw or ""))
    if len(text) != 32:
        raise ValueError(f"ID Notion invalido: {raw}")
    return str(uuid.UUID(text))


def extract_notion_id_from_url(url_or_id: str) -> str:
    parsed = urlparse(str(url_or_id or "").strip())
    if not parsed.scheme and not parsed.netloc:
        return _normalize_notion_id(url_or_id)

    path_candidates = re.findall(r"[0-9A-Fa-f]{32}", parsed.path or "")
    if path_candidates:
        return _normalize_notion_id(path_candidates[-1])

    for values in parse_qs(parsed.query or "").values():
        for value in values:
            found = re.findall(r"[0-9A-Fa-f]{32}", value or "")
            if found:
                return _normalize_notion_id(found[-1])

    found_any = re.findall(r"[0-9A-Fa-f]{32}", str(url_or_id or ""))
    if found_any:
        return _normalize_notion_id(found_any[-1])
    raise ValueError(f"Nao foi possivel extrair ID Notion da URL: {url_or_id}")


def _encode_collection_option_token(raw_id: Any) -> str:
    value = unquote(_normalize_ws(raw_id))
    if not value:
        return ""
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _build_collection_option_url(
    *,
    data_source_id: str,
    property_id: str,
    option_id: str,
) -> str:
    prop_token = _encode_collection_option_token(property_id)
    option_token = _encode_collection_option_token(option_id)
    if not prop_token or not option_token:
        return ""
    return f"collectionPropertyOption://{data_source_id}/{prop_token}/{option_token}"


def _extract_http_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return _normalize_ws(payload.get("message") or payload.get("error") or "")
    except Exception:
        pass
    return _normalize_ws(response.text)


def _retry_delay_s(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            delay = float(retry_after)
            if delay > 0:
                return min(20.0, delay)
        except Exception:
            pass
    return min(20.0, 0.8 * (2 ** max(0, attempt - 1)))


def notion_request(
    session: requests.Session,
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
) -> Dict[str, Any]:
    url = path if path.startswith("http") else f"{NOTION_BASE_URL}{path}"
    attempts = max(1, int(retries) + 1)

    for attempt in range(1, attempts + 1):
        try:
            response = session.request(
                method=method.upper(),
                url=url,
                json=json_body,
                params=params,
                timeout=max(5, int(timeout_s)),
            )
        except requests.RequestException as exc:
            if attempt < attempts:
                time.sleep(_retry_delay_s(attempt))
                continue
            raise RuntimeError(f"Falha de rede no Notion: {exc}") from exc

        if 200 <= response.status_code < 300:
            if not response.text.strip():
                return {}
            payload = response.json()
            return payload if isinstance(payload, dict) else {}

        retryable = response.status_code == 429 or response.status_code >= 500
        message = _extract_http_message(response)
        if retryable and attempt < attempts:
            time.sleep(_retry_delay_s(attempt, response.headers.get("Retry-After")))
            continue
        raise RuntimeError(
            f"Erro Notion {response.status_code} em {method.upper()} {path}: {message or '<sem mensagem>'}"
        )

    raise RuntimeError(f"Falha no Notion em {method.upper()} {path}")


def retrieve_data_source_id(
    session: requests.Session,
    database_id: str,
    *,
    timeout_s: int,
    retries: int,
) -> str:
    payload = notion_request(
        session,
        "GET",
        f"/v1/databases/{database_id}",
        timeout_s=timeout_s,
        retries=retries,
    )
    raw_data_sources = payload.get("data_sources")
    candidates: List[str] = []

    if isinstance(raw_data_sources, list):
        for item in raw_data_sources:
            if isinstance(item, dict) and item.get("id"):
                try:
                    candidates.append(_normalize_notion_id(str(item["id"])))
                except Exception:
                    continue
    elif isinstance(raw_data_sources, dict):
        for key in ("results", "items", "data_sources"):
            value = raw_data_sources.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and item.get("id"):
                    try:
                        candidates.append(_normalize_notion_id(str(item["id"])))
                    except Exception:
                        continue

    if candidates:
        return candidates[0]

    fallback = notion_request(
        session,
        "GET",
        f"/v1/data_sources/{database_id}",
        timeout_s=timeout_s,
        retries=retries,
    )
    fallback_id = fallback.get("id")
    if isinstance(fallback_id, str) and fallback_id:
        return _normalize_notion_id(fallback_id)

    raise RuntimeError("Nao foi possivel identificar data_source_id da base informada.")


def retrieve_data_source(
    session: requests.Session,
    *,
    data_source_id: str,
    timeout_s: int,
    retries: int,
) -> Dict[str, Any]:
    return notion_request(
        session,
        "GET",
        f"/v1/data_sources/{data_source_id}",
        timeout_s=timeout_s,
        retries=retries,
    )


def query_pages(
    session: requests.Session,
    *,
    data_source_id: str,
    database_id: str,
    page_size: int,
    timeout_s: int,
    retries: int,
    max_pages: int,
) -> Iterator[Dict[str, Any]]:
    cursor: Optional[str] = None
    pages_fetched = 0
    use_database_query = False

    while True:
        body: Dict[str, Any] = {"page_size": max(1, min(100, int(page_size)))}
        if cursor:
            body["start_cursor"] = cursor

        path = (
            f"/v1/databases/{database_id}/query"
            if use_database_query
            else f"/v1/data_sources/{data_source_id}/query"
        )
        try:
            payload = notion_request(
                session,
                "POST",
                path,
                json_body=body,
                timeout_s=timeout_s,
                retries=retries,
            )
        except RuntimeError as exc:
            if not use_database_query and ("404" in str(exc) or "object_not_found" in str(exc)):
                LOGGER.info("Endpoint data_sources/query indisponivel; fallback para /databases/{id}/query.")
                use_database_query = True
                continue
            raise

        results = payload.get("results", [])
        if not isinstance(results, list):
            results = []

        pages_fetched += 1
        for item in results:
            if isinstance(item, dict) and str(item.get("object", "")) == "page":
                yield item

        if max_pages > 0 and pages_fetched >= max_pages:
            return
        if not bool(payload.get("has_more")):
            return
        cursor = payload.get("next_cursor")
        if not cursor:
            return


def find_property_name(properties: Dict[str, Any], wanted: str) -> str:
    if wanted in properties:
        return wanted
    wanted_key = _normalize_key(wanted)
    for name in properties.keys():
        if _normalize_key(name) == wanted_key:
            return str(name)
    return ""


def property_to_values(prop: Dict[str, Any]) -> List[str]:
    prop_type = str(prop.get("type", "") or "")
    if prop_type == "select":
        obj = prop.get("select")
        if isinstance(obj, dict):
            value = _normalize_ws(obj.get("name"))
            return [value] if value else []
        return []
    if prop_type == "multi_select":
        items = prop.get("multi_select")
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for item in items:
            if isinstance(item, dict):
                value = _normalize_ws(item.get("name"))
                if value:
                    out.append(value)
        return out
    if prop_type == "rich_text":
        values = prop.get("rich_text")
        if isinstance(values, list):
            text = "".join(str(item.get("plain_text", "")) for item in values if isinstance(item, dict)).strip()
            return [text] if text else []
    if prop_type == "title":
        values = prop.get("title")
        if isinstance(values, list):
            text = "".join(str(item.get("plain_text", "")) for item in values if isinstance(item, dict)).strip()
            return [text] if text else []
    return []


def _find_csv_column(headers: Iterable[str], wanted: str) -> str:
    wanted_key = _normalize_key(wanted)
    for name in headers:
        if _normalize_key(name) == wanted_key:
            return str(name)
    return ""


def _csv_cell_to_values(raw: Any, *, is_multi: bool) -> List[str]:
    text = _normalize_ws(raw)
    if not text:
        return []
    if not is_multi:
        return [text]
    values = [_normalize_ws(part) for part in text.split(",")]
    return [value for value in values if value]


def scan_usage_and_city_uf_from_csv(
    csv_path: Path,
    *,
    tracked_property_names: Iterable[str],
    city_property_name: str,
    uf_property_name: str,
    sigla_classe_property_name: str,
    descricao_classe_property_name: str,
    multi_value_property_names: Iterable[str],
) -> ScanData:
    if not csv_path.exists() or not csv_path.is_file():
        raise RuntimeError(f"CSV base nao encontrado: {csv_path}")

    tracked = [name for name in tracked_property_names if _normalize_ws(name)]
    tracked_set = set(tracked)
    used_keys: Dict[str, set[str]] = {name: set() for name in tracked}
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    descricao_to_sigla_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    multi_keys = {_normalize_key(name) for name in multi_value_property_names if _normalize_ws(name)}

    scanned = 0
    missing_headers_logged: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        headers = [name for name in (reader.fieldnames or []) if name]
        if not headers:
            raise RuntimeError(f"CSV base sem cabecalho: {csv_path}")

        header_by_prop: Dict[str, str] = {}
        for prop_name in tracked_set:
            header_by_prop[prop_name] = _find_csv_column(headers, prop_name)

        city_header = _find_csv_column(headers, city_property_name) if city_property_name else ""
        uf_header = _find_csv_column(headers, uf_property_name) if uf_property_name else ""
        sigla_header = _find_csv_column(headers, sigla_classe_property_name) if sigla_classe_property_name else ""
        descricao_header = _find_csv_column(headers, descricao_classe_property_name) if descricao_classe_property_name else ""

        for row in reader:
            scanned += 1
            for prop_name in tracked_set:
                header = header_by_prop.get(prop_name, "")
                if not header:
                    key = _normalize_key(prop_name)
                    if key not in missing_headers_logged:
                        missing_headers_logged.add(key)
                        LOGGER.warning("CSV base: coluna '%s' nao encontrada no arquivo.", prop_name)
                    continue
                is_multi = _normalize_key(prop_name) in multi_keys
                values = _csv_cell_to_values(row.get(header), is_multi=is_multi)
                if not values:
                    continue
                bucket = used_keys.setdefault(prop_name, set())
                for value in values:
                    key = _normalize_key(value)
                    if key:
                        bucket.add(key)

            if city_header and uf_header:
                city_values = _csv_cell_to_values(row.get(city_header), is_multi=False)
                uf_values = _csv_cell_to_values(row.get(uf_header), is_multi=False)
                if city_values and uf_values:
                    city = city_values[0]
                    uf = _normalize_ws(uf_values[0]).upper()
                    if uf in UF_TO_REGION:
                        counts[_normalize_key(city)][uf] += 1

            if sigla_header and descricao_header:
                sigla_values = _csv_cell_to_values(row.get(sigla_header), is_multi=False)
                descricao_values = _csv_cell_to_values(row.get(descricao_header), is_multi=False)
                if sigla_values and descricao_values:
                    sigla_key = _normalize_key(sigla_values[0])
                    descricao_key = _normalize_key(descricao_values[0])
                    if sigla_key and descricao_key:
                        descricao_to_sigla_counts[descricao_key][sigla_key] += 1

    mapping: Dict[str, str] = {}
    ambiguous = 0
    for city_key, uf_counter in counts.items():
        if not uf_counter:
            continue
        top_two = uf_counter.most_common(2)
        if len(top_two) == 1 or top_two[0][1] > top_two[1][1]:
            mapping[city_key] = top_two[0][0]
        else:
            ambiguous += 1

    LOGGER.info(
        "Mapa municipio->UF coletado (CSV) | linhas=%d | cidades_mapeadas=%d | ambiguas=%d | arquivo=%s",
        scanned,
        len(mapping),
        ambiguous,
        csv_path,
    )
    for prop_name in tracked:
        LOGGER.info(
            "Valores em uso detectados (CSV) | coluna=%s | etiquetas_usadas=%d",
            prop_name,
            len(used_keys.get(prop_name) or set()),
        )
    return ScanData(
        city_to_uf=mapping,
        used_keys_by_property=used_keys,
        descricao_to_sigla_counts=descricao_to_sigla_counts,
        pages_scanned=scanned,
    )


def scan_usage_and_city_uf(
    session: requests.Session,
    *,
    data_source_id: str,
    database_id: str,
    tracked_property_names: Iterable[str],
    city_property_name: str,
    uf_property_name: str,
    sigla_classe_property_name: str,
    descricao_classe_property_name: str,
    page_size: int,
    timeout_s: int,
    retries: int,
    max_pages: int,
) -> ScanData:
    tracked = [name for name in tracked_property_names if _normalize_ws(name)]
    tracked_set = set(tracked)
    used_keys: Dict[str, set[str]] = {name: set() for name in tracked}
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    descricao_to_sigla_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    scanned = 0

    for page in query_pages(
        session,
        data_source_id=data_source_id,
        database_id=database_id,
        page_size=page_size,
        timeout_s=timeout_s,
        retries=retries,
        max_pages=max_pages,
    ):
        scanned += 1
        props = page.get("properties")
        if not isinstance(props, dict):
            continue

        for prop_name in tracked_set:
            prop_obj = props.get(prop_name)
            if not isinstance(prop_obj, dict):
                continue
            values = property_to_values(prop_obj)
            if not values:
                continue
            bucket = used_keys.setdefault(prop_name, set())
            for value in values:
                key = _normalize_key(value)
                if key:
                    bucket.add(key)

        city_prop = props.get(city_property_name)
        uf_prop = props.get(uf_property_name)
        if not isinstance(city_prop, dict) or not isinstance(uf_prop, dict):
            pass
        else:
            city_values = property_to_values(city_prop)
            uf_values = property_to_values(uf_prop)
            if city_values and uf_values:
                city = city_values[0]
                uf = uf_values[0].upper()
                if uf in UF_TO_REGION:
                    counts[_normalize_key(city)][uf] += 1

        if sigla_classe_property_name and descricao_classe_property_name:
            sigla_prop = props.get(sigla_classe_property_name)
            descricao_prop = props.get(descricao_classe_property_name)
            if isinstance(sigla_prop, dict) and isinstance(descricao_prop, dict):
                sigla_values = property_to_values(sigla_prop)
                descricao_values = property_to_values(descricao_prop)
                if sigla_values and descricao_values:
                    sigla_key = _normalize_key(sigla_values[0])
                    descricao_key = _normalize_key(descricao_values[0])
                    if sigla_key and descricao_key:
                        descricao_to_sigla_counts[descricao_key][sigla_key] += 1

    mapping: Dict[str, str] = {}
    ambiguous = 0
    for city_key, uf_counter in counts.items():
        if not uf_counter:
            continue
        top_two = uf_counter.most_common(2)
        if len(top_two) == 1 or top_two[0][1] > top_two[1][1]:
            mapping[city_key] = top_two[0][0]
        else:
            ambiguous += 1

    LOGGER.info(
        "Mapa municipio->UF coletado | paginas=%d | cidades_mapeadas=%d | ambiguas=%d",
        scanned,
        len(mapping),
        ambiguous,
    )
    for prop_name in tracked:
        LOGGER.info(
            "Valores em uso detectados | coluna=%s | etiquetas_usadas=%d",
            prop_name,
            len(used_keys.get(prop_name) or set()),
        )
    return ScanData(
        city_to_uf=mapping,
        used_keys_by_property=used_keys,
        descricao_to_sigla_counts=descricao_to_sigla_counts,
        pages_scanned=scanned,
    )


def collect_tag_properties(properties: Dict[str, Any]) -> Dict[str, TagProperty]:
    out: Dict[str, TagProperty] = {}
    for prop_name, prop_obj in properties.items():
        if not isinstance(prop_obj, dict):
            continue
        prop_id = _normalize_ws(prop_obj.get("id"))
        if not prop_id:
            continue
        prop_type = str(prop_obj.get("type", "") or "")
        if prop_type not in ("select", "multi_select"):
            continue
        cfg = prop_obj.get(prop_type)
        if not isinstance(cfg, dict):
            continue
        options = cfg.get("options")
        if not isinstance(options, list):
            options = []
        cleaned_options = [item for item in options if isinstance(item, dict)]
        out[str(prop_name)] = TagProperty(
            name=str(prop_name),
            property_id=prop_id,
            prop_type=prop_type,
            options=cleaned_options,
        )
    return out


def build_option_url_lookup(
    *,
    data_source_id: str,
    tag_props: Dict[str, TagProperty],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    normalized_ds_id = _normalize_notion_id(data_source_id)

    for prop_name, prop in tag_props.items():
        prop_key = _normalize_key(prop_name)
        if not prop_key:
            continue

        by_exact: Dict[str, str] = {}
        by_key: Dict[str, str] = {}
        duplicate_key_urls: set[str] = set()

        for option in prop.options:
            name = _normalize_ws(option.get("name"))
            option_id = _normalize_ws(option.get("id"))
            if not name or not option_id:
                continue

            option_url = _build_collection_option_url(
                data_source_id=normalized_ds_id,
                property_id=prop.property_id,
                option_id=option_id,
            )
            if not option_url:
                continue

            by_exact[name] = option_url

            option_key = _normalize_key(name)
            if not option_key:
                continue
            existing = by_key.get(option_key)
            if existing and existing != option_url:
                duplicate_key_urls.add(option_key)
            else:
                by_key[option_key] = option_url

        for dup_key in duplicate_key_urls:
            by_key.pop(dup_key, None)

        out[prop_key] = {
            "exact": by_exact,
            "key": by_key,
        }

    return out


def _clean_person_fragment(value: str) -> str:
    text = _normalize_ws(value)
    text = re.sub(r"(?i)^designado\(a\)\s*", "", text)
    text = re.sub(r"(?i)^relator\(a\)\s*", "", text)
    text = _normalize_ws(text.strip(" :;-"))
    return text


def build_composicao_color_lookup(options: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    score: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in options:
        name = _normalize_ws(item.get("name"))
        color = _normalize_ws(item.get("color")) or "default"
        if color not in SUPPORTED_COLORS:
            color = "default"
        if not name:
            continue

        names = [_clean_person_fragment(chunk) for chunk in name.split(",")]
        names = [n for n in names if n]
        if not names:
            continue

        # Opcoes com um nome tendem a ser mapeamentos confiaveis.
        weight = 3 if len(names) == 1 else 1
        for person in names:
            key = _normalize_person_key(person)
            if key:
                score[key][color] += weight

    lookup: Dict[str, str] = {}
    for person_key, colors in score.items():
        if not colors:
            continue
        non_default = [(color, qty) for color, qty in colors.items() if color != "default"]
        if non_default:
            non_default.sort(key=lambda item: item[1], reverse=True)
            lookup[person_key] = non_default[0][0]
            continue
        lookup[person_key] = colors.most_common(1)[0][0]
    return lookup


def extract_relator_candidates(option_name: str) -> List[str]:
    raw = _normalize_ws(option_name)
    if not raw:
        return []

    candidates: List[str] = []
    relator_match = RELATOR_LABEL_RE.search(raw)
    if relator_match:
        candidates.append(_clean_person_fragment(relator_match.group(1)))

    for match in MIN_TOKEN_RE.findall(raw):
        candidates.append(_clean_person_fragment(match))

    for piece in raw.split(","):
        cleaned = _clean_person_fragment(piece)
        if cleaned and ("min" in _normalize_key(cleaned) or " " in cleaned):
            candidates.append(cleaned)

    dedup: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalize_person_key(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(candidate)
    return dedup


def infer_relator_color(option_name: str, composicao_lookup: Dict[str, str]) -> Optional[str]:
    candidates = extract_relator_candidates(option_name)
    for candidate in candidates:
        key = _normalize_person_key(candidate)
        color = composicao_lookup.get(key)
        if color in SUPPORTED_COLORS:
            return color
    return None


def build_plan_sigla_uf(prop: TagProperty) -> PropertyPlan:
    desired: Dict[str, str] = {}
    for item in prop.options:
        name = _normalize_ws(item.get("name")).upper()
        region = UF_TO_REGION.get(name)
        if not region:
            continue
        color = REGION_TO_COLOR.get(region)
        if color:
            desired[_normalize_key(item.get("name"))] = color
    return PropertyPlan(
        property_name=prop.name,
        property_type=prop.prop_type,
        desired_color_by_key=desired,
        reason="Padrao regional por UF",
    )


def build_plan_nome_municipio(prop: TagProperty, city_to_uf: Dict[str, str]) -> PropertyPlan:
    desired: Dict[str, str] = {}
    for item in prop.options:
        name = _normalize_ws(item.get("name"))
        city_key = _normalize_key(name)
        if not city_key:
            continue
        uf = city_to_uf.get(city_key)
        if not uf:
            continue
        region = UF_TO_REGION.get(uf)
        color = REGION_TO_COLOR.get(region) if region else None
        if color:
            desired[city_key] = color
    return PropertyPlan(
        property_name=prop.name,
        property_type=prop.prop_type,
        desired_color_by_key=desired,
        reason="Padrao regional por municipio inferido na propria base",
    )


def build_plan_relator(prop: TagProperty, composicao_lookup: Dict[str, str]) -> PropertyPlan:
    desired: Dict[str, str] = {}
    for item in prop.options:
        name = _normalize_ws(item.get("name"))
        if not name:
            continue
        color = infer_relator_color(name, composicao_lookup)
        if color:
            desired[_normalize_key(name)] = color
    return PropertyPlan(
        property_name=prop.name,
        property_type=prop.prop_type,
        desired_color_by_key=desired,
        reason="Replica cores da coluna composicao",
    )


def build_plan_alphabetical(prop: TagProperty, *, reason: str) -> PropertyPlan:
    desired: Dict[str, str] = {}
    for item in prop.options:
        name = _normalize_ws(item.get("name"))
        if not name:
            continue
        color = alphabetical_color_for_name(name)
        if color:
            desired[_normalize_key(name)] = color
    return PropertyPlan(
        property_name=prop.name,
        property_type=prop.prop_type,
        desired_color_by_key=desired,
        reason=reason,
    )


def infer_sigla_classe_color(sigla_value: Any) -> str:
    key = _normalize_key(sigla_value)
    if not key:
        return CLASS_FAMILY_TO_COLOR["outros"]

    for family, keys in SIGLA_CLASS_FAMILY_KEYS.items():
        if key in keys:
            return CLASS_FAMILY_TO_COLOR[family]

    # Fallback por padrao textual da sigla.
    if "pc" in key:
        return CLASS_FAMILY_TO_COLOR["contas"]
    if "resp" in key or key.startswith("r"):
        return CLASS_FAMILY_TO_COLOR["recursos"]
    if "tut" in key:
        return CLASS_FAMILY_TO_COLOR["tutelas"]
    if any(token in key for token in ("aije", "rp", "rcl", "prop", "ajdes")):
        return CLASS_FAMILY_TO_COLOR["contencioso_eleitoral"]
    if any(token in key for token in ("hc", "ms", "pet", "conf", "cum")):
        return CLASS_FAMILY_TO_COLOR["procedimental"]
    if any(token in key for token in ("cta", "lt", "dp", "czer")):
        return CLASS_FAMILY_TO_COLOR["administrativo"]
    return CLASS_FAMILY_TO_COLOR["outros"]


def infer_descricao_classe_color(
    descricao_value: Any,
    descricao_to_sigla_counts: Dict[str, Counter[str]],
) -> str:
    descricao_key = _normalize_key(descricao_value)
    if not descricao_key:
        return CLASS_FAMILY_TO_COLOR["outros"]

    sigla_counter = descricao_to_sigla_counts.get(descricao_key)
    if sigla_counter:
        mapped_sigla = _top_counter_key(sigla_counter)
        if mapped_sigla:
            return infer_sigla_classe_color(mapped_sigla)

    # Fallback por palavras-chave da descricao.
    if "prestacao de contas" in descricao_key:
        return CLASS_FAMILY_TO_COLOR["contas"]
    if "recurso" in descricao_key:
        return CLASS_FAMILY_TO_COLOR["recursos"]
    if "tutela" in descricao_key:
        return CLASS_FAMILY_TO_COLOR["tutelas"]
    if any(
        token in descricao_key
        for token in (
            "investigacao judicial eleitoral",
            "representacao",
            "reclamacao",
            "propaganda partidaria",
            "desfiliacao partidaria",
        )
    ):
        return CLASS_FAMILY_TO_COLOR["contencioso_eleitoral"]
    if any(
        token in descricao_key
        for token in (
            "habeas corpus",
            "mandado de seguranca",
            "peticao civel",
            "conflito de jurisdicao",
            "cumprimento de sentenca",
        )
    ):
        return CLASS_FAMILY_TO_COLOR["procedimental"]
    if any(
        token in descricao_key
        for token in (
            "consulta",
            "lista triplice",
            "criacao de zona eleitoral",
            "direitos politicos",
            "duplicidade pluralidade de inscricoes",
        )
    ):
        return CLASS_FAMILY_TO_COLOR["administrativo"]
    return CLASS_FAMILY_TO_COLOR["outros"]


def build_plan_sigla_classe(prop: TagProperty) -> PropertyPlan:
    desired: Dict[str, str] = {}
    for item in prop.options:
        name = _normalize_ws(item.get("name"))
        if not name:
            continue
        desired[_normalize_key(name)] = infer_sigla_classe_color(name)
    return PropertyPlan(
        property_name=prop.name,
        property_type=prop.prop_type,
        desired_color_by_key=desired,
        reason="Familias de siglaClasse (recursos, contas, tutela, contencioso etc.)",
    )


def build_plan_descricao_classe(
    prop: TagProperty,
    descricao_to_sigla_counts: Dict[str, Counter[str]],
) -> PropertyPlan:
    desired: Dict[str, str] = {}
    for item in prop.options:
        name = _normalize_ws(item.get("name"))
        if not name:
            continue
        desired[_normalize_key(name)] = infer_descricao_classe_color(name, descricao_to_sigla_counts)
    return PropertyPlan(
        property_name=prop.name,
        property_type=prop.prop_type,
        desired_color_by_key=desired,
        reason="Familias de descricaoClasse alinhadas a siglaClasse",
    )


def build_patch_options(
    options: List[Dict[str, Any]],
    desired_color_by_key: Dict[str, str],
    *,
    used_keys: Optional[set[str]] = None,
    prune_unused: bool = False,
    force_remove_keys: Optional[set[str]] = None,
    strategy: str,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]], List[str]]:
    payload_options: List[Dict[str, Any]] = []
    changes: List[Tuple[str, str, str]] = []
    removed: List[str] = []

    for item in options:
        name = _normalize_ws(item.get("name"))
        current_color = _normalize_ws(item.get("color")) or "default"
        if current_color not in SUPPORTED_COLORS:
            current_color = "default"
        key = _normalize_key(name)
        if force_remove_keys is not None and key in force_remove_keys:
            removed.append(name)
            continue
        if prune_unused and used_keys is not None and key not in used_keys:
            removed.append(name)
            continue
        target_color = desired_color_by_key.get(key)
        if not target_color or target_color not in SUPPORTED_COLORS:
            target_color = current_color

        if target_color != current_color:
            changes.append((name, current_color, target_color))

        # A API publica nao permite alterar cor por ID; a atualizacao efetiva exige
        # substituir a lista de options via nome+cor (sem id).
        if strategy in ("direct", "recreate"):
            payload_options.append({"name": name, "color": target_color})
            continue

        raise ValueError(f"Estrategia desconhecida: {strategy}")

    return payload_options, changes, removed


def _is_actionable_resume_row(row: Dict[str, str]) -> bool:
    current = _normalize_ws(row.get("cor_atual"))
    target = _normalize_ws(row.get("cor_alvo"))
    if current != target:
        return True
    return _to_bool_flag(row.get("remover_se_apply"))


def _split_resume_rows_by_blocks(
    rows: List[Dict[str, str]],
    *,
    block_size: int,
    prefer_existing_blocks: bool,
) -> List[Tuple[str, List[Dict[str, str]]]]:
    actionable = [row for row in rows if _is_actionable_resume_row(row)]
    if not actionable:
        return []

    if prefer_existing_blocks:
        by_block: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in actionable:
            block_id = _normalize_ws(row.get("bloco"))
            if not block_id:
                block_id = "1"
            by_block[block_id].append(row)
        if by_block:
            def _sort_block_id(raw: str) -> Tuple[int, str]:
                text = _normalize_ws(raw)
                if text.isdigit():
                    return (0, f"{int(text):06d}")
                return (1, text)

            ordered: List[Tuple[str, List[Dict[str, str]]]] = []
            for block_id in sorted(by_block.keys(), key=_sort_block_id):
                group_rows = by_block[block_id]
                faixa = _normalize_ws(group_rows[0].get("faixa_alfabetica")) or block_id
                ordered.append((faixa, group_rows))
            return ordered

    return _split_rows_in_blocks(actionable, max_block_size=block_size)


def _build_patch_options_from_resume_scope(
    current_options: List[Dict[str, Any]],
    resume_rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]], List[str], int]:
    current_by_key: Dict[str, Tuple[str, str]] = {}
    for item in current_options:
        name = _normalize_ws(item.get("name"))
        if not name:
            continue
        key = _normalize_key(name)
        if not key or key in current_by_key:
            continue
        color = _normalize_ws(item.get("color")) or "default"
        if color not in SUPPORTED_COLORS:
            color = "default"
        current_by_key[key] = (name, color)

    payload: List[Dict[str, Any]] = []
    changes: List[Tuple[str, str, str]] = []
    remove_candidates: List[str] = []
    missing_in_schema = 0
    seen: set[str] = set()

    for row in resume_rows:
        key = _normalize_key(row.get("etiqueta"))
        if not key or key in seen:
            continue
        seen.add(key)

        if key not in current_by_key:
            missing_in_schema += 1
            continue

        current_name, current_color = current_by_key[key]
        if current_color not in SUPPORTED_COLORS:
            current_color = "default"

        if _to_bool_flag(row.get("remover_se_apply")):
            remove_candidates.append(current_name)
            continue

        target = _normalize_ws(row.get("cor_alvo")) or current_color
        if target not in SUPPORTED_COLORS:
            target = current_color

        if target != current_color:
            changes.append((current_name, current_color, target))
        payload.append({"name": current_name, "color": target})

    return payload, changes, remove_candidates, missing_in_schema


def _build_options_payload_from_raw(
    options: List[Dict[str, Any]],
    *,
    desired_color_by_key: Dict[str, str],
    used_keys: Optional[set[str]],
    prune_unused: bool,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for item in options:
        name = _normalize_ws(item.get("name"))
        if not name:
            continue
        key = _normalize_key(name)
        if prune_unused and used_keys is not None and key not in used_keys:
            continue
        current_color = _normalize_ws(item.get("color")) or "default"
        if current_color not in SUPPORTED_COLORS:
            current_color = "default"
        target_color = desired_color_by_key.get(key)
        if not target_color or target_color not in SUPPORTED_COLORS:
            target_color = current_color
        payload.append({"name": name, "color": target_color})
    return payload


def _make_tmp_label(base_name: str, existing_names: set[str]) -> str:
    base = _normalize_ws(base_name)
    if not base:
        base = "TMP"
    suffix = " __TMP_COLOR__"
    candidate = f"{base}{suffix}"
    if len(candidate) > 95:
        candidate = f"{base[: 95 - len(suffix)]}{suffix}"
    idx = 1
    while candidate in existing_names:
        tail = f"{suffix}{idx}"
        if len(base) + len(tail) > 95:
            candidate = f"{base[: 95 - len(tail)]}{tail}"
        else:
            candidate = f"{base}{tail}"
        idx += 1
    existing_names.add(candidate)
    return candidate


def _property_values_for_page(page_obj: Dict[str, Any], property_name: str) -> List[str]:
    props = page_obj.get("properties")
    if not isinstance(props, dict):
        return []
    prop = props.get(property_name)
    if not isinstance(prop, dict):
        return []
    return property_to_values(prop)


def _patch_page_property_values(
    session: requests.Session,
    *,
    page_id: str,
    property_name: str,
    property_type: str,
    values: List[str],
    timeout_s: int,
    retries: int,
) -> None:
    if property_type == "select":
        value = values[0] if values else ""
        body = {
            "properties": {
                property_name: {
                    "select": {"name": value} if value else None,
                }
            }
        }
    elif property_type == "multi_select":
        body = {
            "properties": {
                property_name: {
                    "multi_select": [{"name": value} for value in values if _normalize_ws(value)],
                }
            }
        }
    else:
        raise RuntimeError(f"Tipo de propriedade nao suportado para migracao de cor: {property_type}")

    notion_request(
        session,
        "PATCH",
        f"/v1/pages/{page_id}",
        json_body=body,
        timeout_s=timeout_s,
        retries=retries,
    )


def _migrate_page_values(
    session: requests.Session,
    *,
    data_source_id: str,
    database_id: str,
    property_name: str,
    property_type: str,
    rename_map: Dict[str, str],
    page_size: int,
    timeout_s: int,
    retries: int,
    max_pages: int,
    min_interval_s: float,
    expected_total_rows: int = 0,
) -> int:
    if not rename_map:
        return 0

    normalized_map = {_normalize_key(src): dst for src, dst in rename_map.items() if _normalize_ws(src) and _normalize_ws(dst)}
    if not normalized_map:
        return 0

    updated = 0
    scanned = 0
    expected = max(0, int(expected_total_rows))
    start_ts = time.monotonic()
    if expected > 0:
        LOGGER.info(
            "[%s] Migracao de valores iniciada | renomeacoes=%d | total_estimado=%d",
            property_name,
            len(normalized_map),
            expected,
        )
    else:
        LOGGER.info(
            "[%s] Migracao de valores iniciada | renomeacoes=%d",
            property_name,
            len(normalized_map),
        )
    for page in query_pages(
        session,
        data_source_id=data_source_id,
        database_id=database_id,
        page_size=page_size,
        timeout_s=timeout_s,
        retries=retries,
        max_pages=max_pages,
    ):
        scanned += 1
        if scanned % 25 == 0:
            elapsed_s = max(0.001, time.monotonic() - start_ts)
            if expected > 0 and scanned > 0:
                rate = scanned / elapsed_s
                remaining = max(0, expected - scanned)
                eta_s = (remaining / rate) if rate > 0 else 0.0
                eta_time = (datetime.now() + timedelta(seconds=eta_s)).strftime("%H:%M:%S")
                pct = min(100.0, (scanned * 100.0) / max(1, expected))
                LOGGER.info(
                    "[%s] Migracao em andamento | %d/%d (%.1f%%) | atualizadas=%d | "
                    "decorrido=%s | ETA=%s (termino ~%s)",
                    property_name,
                    scanned,
                    expected,
                    pct,
                    updated,
                    _format_duration_s(elapsed_s),
                    _format_duration_s(eta_s),
                    eta_time,
                )
            else:
                LOGGER.info(
                    "[%s] Migracao em andamento | paginas_varridas=%d | paginas_atualizadas=%d | decorrido=%s",
                    property_name,
                    scanned,
                    updated,
                    _format_duration_s(elapsed_s),
                )
        page_id = _normalize_ws(page.get("id"))
        if not page_id:
            continue

        current_values = _property_values_for_page(page, property_name)
        if not current_values:
            continue

        changed = False
        new_values: List[str] = []
        for value in current_values:
            mapped = normalized_map.get(_normalize_key(value))
            if mapped:
                new_values.append(mapped)
                changed = True
            else:
                new_values.append(value)
        if not changed:
            continue

        if property_type == "multi_select":
            # Remove duplicatas mantendo ordem.
            dedup: List[str] = []
            seen: set[str] = set()
            for value in new_values:
                key = _normalize_key(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                dedup.append(value)
            new_values = dedup

        _patch_page_property_values(
            session,
            page_id=page_id,
            property_name=property_name,
            property_type=property_type,
            values=new_values,
            timeout_s=timeout_s,
            retries=retries,
        )
        updated += 1
        if min_interval_s > 0:
            time.sleep(float(min_interval_s))

    total_elapsed_s = time.monotonic() - start_ts
    LOGGER.info(
        "[%s] Migracao de valores finalizada | paginas_varridas=%d | paginas_atualizadas=%d | duracao=%s",
        property_name,
        scanned,
        updated,
        _format_duration_s(total_elapsed_s),
    )
    return updated


def apply_color_migration_for_property(
    session: requests.Session,
    *,
    database_id: str,
    data_source_id: str,
    property_name: str,
    property_type: str,
    current_options: List[Dict[str, Any]],
    desired_color_by_key: Dict[str, str],
    used_keys: Optional[set[str]],
    prune_unused: bool,
    page_size: int,
    timeout_s: int,
    retries: int,
    max_pages: int,
    min_interval_s: float,
    expected_total_rows: int = 0,
) -> Dict[str, int]:
    keep_options: List[Dict[str, Any]] = []
    recolor_old_to_tmp: Dict[str, str] = {}
    recolor_tmp_to_final: Dict[str, str] = {}
    unchanged_options: List[Dict[str, Any]] = []
    final_recolor_options: List[Dict[str, Any]] = []

    taken_names: set[str] = set(_normalize_ws(item.get("name")) for item in current_options if _normalize_ws(item.get("name")))
    for item in current_options:
        name = _normalize_ws(item.get("name"))
        if not name:
            continue
        key = _normalize_key(name)
        if prune_unused and used_keys is not None and key not in used_keys:
            continue

        current_color = _normalize_ws(item.get("color")) or "default"
        if current_color not in SUPPORTED_COLORS:
            current_color = "default"
        target_color = desired_color_by_key.get(key) if desired_color_by_key else ""
        if not target_color or target_color not in SUPPORTED_COLORS:
            target_color = current_color

        keep_options.append({"name": name, "color": current_color})
        if target_color == current_color:
            unchanged_options.append({"name": name, "color": current_color})
            continue

        tmp_name = _make_tmp_label(name, taken_names)
        recolor_old_to_tmp[name] = tmp_name
        recolor_tmp_to_final[tmp_name] = name
        final_recolor_options.append({"name": name, "color": target_color})

    if not keep_options:
        return {"page_updates_phase1": 0, "page_updates_phase2": 0}

    if not recolor_old_to_tmp:
        final_payload = _build_options_payload_from_raw(
            current_options,
            desired_color_by_key=desired_color_by_key,
            used_keys=used_keys,
            prune_unused=prune_unused,
        )
        update_property_options(
            session,
            database_id=database_id,
            data_source_id=data_source_id,
            property_name=property_name,
            property_type=property_type,
            options_payload=final_payload,
            timeout_s=timeout_s,
            retries=retries,
        )
        return {"page_updates_phase1": 0, "page_updates_phase2": 0}

    tmp_options = [{"name": tmp_name, "color": desired_color_by_key.get(_normalize_key(final_name), "default")} for tmp_name, final_name in recolor_tmp_to_final.items()]

    # Fase 1: adiciona temporarias mantendo antigas.
    stage1 = keep_options + tmp_options
    if len(stage1) > NOTION_OPTIONS_PATCH_LIMIT:
        raise RuntimeError(
            f"Limite da API: PATCH de options aceita no maximo {NOTION_OPTIONS_PATCH_LIMIT} por chamada; "
            f"coluna '{property_name}' requer {len(stage1)} na fase 1."
        )
    update_property_options(
        session,
        database_id=database_id,
        data_source_id=data_source_id,
        property_name=property_name,
        property_type=property_type,
        options_payload=stage1,
        timeout_s=timeout_s,
        retries=retries,
    )

    phase1_updates = _migrate_page_values(
        session,
        data_source_id=data_source_id,
        database_id=database_id,
        property_name=property_name,
        property_type=property_type,
        rename_map=recolor_old_to_tmp,
        page_size=page_size,
        timeout_s=timeout_s,
        retries=retries,
        max_pages=max_pages,
        min_interval_s=min_interval_s,
        expected_total_rows=expected_total_rows,
    )

    # Fase 2: remove antigas recoloridas (depois de migrar dados para temporarias).
    stage2 = unchanged_options + tmp_options
    if len(stage2) > NOTION_OPTIONS_PATCH_LIMIT:
        raise RuntimeError(
            f"Limite da API: PATCH de options aceita no maximo {NOTION_OPTIONS_PATCH_LIMIT} por chamada; "
            f"coluna '{property_name}' requer {len(stage2)} na fase 2."
        )
    update_property_options(
        session,
        database_id=database_id,
        data_source_id=data_source_id,
        property_name=property_name,
        property_type=property_type,
        options_payload=stage2,
        timeout_s=timeout_s,
        retries=retries,
    )

    # Fase 3: recria nomes finais com cor alvo.
    stage3 = unchanged_options + tmp_options + final_recolor_options
    if len(stage3) > NOTION_OPTIONS_PATCH_LIMIT:
        raise RuntimeError(
            f"Limite da API: PATCH de options aceita no maximo {NOTION_OPTIONS_PATCH_LIMIT} por chamada; "
            f"coluna '{property_name}' requer {len(stage3)} na fase 3."
        )
    update_property_options(
        session,
        database_id=database_id,
        data_source_id=data_source_id,
        property_name=property_name,
        property_type=property_type,
        options_payload=stage3,
        timeout_s=timeout_s,
        retries=retries,
    )

    phase2_updates = _migrate_page_values(
        session,
        data_source_id=data_source_id,
        database_id=database_id,
        property_name=property_name,
        property_type=property_type,
        rename_map=recolor_tmp_to_final,
        page_size=page_size,
        timeout_s=timeout_s,
        retries=retries,
        max_pages=max_pages,
        min_interval_s=min_interval_s,
        expected_total_rows=expected_total_rows,
    )

    # Fase 4: remove temporarias.
    stage4 = unchanged_options + final_recolor_options
    if len(stage4) > NOTION_OPTIONS_PATCH_LIMIT:
        raise RuntimeError(
            f"Limite da API: PATCH de options aceita no maximo {NOTION_OPTIONS_PATCH_LIMIT} por chamada; "
            f"coluna '{property_name}' requer {len(stage4)} na fase 4."
        )
    update_property_options(
        session,
        database_id=database_id,
        data_source_id=data_source_id,
        property_name=property_name,
        property_type=property_type,
        options_payload=stage4,
        timeout_s=timeout_s,
        retries=retries,
    )

    return {
        "page_updates_phase1": phase1_updates,
        "page_updates_phase2": phase2_updates,
    }


def update_property_options(
    session: requests.Session,
    *,
    database_id: str,
    data_source_id: str,
    property_name: str,
    property_type: str,
    options_payload: List[Dict[str, Any]],
    timeout_s: int,
    retries: int,
) -> Dict[str, Any]:
    body = {
        "properties": {
            property_name: {
                property_type: {
                    "options": options_payload,
                }
            }
        }
    }

    return notion_request(
        session,
        "PATCH",
        f"/v1/data_sources/{data_source_id}",
        json_body=body,
        timeout_s=timeout_s,
        retries=retries,
    )


def run(args: argparse.Namespace) -> int:
    requested_keys = _requested_property_keys(args)
    phase1_keys = _phase_property_keys("1")
    phase2_keys = _phase_property_keys("2")
    requested_phase1 = requested_keys & phase1_keys
    requested_phase2 = requested_keys & phase2_keys
    phase2_mode = _normalize_ws(args.phase2_mode).lower()
    source_csv_raw = _normalize_ws(args.source_csv)
    source_csv_path: Optional[Path] = None
    if source_csv_raw:
        source_csv_path = Path(source_csv_raw)
        if not source_csv_path.is_absolute():
            source_csv_path = Path.cwd() / source_csv_path
        if not source_csv_path.exists() or not source_csv_path.is_file():
            raise RuntimeError(f"Arquivo --source-csv nao encontrado: {source_csv_path}")
    queue_from_existing_blocks = bool(
        phase2_mode == "notion-chat"
        and requested_phase2
        and _normalize_ws(args.phase2_blocks_input_dir)
        and not source_csv_path
    )

    if queue_from_existing_blocks:
        queue_path, queue_rows = _generate_phase2_notion_chat_queue(
            args=args,
            selected_phase2_keys=requested_phase2,
        )
        _log_phase2_queue_alert(queue_rows, requested_phase2)
        if not requested_phase1:
            LOGGER.info(
                "Modo phase2=notion-chat concluido sem chamadas API para fase 2. Fila: %s",
                queue_path,
            )
            return 0
        LOGGER.info(
            "Modo phase2=notion-chat: fila da fase 2 gerada; prosseguindo com API apenas para fase 1."
        )

    notion_token = resolve_notion_key()
    if not notion_token:
        raise RuntimeError(
            "Chave do Notion ausente. Crie Chave_Notion.txt ou defina NOTION_API_KEY/NOTION_TOKEN."
        )

    database_id = extract_notion_id_from_url(args.database_url)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {notion_token}",
            "Notion-Version": args.notion_version,
            "Content-Type": "application/json",
        }
    )

    try:
        data_source_id = retrieve_data_source_id(
            session,
            database_id,
            timeout_s=args.timeout,
            retries=args.retries,
        )
        data_source = retrieve_data_source(
            session,
            data_source_id=data_source_id,
            timeout_s=args.timeout,
            retries=args.retries,
        )
        properties = data_source.get("properties")
        if not isinstance(properties, dict):
            raise RuntimeError("Data source sem schema de propriedades.")

        tag_props = collect_tag_properties(properties)
        if not tag_props:
            LOGGER.info("Nenhuma propriedade select/multi_select encontrada. Nada a fazer.")
            return 0
        option_url_lookup_by_property = build_option_url_lookup(
            data_source_id=data_source_id,
            tag_props=tag_props,
        )

        composicao_name = find_property_name(properties, "composicao")
        relator_name = find_property_name(properties, "relator")
        uf_name = find_property_name(properties, "siglaUF")
        city_name = find_property_name(properties, "nomeMunicipio")
        partes_name = find_property_name(properties, "partes")
        advogados_name = find_property_name(properties, "advogados")
        sigla_classe_name = find_property_name(properties, "siglaClasse")
        descricao_classe_name = find_property_name(properties, "descricaoClasse")

        if not composicao_name:
            LOGGER.warning("Coluna 'composicao' nao encontrada; regra de replica para relator sera ignorada.")

        composicao_lookup: Dict[str, str] = {}
        if composicao_name and composicao_name in tag_props:
            composicao_lookup = build_composicao_color_lookup(tag_props[composicao_name].options)
            LOGGER.info("Mapa de cores da composicao carregado: %d nomes.", len(composicao_lookup))

        tracked_for_usage = [
            name
            for name in (
                uf_name,
                city_name,
                relator_name,
                partes_name,
                advogados_name,
                sigla_classe_name,
                descricao_classe_name,
            )
            if name
        ]
        if source_csv_path:
            multi_props = [
                name
                for name, prop in tag_props.items()
                if _normalize_ws(prop.prop_type) == "multi_select"
            ]
            scan_data = scan_usage_and_city_uf_from_csv(
                source_csv_path,
                tracked_property_names=tracked_for_usage,
                city_property_name=city_name,
                uf_property_name=uf_name,
                sigla_classe_property_name=sigla_classe_name,
                descricao_classe_property_name=descricao_classe_name,
                multi_value_property_names=multi_props,
            )
        else:
            scan_data = scan_usage_and_city_uf(
                session,
                data_source_id=data_source_id,
                database_id=database_id,
                tracked_property_names=tracked_for_usage,
                city_property_name=city_name,
                uf_property_name=uf_name,
                sigla_classe_property_name=sigla_classe_name,
                descricao_classe_property_name=descricao_classe_name,
                page_size=args.page_size,
                timeout_s=args.timeout,
                retries=args.retries,
                max_pages=args.max_pages,
            )
        city_to_uf = scan_data.city_to_uf
        used_keys_by_property = scan_data.used_keys_by_property
        descricao_to_sigla_counts = scan_data.descricao_to_sigla_counts

        plans: List[PropertyPlan] = []

        if uf_name and uf_name in tag_props:
            plans.append(build_plan_sigla_uf(tag_props[uf_name]))
        else:
            LOGGER.warning("Coluna siglaUF nao encontrada como select/multi_select.")

        if city_name and city_name in tag_props:
            plans.append(build_plan_nome_municipio(tag_props[city_name], city_to_uf))
        else:
            LOGGER.warning("Coluna nomeMunicipio nao encontrada como select/multi_select.")

        if relator_name and relator_name in tag_props and composicao_lookup:
            plans.append(build_plan_relator(tag_props[relator_name], composicao_lookup))
        elif relator_name and relator_name in tag_props:
            LOGGER.warning("Coluna relator encontrada, mas sem mapa de composicao; regra pulada.")
        else:
            LOGGER.warning("Coluna relator nao encontrada como select/multi_select.")

        if partes_name and partes_name in tag_props:
            plans.append(
                build_plan_alphabetical(
                    tag_props[partes_name],
                    reason="Padrao alfabetico de nomes (partes)",
                )
            )
        else:
            LOGGER.warning("Coluna partes nao encontrada como select/multi_select.")

        if advogados_name and advogados_name in tag_props:
            plans.append(
                build_plan_alphabetical(
                    tag_props[advogados_name],
                    reason="Padrao alfabetico de nomes (advogados)",
                )
            )
        else:
            LOGGER.warning("Coluna advogados nao encontrada como select/multi_select.")

        if sigla_classe_name and sigla_classe_name in tag_props:
            plans.append(build_plan_sigla_classe(tag_props[sigla_classe_name]))
        else:
            LOGGER.warning("Coluna siglaClasse nao encontrada como select/multi_select.")

        if descricao_classe_name and descricao_classe_name in tag_props:
            plans.append(
                build_plan_descricao_classe(
                    tag_props[descricao_classe_name],
                    descricao_to_sigla_counts=descricao_to_sigla_counts,
                )
            )
        else:
            LOGGER.warning("Coluna descricaoClasse nao encontrada como select/multi_select.")

        # Garantia explicita do requisito do usuario.
        plans = [plan for plan in plans if _normalize_key(plan.property_name) != _normalize_key("composicao")]
        phase_keys = _phase_property_keys(args.phase)
        plans = [plan for plan in plans if _normalize_key(plan.property_name) in phase_keys]
        if args.phase != "all":
            LOGGER.info("Filtro --phase ativo | fase=%s | planos_restantes=%d", args.phase, len(plans))

        only_property_keys = _parse_only_properties(args.only_properties)
        if only_property_keys:
            plans = [plan for plan in plans if _normalize_key(plan.property_name) in only_property_keys]
            LOGGER.info(
                "Filtro --only-properties ativo | colunas=%s | planos_restantes=%d",
                ",".join(sorted(only_property_keys)),
                len(plans),
            )

        if queue_from_existing_blocks:
            before = len(plans)
            plans = [plan for plan in plans if _normalize_key(plan.property_name) not in phase2_keys]
            removed = before - len(plans)
            if removed > 0:
                LOGGER.info(
                    "Modo phase2=notion-chat: %d plano(s) da fase 2 retirados da execucao API.",
                    removed,
                )

        resume_data: Optional[ResumePlanData] = None
        if _normalize_ws(args.resume_plan_input):
            resume_path = Path(_normalize_ws(args.resume_plan_input))
            if not resume_path.is_absolute():
                resume_path = Path.cwd() / resume_path
            resume_data = _load_resume_plan_csv(resume_path)
            resume_prop_keys = (
                set(resume_data.target_colors_by_property)
                | set(resume_data.remove_keys_by_property)
                | set(resume_data.rows_by_property)
            )
            plans = [plan for plan in plans if _normalize_key(plan.property_name) in resume_prop_keys]
            LOGGER.info(
                "Filtro de retomada ativo | arquivo=%s | colunas_com_pendencia=%d | planos_restantes=%d",
                resume_path,
                len(resume_prop_keys),
                len(plans),
            )
            if not plans:
                LOGGER.info("CSV de retomada sem pendencias para as colunas selecionadas.")
                return 0

        total_changes = 0
        total_removed = 0
        updated_props = 0
        failed_props = 0
        skipped_props = 0
        manual_plan_rows: List[Dict[str, str]] = []
        phase2_keys = _phase_property_keys("2")
        phase2_notion_chat_mode = phase2_mode == "notion-chat"
        phase2_selected_names: List[str] = []
        phase2_seen_names: set[str] = set()
        phase2_pending_by_property: Dict[str, int] = {}
        for plan in plans:
            if _normalize_key(plan.property_name) not in phase2_keys:
                continue
            prop_name = plan.property_name
            if prop_name in phase2_seen_names:
                continue
            phase2_seen_names.add(prop_name)
            phase2_selected_names.append(prop_name)
            phase2_pending_by_property[prop_name] = 0

        LOGGER.info("Modo: %s", "APPLY" if args.apply else "DRY-RUN")
        prune_unused_effective = bool(args.prune_unused)
        if prune_unused_effective and args.max_pages > 0 and not args.allow_partial_prune:
            LOGGER.warning(
                "Poda de etiquetas desativada para evitar falso-positivo com --max-pages=%d. "
                "Use --allow-partial-prune para forcar.",
                args.max_pages,
            )
            prune_unused_effective = False

        for plan in plans:
            prop = tag_props.get(plan.property_name)
            if not prop:
                continue

            prop_key = _normalize_key(plan.property_name)
            resume_target_colors = (
                resume_data.target_colors_by_property.get(prop_key, {})
                if resume_data
                else {}
            )
            resume_remove_keys = (
                resume_data.remove_keys_by_property.get(prop_key, set())
                if resume_data
                else set()
            )
            resume_rows_for_prop = (
                resume_data.rows_by_property.get(prop_key, [])
                if resume_data
                else []
            )
            desired_colors = plan.desired_color_by_key
            if resume_data:
                desired_colors = dict(resume_target_colors)
            if phase2_notion_chat_mode and prop_key in phase2_keys and bool(args.phase2_force_default):
                desired_colors = {
                    _normalize_key(_normalize_ws(item.get("name"))): "default"
                    for item in prop.options
                    if _normalize_ws(item.get("name"))
                }

            if phase2_notion_chat_mode and prop_key in phase2_keys:
                include_removals = bool(args.phase2_include_removals)
                used_keys_for_phase2 = used_keys_by_property.get(plan.property_name) if include_removals else None
                phase2_rows = _manual_plan_rows_for_property(
                    prop=prop,
                    plan=PropertyPlan(
                        property_name=plan.property_name,
                        property_type=plan.property_type,
                        desired_color_by_key=desired_colors,
                        reason="Fase 2 / Notion Chat",
                    ),
                    used_keys=used_keys_for_phase2,
                    prune_unused=include_removals,
                    force_remove_keys=resume_remove_keys if resume_data else None,
                )
                manual_plan_rows.extend(phase2_rows)
                set_default_count = len(phase2_rows) if bool(args.phase2_force_default) else sum(
                    1 for row in phase2_rows if _normalize_ws(row.get("cor_atual")) != _normalize_ws(row.get("cor_alvo"))
                )
                remove_count = sum(1 for row in phase2_rows if _to_bool_flag(row.get("remover_se_apply")))
                phase2_pending_by_property[plan.property_name] = set_default_count + remove_count
                total_changes += set_default_count
                total_removed += remove_count
                LOGGER.info(
                    "[%s] Fase 2 / Notion Chat preparada | set_default=%d | remocoes=%d | etiquetas=%d",
                    plan.property_name,
                    set_default_count,
                    remove_count,
                    len(phase2_rows),
                )
                continue

            use_resume_block_apply = bool(
                resume_data
                and args.apply
                and resume_rows_for_prop
                and prop_key in _phase_property_keys("2")
            )

            if use_resume_block_apply:
                resume_blocks = _split_resume_rows_by_blocks(
                    resume_rows_for_prop,
                    block_size=args.phase2_block_size,
                    prefer_existing_blocks=resume_data.has_block_columns,
                )
                pending_total_resume = sum(1 for row in resume_rows_for_prop if _is_actionable_resume_row(row))
                phase2_pending_by_property[plan.property_name] = pending_total_resume
                if not resume_blocks:
                    LOGGER.info("[%s] CSV de retomada sem acoes pendentes.", plan.property_name)
                    continue

                has_color_recolor = any(
                    _normalize_ws(row.get("cor_atual")) != _normalize_ws(row.get("cor_alvo"))
                    for row in resume_rows_for_prop
                )
                if has_color_recolor:
                    LOGGER.error(
                        "[%s] Bloqueio da API oficial do Notion: cor de option existente em "
                        "select/multi_select nao pode ser atualizada via API ('Cannot update color ...'). "
                        "Para colunas grandes, este fluxo em blocos nao consegue concluir recoloracao apenas por API.",
                        plan.property_name,
                    )
                    failed_props += 1
                    continue

                prop_changes_total = 0
                prop_remove_total = 0
                block_failures = 0
                block_updates = 0
                warned_remove_block_mode = False

                for block_idx, (range_label, block_rows) in enumerate(resume_blocks, start=1):
                    payload_block, changes_block, removed_block, missing_count = _build_patch_options_from_resume_scope(
                        prop.options,
                        block_rows,
                    )

                    prop_changes_total += len(changes_block)
                    prop_remove_total += len(removed_block)

                    if len(payload_block) > NOTION_OPTIONS_PATCH_LIMIT:
                        block_failures += 1
                        skipped_props += 1
                        LOGGER.warning(
                            "[%s][bloco %d - %s] Ignorado no apply: payload com %d opcoes (limite=%d).",
                            plan.property_name,
                            block_idx,
                            range_label,
                            len(payload_block),
                            NOTION_OPTIONS_PATCH_LIMIT,
                        )
                        continue

                    if removed_block and not warned_remove_block_mode:
                        warned_remove_block_mode = True
                        LOGGER.warning(
                            "[%s] CSV em blocos contem remocoes. Em modo retomada por blocos, a API oficial nao garante "
                            "apagar etiquetas fora do payload; este passo prioriza recoloracao.",
                            plan.property_name,
                        )

                    if not payload_block:
                        LOGGER.info(
                            "[%s][bloco %d - %s] Sem mudancas de cor aplicaveis (somente remocoes ou sem acao).",
                            plan.property_name,
                            block_idx,
                            range_label,
                        )
                        continue

                    LOGGER.info(
                        "[%s][bloco %d/%d - %s] payload=%d | mudancas_cor=%d | remocoes_planejadas=%d | labels_fora_schema=%d",
                        plan.property_name,
                        block_idx,
                        len(resume_blocks),
                        range_label,
                        len(payload_block),
                        len(changes_block),
                        len(removed_block),
                        missing_count,
                    )
                    for name, old_color, new_color in changes_block[:8]:
                        LOGGER.info("  - %s | %s -> %s", name, old_color, new_color)
                    if len(changes_block) > 8:
                        LOGGER.info("  ... +%d alteracoes", len(changes_block) - 8)

                    try:
                        update_property_options(
                            session,
                            database_id=database_id,
                            data_source_id=data_source_id,
                            property_name=plan.property_name,
                            property_type=plan.property_type,
                            options_payload=payload_block,
                            timeout_s=args.timeout,
                            retries=args.retries,
                        )
                        block_updates += 1
                        if args.min_interval > 0:
                            time.sleep(float(args.min_interval))
                    except Exception as block_exc:
                        block_failures += 1
                        LOGGER.error(
                            "[%s][bloco %d - %s] Falha no PATCH: %s",
                            plan.property_name,
                            block_idx,
                            range_label,
                            block_exc,
                        )

                total_changes += prop_changes_total
                total_removed += prop_remove_total
                if block_updates > 0 and block_failures == 0:
                    updated_props += 1
                elif block_updates > 0 and block_failures > 0:
                    failed_props += 1
                elif block_failures > 0:
                    failed_props += 1
                continue

            effective_prune_unused = prune_unused_effective and not resume_data
            used_keys = used_keys_by_property.get(plan.property_name) if effective_prune_unused else None

            payload_direct, changes, removed = build_patch_options(
                prop.options,
                desired_colors,
                used_keys=used_keys,
                prune_unused=effective_prune_unused,
                force_remove_keys=resume_remove_keys if resume_data else None,
                strategy="direct",
            )

            if (effective_prune_unused or (resume_data and resume_remove_keys)) and not payload_direct and prop.options:
                LOGGER.warning(
                    "[%s] Poda resultou em 0 etiquetas; update ignorado por seguranca.",
                    plan.property_name,
                )
                continue

            pending_actions = len(changes) + len(removed)
            if prop_key in phase2_keys:
                phase2_pending_by_property[plan.property_name] = pending_actions

            if not changes and not removed:
                LOGGER.info("[%s] Sem mudancas necessarias (%s).", plan.property_name, plan.reason)
                continue

            total_changes += len(changes)
            total_removed += len(removed)
            LOGGER.info(
                "[%s] mudancas_cor=%d | removidas=%d | opcoes=%d | motivo=%s",
                plan.property_name,
                len(changes),
                len(removed),
                len(prop.options),
                plan.reason,
            )
            for name, old_color, new_color in changes[:12]:
                LOGGER.info("  - %s | %s -> %s", name, old_color, new_color)
            if len(changes) > 12:
                LOGGER.info("  ... +%d alteracoes", len(changes) - 12)
            for name in removed[:12]:
                LOGGER.info("  - removida (sem uso): %s", name)
            if len(removed) > 12:
                LOGGER.info("  ... +%d remocoes", len(removed) - 12)

            if not args.apply:
                continue

            if len(payload_direct) > NOTION_OPTIONS_PATCH_LIMIT:
                skipped_props += 1
                LOGGER.warning(
                    "[%s] Ignorado no apply: payload com %d opcoes (limite da API=%d).",
                    plan.property_name,
                    len(payload_direct),
                    NOTION_OPTIONS_PATCH_LIMIT,
                )
                manual_plan_rows.extend(
                    _manual_plan_rows_for_property(
                        prop=prop,
                        plan=PropertyPlan(
                            property_name=plan.property_name,
                            property_type=plan.property_type,
                            desired_color_by_key=desired_colors,
                            reason=plan.reason,
                        ),
                        used_keys=used_keys,
                        prune_unused=effective_prune_unused,
                        force_remove_keys=resume_remove_keys if resume_data else None,
                    )
                )
                continue

            try:
                update_property_options(
                    session,
                    database_id=database_id,
                    data_source_id=data_source_id,
                    property_name=plan.property_name,
                    property_type=plan.property_type,
                    options_payload=payload_direct,
                    timeout_s=args.timeout,
                    retries=args.retries,
                )
                updated_props += 1
                if args.min_interval > 0:
                    time.sleep(float(args.min_interval))
                continue
            except Exception as direct_exc:
                direct_msg = str(direct_exc)
                if "Cannot update color of select" in direct_msg or "Cannot update color of multi_select" in direct_msg:
                    LOGGER.warning(
                        "[%s] API bloqueou recolor direto; executando migracao por etiqueta temporaria.",
                        plan.property_name,
                    )
                    try:
                        migration_stats = apply_color_migration_for_property(
                            session,
                            database_id=database_id,
                            data_source_id=data_source_id,
                            property_name=plan.property_name,
                            property_type=plan.property_type,
                            current_options=prop.options,
                            desired_color_by_key=desired_colors,
                            used_keys=used_keys,
                            prune_unused=effective_prune_unused,
                            page_size=args.page_size,
                            timeout_s=args.timeout,
                            retries=args.retries,
                            max_pages=args.max_pages,
                            min_interval_s=args.min_interval,
                            expected_total_rows=scan_data.pages_scanned,
                        )
                        LOGGER.info(
                            "[%s] Migracao de cor concluida | paginas_fase1=%d | paginas_fase2=%d",
                            plan.property_name,
                            migration_stats.get("page_updates_phase1", 0),
                            migration_stats.get("page_updates_phase2", 0),
                        )
                        updated_props += 1
                        continue
                    except Exception as migrate_exc:
                        failed_props += 1
                        LOGGER.error("[%s] Falha na migracao de cor: %s", plan.property_name, migrate_exc)
                        continue

                if not args.allow_recreate:
                    failed_props += 1
                    LOGGER.error("[%s] Falha no PATCH (direct): %s", plan.property_name, direct_msg)
                    continue

                LOGGER.warning("[%s] PATCH direct falhou; tentando estrategia recreate.", plan.property_name)
                payload_recreate, _, _ = build_patch_options(
                    prop.options,
                    desired_colors,
                    used_keys=used_keys,
                    prune_unused=effective_prune_unused,
                    force_remove_keys=resume_remove_keys if resume_data else None,
                    strategy="recreate",
                )
                if len(payload_recreate) > NOTION_OPTIONS_PATCH_LIMIT:
                    skipped_props += 1
                    LOGGER.warning(
                        "[%s] Fallback recreate ignorado: payload com %d opcoes (limite=%d).",
                        plan.property_name,
                        len(payload_recreate),
                        NOTION_OPTIONS_PATCH_LIMIT,
                    )
                    manual_plan_rows.extend(
                        _manual_plan_rows_for_property(
                            prop=prop,
                            plan=PropertyPlan(
                                property_name=plan.property_name,
                                property_type=plan.property_type,
                                desired_color_by_key=desired_colors,
                                reason=plan.reason,
                            ),
                            used_keys=used_keys,
                            prune_unused=effective_prune_unused,
                            force_remove_keys=resume_remove_keys if resume_data else None,
                        )
                    )
                    continue
                try:
                    update_property_options(
                        session,
                        database_id=database_id,
                        data_source_id=data_source_id,
                        property_name=plan.property_name,
                        property_type=plan.property_type,
                        options_payload=payload_recreate,
                        timeout_s=args.timeout,
                        retries=args.retries,
                    )
                    updated_props += 1
                    if args.min_interval > 0:
                        time.sleep(float(args.min_interval))
                except Exception as recreate_exc:
                    failed_props += 1
                    LOGGER.error(
                        "[%s] Falha no PATCH (recreate): %s",
                        plan.property_name,
                        recreate_exc,
                    )

        LOGGER.info("Resumo:")
        LOGGER.info("  Propriedades planejadas: %d", len(plans))
        LOGGER.info("  Alteracoes de etiqueta previstas: %d", total_changes)
        LOGGER.info("  Etiquetas sem uso para remover: %d", total_removed)
        LOGGER.info("  Propriedades atualizadas: %d", updated_props)
        LOGGER.info("  Propriedades ignoradas por limite API: %d", skipped_props)
        LOGGER.info("  Propriedades com erro: %d", failed_props)
        if phase2_selected_names:
            LOGGER.warning("ALERTA FASE 2 | Status de pendencias por coluna:")
            for prop_name in phase2_selected_names:
                pending = int(phase2_pending_by_property.get(prop_name, 0))
                if pending <= 0:
                    LOGGER.warning("  - %s: SEM PENDENCIAS.", prop_name)
                else:
                    LOGGER.warning("  - %s: %d pendencias (cor/remocao).", prop_name, pending)

            all_selected_phase2_clean = all(int(phase2_pending_by_property.get(prop_name, 0)) <= 0 for prop_name in phase2_selected_names)
            selected_phase2_keys = {_normalize_key(name) for name in phase2_selected_names}
            full_phase2_keys = {_normalize_key("advogados"), _normalize_key("partes"), _normalize_key("nomeMunicipio")}
            if all_selected_phase2_clean and selected_phase2_keys >= full_phase2_keys:
                LOGGER.warning(
                    "ALERTA FINAL: SEM PENDENCIAS em advogados, partes e nomeMunicipio."
                )
            elif all_selected_phase2_clean:
                LOGGER.warning(
                    "ALERTA FINAL: SEM PENDENCIAS nas colunas de fase 2 selecionadas."
                )
        if args.apply and updated_props == 0 and skipped_props > 0 and failed_props == 0 and not phase2_notion_chat_mode:
            LOGGER.warning(
                "APPLY executado, mas nenhuma coluna foi atualizada porque a API oficial do Notion "
                "bloqueou payloads acima de %d options. Use o CSV de plano manual e os blocos da fase 2.",
                NOTION_OPTIONS_PATCH_LIMIT,
            )
        if manual_plan_rows:
            out_path = _resolve_manual_plan_path(args.manual_plan_output)
            _write_manual_plan_csv(out_path, manual_plan_rows)
            if phase2_notion_chat_mode and requested_phase2:
                LOGGER.warning(
                    "Plano base da fase 2 gerado a partir da fonte atual (CSV/Notion): %s | linhas=%d",
                    out_path,
                    len(manual_plan_rows),
                )
            else:
                LOGGER.warning(
                    "Plano manual gerado para colunas ignoradas por limite API: %s | linhas=%d",
                    out_path,
                    len(manual_plan_rows),
                )

            if phase2_notion_chat_mode and requested_phase2:
                queue_path, queue_rows = _generate_phase2_notion_chat_compact_queue(
                    args=args,
                    selected_phase2_keys=requested_phase2,
                    manual_rows=manual_plan_rows,
                    option_url_lookup_by_property=option_url_lookup_by_property,
                    fallback_parent=out_path.parent,
                )
                LOGGER.warning("Fila do Notion Chat atualizada: %s", queue_path)
                _log_phase2_queue_alert(queue_rows, requested_phase2)
            else:
                should_generate_phase2_blocks = bool(args.apply)
                if should_generate_phase2_blocks:
                    block_dir = _resolve_phase2_blocks_dir(args.phase2_blocks_output_dir, out_path.parent)
                    summary_path, phase2_blocks = _write_phase2_block_files(
                        manual_rows=manual_plan_rows,
                        out_dir=block_dir,
                        block_size=args.phase2_block_size,
                        include_all_phase2_labels=False,
                    )
                    if phase2_blocks:
                        LOGGER.warning(
                            "Fase 2 em blocos (A-C, D-F, etc.) gerada para contorno do limite da API: %s | blocos=%d",
                            block_dir,
                            len(phase2_blocks),
                        )
                        for row in phase2_blocks[:12]:
                            LOGGER.warning(
                                "  - [%s] bloco=%s faixa=%s linhas=%s | cores=%s | remocoes=%s",
                                row.get("coluna"),
                                row.get("bloco"),
                                row.get("faixa_alfabetica"),
                                row.get("linhas"),
                                row.get("mudancas_cor"),
                                row.get("remocoes"),
                            )
                        if len(phase2_blocks) > 12:
                            LOGGER.warning("  ... +%d blocos", len(phase2_blocks) - 12)
                        LOGGER.warning("Resumo dos blocos da fase 2: %s", summary_path)
                    else:
                        LOGGER.info(
                            "Nenhum bloco da fase 2 foi necessario no plano manual (apenas fase 1 ou sem pendencias de acao)."
                        )

        if args.apply and failed_props > 0:
            return 2
        return 0
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Padroniza cores de etiquetas no Notion (DJe): siglaUF, nomeMunicipio, "
            "relator, partes, advogados, siglaClasse e descricaoClasse."
        )
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help="URL (ou ID) da base do Notion.",
    )
    parser.add_argument(
        "--source-csv",
        default="",
        help=(
            "CSV base para calcular uso real e mapa municipio->UF sem varrer paginas do Notion. "
            "Ex.: 'DJe - 2ª semana - FEV_26_atualizado.csv'."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Efetiva alteracoes no Notion. Sem esta flag, roda em dry-run.",
    )
    parser.add_argument(
        "--only-properties",
        default="",
        help=(
            "Lista de colunas separadas por virgula para processar exclusivamente "
            "(ex.: siglaUF,relator,siglaClasse,descricaoClasse)."
        ),
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=("all", "1", "2"),
        help=(
            "Fase de processamento: "
            "1=siglaUF/relator/siglaClasse/descricaoClasse, "
            "2=nomeMunicipio/partes/advogados, "
            "all=todas."
        ),
    )
    parser.add_argument(
        "--phase2-mode",
        default="api",
        choices=("api", "notion-chat"),
        help=(
            "Modo da fase 2: "
            "api=fluxo legado via API oficial; "
            "notion-chat=gera fila JSON/prompt para execucao manual assistida no Notion Chat."
        ),
    )
    parser.add_argument(
        "--phase2-blocks-input-dir",
        default="",
        help=(
            "Pasta de entrada com CSVs de blocos da fase 2 (phase2_*_bloco_*.csv). "
            "Uso legado/opcional para reaproveitar blocos ja existentes."
        ),
    )
    parser.add_argument(
        "--phase2-chat-output-dir",
        default="",
        help=(
            "Pasta de saida para artefatos do modo notion-chat (json/prompt/fila). "
            "Vazio = pasta atual (sem criar subpasta automatica)."
        ),
    )
    parser.add_argument(
        "--phase2-queue-file",
        default="",
        help=(
            "Arquivo CSV da fila do mode notion-chat. "
            "Vazio = <phase2-chat-output-dir>/phase2_chat_queue.csv."
        ),
    )
    parser.add_argument(
        "--no-phase2-force-default",
        action="store_false",
        dest="phase2_force_default",
        help=(
            "No mode notion-chat, nao forca default em todas as etiquetas do bloco "
            "(usa somente linhas com divergencia cor_atual vs cor_alvo)."
        ),
    )
    parser.add_argument(
        "--no-phase2-include-removals",
        action="store_false",
        dest="phase2_include_removals",
        help="No mode notion-chat, nao inclui remocoes de etiquetas sem uso na fila.",
    )
    parser.set_defaults(phase2_force_default=True, phase2_include_removals=True)
    parser.add_argument(
        "--manual-plan-output",
        default="",
        help=(
            "Arquivo CSV para registrar plano manual de colunas que a API nao conseguiu aplicar "
            "por limite de options (>100). Vazio = gera nome automatico."
        ),
    )
    parser.add_argument(
        "--phase2-block-size",
        type=int,
        default=95,
        help=(
            "Tamanho maximo de cada bloco alfabetico da fase 2 no plano manual "
            f"(padrao: 95, limite API: {NOTION_OPTIONS_PATCH_LIMIT})."
        ),
    )
    parser.add_argument(
        "--phase2-blocks-output-dir",
        default="",
        help=(
            "Pasta para salvar os CSVs de blocos alfabeticos da fase 2 "
            "(A-C, D-F, etc.) quando houver colunas ignoradas por limite API."
        ),
    )
    parser.add_argument(
        "--resume-plan-input",
        default="",
        help=(
            "CSV de checkpoint (manual-plan-output) para retomar automaticamente apenas pendencias. "
            "Quando informado, o script aplica somente colunas/etiquetas pendentes desse arquivo. "
            "Na fase 2 com --apply, o CSV pode ser processado em blocos alfabeticos automaticamente "
            f"(ate {NOTION_OPTIONS_PATCH_LIMIT} options por chamada)."
        ),
    )
    parser.add_argument(
        "--no-prune-unused",
        action="store_false",
        dest="prune_unused",
        help="Nao remove etiquetas sem uso real nos registros da base.",
    )
    parser.set_defaults(prune_unused=True)
    parser.add_argument(
        "--allow-partial-prune",
        action="store_true",
        help=(
            "Permite remover etiquetas sem uso mesmo quando --max-pages > 0 "
            "(risco de poda indevida por varredura parcial)."
        ),
    )
    parser.add_argument(
        "--no-recreate",
        action="store_false",
        dest="allow_recreate",
        help=(
            "Desativa fallback que recria opcoes quando o endpoint bloqueia update direto de cor "
            "em option existente."
        ),
    )
    parser.set_defaults(allow_recreate=True)
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Itens por pagina no query do Notion (padrao: {DEFAULT_PAGE_SIZE}).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Limite de paginas para mapear municipio->UF (0 = sem limite).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Timeout por request em segundos (padrao: {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries por request (padrao: {DEFAULT_RETRIES}).",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=DEFAULT_MIN_INTERVAL_S,
        help=f"Intervalo minimo entre updates (padrao: {DEFAULT_MIN_INTERVAL_S:.2f}s).",
    )
    parser.add_argument(
        "--notion-version",
        default=DEFAULT_NOTION_VERSION,
        help=f"Cabecalho Notion-Version (padrao: {DEFAULT_NOTION_VERSION}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa logs em nivel DEBUG.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        return run(args)
    except Exception as exc:
        LOGGER.error("Falha: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
