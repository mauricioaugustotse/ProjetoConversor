#!/usr/bin/env python3
"""Gera JSON e prompt TXT para corrigir etiquetas no Notion (fase 2).

Fluxo:
1) Le um CSV de plano manual com colunas:
   - coluna, etiqueta, cor_atual, cor_alvo, remover_se_apply
2) Busca no schema do Notion os IDs internos das opcoes (url collectionPropertyOption://...).
3) Gera, para cada coluna selecionada:
   - phase2_<coluna>.json
   - phase2_<coluna>.prompt.txt
4) Gera um resumo em phase2_chat_queue.csv.

Uso tipico:
python NOTION_corrigir_etiquetas.py \
  --manual-plan-csv notion_etiquetas_plano_manual_fase2.csv \
  --database-url "https://www.notion.so/...."
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
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import requests


NOTION_BASE_URL = "https://api.notion.com"
DEFAULT_NOTION_VERSION = "2025-09-03"
DEFAULT_DATABASE_URL = "https://www.notion.so/317721955c6480d3b642cc296d6074c7?v=6dde3c179e6b400ab0309cd7eac7d61d"
DEFAULT_PROPERTIES = ("nomeMunicipio", "partes", "advogados")

LOGGER = logging.getLogger("NOTION_corrigir_etiquetas")


def _normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_key(value: Any) -> str:
    text = _normalize_ws(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _normalize_ws(text)


def _to_bool_flag(raw: Any) -> bool:
    text = _normalize_ws(raw).lower()
    return text in {"1", "true", "t", "yes", "y", "sim", "s"}


def _sanitize_filename_token(value: str) -> str:
    token = _normalize_key(value).replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]+", "", token)
    return token or "coluna"


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
    timeout_s: int,
    retries: int,
) -> Dict[str, Any]:
    url = path if path.startswith("http") else f"{NOTION_BASE_URL}{path}"
    attempts = max(1, int(retries) + 1)

    for attempt in range(1, attempts + 1):
        try:
            response = session.request(
                method=method.upper(),
                url=url,
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


def collect_tag_properties(properties: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
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
        out[str(prop_name)] = {
            "name": str(prop_name),
            "property_id": prop_id,
            "type": prop_type,
            "options": cleaned_options,
        }
    return out


def build_option_url_lookup(
    *,
    data_source_id: str,
    tag_props: Dict[str, Dict[str, Any]],
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

        property_id = _normalize_ws(prop.get("property_id"))
        for option in prop.get("options", []):
            if not isinstance(option, dict):
                continue
            name = _normalize_ws(option.get("name"))
            option_id = _normalize_ws(option.get("id"))
            if not name or not option_id:
                continue

            option_url = _build_collection_option_url(
                data_source_id=normalized_ds_id,
                property_id=property_id,
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

        out[prop_key] = {"exact": by_exact, "key": by_key}

    return out


def load_manual_plan_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"CSV de plano manual nao encontrado: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"coluna", "etiqueta", "cor_atual", "cor_alvo", "remover_se_apply"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"CSV invalido: faltam colunas {', '.join(sorted(missing))}"
            )

        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "coluna": _normalize_ws(row.get("coluna")),
                    "etiqueta": _normalize_ws(row.get("etiqueta")),
                    "cor_atual": _normalize_ws(row.get("cor_atual")) or "default",
                    "cor_alvo": _normalize_ws(row.get("cor_alvo")) or "default",
                    "remover_se_apply": "1" if _to_bool_flag(row.get("remover_se_apply")) else "0",
                }
            )
    return rows


def _write_manual_plan_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = ["coluna", "etiqueta", "cor_atual", "cor_alvo", "remover_se_apply"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_selected_property_keys(raw: str) -> set[str]:
    return {_normalize_key(item) for item in str(raw).split(",") if _normalize_key(item)}


def _resolve_manual_plan_csv_path(raw_path: str) -> Path:
    path = Path(_normalize_ws(raw_path))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _resolve_manual_plan_output_path(*, manual_plan_output: str, manual_plan_csv: str) -> Path:
    candidate = _normalize_ws(manual_plan_output) or _normalize_ws(manual_plan_csv) or "notion_etiquetas_plano_manual_fase2.csv"
    path = Path(candidate)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _build_manual_plan_rows_from_schema(
    *,
    tag_props: Dict[str, Dict[str, Any]],
    selected_keys: set[str],
) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    rows: List[Dict[str, str]] = []
    available_by_key = {_normalize_key(name): name for name in tag_props}
    missing = [key for key in sorted(selected_keys) if key not in available_by_key]
    selected_names: List[str] = []

    for key in sorted(selected_keys):
        property_name = available_by_key.get(key)
        if not property_name:
            continue
        selected_names.append(property_name)
        prop = tag_props.get(property_name, {})
        for option in prop.get("options", []):
            if not isinstance(option, dict):
                continue
            label = _normalize_ws(option.get("name"))
            if not label:
                continue
            current_color = _normalize_ws(option.get("color")) or "default"
            rows.append(
                {
                    "coluna": property_name,
                    "etiqueta": label,
                    "cor_atual": current_color,
                    "cor_alvo": current_color,
                    "remover_se_apply": "0",
                }
            )
    return rows, selected_names, missing


def _dedupe_labels(labels: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in labels:
        name = _normalize_ws(item)
        key = _normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def build_payload_for_property(
    *,
    property_name: str,
    rows: List[Dict[str, str]],
    option_lookup: Dict[str, Dict[str, Dict[str, str]]],
    color_mode: str,
    include_removals: bool,
) -> Tuple[Dict[str, Any], int, int]:
    prop_lookup = option_lookup.get(_normalize_key(property_name), {})
    url_by_exact = prop_lookup.get("exact", {})
    url_by_key = prop_lookup.get("key", {})

    remove_unused_raw: List[str] = []
    if include_removals:
        remove_unused_raw = [
            row.get("etiqueta", "")
            for row in rows
            if _to_bool_flag(row.get("remover_se_apply"))
        ]
    remove_unused = _dedupe_labels(remove_unused_raw)
    remove_keys = {_normalize_key(label) for label in remove_unused}

    options: List[Dict[str, str]] = []
    missing_urls: List[str] = []
    seen: set[str] = set()
    for row in rows:
        name = _normalize_ws(row.get("etiqueta"))
        key = _normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in remove_keys:
            continue

        color = "default" if color_mode == "default" else (_normalize_ws(row.get("cor_alvo")) or "default")
        option_url = _normalize_ws(url_by_exact.get(name) or url_by_key.get(key))
        if not option_url:
            missing_urls.append(name)
            continue

        options.append(
            {
                "name": name,
                "url": option_url,
                "color": color,
            }
        )

    if missing_urls:
        sample = ", ".join(missing_urls[:8])
        if len(missing_urls) > 8:
            sample += f" ... +{len(missing_urls) - 8}"
        raise RuntimeError(
            f"[{property_name}] {len(missing_urls)} etiqueta(s) sem URL interna no schema do Notion. Exemplos: {sample}"
        )

    payload = {"property": property_name, "options": options}
    return payload, len(options), len(remove_unused)


def build_prompt_text(
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
        "Formato solicitado:\n"
        "{\n"
        '  "property": "...",\n'
        '  "options": [\n'
        '    {"name": "...", "url": "...", "color": "..."}\n'
        "  ]\n"
        "}\n\n"
        "JSON da propriedade:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera JSON + prompt TXT para correcao de etiquetas (fase 2) no Notion Chat."
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help="URL (ou ID) da base do Notion.",
    )
    parser.add_argument(
        "--manual-plan-csv",
        default="notion_etiquetas_plano_manual_fase2.csv",
        help="CSV de plano manual com colunas/etiquetas alvo.",
    )
    parser.add_argument(
        "--generate-manual-plan-only",
        action="store_true",
        help="Gera somente o CSV de plano manual a partir do schema da base (sem gerar JSON/TXT).",
    )
    parser.add_argument(
        "--manual-plan-output",
        default="",
        help="Arquivo de saida para o CSV manual quando usado com --generate-manual-plan-only.",
    )
    parser.add_argument(
        "--properties",
        default=",".join(DEFAULT_PROPERTIES),
        help="Colunas alvo separadas por virgula (padrao: nomeMunicipio,partes,advogados).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Pasta de saida dos arquivos JSON/TXT/queue.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("default", "target"),
        default="default",
        help="default=todas as opcoes com color=default; target=usa cor_alvo do CSV.",
    )
    parser.add_argument(
        "--no-removals",
        action="store_true",
        help="Nao considerar remover_se_apply para contagem de removidos.",
    )
    parser.add_argument(
        "--notion-version",
        default=DEFAULT_NOTION_VERSION,
        help=f"Cabecalho Notion-Version (padrao: {DEFAULT_NOTION_VERSION}).",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Timeout HTTP em segundos.")
    parser.add_argument("--retries", type=int, default=4, help="Tentativas adicionais em caso de falha.")
    parser.add_argument("--debug", action="store_true", help="Habilita logs DEBUG.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    selected_keys = _parse_selected_property_keys(args.properties)
    if not selected_keys:
        raise RuntimeError("Nenhuma coluna selecionada em --properties.")

    if args.generate_manual_plan_only:
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
        data_source_id = retrieve_data_source_id(
            session,
            database_id=database_id,
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
        manual_rows, selected_names, missing_keys = _build_manual_plan_rows_from_schema(
            tag_props=tag_props,
            selected_keys=selected_keys,
        )
        if missing_keys:
            human = ", ".join(missing_keys)
            raise RuntimeError(
                f"Colunas nao encontradas no schema da base: {human}"
            )

        manual_out = _resolve_manual_plan_output_path(
            manual_plan_output=args.manual_plan_output,
            manual_plan_csv=args.manual_plan_csv,
        )
        _write_manual_plan_csv(manual_out, manual_rows)
        LOGGER.info(
            "CSV manual gerado | arquivo=%s | linhas=%d | colunas=%s",
            manual_out,
            len(manual_rows),
            ",".join(selected_names),
        )
        print("[OK] CSV manual gerado.")
        return 0

    manual_plan_path = _resolve_manual_plan_csv_path(args.manual_plan_csv)
    rows = load_manual_plan_rows(manual_plan_path)
    LOGGER.info("CSV carregado | arquivo=%s | linhas=%d", manual_plan_path, len(rows))

    output_dir = Path(_normalize_ws(args.output_dir) or ".")
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    notion_token = resolve_notion_key()
    if not notion_token:
        raise RuntimeError(
            "Chave do Notion ausente. Crie Chave_Notion.txt ou defina NOTION_API_KEY/NOTION_TOKEN."
        )

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        col = _normalize_ws(row.get("coluna"))
        col_key = _normalize_key(col)
        if col_key not in selected_keys:
            continue
        grouped.setdefault(col, []).append(row)
    if not grouped:
        raise RuntimeError("Nenhuma linha do CSV corresponde as colunas selecionadas.")

    database_id = extract_notion_id_from_url(args.database_url)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {notion_token}",
            "Notion-Version": args.notion_version,
            "Content-Type": "application/json",
        }
    )

    data_source_id = retrieve_data_source_id(
        session,
        database_id=database_id,
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
    option_lookup = build_option_url_lookup(data_source_id=data_source_id, tag_props=tag_props)

    queue_rows: List[Dict[str, str]] = []
    for property_name in sorted(grouped.keys(), key=_normalize_key):
        payload, set_default_count, remove_unused_count = build_payload_for_property(
            property_name=property_name,
            rows=grouped[property_name],
            option_lookup=option_lookup,
            color_mode=args.color_mode,
            include_removals=not args.no_removals,
        )

        prop_token = _sanitize_filename_token(property_name)
        json_path = output_dir / f"phase2_{prop_token}.json"
        prompt_path = output_dir / f"phase2_{prop_token}.prompt.txt"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prompt_path.write_text(
            build_prompt_text(
                database_url=args.database_url,
                payload=payload,
                set_default_count=set_default_count,
                remove_unused_count=remove_unused_count,
            ),
            encoding="utf-8",
        )

        queue_rows.append(
            {
                "coluna": property_name,
                "arquivo_json": str(json_path),
                "arquivo_prompt": str(prompt_path),
                "total_options": str(set_default_count),
                "total_remove_unused": str(remove_unused_count),
                "status": "GERADO",
            }
        )
        LOGGER.info(
            "[%s] Arquivos gerados | options=%d | removidos=%d | json=%s",
            property_name,
            set_default_count,
            remove_unused_count,
            json_path,
        )

    queue_path = output_dir / "phase2_chat_queue.csv"
    with queue_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["coluna", "arquivo_json", "arquivo_prompt", "total_options", "total_remove_unused", "status"],
        )
        writer.writeheader()
        writer.writerows(queue_rows)

    LOGGER.info("Fila gerada: %s | itens=%d", queue_path, len(queue_rows))
    print("[OK] Processo finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

