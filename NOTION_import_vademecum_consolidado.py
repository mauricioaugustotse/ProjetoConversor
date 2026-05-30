#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import NOTION_import_codigo_eleitoral_2026_csvs as notion_import


DEFAULT_VADEMECUM_PAGE_URL = "https://www.notion.so/vademecum-323721955c64809bb7e6e1d98b27246f"
DEFAULT_RICD_DATABASE_URL = "https://www.notion.so/356721955c648014bff1c0cbd6f0484a?v=b826e79c86354644ad90675d2e748e90"
DEFAULT_OUTPUT_CSV = Path("vademecum_rag_consolidado.csv")
DEFAULT_CHECKPOINT = Path("vademecum_rag_consolidado_checkpoint.json")
DEFAULT_DATABASE_TITLE = "Vademecum - RAG consolidado"
RELATION_PROPERTY = "dispositivo_pai_relacao"
TITLE_PROPERTY = "Nome"

LOGGER = logging.getLogger("notion_vademecum_consolidado")

CSV_COLUMNS = [
    "id",
    "ordem_global",
    "row_key",
    "titulo_linha",
    "tipo_linha",
    "tipo_base",
    "linha_sintetica",
    "incluir_no_rag",
    "norma_id",
    "norma_titulo",
    "norma_numero",
    "norma_ano",
    "norma_nome_popular",
    "grupo",
    "fonte_tipo",
    "fonte_database_titulo",
    "fonte_database_id",
    "fonte_data_source_id",
    "fonte_page_id",
    "id_fonte",
    "ordem_doc",
    "tipo_dispositivo",
    "dispositivo",
    "dispositivo_pai",
    "dispositivo_pai_row_key",
    "hierarquia_normativa",
    "documento_extraido",
    "texto_dispositivo",
    "texto_em_vigor",
    "notas_texto",
    "notas_json",
    "qtd_notas",
    "referencias_normativas",
    "referencias_jurisprudenciais",
    "resumo_curto",
    "palavras_chave",
    "alerta_qualidade",
    "qualidade_texto",
    "texto_rag",
    "chars",
    "tokens_estimados",
]

NUMBER_COLUMNS = {
    "id",
    "ordem_global",
    "norma_ano",
    "id_fonte",
    "ordem_doc",
    "qtd_notas",
    "chars",
    "tokens_estimados",
}
CHECKBOX_COLUMNS = {"linha_sintetica", "incluir_no_rag"}
SELECT_COLUMNS = {"tipo_linha", "tipo_base", "grupo", "fonte_tipo", "tipo_dispositivo", "qualidade_texto"}

KNOWN_NORMS: Dict[str, Dict[str, str]] = {
    "constituicao": {
        "norma_id": "constituicao_federal_1988",
        "norma_titulo": "Constituicao da Republica Federativa do Brasil de 1988",
        "norma_numero": "",
        "norma_ano": "1988",
        "norma_nome_popular": "Constituicao Federal",
        "grupo": "constitucional",
    },
    "codigo eleitoral": {
        "norma_id": "lei_4737_1965",
        "norma_titulo": "Lei n. 4.737, de 15 de julho de 1965",
        "norma_numero": "4737",
        "norma_ano": "1965",
        "norma_nome_popular": "Codigo Eleitoral",
        "grupo": "eleitoral",
    },
    "lei das eleicoes": {
        "norma_id": "lei_9504_1997",
        "norma_titulo": "Lei n. 9.504, de 30 de setembro de 1997",
        "norma_numero": "9504",
        "norma_ano": "1997",
        "norma_nome_popular": "Lei das Eleicoes",
        "grupo": "eleitoral",
    },
    "lc64": {
        "norma_id": "lei_complementar_64_1990",
        "norma_titulo": "Lei Complementar n. 64, de 18 de maio de 1990",
        "norma_numero": "64",
        "norma_ano": "1990",
        "norma_nome_popular": "Lei de Inelegibilidade",
        "grupo": "eleitoral",
    },
    "lindb": {
        "norma_id": "decreto_lei_4657_1942",
        "norma_titulo": "Decreto-Lei n. 4.657, de 4 de setembro de 1942",
        "norma_numero": "4657",
        "norma_ano": "1942",
        "norma_nome_popular": "LINDB",
        "grupo": "geral",
    },
    "decreto da lindb": {
        "norma_id": "decreto_9830_2019",
        "norma_titulo": "Decreto n. 9.830, de 10 de junho de 2019",
        "norma_numero": "9830",
        "norma_ano": "2019",
        "norma_nome_popular": "Decreto da LINDB",
        "grupo": "geral",
    },
    "cc": {
        "norma_id": "lei_10406_2002",
        "norma_titulo": "Lei n. 10.406, de 10 de janeiro de 2002",
        "norma_numero": "10406",
        "norma_ano": "2002",
        "norma_nome_popular": "Codigo Civil",
        "grupo": "civil",
    },
    "cpc": {
        "norma_id": "lei_13105_2015",
        "norma_titulo": "Lei n. 13.105, de 16 de marco de 2015",
        "norma_numero": "13105",
        "norma_ano": "2015",
        "norma_nome_popular": "Codigo de Processo Civil",
        "grupo": "processual_civil",
    },
    "cdc": {
        "norma_id": "lei_8078_1990",
        "norma_titulo": "Lei n. 8.078, de 11 de setembro de 1990",
        "norma_numero": "8078",
        "norma_ano": "1990",
        "norma_nome_popular": "Codigo de Defesa do Consumidor",
        "grupo": "consumidor",
    },
    "ctn": {
        "norma_id": "lei_5172_1966",
        "norma_titulo": "Lei n. 5.172, de 25 de outubro de 1966",
        "norma_numero": "5172",
        "norma_ano": "1966",
        "norma_nome_popular": "Codigo Tributario Nacional",
        "grupo": "tributario",
    },
    "clt": {
        "norma_id": "decreto_lei_5452_1943",
        "norma_titulo": "Decreto-Lei n. 5.452, de 1 de maio de 1943",
        "norma_numero": "5452",
        "norma_ano": "1943",
        "norma_nome_popular": "CLT",
        "grupo": "trabalhista",
    },
    "ctb": {
        "norma_id": "lei_9503_1997",
        "norma_titulo": "Lei n. 9.503, de 23 de setembro de 1997",
        "norma_numero": "9503",
        "norma_ano": "1997",
        "norma_nome_popular": "Codigo de Transito Brasileiro",
        "grupo": "transito",
    },
    "lei de introducao ao codigo penal": {
        "norma_id": "decreto_lei_3914_1941",
        "norma_titulo": "Decreto-Lei n. 3.914, de 9 de dezembro de 1941",
        "norma_numero": "3914",
        "norma_ano": "1941",
        "norma_nome_popular": "Lei de Introducao ao Codigo Penal",
        "grupo": "penal",
    },
    "cp": {
        "norma_id": "decreto_lei_2848_1940",
        "norma_titulo": "Decreto-Lei n. 2.848, de 7 de dezembro de 1940",
        "norma_numero": "2848",
        "norma_ano": "1940",
        "norma_nome_popular": "Codigo Penal",
        "grupo": "penal",
    },
    "lei de introducao ao codigo de processo penal": {
        "norma_id": "decreto_lei_3931_1941",
        "norma_titulo": "Decreto-Lei n. 3.931, de 11 de dezembro de 1941",
        "norma_numero": "3931",
        "norma_ano": "1941",
        "norma_nome_popular": "Lei de Introducao ao Codigo de Processo Penal",
        "grupo": "processual_penal",
    },
    "cpp": {
        "norma_id": "decreto_lei_3689_1941",
        "norma_titulo": "Decreto-Lei n. 3.689, de 3 de outubro de 1941",
        "norma_numero": "3689",
        "norma_ano": "1941",
        "norma_nome_popular": "Codigo de Processo Penal",
        "grupo": "processual_penal",
    },
    "cpm": {
        "norma_id": "decreto_lei_1001_1969",
        "norma_titulo": "Decreto-Lei n. 1.001, de 21 de outubro de 1969",
        "norma_numero": "1001",
        "norma_ano": "1969",
        "norma_nome_popular": "Codigo Penal Militar",
        "grupo": "penal_militar",
    },
    "cppm": {
        "norma_id": "decreto_lei_1002_1969",
        "norma_titulo": "Decreto-Lei n. 1.002, de 21 de outubro de 1969",
        "norma_numero": "1002",
        "norma_ano": "1969",
        "norma_nome_popular": "Codigo de Processo Penal Militar",
        "grupo": "processual_penal_militar",
    },
    "lei 15190": {
        "norma_id": "lei_15190",
        "norma_titulo": "Lei n. 15.190",
        "norma_numero": "15190",
        "norma_ano": "",
        "norma_nome_popular": "Lei n. 15.190",
        "grupo": "legislacao_especial",
    },
    "l1079": {
        "norma_id": "lei_1079_1950",
        "norma_titulo": "Lei n. 1.079, de 10 de abril de 1950",
        "norma_numero": "1079",
        "norma_ano": "1950",
        "norma_nome_popular": "Lei dos Crimes de Responsabilidade",
        "grupo": "legislacao_especial",
    },
    "l9605.csv": {
        "norma_id": "lei_9605_1998",
        "norma_titulo": "Lei n. 9.605, de 12 de fevereiro de 1998",
        "norma_numero": "9605",
        "norma_ano": "1998",
        "norma_nome_popular": "Lei de Crimes Ambientais",
        "grupo": "ambiental",
    },
}


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\ufeff", " ")).strip()


def normalize_text_block(text: Any) -> str:
    raw = str(text or "").replace("\ufeff", " ").replace("\x00", "")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [normalize_ws(line) for line in raw.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines) if lines else normalize_ws(raw)


def ascii_key(text: Any) -> str:
    value = str(text or "").replace("º", "o").replace("ª", "a").replace("–", "-").replace("—", "-")
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()


def slugify(text: Any, *, max_len: int = 96) -> str:
    value = ascii_key(text)
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return (value[:max_len].strip("_") or "item")


def stable_key_slug(text: Any, *, max_slug_len: int = 72) -> str:
    base = slugify(text, max_len=max_slug_len)
    digest = hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{base}_{digest}"


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def limit_text(text: str, max_chars: int = 420) -> str:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}..."


def first_sentence(text: str) -> str:
    text = normalize_ws(text)
    if not text:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-ZÁ-Ú0-9])", text)
    return normalize_ws(pieces[0] if pieces else text)


def split_device_path(path: str) -> List[str]:
    return [normalize_ws(part) for part in normalize_ws(path).split(",") if normalize_ws(part)]


def join_device_path(parts: Sequence[str]) -> str:
    return ", ".join(part for part in parts if part)


def parent_device_path(path: str) -> str:
    parts = split_device_path(path)
    return join_device_path(parts[:-1]) if len(parts) > 1 else ""


def last_device_part(path: str) -> str:
    parts = split_device_path(path)
    return parts[-1] if parts else normalize_ws(path)


def split_hierarchy(path: str) -> List[str]:
    raw = normalize_ws(path)
    if not raw:
        return []
    separator = ">" if ">" in raw else "|"
    return [normalize_ws(part) for part in raw.split(separator) if normalize_ws(part)]


def join_hierarchy(parts: Sequence[str]) -> str:
    return " > ".join(part for part in parts if part)


def infer_tipo_dispositivo(path: str, *, structure: bool = False) -> str:
    if structure:
        part = normalize_ws(path)
    else:
        part = last_device_part(path)
    key = ascii_key(part)
    if not part:
        return "dispositivo"
    if structure:
        if key.startswith("parte"):
            return "parte"
        if key.startswith("livro"):
            return "livro"
        if key.startswith("titulo"):
            return "titulo"
        if key.startswith("capitulo"):
            return "capitulo"
        if key.startswith("secao"):
            return "secao"
        if key.startswith("subsecao"):
            return "subsecao"
        return "estrutura"
    if key.startswith("art."):
        return "artigo"
    if "paragrafo unico" in key or key.startswith("§"):
        return "paragrafo"
    if key.startswith("inciso ") or re.fullmatch(r"[ivxlcdm]+", part, flags=re.IGNORECASE):
        return "inciso"
    if key.startswith("alinea ") or re.fullmatch(r"[a-z]\)?", part, flags=re.IGNORECASE):
        return "alinea"
    if key.startswith("item ") or re.fullmatch(r"\d+(?:\.\d+)*", part):
        return "item"
    return "dispositivo"


def property_plain_value(prop: Mapping[str, Any]) -> str:
    prop_type = normalize_ws(prop.get("type"))
    if prop_type in {"title", "rich_text"}:
        return notion_import.property_plain_text(prop)
    if prop_type == "number":
        value = prop.get("number")
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if prop_type == "select":
        value = prop.get("select") or {}
        return normalize_ws(value.get("name")) if isinstance(value, dict) else ""
    if prop_type == "multi_select":
        values = prop.get("multi_select") or []
        return "; ".join(normalize_ws(item.get("name")) for item in values if isinstance(item, dict) and item.get("name"))
    if prop_type == "checkbox":
        return "true" if prop.get("checkbox") else "false"
    if prop_type == "date":
        value = prop.get("date") or {}
        if isinstance(value, dict):
            return normalize_ws(value.get("start"))
    return ""


def extract_title_from_rich_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    return "".join(str(item.get("plain_text") or "") for item in items if isinstance(item, dict)).strip()


def database_title(payload: Mapping[str, Any], fallback: str = "") -> str:
    return extract_title_from_rich_text(payload.get("title")) or fallback


def data_source_id_for_database(client: notion_import.NotionClient, database_id: str) -> str:
    payload = client.request("GET", f"/v1/databases/{database_id}")
    return notion_import.extract_data_source_id(client, database_id, payload)


def list_child_databases(client: notion_import.NotionClient, page_id: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    cursor = ""
    while True:
        params: Dict[str, Any] = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        payload = client.request("GET", f"/v1/blocks/{page_id}/children", params=params)
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict) or item.get("type") != "child_database":
                continue
            child = item.get("child_database") or {}
            database_id = notion_import.normalize_notion_id(str(item.get("id", "")))
            results.append(
                {
                    "title": normalize_ws(child.get("title")),
                    "database_id": database_id,
                    "data_source_id": data_source_id_for_database(client, database_id),
                    "source_kind": "vademecum_child_database",
                }
            )
        if not payload.get("has_more"):
            break
        cursor = normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return results


def get_database_source(client: notion_import.NotionClient, database_url_or_id: str, *, source_kind: str) -> Dict[str, str]:
    database_id = notion_import.extract_notion_id_from_url(database_url_or_id)
    payload = client.request("GET", f"/v1/databases/{database_id}")
    return {
        "title": database_title(payload, fallback=database_id),
        "database_id": database_id,
        "data_source_id": notion_import.extract_data_source_id(client, database_id, payload),
        "source_kind": source_kind,
    }


def find_source_property(props: Mapping[str, Any], names: Sequence[str], *, prop_type: str = "") -> str:
    exact = {ascii_key(name): name for name in props}
    for name in names:
        key = ascii_key(name)
        if key in exact:
            return exact[key]
    for name, prop in props.items():
        if prop_type and normalize_ws(prop.get("type")) != prop_type:
            continue
        name_key = ascii_key(name)
        if any(ascii_key(candidate) in name_key for candidate in names):
            return name
    return ""


def query_source_rows(client: notion_import.NotionClient, source: Mapping[str, str]) -> List[Dict[str, str]]:
    data_source_id = normalize_ws(source.get("data_source_id"))
    rows: List[Dict[str, str]] = []
    cursor = ""
    while True:
        body: Dict[str, Any] = {"page_size": 100, "result_type": "page"}
        if cursor:
            body["start_cursor"] = cursor
        payload = client.request("POST", f"/v1/data_sources/{data_source_id}/query", json_body=body)
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict) or item.get("object") != "page":
                continue
            props = item.get("properties") or {}
            if not isinstance(props, dict):
                continue
            id_prop = find_source_property(props, ["ID", "id"], prop_type="number")
            name_prop = find_source_property(props, ["Nome (Artigo)", "Nome", "Artigo"], prop_type="title")
            text_prop = find_source_property(props, ["Texto do Artigo", "Texto", "texto_dispositivo"], prop_type="rich_text")
            rows.append(
                {
                    "fonte_page_id": notion_import.normalize_notion_id(str(item.get("id", ""))),
                    "ID": property_plain_value(props.get(id_prop, {}) or {}) if id_prop else "",
                    "Nome (Artigo)": property_plain_value(props.get(name_prop, {}) or {}) if name_prop else "",
                    "Texto do Artigo": property_plain_value(props.get(text_prop, {}) or {}) if text_prop else "",
                }
            )
        if not payload.get("has_more"):
            break
        cursor = normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break

    def row_sort_key(row: Mapping[str, str]) -> Tuple[int, str]:
        raw = normalize_ws(row.get("ID"))
        try:
            return (int(float(raw)), raw)
        except ValueError:
            return (10**12, raw)

    return sorted(rows, key=row_sort_key)


def parse_embedded_metadata(text: str) -> Tuple[Dict[str, str], str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw.split("\n")]
    meta: Dict[str, str] = {}
    consumed_until = 0
    for index, line in enumerate(lines[:10]):
        if not line:
            consumed_until = index + 1 if meta else consumed_until
            if meta:
                continue
            break
        match = re.match(r"^(Documento|Hierarquia|Dispositivo)\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if not match:
            break
        key = ascii_key(match.group(1))
        meta[key] = normalize_ws(match.group(2))
        consumed_until = index + 1
    body = "\n".join(line for line in lines[consumed_until:] if line.strip()) if meta else raw
    return meta, normalize_text_block(body)


def derive_norm_from_source(source_title: str, document: str = "") -> Dict[str, str]:
    source_key = ascii_key(source_title)
    document_key = ascii_key(document)
    if document:
        if "regimento interno da camara dos deputados" in document_key:
            return {
                "norma_id": "regimento_interno_camara_deputados",
                "norma_titulo": "Regimento Interno da Camara dos Deputados",
                "norma_numero": "",
                "norma_ano": "",
                "norma_nome_popular": "RICD",
                "grupo": "camara_deputados",
            }
        if "codigo de etica" in document_key and "camara dos deputados" in document_key:
            return {
                "norma_id": "codigo_etica_decoro_camara_deputados",
                "norma_titulo": "Codigo de Etica e Decoro Parlamentar da Camara dos Deputados",
                "norma_numero": "",
                "norma_ano": "",
                "norma_nome_popular": "Codigo de Etica da Camara",
                "grupo": "camara_deputados",
            }
        match = re.search(r"\b(?:lei|decreto|resolucao|lei complementar|decreto-lei)\s*(?:n[.oº]*)?\s*([\d.]+)(?:[,/ ]+.*?\b(19|20)\d{2})?", document, flags=re.IGNORECASE)
        year_match = re.search(r"\b(19|20)\d{2}\b", document)
        number = re.sub(r"\D+", "", match.group(1)) if match else ""
        year = year_match.group(0) if year_match else ""
        return {
            "norma_id": slugify("_".join(part for part in (source_title, document, number, year) if part), max_len=90),
            "norma_titulo": document,
            "norma_numero": number,
            "norma_ano": year,
            "norma_nome_popular": document,
            "grupo": "vademecum",
        }
    if source_key in KNOWN_NORMS:
        return dict(KNOWN_NORMS[source_key])
    clean_title = normalize_ws(source_title)
    return {
        "norma_id": slugify(clean_title),
        "norma_titulo": clean_title,
        "norma_numero": "",
        "norma_ano": "",
        "norma_nome_popular": clean_title,
        "grupo": "vademecum",
    }


def extract_normative_refs(*texts: str) -> str:
    text = normalize_ws(" ".join(t for t in texts if t))
    if not text:
        return ""
    patterns = [
        r"\b(?:Constituicao Federal|CF|Lei(?: Complementar)?|LC|Decreto(?:-Lei)?|Res(?:\.|olucao)?|Emenda Constitucional|EC)\s+n[.oº]*\s*[\d.]+(?:/\d{4})?",
        r"\b(?:Lei|Decreto|Res(?:\.|olucao)?)\s+n[.oº]*\s*[\d.]+,\s*de\s*\d{1,2}\s+de\s+[a-zç]+\s+de\s+\d{4}",
        r"\bart[.]?\s*\d+[A-Z]?(?:-[A-Z])?(?:[oº])?(?:,\s*(?:§|inciso|paragrafo)\s*[\wº.-]+)?",
    ]
    found: List[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = normalize_ws(match.group(0)).rstrip(" ,;:.")
            key = ascii_key(value)
            if value and key not in seen:
                seen.add(key)
                found.append(value)
            if len(found) >= 80:
                return limit_text("; ".join(found), 1800)
    return limit_text("; ".join(found), 1800)


def extract_juris_refs(*texts: str) -> str:
    text = normalize_ws(" ".join(t for t in texts if t))
    if not text:
        return ""
    patterns = [
        r"\b(?:ADI|ADC|ADPF|ADO|ARE|HC|MS|MI|RE|REsp|REspe|REspEl|RO|RMS|Rcl|Pet|AgR-[A-Za-z]+)\s+n[.oº]*\s*[\d./-]+",
        r"\bTema\s+n[.oº]*\s*\d+",
        r"\bSumula(?: Vinculante)?\s+n[.oº]*\s*\d+",
    ]
    found: List[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = normalize_ws(match.group(0)).rstrip(" ,;:.")
            key = ascii_key(value)
            if value and key not in seen:
                seen.add(key)
                found.append(value)
            if len(found) >= 50:
                return limit_text("; ".join(found), 1800)
    return limit_text("; ".join(found), 1800)


def quality_alert(text: str) -> Tuple[str, str]:
    clean = normalize_ws(text)
    if not clean:
        return "texto_ausente", "critico"
    alerts: List[str] = []
    if clean.count("(") > clean.count(")"):
        alerts.append("possivel_truncamento_parenteses")
    if re.search(r"\b(?:no|na|nos|nas|do|da|dos|das|de|em|previsto no|disposto no|caput e no|referidas no)$", clean, flags=re.IGNORECASE):
        alerts.append("possivel_truncamento_final")
    elif re.search(r"\b(?:inciso|artigo|paragrafo|alinea|Lei|Decreto|Resolucao|n[.oº]*)$", clean, flags=re.IGNORECASE):
        alerts.append("possivel_truncamento_final")
    if len(clean) >= 1950 and not re.search(r"[.;:!?)]$", clean):
        alerts.append("texto_longo_sem_pontuacao_final")
    if alerts:
        return "; ".join(dict.fromkeys(alerts)), "verificar"
    return "", "ok"


def make_summary(row: Mapping[str, str]) -> str:
    tipo_linha = normalize_ws(row.get("tipo_linha"))
    norma = normalize_ws(row.get("norma_nome_popular")) or normalize_ws(row.get("norma_titulo"))
    dispositivo = normalize_ws(row.get("dispositivo"))
    if tipo_linha == "norma":
        return limit_text(f"Raiz normativa para navegacao e busca RAG: {norma}.")
    if tipo_linha == "estrutura":
        return limit_text(f"Estrutura hierarquica de {norma}: {normalize_ws(row.get('hierarquia_normativa'))}.")
    sentence = first_sentence(row.get("texto_em_vigor", "") or row.get("texto_dispositivo", ""))
    if sentence:
        return limit_text(f"{norma}, {dispositivo}: {sentence}")
    return limit_text(f"{norma}, {dispositivo}.")


def build_text_rag(row: Mapping[str, str]) -> str:
    labels = [
        ("Base", "Vademecum"),
        ("Norma", row.get("norma_titulo", "")),
        ("Nome popular", row.get("norma_nome_popular", "")),
        ("Grupo", row.get("grupo", "")),
        ("Tipo da linha", row.get("tipo_linha", "")),
        ("Tipo do dispositivo", row.get("tipo_dispositivo", "")),
        ("Dispositivo", row.get("dispositivo", "")),
        ("Hierarquia", row.get("hierarquia_normativa", "")),
        ("Dispositivo pai", row.get("dispositivo_pai", "")),
        ("Row key do pai", row.get("dispositivo_pai_row_key", "")),
        ("Fonte", row.get("fonte_database_titulo", "")),
        ("Fonte page id", row.get("fonte_page_id", "")),
        ("Qualidade", row.get("qualidade_texto", "")),
        ("Alerta de qualidade", row.get("alerta_qualidade", "")),
    ]
    parts = [f"{label}: {normalize_ws(value)}" for label, value in labels if normalize_ws(value)]
    for label, column in [
        ("Resumo", "resumo_curto"),
        ("Referencias normativas", "referencias_normativas"),
        ("Referencias jurisprudenciais", "referencias_jurisprudenciais"),
        ("Texto em vigor", "texto_em_vigor"),
        ("Notas", "notas_texto"),
    ]:
        value = normalize_text_block(row.get(column, ""))
        if value:
            parts.extend(["", f"{label}:", value])
    return "\n".join(parts).strip()


def base_row(
    *,
    row_key: str,
    titulo_linha: str,
    tipo_linha: str,
    tipo_base: str,
    metadata: Mapping[str, str],
    linha_sintetica: bool,
    source: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    source = source or {}
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "row_key": row_key,
            "titulo_linha": titulo_linha,
            "tipo_linha": tipo_linha,
            "tipo_base": tipo_base,
            "linha_sintetica": "true" if linha_sintetica else "false",
            "incluir_no_rag": "true",
            "norma_id": metadata.get("norma_id", ""),
            "norma_titulo": metadata.get("norma_titulo", ""),
            "norma_numero": metadata.get("norma_numero", ""),
            "norma_ano": metadata.get("norma_ano", ""),
            "norma_nome_popular": metadata.get("norma_nome_popular", ""),
            "grupo": metadata.get("grupo", ""),
            "fonte_tipo": source.get("source_kind", ""),
            "fonte_database_titulo": source.get("title", ""),
            "fonte_database_id": source.get("database_id", ""),
            "fonte_data_source_id": source.get("data_source_id", ""),
            "qtd_notas": "0",
        }
    )
    return row


class ConsolidatedBuilder:
    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []
        self.rows_by_key: Dict[str, Dict[str, str]] = {}
        self.root_by_norma: Dict[str, str] = {}
        self.sources_by_norma: Dict[str, set[str]] = defaultdict(set)
        self.structure_by_norm_path: Dict[Tuple[str, str], str] = {}
        self.actual_by_norm_device_path: Dict[Tuple[str, str], str] = {}

    def add_row(self, row: Dict[str, str]) -> str:
        row_key = normalize_ws(row.get("row_key"))
        if not row_key:
            raise RuntimeError("Linha sem row_key.")
        if row_key in self.rows_by_key:
            raise RuntimeError(f"row_key duplicado: {row_key}")
        row["referencias_normativas"] = extract_normative_refs(row.get("texto_em_vigor", ""), row.get("notas_texto", ""))
        row["referencias_jurisprudenciais"] = extract_juris_refs(row.get("texto_em_vigor", ""), row.get("notas_texto", ""))
        row["resumo_curto"] = make_summary(row)
        row["texto_rag"] = build_text_rag(row)
        row["chars"] = str(len(row["texto_rag"]))
        row["tokens_estimados"] = str(estimate_tokens(row["texto_rag"]))
        self.rows_by_key[row_key] = row
        self.rows.append(row)
        return row_key

    def refresh_row(self, row_key: str) -> None:
        row = self.rows_by_key[row_key]
        row["referencias_normativas"] = extract_normative_refs(row.get("texto_em_vigor", ""), row.get("notas_texto", ""))
        row["referencias_jurisprudenciais"] = extract_juris_refs(row.get("texto_em_vigor", ""), row.get("notas_texto", ""))
        row["resumo_curto"] = make_summary(row)
        row["texto_rag"] = build_text_rag(row)
        row["chars"] = str(len(row["texto_rag"]))
        row["tokens_estimados"] = str(estimate_tokens(row["texto_rag"]))

    def ensure_root(self, metadata: Mapping[str, str], source: Mapping[str, str]) -> str:
        norma_id = normalize_ws(metadata.get("norma_id"))
        if not norma_id:
            raise RuntimeError(f"Metadata sem norma_id: {metadata}")
        if source.get("title"):
            self.sources_by_norma[norma_id].add(source["title"])
        existing = self.root_by_norma.get(norma_id)
        if existing:
            return existing
        title = metadata.get("norma_titulo") or metadata.get("norma_nome_popular") or norma_id
        row_key = f"norma:{norma_id}"
        row = base_row(
            row_key=row_key,
            titulo_linha=title,
            tipo_linha="norma",
            tipo_base="sintetico",
            metadata=metadata,
            linha_sintetica=True,
            source=source,
        )
        row.update(
            {
                "tipo_dispositivo": "norma",
                "dispositivo": title,
                "hierarquia_normativa": title,
                "texto_dispositivo": title,
                "texto_em_vigor": title,
                "qualidade_texto": "ok",
            }
        )
        self.add_row(row)
        self.root_by_norma[norma_id] = row_key
        return row_key

    def ensure_structure(
        self,
        *,
        metadata: Mapping[str, str],
        source: Mapping[str, str],
        hierarchy_parts: Sequence[str],
    ) -> str:
        if not hierarchy_parts:
            return self.ensure_root(metadata, source)
        parent_key = self.ensure_root(metadata, source)
        for depth in range(1, len(hierarchy_parts) + 1):
            current_parts = list(hierarchy_parts[:depth])
            full_path = join_hierarchy(current_parts)
            key = (metadata["norma_id"], full_path)
            existing = self.structure_by_norm_path.get(key)
            if existing:
                parent_key = existing
                continue
            row_key = f"estrutura:{metadata['norma_id']}:{stable_key_slug(full_path)}"
            parent_label = self.rows_by_key.get(parent_key, {}).get("dispositivo", "")
            row = base_row(
                row_key=row_key,
                titulo_linha=f"{metadata.get('norma_nome_popular') or metadata.get('norma_titulo')} - {current_parts[-1]}",
                tipo_linha="estrutura",
                tipo_base="sintetico",
                metadata=metadata,
                linha_sintetica=True,
                source=source,
            )
            row.update(
                {
                    "tipo_dispositivo": infer_tipo_dispositivo(current_parts[-1], structure=True),
                    "dispositivo": current_parts[-1],
                    "dispositivo_pai": parent_label,
                    "dispositivo_pai_row_key": parent_key,
                    "hierarquia_normativa": full_path,
                    "texto_dispositivo": full_path,
                    "texto_em_vigor": full_path,
                    "qualidade_texto": "ok",
                }
            )
            self.add_row(row)
            self.structure_by_norm_path[key] = row_key
            parent_key = row_key
        return parent_key

    def add_source(self, source: Mapping[str, str], rows: Sequence[Mapping[str, str]]) -> None:
        prepared: List[Dict[str, str]] = []
        source_title = normalize_ws(source.get("title"))
        for index, item in enumerate(rows, start=1):
            raw_name = normalize_ws(item.get("Nome (Artigo)"))
            raw_text = str(item.get("Texto do Artigo") or "")
            embedded, body = parse_embedded_metadata(raw_text)
            document = normalize_ws(embedded.get("documento"))
            metadata = derive_norm_from_source(source_title, document)
            device_path = normalize_ws(embedded.get("dispositivo")) or raw_name
            hierarchy_parts = split_hierarchy(embedded.get("hierarquia", ""))
            text = body or normalize_text_block(raw_text)
            prepared.append(
                {
                    "index": str(index),
                    "source_id": normalize_ws(item.get("ID")) or str(index),
                    "source_page_id": normalize_ws(item.get("fonte_page_id")),
                    "source_name": raw_name,
                    "source_text": raw_text,
                    "document": document,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                    "device_path": device_path,
                    "hierarchy": join_hierarchy(hierarchy_parts),
                    "text": text,
                }
            )

        count_by_norm_path = Counter(
            (json.loads(item["metadata_json"])["norma_id"], item["device_path"])
            for item in prepared
            if item["device_path"]
        )
        unique_actual_key: Dict[Tuple[str, str], str] = {}
        for item in prepared:
            metadata = json.loads(item["metadata_json"])
            key = (metadata["norma_id"], item["device_path"])
            if item["device_path"] and count_by_norm_path[key] == 1:
                unique_actual_key[key] = f"disp:{metadata['norma_id']}:{slugify(source_title, max_len=36)}:{item['source_id']}"

        required_device_structures: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in prepared:
            metadata = json.loads(item["metadata_json"])
            path = item["device_path"]
            if not path:
                continue
            parts = split_device_path(path)
            for depth in range(1, len(parts)):
                prefix = join_device_path(parts[:depth])
                key = (metadata["norma_id"], prefix)
                if key not in unique_actual_key:
                    required_device_structures[key] = {
                        "metadata": metadata,
                        "source": source,
                        "hierarchy": item["hierarchy"],
                    }
            own_key = (metadata["norma_id"], path)
            if count_by_norm_path[own_key] > 1:
                required_device_structures[own_key] = {
                    "metadata": metadata,
                    "source": source,
                    "hierarchy": item["hierarchy"],
                }

        for item in prepared:
            metadata = json.loads(item["metadata_json"])
            root_key = self.ensure_root(metadata, source)
            if item["hierarchy"]:
                self.ensure_structure(metadata=metadata, source=source, hierarchy_parts=split_hierarchy(item["hierarchy"]))
            self.rows_by_key[root_key]["fonte_database_titulo"] = "; ".join(sorted(self.sources_by_norma[metadata["norma_id"]]))
            self.refresh_row(root_key)

        def device_structure_sort(entry: Tuple[Tuple[str, str], Dict[str, Any]]) -> Tuple[str, int, str]:
            (norma_id, path), _info = entry
            return (norma_id, len(split_device_path(path)), path)

        for (norma_id, path), info in sorted(required_device_structures.items(), key=device_structure_sort):
            metadata = info["metadata"]
            hierarchy = info.get("hierarchy") or ""
            hierarchy_parent = self.ensure_structure(metadata=metadata, source=source, hierarchy_parts=split_hierarchy(hierarchy)) if hierarchy else self.ensure_root(metadata, source)
            parent_path = parent_device_path(path)
            parent_key = self.actual_by_norm_device_path.get((norma_id, parent_path)) or unique_actual_key.get((norma_id, parent_path)) or hierarchy_parent
            row_key = f"estrutura_disp:{norma_id}:{stable_key_slug(path)}"
            if row_key in self.rows_by_key:
                self.actual_by_norm_device_path[(norma_id, path)] = row_key
                continue
            parent_label = self.rows_by_key.get(parent_key, {}).get("dispositivo", "")
            row = base_row(
                row_key=row_key,
                titulo_linha=f"{metadata.get('norma_nome_popular')} - {path}",
                tipo_linha="estrutura",
                tipo_base="sintetico",
                metadata=metadata,
                linha_sintetica=True,
                source=source,
            )
            row.update(
                {
                    "tipo_dispositivo": infer_tipo_dispositivo(path),
                    "dispositivo": last_device_part(path),
                    "dispositivo_pai": parent_label,
                    "dispositivo_pai_row_key": parent_key,
                    "hierarquia_normativa": " > ".join(part for part in (hierarchy, path) if part),
                    "texto_dispositivo": path,
                    "texto_em_vigor": path,
                    "qualidade_texto": "ok",
                }
            )
            self.add_row(row)
            self.actual_by_norm_device_path[(norma_id, path)] = row_key

        for item in prepared:
            metadata = json.loads(item["metadata_json"])
            hierarchy = item["hierarchy"]
            hierarchy_parent = self.ensure_structure(metadata=metadata, source=source, hierarchy_parts=split_hierarchy(hierarchy)) if hierarchy else self.ensure_root(metadata, source)
            path = item["device_path"]
            source_id = item["source_id"]
            row_key = f"disp:{metadata['norma_id']}:{slugify(source_title, max_len=36)}:{source_id}"
            own_path_key = (metadata["norma_id"], path)
            if count_by_norm_path.get(own_path_key, 0) > 1:
                parent_key = self.actual_by_norm_device_path.get(own_path_key, hierarchy_parent)
            else:
                parent_path = parent_device_path(path)
                parent_key = (
                    self.actual_by_norm_device_path.get((metadata["norma_id"], parent_path))
                    or unique_actual_key.get((metadata["norma_id"], parent_path))
                    or hierarchy_parent
                )
            parent_label = self.rows_by_key.get(parent_key, {}).get("dispositivo", "")
            alert, quality = quality_alert(item["text"])
            row = base_row(
                row_key=row_key,
                titulo_linha=f"{metadata.get('norma_nome_popular') or metadata.get('norma_titulo')} - {path or source_id}",
                tipo_linha="dispositivo",
                tipo_base="matriz_notion",
                metadata=metadata,
                linha_sintetica=False,
                source=source,
            )
            row.update(
                {
                    "fonte_page_id": item["source_page_id"],
                    "id_fonte": source_id,
                    "ordem_doc": item["index"],
                    "tipo_dispositivo": infer_tipo_dispositivo(path),
                    "dispositivo": last_device_part(path) if path else source_id,
                    "dispositivo_pai": parent_label,
                    "dispositivo_pai_row_key": parent_key,
                    "hierarquia_normativa": " > ".join(part for part in (hierarchy, path) if part),
                    "documento_extraido": item["document"],
                    "texto_dispositivo": item["text"],
                    "texto_em_vigor": item["text"],
                    "palavras_chave": "; ".join(filter(None, [metadata.get("norma_nome_popular", ""), metadata.get("grupo", ""), path])),
                    "alerta_qualidade": alert,
                    "qualidade_texto": quality,
                }
            )
            if quality == "critico":
                row["incluir_no_rag"] = "false"
            self.add_row(row)
            if path and count_by_norm_path.get(own_path_key, 0) == 1:
                self.actual_by_norm_device_path[own_path_key] = row_key


def build_consolidated_rows(client: notion_import.NotionClient, sources: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    builder = ConsolidatedBuilder()
    for source in sources:
        rows = query_source_rows(client, source)
        LOGGER.info("Fonte %s: %s linhas.", source.get("title"), len(rows))
        builder.add_source(source, rows)
    for index, row in enumerate(builder.rows, start=1):
        row["id"] = str(index)
        row["ordem_global"] = str(index)
        builder.refresh_row(row["row_key"])
    return builder.rows


def write_consolidated_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_consolidated_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in CSV_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"{path} nao tem colunas esperadas: {', '.join(missing)}")
        return [{name: row.get(name, "") for name in CSV_COLUMNS} for row in reader]


def validate_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    row_keys = [normalize_ws(row.get("row_key")) for row in rows]
    row_key_set = set(row_keys)
    totals = {
        "rows": len(rows),
        "duplicate_row_keys": len(row_keys) - len(row_key_set),
        "normas": len({normalize_ws(row.get("norma_id")) for row in rows if normalize_ws(row.get("norma_id"))}),
        "source_databases": len({normalize_ws(row.get("fonte_database_id")) for row in rows if normalize_ws(row.get("fonte_database_id"))}),
        "missing_norma_nome_popular": 0,
        "missing_texto_rag": 0,
        "missing_parent_key": 0,
        "parent_key_not_found": 0,
        "root_rows": 0,
        "synthetic_rows": 0,
        "quality_ok": 0,
        "quality_verify": 0,
        "quality_critical": 0,
        "incluir_no_rag_false": 0,
    }
    for row in rows:
        if not normalize_ws(row.get("norma_nome_popular")):
            totals["missing_norma_nome_popular"] += 1
        if normalize_ws(row.get("incluir_no_rag")).lower() == "true" and not normalize_ws(row.get("texto_rag")):
            totals["missing_texto_rag"] += 1
        if normalize_ws(row.get("linha_sintetica")).lower() == "true":
            totals["synthetic_rows"] += 1
        if normalize_ws(row.get("incluir_no_rag")).lower() != "true":
            totals["incluir_no_rag_false"] += 1
        quality = normalize_ws(row.get("qualidade_texto"))
        if quality == "ok":
            totals["quality_ok"] += 1
        elif quality == "verificar":
            totals["quality_verify"] += 1
        elif quality == "critico":
            totals["quality_critical"] += 1
        parent_key = normalize_ws(row.get("dispositivo_pai_row_key"))
        if normalize_ws(row.get("tipo_linha")) == "norma":
            totals["root_rows"] += 1
            if parent_key:
                totals["missing_parent_key"] += 1
        elif not parent_key:
            totals["missing_parent_key"] += 1
        elif parent_key not in row_key_set:
            totals["parent_key_not_found"] += 1
    return totals


def schema_for_import() -> Dict[str, Any]:
    properties: Dict[str, Any] = {TITLE_PROPERTY: {"title": {}}}
    for column in CSV_COLUMNS:
        if column in NUMBER_COLUMNS:
            properties[column] = {"number": {"format": "number"}}
        elif column in CHECKBOX_COLUMNS:
            properties[column] = {"checkbox": {}}
        elif column in SELECT_COLUMNS:
            properties[column] = {"select": {"options": []}}
        else:
            properties[column] = {"rich_text": {}}
    return properties


def create_database(client: notion_import.NotionClient, parent_page_id: str, title: str) -> Tuple[str, str]:
    title_rich_text = [notion_import.text_object(title)]
    body = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": title_rich_text,
        "is_inline": True,
        "initial_data_source": {"title": title_rich_text, "properties": schema_for_import()},
    }
    payload = client.request("POST", "/v1/databases", json_body=body)
    database_id = notion_import.normalize_notion_id(str(payload.get("id", "")))
    data_source_id = notion_import.extract_data_source_id(client, database_id, payload)
    return database_id, data_source_id


def page_title(row: Mapping[str, str]) -> str:
    return normalize_ws(row.get("titulo_linha")) or normalize_ws(row.get("row_key")) or "Vademecum"


def build_page_properties(row: Mapping[str, str], *, max_rich_text_chars: int) -> Dict[str, Any]:
    props: Dict[str, Any] = {TITLE_PROPERTY: notion_import.title_property(page_title(row))}
    for column in CSV_COLUMNS:
        value = row.get(column, "")
        if column in NUMBER_COLUMNS:
            props[column] = notion_import.number_property(value)
        elif column in CHECKBOX_COLUMNS:
            props[column] = notion_import.checkbox_property(value)
        elif column in SELECT_COLUMNS:
            props[column] = notion_import.select_property(value)
        else:
            props[column] = notion_import.rich_text_property(value, max_chars=max_rich_text_chars)
    return props


def create_page(client: notion_import.NotionClient, data_source_id: str, properties: Mapping[str, Any]) -> str:
    payload = client.request(
        "POST",
        "/v1/pages",
        json_body={"parent": {"type": "data_source_id", "data_source_id": data_source_id}, "properties": dict(properties)},
    )
    return notion_import.normalize_notion_id(str(payload.get("id", "")))


def read_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_checkpoint(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sync_existing_pages_by_row_key(client: notion_import.NotionClient, data_source_id: str) -> Dict[str, str]:
    existing: Dict[str, str] = {}
    cursor = ""
    while True:
        body: Dict[str, Any] = {"page_size": 100, "result_type": "page"}
        if cursor:
            body["start_cursor"] = cursor
        payload = client.request("POST", f"/v1/data_sources/{data_source_id}/query", json_body=body)
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict) or item.get("object") != "page":
                continue
            props = item.get("properties") or {}
            row_key = normalize_ws(notion_import.property_plain_text(props.get("row_key", {}) or {}))
            page_id = normalize_ws(item.get("id"))
            if row_key and page_id:
                existing[row_key] = notion_import.normalize_notion_id(page_id)
        if not payload.get("has_more"):
            break
        cursor = normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return existing


def import_pages(
    client: notion_import.NotionClient,
    rows: Sequence[Mapping[str, str]],
    *,
    target_page_id: str,
    database_title: str,
    checkpoint_path: Path,
    workers: int,
    checkpoint_every: int,
    max_rich_text_chars: int,
) -> Dict[str, Any]:
    checkpoint = read_checkpoint(checkpoint_path)
    if not checkpoint:
        checkpoint = {
            "started_at": notion_import.now_iso(),
            "target_page_id": target_page_id,
            "database_title": database_title,
            "created_pages": {},
            "relation_updates": {},
        }
    created_pages = checkpoint.setdefault("created_pages", {})
    if "database_id" not in checkpoint or "data_source_id" not in checkpoint:
        LOGGER.info("Criando database consolidada '%s'...", database_title)
        database_id, data_source_id = create_database(client, target_page_id, database_title)
        checkpoint["database_id"] = database_id
        checkpoint["data_source_id"] = data_source_id
        checkpoint["database_url"] = notion_import.notion_url_from_id(database_id)
        checkpoint["created_at"] = notion_import.now_iso()
        write_checkpoint(checkpoint_path, checkpoint)
        LOGGER.info("Database criada: %s", checkpoint["database_url"])
    data_source_id = normalize_ws(checkpoint["data_source_id"])
    existing = sync_existing_pages_by_row_key(client, data_source_id)
    new_existing = {key: value for key, value in existing.items() if key not in created_pages}
    if new_existing:
        created_pages.update(new_existing)
        checkpoint["created_count"] = len(created_pages)
        checkpoint["updated_at"] = notion_import.now_iso()
        write_checkpoint(checkpoint_path, checkpoint)

    pending = [row for row in rows if normalize_ws(row.get("row_key")) not in created_pages]
    LOGGER.info("Paginas pendentes para criar: %s de %s.", len(pending), len(rows))
    submitted = 0
    imported = 0
    in_flight: Dict[Future[str], str] = {}
    max_in_flight = max(1, workers * 3)

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal submitted
        while submitted < len(pending) and len(in_flight) < max_in_flight:
            row = pending[submitted]
            submitted += 1
            row_key = normalize_ws(row.get("row_key"))
            props = build_page_properties(row, max_rich_text_chars=max_rich_text_chars)
            future = executor.submit(create_page, client, data_source_id, props)
            in_flight[future] = row_key

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_next(executor)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                row_key = in_flight.pop(future)
                try:
                    page_id = future.result()
                except Exception:
                    checkpoint["created_count"] = len(created_pages)
                    checkpoint["updated_at"] = notion_import.now_iso()
                    write_checkpoint(checkpoint_path, checkpoint)
                    raise
                created_pages[row_key] = page_id
                imported += 1
                if imported % checkpoint_every == 0 or imported == len(pending):
                    checkpoint["created_count"] = len(created_pages)
                    checkpoint["updated_at"] = notion_import.now_iso()
                    write_checkpoint(checkpoint_path, checkpoint)
                    LOGGER.info("Importacao: %s/%s paginas criadas/sincronizadas.", len(created_pages), len(rows))
            submit_next(executor)
    checkpoint["created_count"] = len(created_pages)
    checkpoint["rows_total"] = len(rows)
    checkpoint["updated_at"] = notion_import.now_iso()
    write_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def ensure_relation_schema(client: notion_import.NotionClient, data_source_id: str) -> None:
    payload = client.request("GET", f"/v1/data_sources/{data_source_id}")
    schema = payload.get("properties") or {}
    current = schema.get(RELATION_PROPERTY) if isinstance(schema, dict) else None
    if isinstance(current, dict) and current.get("type") == "relation":
        return
    LOGGER.info("Criando propriedade de relacao pai-filho '%s'.", RELATION_PROPERTY)
    client.request(
        "PATCH",
        f"/v1/data_sources/{data_source_id}",
        json_body={"properties": {RELATION_PROPERTY: {"relation": {"data_source_id": data_source_id, "single_property": {}}}}},
    )


def update_page_relation(client: notion_import.NotionClient, page_id: str, parent_page_id: str) -> None:
    client.request(
        "PATCH",
        f"/v1/pages/{page_id}",
        json_body={"properties": {RELATION_PROPERTY: {"relation": [{"id": parent_page_id}] if parent_page_id else []}}},
    )


def update_parent_relations(
    client: notion_import.NotionClient,
    rows: Sequence[Mapping[str, str]],
    *,
    checkpoint_path: Path,
    workers: int,
    checkpoint_every: int,
) -> Dict[str, int]:
    checkpoint = read_checkpoint(checkpoint_path)
    data_source_id = normalize_ws(checkpoint.get("data_source_id"))
    created_pages = checkpoint.get("created_pages") or {}
    relation_updates = checkpoint.setdefault("relation_updates", {})
    if not data_source_id or not isinstance(created_pages, dict):
        raise RuntimeError("Checkpoint sem data_source_id/created_pages para atualizar relacoes.")
    ensure_relation_schema(client, data_source_id)
    pending: List[Tuple[str, str, str, str]] = []
    missing_parent = 0
    for row in rows:
        row_key = normalize_ws(row.get("row_key"))
        parent_key = normalize_ws(row.get("dispositivo_pai_row_key"))
        if not parent_key:
            continue
        if relation_updates.get(row_key) == parent_key:
            continue
        page_id = created_pages.get(row_key, "")
        parent_page_id = created_pages.get(parent_key, "")
        if not page_id or not parent_page_id:
            missing_parent += 1
            continue
        pending.append((row_key, parent_key, page_id, parent_page_id))
    LOGGER.info("Relacoes pai-filho pendentes: %s; pais ausentes: %s.", len(pending), missing_parent)
    submitted = 0
    updated = 0
    in_flight: Dict[Future[None], Tuple[str, str]] = {}
    max_in_flight = max(1, workers * 3)
    lock = threading.Lock()

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal submitted
        while submitted < len(pending) and len(in_flight) < max_in_flight:
            row_key, parent_key, page_id, parent_page_id = pending[submitted]
            submitted += 1
            future = executor.submit(update_page_relation, client, page_id, parent_page_id)
            in_flight[future] = (row_key, parent_key)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_next(executor)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                row_key, parent_key = in_flight.pop(future)
                future.result()
                with lock:
                    relation_updates[row_key] = parent_key
                    updated += 1
                    if updated % checkpoint_every == 0 or updated == len(pending):
                        checkpoint["relation_updated_count"] = len(relation_updates)
                        checkpoint["updated_at"] = notion_import.now_iso()
                        write_checkpoint(checkpoint_path, checkpoint)
                        LOGGER.info("Relacoes atualizadas: %s/%s.", updated, len(pending))
            submit_next(executor)
    checkpoint["relation_updated_count"] = len(relation_updates)
    checkpoint["updated_at"] = notion_import.now_iso()
    write_checkpoint(checkpoint_path, checkpoint)
    return {"pending": len(pending), "updated": updated, "missing_parent": missing_parent, "total_marked": len(relation_updates)}


def validate_notion_import(client: notion_import.NotionClient, rows: Sequence[Mapping[str, str]], data_source_id: str) -> Dict[str, int]:
    pages_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pages_without_row_key = 0
    parent_key_pages = 0
    relation_pages = 0
    parent_no_relation = 0
    notion_pages = 0
    groups = sorted({normalize_ws(row.get("grupo")) for row in rows if normalize_ws(row.get("grupo"))})
    partitions: List[Optional[Dict[str, Any]]] = []
    if groups:
        partitions.extend({"property": "grupo", "select": {"equals": group}} for group in groups)
    else:
        partitions.append(None)

    for filter_body in partitions:
        cursor = ""
        while True:
            body: Dict[str, Any] = {"page_size": 100, "result_type": "page"}
            if filter_body:
                body["filter"] = filter_body
            if cursor:
                body["start_cursor"] = cursor
            payload = client.request("POST", f"/v1/data_sources/{data_source_id}/query", json_body=body)
            for item in payload.get("results", []) or []:
                if not isinstance(item, dict) or item.get("object") != "page":
                    continue
                notion_pages += 1
                props = item.get("properties") or {}
                row_key = normalize_ws(notion_import.property_plain_text(props.get("row_key", {}) or {}))
                parent_key = normalize_ws(notion_import.property_plain_text(props.get("dispositivo_pai_row_key", {}) or {}))
                relation_prop = props.get(RELATION_PROPERTY, {}) or {}
                relation_items = relation_prop.get("relation") if isinstance(relation_prop, dict) else []
                has_relation = bool(relation_items)
                if row_key:
                    pages_by_key[row_key].append(item)
                else:
                    pages_without_row_key += 1
                if parent_key:
                    parent_key_pages += 1
                    if has_relation:
                        relation_pages += 1
                    else:
                        parent_no_relation += 1
            if not payload.get("has_more"):
                break
            cursor = normalize_ws(payload.get("next_cursor"))
            if not cursor:
                break
    expected_keys = {normalize_ws(row.get("row_key")) for row in rows}
    notion_keys = set(pages_by_key)
    duplicate_pages = sum(max(0, len(items) - 1) for items in pages_by_key.values())
    return {
        "expected_rows": len(rows),
        "notion_pages": notion_pages,
        "notion_unique_row_keys": len(notion_keys),
        "missing_in_notion": len(expected_keys - notion_keys),
        "extra_row_keys_in_notion": len(notion_keys - expected_keys),
        "duplicate_row_key_pages": duplicate_pages,
        "pages_without_row_key": pages_without_row_key,
        "parent_key_pages": parent_key_pages,
        "relation_pages": relation_pages,
        "parent_no_relation": parent_no_relation,
    }


def cleanup_local_noise(paths: Sequence[Path]) -> List[str]:
    removed: List[str] = []
    for path in paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(str(path))
        except OSError as exc:
            LOGGER.warning("Nao foi possivel remover %s: %s", path, exc)
    return removed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolida bases Vademecum/RICD do Notion para uma base RAG estruturada.")
    parser.add_argument("--vademecum-page-url", default=DEFAULT_VADEMECUM_PAGE_URL)
    parser.add_argument("--ricd-database-url", default=DEFAULT_RICD_DATABASE_URL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--database-title", default=DEFAULT_DATABASE_TITLE)
    parser.add_argument("--from-csv", action="store_true")
    parser.add_argument("--csv-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-rps", type=float, default=2.7)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-rich-text-chars", type=int, default=30000)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    target_page_id = notion_import.extract_notion_id_from_url(args.vademecum_page_url)
    token = notion_import.resolve_notion_token()
    if not token and not args.dry_run:
        raise RuntimeError("Token Notion nao encontrado. Configure NOTION_API_KEY ou Chave_Notion.txt.")
    client = notion_import.NotionClient(
        token or "dry-run",
        notion_version=notion_import.DEFAULT_NOTION_VERSION,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        rate_rps=args.rate_rps,
    )

    if args.from_csv:
        rows = read_consolidated_csv(args.output_csv)
    else:
        sources = list_child_databases(client, target_page_id)
        ricd_source = get_database_source(client, args.ricd_database_url, source_kind="ricd_database")
        sources.append(ricd_source)
        LOGGER.info("Fontes Notion selecionadas: %s.", len(sources))
        rows = build_consolidated_rows(client, sources)
        write_consolidated_csv(args.output_csv, rows)
        LOGGER.info("CSV consolidado escrito em %s.", args.output_csv)

    validation = validate_rows(rows)
    LOGGER.info("Validacao local: %s", json.dumps(validation, ensure_ascii=False, sort_keys=True))
    blocking = [
        "duplicate_row_keys",
        "missing_norma_nome_popular",
        "missing_texto_rag",
        "missing_parent_key",
        "parent_key_not_found",
        "quality_critical",
    ]
    if any(validation[key] for key in blocking):
        raise RuntimeError("Validacao local encontrou pendencias bloqueantes; importacao interrompida.")
    if args.csv_only or args.dry_run:
        LOGGER.info("%s CSV consolidado: %s", "[dry-run]" if args.dry_run else "Gerado", args.output_csv)
        return 0

    checkpoint = import_pages(
        client,
        rows,
        target_page_id=target_page_id,
        database_title=args.database_title,
        checkpoint_path=args.checkpoint,
        workers=max(1, int(args.workers or 1)),
        checkpoint_every=max(1, int(args.checkpoint_every or 1)),
        max_rich_text_chars=max(2000, int(args.max_rich_text_chars or 0)),
    )
    relation_result = update_parent_relations(
        client,
        rows,
        checkpoint_path=args.checkpoint,
        workers=max(1, int(args.workers or 1)),
        checkpoint_every=max(1, int(args.checkpoint_every or 1)),
    )
    checkpoint = read_checkpoint(args.checkpoint)
    notion_validation = validate_notion_import(client, rows, normalize_ws(checkpoint["data_source_id"]))
    checkpoint["final_validation"] = notion_validation
    checkpoint["relation_result"] = relation_result
    checkpoint["completed_at"] = notion_import.now_iso()
    write_checkpoint(args.checkpoint, checkpoint)
    LOGGER.info("Validacao Notion: %s", json.dumps(notion_validation, ensure_ascii=False, sort_keys=True))
    LOGGER.info("Resultado das relacoes: %s", json.dumps(relation_result, ensure_ascii=False, sort_keys=True))
    if (
        notion_validation["missing_in_notion"]
        or notion_validation["extra_row_keys_in_notion"]
        or notion_validation["duplicate_row_key_pages"]
        or notion_validation["pages_without_row_key"]
        or notion_validation["parent_no_relation"]
        or relation_result["missing_parent"]
    ):
        raise RuntimeError("Validacao Notion encontrou pendencias.")
    if not args.no_cleanup:
        removed = cleanup_local_noise([Path("diagnostico_vademecum_notion.json"), Path("diagnostico_ricd_notion.json")])
        if removed:
            LOGGER.info("Artefatos locais de diagnostico removidos: %s", ", ".join(removed))
    LOGGER.info("Database consolidada: %s", checkpoint.get("database_url", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
