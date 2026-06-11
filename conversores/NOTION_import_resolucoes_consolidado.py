#!/usr/bin/env python3
from __future__ import annotations

# Script movido da raiz do projeto para conversores/. O bloco abaixo mantem
# imports (Artefatos.*, gui_intuitiva e scripts irmaos) e caminhos relativos
# ancorados na raiz do projeto.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "conversores")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_PROJECT_ROOT)

import argparse
import csv
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


DEFAULT_SOURCE_DIR = Path("/mnt/c/Users/mauri/resolucoes")
DEFAULT_TARGET_PAGE_URL = "https://www.notion.so/resolu-es-2f9721955c64804c9ef5ff8fc2ac9967"
DEFAULT_OUTPUT_CSV = Path("resolucoes_rag_consolidado.csv")
DEFAULT_CHECKPOINT = Path("Artefatos/checkpoints/resolucoes_rag_consolidado_checkpoint.json")
DEFAULT_DATABASE_TITLE = "Resolucoes TSE - RAG consolidado"
RELATION_PROPERTY = "dispositivo_pai_relacao"

LOGGER = logging.getLogger("notion_resolucoes_consolidado")

RESOLUTION_INFO = {
    "23.605": ("2019", "FEFC", "Res.-TSE n. 23.605/2019 - FEFC"),
    "23.607": ("2019", "Prestacao de contas", "Res.-TSE n. 23.607/2019 - Prestacao de contas"),
    "23.608": (
        "2019",
        "Representacoes, reclamacoes e direito de resposta",
        "Res.-TSE n. 23.608/2019 - Representacoes, reclamacoes e direito de resposta",
    ),
    "23.609": ("2019", "Registro de candidatura", "Res.-TSE n. 23.609/2019 - Registro de candidatura"),
    "23.610": ("2019", "Propaganda eleitoral", "Res.-TSE n. 23.610/2019 - Propaganda eleitoral"),
    "23.735": ("2024", "Ilicitos eleitorais", "Res.-TSE n. 23.735/2024 - Ilicitos eleitorais"),
    "23.444": ("2015", "Teste Publico de Seguranca (TPS)", "Resolucao-TSE n. 23.444/2015 - Teste Publico de Seguranca (TPS)"),
    "23.673": (
        "2021",
        "Fiscalizacao e auditoria do sistema eletronico de votacao",
        "Resolucao-TSE n. 23.673/2021 - Fiscalizacao e auditoria do sistema eletronico de votacao",
    ),
    "23.677": (
        "2021",
        "Sistemas eleitorais majoritario e proporcional",
        "Resolucao-TSE n. 23.677/2021 - Sistemas eleitorais majoritario e proporcional",
    ),
    "23.750": (
        "2026",
        "Cronograma operacional do Cadastro Eleitoral 2026",
        "Resolucao-TSE n. 23.750/2026 - Cronograma operacional do Cadastro Eleitoral para as Eleicoes 2026",
    ),
    "23.751": (
        "2026",
        "Atos gerais do processo eleitoral 2026",
        "Resolucao-TSE n. 23.751/2026 - Atos gerais do processo eleitoral para as Eleicoes 2026",
    ),
    "23.760": ("2026", "Calendario Eleitoral 2026", "Resolucao-TSE n. 23.760/2026 - Calendario Eleitoral 2026"),
}

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
    "grupo_resolucao",
    "fonte_csv",
    "fonte_pdf",
    "fonte_database_titulo",
    "fonte_database_id",
    "id_fonte",
    "ordem_doc",
    "tipo_dispositivo",
    "dispositivo",
    "dispositivo_pai",
    "dispositivo_pai_row_key",
    "hierarquia_normativa",
    "texto_dispositivo",
    "texto_em_vigor",
    "classe_vigencia_texto",
    "usar_como_texto_vigente",
    "alerta_vigencia",
    "prioridade_rag",
    "texto_redacao_anterior",
    "texto_redacao_proposta",
    "texto_redacao_consolidada_proposta",
    "texto_marcado",
    "alteracoes_json",
    "status_alteracao",
    "tipo_operacao",
    "ordem_alteracao",
    "texto_fragmento_anterior",
    "texto_fragmento_novo",
    "resolucao_alterada",
    "resolucao_alteradora",
    "notas_texto",
    "notas_json",
    "qtd_notas",
    "referencias_normativas",
    "referencias_jurisprudenciais",
    "resumo_curto",
    "palavras_chave",
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
    "ordem_alteracao",
    "qtd_notas",
    "chars",
    "tokens_estimados",
}
CHECKBOX_COLUMNS = {"linha_sintetica", "incluir_no_rag", "usar_como_texto_vigente"}
SELECT_COLUMNS = {
    "tipo_linha",
    "tipo_base",
    "tipo_dispositivo",
    "status_alteracao",
    "tipo_operacao",
    "grupo_resolucao",
    "classe_vigencia_texto",
    "prioridade_rag",
}
TITLE_PROPERTY = "Nome"


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
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return normalize_ws(pieces[0] if pieces else text)


def extract_resolution_number(text: Any) -> str:
    match = re.search(r"\b(\d{2}\.\d{3})\b", str(text or ""))
    return match.group(1) if match else ""


def extract_year(text: Any) -> str:
    years = re.findall(r"\b(?:19|20)\d{2}\b", str(text or ""))
    return years[-1] if years else ""


def norm_id(number: str, year: str) -> str:
    compact = re.sub(r"\D+", "", number)
    return f"res_tse_{compact}_{year}" if year else f"res_tse_{compact}"


def split_path(path: str) -> List[str]:
    return [normalize_ws(part) for part in normalize_ws(path).split(",") if normalize_ws(part)]


def join_path(parts: Sequence[str]) -> str:
    return ", ".join(part for part in parts if part)


def parent_path(path: str) -> str:
    parts = split_path(path)
    return join_path(parts[:-1]) if len(parts) > 1 else ""


def last_path_part(path: str) -> str:
    parts = split_path(path)
    return parts[-1] if parts else normalize_ws(path)


def infer_tipo_dispositivo(path: str, source_tipo: str = "") -> str:
    source_tipo = normalize_ws(source_tipo).lower()
    if source_tipo:
        return source_tipo
    part = last_path_part(path)
    lower = ascii_key(part)
    full = ascii_key(path)
    if lower.startswith("art."):
        return "artigo"
    if "paragrafo unico" in lower or lower.startswith("§"):
        return "paragrafo"
    if re.fullmatch(r"[ivxlcdm]+", part.strip(), flags=re.IGNORECASE):
        return "inciso"
    if re.fullmatch(r"[a-z]\)?", part.strip(), flags=re.IGNORECASE):
        return "alinea"
    if full.startswith("anexo"):
        return "item_anexo" if "," in path else "anexo"
    if re.fullmatch(r"\d+(?:\.\d+)*", part.strip()):
        return "item"
    return "dispositivo"


def discover_source_csvs(source_dir: Path) -> List[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Pasta fonte nao encontrada: {source_dir}")
    selected: List[Path] = []
    for csv_path in sorted(source_dir.glob("*.csv")):
        key = ascii_key(csv_path.name)
        if "_old" in key or key.startswith("pool_"):
            continue
        if key.startswith("estruturado ia -"):
            selected.append(csv_path)
            continue
        if key.startswith("resolucao no") or key.startswith("resolucao n "):
            selected.append(csv_path)
    if not selected:
        raise FileNotFoundError(f"Nenhum CSV matriz encontrado em {source_dir}")
    return selected


def source_kind(csv_path: Path) -> str:
    key = ascii_key(csv_path.name)
    if key.startswith("estruturado ia -") and " granular" in key:
        return "comparativo_granular"
    if key.startswith("estruturado ia -"):
        return "comparativo_base"
    return "operacional"


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def metadata_for_source(csv_path: Path, row: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    row = row or {}
    number = normalize_ws(row.get("resolucao_alterada")) or extract_resolution_number(csv_path.name)
    if re.fullmatch(r"\d+(?:\.\d+)?", number) and "." not in number and len(number) == 5:
        number = f"{number[:2]}.{number[2:]}"
    year = extract_year(csv_path.name)
    known = RESOLUTION_INFO.get(number)
    if known:
        known_year, popular, title = known
        year = year or known_year
    else:
        popular = csv_path.stem
        title = f"Resolucao-TSE n. {number}/{year}" if number and year else csv_path.stem
    kind = source_kind(csv_path)
    group = "comparativos_de_alteracao" if kind.startswith("comparativo") else "textos_operacionais"
    return {
        "norma_numero": number,
        "norma_ano": year,
        "norma_id": norm_id(number, year),
        "norma_nome_popular": popular,
        "norma_titulo": title,
        "grupo_resolucao": group,
    }


def extract_normative_refs(*texts: str) -> str:
    text = normalize_ws(" ".join(t for t in texts if t))
    if not text:
        return ""
    patterns = [
        r"\b(?:Lei(?: Complementar)?|LC|Res(?:\.|olucao)?(?:-TSE)?|Emenda Constitucional|EC|Decreto|Portaria)\s+n[.oº]*\s*[\d.]+(?:/\d{4})?",
        r"\bRes\.-TSE\s+n[.oº]*\s*[\d.]+(?:/\d{4})?",
        r"\bart[.]?\s*\d+[A-Z]?(?:-[A-Z])?(?:[oº])?(?:,\s*§\s*\d+[oº]?)?",
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
        r"Ac\.-(?:TSE|STF|STJ)(?:,\s*de\s*\d{1,2}/\d{1,2}/\d{4})?,\s*(?:no|na|nos|nas)\s*[^:;.]{1,120}?\bn[.oº]*\s*[\d./-]+",
        r"\b(?:ADI|ADC|ADPF|ADO|ARE|HC|MS|MI|RE|REspe|AgR-REspe|REspEl|AgR-REspEl|RO|RMS|PA|CtaEl|Cta|Rp|Rcl|Pet|AI)\s+n[.oº]*\s*[\d./-]+",
        r"\bTema\s+n[.oº]*\s*\d+",
        r"\bTema\s+\d+\b",
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
            if len(found) >= 40:
                return limit_text("; ".join(found), 1800)
    return limit_text("; ".join(found), 1800)


def build_text_rag(row: Mapping[str, str]) -> str:
    labels = [
        ("Base", "Resolucoes TSE"),
        ("Norma", row.get("norma_titulo", "")),
        ("Nome popular", row.get("norma_nome_popular", "")),
        ("Grupo", row.get("grupo_resolucao", "")),
        ("Tipo da linha", row.get("tipo_linha", "")),
        ("Tipo do dispositivo", row.get("tipo_dispositivo", "")),
        ("Dispositivo", row.get("dispositivo", "")),
        ("Hierarquia", row.get("hierarquia_normativa", "")),
        ("Dispositivo pai", row.get("dispositivo_pai", "")),
        ("Row key do pai", row.get("dispositivo_pai_row_key", "")),
        ("Fonte CSV", row.get("fonte_csv", "")),
    ]
    parts = [f"{label}: {normalize_ws(value)}" for label, value in labels if normalize_ws(value)]
    for label, column in [
        ("Resumo", "resumo_curto"),
        ("Referencias normativas", "referencias_normativas"),
        ("Referencias jurisprudenciais", "referencias_jurisprudenciais"),
        ("Texto do dispositivo", "texto_dispositivo"),
        ("Texto vigente", "texto_vigente"),
        ("Texto proposto", "texto_proposto"),
        ("Texto consolidado", "texto_consolidado"),
        ("Texto marcado", "texto_marcado"),
        ("Texto anterior", "texto_anterior"),
        ("Texto novo", "texto_novo"),
        ("Alteracoes JSON", "alteracoes_json"),
        ("Notas", "notas_texto"),
    ]:
        value = normalize_text_block(row.get(column, ""))
        if value:
            parts.extend(["", f"{label}:", value])
    return "\n".join(parts).strip()


def make_summary(row: Mapping[str, str]) -> str:
    tipo_linha = normalize_ws(row.get("tipo_linha"))
    dispositivo = normalize_ws(row.get("dispositivo"))
    popular = normalize_ws(row.get("norma_nome_popular"))
    if tipo_linha == "norma":
        return limit_text(f"Norma consolidada: {normalize_ws(row.get('norma_titulo')) or popular}")
    if tipo_linha == "estrutura":
        return limit_text(f"Estrutura de navegacao em {popular}: {normalize_ws(row.get('hierarquia_normativa'))}")
    if tipo_linha == "alteracao_granular":
        operation = normalize_ws(row.get("tipo_operacao")) or "alteracao"
        sentence = first_sentence(row.get("texto_novo", "") or row.get("texto_dispositivo", ""))
        return limit_text(f"Alteracao granular ({operation}) em {dispositivo}: {sentence}")
    if tipo_linha == "alteracao":
        status = normalize_ws(row.get("status_alteracao")) or "alterado"
        sentence = first_sentence(row.get("texto_consolidado", "") or row.get("texto_proposto", "") or row.get("texto_vigente", ""))
        return limit_text(f"Dispositivo {status} em {popular}, {dispositivo}: {sentence}")
    sentence = first_sentence(row.get("texto_dispositivo", ""))
    return limit_text(f"{popular}, {dispositivo}: {sentence}" if sentence else f"{popular}, {dispositivo}")


def base_row(
    *,
    row_key: str,
    titulo_linha: str,
    tipo_linha: str,
    tipo_base: str,
    metadata: Mapping[str, str],
    linha_sintetica: bool,
    fonte_csv: str = "",
    fonte_pdf: str = "",
    fonte_database_titulo: str = "",
    fonte_database_id: str = "",
) -> Dict[str, str]:
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
            "grupo_resolucao": metadata.get("grupo_resolucao", ""),
            "fonte_csv": fonte_csv,
            "fonte_pdf": fonte_pdf,
            "fonte_database_titulo": fonte_database_titulo,
            "fonte_database_id": fonte_database_id,
            "qtd_notas": "0",
        }
    )
    return row


class ConsolidatedBuilder:
    def __init__(self, source_db_map: Mapping[str, Mapping[str, str]]) -> None:
        self.source_db_map = source_db_map
        self.rows: List[Dict[str, str]] = []
        self.rows_by_key: Dict[str, Dict[str, str]] = {}
        self.root_by_norma: Dict[str, str] = {}
        self.root_sources: Dict[str, set[str]] = defaultdict(set)
        self.base_key_by_norm_path: Dict[Tuple[str, str], str] = {}

    def source_info(self, csv_path: Path) -> Dict[str, str]:
        info = self.source_db_map.get(csv_path.name, {})
        return {
            "fonte_database_titulo": info.get("title", ""),
            "fonte_database_id": info.get("database_id", ""),
        }

    def add_row(self, row: Dict[str, str]) -> str:
        row_key = normalize_ws(row.get("row_key"))
        if not row_key:
            raise RuntimeError("Linha sem row_key.")
        if row_key in self.rows_by_key:
            raise RuntimeError(f"row_key duplicado: {row_key}")
        row["resumo_curto"] = make_summary(row)
        row["referencias_normativas"] = extract_normative_refs(
            row.get("texto_dispositivo", ""),
            row.get("texto_vigente", ""),
            row.get("texto_proposto", ""),
            row.get("texto_consolidado", ""),
            row.get("alteracoes_json", ""),
        )
        row["referencias_jurisprudenciais"] = extract_juris_refs(
            row.get("texto_dispositivo", ""),
            row.get("texto_vigente", ""),
            row.get("texto_proposto", ""),
            row.get("texto_consolidado", ""),
            row.get("notas_texto", ""),
        )
        row["texto_rag"] = build_text_rag(row)
        row["chars"] = str(len(row["texto_rag"]))
        row["tokens_estimados"] = str(estimate_tokens(row["texto_rag"]))
        self.rows_by_key[row_key] = row
        self.rows.append(row)
        return row_key

    def ensure_root(self, metadata: Mapping[str, str], source_csv: str = "") -> str:
        norma_id = metadata.get("norma_id", "")
        if not norma_id:
            raise RuntimeError(f"Metadata sem norma_id: {metadata}")
        if source_csv:
            self.root_sources[norma_id].add(source_csv)
        existing = self.root_by_norma.get(norma_id)
        if existing:
            return existing
        row_key = f"norma:{norma_id}"
        title = metadata.get("norma_titulo", "") or metadata.get("norma_nome_popular", "") or norma_id
        row = base_row(
            row_key=row_key,
            titulo_linha=title,
            tipo_linha="norma",
            tipo_base="sintetico",
            metadata=metadata,
            linha_sintetica=True,
        )
        row["tipo_dispositivo"] = "norma"
        row["dispositivo"] = title
        row["hierarquia_normativa"] = title
        row["texto_dispositivo"] = title
        self.add_row(row)
        self.root_by_norma[norma_id] = row_key
        return row_key

    def refresh_root_sources(self) -> None:
        for norma_id, sources in self.root_sources.items():
            root_key = self.root_by_norma.get(norma_id)
            if root_key and root_key in self.rows_by_key:
                row = self.rows_by_key[root_key]
                row["fonte_csv"] = "; ".join(sorted(sources))
                row["texto_rag"] = build_text_rag(row)
                row["chars"] = str(len(row["texto_rag"]))
                row["tokens_estimados"] = str(estimate_tokens(row["texto_rag"]))

    def add_operational_source(self, csv_path: Path) -> None:
        rows = read_csv_rows(csv_path)
        metadata = metadata_for_source(csv_path)
        root_key = self.ensure_root(metadata, csv_path.name)
        source_info = self.source_info(csv_path)
        path_counts = Counter(normalize_ws(row.get("Nome (Artigo)")) for row in rows if normalize_ws(row.get("Nome (Artigo)")))
        required_structures: set[str] = set()
        paths_seen: Dict[str, int] = {}
        for item in rows:
            path = normalize_ws(item.get("Nome (Artigo)"))
            if not path:
                continue
            paths_seen.setdefault(path, len(paths_seen))
            parts = split_path(path)
            for idx in range(1, len(parts)):
                prefix = join_path(parts[:idx])
                if path_counts.get(prefix, 0) != 1:
                    required_structures.add(prefix)
            if path_counts[path] > 1:
                required_structures.add(path)

        structure_order = sorted(required_structures, key=lambda value: (len(split_path(value)), paths_seen.get(value, 10**9), value))
        path_anchor: Dict[str, str] = {}
        for path, count in path_counts.items():
            if count == 1 and path not in required_structures:
                matching = next(row for row in rows if normalize_ws(row.get("Nome (Artigo)")) == path)
                path_anchor[path] = f"op:{metadata['norma_id']}:{slugify(csv_path.stem, max_len=48)}:{normalize_ws(matching.get('ID'))}"
        for path in structure_order:
            row_key = f"estrutura:{metadata['norma_id']}:{slugify(path)}"
            path_anchor[path] = row_key
            parent = parent_path(path)
            parent_key = path_anchor.get(parent, root_key) if parent else root_key
            parent_label = self.rows_by_key.get(parent_key, {}).get("dispositivo", "")
            row = base_row(
                row_key=row_key,
                titulo_linha=f"{metadata['norma_nome_popular']} - {path}",
                tipo_linha="estrutura",
                tipo_base="sintetico",
                metadata=metadata,
                linha_sintetica=True,
                fonte_csv=csv_path.name,
                **source_info,
            )
            row.update(
                {
                    "ordem_doc": "0",
                    "tipo_dispositivo": "estrutura",
                    "dispositivo": last_path_part(path),
                    "dispositivo_pai": parent_label,
                    "dispositivo_pai_row_key": parent_key,
                    "hierarquia_normativa": path,
                    "texto_dispositivo": path,
                }
            )
            self.add_row(row)

        for index, item in enumerate(rows, start=1):
            path = normalize_ws(item.get("Nome (Artigo)"))
            text = normalize_text_block(item.get("Texto do Artigo"))
            if not path and not text:
                continue
            source_id = normalize_ws(item.get("ID")) or str(index)
            row_key = f"op:{metadata['norma_id']}:{slugify(csv_path.stem, max_len=48)}:{source_id}"
            if path_counts.get(path, 0) > 1:
                parent_key = path_anchor.get(path, root_key)
            else:
                parent = parent_path(path)
                parent_key = path_anchor.get(parent, root_key) if parent else root_key
            parent_label = self.rows_by_key.get(parent_key, {}).get("dispositivo", "")
            row = base_row(
                row_key=row_key,
                titulo_linha=f"{metadata['norma_nome_popular']} - {path or source_id}",
                tipo_linha="dispositivo",
                tipo_base="operacional",
                metadata=metadata,
                linha_sintetica=False,
                fonte_csv=csv_path.name,
                **source_info,
            )
            row.update(
                {
                    "id_fonte": source_id,
                    "ordem_doc": str(index),
                    "tipo_dispositivo": infer_tipo_dispositivo(path),
                    "dispositivo": last_path_part(path) if path else source_id,
                    "dispositivo_pai": parent_label,
                    "dispositivo_pai_row_key": parent_key,
                    "hierarquia_normativa": path,
                    "texto_dispositivo": text,
                    "palavras_chave": "; ".join(filter(None, [metadata["norma_nome_popular"], path])),
                }
            )
            self.add_row(row)

    def add_comparative_base_source(self, csv_path: Path) -> None:
        rows = read_csv_rows(csv_path)
        source_info = self.source_info(csv_path)
        for index, item in enumerate(rows, start=1):
            metadata = metadata_for_source(csv_path, item)
            root_key = self.ensure_root(metadata, csv_path.name)
            source_id = normalize_ws(item.get("id")) or str(index)
            path = normalize_ws(item.get("caminho_dispositivo")) or normalize_ws(item.get("artigo_base")) or normalize_ws(item.get("nome_dispositivo"))
            text_vigente = normalize_text_block(item.get("texto_vigente"))
            text_proposto = normalize_text_block(item.get("texto_proposto"))
            text_consolidado = normalize_text_block(item.get("texto_consolidado"))
            row_key = f"comp_base:{metadata['norma_id']}:{source_id}"
            row = base_row(
                row_key=row_key,
                titulo_linha=f"{metadata['norma_nome_popular']} - {path or source_id} - alteracao",
                tipo_linha="alteracao",
                tipo_base="comparativo_base",
                metadata=metadata,
                linha_sintetica=False,
                fonte_csv=csv_path.name,
                fonte_pdf=normalize_ws(item.get("fonte_pdf")),
                **source_info,
            )
            row.update(
                {
                    "id_fonte": source_id,
                    "ordem_doc": str(index),
                    "tipo_dispositivo": infer_tipo_dispositivo(path, item.get("tipo_dispositivo", "")),
                    "dispositivo": path or normalize_ws(item.get("nome_dispositivo")) or source_id,
                    "dispositivo_pai": self.rows_by_key[root_key]["dispositivo"],
                    "dispositivo_pai_row_key": root_key,
                    "hierarquia_normativa": path,
                    "texto_dispositivo": text_consolidado or text_proposto or text_vigente,
                    "texto_vigente": text_vigente,
                    "texto_proposto": text_proposto,
                    "texto_consolidado": text_consolidado,
                    "texto_marcado": normalize_text_block(item.get("texto_proposto_marcado")),
                    "alteracoes_json": normalize_text_block(item.get("alteracoes_json")),
                    "status_alteracao": normalize_ws(item.get("status_alteracao")),
                    "resolucao_alterada": normalize_ws(item.get("resolucao_alterada")),
                    "resolucao_alteradora": normalize_ws(item.get("resolucao_alteradora")),
                    "palavras_chave": "; ".join(filter(None, [metadata["norma_nome_popular"], path, normalize_ws(item.get("status_alteracao"))])),
                }
            )
            self.add_row(row)
            if path:
                self.base_key_by_norm_path[(metadata["norma_id"], path)] = row_key

    def add_comparative_granular_source(self, csv_path: Path) -> None:
        rows = read_csv_rows(csv_path)
        source_info = self.source_info(csv_path)
        for index, item in enumerate(rows, start=1):
            metadata = metadata_for_source(csv_path, item)
            root_key = self.ensure_root(metadata, csv_path.name)
            source_id = normalize_ws(item.get("id")) or str(index)
            path = normalize_ws(item.get("caminho_dispositivo")) or normalize_ws(item.get("artigo_base")) or normalize_ws(item.get("nome_dispositivo"))
            parent_key = self.base_key_by_norm_path.get((metadata["norma_id"], path), root_key)
            parent_label = self.rows_by_key.get(parent_key, {}).get("dispositivo", "")
            text_dispositivo = normalize_text_block(item.get("texto_consolidado_dispositivo") or item.get("texto_proposto_dispositivo") or item.get("texto_vigente_dispositivo"))
            operation = normalize_ws(item.get("tipo_operacao"))
            row_key = f"comp_gran:{metadata['norma_id']}:{source_id}"
            row = base_row(
                row_key=row_key,
                titulo_linha=f"{metadata['norma_nome_popular']} - {path or source_id} - {operation or 'alteracao'} {source_id}",
                tipo_linha="alteracao_granular",
                tipo_base="comparativo_granular",
                metadata=metadata,
                linha_sintetica=False,
                fonte_csv=csv_path.name,
                fonte_pdf=normalize_ws(item.get("fonte_pdf")),
                **source_info,
            )
            row.update(
                {
                    "id_fonte": source_id,
                    "ordem_doc": str(index),
                    "tipo_dispositivo": infer_tipo_dispositivo(path, item.get("tipo_dispositivo", "")),
                    "dispositivo": path or normalize_ws(item.get("nome_dispositivo")) or source_id,
                    "dispositivo_pai": parent_label,
                    "dispositivo_pai_row_key": parent_key,
                    "hierarquia_normativa": path,
                    "texto_dispositivo": text_dispositivo,
                    "texto_vigente": normalize_text_block(item.get("texto_vigente_dispositivo")),
                    "texto_proposto": normalize_text_block(item.get("texto_proposto_dispositivo")),
                    "texto_consolidado": normalize_text_block(item.get("texto_consolidado_dispositivo")),
                    "texto_marcado": normalize_text_block(item.get("texto_novo")),
                    "alteracoes_json": json.dumps(
                        {
                            "texto_anterior": normalize_text_block(item.get("texto_anterior")),
                            "texto_novo": normalize_text_block(item.get("texto_novo")),
                        },
                        ensure_ascii=False,
                    ),
                    "status_alteracao": normalize_ws(item.get("status_alteracao")),
                    "tipo_operacao": operation,
                    "ordem_alteracao": normalize_ws(item.get("ordem_alteracao")),
                    "texto_anterior": normalize_text_block(item.get("texto_anterior")),
                    "texto_novo": normalize_text_block(item.get("texto_novo")),
                    "resolucao_alterada": normalize_ws(item.get("resolucao_alterada")),
                    "resolucao_alteradora": normalize_ws(item.get("resolucao_alteradora")),
                    "palavras_chave": "; ".join(filter(None, [metadata["norma_nome_popular"], path, operation])),
                }
            )
            self.add_row(row)


def build_consolidated_rows(source_paths: Sequence[Path], source_db_map: Mapping[str, Mapping[str, str]]) -> List[Dict[str, str]]:
    builder = ConsolidatedBuilder(source_db_map)
    comparative_base = [path for path in source_paths if source_kind(path) == "comparativo_base"]
    comparative_granular = [path for path in source_paths if source_kind(path) == "comparativo_granular"]
    operational = [path for path in source_paths if source_kind(path) == "operacional"]
    for path in comparative_base:
        builder.add_comparative_base_source(path)
    for path in comparative_granular:
        builder.add_comparative_granular_source(path)
    for path in operational:
        builder.add_operational_source(path)
    builder.refresh_root_sources()
    for index, row in enumerate(builder.rows, start=1):
        row["id"] = str(index)
        row["ordem_global"] = str(index)
        row["texto_rag"] = build_text_rag(row)
        row["chars"] = str(len(row["texto_rag"]))
        row["tokens_estimados"] = str(estimate_tokens(row["texto_rag"]))
    from NOTION_repair_resolucoes_vigencia import repair_row

    repaired_rows = [repair_row(row) for row in builder.rows]
    for index, row in enumerate(repaired_rows, start=1):
        row["id"] = str(index)
        row["ordem_global"] = str(index)
    return repaired_rows


def write_consolidated_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    row_keys = [normalize_ws(row.get("row_key")) for row in rows]
    row_key_set = set(row_keys)
    source_names: set[str] = set()
    for row in rows:
        for item in normalize_ws(row.get("fonte_csv")).split(";"):
            item = normalize_ws(item)
            if item:
                source_names.add(item)
    totals = {
        "rows": len(rows),
        "duplicate_row_keys": len(row_keys) - len(row_key_set),
        "missing_norma_nome_popular": 0,
        "missing_texto_rag": 0,
        "missing_parent_key": 0,
        "parent_key_not_found": 0,
        "root_rows": 0,
        "synthetic_rows": 0,
        "source_csvs": len(source_names),
    }
    for row in rows:
        if not normalize_ws(row.get("norma_nome_popular")):
            totals["missing_norma_nome_popular"] += 1
        if normalize_ws(row.get("incluir_no_rag")).lower() == "true" and not normalize_ws(row.get("texto_rag")):
            totals["missing_texto_rag"] += 1
        if normalize_ws(row.get("linha_sintetica")).lower() == "true":
            totals["synthetic_rows"] += 1
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
        "initial_data_source": {
            "title": title_rich_text,
            "properties": schema_for_import(),
        },
    }
    payload = client.request("POST", "/v1/databases", json_body=body)
    database_id = notion_import.normalize_notion_id(str(payload.get("id", "")))
    data_source_id = notion_import.extract_data_source_id(client, database_id, payload)
    return database_id, data_source_id


def page_title(row: Mapping[str, str]) -> str:
    return normalize_ws(row.get("titulo_linha")) or normalize_ws(row.get("row_key")) or "Resolucao"


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
            title = normalize_ws(child.get("title"))
            database_id = notion_import.normalize_notion_id(str(item.get("id", "")))
            results.append({"title": title, "database_id": database_id})
        if not payload.get("has_more"):
            break
        cursor = normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return results


def match_source_databases(source_paths: Sequence[Path], child_databases: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, str]]:
    by_exact = {ascii_key(item.get("title")): dict(item) for item in child_databases}
    result: Dict[str, Dict[str, str]] = {}
    for csv_path in source_paths:
        exact = by_exact.get(ascii_key(csv_path.name))
        if exact:
            result[csv_path.name] = exact
            continue
        number = extract_resolution_number(csv_path.name)
        candidates = [dict(item) for item in child_databases if number and number in normalize_ws(item.get("title"))]
        if len(candidates) == 1:
            result[csv_path.name] = candidates[0]
    return result


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
                page_id = future.result()
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
    cursor = ""
    while True:
        body: Dict[str, Any] = {"page_size": 100, "result_type": "page"}
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
    parser = argparse.ArgumentParser(description="Consolida CSVs de resolucoes eleitorais e importa uma base RAG limpa no Notion.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-page-url", default=DEFAULT_TARGET_PAGE_URL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--database-title", default=DEFAULT_DATABASE_TITLE)
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
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    source_paths = discover_source_csvs(args.source_dir)
    target_page_id = notion_import.extract_notion_id_from_url(args.target_page_url)
    token = notion_import.resolve_notion_token()
    if not token and not args.csv_only:
        raise RuntimeError("Token Notion nao encontrado. Configure NOTION_API_KEY ou Chave_Notion.txt.")
    client: Optional[notion_import.NotionClient] = None
    source_db_map: Dict[str, Dict[str, str]] = {}
    if token:
        client = notion_import.NotionClient(
            token,
            notion_version=notion_import.DEFAULT_NOTION_VERSION,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            rate_rps=args.rate_rps,
        )
        child_databases = list_child_databases(client, target_page_id)
        source_db_map = match_source_databases(source_paths, child_databases)
        LOGGER.info("Bases-fonte encontradas no Notion: %s/%s.", len(source_db_map), len(source_paths))

    rows = build_consolidated_rows(source_paths, source_db_map)
    write_consolidated_csv(args.output_csv, rows)
    validation = validate_rows(rows)
    LOGGER.info("Validacao local: %s", json.dumps(validation, ensure_ascii=False, sort_keys=True))
    if validation["duplicate_row_keys"] or validation["missing_norma_nome_popular"] or validation["missing_texto_rag"] or validation["missing_parent_key"] or validation["parent_key_not_found"]:
        raise RuntimeError("Validacao local encontrou pendencias; importacao interrompida.")

    if args.csv_only or args.dry_run:
        LOGGER.info("%s CSV consolidado: %s", "[dry-run]" if args.dry_run else "Gerado", args.output_csv)
        return 0
    if client is None:
        raise RuntimeError("Cliente Notion nao inicializado.")

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
        removed = cleanup_local_noise([Path("diagnostico_resolucoes_notion.json"), Path("diagnostico_resolucoes_notion_schema.json")])
        if removed:
            LOGGER.info("Artefatos locais de diagnostico removidos: %s", ", ".join(removed))
    LOGGER.info("Database consolidada: %s", checkpoint.get("database_url", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
