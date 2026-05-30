#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrai o Código Eleitoral Anotado 2026 em CSVs por norma.

O PDF do TSE traz informação tipográfica suficiente para separar texto legal,
hierarquia normativa e notas anotadas. Este conversor evita API/LLM e usa
PyMuPDF para preservar a relação entre cada nota e o dispositivo imediatamente
anterior.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import fitz  # type: ignore
except Exception:
    fitz = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_NAME = "Codigo_Eleitoral_2026_anotado.pdf"
DEFAULT_OUTPUT_DIR = "Codigo_Eleitoral_2026_csv_normas"
DOCUMENT_TITLE = "Código Eleitoral Anotado e Legislação Complementar 2026"

BLUE = 2301728
GREEN = 3953780
WHITE = 16777215

CSV_HEADER = [
    "id",
    "ordem_doc",
    "arquivo_fonte",
    "norma_id",
    "norma_titulo",
    "norma_nome_popular",
    "macrogrupo",
    "grupo_sumario",
    "pagina_inicial",
    "pagina_final",
    "hierarquia_normativa",
    "tipo_dispositivo",
    "dispositivo",
    "dispositivo_pai",
    "dispositivo_pai_id",
    "dispositivo_pai_row_key",
    "artigo",
    "paragrafo",
    "inciso",
    "alinea",
    "item",
    "texto_dispositivo",
    "notas_texto",
    "notas_json",
    "qtd_notas",
    "referencias_normativas",
    "referencias_jurisprudenciais",
    "referencias_sumulas",
    "texto_rag",
    "resumo_curto",
    "palavras_chave",
    "incluir_no_rag",
    "chars",
    "tokens_estimados",
    "row_key",
    "csv_arquivo",
]

SECTION_RANGES = [
    (25, "Código Eleitoral", "Código Eleitoral", "Código Eleitoral"),
    (151, "Constituição Federal", "Constituição Federal", "Constituição Federal"),
    (393, "Emendas Constitucionais", "Emendas Constitucionais", "Emendas Constitucionais"),
    (405, "Lei de Inelegibilidade", "Lei de Inelegibilidade", "Lei de Inelegibilidade"),
    (443, "Lei dos Partidos Políticos", "Lei dos Partidos Políticos", "Lei dos Partidos Políticos"),
    (487, "Lei das Eleições", "Lei das Eleições", "Lei das Eleições"),
    (609, "Lei n. 13.831/2019", "Lei n. 13.831/2019", "Lei n. 13.831/2019"),
    (613, "Lei n. 13.877/2019", "Lei n. 13.877/2019", "Lei n. 13.877/2019"),
    (617, "Legislação correlata", "Legislação correlata", "Legislação correlata"),
    (733, "Regimento Interno do TSE", "Regimento Interno do TSE", "Regimento Interno do TSE"),
    (769, "Normas editadas pelo TSE", "Normas editadas pelo TSE", "Normas editadas pelo TSE"),
    (771, "Normas editadas pelo TSE", "Resoluções", "Resoluções"),
    (1145, "Normas editadas pelo TSE", "Portarias", "Portarias"),
    (1197, "Normas editadas pelo TSE", "Instruções Normativas", "Instruções Normativas"),
    (1205, "Normas editadas pelo TSE", "Provimentos-CGE", "Provimentos-CGE"),
    (1303, "Normas editadas pelo TSE", "Protocolo de Cooperação Técnica", "Protocolo de Cooperação Técnica"),
    (1311, "Súmulas do TSE, do STF e do STJ", "Súmulas do TSE, do STF e do STJ", "Súmulas"),
    (1341, "Resolução do TCU", "Resolução do TCU", "Resolução do TCU"),
]

POPULAR_NAMES = {
    "lei_4737_1965": "Código Eleitoral",
    "constituicao_federal_1988": "Constituição Federal",
    "lei_complementar_64_1990": "Lei de Inelegibilidade",
    "lei_9096_1995": "Lei dos Partidos Políticos",
    "lei_9504_1997": "Lei das Eleições",
    "resolucao_4510_1952": "Regimento Interno do TSE",
}

GROUP_TITLE_EXCLUSIONS = {
    "Súmulas do TSE",
    "Súmulas do STF",
    "Súmulas do STJ",
}

NOTE_START_MARKERS = {"\x83": "quadrado", "9": "ticado"}

ARTICLE_RE = re.compile(
    r"^(Art\.\s*\d+(?:\.\d+)?(?:[ºª])?(?:-[A-Z])?\.?)\b",
    re.IGNORECASE,
)
PARAGRAPH_RE = re.compile(
    r"^((?:§\s*\d+(?:[ºª])?(?:-[A-Z])?\.?)|(?:Parágrafo\s+único\.?))\b",
    re.IGNORECASE,
)
INCISO_RE = re.compile(r"^([IVXLCDM]+)\s*[-–—]\s+", re.IGNORECASE)
ALINEA_RE = re.compile(r"^([a-z])\)\s+", re.IGNORECASE)
ITEM_RE = re.compile(r"^(\d+)[.)-]\s+")
SUMULA_RE = re.compile(r"^Súmula(?:\s+Vinculante)?-(?:TSE|STF|STJ)\s+n\.\s*\d+", re.IGNORECASE)
NORM_TITLE_RE = re.compile(
    r"^(?:"
    r"Constituição da República Federativa do Brasil"
    r"|Emenda Constitucional n\."
    r"|Lei(?: Complementar)? n\."
    r"|Decreto(?:-Lei)? n\."
    r"|Resolução(?:-TCU)? n\."
    r"|Portaria(?: Conjunta-[^ ]+)? n\."
    r"|Instrução Normativa(?: Conjunta-[^ ]+)? n\."
    r"|Provimento-CGE n\."
    r"|Protocolo de Cooperação Técnica n\."
    r")",
    re.IGNORECASE,
)
STRUCTURAL_LABEL_RE = re.compile(
    r"^(PARTE|LIVRO|TÍTULO|CAPÍTULO|SEÇÃO|SUBSEÇÃO)\b(?:\s+(.+))?$",
    re.IGNORECASE,
)

PT_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "lei",
    "n",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "que",
    "se",
    "sua",
    "suas",
    "um",
    "uma",
}


@dataclass(frozen=True)
class PdfLine:
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str
    color: int
    spans: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def column(self) -> int:
        return 0 if self.x0 < 228 else 1

    @property
    def order_key(self) -> Tuple[int, int, float, float]:
        return (self.page, self.column, self.y0, self.x0)


@dataclass
class NormStart:
    order_index: int
    start_key: Tuple[int, int, float, float]
    page: int
    title: str
    norm_id: str
    popular_name: str
    macrogroup: str
    summary_group: str
    title_keys: Tuple[Tuple[int, int, float, float], ...]
    is_sumula: bool = False


@dataclass
class Note:
    marker: str
    text_parts: List[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0

    def add_line(self, text: str, page: int) -> None:
        cleaned = clean_text_for_csv(strip_note_marker(text))
        if cleaned:
            self.text_parts.append(cleaned)
        if not self.page_start:
            self.page_start = page
        self.page_end = page

    @property
    def text(self) -> str:
        text = ""
        for part in self.text_parts:
            text = merge_text_fragments(text, part)
        return clean_text_for_csv(text)

    def as_dict(self, ordem: int) -> Dict[str, Any]:
        return {
            "ordem": ordem,
            "marcador": self.marker,
            "texto": self.text,
            "pagina_inicial": self.page_start,
            "pagina_final": self.page_end,
        }


@dataclass
class ParsedRow:
    norm: NormStart
    tipo_dispositivo: str
    dispositivo: str
    dispositivo_pai: str
    artigo: str
    paragrafo: str
    inciso: str
    alinea: str
    item: str
    hierarchy: str
    text_parts: List[str] = field(default_factory=list)
    notes: List[Note] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0

    def add_text(self, text: str, page: int) -> None:
        cleaned = clean_text_for_csv(text)
        if cleaned:
            self.text_parts.append(cleaned)
        self.touch_page(page)

    def touch_page(self, page: int) -> None:
        if not self.page_start:
            self.page_start = page
        self.page_end = page

    @property
    def texto_dispositivo(self) -> str:
        text = ""
        for part in self.text_parts:
            text = merge_text_fragments(text, part)
        return clean_text_for_csv(text)

    @property
    def notas_texto(self) -> str:
        return "\n".join(note.text for note in self.notes if note.text)


def ensure_pymupdf_available() -> None:
    if fitz is None:
        raise RuntimeError("PyMuPDF não está instalado. Instale com: python -m pip install pymupdf")


def normalize_space(text: str) -> str:
    text = (text or "").replace("\ufeff", "")
    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060]", "", text)
    return text


def clean_text_for_csv(text: str) -> str:
    text = normalize_space(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


def normalize_for_matching(text: str) -> str:
    text = clean_text_for_csv(text)
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents).strip().upper()


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower()
    ascii_text = ascii_text.replace("º", "").replace("ª", "")
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text)
    ascii_text = re.sub(r"_+", "_", ascii_text).strip("_")
    return ascii_text or "norma"


def merge_text_fragments(current: str, fragment: str) -> str:
    fragment = clean_text_for_csv(fragment)
    if not fragment:
        return current
    if not current:
        return fragment
    if current.endswith("-"):
        return current[:-1] + fragment
    return f"{current} {fragment}"


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text or "") / 4))


def build_short_summary(text: str, max_chars: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", clean_text_for_csv(text))
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(" .,;:") + "..."


def clean_text_for_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", clean_text_for_csv(text))
    text = re.sub(r"^Art\.\s*[\wº°.-]+(?:-[A-Z])?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Parágrafo único\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^§\s*\d+[º°]?(?:-[A-Z])?\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[IVXLCDM]+\s*[-–—]\s*", "", text)
    text = re.sub(r"^[a-z]\)\s*", "", text)
    text = re.sub(r"^\d+[.)]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def first_legal_sentence(text: str) -> str:
    cleaned = clean_text_for_summary(text)
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])", cleaned)
    return (parts[0] if parts else cleaned).strip()


def limit_text(text: str, max_chars: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", clean_text_for_csv(text))
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}..."


def build_keywords(*texts: str, max_terms: int = 10) -> str:
    words: List[str] = []
    for text in texts:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()
        words.extend(re.findall(r"[a-z][a-z0-9]{3,}", normalized))
    counts = Counter(word for word in words if word not in PT_STOPWORDS)
    return "; ".join(word for word, _ in counts.most_common(max_terms))


def section_for_page(page: int) -> Tuple[str, str, str]:
    current = SECTION_RANGES[0]
    for item in SECTION_RANGES:
        if page >= item[0]:
            current = item
        else:
            break
    _, macrogroup, summary_group, popular = current
    return macrogroup, summary_group, popular


def year_from_norm(norm: NormStart) -> str:
    candidates = [norm.title, norm.norm_id]
    found: List[str] = []
    for candidate in candidates:
        found.extend(re.findall(r"(?:19|20)\d{2}", candidate or ""))
    return found[-1] if found else ""


def canonical_norm_label(norm: NormStart) -> str:
    if norm.popular_name:
        return norm.popular_name
    title = clean_text_for_csv(norm.title)
    year = year_from_norm(norm)
    if norm.norm_id == "constituicao_federal_1988" or title.lower().startswith("constituição federal"):
        return "Constituição Federal"
    if title.startswith("Súmula"):
        return title
    match = re.match(r"^(?P<prefix>.+?\bn\.)\s*(?P<num>[\d.]+)(?:\s*,?\s+de\b.*)?$", title, flags=re.IGNORECASE)
    if match and year:
        prefix = clean_text_for_csv(match.group("prefix"))
        number = clean_text_for_csv(match.group("num"))
        prefix = re.sub(r"(?i)^Lei Complementar n\.$", "LC n.", prefix)
        prefix = re.sub(r"(?i)^Emenda Constitucional n\.$", "EC n.", prefix)
        return f"{prefix} {number}/{year}"
    return title


def year_from_portuguese_date(text: str) -> str:
    match = re.search(r"\b(?:de|em)\s+(\d{4})\b", text)
    if match:
        return match.group(1)
    match = re.search(r"/(\d{4})\b", text)
    return match.group(1) if match else ""


def build_norm_id(title: str, page: int) -> str:
    clean = re.sub(r"\s+", " ", title).strip()
    year = year_from_portuguese_date(clean)
    number_match = re.search(r"\bn\.\s*([\d.]+)", clean, re.IGNORECASE)
    number = number_match.group(1).replace(".", "") if number_match else ""
    normalized = normalize_for_matching(clean)

    if normalized.startswith("CONSTITUICAO DA REPUBLICA"):
        return "constituicao_federal_1988"
    if normalized.startswith("SUMULA VINCULANTE-STF"):
        number = number or re.search(r"(\d+)", clean).group(1)
        return f"sumula_vinculante_stf_{number}"
    if normalized.startswith("SUMULA-TSE"):
        number = number or re.search(r"(\d+)", clean).group(1)
        suffix = "_cancelada" if "CANCELADA" in normalized else ""
        return f"sumula_tse_{number}{suffix}"
    if normalized.startswith("SUMULA-STF"):
        number = number or re.search(r"(\d+)", clean).group(1)
        return f"sumula_stf_{number}"
    if normalized.startswith("SUMULA-STJ"):
        number = number or re.search(r"(\d+)", clean).group(1)
        return f"sumula_stj_{number}"

    if normalized.startswith("EMENDA CONSTITUCIONAL"):
        return f"emenda_constitucional_{number}_{year}".strip("_")
    if normalized.startswith("LEI COMPLEMENTAR"):
        return f"lei_complementar_{number}_{year}".strip("_")
    if normalized.startswith("LEI "):
        return f"lei_{number}_{year}".strip("_")
    if normalized.startswith("DECRETO-LEI"):
        return f"decreto_lei_{number}_{year}".strip("_")
    if normalized.startswith("DECRETO "):
        return f"decreto_{number}_{year}".strip("_")
    if normalized.startswith("RESOLUCAO-TCU"):
        return f"resolucao_tcu_{number}_{year}".strip("_")
    if normalized.startswith("RESOLUCAO"):
        return f"resolucao_{number}_{year}".strip("_")
    if normalized.startswith("PORTARIA CONJUNTA"):
        return f"portaria_conjunta_{number}_{year}".strip("_")
    if normalized.startswith("PORTARIA"):
        return f"portaria_{number}_{year}".strip("_")
    if normalized.startswith("INSTRUCAO NORMATIVA CONJUNTA"):
        return f"instrucao_normativa_conjunta_{number}_{year}".strip("_")
    if normalized.startswith("INSTRUCAO NORMATIVA"):
        return f"instrucao_normativa_{number}_{year}".strip("_")
    if normalized.startswith("PROVIMENTO-CGE"):
        return f"provimento_cge_{number}_{year}".strip("_")
    if normalized.startswith("PROTOCOLO DE COOPERACAO TECNICA"):
        return f"protocolo_cooperacao_tecnica_{number}_{year}".strip("_")
    return f"{slugify(clean)}_{page}"


def popular_name_for_norm(norm_id: str, page: int) -> str:
    if norm_id in POPULAR_NAMES:
        return POPULAR_NAMES[norm_id]
    _, _, popular = section_for_page(page)
    if popular in {"Legislação correlata", "Normas editadas pelo TSE", "Resoluções", "Portarias", "Instruções Normativas", "Provimentos-CGE", "Protocolo de Cooperação Técnica", "Súmulas", "Resolução do TCU"}:
        return ""
    return popular


def is_mergeable_title_continuation(first: str, second: str) -> bool:
    if first.startswith("Protocolo de Cooperação Técnica") and second.lower().startswith("de "):
        return True
    return False


def extract_lines_from_page(page: Any, page_number: int) -> List[PdfLine]:
    lines: List[PdfLine] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            raw_spans = [span for span in line.get("spans", []) if (span.get("text") or "").strip()]
            if not raw_spans:
                continue
            text = "".join(span.get("text", "") for span in raw_spans)
            text = clean_text_for_csv(text)
            if not text:
                continue
            x0 = min(float(span["bbox"][0]) for span in raw_spans)
            y0 = min(float(span["bbox"][1]) for span in raw_spans)
            x1 = max(float(span["bbox"][2]) for span in raw_spans)
            y1 = max(float(span["bbox"][3]) for span in raw_spans)
            first = raw_spans[0]
            lines.append(
                PdfLine(
                    page=page_number,
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    size=round(float(first.get("size", 0.0)), 1),
                    font=str(first.get("font", "")),
                    color=int(first.get("color", 0)),
                    spans=tuple(raw_spans),
                )
            )
    lines.sort(key=lambda item: item.order_key)
    return merge_split_article_prefixes(lines)


def merge_split_article_prefixes(lines: Sequence[PdfLine]) -> List[PdfLine]:
    merged: List[PdfLine] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if (
            current.text == "Art."
            and index + 1 < len(lines)
            and lines[index + 1].page == current.page
            and lines[index + 1].column == current.column
            and re.match(r"^\d", lines[index + 1].text)
            and abs(lines[index + 1].y0 - current.y0) < 18
        ):
            nxt = lines[index + 1]
            merged.append(
                PdfLine(
                    page=current.page,
                    text=f"{current.text} {nxt.text}",
                    x0=current.x0,
                    y0=current.y0,
                    x1=max(current.x1, nxt.x1),
                    y1=max(current.y1, nxt.y1),
                    size=current.size,
                    font=current.font,
                    color=current.color,
                    spans=current.spans + nxt.spans,
                )
            )
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def is_page_number(line: PdfLine) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", line.text)) and line.y0 > 600 and line.font == "Ubuntu"


def is_running_header(line: PdfLine) -> bool:
    if line.y0 < 38 and line.color == BLUE and line.font in {"Ubuntu-Light", "Ubuntu"}:
        return True
    if line.x0 > 418 and line.font == "Ubuntu-Light" and line.size <= 8.1:
        return True
    return False


def is_noise_line(line: PdfLine) -> bool:
    if is_page_number(line) or is_running_header(line):
        return True
    if line.text.strip() in {"", "\uf0a7"}:
        return True
    return False


def is_non_sumula_norm_title(line: PdfLine) -> bool:
    if line.font != "AGaramondPro-Regular" or line.size != 17.0 or line.color != GREEN or line.y0 > 95:
        return False
    if line.text in GROUP_TITLE_EXCLUSIONS:
        return False
    return bool(NORM_TITLE_RE.match(line.text))


def is_sumula_title(line: PdfLine) -> bool:
    return (
        line.font == "Ubuntu"
        and line.size == 9.0
        and line.color == GREEN
        and bool(SUMULA_RE.match(line.text))
    )


def detect_norm_starts(page_lines_by_page: Sequence[Sequence[PdfLine]]) -> List[NormStart]:
    starts: List[NormStart] = []
    for page_lines in page_lines_by_page:
        index = 0
        while index < len(page_lines):
            line = page_lines[index]
            if is_non_sumula_norm_title(line):
                title = line.text
                title_keys = [line.order_key]
                if index + 1 < len(page_lines):
                    nxt = page_lines[index + 1]
                    if (
                        nxt.font == line.font
                        and nxt.size == line.size
                        and nxt.color == line.color
                        and nxt.page == line.page
                        and is_mergeable_title_continuation(line.text, nxt.text)
                    ):
                        title = f"{title} {nxt.text}"
                        title_keys.append(nxt.order_key)
                        index += 1
                norm_id = build_norm_id(title, line.page)
                macro, summary_group, _ = section_for_page(line.page)
                starts.append(
                    NormStart(
                        order_index=0,
                        start_key=line.order_key,
                        page=line.page,
                        title=clean_text_for_csv(title),
                        norm_id=norm_id,
                        popular_name=popular_name_for_norm(norm_id, line.page),
                        macrogroup=macro,
                        summary_group=summary_group,
                        title_keys=tuple(title_keys),
                    )
                )
            elif line.page >= 1313 and is_sumula_title(line):
                title = clean_text_for_csv(line.text)
                norm_id = build_norm_id(title, line.page)
                macro, summary_group, _ = section_for_page(line.page)
                starts.append(
                    NormStart(
                        order_index=0,
                        start_key=line.order_key,
                        page=line.page,
                        title=title,
                        norm_id=norm_id,
                        popular_name="",
                        macrogroup=macro,
                        summary_group=summary_group,
                        title_keys=(line.order_key,),
                        is_sumula=True,
                    )
                )
            index += 1

    starts.sort(key=lambda item: item.start_key)
    unique: List[NormStart] = []
    seen: set[Tuple[str, Tuple[int, int, float, float]]] = set()
    for start in starts:
        key = (start.norm_id, start.start_key)
        if key in seen:
            continue
        seen.add(key)
        start.order_index = len(unique) + 1
        unique.append(start)
    return unique


def line_starts_note(line: PdfLine) -> Optional[str]:
    for span in line.spans:
        text = span.get("text", "").strip()
        if span.get("font") == "Wingdings-Regular" and text in NOTE_START_MARKERS:
            return NOTE_START_MARKERS[text]
    return None


def is_note_line(line: PdfLine) -> bool:
    if line_starts_note(line):
        return True
    if line.color == GREEN and line.size == 8.5 and line.font in {"MyriadPro-Light", "MyriadPro-LightIt"}:
        return True
    return False


def strip_note_marker(text: str) -> str:
    text = text.lstrip()
    text = re.sub(r"^[\x83]\s*", "", text)
    text = re.sub(r"^9(?=[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ0-9])", "", text)
    return text.strip()


def is_publication_line(line: PdfLine) -> bool:
    text = line.text.strip()
    if text.startswith("________"):
        return True
    if re.match(r"^Brasília,?\s+\d{1,2}\s+de\s+", text, re.IGNORECASE):
        return True
    if re.match(r"^(?:Publicad[ao]s?|Republicad[ao]s?)\b", text):
        return True
    return False


def is_structural_heading(line: PdfLine) -> bool:
    text = line.text.strip()
    if not text or is_sumula_title(line):
        return False
    if line.font != "Ubuntu" or line.color != GREEN or line.size not in {9.0, 10.0}:
        return False
    if ARTICLE_RE.match(text) or PARAGRAPH_RE.match(text) or INCISO_RE.match(text):
        return False
    normalized = normalize_for_matching(text)
    if STRUCTURAL_LABEL_RE.match(text):
        return True
    if len(text) <= 90 and normalized == normalized.upper() and any(ch.isalpha() for ch in text):
        return True
    return False


def structural_level(text: str) -> int:
    normalized = normalize_for_matching(text)
    if normalized.startswith("PARTE"):
        return 1
    if normalized.startswith("LIVRO"):
        return 2
    if normalized.startswith("TITULO"):
        return 3
    if normalized.startswith("CAPITULO"):
        return 4
    if normalized.startswith("SECAO"):
        return 5
    if normalized.startswith("SUBSECAO"):
        return 6
    return 7


def structural_label_waits_description(text: str) -> bool:
    match = STRUCTURAL_LABEL_RE.match(text.strip())
    if not match:
        return False
    rest = (match.group(2) or "").strip()
    return not rest or bool(re.fullmatch(r"(?:[IVXLCDM]+|\d+[ºª]?|ÚNICO|ÚNICA)", rest, re.IGNORECASE))


def classify_device(line: PdfLine) -> Optional[Tuple[str, str]]:
    text = line.text.strip()
    match = ARTICLE_RE.match(text)
    if match:
        return "artigo", normalize_device_label(match.group(1))
    match = PARAGRAPH_RE.match(text)
    if match:
        return "paragrafo", normalize_device_label(match.group(1))
    match = INCISO_RE.match(text)
    if match:
        return "inciso", match.group(1).upper()
    match = ALINEA_RE.match(text)
    if match:
        return "alinea", f"{match.group(1).lower()})"
    match = ITEM_RE.match(text)
    if match:
        return "item", match.group(1)
    return None


def normalize_device_label(label: str) -> str:
    label = clean_text_for_csv(label)
    label = re.sub(r"\s+", " ", label)
    label = label.replace("§ ", "§")
    label = re.sub(r"\.$", "", label)
    if label.lower().startswith("art."):
        label = re.sub(r"(?i)^art\.\s*", "Art. ", label)
    if label.lower().startswith("parágrafo"):
        label = "Parágrafo único"
    return label


def build_device_display(context: Dict[str, str], kind: str, label: str) -> Tuple[str, str]:
    if kind == "artigo":
        return label, ""

    parent_parts = [context.get("artigo", "")]
    if kind in {"inciso", "alinea", "item"} and context.get("paragrafo"):
        parent_parts.append(context["paragrafo"])
    if kind in {"alinea", "item"} and context.get("inciso"):
        parent_parts.append(context["inciso"])
    if kind == "item" and context.get("alinea"):
        parent_parts.append(context["alinea"])
    parent = ", ".join(part for part in parent_parts if part)
    display = ", ".join(part for part in [parent, label] if part)
    return display or label, parent


def references_from_text(text: str) -> Tuple[str, str, str]:
    normalized = clean_text_for_csv(text)
    normative_patterns = [
        r"CF/1988(?:,\s*art\.[^.;\n]*)?",
        r"CE/1965(?:,\s*art\.[^.;\n]*)?",
        r"ADCT(?:,\s*art\.[^.;\n]*)?",
        r"Lei(?: Complementar)? n\.\s*[\d.]+/?\d{0,4}",
        r"LC n\.\s*\d+/\d{4}",
        r"EC n\.\s*\d+/\d{4}",
        r"Res\.-TSE n\.\s*\d+/\d{4}",
        r"Resolução(?:-TSE|-TCU)? n\.\s*[\d.]+(?:/\d{4})?",
        r"Port\.-TSE n\.\s*\d+/\d{4}",
        r"Prov\.-CGE n\.\s*\d+/\d{4}",
        r"Decreto(?:-Lei)? n\.\s*[\d.]+/?\d{0,4}",
    ]
    jurisprudence_patterns = [
        r"Ac\.-(?:TSE|STF|STJ)[^.;\n]*",
        r"\b(?:ADI|ADC|ADPF|MS|REspe|AgR-REspe|REspEl|RO|PA|CtaEl|Rp|MC)\s+n\.\s*[\d.]+",
    ]
    sumula_patterns = [
        r"Súm\.-(?:TSE|STF|STJ)\s+n\.\s*[\d/,\s]+",
        r"Súmula(?: Vinculante)?-(?:TSE|STF|STJ)\s+n\.\s*\d+",
    ]
    return (
        "; ".join(unique_matches(normalized, normative_patterns)),
        extract_jurisprudence_references(normalized),
        "; ".join(unique_matches(normalized, sumula_patterns)),
    )


def extract_jurisprudence_references(text: str) -> str:
    normalized = clean_text_for_csv(text)
    if not normalized:
        return ""
    patterns = [
        r"Ac\.-(?:TSE|STF|STJ)\s+n\.\s*[\d./-]+",
        r"Ac\.-(?:TSE|STF|STJ)(?:,\s*de\s*\d{1,2}/\d{1,2}/\d{4})?,\s*(?:no|na|nos|nas)\s*[^:;.]{1,100}?\bn\.\s*[\d./-]+",
        r"\b(?:ADI|ADC|ADPF|ADO|ARE|HC|MS|MI|RE|REspe|AgR-REspe|REspEl|AgR-REspEl|RO|RMS|PA|CtaEl|Cta|Rp|Rcl|Pet|AI)\s+n\.\s*[\d./-]+",
        r"\bTema\s+n\.\s*\d+",
        r"\bTema\s+\d+\b",
        r"\bRepercussão Geral no RE\s+n\.\s*[\d./-]+",
    ]
    found = unique_matches(normalized, patterns)
    filtered: List[str] = []
    for value in found:
        value_key = value.lower()
        if any(value_key != other.lower() and value_key in other.lower() for other in found):
            continue
        filtered.append(value)
    return "; ".join(filtered)


def unique_matches(text: str, patterns: Sequence[str]) -> List[str]:
    found: List[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = clean_text_for_csv(match.group(0)).rstrip(" ,;:")
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                found.append(value)
    return found


def build_text_rag(row: ParsedRow) -> str:
    parts = [
        f"Documento: {DOCUMENT_TITLE}",
        f"Norma: {row.norm.title}",
    ]
    popular_name = canonical_norm_label(row.norm)
    if popular_name:
        parts.append(f"Nome popular/citação curta: {popular_name}")
    parts.extend(
        [
            f"Macrogrupo: {row.norm.macrogroup}",
            f"Grupo do sumário: {row.norm.summary_group}",
            f"Dispositivo: {row.dispositivo}",
            f"Tipo: {row.tipo_dispositivo}",
        ]
    )
    if row.hierarchy:
        parts.append(f"Hierarquia: {row.hierarchy}")
    if row.dispositivo_pai:
        parts.append(f"Dispositivo pai: {row.dispositivo_pai}")
    if row.page_start:
        parts.append(f"Páginas: {row.page_start}-{row.page_end or row.page_start}")
    if row.texto_dispositivo:
        parts.extend(["", row.texto_dispositivo])
    if row.notas_texto:
        parts.extend(["", "Notas vinculadas:", row.notas_texto])
    return clean_text_for_csv("\n".join(parts))


def build_row_summary(row: ParsedRow, refs_juris: str = "") -> str:
    norm_label = canonical_norm_label(row.norm)
    sentence = first_legal_sentence(row.texto_dispositivo) or first_legal_sentence(row.notas_texto)
    if row.tipo_dispositivo == "norma":
        prefix = f"Norma {norm_label}"
    elif row.tipo_dispositivo in {"estrutura", "artigo", "paragrafo", "inciso", "alinea", "item"} and row.dispositivo_pai:
        prefix = f"{row.dispositivo}, subordinado a {row.dispositivo_pai}"
    elif row.tipo_dispositivo == "estrutura":
        prefix = f"Estrutura da {norm_label}"
    elif row.tipo_dispositivo == "sumula":
        prefix = f"Enunciado {row.dispositivo}"
    else:
        prefix = row.dispositivo or norm_label

    if sentence and sentence.lower() != clean_text_for_csv(row.dispositivo).lower():
        summary = f"{prefix}: {sentence}"
    else:
        summary = prefix

    flags: List[str] = []
    combined = f"{row.texto_dispositivo} {row.notas_texto}".lower()
    if re.search(r"\b(revogado|revogada|cancelada|cancelado|vetado|vetada)\b", combined):
        flags.append("há marcação de revogação/cancelamento/veto")
    if row.notes:
        flags.append(f"{len(row.notes)} nota(s) vinculada(s)")
    if refs_juris:
        flags.append("jurisprudência extraída das notas")
    if flags:
        summary = f"{summary} | " + "; ".join(flags)
    return limit_text(summary)


def row_to_csv_dict(row: ParsedRow, row_id: int, source_name: str = DEFAULT_INPUT_NAME) -> Dict[str, Any]:
    notes_json = [note.as_dict(index) for index, note in enumerate(row.notes, start=1) if note.text]
    combined_text = "\n".join([row.texto_dispositivo, row.notas_texto])
    refs_norm, _, refs_sumulas = references_from_text(combined_text)
    refs_juris = extract_jurisprudence_references(row.notas_texto)
    popular_name = canonical_norm_label(row.norm)
    summary = build_row_summary(row, refs_juris)
    rag_parts = [build_text_rag(row)]
    if summary:
        rag_parts.extend(["", f"Resumo: {summary}"])
    if refs_juris:
        rag_parts.extend(["", f"Referências jurisprudenciais: {refs_juris}"])
    rag_text = clean_text_for_csv("\n".join(part for part in rag_parts if part))
    return {
        "id": row_id,
        "ordem_doc": row_id,
        "arquivo_fonte": source_name,
        "norma_id": row.norm.norm_id,
        "norma_titulo": row.norm.title,
        "norma_nome_popular": popular_name,
        "macrogrupo": row.norm.macrogroup,
        "grupo_sumario": row.norm.summary_group,
        "pagina_inicial": row.page_start,
        "pagina_final": row.page_end or row.page_start,
        "hierarquia_normativa": row.hierarchy,
        "tipo_dispositivo": row.tipo_dispositivo,
        "dispositivo": row.dispositivo,
        "dispositivo_pai": row.dispositivo_pai,
        "dispositivo_pai_id": "",
        "dispositivo_pai_row_key": "",
        "artigo": row.artigo,
        "paragrafo": row.paragrafo,
        "inciso": row.inciso,
        "alinea": row.alinea,
        "item": row.item,
        "texto_dispositivo": row.texto_dispositivo,
        "notas_texto": row.notas_texto,
        "notas_json": json.dumps(notes_json, ensure_ascii=False),
        "qtd_notas": len(notes_json),
        "referencias_normativas": refs_norm,
        "referencias_jurisprudenciais": refs_juris,
        "referencias_sumulas": refs_sumulas,
        "texto_rag": rag_text,
        "resumo_curto": summary,
        "palavras_chave": build_keywords(row.norm.title, row.hierarchy, row.dispositivo, combined_text),
        "incluir_no_rag": True,
        "chars": len(rag_text),
        "tokens_estimados": estimate_tokens(rag_text),
        "row_key": "",
        "csv_arquivo": "",
    }


def enrich_csv_rows_with_links(norm: NormStart, rows: List[Dict[str, Any]]) -> None:
    csv_name = filename_for_norm(norm)
    stem = Path(csv_name).stem
    for row in rows:
        row["csv_arquivo"] = csv_name
        row["row_key"] = f"{stem}:{row.get('id')}"

    rows_by_dispositivo = {
        clean_text_for_csv(str(row.get("dispositivo", ""))).lower(): row
        for row in rows
        if clean_text_for_csv(str(row.get("dispositivo", "")))
    }
    norm_row = next((row for row in rows if clean_text_for_csv(str(row.get("tipo_dispositivo"))) == "norma"), None)
    if norm_row is None:
        norm_row = next((row for row in rows if clean_text_for_csv(str(row.get("tipo_dispositivo"))) == "sumula"), None)
    structure_by_hierarchy: Dict[str, Dict[str, Any]] = {}

    def parent_hierarchy(hierarchy: str) -> str:
        parts = [part.strip() for part in clean_text_for_csv(hierarchy).split(" > ") if part.strip()]
        if len(parts) <= 1:
            return ""
        return " > ".join(parts[:-1])

    def assign_parent(row: Dict[str, Any], parent_row: Optional[Dict[str, Any]]) -> None:
        if parent_row and parent_row is not row:
            row["dispositivo_pai"] = parent_row.get("dispositivo", "")
            row["dispositivo_pai_id"] = parent_row.get("id", "")
            row["dispositivo_pai_row_key"] = parent_row.get("row_key", "")
        else:
            row["dispositivo_pai"] = ""
            row["dispositivo_pai_id"] = ""
            row["dispositivo_pai_row_key"] = ""

    def rebuild_summary(row: Dict[str, Any]) -> str:
        tipo = clean_text_for_csv(str(row.get("tipo_dispositivo", "")))
        dispositivo = clean_text_for_csv(str(row.get("dispositivo", "")))
        parent = clean_text_for_csv(str(row.get("dispositivo_pai", "")))
        sentence = first_legal_sentence(str(row.get("texto_dispositivo", ""))) or first_legal_sentence(str(row.get("notas_texto", "")))
        norm_label = clean_text_for_csv(str(row.get("norma_nome_popular", ""))) or canonical_norm_label(norm)
        if tipo == "norma":
            prefix = f"Norma {norm_label}"
        elif tipo in {"estrutura", "artigo", "paragrafo", "inciso", "alinea", "item"} and parent:
            prefix = f"{dispositivo}, subordinado a {parent}"
        elif tipo == "estrutura":
            prefix = f"Estrutura da {norm_label}"
        elif tipo == "sumula":
            prefix = f"Enunciado {dispositivo}"
        else:
            prefix = dispositivo or norm_label
        summary = f"{prefix}: {sentence}" if sentence and sentence.lower() != dispositivo.lower() else prefix
        flags: List[str] = []
        combined = f"{row.get('texto_dispositivo', '')} {row.get('notas_texto', '')}".lower()
        if re.search(r"\b(revogado|revogada|cancelada|cancelado|vetado|vetada)\b", combined):
            flags.append("há marcação de revogação/cancelamento/veto")
        try:
            notes_count = int(float(str(row.get("qtd_notas") or "0")))
        except ValueError:
            notes_count = 0
        if notes_count:
            flags.append(f"{notes_count} nota(s) vinculada(s)")
        if clean_text_for_csv(str(row.get("referencias_jurisprudenciais", ""))):
            flags.append("jurisprudência extraída das notas")
        if flags:
            summary = f"{summary} | " + "; ".join(flags)
        return limit_text(summary)

    for row in rows:
        tipo = clean_text_for_csv(str(row.get("tipo_dispositivo")))
        parent_row: Optional[Dict[str, Any]] = None
        if tipo == "estrutura":
            hierarchy = clean_text_for_csv(str(row.get("hierarquia_normativa", "")))
            parent_row = structure_by_hierarchy.get(parent_hierarchy(hierarchy)) or norm_row
        elif tipo == "artigo":
            hierarchy = clean_text_for_csv(str(row.get("hierarquia_normativa", "")))
            parent_row = structure_by_hierarchy.get(hierarchy) or norm_row
        elif tipo == "paragrafo" and not clean_text_for_csv(str(row.get("dispositivo_pai", ""))):
            parent_row = rows_by_dispositivo.get(clean_text_for_csv(str(row.get("artigo", ""))).lower()) or norm_row
        elif tipo in {"paragrafo", "inciso", "alinea", "item"}:
            parent = clean_text_for_csv(str(row.get("dispositivo_pai", ""))).lower()
            parent_row = rows_by_dispositivo.get(parent) if parent else None
        assign_parent(row, parent_row)
        if tipo == "estrutura":
            hierarchy = clean_text_for_csv(str(row.get("hierarquia_normativa", "")))
            if hierarchy:
                structure_by_hierarchy[hierarchy] = row
        row["resumo_curto"] = rebuild_summary(row)
        if row.get("texto_rag") and row.get("dispositivo_pai_row_key"):
            extra_parts: List[str] = []
            if "Dispositivo pai:" not in str(row["texto_rag"]):
                extra_parts.append(f"Dispositivo pai: {row['dispositivo_pai']}")
            if "ID do dispositivo pai:" not in str(row["texto_rag"]):
                extra_parts.append(f"ID do dispositivo pai: {row['dispositivo_pai_row_key']}")
            if extra_parts:
                row["texto_rag"] = clean_text_for_csv(f"{row['texto_rag']}\n" + "\n".join(extra_parts))
            row["chars"] = len(str(row["texto_rag"]))
            row["tokens_estimados"] = estimate_tokens(str(row["texto_rag"]))


def create_row(
    norm: NormStart,
    tipo: str,
    dispositivo: str,
    parent: str,
    context: Dict[str, str],
    hierarchy: str,
) -> ParsedRow:
    return ParsedRow(
        norm=norm,
        tipo_dispositivo=tipo,
        dispositivo=dispositivo,
        dispositivo_pai=parent,
        artigo=context.get("artigo", ""),
        paragrafo=context.get("paragrafo", ""),
        inciso=context.get("inciso", ""),
        alinea=context.get("alinea", ""),
        item=context.get("item", ""),
        hierarchy=hierarchy,
    )


def parse_norm_lines(norm: NormStart, lines: Sequence[PdfLine]) -> List[ParsedRow]:
    rows: List[ParsedRow] = []
    context = {"artigo": "", "paragrafo": "", "inciso": "", "alinea": "", "item": ""}
    hierarchy_stack: Dict[int, str] = {}
    pending_structure: Optional[Tuple[int, str]] = None
    current_row: Optional[ParsedRow] = None
    last_anchor: Optional[ParsedRow] = None
    current_note: Optional[Note] = None
    norm_row: Optional[ParsedRow] = None
    in_footer = False
    title_keys = set(norm.title_keys)

    def current_hierarchy() -> str:
        return " > ".join(hierarchy_stack[level] for level in sorted(hierarchy_stack) if hierarchy_stack[level])

    def ensure_norm_row() -> ParsedRow:
        nonlocal norm_row, current_row, last_anchor
        if norm_row is None:
            empty_context = {"artigo": "", "paragrafo": "", "inciso": "", "alinea": "", "item": ""}
            norm_row = create_row(norm, "norma", norm.title, "", empty_context, current_hierarchy())
            rows.append(norm_row)
        current_row = norm_row
        last_anchor = norm_row
        return norm_row

    if norm.is_sumula:
        current_row = create_row(norm, "sumula", norm.title, "", context, current_hierarchy())
        rows.append(current_row)
        last_anchor = current_row

    for line in lines:
        if line.order_key in title_keys or is_noise_line(line):
            continue

        if line.page == norm.page and line.font == "AGaramondPro-Regular" and line.size == 17.0:
            continue

        marker = line_starts_note(line)
        if is_note_line(line):
            anchor = last_anchor or ensure_norm_row()
            if marker or current_note is None or current_note not in anchor.notes:
                current_note = Note(marker=marker or "continuação", page_start=line.page, page_end=line.page)
                anchor.notes.append(current_note)
            current_note.add_line(line.text, line.page)
            anchor.touch_page(line.page)
            continue

        current_note = None

        structural_heading = is_structural_heading(line)
        device = classify_device(line)
        if in_footer and (structural_heading or device):
            in_footer = False
        elif is_publication_line(line) or in_footer:
            in_footer = True
            continue

        if structural_heading:
            level = structural_level(line.text)
            label = line.text
            if pending_structure and not STRUCTURAL_LABEL_RE.match(line.text):
                pending_level, pending_label = pending_structure
                level = pending_level
                label = f"{pending_label} - {line.text}"
                pending_structure = None
                if rows and rows[-1].tipo_dispositivo == "estrutura" and rows[-1].dispositivo == pending_label:
                    rows.pop()
            else:
                pending_structure = (level, line.text) if structural_label_waits_description(line.text) else None

            for lower in list(hierarchy_stack):
                if lower >= level:
                    del hierarchy_stack[lower]
            hierarchy_stack[level] = label
            current_row = create_row(norm, "estrutura", label, "", context, current_hierarchy())
            current_row.add_text(label, line.page)
            rows.append(current_row)
            last_anchor = current_row
            continue

        pending_structure = None
        if device:
            kind, label = device
            if kind == "artigo":
                context = {"artigo": label, "paragrafo": "", "inciso": "", "alinea": "", "item": ""}
            elif kind == "paragrafo":
                context["paragrafo"] = label
                context["inciso"] = ""
                context["alinea"] = ""
                context["item"] = ""
            elif kind == "inciso":
                context["inciso"] = label
                context["alinea"] = ""
                context["item"] = ""
            elif kind == "alinea":
                context["alinea"] = label
                context["item"] = ""
            elif kind == "item":
                context["item"] = label

            display, parent = build_device_display(context, kind, label)
            current_row = create_row(norm, kind, display, parent, context, current_hierarchy())
            current_row.add_text(line.text, line.page)
            rows.append(current_row)
            last_anchor = current_row
            in_footer = False
            continue

        if current_row is None:
            current_row = ensure_norm_row()
        current_row.add_text(line.text, line.page)
        last_anchor = current_row

    return [row for row in rows if row.texto_dispositivo or row.notes]


def flatten_page_lines(page_lines_by_page: Sequence[Sequence[PdfLine]]) -> List[PdfLine]:
    lines = [line for page_lines in page_lines_by_page for line in page_lines]
    lines.sort(key=lambda item: item.order_key)
    return lines


def lines_for_norm(norm: NormStart, next_norm: Optional[NormStart], all_lines: Sequence[PdfLine]) -> List[PdfLine]:
    end_key = next_norm.start_key if next_norm else (10**9, 0, 0.0, 0.0)
    return [line for line in all_lines if norm.start_key <= line.order_key < end_key]


def extract_document_rows_by_norm(input_file: str) -> Tuple[List[NormStart], Dict[str, List[Dict[str, Any]]]]:
    ensure_pymupdf_available()
    source_name = Path(input_file).name
    with fitz.open(input_file) as document:
        page_lines_by_page = [
            extract_lines_from_page(document[page_index], page_index + 1)
            for page_index in range(document.page_count)
        ]

    starts = detect_norm_starts(page_lines_by_page)
    all_lines = flatten_page_lines(page_lines_by_page)
    rows_by_norm: Dict[str, List[Dict[str, Any]]] = {}

    for index, norm in enumerate(starts):
        next_norm = starts[index + 1] if index + 1 < len(starts) else None
        parsed_rows = parse_norm_lines(norm, lines_for_norm(norm, next_norm, all_lines))
        csv_rows = [
            row_to_csv_dict(row, row_index, source_name=source_name)
            for row_index, row in enumerate(parsed_rows, start=1)
        ]
        if csv_rows:
            enrich_csv_rows_with_links(norm, csv_rows)
            rows_by_norm[norm.norm_id] = csv_rows

    return starts, rows_by_norm


def filename_for_norm(norm: NormStart) -> str:
    suffix = slugify(norm.popular_name) if norm.popular_name else ""
    base = norm.norm_id
    if suffix and suffix not in base:
        base = f"{base}_{suffix}"
    return f"{norm.order_index:03d}_{base}.csv"


def write_rows_to_csv(output_file: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_HEADER})


def convert_pdf_to_csvs(input_file: str, output_dir: str) -> List[Path]:
    input_path = Path(input_file).expanduser()
    if not input_path.is_absolute():
        input_path = SCRIPT_DIR / input_path
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    starts, rows_by_norm = extract_document_rows_by_norm(str(input_path))
    written: List[Path] = []
    for norm in starts:
        rows = rows_by_norm.get(norm.norm_id, [])
        if not rows:
            continue
        output_file = output_path / filename_for_norm(norm)
        write_rows_to_csv(output_file, rows)
        written.append(output_file)
    return written


def resolve_default_input() -> str:
    candidate = SCRIPT_DIR / DEFAULT_INPUT_NAME
    return str(candidate.resolve())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Converte o Código Eleitoral Anotado 2026 em CSVs por norma.")
    parser.add_argument("--input-file", default=resolve_default_input(), help="PDF de entrada.")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / DEFAULT_OUTPUT_DIR), help="Diretório dos CSVs de saída.")
    parser.add_argument("--no-gui", action="store_true", help="Mantido para compatibilidade; este conversor usa CLI.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    written = convert_pdf_to_csvs(args.input_file, args.output_dir)
    print(f"[SUCESSO] {len(written)} CSVs escritos em: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
