#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converte o relatório do IDP em CSV amigável para Notion/RAG.

Fluxo:
1. Localiza o PDF do relatório ou recebe um caminho explícito via CLI/GUI.
2. Extrai linhas, fontes e coordenadas com PyMuPDF.
3. Detecta `PARTE`, seções numeradas, autores e blocos tabulares da Parte III.
4. Gera chunks narrativos e linhas por recomendação da tabela.
5. Escreve um CSV rico em metadados para importação no Notion.
"""

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
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gui_intuitiva import open_file_panel

try:
    import fitz  # type: ignore
except Exception:
    fitz = None


# O script residia na raiz do projeto; SCRIPT_DIR continua apontando para ela
# (base para PDFs de entrada, CSV de saida e caminhos relativos).
SCRIPT_DIR = _PROJECT_ROOT
DEFAULT_INPUT_NAME = (
    "IDP-LIA-Relatorio-de-Pesquisa-Integridade-da-informacao-nas-eleicoes-e-"
    "plataformas-digitais-caminhos-para-a-corregulacao-2026-2.pdf"
)
DEFAULT_OUTPUT_NAME = "IDP_relatorio_rag.csv"
DOCUMENT_TITLE = (
    "Integridade da informação nas eleições e plataformas digitais: caminhos para a corregulação"
)
HEADER_BANNER = (
    "INTEGRIDADE DA INFORMAÇÃO NAS ELEIÇÕES E PLATAFORMAS DIGITAIS: CAMINHOS PARA A CORREGULAÇÃO"
)
CSV_HEADER = [
    "id",
    "ordem_doc",
    "arquivo_fonte",
    "titulo_documento",
    "parte",
    "secao",
    "subsecao",
    "caminho_hierarquico",
    "tipo_chunk",
    "autores",
    "pagina_inicial",
    "pagina_final",
    "bloco_tabela",
    "subtema_tabela",
    "fonte_parte_ii",
    "titulo_chunk",
    "texto_limpo",
    "texto_rag",
    "resumo_curto",
    "palavras_chave",
    "incluir_no_rag",
    "chars",
    "tokens_estimados",
]
MIN_FONT_SIZE = 9.0
PART_RE = re.compile(r"^PARTE\s+([IVXLC]+)\s*[-–]\s*(.+)$", re.IGNORECASE)
NUMBERED_HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)+\.?)\s+(?P<title>.+)$")
SUBTHEME_RE = re.compile(r"^\d+(?:\.\d+)+\.?\s+")
BLOCK_RE = re.compile(r"^(Bloco\s+\d+):\s*(.+)$", re.IGNORECASE)
REFERENCE_RE = re.compile(r"^REFERÊNCIAS$", re.IGNORECASE)
PAGE_ONLY_RE = re.compile(r"^\d{1,3}$")
URL_RE = re.compile(r"https?://\S+")

PT_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "aquilo",
    "as",
    "até",
    "com",
    "como",
    "da",
    "das",
    "de",
    "dela",
    "dele",
    "deles",
    "demais",
    "depois",
    "do",
    "dos",
    "e",
    "ela",
    "elas",
    "ele",
    "eles",
    "em",
    "entre",
    "era",
    "essa",
    "essas",
    "esse",
    "esses",
    "esta",
    "estas",
    "este",
    "estes",
    "eu",
    "foi",
    "foram",
    "há",
    "isso",
    "isto",
    "já",
    "la",
    "lhe",
    "lhes",
    "mais",
    "mas",
    "me",
    "mesmo",
    "meu",
    "meus",
    "minha",
    "minhas",
    "na",
    "nas",
    "nem",
    "no",
    "nos",
    "nós",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "qual",
    "quando",
    "que",
    "quem",
    "se",
    "sem",
    "ser",
    "seu",
    "seus",
    "só",
    "sua",
    "suas",
    "também",
    "te",
    "tem",
    "tendo",
    "ter",
    "teve",
    "tu",
    "um",
    "uma",
    "vários",
    "várias",
}
KEYWORD_PRIORITY_TERMS = {
    "algoritmica",
    "algoritmico",
    "algoritmos",
    "anuncios",
    "chatbots",
    "conteudo",
    "corregulacao",
    "dados",
    "deepfakes",
    "desinformacao",
    "eleicoes",
    "eleitoral",
    "integridade",
    "inteligencia",
    "moderação",
    "moderacao",
    "plataformas",
    "politicas",
    "propaganda",
    "protecao",
    "riscos",
    "transparencia",
}


@dataclass(frozen=True)
class PageLine:
    page_number: int
    text: str
    x0: float
    y0: float
    size: float
    font: str
    is_bold: bool
    is_italic: bool


@dataclass
class SourceRecord:
    part: str
    section: str
    subsection: str
    path: str
    chunk_type: str
    authors: str
    page_start: int
    page_end: int
    text: str
    block_label: str = ""
    block_display: str = ""
    block_intro: str = ""
    subtheme: str = ""
    source_part_ii: str = ""
    order_hint: Tuple[int, float, int] = (0, 0.0, 0)


@dataclass
class TableLayout:
    left_anchor: float
    middle_anchor: float
    right_anchor: float

    @property
    def left_cut(self) -> float:
        return (self.left_anchor + self.middle_anchor) / 2

    @property
    def right_cut(self) -> float:
        return (self.middle_anchor + self.right_anchor) / 2


@dataclass
class TableRowBuilder:
    block_label: str
    block_display: str
    block_intro: str
    page_start: int
    order_hint: Tuple[int, float, int]
    text_parts: List[str]
    source_parts: List[str]
    page_end: int


def ensure_pymupdf_available() -> None:
    if fitz is not None:
        return
    raise RuntimeError(
        "PyMuPDF não está disponível no ambiente atual. Instale com "
        "`python -m pip install PyMuPDF` no ambiente ativo. Neste repositório, a "
        "implementação foi validada com `.venv/bin/python`."
    )


def normalize_space_characters(text: str) -> str:
    text = str(text or "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060]", "", text)
    return text


def normalize_inline_whitespace(text: str) -> str:
    text = normalize_space_characters(text).replace("\r", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def merge_text_fragments(left: str, right: str) -> str:
    left = str(left or "").rstrip()
    right = str(right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-") and right and right[0].isalnum():
        return left[:-1] + right
    if right[:1] in ",.;:!?)]}":
        return left + right
    return f"{left} {right}"


def clean_text_for_csv(text: str) -> str:
    text = normalize_space_characters(text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_heading_text(text: str) -> str:
    text = clean_text_for_csv(text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" -  ", " - ")
    return text.strip()


def extract_raw_page_lines(page: Any, page_number: int) -> List[PageLine]:
    blocks = page.get_text("dict").get("blocks", [])
    lines: List[PageLine] = []
    for block in blocks:
        if block.get("type", 0) != 0:
            continue
        for raw_line in block.get("lines", []):
            spans = raw_line.get("spans", [])
            texts = []
            x_positions = []
            sizes = []
            fonts = []
            is_bold = False
            is_italic = False
            for span in spans:
                text = normalize_inline_whitespace(span.get("text", ""))
                if not text:
                    continue
                texts.append(text)
                x_positions.append(float(span["bbox"][0]))
                sizes.append(float(span["size"]))
                font = str(span.get("font", ""))
                fonts.append(font)
                if "Bold" in font or "SemiBold" in font:
                    is_bold = True
                if "Italic" in font or font.endswith("It"):
                    is_italic = True
            if not texts:
                continue
            font = fonts[0] if fonts else ""
            size = max(sizes) if sizes else 0.0
            lines.append(
                PageLine(
                    page_number=page_number,
                    text=" ".join(texts),
                    x0=min(x_positions) if x_positions else 0.0,
                    y0=float(raw_line["bbox"][1]),
                    size=size,
                    font=font,
                    is_bold=is_bold,
                    is_italic=is_italic,
                )
            )
    lines.sort(key=lambda item: (round(item.y0, 1), round(item.x0, 1), -item.size))
    return lines


def is_noise_line(line: PageLine) -> bool:
    text = normalize_inline_whitespace(line.text)
    if not text:
        return True
    if text == HEADER_BANNER:
        return True
    if PAGE_ONLY_RE.fullmatch(text):
        return True
    if line.size < MIN_FONT_SIZE:
        return True
    return False


def is_part_heading(line: PageLine) -> bool:
    text = clean_heading_text(line.text)
    return line.size >= 18.0 and "Conthrax" in line.font and bool(PART_RE.match(text))


def is_numbered_heading(line: PageLine) -> bool:
    if line.size < 14.0:
        return False
    text = clean_heading_text(line.text)
    return "Conthrax" in line.font and bool(NUMBERED_HEADING_RE.match(text))


def is_heading_merge_candidate(line: PageLine) -> bool:
    if is_part_heading(line) or is_numbered_heading(line):
        return True
    return line.size >= 14.0 and "Conthrax" in line.font and REFERENCE_RE.match(clean_heading_text(line.text)) is not None


def is_heading_continuation(seed: PageLine, candidate: PageLine) -> bool:
    if candidate.page_number != seed.page_number:
        return False
    if candidate.size < 14.0:
        return False
    if "Conthrax" not in candidate.font:
        return False
    if abs(candidate.x0 - seed.x0) > 16:
        return False
    if abs(candidate.size - seed.size) > 1.1:
        return False
    return True


def merge_multiline_headings(lines: Sequence[PageLine]) -> List[PageLine]:
    merged: List[PageLine] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if not is_heading_merge_candidate(current):
            merged.append(current)
            index += 1
            continue

        parts = [current.text]
        last = current
        cursor = index + 1
        while cursor < len(lines):
            nxt = lines[cursor]
            if not is_heading_continuation(current, nxt):
                break
            if nxt.y0 - last.y0 > 44:
                break
            parts.append(nxt.text)
            last = nxt
            cursor += 1

        if len(parts) == 1:
            merged.append(current)
            index += 1
            continue

        text = ""
        for part in parts:
            text = merge_text_fragments(text, part)
        merged.append(
            PageLine(
                page_number=current.page_number,
                text=clean_heading_text(text),
                x0=current.x0,
                y0=current.y0,
                size=max(current.size, last.size),
                font=current.font,
                is_bold=True,
                is_italic=current.is_italic and last.is_italic,
            )
        )
        index = cursor
    return merged


def extract_page_lines(page: Any, page_number: int) -> List[PageLine]:
    raw_lines = [line for line in extract_raw_page_lines(page, page_number) if not is_noise_line(line)]
    return merge_multiline_headings(raw_lines)


def looks_like_author_line(line: PageLine) -> bool:
    text = clean_heading_text(line.text)
    if not text or len(text.split()) < 2:
        return False
    if line.size < 9.5 or line.size > 12.5:
        return False
    if not (line.is_italic or "Light" in line.font or "RegularIt" in line.font):
        return False
    if re.search(r"\b(?:ISBN|http|www\.|\d{4})\b", text, re.IGNORECASE):
        return False
    return bool(re.search(r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+", text))


def is_references_heading(line: PageLine) -> bool:
    return REFERENCE_RE.match(clean_heading_text(line.text)) is not None


def resolve_input_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return str(path.resolve())


def resolve_output_path(path_value: str) -> str:
    raw_value = str(path_value or DEFAULT_OUTPUT_NAME).strip()
    path = Path(raw_value).expanduser()
    if path.exists() and path.is_dir():
        path = path / DEFAULT_OUTPUT_NAME
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def find_local_input_file() -> Optional[str]:
    candidate = _PROJECT_ROOT / "Artefatos" / "dados" / "insumos" / DEFAULT_INPUT_NAME
    if candidate.is_file():
        return str(candidate.resolve())
    pdfs = sorted(SCRIPT_DIR.glob("*.pdf"))
    if not pdfs:
        return None
    return str(pdfs[0].resolve())


def select_input_and_output_via_gui(default_output_name: str = DEFAULT_OUTPUT_NAME) -> Tuple[Optional[str], Optional[str]]:
    gui = open_file_panel(
        title="Relatório IDP para CSV (RAG)",
        subtitle="Selecione o PDF do relatório e configure o CSV de saída.",
        filetypes=[("PDF", "*.pdf"), ("Todos os arquivos", "*.*")],
        extensions=[".pdf"],
        initial_files=[],
        allow_add_dir=False,
        recursive_dir=False,
        min_files=1,
        output_label="Pasta de saída",
        initial_output=str(SCRIPT_DIR),
        extra_texts=[("output_name", "Nome do CSV de saída", default_output_name)],
    )
    if not gui or not gui.get("confirmed"):
        return None, None
    files = list(gui.get("files") or [])
    if not files:
        return None, None
    output_dir = str(gui.get("output") or "").strip() or str(SCRIPT_DIR)
    output_name = str((gui.get("texts") or {}).get("output_name", "")).strip() or default_output_name
    output_file = resolve_output_path(str(Path(output_dir) / Path(output_name).name))
    return files[0], output_file


def parse_heading(line: PageLine) -> Tuple[str, str]:
    text = clean_heading_text(line.text)
    match = NUMBERED_HEADING_RE.match(text)
    if not match:
        return "", text
    return match.group("num").rstrip("."), clean_heading_text(match.group("title"))


def parse_part_heading(text: str) -> str:
    text = clean_heading_text(text)
    match = PART_RE.match(text)
    if not match:
        return text
    roman = match.group(1).upper()
    title = clean_heading_text(match.group(2))
    return f"PARTE {roman} - {title}"


def format_page_range(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"pp. {page_start}"
    return f"pp. {page_start}-{page_end}"


def should_start_new_paragraph(previous: PageLine, current: PageLine) -> bool:
    if current.page_number != previous.page_number:
        if previous.text.rstrip().endswith("-"):
            return False
        if current.x0 >= 80:
            return True
        return previous.text.rstrip().endswith((".", "!", "?", ":"))
    vertical_gap = current.y0 - previous.y0
    if vertical_gap > max(previous.size, current.size) * 1.45:
        return True
    if current.x0 >= 80 and previous.x0 < 70:
        return True
    return False


def build_text_from_lines(lines: Sequence[PageLine]) -> str:
    if not lines:
        return ""
    paragraphs: List[str] = []
    current_parts: List[str] = []
    previous: Optional[PageLine] = None
    for line in lines:
        if previous is not None and should_start_new_paragraph(previous, line):
            paragraph = ""
            for part in current_parts:
                paragraph = merge_text_fragments(paragraph, part)
            paragraph = clean_text_for_csv(paragraph)
            if paragraph:
                paragraphs.append(paragraph)
            current_parts = []
        current_parts.append(line.text)
        previous = line

    if current_parts:
        paragraph = ""
        for part in current_parts:
            paragraph = merge_text_fragments(paragraph, part)
        paragraph = clean_text_for_csv(paragraph)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def split_text_with_overlap(text: str, max_chars: int, min_chars: int, overlap_chars: int) -> List[str]:
    text = clean_text_for_csv(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    segments: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            segments.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                segments.append(current)
            if len(sentence) <= max_chars:
                current = sentence
                continue
            words = sentence.split()
            chunk = ""
            for word in words:
                candidate_word = word if not chunk else f"{chunk} {word}"
                if len(candidate_word) <= max_chars:
                    chunk = candidate_word
                    continue
                if chunk:
                    segments.append(chunk)
                chunk = word
            current = chunk
        if current:
            segments.append(current)

    chunks: List[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else f"{current}\n\n{segment}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        overlap = ""
        if chunks and overlap_chars > 0:
            overlap = chunks[-1][-overlap_chars:].strip()
        current = segment if not overlap else clean_text_for_csv(f"{overlap} {segment}")
        if len(current) > max_chars:
            chunks.append(current[:max_chars].rstrip())
            current = current[max_chars - overlap_chars :].lstrip() if overlap_chars > 0 else ""
    if current:
        chunks.append(current)

    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        chunks[-2] = clean_text_for_csv(f"{chunks[-2]}\n\n{chunks[-1]}")
        chunks.pop()

    return [clean_text_for_csv(chunk) for chunk in chunks if clean_text_for_csv(chunk)]


def sentence_candidates(text: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?])\s+", clean_text_for_csv(text))
    return [piece.strip() for piece in pieces if piece.strip()]


def build_short_summary(text: str) -> str:
    for sentence in sentence_candidates(text):
        if len(sentence) < 40:
            continue
        summary = sentence.rstrip()
        if len(summary) <= 280:
            return summary
        return summary[:277].rstrip() + "..."
    text = clean_text_for_csv(text)
    if len(text) <= 280:
        return text
    return text[:277].rstrip() + "..."


def tokenize_keyword_terms(text: str) -> List[str]:
    base = strip_accents(text).lower()
    return re.findall(r"[a-z0-9]{3,}", base)


def restore_keyword_case(term: str) -> str:
    specials = {
        "ai": "IA",
        "anpd": "ANPD",
        "dsa": "DSA",
        "idp": "IDP",
        "lgpd": "LGPD",
        "osa": "OSA",
        "stf": "STF",
        "tse": "TSE",
        "ue": "UE",
    }
    if term in specials:
        return specials[term]
    return " ".join(piece.capitalize() for piece in term.split())


def build_keywords(*texts: str, max_terms: int = 8) -> str:
    score = Counter()
    for index, raw_text in enumerate(texts):
        text = clean_text_for_csv(raw_text)
        if not text:
            continue
        weight = 4 if index == 0 else 2 if index == 1 else 1
        tokens = tokenize_keyword_terms(text)
        for token in tokens:
            if token in PT_STOPWORDS:
                continue
            score[token] += weight + (2 if token in KEYWORD_PRIORITY_TERMS else 0)
        for first, second in zip(tokens, tokens[1:]):
            if first in PT_STOPWORDS or second in PT_STOPWORDS:
                continue
            bigram = f"{first} {second}"
            score[bigram] += weight + 1 + (2 if first in KEYWORD_PRIORITY_TERMS or second in KEYWORD_PRIORITY_TERMS else 0)
    ordered = [item for item, _count in score.most_common() if item]
    cleaned: List[str] = []
    seen = set()
    for item in ordered:
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(restore_keyword_case(item))
        if len(cleaned) >= max_terms:
            break
    return ", ".join(cleaned)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(clean_text_for_csv(text)) / 4))


def extract_lead_topic(text: str, max_words: int = 12) -> str:
    first_line = clean_text_for_csv(text).split("\n", 1)[0]
    if ":" in first_line:
        lead = first_line.split(":", 1)[0].strip()
        if 4 <= len(lead) <= 100:
            return lead
    words = first_line.split()
    return " ".join(words[:max_words]).strip(" ,.;:-")


def detect_part_pages(page_lines_by_page: Sequence[Sequence[PageLine]]) -> Dict[str, int]:
    part_pages: Dict[str, int] = {}
    for lines in page_lines_by_page:
        for line in lines:
            if not is_part_heading(line):
                continue
            normalized = parse_part_heading(line.text)
            roman_match = PART_RE.match(normalized)
            if not roman_match:
                continue
            roman = roman_match.group(1).upper()
            if roman not in part_pages and line.page_number > 5:
                part_pages[roman] = line.page_number
    return part_pages


def collect_narrative_sources(
    page_lines_by_page: Sequence[Sequence[PageLine]],
    table_start_page: int,
    table_end_page: int,
) -> List[SourceRecord]:
    sources: List[SourceRecord] = []
    started = False
    current_part = ""
    current_section = ""
    current_subsection = ""
    current_authors = ""
    current_lines: List[PageLine] = []
    reference_lines: List[PageLine] = []
    in_references = False
    allow_author_capture = False
    section_map: Dict[str, str] = {}
    source_counter = 0

    def flush_current(chunk_type: str) -> None:
        nonlocal current_lines, reference_lines, source_counter
        target_lines = reference_lines if chunk_type == "referencia_bibliografica" else current_lines
        text = build_text_from_lines(target_lines)
        if text:
            first_line = target_lines[0]
            last_line = target_lines[-1]
            path_parts = [current_part]
            if current_section:
                path_parts.append(current_section)
            if current_subsection:
                path_parts.append(current_subsection)
            sources.append(
                SourceRecord(
                    part=current_part,
                    section=current_section,
                    subsection=current_subsection,
                    path=" > ".join(part for part in path_parts if part),
                    chunk_type=chunk_type,
                    authors=current_authors,
                    page_start=first_line.page_number,
                    page_end=last_line.page_number,
                    text=text,
                    order_hint=(first_line.page_number, first_line.y0, source_counter),
                )
            )
            source_counter += 1
        if chunk_type == "referencia_bibliografica":
            reference_lines = []
        else:
            current_lines = []

    for page_lines in page_lines_by_page:
        if not page_lines:
            continue
        page_number = page_lines[0].page_number
        if table_start_page <= page_number < table_end_page:
            continue

        for line in page_lines:
            if not started and not is_part_heading(line):
                continue
            if is_part_heading(line):
                flush_current("referencia_bibliografica" if in_references else "narrativo")
                in_references = False
                started = True
                current_part = parse_part_heading(line.text)
                current_section = ""
                current_subsection = ""
                current_authors = ""
                allow_author_capture = True
                continue

            if not started:
                continue

            if is_numbered_heading(line):
                flush_current("referencia_bibliografica" if in_references else "narrativo")
                in_references = False
                number, title = parse_heading(line)
                label = clean_heading_text(f"{number} {title}")
                segments = number.split(".")
                if len(segments) == 2:
                    current_section = label
                    section_map[number] = label
                    current_subsection = ""
                else:
                    root_number = ".".join(segments[:2])
                    current_section = section_map.get(root_number, current_section or root_number)
                    current_subsection = label
                allow_author_capture = True
                continue

            if is_references_heading(line):
                flush_current("narrativo")
                in_references = True
                allow_author_capture = False
                continue

            if allow_author_capture and looks_like_author_line(line):
                current_authors = merge_text_fragments(current_authors, line.text)
                continue

            if allow_author_capture:
                allow_author_capture = False

            if in_references:
                reference_lines.append(line)
            else:
                current_lines.append(line)

    flush_current("referencia_bibliografica" if in_references else "narrativo")
    return sources


def detect_table_layout(lines: Sequence[PageLine], previous: Optional[TableLayout]) -> Optional[TableLayout]:
    left_anchor = previous.left_anchor if previous else 84.0
    middle_anchor = previous.middle_anchor if previous else 160.0
    right_anchor = previous.right_anchor if previous else 420.0

    bins = Counter()
    for line in lines:
        text = clean_heading_text(line.text)
        if text.startswith("Subtema do"):
            left_anchor = line.x0
        elif text.startswith("Fonte na Parte"):
            right_anchor = line.x0
        if 70 <= line.x0 <= 480 and not PAGE_ONLY_RE.fullmatch(text):
            bins[round(line.x0 / 5) * 5] += 1

    left_candidates = [value for value, _count in bins.most_common() if 70 <= value <= 130]
    middle_candidates = [value for value, _count in bins.most_common() if 130 < value <= 260]
    right_candidates = [value for value, _count in bins.most_common() if 350 <= value <= 470]
    if left_candidates:
        left_anchor = min(left_candidates)
    if middle_candidates:
        middle_anchor = min(middle_candidates)
    if right_candidates:
        right_anchor = min(right_candidates)
    if left_anchor and middle_anchor and right_anchor:
        return TableLayout(left_anchor=left_anchor, middle_anchor=middle_anchor, right_anchor=right_anchor)
    return previous


def extract_table_fragments(page: Any, page_number: int) -> List[PageLine]:
    blocks = page.get_text("dict").get("blocks", [])
    fragments: List[PageLine] = []
    for block in blocks:
        if block.get("type", 0) != 0:
            continue
        for raw_line in block.get("lines", []):
            for span in raw_line.get("spans", []):
                text = normalize_inline_whitespace(span.get("text", ""))
                if not text:
                    continue
                font = str(span.get("font", ""))
                size = float(span.get("size", 0.0))
                fragment = PageLine(
                    page_number=page_number,
                    text=text,
                    x0=float(span["bbox"][0]),
                    y0=float(raw_line["bbox"][1]),
                    size=size,
                    font=font,
                    is_bold=("Bold" in font or "SemiBold" in font),
                    is_italic=("Italic" in font or font.endswith("It")),
                )
                if is_noise_line(fragment):
                    continue
                fragments.append(fragment)
    fragments.sort(key=lambda item: (round(item.y0, 1), round(item.x0, 1)))
    return fragments


def group_table_lines(lines: Sequence[PageLine], layout: TableLayout) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def ensure_group(seed: PageLine) -> Dict[str, Any]:
        return {
            "page_number": seed.page_number,
            "y0": seed.y0,
            "left": [],
            "middle": [],
            "right": [],
            "middle_bold": False,
            "combined": [],
        }

    for line in lines:
        if current is None or abs(line.y0 - current["y0"]) > 3.5:
            current = ensure_group(line)
            grouped.append(current)
        current["combined"].append(line.text)
        target = "left"
        if line.x0 >= layout.right_cut:
            target = "right"
        elif line.x0 >= layout.left_cut:
            target = "middle"
        current[target].append(line.text)
        if target == "middle" and line.is_bold:
            current["middle_bold"] = True

    for item in grouped:
        item["left_text"] = clean_text_for_csv(" ".join(item["left"]))
        item["middle_text"] = clean_text_for_csv(" ".join(item["middle"]))
        item["right_text"] = clean_text_for_csv(" ".join(item["right"]))
        item["combined_text"] = clean_text_for_csv(" ".join(item["combined"]))
    return grouped


def is_table_header_group(group: Dict[str, Any]) -> bool:
    text = group["combined_text"]
    return "Recomendações" in text or "Fonte na Parte" in text or text.startswith("Subtema do")


def append_table_text(parts: List[str], text: str) -> None:
    text = clean_text_for_csv(text)
    if not text:
        return
    if not parts:
        parts.append(text)
        return
    if parts[-1].endswith("-") and text[:1].isalnum():
        parts[-1] = parts[-1][:-1] + text
        return
    parts.append(text)


def finalize_table_row(current_row: Optional[TableRowBuilder], rows: List[SourceRecord], order_counter: int) -> int:
    if current_row is None:
        return order_counter
    text = clean_text_for_csv(" ".join(current_row.text_parts))
    source = clean_text_for_csv(" ".join(current_row.source_parts))
    if text:
        rows.append(
            SourceRecord(
                part="PARTE III - TABELA UNIFICADA DAS RECOMENDAÇÕES NORMATIVAS E OPERACIONAIS",
                section="",
                subsection="",
                path="",
                chunk_type="tabela_recomendacao",
                authors="",
                page_start=current_row.page_start,
                page_end=current_row.page_end,
                text=text,
                block_label=current_row.block_label,
                block_display=current_row.block_display,
                block_intro=current_row.block_intro,
                subtheme="",
                source_part_ii=source,
                order_hint=(current_row.page_start, current_row.order_hint[1], order_counter),
            )
        )
        order_counter += 1
    return order_counter


def apply_subtheme_to_rows(rows: List[SourceRecord], start_index: int, subtheme_parts: Sequence[str]) -> None:
    label = clean_text_for_csv(" ".join(part for part in subtheme_parts if clean_text_for_csv(part)))
    if not label:
        return
    for index in range(start_index, len(rows)):
        if not rows[index].subtheme:
            rows[index].subtheme = label


def collect_table_sources(
    document: Any,
    table_start_page: int,
    table_end_page: int,
) -> List[SourceRecord]:
    rows: List[SourceRecord] = []
    layout: Optional[TableLayout] = None
    current_block_label = ""
    current_block_display = ""
    current_block_intro_parts: List[str] = []
    table_preamble_parts: List[str] = []
    collecting_block_intro = False
    pending_subtheme_parts: List[str] = []
    pending_subtheme_start_index = 0
    block_row_start_index = 0
    current_row: Optional[TableRowBuilder] = None
    previous_middle_bold = False
    order_counter = 0

    for page_number in range(table_start_page, table_end_page):
        page = document[page_number - 1]
        raw_lines = extract_table_fragments(page, page_number)
        layout = detect_table_layout(raw_lines, layout)
        if layout is None:
            continue
        grouped_lines = group_table_lines(raw_lines, layout)

        for group in grouped_lines:
            combined = group["combined_text"]
            left_text = group["left_text"]
            middle_text = group["middle_text"]
            right_text = group["right_text"]
            middle_bold = bool(group["middle_bold"])

            if not combined:
                continue

            block_match = BLOCK_RE.match(combined)
            if block_match:
                order_counter = finalize_table_row(current_row, rows, order_counter)
                current_row = None
                if pending_subtheme_parts:
                    apply_subtheme_to_rows(rows, pending_subtheme_start_index, pending_subtheme_parts)
                    pending_subtheme_parts = []
                current_block_label = block_match.group(1).title()
                current_block_display = clean_text_for_csv(f"{block_match.group(1)}: {block_match.group(2)}")
                current_block_intro_parts = []
                if table_preamble_parts and current_block_label == "Bloco 1":
                    current_block_intro_parts.extend(table_preamble_parts)
                    table_preamble_parts = []
                collecting_block_intro = True
                block_row_start_index = len(rows)
                previous_middle_bold = False
                continue

            if is_table_header_group(group):
                collecting_block_intro = False
                previous_middle_bold = False
                continue

            if collecting_block_intro and not middle_text and not right_text and not SUBTHEME_RE.match(left_text):
                append_table_text(current_block_intro_parts, combined)
                continue

            if not current_block_label:
                table_preamble_parts.append(combined)
                continue

            numeric_subtheme = bool(left_text and SUBTHEME_RE.match(left_text))
            if numeric_subtheme:
                if pending_subtheme_parts:
                    apply_subtheme_to_rows(rows, pending_subtheme_start_index, pending_subtheme_parts)
                pending_subtheme_parts = [left_text]
                pending_subtheme_start_index = len(rows)
            elif left_text and pending_subtheme_parts:
                pending_subtheme_parts.append(left_text)

            has_table_payload = bool(middle_text or right_text)
            if not has_table_payload:
                previous_middle_bold = False
                continue

            row_needs_restart = False
            if current_row is None:
                row_needs_restart = True
            elif middle_bold and not previous_middle_bold and current_row.text_parts:
                order_counter = finalize_table_row(current_row, rows, order_counter)
                current_row = None
                row_needs_restart = True

            if row_needs_restart:
                current_row = TableRowBuilder(
                    block_label=current_block_label,
                    block_display=current_block_display,
                    block_intro=clean_text_for_csv(" ".join(current_block_intro_parts)),
                    page_start=page_number,
                    order_hint=(page_number, float(group["y0"]), order_counter),
                    text_parts=[],
                    source_parts=[],
                    page_end=page_number,
                )
                if not pending_subtheme_parts:
                    pending_subtheme_start_index = block_row_start_index

            current_row.page_end = page_number
            append_table_text(current_row.text_parts, middle_text)
            append_table_text(current_row.source_parts, right_text)
            previous_middle_bold = middle_bold

        previous_middle_bold = False

    order_counter = finalize_table_row(current_row, rows, order_counter)
    if pending_subtheme_parts:
        apply_subtheme_to_rows(rows, pending_subtheme_start_index, pending_subtheme_parts)
    for row in rows:
        path_parts = [
            row.part,
            row.block_display or row.block_label,
            row.subtheme,
        ]
        row.path = " > ".join(part for part in path_parts if part)
    return rows


def build_chunk_title(source: SourceRecord, chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    if source.chunk_type == "tabela_recomendacao":
        base = source.subtheme or source.block_display or source.block_label or "Recomendação"
        lead = extract_lead_topic(chunk_text)
        if lead and lead not in base:
            title = f"{base} - {lead}"
        else:
            title = base
    elif source.chunk_type == "referencia_bibliografica":
        base = source.subsection or source.section or source.part or "Referências"
        title = f"Referências - {base}"
    else:
        title = source.subsection or source.section or source.part

    title = clean_heading_text(title)
    if total_chunks > 1:
        title = f"{title} (trecho {chunk_index + 1})"
    return title


def build_rag_text(source: SourceRecord, chunk_text: str) -> str:
    breadcrumb = source.path or source.part
    prefix = f"{breadcrumb} | {format_page_range(source.page_start, source.page_end)}"
    extras = []
    if source.chunk_type == "tabela_recomendacao":
        if source.block_display:
            extras.append(f"Bloco: {source.block_display}")
        if source.subtheme:
            extras.append(f"Subtema: {source.subtheme}")
        if source.source_part_ii:
            extras.append(f"Fonte na Parte II: {source.source_part_ii}")
        if source.block_intro:
            extras.append(f"Contexto do bloco: {source.block_intro}")
    if source.chunk_type == "referencia_bibliografica":
        extras.append("Tipo: Referências bibliográficas")
    meta = prefix if not extras else f"{prefix}\n" + "\n".join(extras)
    return clean_text_for_csv(f"{meta}\n\n{chunk_text}")


def chunk_source_record(
    source: SourceRecord,
    file_name: str,
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
) -> List[Dict[str, Any]]:
    if source.chunk_type == "referencia_bibliografica":
        pieces = [clean_text_for_csv(source.text)]
    else:
        pieces = split_text_with_overlap(source.text, max_chars=max_chars, min_chars=min_chars, overlap_chars=overlap_chars)

    rows: List[Dict[str, Any]] = []
    include_in_rag = source.chunk_type != "referencia_bibliografica"
    for index, piece in enumerate(pieces):
        title = build_chunk_title(source, piece, index, len(pieces))
        rag_text = build_rag_text(source, piece)
        rows.append(
            {
                "arquivo_fonte": file_name,
                "titulo_documento": DOCUMENT_TITLE,
                "parte": source.part,
                "secao": source.section,
                "subsecao": source.subsection,
                "caminho_hierarquico": source.path,
                "tipo_chunk": source.chunk_type,
                "autores": clean_text_for_csv(source.authors),
                "pagina_inicial": source.page_start,
                "pagina_final": source.page_end,
                "bloco_tabela": source.block_label,
                "subtema_tabela": source.subtheme,
                "fonte_parte_ii": source.source_part_ii,
                "titulo_chunk": title,
                "texto_limpo": piece,
                "texto_rag": rag_text,
                "resumo_curto": build_short_summary(piece),
                "palavras_chave": build_keywords(
                    title,
                    source.subtheme or source.subsection or source.section or source.part,
                    piece,
                ),
                "incluir_no_rag": include_in_rag,
                "chars": len(piece),
                "tokens_estimados": estimate_tokens(rag_text),
                "__order_hint": source.order_hint,
            }
        )
    return rows


def extract_document_rows(
    input_file: str,
    *,
    max_chars: int = 2200,
    min_chars: int = 700,
    overlap_chars: int = 180,
) -> List[Dict[str, Any]]:
    ensure_pymupdf_available()
    with fitz.open(input_file) as document:
        page_lines_by_page = [
            extract_page_lines(document[page_index], page_index + 1)
            for page_index in range(document.page_count)
        ]
        part_pages = detect_part_pages(page_lines_by_page)
        table_start_page = int(part_pages.get("III", 266))
        table_end_page = int(part_pages.get("IV", document.page_count + 1))

        narrative_sources = collect_narrative_sources(
            page_lines_by_page,
            table_start_page=table_start_page,
            table_end_page=table_end_page,
        )
        table_sources = collect_table_sources(
            document,
            table_start_page=table_start_page,
            table_end_page=table_end_page,
        )

    all_sources = narrative_sources + table_sources
    all_rows: List[Dict[str, Any]] = []
    file_name = Path(input_file).name
    for source in all_sources:
        all_rows.extend(
            chunk_source_record(
                source,
                file_name=file_name,
                max_chars=max_chars,
                min_chars=min_chars,
                overlap_chars=overlap_chars,
            )
        )

    all_rows.sort(key=lambda row: (row["pagina_inicial"], row["__order_hint"][1], row["__order_hint"][2]))
    final_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(all_rows, start=1):
        final_row = dict(row)
        final_row.pop("__order_hint", None)
        final_row["id"] = index
        final_row["ordem_doc"] = index
        final_rows.append(final_row)
    return final_rows


def write_rows_to_csv(output_file: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(output_file, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_HEADER})


def convert_idp_report_to_csv(
    input_filename: str = "",
    output_filename: str = "",
    *,
    max_chars: int = 2200,
    min_chars: int = 700,
    overlap_chars: int = 180,
    use_gui: bool = True,
) -> str:
    input_filename = str(input_filename or "").strip()
    output_filename = str(output_filename or "").strip()

    if not input_filename and use_gui:
        gui_input, gui_output = select_input_and_output_via_gui()
        if gui_input:
            input_filename = gui_input
            if gui_output:
                output_filename = gui_output

    if input_filename:
        input_filename = resolve_input_path(input_filename)
    else:
        found = find_local_input_file()
        if not found:
            raise FileNotFoundError(
                "Nenhum PDF foi encontrado. Informe `--input-file` ou coloque o relatório na pasta do script."
            )
        input_filename = found

    if not output_filename:
        output_filename = resolve_output_path(DEFAULT_OUTPUT_NAME)
    else:
        output_filename = resolve_output_path(output_filename)

    rows = extract_document_rows(
        input_filename,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap_chars,
    )
    write_rows_to_csv(output_filename, rows)
    return output_filename


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conversor do relatório IDP para CSV RAG-friendly.")
    parser.add_argument("--input-file", default="", help="PDF de entrada.")
    parser.add_argument("--output-csv", default="", help="CSV de saída.")
    parser.add_argument("--max-chars", type=int, default=2200, help="Tamanho máximo por chunk narrativo.")
    parser.add_argument("--min-chars", type=int, default=700, help="Tamanho mínimo desejado por chunk.")
    parser.add_argument("--overlap-chars", type=int, default=180, help="Overlap entre chunks consecutivos.")
    parser.add_argument("--no-gui", action="store_true", help="Desativa a seleção por GUI.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = convert_idp_report_to_csv(
        input_filename=args.input_file,
        output_filename=args.output_csv,
        max_chars=max(300, int(args.max_chars)),
        min_chars=max(100, int(args.min_chars)),
        overlap_chars=max(0, int(args.overlap_chars)),
        use_gui=not bool(args.no_gui),
    )
    print(f"[SUCESSO] CSV gerado em: {output_path}")


if __name__ == "__main__":
    main()
