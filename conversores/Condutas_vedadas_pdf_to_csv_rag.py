#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converte a cartilha Condutas Vedadas em CSV amigável para Notion/RAG.

Fluxo:
1. Localiza o PDF da cartilha ou recebe um caminho explícito via CLI/GUI.
2. Extrai linhas, fontes e coordenadas com PyMuPDF.
3. Detecta Apresentação, capítulos, seções numeradas, subseções e marcos do calendário.
4. Gera chunks narrativos com metadados úteis para recuperação semântica.
5. Escreve um CSV rico em contexto para importação no Notion.
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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gui_intuitiva import open_file_panel
from IDP_relatorio_pdf_to_csv_rag import (
    CSV_HEADER,
    PageLine,
    build_keywords,
    build_short_summary,
    clean_heading_text,
    clean_text_for_csv,
    ensure_pymupdf_available,
    estimate_tokens,
    extract_raw_page_lines,
    fitz,
    merge_text_fragments,
    split_text_with_overlap,
)


SCRIPT_DIR = _PROJECT_ROOT  # raiz do projeto (o script foi movido para conversores/)
DEFAULT_INPUT_NAME = "Artefatos/dados/insumos/Condutas_vedadas_2024_Digital_15mb.pdf"
DEFAULT_OUTPUT_NAME = "Condutas_vedadas_2024_rag.csv"
DOCUMENT_TITLE = "Condutas vedadas aos agentes públicos federais em eleições 2024"

PAGE_ONLY_RE = re.compile(r"^\d{1,3}$")
NUMBERED_HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)+\.?)\s+(?P<title>.+)$")
CALENDAR_HEADING_RE = re.compile(
    r"^\d{1,2}(?:º)?\s+de\s+[A-Za-zÀ-ÿçÇ]+\s+[–-]\s+.+$",
    re.IGNORECASE,
)
INTRO_HEADING_RE = re.compile(r"^Apresentação$", re.IGNORECASE)

HEADER_BANNERS = {
    "CONDUTAS vedadas",
    "AOS AGENTES PÚBLICOS FEDERAIS EM ELEIÇÕES",
    "10ª edição, revista e atualizada pela Advocacia-Geral da União",
    "2024",
    "Capítulo",
}


@dataclass
class SourceRecord:
    part: str
    section: str
    subsection: str
    path: str
    page_start: int
    page_end: int
    text: str
    order_hint: Tuple[int, float, int]


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
    candidate = SCRIPT_DIR / DEFAULT_INPUT_NAME
    if candidate.is_file():
        return str(candidate.resolve())
    pdfs = sorted(SCRIPT_DIR.glob("*.pdf"))
    if not pdfs:
        return None
    return str(pdfs[0].resolve())


def select_input_and_output_via_gui(default_output_name: str = DEFAULT_OUTPUT_NAME) -> Tuple[Optional[str], Optional[str]]:
    gui = open_file_panel(
        title="Condutas Vedadas para CSV (RAG)",
        subtitle="Selecione o PDF da cartilha e configure o CSV de saída.",
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


def find_chapter_number(raw_lines: Sequence[PageLine]) -> str:
    has_capitulo = any(clean_heading_text(line.text) == "Capítulo" for line in raw_lines)
    if not has_capitulo:
        return ""
    for line in raw_lines:
        text = clean_heading_text(line.text)
        if line.size >= 45.0 and line.x0 >= 450 and PAGE_ONLY_RE.fullmatch(text):
            return text.zfill(2)
    return ""


def is_noise_line(line: PageLine) -> bool:
    text = clean_heading_text(line.text)
    if not text:
        return True
    if text in HEADER_BANNERS:
        return True
    if PAGE_ONLY_RE.fullmatch(text):
        return True
    if line.size < 10.0:
        return True
    return False


def is_heading_merge_candidate(line: PageLine) -> bool:
    text = clean_heading_text(line.text)
    return bool(text) and "Alice" in line.font and line.size >= 15.0 and not PAGE_ONLY_RE.fullmatch(text)


def is_heading_continuation(seed: PageLine, candidate: PageLine) -> bool:
    if candidate.page_number != seed.page_number:
        return False
    if not is_heading_merge_candidate(candidate):
        return False
    if abs(candidate.x0 - seed.x0) > 28:
        return False
    if abs(candidate.size - seed.size) > 2.5:
        return False
    max_gap = max(28.0, max(seed.size, candidate.size) * 1.35)
    return (candidate.y0 - seed.y0) <= max_gap


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
            if not is_heading_continuation(last, nxt):
                break
            parts.append(nxt.text)
            last = nxt
            cursor += 1

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
                is_bold=current.is_bold or last.is_bold,
                is_italic=current.is_italic and last.is_italic,
            )
        )
        index = cursor
    return merged


def extract_page_lines(page: Any, page_number: int) -> List[PageLine]:
    raw_lines = extract_raw_page_lines(page, page_number)
    merged = merge_multiline_headings(raw_lines)
    return [line for line in merged if not is_noise_line(line)]


def is_intro_heading(line: PageLine) -> bool:
    text = clean_heading_text(line.text)
    return line.size >= 35.0 and "Alice" in line.font and INTRO_HEADING_RE.match(text) is not None


def is_chapter_title(line: PageLine, chapter_numbers: Dict[int, str]) -> bool:
    text = clean_heading_text(line.text)
    if not text or line.page_number not in chapter_numbers:
        return False
    if line.size < 35.0 or "Alice" not in line.font:
        return False
    return PAGE_ONLY_RE.fullmatch(text) is None


def is_numbered_heading(line: PageLine) -> bool:
    text = clean_heading_text(line.text)
    return line.size >= 15.0 and "Alice" in line.font and NUMBERED_HEADING_RE.match(text) is not None


def is_calendar_heading(line: PageLine) -> bool:
    text = clean_heading_text(line.text)
    return line.size >= 24.0 and "Alice" in line.font and CALENDAR_HEADING_RE.match(text) is not None


def parse_numbered_heading(text: str) -> Tuple[str, str]:
    normalized = clean_heading_text(text)
    match = NUMBERED_HEADING_RE.match(normalized)
    if not match:
        return "", normalized
    return match.group("num").rstrip("."), clean_heading_text(match.group("title"))


def format_page_range(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"pp. {page_start}"
    return f"pp. {page_start}-{page_end}"


def should_start_new_paragraph(previous: PageLine, current: PageLine) -> bool:
    if current.page_number != previous.page_number:
        return not previous.text.rstrip().endswith("-")

    vertical_gap = current.y0 - previous.y0
    if vertical_gap > max(previous.size, current.size) * 1.75:
        return True
    if current.x0 >= 95 and previous.x0 <= 75 and vertical_gap > max(previous.size, current.size) * 1.25:
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


def build_chunk_title(source: SourceRecord, chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    del chunk_text
    title = source.subsection or source.section or source.part
    title = clean_heading_text(title)
    if total_chunks > 1:
        title = f"{title} (trecho {chunk_index + 1})"
    return title


def build_rag_text(source: SourceRecord, chunk_text: str) -> str:
    breadcrumb = source.path or source.part
    prefix = f"{breadcrumb} | {format_page_range(source.page_start, source.page_end)}"
    return clean_text_for_csv(f"{prefix}\n\n{chunk_text}")


def collect_narrative_sources(
    page_lines_by_page: Sequence[Sequence[PageLine]],
    chapter_numbers: Dict[int, str],
) -> List[SourceRecord]:
    sources: List[SourceRecord] = []
    started = False
    current_part = ""
    current_section = ""
    current_subsection = ""
    current_lines: List[PageLine] = []
    section_map: Dict[str, str] = {}
    source_counter = 0

    def flush_current() -> None:
        nonlocal current_lines, source_counter
        text = build_text_from_lines(current_lines)
        if text and current_part:
            first_line = current_lines[0]
            last_line = current_lines[-1]
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
                    page_start=first_line.page_number,
                    page_end=last_line.page_number,
                    text=text,
                    order_hint=(first_line.page_number, first_line.y0, source_counter),
                )
            )
            source_counter += 1
        current_lines = []

    for page_lines in page_lines_by_page:
        if not page_lines:
            continue

        for line in page_lines:
            if is_intro_heading(line):
                flush_current()
                started = True
                current_part = "Apresentação"
                current_section = ""
                current_subsection = ""
                continue

            if is_chapter_title(line, chapter_numbers):
                flush_current()
                started = True
                current_section = ""
                current_subsection = ""
                chapter_number = chapter_numbers.get(line.page_number, "")
                current_part = f"Capítulo {chapter_number} - {clean_heading_text(line.text)}"
                continue

            if not started:
                continue

            if is_numbered_heading(line):
                flush_current()
                number, title = parse_numbered_heading(line.text)
                label = clean_heading_text(f"{number} {title}")
                segments = number.split(".")
                if len(segments) == 2:
                    current_section = label
                    current_subsection = ""
                    section_map[number] = label
                else:
                    root_number = ".".join(segments[:2])
                    current_section = section_map.get(root_number, current_section or root_number)
                    current_subsection = label
                continue

            if is_calendar_heading(line):
                flush_current()
                current_section = clean_heading_text(line.text)
                current_subsection = ""
                continue

            current_lines.append(line)

    flush_current()
    return sources


def chunk_source_record(
    source: SourceRecord,
    file_name: str,
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
) -> List[Dict[str, Any]]:
    pieces = split_text_with_overlap(
        source.text,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap_chars,
    )

    rows: List[Dict[str, Any]] = []
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
                "tipo_chunk": "narrativo",
                "autores": "",
                "pagina_inicial": source.page_start,
                "pagina_final": source.page_end,
                "bloco_tabela": "",
                "subtema_tabela": "",
                "fonte_parte_ii": "",
                "titulo_chunk": title,
                "texto_limpo": piece,
                "texto_rag": rag_text,
                "resumo_curto": build_short_summary(piece),
                "palavras_chave": build_keywords(
                    title,
                    source.subsection or source.section or source.part,
                    piece,
                ),
                "incluir_no_rag": True,
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
        chapter_numbers: Dict[int, str] = {}
        page_lines_by_page: List[List[PageLine]] = []
        for page_index in range(document.page_count):
            raw_lines = extract_raw_page_lines(document[page_index], page_index + 1)
            chapter_number = find_chapter_number(raw_lines)
            if chapter_number:
                chapter_numbers[page_index + 1] = chapter_number
            merged = merge_multiline_headings(raw_lines)
            page_lines_by_page.append([line for line in merged if not is_noise_line(line)])

    sources = collect_narrative_sources(page_lines_by_page, chapter_numbers)
    all_rows: List[Dict[str, Any]] = []
    file_name = Path(input_file).name
    for source in sources:
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


def convert_condutas_pdf_to_csv(
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
                "Nenhum PDF foi encontrado. Informe `--input-file` ou coloque a cartilha na pasta do script."
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
    parser = argparse.ArgumentParser(description="Conversor da cartilha Condutas Vedadas para CSV RAG-friendly.")
    parser.add_argument("--input-file", default="", help="PDF de entrada.")
    parser.add_argument("--output-csv", default="", help="CSV de saída.")
    parser.add_argument("--max-chars", type=int, default=2200, help="Tamanho máximo por chunk narrativo.")
    parser.add_argument("--min-chars", type=int, default=700, help="Tamanho mínimo desejado por chunk.")
    parser.add_argument("--overlap-chars", type=int, default=180, help="Overlap entre chunks consecutivos.")
    parser.add_argument("--no-gui", action="store_true", help="Desativa a seleção por GUI.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = convert_condutas_pdf_to_csv(
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
