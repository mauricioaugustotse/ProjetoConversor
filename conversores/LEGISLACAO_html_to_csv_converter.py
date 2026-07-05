# -*- coding: utf-8 -*-
"""
Converte HTMLs de legislação salvos do portal do TSE em CSV compatível com o
script paradigma, preservando o schema:

    ID, Nome (Artigo), Texto do Artigo

O parser cobre:
- corpo principal da resolução (artigos, parágrafos, incisos, alíneas e itens);
- anexos em texto;
- anexos em tabelas cronológicas;
- anexos em tabelas binárias, como "VEDADA(O)" x "PERMITIDA".
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "Artefatos"
SUPPORTED_EXTENSIONS = (".html", ".htm")

if getattr(sys, "pycache_prefix", None) is None:
    sys.pycache_prefix = str(ARTIFACTS_DIR / "__pycache__")


FOOTER_PATTERNS = (
    re.compile(r"^Este texto não substitui o publicado", re.IGNORECASE),
    re.compile(r"^Funcionamento dos protocolos", re.IGNORECASE),
    re.compile(r"^Caso necessite de informações institucionais", re.IGNORECASE),
    re.compile(r"^Ouvidoria\.?$", re.IGNORECASE),
    re.compile(r"^Este site é protegido por hCaptcha", re.IGNORECASE),
    re.compile(r"^Carregando formulário", re.IGNORECASE),
    re.compile(r"^O Portal do TSE coleta dados", re.IGNORECASE),
    re.compile(r"^Usamos cookies", re.IGNORECASE),
    re.compile(r"^Cookies ", re.IGNORECASE),
)

SIGNATURE_PATTERNS = (
    re.compile(r"^Brasília,\s+\d{1,2}\s+de\s+.+\s+de\s+\d{4}\.?$", re.IGNORECASE),
    re.compile(r"^(MINISTRA?|MINISTRO)\b", re.IGNORECASE),
    re.compile(r"^(RELATORA?|RELATOR)\b", re.IGNORECASE),
)

ANNEX_HEADER_PATTERN = re.compile(r"^ANEXO(?:\s+[IVXLCDM]+)?(?:\s*[-–—]\s*.+)?$", re.IGNORECASE)
MONTH_HEADER_PATTERN = re.compile(
    r"^[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇÜ]+\s+DE\s+\d{4}$"
)
NUMERIC_ITEM_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\.\s*(.*)$")
ROMAN_ITEM_PATTERN = re.compile(r"^([IVXLCDM]+)\s*[-–—]\s*(.*)$")


class HTMLBlockParser(HTMLParser):
    """Extrai blocos ordenados de <p> e <table> sem depender de libs externas."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict] = []
        self.table_depth = 0

        self.in_paragraph = False
        self.current_paragraph: list[str] = []

        self.in_row = False
        self.in_cell = False
        self.current_table_rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: list[str] = []

    def _append_block(self, block_type: str, **payload) -> None:
        self.blocks.append({"type": block_type, **payload})

    def _push_break_to_cell(self) -> None:
        if self.in_cell and self.current_cell:
            if not self.current_cell[-1].endswith("\n"):
                self.current_cell.append("\n")

    def _push_break_to_paragraph(self) -> None:
        if self.in_paragraph and self.current_paragraph:
            if not self.current_paragraph[-1].endswith("\n"):
                self.current_paragraph.append("\n")

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()

        if tag == "table":
            if self.table_depth == 0:
                self.current_table_rows = []
            self.table_depth += 1
            return

        if self.table_depth > 0:
            if tag == "tr":
                self.in_row = True
                self.current_row = []
            elif tag in {"td", "th"} and self.in_row:
                self.in_cell = True
                self.current_cell = []
            elif tag == "p" and self.in_cell:
                self._push_break_to_cell()
            elif tag == "br" and self.in_cell:
                self._push_break_to_cell()
            return

        if tag == "p":
            self.in_paragraph = True
            self.current_paragraph = []
        elif tag == "br" and self.in_paragraph:
            self._push_break_to_paragraph()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "table":
            if self.table_depth > 0:
                self.table_depth -= 1
                if self.table_depth == 0 and self.current_table_rows:
                    self._append_block("table", rows=self.current_table_rows)
                    self.current_table_rows = []
            return

        if self.table_depth > 0:
            if tag in {"td", "th"} and self.in_cell:
                text = "".join(self.current_cell).strip()
                self.current_row.append(text)
                self.current_cell = []
                self.in_cell = False
            elif tag == "tr" and self.in_row:
                if any(cell.strip() for cell in self.current_row):
                    self.current_table_rows.append(self.current_row)
                self.current_row = []
                self.in_row = False
            elif tag == "p" and self.in_cell:
                self._push_break_to_cell()
            return

        if tag == "p" and self.in_paragraph:
            text = "".join(self.current_paragraph).strip()
            if text:
                self._append_block("p", text=text)
            self.current_paragraph = []
            self.in_paragraph = False

    def handle_data(self, data: str) -> None:
        if self.table_depth > 0 and self.in_cell:
            self.current_cell.append(data)
        elif self.in_paragraph:
            self.current_paragraph.append(data)


def normalize_ordinals(text: str) -> str:
    if not text:
        return text
    text = text.replace(".\u00ba", "\u00ba").replace(".\u00aa", "\u00aa")
    text = re.sub(r"\u00ba{2,}", "\u00ba", text)
    text = re.sub(r"\u00aa{2,}", "\u00aa", text)
    text = re.sub(r"(\d+)\s*[oO\u00ba]\b", lambda m: f"{m.group(1)}\u00ba", text)
    return text


def normalize_decimal_commas(text: str) -> str:
    if not text:
        return text
    return re.sub(r"(?<=\d)\s*,\s*(?=\d)", ",", text)


def normalize_space_characters(text: str) -> str:
    if not text:
        return text
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060]", "", text)
    return text


article_sequence_pattern = re.compile(
    r"(\barts?\.)"
    r"(\s*)"
    r"(\d[\wºª-]*(?:,\s*\d[\wºª-]*)+)",
    flags=re.IGNORECASE,
)


def fix_article_number_spacing(text: str) -> str:
    if not text:
        return text

    def repl(match):
        prefix, spacing, numbers = match.groups()
        spacing = spacing if spacing else " "
        normalized_numbers = re.sub(r",\s*(?=\d)", ", ", numbers)
        return f"{prefix}{spacing}{normalized_numbers}"

    return article_sequence_pattern.sub(repl, text)


def clean_title(raw_title: str) -> str:
    cleaned = normalize_space_characters(raw_title)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"(?i)^Art\.\s*", "Art. ", cleaned)
    cleaned = normalize_ordinals(cleaned)
    cleaned = cleaned.rstrip(".")
    cleaned = re.sub(r"\u00a7\s+", "\u00a7", cleaned)
    cleaned = re.sub(
        r"(\u00a7)([1-9])(?!\u00ba|\d)",
        lambda m: f"{m.group(1)}{m.group(2)}\u00ba",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)(Lei)\s+n[\s.\-]*(?:\u00ba|o)?\s*(?=\d)",
        lambda m: f"{m.group(1)} n\u00ba ",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def collapse_duplicate_text(text: str) -> str:
    if not text:
        return text

    leading_len = len(text) - len(text.lstrip())
    trailing_len = len(text) - len(text.rstrip())
    core = text[leading_len : len(text) - trailing_len if trailing_len else None]

    starts_with_newline = core.startswith("\n") if core else False
    core_stripped = core.strip()

    if core_stripped:
        duplicate_match = re.match(r"(?P<block>.+?)\s+(?P=block)$", core_stripped, re.DOTALL)
        if duplicate_match:
            core_stripped = duplicate_match.group("block")

    if starts_with_newline:
        core_stripped = "\n" + core_stripped

    prefix = text[:leading_len] if leading_len else ""
    suffix = text[len(text) - trailing_len :] if trailing_len else ""
    return f"{prefix}{core_stripped}{suffix}"


def clean_text_candidate(candidate: str) -> str:
    candidate = normalize_space_characters(candidate)
    candidate = candidate.replace("\r", "")
    candidate = candidate.strip()
    if not candidate:
        return ""

    candidate = re.sub(r"^#{1,6}\s*", "", candidate)
    candidate = re.sub(r"^[_*]+", "", candidate)
    candidate = re.sub(r"\[\s*\]\s*\([^)]+\)", "", candidate)
    candidate = re.sub(r"\[([^\]]+)\]\s*\([^)]*\)", r"\1", candidate)
    candidate = re.sub(r"\[([^\]]+)\]", r"\1", candidate)
    candidate = re.sub(r"\(\s*(?:https?://|\.\.?/|#)[^)]+\)", "", candidate)
    candidate = re.sub(
        r"\(\s*[\w\u00C0-\u024F./#%-]*\.html?[^)]*\)",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"https?://\S+", "", candidate)
    candidate = candidate.replace("_", "")
    candidate = normalize_ordinals(candidate)
    candidate = re.sub(r"(?i)\bArt\.\s*(?=\d)", "Art. ", candidate)
    candidate = re.sub(r"\u00a7\s+", "\u00a7", candidate)
    candidate = re.sub(
        r"(\u00a7)([1-9])(?!\u00ba|\d)",
        lambda m: f"{m.group(1)}{m.group(2)}\u00ba",
        candidate,
    )
    candidate = re.sub(
        r"(?i)(Lei)\s+n[\s.\-]*(?:\u00ba|o)?\s*(?=\d)",
        lambda m: f"{m.group(1)} n\u00ba ",
        candidate,
    )
    candidate = re.sub(r"\s*\(\s*", " (", candidate)
    candidate = re.sub(r"\s+\)", ")", candidate)
    candidate = re.sub(r"\)\s*(?=\()", ") (", candidate)
    candidate = re.sub(r"^\s+\(", "(", candidate)
    candidate = re.sub(r"\(\s*\(", "(", candidate)
    candidate = re.sub(r"\)\s*\)", ")", candidate)
    candidate = re.sub(r"\s*>\s*", " ", candidate)
    candidate = re.sub(r"\s+([;:,])", r"\1", candidate)
    candidate = re.sub(r"([;:])(?=\S)", r"\1 ", candidate)
    candidate = re.sub(r"(,)(?=[^\s\d])", r"\1 ", candidate)
    candidate = re.sub(r"[ \t]+", " ", candidate)
    candidate = normalize_decimal_commas(candidate)
    candidate = fix_article_number_spacing(candidate)
    candidate = candidate.replace("\\", "")
    candidate = candidate.rstrip(",[]* ")
    candidate = collapse_duplicate_text(candidate)
    return candidate.strip()


def clean_multiline_text(text: str) -> str:
    lines = []
    for raw_line in normalize_space_characters(text).replace("\r", "").splitlines():
        cleaned = clean_text_candidate(raw_line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def clean_heading_part(text: str) -> str:
    cleaned = clean_multiline_text(text).replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.rstrip(".")


def save_current_item(rows: list[dict], title: str, text_lines: list[str]):
    cleaned_text = None
    title_normalized = clean_title(title) if title else title

    if title and text_lines:
        normalized_lines = []
        for segment in text_lines:
            if segment is None:
                continue
            for raw_line in normalize_space_characters(segment).replace("\r", "").split("\n"):
                candidate = clean_text_candidate(raw_line)
                if candidate:
                    normalized_lines.append(candidate)

        if not normalized_lines and title_normalized:
            normalized_lines.append(title_normalized.strip())

        if normalized_lines:
            cleaned_text = "\n".join(normalized_lines)
            rows.append({"title": title_normalized or title, "text": cleaned_text})
    return [], "", cleaned_text


def is_footer_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in FOOTER_PATTERNS)


def is_signature_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in SIGNATURE_PATTERNS)


def is_content_start_line(line: str) -> bool:
    return line.upper().startswith("RESOLVE:") or bool(re.match(r"^Art\.\s*1", line, re.IGNORECASE))


def normalize_annex_name(text: str) -> str:
    return clean_heading_part(text)


def annex_header_from_lines(lines: list[str]) -> str:
    for line in lines:
        if ANNEX_HEADER_PATTERN.match(line):
            return normalize_annex_name(line)
    return ""


def annex_header_from_table_rows(rows: list[list[str]]) -> str:
    for row in rows[:3]:
        for cell in row:
            cleaned_cell = clean_multiline_text(cell)
            if not cleaned_cell:
                continue
            first_line = cleaned_cell.splitlines()[0].strip()
            if first_line.upper().startswith("ANEXO"):
                return normalize_annex_name(cleaned_cell.replace("\n", " "))
    return ""


def is_all_capsish(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and all(char == char.upper() for char in letters)


def is_month_header(text: str) -> bool:
    return bool(MONTH_HEADER_PATTERN.match(clean_heading_part(text)))


def is_date_header(text: str) -> bool:
    text = clean_heading_part(text)
    return bool(
        re.match(
            r"^(?:\d{1,2}|1º)\s+de\s+[A-Za-zÀ-ÿç]+(?:[^\n]*)$",
            text,
            flags=re.IGNORECASE,
        )
    )


def marker_chain(marker: str) -> list[str]:
    parts = marker.split(".")
    return [".".join(parts[:idx]) for idx in range(1, len(parts) + 1)]


def join_title_parts(parts: list[str]) -> str:
    cleaned_parts = []
    for part in parts:
        cleaned = clean_heading_part(part)
        if cleaned:
            cleaned = re.sub(r"\s+\(", "(", cleaned)
            cleaned = re.sub(r"\)\s+", ")", cleaned)
            cleaned_parts.append(cleaned)
    return ", ".join(cleaned_parts)


def join_text_lines(lines: list[str]) -> str:
    cleaned_lines = []
    for line in lines:
        cleaned = clean_text_candidate(line)
        if cleaned:
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def append_row(rows: list[dict], title_parts: list[str], text_lines: list[str]) -> dict | None:
    title = join_title_parts(title_parts)
    text = join_text_lines(text_lines)
    if not title or not text:
        return None
    row = {"title": title, "text": text}
    rows.append(row)
    return row


def append_line_to_row(row: dict | None, line: str) -> None:
    if row is None:
        return
    existing = row.get("text", "")
    addition = clean_text_candidate(line)
    if not addition:
        return
    row["text"] = join_text_lines(existing.split("\n") + [addition])


def clean_table_rows(rows: list[list[str]]) -> list[list[str]]:
    cleaned_rows = []
    for row in rows:
        cleaned_row = [clean_multiline_text(cell) for cell in row]
        if any(cell.strip() for cell in cleaned_row):
            cleaned_rows.append(cleaned_row)
    return cleaned_rows


def parse_html_blocks(html_text: str) -> list[dict]:
    parser = HTMLBlockParser()
    parser.feed(html_text)
    parser.close()
    return parser.blocks


def split_body_and_annex_segments(blocks: list[dict]):
    started = False
    body_lines: list[str] = []
    annex_segments: list[dict] = []
    current_annex: dict | None = None

    for block in blocks:
        if block["type"] == "p":
            cleaned_lines = [
                cleaned
                for raw_line in normalize_space_characters(block["text"]).replace("\r", "").splitlines()
                for cleaned in [clean_text_candidate(raw_line)]
                if cleaned
            ]
            if not cleaned_lines:
                continue

            if not started:
                if any(is_content_start_line(line) for line in cleaned_lines):
                    started = True
                else:
                    continue

            if any(is_footer_line(line) for line in cleaned_lines):
                break

            annex_header = annex_header_from_lines(cleaned_lines)
            if annex_header:
                if current_annex is None or current_annex["name"] != annex_header:
                    current_annex = {"name": annex_header, "blocks": []}
                    annex_segments.append(current_annex)
                current_annex["blocks"].append({"type": "p", "text": "\n".join(cleaned_lines)})
                continue

            if current_annex is not None:
                current_annex["blocks"].append({"type": "p", "text": "\n".join(cleaned_lines)})
                continue

            body_lines.extend(cleaned_lines)
            continue

        cleaned_rows = clean_table_rows(block["rows"])
        if not cleaned_rows:
            continue
        if not started:
            continue

        annex_header = annex_header_from_table_rows(cleaned_rows)
        if annex_header:
            if current_annex is None or current_annex["name"] != annex_header:
                current_annex = {"name": annex_header, "blocks": []}
                annex_segments.append(current_annex)
            current_annex["blocks"].append({"type": "table", "rows": cleaned_rows})
            continue

        if current_annex is not None:
            current_annex["blocks"].append({"type": "table", "rows": cleaned_rows})
        else:
            for row in cleaned_rows:
                for cell in row:
                    if cell:
                        body_lines.extend(cell.splitlines())

    return body_lines, annex_segments


def parse_main_body_lines(lines: list[str]) -> list[dict]:
    art_pattern = re.compile(
        r"^(Art\.\s*\d+(?:\.\d+)*(?:[.\s]*(?:\u00ba|\u00aa|o))?(?:-[A-Z]+)?(?:\.\s*)?)(.*)"
    )
    par_pattern = re.compile(
        r"^("
        r"\u00a7\s*\d+(?:-[A-Z]+)?[\w.]*\u00ba?"
        r"|Par\u00e1grafo \u00fanico"
        r")"
        r"(?:\s*[-–—:]\s*|\.\s*)?"
        r"(.*)"
    )
    inc_pattern = re.compile(r"^([IVXLCDM]+)\s*[-\u2013\u2014]\s*(.*)")
    ali_pattern = re.compile(r"^([a-z])\)(.*)")
    num_pattern = re.compile(r"^(\d+)[\.\)\-]\s*(.*)")
    header_pattern = re.compile(
        r"^\s*(PARTE|LIVRO|T\u00cdTULO|CAP\u00cdTULO|Se\u00e7\u00e3o|Subse\u00e7\u00e3o|ANEXO)\s",
        re.IGNORECASE,
    )
    header_search_pattern = re.compile(
        r"\s*(PARTE|LIVRO|T\u00cdTULO|CAP\u00cdTULO|Se\u00e7\u00e3o|Subse\u00e7\u00e3o|ANEXO)\s",
        re.IGNORECASE,
    )
    inline_art_pattern = re.compile(
        r"Art\.\s*\d+(?:\.\d+)*(?:[.\s]*(?:\u00ba|\u00aa|o))?(?:-[A-Z]+)?(?:\.\s*)?"
    )
    inline_paragraph_split_pattern = re.compile(
        r"(?=(?:\u00a7\s*\d+(?:-[A-Z]+)?[\w.]*\u00ba?|Par\u00e1grafo \u00fanico)(?:\s*[-–—:]\s*|\s+))",
        re.IGNORECASE,
    )

    all_rows: list[dict] = []
    current_text_lines: list[str] = []
    current_title = ""
    context = {"art": "", "par": "", "inc": "", "ali": "", "num": ""}
    hierarchy_text = {"art": "", "par": "", "inc": "", "ali": "", "num": ""}
    current_level = None
    last_line_was_header_desc = False

    def get_parent_text(*levels):
        for level in levels:
            parent_text = hierarchy_text.get(level)
            if parent_text:
                return parent_text

        fallback_parts = []
        for level in ("art", "par", "inc", "ali", "num"):
            if level in levels and context.get(level):
                fallback_parts.append(context[level])
        return "\n".join(part for part in fallback_parts if part)

    for original in lines:
        if is_footer_line(original):
            break
        if is_signature_line(original):
            continue

        line = normalize_space_characters(original)
        original_line = line.strip()
        if not original_line:
            continue

        expanded_line = original_line
        expanded_line = re.sub(r";\s*(?=[IVXLCDM]+\s*[-–—])", ";\n", expanded_line)
        expanded_line = re.sub(r";\s*(?=[a-z]\))", ";\n", expanded_line)
        expanded_line = re.sub(r";\s*(?=\d+[\.\)\-])", ";\n", expanded_line)
        pending_lines = expanded_line.splitlines()

        while pending_lines:
            sub_line = pending_lines.pop(0)
            line_stripped = clean_text_candidate(sub_line)
            if not line_stripped or line_stripped.upper().startswith("RESOLVE:"):
                continue

            if ">" in line_stripped:
                normalized = re.sub(r"(?:\s*>\s*){2,}", "\n", line_stripped)
                if normalized != line_stripped:
                    split_parts = normalized.splitlines()
                    if split_parts:
                        pending_lines = split_parts + pending_lines
                        continue
                    line_stripped = normalized
                line_stripped = re.sub(r"\s*>\s*", " ", line_stripped)
                line_stripped = clean_text_candidate(line_stripped)

            if line_stripped:
                search_pos = 1
                while True:
                    split_match = inline_paragraph_split_pattern.search(line_stripped, search_pos)
                    if not split_match:
                        break
                    prefix = line_stripped[: split_match.start()]
                    trimmed_prefix = prefix.rstrip()
                    has_boundary = not trimmed_prefix or trimmed_prefix[-1] in ".;:–—-)]}"
                    if has_boundary:
                        before = prefix.rstrip(" -–—")
                        after = line_stripped[split_match.start() :].lstrip(" -–—")
                        if after:
                            pending_lines.insert(0, after)
                        line_stripped = before.strip()
                        search_pos = 1
                        if not line_stripped:
                            break
                    else:
                        search_pos = split_match.start() + 1
                if not line_stripped:
                    continue

            matches = list(inline_art_pattern.finditer(line_stripped))
            if matches and matches[0].start() == 0:
                segments = []
                last_pos = 0
                for idx, match in enumerate(matches):
                    if match.start() > last_pos:
                        segment = line_stripped[last_pos : match.start()].strip()
                        if segment:
                            segments.append(segment)
                    next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(line_stripped)
                    segment = line_stripped[match.start() : next_start].strip()
                    if segment:
                        segments.append(segment)
                    last_pos = next_start
                if len(segments) > 1 or (segments and segments[0] != line_stripped):
                    pending_lines = segments + pending_lines
                    continue
                if segments:
                    line_stripped = segments[0]

            art_match = art_pattern.match(line_stripped)
            par_match = par_pattern.match(line_stripped)
            inc_match = inc_pattern.match(line_stripped)
            ali_match = ali_pattern.match(line_stripped)
            num_match = num_pattern.match(line_stripped)
            header_match = header_pattern.match(line_stripped)

            is_all_caps_desc = line_stripped.isupper() and len(line_stripped.split()) < 10

            if art_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(
                    all_rows, current_title, current_text_lines
                )
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                art_title = clean_title(art_match.group(1))
                context = {"art": art_title, "par": "", "inc": "", "ali": "", "num": ""}
                current_level = "art"
                current_title = context["art"]
                current_text_lines = [line_stripped]
                hierarchy_text = {"art": "", "par": "", "inc": "", "ali": "", "num": ""}
                last_line_was_header_desc = False

            elif par_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(
                    all_rows, current_title, current_text_lines
                )
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                par_title = clean_title(par_match.group(1))
                context["par"] = par_title
                context["inc"] = ""
                context["ali"] = ""
                context["num"] = ""
                current_level = "par"
                current_title = f'{context["art"]}, {context["par"]}'
                parent_text = get_parent_text("art")
                current_text_lines = []
                if parent_text:
                    current_text_lines.append(parent_text)
                current_text_lines.append(f"\n{line_stripped}")
                hierarchy_text["inc"] = ""
                hierarchy_text["ali"] = ""
                hierarchy_text["num"] = ""
                last_line_was_header_desc = False

            elif inc_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(
                    all_rows, current_title, current_text_lines
                )
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                context["inc"] = inc_match.group(1).strip()
                context["ali"] = ""
                context["num"] = ""
                current_level = "inc"
                title_parts = [context["art"]]
                if context["par"]:
                    title_parts.append(context["par"])
                title_parts.append(context["inc"])
                current_title = ", ".join(title_parts)
                parent_text = get_parent_text("par", "art")
                current_text_lines = []
                if parent_text:
                    current_text_lines.append(parent_text)
                current_text_lines.append(f"\n{line_stripped}")
                hierarchy_text["ali"] = ""
                hierarchy_text["num"] = ""
                last_line_was_header_desc = False

            elif ali_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(
                    all_rows, current_title, current_text_lines
                )
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                context["ali"] = ali_match.group(1).strip() + ")"
                context["num"] = ""
                current_level = "ali"
                title_parts = [context["art"]]
                if context["par"]:
                    title_parts.append(context["par"])
                if context["inc"]:
                    title_parts.append(context["inc"])
                title_parts.append(context["ali"])
                current_title = ", ".join(title_parts)
                parent_text = get_parent_text("inc", "par", "art")
                current_text_lines = []
                if parent_text:
                    current_text_lines.append(parent_text)
                current_text_lines.append(f"\n{line_stripped}")
                hierarchy_text["num"] = ""
                last_line_was_header_desc = False

            elif num_match:
                current_text_lines, current_title, cleaned_parent = save_current_item(
                    all_rows, current_title, current_text_lines
                )
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                context["num"] = num_match.group(1).strip()
                current_level = "num"
                title_parts = [context["art"]]
                if context["par"]:
                    title_parts.append(context["par"])
                if context["inc"]:
                    title_parts.append(context["inc"])
                if context["ali"]:
                    title_parts.append(context["ali"])
                title_parts.append(context["num"])
                current_title = ", ".join(title_parts)
                parent_text = get_parent_text("ali", "inc", "par", "art")
                current_text_lines = []
                if parent_text:
                    current_text_lines.append(parent_text)
                current_text_lines.append(f"\n{line_stripped}")
                last_line_was_header_desc = False

            elif header_match or (last_line_was_header_desc and is_all_caps_desc):
                current_text_lines, current_title, cleaned_parent = save_current_item(
                    all_rows, current_title, current_text_lines
                )
                if cleaned_parent and current_level:
                    hierarchy_text[current_level] = cleaned_parent
                last_line_was_header_desc = True
                current_title = ""
                current_text_lines = []
                current_level = None

            else:
                if current_title:
                    header_in_line_match = header_search_pattern.search(line_stripped)
                    if header_in_line_match:
                        header_value = header_in_line_match.group(1)
                        if not (header_value.isupper() or header_value.istitle()):
                            header_in_line_match = None

                    if header_in_line_match:
                        text_before_header = line_stripped[: header_in_line_match.start()].strip()
                        if text_before_header:
                            current_text_lines.append(text_before_header)

                        current_text_lines, current_title, cleaned_parent = save_current_item(
                            all_rows, current_title, current_text_lines
                        )
                        if cleaned_parent and current_level:
                            hierarchy_text[current_level] = cleaned_parent
                        last_line_was_header_desc = True
                        current_title = ""
                        current_text_lines = []
                        current_level = None
                    else:
                        if current_text_lines:
                            current_text_lines[-1] = f"{current_text_lines[-1]} {line_stripped}"
                        else:
                            current_text_lines.append(line_stripped)
                        last_line_was_header_desc = False
                else:
                    last_line_was_header_desc = False

    _, _, cleaned_parent = save_current_item(all_rows, current_title, current_text_lines)
    if cleaned_parent and current_level:
        hierarchy_text[current_level] = cleaned_parent

    if all_rows:
        last_text = all_rows[-1]["text"]
        footer_match = re.search(
            r"(\b[A-Za-zÀ-ÿ][^,\n\d]+,\s+\d{1,2}\s+de\s+[A-Za-zÀ-ÿ]+\s+de\s+\d{4}.*$)",
            last_text,
            re.IGNORECASE | re.DOTALL,
        )
        if footer_match and "lei n" not in footer_match.group(1).lower():
            all_rows[-1]["text"] = last_text[: footer_match.start()].rstrip()

    return all_rows


def is_numeric_heading_like(rest: str) -> bool:
    rest = rest.strip()
    if not rest:
        return True
    head, _, tail = rest.partition(":")
    if not is_all_capsish(head):
        return False
    if tail.strip() and any(char.islower() for char in tail):
        return False
    return len(rest.split()) <= 18


def parse_numbered_annex_lines(
    lines: list[str],
    title_prefix: list[str],
    text_prefix: list[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    contexts: dict[str, str] = {}
    current_numeric_marker = ""
    last_row: dict | None = None
    prefix_lines = list(text_prefix or [])

    for raw_line in lines:
        line = clean_text_candidate(raw_line)
        if not line or is_signature_line(line) or is_footer_line(line):
            continue
        if ANNEX_HEADER_PATTERN.match(line):
            continue

        numeric_match = NUMERIC_ITEM_PATTERN.match(line)
        if numeric_match:
            marker, rest = numeric_match.groups()
            rest = rest.strip()
            full = f"{marker}. {rest}" if rest else f"{marker}."
            contexts[marker] = full
            current_numeric_marker = marker
            last_row = None

            if is_numeric_heading_like(rest):
                continue

            chain = marker_chain(marker)
            text_lines = list(prefix_lines)
            for parent in chain[:-1]:
                parent_text = contexts.get(parent)
                if parent_text:
                    text_lines.append(parent_text)
            text_lines.append(full)
            last_row = append_row(rows, title_prefix + chain, text_lines)
            continue

        roman_match = ROMAN_ITEM_PATTERN.match(line)
        if roman_match and current_numeric_marker:
            roman_marker, rest = roman_match.groups()
            full = f"{roman_marker} - {rest.strip()}"
            chain = marker_chain(current_numeric_marker)
            text_lines = list(prefix_lines)
            for parent in chain:
                parent_text = contexts.get(parent)
                if parent_text:
                    text_lines.append(parent_text)
            text_lines.append(full)
            last_row = append_row(rows, title_prefix + chain + [roman_marker], text_lines)
            continue

        if last_row is not None:
            append_line_to_row(last_row, line)
            continue

        if current_numeric_marker:
            chain = marker_chain(current_numeric_marker)
            text_lines = list(prefix_lines)
            for parent in chain[:-1]:
                parent_text = contexts.get(parent)
                if parent_text:
                    text_lines.append(parent_text)
            current_context = contexts.get(current_numeric_marker)
            if current_context:
                text_lines.append(current_context)
            text_lines.append(line)
            last_row = append_row(rows, title_prefix + chain, text_lines)

    return rows


def process_chronology_table_annex(segment: dict) -> list[dict]:
    rows: list[dict] = []
    annex_name = segment["name"]
    current_month = ""

    for block in segment["blocks"]:
        if block["type"] != "table":
            continue

        for row in block["rows"]:
            if not any(cell.strip() for cell in row):
                continue

            nonempty = [cell for cell in row if cell.strip()]
            if not nonempty:
                continue

            row_annex_header = annex_header_from_table_rows([row])
            if row_annex_header:
                annex_name = row_annex_header
                continue

            if len(nonempty) == 1 and is_month_header(nonempty[0]):
                current_month = clean_heading_part(nonempty[0])
                continue

            left = row[0].strip() if len(row) > 0 else ""
            right = row[1].strip() if len(row) > 1 else ""

            if is_month_header(left) and not right:
                current_month = clean_heading_part(left)
                continue

            if not left or not right:
                continue

            append_row(
                rows,
                [annex_name, current_month, left],
                [left, right],
            )

    return rows


def process_chronology_paragraph_annex(segment: dict) -> list[dict]:
    rows: list[dict] = []
    annex_name = segment["name"]
    current_month = ""
    current_date = ""
    current_date_row: dict | None = None
    current_number_marker = ""
    current_number_text = ""
    last_row: dict | None = None

    for block in segment["blocks"]:
        if block["type"] != "p":
            continue
        for raw_line in block["text"].splitlines():
            line = clean_text_candidate(raw_line)
            if not line or is_signature_line(line):
                continue
            if is_footer_line(line):
                return rows
            if ANNEX_HEADER_PATTERN.match(line):
                annex_name = normalize_annex_name(line)
                current_date = ""
                current_month = ""
                current_date_row = None
                current_number_marker = ""
                current_number_text = ""
                last_row = None
                continue
            if is_month_header(line):
                current_month = clean_heading_part(line)
                current_date = ""
                current_date_row = None
                current_number_marker = ""
                current_number_text = ""
                last_row = None
                continue
            if is_date_header(line):
                current_date = clean_heading_part(line)
                current_date_row = None
                current_number_marker = ""
                current_number_text = ""
                last_row = None
                continue

            numeric_match = NUMERIC_ITEM_PATTERN.match(line)
            if current_date and numeric_match:
                marker, rest = numeric_match.groups()
                full = f"{marker}. {rest.strip()}" if rest.strip() else f"{marker}."
                current_number_marker = marker
                current_number_text = full
                last_row = append_row(
                    rows,
                    [annex_name, current_month, current_date] + marker_chain(marker),
                    [current_date, full],
                )
                continue

            roman_match = ROMAN_ITEM_PATTERN.match(line)
            if current_date and roman_match and current_number_marker:
                roman_marker, rest = roman_match.groups()
                full = f"{roman_marker} - {rest.strip()}"
                last_row = append_row(
                    rows,
                    [annex_name, current_month, current_date] + marker_chain(current_number_marker) + [roman_marker],
                    [current_date, current_number_text, full],
                )
                continue

            if current_date:
                if last_row is not None and current_number_marker:
                    append_line_to_row(last_row, line)
                else:
                    if current_date_row is None:
                        current_date_row = append_row(
                            rows,
                            [annex_name, current_month, current_date],
                            [current_date, line],
                        )
                    else:
                        append_line_to_row(current_date_row, line)
                    last_row = current_date_row

    return rows


def canonical_label(text: str) -> str:
    label = clean_heading_part(text).upper().rstrip(":")
    return re.sub(r"\s+", "", label)


def display_label(text: str) -> str:
    label = clean_heading_part(text).rstrip(":")
    label = re.sub(r"\s*\(\s*", "(", label)
    label = re.sub(r"\s*\)\s*", ")", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip()


def is_side_heading(text: str) -> bool:
    return canonical_label(text) in {
        "VEDADA(O)",
        "VEDADO(A)",
        "VEDADO",
        "VEDADA",
        "PERMITIDA(O)",
        "PERMITIDO(A)",
        "PERMITIDO",
        "PERMITIDA",
        "PROIBIDA(O)",
        "PROIBIDO(A)",
        "PROIBIDA",
        "PROIBIDO",
        "PROIBIDAS",
        "PERMITIDAS",
    }


def is_binary_section_heading(text: str) -> bool:
    cleaned = clean_heading_part(text)
    return bool(cleaned) and is_all_capsish(cleaned) and not is_side_heading(cleaned) and len(cleaned.split()) <= 18


def extract_simple_marker(text: str) -> str:
    numeric_match = NUMERIC_ITEM_PATTERN.match(text)
    if numeric_match:
        return f"{numeric_match.group(1)}."
    roman_match = ROMAN_ITEM_PATTERN.match(text)
    if roman_match:
        return f"{roman_match.group(1)} -"
    return ""


def starts_with_marker(text: str) -> bool:
    return bool(NUMERIC_ITEM_PATTERN.match(text) or ROMAN_ITEM_PATTERN.match(text))


def process_binary_table_annex(segment: dict) -> list[dict]:
    rows: list[dict] = []
    annex_name = segment["name"]
    current_section = ""
    current_labels: list[str] = []
    lines_by_label: dict[str, list[str]] = {}

    def flush_section():
        nonlocal rows, lines_by_label
        if not current_section or not current_labels:
            lines_by_label = {}
            return
        for label in current_labels:
            label_lines = lines_by_label.get(label, [])
            if not label_lines:
                continue
            rows.extend(
                parse_numbered_annex_lines(
                    label_lines,
                    [annex_name, current_section, label],
                    [current_section, f"{label}:"],
                )
            )
        lines_by_label = {}

    for block in segment["blocks"]:
        if block["type"] != "table":
            continue
        for row in block["rows"]:
            nonempty = [cell for cell in row if cell.strip()]
            if not nonempty:
                continue

            if any(ANNEX_HEADER_PATTERN.match(cell) for cell in nonempty):
                annex_name = normalize_annex_name(nonempty[0])
                continue

            if len(nonempty) == 1 and is_binary_section_heading(nonempty[0]):
                flush_section()
                current_section = clean_heading_part(nonempty[0])
                current_labels = []
                continue

            left = row[0].strip() if len(row) > 0 else ""
            right = row[1].strip() if len(row) > 1 else ""

            if left and right and is_side_heading(left) and is_side_heading(right):
                flush_section()
                current_labels = [display_label(left), display_label(right)]
                lines_by_label = {label: [] for label in current_labels}
                continue

            if not current_labels:
                continue

            left_label = current_labels[0] if len(current_labels) > 0 else ""
            right_label = current_labels[1] if len(current_labels) > 1 else ""
            left_text = left
            right_text = right

            inherited_marker = extract_simple_marker(left_text)
            if right_text and inherited_marker and not starts_with_marker(right_text):
                right_text = f"{inherited_marker} {right_text}"

            if left_text and left_label:
                lines_by_label.setdefault(left_label, []).append(left_text)
            if right_text and right_label:
                lines_by_label.setdefault(right_label, []).append(right_text)

    flush_section()
    return rows


def process_generic_text_annex(segment: dict) -> list[dict]:
    lines: list[str] = []
    for block in segment["blocks"]:
        if block["type"] != "p":
            continue
        lines.extend(block["text"].splitlines())
    return parse_numbered_annex_lines(lines, [segment["name"]])


def segment_has_binary_table(segment: dict) -> bool:
    for block in segment["blocks"]:
        if block["type"] != "table":
            continue
        for row in block["rows"]:
            if len(row) >= 2:
                left = row[0].strip() if len(row) > 0 else ""
                right = row[1].strip() if len(row) > 1 else ""
                if left and right and is_side_heading(left) and is_side_heading(right):
                    return True
    return False


def segment_has_table(segment: dict) -> bool:
    return any(block["type"] == "table" for block in segment["blocks"])


def segment_looks_like_chronology_paragraph(segment: dict) -> bool:
    month_hits = 0
    date_hits = 0
    for block in segment["blocks"]:
        if block["type"] != "p":
            continue
        for line in block["text"].splitlines():
            cleaned = clean_text_candidate(line)
            if not cleaned:
                continue
            if is_month_header(cleaned):
                month_hits += 1
            elif is_date_header(cleaned):
                date_hits += 1
    return month_hits >= 1 and date_hits >= 2


def parse_annex_segment(segment: dict) -> list[dict]:
    if segment_has_binary_table(segment):
        return process_binary_table_annex(segment)
    if segment_has_table(segment):
        return process_chronology_table_annex(segment)
    if segment_looks_like_chronology_paragraph(segment):
        return process_chronology_paragraph_annex(segment)
    return process_generic_text_annex(segment)


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped = []
    for row in rows:
        key = (row["title"], row["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def sanitize_output_name(input_path: Path) -> str:
    base_name = input_path.stem
    base_name = re.sub(r"\.?\s*[—-]\s*Tribunal Superior Eleitoral$", "", base_name).strip()
    return f"{base_name}.csv"


def write_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        writer.writerow(["ID", "Nome (Artigo)", "Texto do Artigo"])
        for idx, row in enumerate(rows, start=1):
            writer.writerow([idx, row["title"], row["text"]])


def convert_html_to_csv(input_path: Path, output_dir: Path) -> tuple[Path, int]:
    html_text = input_path.read_text(encoding="utf-8", errors="ignore")
    blocks = parse_html_blocks(html_text)
    body_lines, annex_segments = split_body_and_annex_segments(blocks)

    all_rows = parse_main_body_lines(body_lines)
    for segment in annex_segments:
        all_rows.extend(parse_annex_segment(segment))

    all_rows = dedupe_rows(all_rows)
    output_path = output_dir / sanitize_output_name(input_path)
    write_csv(output_path, all_rows)
    return output_path, len(all_rows)


def discover_inputs(search_dir: Path) -> list[Path]:
    candidates = sorted(
        path
        for path in search_dir.glob("RESOLUÇÃO Nº *.html")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if candidates:
        return candidates
    return sorted(
        path
        for path in search_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.name.upper().startswith("RESOLUÇÃO Nº ")
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Converte HTMLs do portal do TSE em CSV compatível com o paradigma."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Arquivos HTML de entrada. Se omitido, faz autodiscovery no diretório atual.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Diretório de saída dos CSVs gerados. Padrão: diretório atual.",
    )
    return parser


def resolve_inputs(raw_inputs: list[str]) -> list[Path]:
    if raw_inputs:
        return [Path(raw).expanduser().resolve() for raw in raw_inputs]
    return discover_inputs(Path.cwd())


def main() -> int:
    args = build_arg_parser().parse_args()
    input_paths = resolve_inputs(args.inputs)
    if not input_paths:
        print("[ERRO] Nenhum HTML foi encontrado para conversão.")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    failures = 0

    print("--- Conversor Estruturado de HTML TSE para CSV (compatível com o paradigma) ---")
    for input_path in input_paths:
        if not input_path.exists():
            print(f"[ERRO] Arquivo não encontrado: {input_path}")
            failures += 1
            continue
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"[ERRO] Extensão não suportada: {input_path}")
            failures += 1
            continue

        try:
            output_path, count = convert_html_to_csv(input_path, output_dir)
            print(f"[SUCESSO] {input_path.name} -> {output_path.name} ({count} itens)")
        except Exception as exc:
            print(f"[ERRO] Falha ao converter '{input_path.name}': {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
