#!/usr/bin/env python3
"""
Extrai teses do STJ em PDF e gera CSV estruturado.

Fluxo de trabalho:
1. Lê o PDF com `pdfplumber` e reconstrói linhas/posicionamento por página.
2. Detecta ramo, edição, tema, tese, precedentes, referências e links associados.
3. Aplica limpeza de ruído e pós-processamento para unificar registros quebrados.
4. Expande múltiplos links em colunas `Site 1..N` e numera os itens com `ID`.
5. Escreve o CSV final com colunas jurídicas padronizadas.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from gui_intuitiva import open_file_panel

try:
    import pdfplumber
except ModuleNotFoundError:
    print("Este script depende do pacote 'pdfplumber'. Execute: pip install pdfplumber")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent

MONTHS = {
    "janeiro": "01",
    "fevereiro": "02",
    "marco": "03",
    "abril": "04",
    "maio": "05",
    "junho": "06",
    "julho": "07",
    "agosto": "08",
    "setembro": "09",
    "outubro": "10",
    "novembro": "11",
    "dezembro": "12",
}

COLUMN_HEADERS = ["Ramo", "Edição", "Tema", "Tese", "Precedentes", "Referências", "Data", "ID"]
ITEM_PATTERN = re.compile(r"^(\d{1,3})\s*[\.\-–]\s+(.*)")
ITEM_MAX_NUMBER = 200
THESIS_NOISE_PATTERNS = [
    re.compile(r'^TEMA\b', re.IGNORECASE),
    re.compile(r'^SCON\.STJ\.JUS\.BR', re.IGNORECASE),
    re.compile(r'^@?CNOT', re.IGNORECASE),
]
PRECEDENTES_PATTERN = re.compile(r"^Precedentes\s*[:\-–]\s*(.*)", re.IGNORECASE)
JULGADOS_PATTERN = re.compile(r"\bjulgad(?:o|os)\s*[:\-–]\s*(.+)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"(\d{1,2})\s+de\s+([a-z]+)\s+de\s+(\d{4})")
REFERENCE_KEYWORDS = ("vide ", "consulte ", "disponivel ", "legislacao aplicada ")
COLUMN_GAP = 60.0
LINE_TOLERANCE = 2.5
RAMO_MAX_LENGTH = 60
RAMO_CONTINUATION_TRAILING_WORDS = {"DE", "DA", "DO", "DOS", "DAS", "E"}
RAMO_SINGLE_WORD_CONTINUATIONS = {"ORIENTACOES"}
RAMO_CONTINUATION_PHRASES = {"E PROCESSUAL"}
RAMO_CONTINUATION_MAX_X_OFFSET = 120.0
THEME_CONTINUATION_TRAILING_WORDS = {
    "DE",
    "DA",
    "DO",
    "DOS",
    "DAS",
    "E",
    "SOBRE",
    "PARA",
}
THEME_CONTINUATION_MAX_X_OFFSET = 320.0
NOISE_PATTERNS = [
    re.compile(
        r"as teses apresentadas foram elaboradas.*?(?:\d{1,2}/\d{1,2}/\d{4})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"este periodico nao e um repositorio oficial de jurisprudencia\.?",
        re.IGNORECASE,
    ),
    re.compile(r"jurisprudencia em teses", re.IGNORECASE),
]
MERGE_SUFFIX_PATTERN = re.compile(
    r"(?:lei|decreto|portaria|art|artigo|inc|inciso|paragrafo|§)\s*(?:n\.?|unico)?\s*$",
    re.IGNORECASE,
)


@dataclass
class Line:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    page_number: int


@dataclass
class ReferenceEntry:
    text: str
    page_number: int
    x0: float
    x1: float
    top: float
    bottom: float

    def merge(self, line: Line) -> None:
        joined = f"{self.text} {line.text}".strip()
        self.text = normalize_whitespace(joined)
        self.x0 = min(self.x0, line.x0)
        self.x1 = max(self.x1, line.x1)
        self.top = min(self.top, line.top)
        self.bottom = max(self.bottom, line.bottom)


@dataclass
class LinkEntry:
    uri: str
    x0: float
    x1: float
    top: float
    bottom: float
    used: bool = False


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def remove_accents(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ASCII", "ignore").decode("ASCII")


def strip_noise_phrases(text: str) -> str:
    if not text:
        return ""
    result = text
    ascii_result = remove_accents(result).lower()
    for pattern in NOISE_PATTERNS:
        while True:
            match = pattern.search(ascii_result)
            if not match:
                break
            start, end = match.span()
            result = result[:start] + " " + result[end:]
            ascii_result = remove_accents(result).lower()
    return result




def is_noise_thesis_text(text: str) -> bool:
    if not text:
        return False
    normalized = remove_accents(text).strip()
    for pattern in THESIS_NOISE_PATTERNS:
        if pattern.match(normalized):
            return True
    return False

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = strip_noise_phrases(text)
    text = normalize_whitespace(text)
    text = re.sub(r"\s+([,;:.])", r"\1", text)

    def _space_after(match: re.Match[str]) -> str:
        punct = match.group(1)
        next_char = match.group(2)
        prev_char = match.string[match.start() - 1] if match.start() > 0 else ""
        if punct == "." and prev_char.isdigit() and next_char.isdigit():
            return punct + next_char
        return punct + " " + next_char

    text = re.sub(r"([,;:.])(\S)", _space_after, text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;")


def combine_csv_field(*values: str) -> str:
    items: List[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for part in re.split(r"\s*,\s*", value.strip()):
            cleaned = normalize_text(part)
            if not cleaned:
                continue
            key = remove_accents(cleaned).lower()
            if key not in seen:
                seen.add(key)
                items.append(cleaned)
    return ", ".join(items)


def combine_site_field(*values: str) -> str:
    items: List[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for part in re.split(r"\s*,\s*", value.strip()):
            cleaned = part.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                items.append(cleaned)
    return ", ".join(items)


PRECEDENT_NUMBER_PATTERN = re.compile(r"^\d+\)\s*")


def is_noise_precedent_entry(entry: str) -> bool:
    if not entry:
        return False
    normalized = normalize_whitespace(remove_accents(entry).lower())
    return normalized.startswith(
        "jurisprudencia em teses as teses apresentadas foram elaboradas"
    )


def split_precedent_entries(text: str) -> List[str]:
    if not text:
        return []
    lines: List[str] = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        stripped = PRECEDENT_NUMBER_PATTERN.sub("", raw_line.strip())
        if stripped:
            lines.append(stripped)
    combined = normalize_whitespace(" ".join(lines))
    if not combined:
        return []
    segments = [segment.strip() for segment in re.split(r";\s*", combined) if segment.strip()]
    results: List[str] = []
    for segment in segments:
        entry = segment if segment.endswith(";") else f"{segment};"
        if is_noise_precedent_entry(entry):
            continue
        results.append(entry)
    return results


def format_precedent_entries(entries: Iterable[str]) -> str:
    cleaned_entries: List[str] = []
    for entry in entries:
        normalized = normalize_whitespace(entry.rstrip())
        if not normalized:
            continue
        if is_noise_precedent_entry(normalized):
            continue
        if not normalized.endswith(";"):
            normalized = f"{normalized};"
        cleaned_entries.append(normalized)
    if not cleaned_entries:
        return ""
    return "\n".join(f"{idx}) {value}" for idx, value in enumerate(cleaned_entries, start=1))


def prepare_precedentes(precedentes_lines: List[str]) -> str:
    if not precedentes_lines:
        return ""
    combined = " ".join(precedentes_lines)
    combined = remove_inline_urls(combined)
    combined = normalize_whitespace(combined)
    if combined.lower().startswith("julgados:"):
        combined = combined[len("julgados:") :].strip()
    entries = split_precedent_entries(combined)
    return format_precedent_entries(entries)


def split_site_links(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in re.split(r",\s*", value) if part.strip()]


def decode_pdf_bytes(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
        return value.decode("latin-1", errors="ignore")
    return str(value)


def parse_pt_br_date(text: str) -> Optional[str]:
    plain = remove_accents(text).lower()
    match = DATE_PATTERN.search(plain)
    if not match:
        return None
    day = int(match.group(1))
    month_name = match.group(2)
    year = match.group(3)
    month = MONTHS.get(month_name)
    if not month:
        return None
    month_int = int(month)
    year_int = int(year)
    try:
        date(year_int, month_int, day)
    except ValueError:
        return None
    return f"{month_int}/{day}/{year_int}"


def remove_inline_urls(text: str) -> str:
    if not text:
        return text

    cleaned = text
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\bwww\.\S+", "", cleaned, flags=re.IGNORECASE)
    for domain in ("scon.stj.jus.br", "portal.stf.jus.br", "ww2.stj.jus.br"):
        domain_pattern = re.compile(rf"\s*{re.escape(domain)}\S*(?:\s+\d+/\d+)?", re.IGNORECASE)
        cleaned = domain_pattern.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;")


def extract_edition_label(text: str) -> Optional[str]:
    plain = remove_accents(text).lower()
    match = re.search(r"edicao\s*n[ºo.]?\s*([0-9]+(?:[-/][0-9a-z]+)?)", plain)
    if not match:
        return None
    number = match.group(1).upper()
    return f"Edição n. {number}"


def split_edition_and_theme(text: str) -> Optional[tuple[str, str]]:
    for separator in (":", " – ", " - ", " — "):
        if separator in text:
            left, right = text.split(separator, 1)
            label = extract_edition_label(left)
            if label:
                return label, normalize_whitespace(right)
    return None


def is_ramo_line(text: str) -> bool:
    normalized = remove_accents(text).upper().strip()
    if len(normalized) < 6 or len(normalized) > RAMO_MAX_LENGTH:
        return False
    if any(char.isdigit() for char in normalized):
        return False
    if ":" in normalized or "EDICAO" in normalized:
        return False
    if not (
        normalized.startswith("DIREITO")
        or normalized.startswith("DIREITOS")
        or normalized.startswith("ORIENTACOES")
    ):
        return False
    return re.fullmatch(r"[-A-Z /&]+", normalized) is not None


def should_expect_ramo_continuation(text: str) -> bool:
    normalized = remove_accents(text).upper().strip()
    if not normalized:
        return False
    if normalized in RAMO_SINGLE_WORD_CONTINUATIONS:
        return True
    tokens = normalized.split()
    if not tokens:
        return False
    last = tokens[-1]
    if last in RAMO_CONTINUATION_TRAILING_WORDS:
        return True
    if len(tokens) >= 2:
        last_two = " ".join(tokens[-2:])
        if last_two in RAMO_CONTINUATION_PHRASES:
            return True
    return False


def is_ramo_continuation_line(text: str, line: Line, previous: Optional[Line]) -> bool:
    normalized = remove_accents(text).upper().strip()
    if not normalized or len(normalized) > RAMO_MAX_LENGTH:
        return False
    if any(char.isdigit() for char in normalized):
        return False
    if ":" in normalized or "EDICAO" in normalized:
        return False
    if not re.fullmatch(r"[-A-Z /&]+", normalized):
        return False
    if previous:
        if line.page_number != previous.page_number:
            return False
        if abs(line.x0 - previous.x0) > RAMO_CONTINUATION_MAX_X_OFFSET:
            return False
    return True


def normalize_theme_segment(text: str) -> str:
    return normalize_whitespace(text)


def merge_theme_segments(segments: Iterable[str]) -> str:
    merged = ""
    for raw_segment in segments:
        segment = normalize_theme_segment(raw_segment)
        if not segment:
            continue
        if not merged:
            merged = segment
            continue
        trimmed = merged.rstrip()
        next_segment = segment.strip()
        if trimmed.endswith("-") and next_segment and not next_segment.startswith("-"):
            first_part, *rest = next_segment.split(" ", 1)
            merged = trimmed + first_part
            if rest:
                merged = f"{merged} {rest[0]}"
        else:
            merged = f"{merged} {next_segment}" if next_segment else merged
    merged = re.sub(r"(\s-)([A-Z0-9])", r"\1 \2", merged)
    return normalize_text(merged)


def should_expect_theme_continuation(theme: str) -> bool:
    normalized = remove_accents(theme).upper().strip()
    if not normalized:
        return False
    if normalized.endswith("-"):
        return True
    tokens = normalized.split()
    if not tokens:
        return False
    last = re.sub(r"[^A-Z0-9]+$", "", tokens[-1])
    if not last:
        return True
    if last in THEME_CONTINUATION_TRAILING_WORDS:
        return True
    return False


def is_theme_continuation_line(text: str, line: Line, previous: Optional[Line]) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    if any(char.islower() for char in cleaned if char.isalpha()):
        return False
    if ":" in cleaned:
        return False
    if previous:
        if line.page_number != previous.page_number:
            return False
        if abs(line.x0 - previous.x0) > THEME_CONTINUATION_MAX_X_OFFSET:
            return False
    return True


def harmonize_theme_by_edition(records: Iterable[Dict[str, str]]) -> None:
    edition_theme: Dict[str, str] = {}
    for record in records:
        edition = normalize_whitespace(record.get("Edição", ""))
        theme = normalize_text(record.get("Tema", ""))
        if not edition or not theme:
            continue
        current = edition_theme.get(edition)
        if not current or len(theme) > len(current):
            edition_theme[edition] = theme

    for record in records:
        edition = normalize_whitespace(record.get("Edição", ""))
        if edition and edition in edition_theme:
            record["Tema"] = edition_theme[edition]


def is_noise_line(text: str) -> bool:
    plain = remove_accents(text).lower().strip()
    if not plain:
        return True
    noise_prefixes = (
        "jurisprudencia em teses as teses apresentadas foram elaboradas",
        "este periodico nao e um repositorio oficial",
        "tribunal da cidadania",
        "superior tribunal de justica",
        "fonte:",
        "atualizada ate",
    )
    if plain in {"jurisprudencia em teses", "informacoes adicionais"}:
        return True
    return any(plain.startswith(prefix) for prefix in noise_prefixes)


def is_reference_line(text: str) -> bool:
    stripped = text.strip().lstrip("-•*–—·")
    plain = remove_accents(stripped).lower()
    while plain.startswith("("):
        plain = plain[1:].lstrip()
    plain = plain.rstrip(")").rstrip()
    return any(plain.startswith(keyword) for keyword in REFERENCE_KEYWORDS)


def clean_reference_segments(raw_texts: Iterable[str]) -> List[str]:
    results: List[str] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"(?:Vide|Consulte|Disponivel)\s+.*?(?=(?:\s+(?:Vide|Consulte|Disponivel)\b)|$)",
        re.IGNORECASE,
    )

    for text in raw_texts:
        cleaned = remove_inline_urls(text)
        cleaned = cleaned.replace("(", " ").replace(")", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        segments = []
        for match in pattern.finditer(cleaned):
            segment = match.group(0).strip()
            if segment:
                segments.append(segment)
        if not segments and any(remove_accents(cleaned).lower().startswith(keyword) for keyword in REFERENCE_KEYWORDS):
            segments = [cleaned]

        for segment in segments:
            raw_normalized = remove_accents(segment).lower()
            if "jurisprudencia em teses" in raw_normalized:
                continue
            segment = normalize_text(segment)
            if not segment:
                continue
            normalized = remove_accents(segment).lower()
            if "pesquisa pronta" in normalized:
                continue
            if "scon.stj.jus.br" in normalized:
                continue
            if normalized not in seen:
                seen.add(normalized)
                results.append(segment)
    return results


def extract_lines(page: pdfplumber.page.Page) -> List[Line]:
    # Agrupa palavras no mesmo alinhamento horizontal para reconstruir cada linha.
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    words.sort(key=lambda w: (float(w["top"]), float(w["x0"])))

    lines: List[Line] = []
    buffer: List[str] = []
    bbox: Optional[dict[str, float]] = None

    for word in words:
        text = normalize_whitespace(word.get("text", ""))
        if not text:
            continue
        top = float(word["top"])
        bottom = float(word["bottom"])
        x0 = float(word["x0"])
        x1 = float(word["x1"])

        if buffer:
            same_row = abs(top - bbox["top"]) <= LINE_TOLERANCE if bbox else False
            gap = x0 - (bbox["x1"] if bbox else x0)
            if not same_row or gap > COLUMN_GAP:
                line_text = normalize_whitespace(" ".join(buffer))
                if line_text:
                    lines.append(Line(line_text, bbox["x0"], bbox["x1"], bbox["top"], bbox["bottom"], page.page_number))
                buffer = []
                bbox = None

        if not buffer:
            buffer = [text]
            bbox = {"x0": x0, "x1": x1, "top": top, "bottom": bottom}
        else:
            buffer.append(text)
            bbox["x0"] = min(bbox["x0"], x0)
            bbox["x1"] = max(bbox["x1"], x1)
            bbox["top"] = min(bbox["top"], top)
            bbox["bottom"] = max(bbox["bottom"], bottom)

    if buffer and bbox:
        line_text = normalize_whitespace(" ".join(buffer))
        if line_text:
            lines.append(Line(line_text, bbox["x0"], bbox["x1"], bbox["top"], bbox["bottom"], page.page_number))

    return lines


def extract_links(page: pdfplumber.page.Page, *, debug: bool = False) -> List[LinkEntry]:
    links: List[LinkEntry] = []
    try:
        raw_links = getattr(page, "hyperlinks", []) or []
    except Exception as exc:
        if debug:
            print(f"[DEBUG] Falha ao obter hyperlinks na página {page.page_number}: {exc}")
        raw_links = []

    if not raw_links:
        try:
            raw_annots = page.objects.get("annots", []) or []
        except Exception as exc:
            raw_annots = []
            if debug:
                print(f"[DEBUG] Falha ao acessar anotações na página {page.page_number}: {exc}")
        else:
            height = float(page.height)
            for annot in raw_annots:
                action = (
                    annot.get("A")
                    or annot.get("a")
                    or annot.get("PA")
                    or annot.get("pa")
                    or {}
                )
                uri_value = (
                    action.get("URI")
                    or action.get("Uri")
                    or action.get("uri")
                    or action.get("F")
                    or action.get("f")
                )
                uri = decode_pdf_bytes(uri_value).strip() if uri_value else ""
                if not uri:
                    continue
                rect = annot.get("Rect") or annot.get("rect") or annot.get("BBox") or annot.get("bbox")
                if not rect or len(rect) < 4:
                    continue
                try:
                    x0, y0, x1, y1 = [float(v) for v in rect[:4]]
                except (TypeError, ValueError):
                    continue
                x0, x1 = min(x0, x1), max(x0, x1)
                y0, y1 = min(y0, y1), max(y0, y1)
                top = height - y1
                bottom = height - y0
                raw_links.append(
                    {
                        "uri": uri,
                        "x0": x0,
                        "x1": x1,
                        "top": top,
                        "bottom": bottom,
                    }
                )
            if debug and raw_annots and not raw_links:
                print(f"[DEBUG] Nenhum hyperlink válido encontrado via anotações na página {page.page_number}.")

    for raw in raw_links:
        uri = raw.get("uri") or raw.get("target") or raw.get("url")
        if not uri:
            continue
        uri = decode_pdf_bytes(uri).strip()
        if not uri:
            continue
        x0 = float(raw.get("x0", raw.get("left", raw.get("x", 0.0))))
        x1 = float(raw.get("x1", raw.get("right", raw.get("x0", 0.0))))
        top = float(raw.get("top", raw.get("y0", raw.get("y", 0.0))))
        bottom = float(raw.get("bottom", raw.get("y1", raw.get("y0", 0.0))))
        links.append(LinkEntry(uri=uri.strip(), x0=x0, x1=x1, top=top, bottom=bottom))
    return links


def overlaps(a0: float, a1: float, b0: float, b1: float, slack: float) -> bool:
    return min(a1, b1) - max(a0, b0) >= -slack


def find_links_for_reference(ref: ReferenceEntry, links: List[LinkEntry]) -> List[str]:
    urls: List[str] = []
    for link in links:
        if link.used:
            continue
        vertical_match = overlaps(ref.top, ref.bottom, link.top, link.bottom, slack=3.0)
        horizontal_match = overlaps(ref.x0, ref.x1, link.x0, link.x1, slack=5.0)
        if vertical_match and horizontal_match:
            link.used = True
            urls.append(link.uri)
    return urls


def should_merge_records(current: Dict[str, str], nxt: Dict[str, str]) -> bool:
    curr_tese = current.get("Tese", "").strip()
    next_tese = nxt.get("Tese", "").strip()
    if not curr_tese or not next_tese:
        return False
    if current.get("Precedentes", "").strip():
        return False
    if not nxt.get("Precedentes", "").strip():
        return False
    curr_ascii = remove_accents(curr_tese).lower()
    if MERGE_SUFFIX_PATTERN.search(curr_ascii):
        return True
    if curr_tese.rstrip().endswith(("/", "-")):
        return True
    if curr_ascii.endswith(" n.") or curr_ascii.endswith(" art."):
        return True
    if curr_ascii.startswith(("tema ", "vide ", "n. ", "nº ")):
        return True
    return False


def should_drop_record(record: Dict[str, str]) -> bool:
    tese_raw = record.get("Tese", "") or ""
    tese = strip_noise_phrases(tese_raw).strip()
    precedentes = strip_noise_phrases(record.get("Precedentes", "") or "").strip()
    referencias = strip_noise_phrases(record.get("Referências", "") or "").strip()
    if not tese and not precedentes and not referencias:
        return True
    if (
        is_noise_thesis_text(tese_raw)
        and not precedentes
        and not referencias
        and not (record.get("Site", "") or "").strip()
    ):
        return True
    return False


def post_process_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merge_count = 0
    i = 0
    while i < len(records) - 1:
        current = records[i]
        nxt = records[i + 1]
        if should_merge_records(current, nxt):
            merged_tese = f"{current.get('Tese', '')} {nxt.get('Tese', '')}"
            current["Tese"] = normalize_text(merged_tese)
            current_entries = split_precedent_entries(current.get("Precedentes", ""))
            next_entries = split_precedent_entries(nxt.get("Precedentes", ""))
            combined_precedentes = format_precedent_entries(current_entries + next_entries)
            current["Precedentes"] = combined_precedentes
            current["Referências"] = combine_csv_field(
                current.get("Referências", ""), nxt.get("Referências", "")
            )
            current["Site"] = combine_site_field(
                current.get("Site", ""), nxt.get("Site", "")
            )
            records.pop(i + 1)
            merge_count += 1
            continue
        if should_drop_record(current):
            records.pop(i)
            continue
        i += 1

    if records and should_drop_record(records[-1]):
        records.pop()

    for record in records:
        record["Tese"] = normalize_text(record.get("Tese", ""))
        prec_entries = split_precedent_entries(record.get("Precedentes", ""))
        record["Precedentes"] = format_precedent_entries(prec_entries) if prec_entries else ""
        record["Referências"] = combine_csv_field(record.get("Referências", ""))
        record["Site"] = combine_site_field(record.get("Site", ""))
    print(f"[DEBUG] merges executed: {merge_count}")
    return post_process_records(records)


def parse_pdf(pdf_path: Path, debug: bool) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    edition_dates: Dict[str, str] = {}
    page_links: Dict[int, List[LinkEntry]] = {}

    current_ramo = ""
    current_edition = ""
    current_tema = ""
    current_date = ""
    current_thesis_number: Optional[str] = None
    thesis_lines: List[str] = []
    precedentes_lines: List[str] = []
    referencias: List[ReferenceEntry] = []
    current_section: Optional[str] = None
    expected_ramo_continuation = False
    last_ramo_line: Optional[Line] = None
    current_theme_lines: List[str] = []
    expected_theme_continuation = False
    last_theme_line: Optional[Line] = None

    def finalize_current_thesis() -> None:
        nonlocal current_thesis_number, thesis_lines, precedentes_lines, referencias, current_section
        if not current_thesis_number or not thesis_lines:
            current_thesis_number = None
            thesis_lines = []
            precedentes_lines = []
            referencias = []
            current_section = None
            return

        raw_tese = " ".join(thesis_lines)
        tese = normalize_text(raw_tese)
        precedentes = prepare_precedentes(precedentes_lines)

        referencias_brutas: List[str] = []
        links_site: List[str] = []
        for referencia in referencias:
            texto = normalize_whitespace(referencia.text)
            if texto:
                referencias_brutas.append(texto)
            for uri in find_links_for_reference(referencia, page_links.get(referencia.page_number, [])):
                if uri not in links_site:
                    links_site.append(uri)

        referencias_textos = clean_reference_segments(referencias_brutas)
        links_site = [
            uri for uri in links_site if "pesquisa_pronta" not in uri.lower()
        ]

        data_registro = current_date or edition_dates.get(current_edition.upper(), "")

        if debug:
            print(
                f"[DEBUG] Referências brutas ({current_edition}): {referencias_brutas}"
            )
            print(
                f"[DEBUG] Referências limpas ({current_edition}): {referencias_textos}"
            )
            if links_site:
                print(f"[DEBUG] Links associados ({current_edition}): {links_site}")

        records.append(
            {
                "Ramo": current_ramo,
                "Edição": current_edition,
                "Tema": current_tema,
                "Tese": tese,
                "Precedentes": precedentes,
                "Referências": ", ".join(referencias_textos),
                "Site": ", ".join(links_site),
                "Data": data_registro,
            }
        )

        if debug:
            print(f"[DEBUG] Registro gerado: {current_ramo} | {current_edition} | {current_tema}")

        current_thesis_number = None
        thesis_lines = []
        precedentes_lines = []
        referencias = []
        current_section = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_links[page.page_number] = extract_links(page, debug=debug)
            for line in extract_lines(page):
                if not line.text:
                    continue

                texto = line.text.strip()
                if is_noise_line(texto):
                    continue

                if expected_theme_continuation and current_theme_lines:
                    if is_theme_continuation_line(texto, line, last_theme_line):
                        current_theme_lines.append(texto)
                        current_tema = merge_theme_segments(current_theme_lines)
                        last_theme_line = line
                        expected_theme_continuation = should_expect_theme_continuation(current_tema)
                        if debug:
                            print(
                                f"[DEBUG] Tema complementado: {current_edition} => {current_tema}"
                            )
                        continue
                    expected_theme_continuation = False

                if texto.isdigit():
                    continue

                if expected_ramo_continuation:
                    if is_ramo_continuation_line(texto, line, last_ramo_line):
                        current_ramo = normalize_whitespace(f"{current_ramo} {texto}")
                        last_ramo_line = line
                        expected_ramo_continuation = should_expect_ramo_continuation(current_ramo)
                        if debug:
                            print(
                                f"[DEBUG] Ramo complementado: {current_ramo} (página {line.page_number})"
                            )
                        continue
                    expected_ramo_continuation = False

                if is_ramo_line(texto):
                    finalize_current_thesis()
                    current_ramo = normalize_whitespace(texto)
                    current_edition = ""
                    current_tema = ""
                    current_date = ""
                    last_ramo_line = line
                    current_theme_lines = []
                    last_theme_line = None
                    expected_theme_continuation = False
                    expected_ramo_continuation = should_expect_ramo_continuation(current_ramo)
                    if debug:
                        print(f"[DEBUG] Novo ramo identificado: {current_ramo} (página {line.page_number})")
                    continue

                label_para_data = extract_edition_label(texto)
                data_extraida = parse_pt_br_date(texto)
                if label_para_data and data_extraida:
                    edition_dates[label_para_data.upper()] = data_extraida
                    current_date = data_extraida
                    continue

                edicao_tema = split_edition_and_theme(texto)
                if edicao_tema:
                    finalize_current_thesis()
                    current_edition, tema_inicial = edicao_tema
                    current_theme_lines = [tema_inicial]
                    current_tema = merge_theme_segments(current_theme_lines)
                    current_date = edition_dates.get(current_edition.upper(), current_date)
                    last_theme_line = line
                    expected_theme_continuation = should_expect_theme_continuation(current_tema)
                    if debug:
                        print(f"[DEBUG] Nova edição e tema: {current_edition} => {current_tema}")
                    continue

                item_match = ITEM_PATTERN.match(texto)
                if item_match:
                    try:
                        item_number = int(item_match.group(1))
                    except ValueError:
                        item_match = None
                    else:
                        if item_number > ITEM_MAX_NUMBER:
                            item_match = None
                if item_match and current_edition and current_tema:
                    finalize_current_thesis()
                    current_thesis_number = item_match.group(1)
                    tese_inicial = item_match.group(2).strip()
                    thesis_lines = [tese_inicial] if tese_inicial else []
                    precedentes_lines = []
                    referencias = []
                    current_section = "thesis"
                    continue

                if not current_thesis_number:
                    continue

                if current_section in (None, "thesis"):
                    julgados_match = JULGADOS_PATTERN.search(texto)
                    if julgados_match:
                        before = texto[: julgados_match.start()].strip()
                        if before:
                            thesis_lines.append(before)
                        conteudo = julgados_match.group(1).strip()
                        precedentes_lines = [conteudo] if conteudo else []
                        current_section = "precedentes"
                        continue

                precedentes_match = PRECEDENTES_PATTERN.match(texto)
                if precedentes_match:
                    current_section = "precedentes"
                    conteudo = precedentes_match.group(1).strip()
                    if conteudo:
                        if not precedentes_lines:
                            precedentes_lines = []
                        precedentes_lines.append(conteudo)
                    continue

                if (
                    is_reference_line(texto)
                    and current_thesis_number
                    and (current_section in ("precedentes", "references") or bool(precedentes_lines))
                ):
                    referencia = ReferenceEntry(
                        text=texto,
                        page_number=line.page_number,
                        x0=line.x0,
                        x1=line.x1,
                        top=line.top,
                        bottom=line.bottom,
                    )
                    referencias.append(referencia)
                    current_section = "references"
                    continue

                if current_section == "precedentes":
                    precedentes_lines.append(texto)
                    continue

                if current_section == "references":
                    if referencias:
                        referencias[-1].merge(line)
                    continue

                thesis_lines.append(texto)

        finalize_current_thesis()

    harmonize_theme_by_edition(records)
    records = [
        record
        for record in records
        if not (
            is_noise_thesis_text(record.get("Tese", ""))
            and not (record.get("Precedentes", "") or "").strip()
            and not (record.get("Referências", "") or "").strip()
            and not (record.get("Site", "") or "").strip()
        )
    ]
    return records


def write_csv(records: Iterable[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_list = list(records)
    max_links = 0
    for record in records_list:
        site_links = split_site_links(record.get("Site", ""))
        record["_site_links"] = site_links
        if len(site_links) > max_links:
            max_links = len(site_links)
    max_links = max(1, max_links)
    site_headers = [f"Site {i}" for i in range(1, max_links + 1)]
    fieldnames = ["Ramo", "Edição", "Tema", "Tese", "Precedentes", "Referências", *site_headers, "Data", "ID"]

    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(records_list, start=1):
            row = {
                "Ramo": record.get("Ramo", ""),
                "Edição": record.get("Edição", ""),
                "Tema": record.get("Tema", ""),
                "Tese": record.get("Tese", ""),
                "Precedentes": record.get("Precedentes", ""),
                "Referências": record.get("Referências", ""),
                "Data": record.get("Data", ""),
                "ID": str(idx),
            }
            links = record.get("_site_links", [])
            for col_index, header in enumerate(site_headers):
                row[header] = links[col_index] if col_index < len(links) else ""
            writer.writerow(row)


def resolve_input_path(path_value: Path | str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def force_output_path(path_value: Path | str) -> Path:
    name = Path(path_value).name
    if not name:
        name = "Jurisprudencia_em_teses.csv"
    return (SCRIPT_DIR / name).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai colunas específicas do PDF 'Jurisprudencia_em_teses.pdf' e gera um CSV estruturado."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=(SCRIPT_DIR / "Jurisprudencia_em_teses.pdf"),
        help="Caminho do PDF de origem.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=(SCRIPT_DIR / "Jurisprudencia_em_teses.csv"),
        help="Arquivo CSV de destino.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostra mensagens adicionais durante o processamento.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Desativa o painel GUI e usa apenas argumentos CLI.",
    )
    return parser.parse_args()


def maybe_collect_gui_inputs(args: argparse.Namespace) -> argparse.Namespace:
    cli_tokens = list(sys.argv[1:])
    pdf_explicit = any(tok == "--pdf" or tok.startswith("--pdf=") for tok in cli_tokens)
    if args.no_gui or pdf_explicit:
        return args

    current_output = force_output_path(args.output)
    initial_output_dir = str(SCRIPT_DIR)
    gui = open_file_panel(
        title="Jurisprudencia em Teses STJ - PDF para CSV",
        subtitle="Selecione o PDF de entrada e configure a saida.",
        filetypes=[("PDF", "*.pdf"), ("Todos os arquivos", "*.*")],
        extensions=[".pdf"],
        initial_files=[str(args.pdf)] if args.pdf else [],
        allow_add_dir=True,
        recursive_dir=True,
        min_files=1,
        output_label="Pasta de saida",
        initial_output=initial_output_dir,
        extra_bools=[("debug", "Ativar logs de debug", bool(args.debug))],
        extra_texts=[("output_name", "Nome do CSV de saida", current_output.name or "jurisprudencia_em_teses.csv")],
    )
    if not gui or not gui.get("confirmed"):
        raise SystemExit("Execucao cancelada no painel GUI.")

    files = list(gui.get("files") or [])
    if not files:
        raise SystemExit("Nenhum PDF selecionado no painel GUI.")

    output_name = str((gui.get("texts") or {}).get("output_name", "")).strip() or current_output.name
    args.pdf = resolve_input_path(files[0])
    args.output = force_output_path(output_name)
    args.debug = bool((gui.get("bools") or {}).get("debug", args.debug))
    return args


def main() -> None:
    args = parse_args()
    args = maybe_collect_gui_inputs(args)
    pdf_path = resolve_input_path(args.pdf)

    if not pdf_path.exists():
        print(f"Arquivo PDF não encontrado: {pdf_path}")
        sys.exit(1)

    registros = parse_pdf(pdf_path, debug=args.debug)
    if not registros:
        print("Nenhuma tese foi identificada. Use --debug para investigar o formato do PDF.")
        sys.exit(1)

    output_path = force_output_path(args.output)
    write_csv(registros, output_path)
    print(f"CSV gerado em: {output_path}")
    print(f"Total de teses exportadas: {len(registros)}")


if __name__ == "__main__":
    main()
