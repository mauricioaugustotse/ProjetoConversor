#!/usr/bin/env python3
"""Versao v2 do conversor SJUR -> Notion-friendly.

Mantem a interface publica da v1 e substitui o motor de extracao de
`partes`/`advogados` por um parser de cabecalho orientado a labels.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import Artefatos.legado.SJUR_csv_to_csv_NOTIONfriendly as _base
from Artefatos.legado.SJUR_csv_to_csv_NOTIONfriendly import *  # noqa: F401,F403


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_ROOT = (SCRIPT_DIR / "Artefatos").resolve()

_CONNECTOR_WORDS = {
    "a",
    "as",
    "da",
    "das",
    "de",
    "del",
    "di",
    "do",
    "dos",
    "d",
    "e",
    "la",
    "o",
    "os",
    "van",
    "von",
}
_INSTITUTION_HINT_RE = re.compile(
    r"(?i)\b(?:partido|coliga[cç][aã]o|federa[cç][aã]o|minist[eé]rio|procuradoria|defensoria|"
    r"tribunal|ju[ií]zo|advocacia|prefeitura|c[aâ]mara|secretaria|associa[cç][aã]o|instituto|"
    r"funda[cç][aã]o|cooperativa|empresa|sociedade|uni[aã]o(?:\s+brasil)?)\b"
)
_BUSINESS_HINT_RE = re.compile(
    r"(?i)\b(?:ltda|eireli|me|epp|marketing|pesquisas?|mercado|com[eé]rcio|ind[uú]stria)\b"
)
_OTHER_SUFFIX_RE = re.compile(r"(?is)\s+e\s+outr[oa]s?\b.*$")
_OTHER_ONLY_RE = re.compile(r"(?i)^e\s+outr[oa]s?$")
_TRAILING_EMPTY_PARENS_RE = re.compile(r"\s*[\(\[]+\s*$")
_UF_REG_TRAILING_RE = re.compile(r"\s*(?:[-–—]\s*)?[A-Z]{2}\d{3,}(?:-[A-Z])?\s*$")
_OAB_NUMBER_PATTERN = r"[\d./]+(?:[-–—][A-Z])?(?:/[A-Z]{2})?"
_OAB_TAIL_PATTERN = (
    rf"(?:[-–—]\s*)?OAB(?:/[A-Z]{{2}})?(?:\s*[-–—]\s*|\s*:?\s*){_OAB_NUMBER_PATTERN}"
    r"(?:\s+e\s+outr[oa]s?)?"
)
_OAB_PAREN_RE = re.compile(r"\(\s*OAB[^)]*\)", re.IGNORECASE)
_OAB_INLINE_RE = re.compile(rf"\s*{_OAB_TAIL_PATTERN}\s*$", re.IGNORECASE)
_PUNCT_SPAN_RE = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ]+")
_PERSON_NAME_PATTERN = (
    r"[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]*"
    r"(?:\s+(?:[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]*|da|de|do|das|dos|e|d|del|la|van|von|di)){1,12}"
)
_ATTORNEY_REG_PATTERN = (
    rf"(?:\(\s*OAB[^)]*\)"
    rf"|{_OAB_TAIL_PATTERN}"
    r"|[-–—]\s*[A-Z]{2}\d{3,}(?:-[A-Z])?)"
)
_ATTORNEY_WITH_REG_RE = re.compile(
    rf"(?P<name>{_PERSON_NAME_PATTERN})\s*(?P<reg>{_ATTORNEY_REG_PATTERN})",
    re.IGNORECASE | re.UNICODE,
)
_PARTY_ROLE_PATTERN = (
    r"(?:recorrentes?|recorrid[oa]s?|agravantes?|agravad[oa]s?|impetrantes?|impetrad[oa]s?|"
    r"requerentes?|requerid[oa]s?|exequentes?|executad[oa]s?|embargantes?|embargad[oa]s?|"
    r"apelantes?|apelad[oa]s?|autor(?:es)?|r[eé]us?|interessad[oa]s?|representantes?|"
    r"representad[oa]s?|reclamantes?|reclamad[oa]s?|pacientes?|impugnantes?|impugnad[oa]s?|"
    r"noticiantes?|noticiad[oa]s?|investigantes?|investigad[oa]s?|org[aã]o\s+coator|"
    r"autoridade\s+coator[ao])"
)
_ATTORNEY_LABEL_PATTERN = (
    rf"(?:advogad(?:o|a|os|as)|representantes?\s+(?:do\s*\(a\)|do|da|dos|das)\s+{_PARTY_ROLE_PATTERN})"
)
HEADER_LABEL_RE = re.compile(
    rf"(?i)\b(?P<label>{_ATTORNEY_LABEL_PATTERN}|{_PARTY_ROLE_PATTERN})\s*:"
)
_ROLE_NOISE_RE = re.compile(
    rf"(?i)\b(?:{_ATTORNEY_LABEL_PATTERN}|{_PARTY_ROLE_PATTERN}|relator(?:a)?|composi[çc][aã]o)\b"
)
_HEADER_DECISION_NOISE_RE = re.compile(
    r"(?i)\b(?:decis[aã]o|ac[oó]rd[aã]o|ementa|despacho|ante\s+o\s+exposto|"
    r"publique-?se|intimem-?se|julgamento\s+conjunto)\b"
)


def _sync_base_state() -> None:
    _base.ARTIFACTS_ROOT = Path(ARTIFACTS_ROOT)
    _base.extract_header_metadata = extract_header_metadata
    _base.extract_partes_multiselect = extract_partes_multiselect
    _base.extract_advogados_multiselect = extract_advogados_multiselect
    _base.sanitize_advogados_multiselect = sanitize_advogados_multiselect
    _base.merge_multiselect_values = merge_multiselect_values
    _base.HEADER_LABEL_RE = HEADER_LABEL_RE
    if "buscar_todas_noticias_async" in globals():
        _base.buscar_todas_noticias_async = globals()["buscar_todas_noticias_async"]


def _compare_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = _PUNCT_SPAN_RE.sub(" ", normalized)
    return _base.SPACE_RE.sub(" ", normalized).strip()


def _meaningful_tokens(value: str) -> list[str]:
    return [token for token in _compare_text(value).split() if token and token not in _CONNECTOR_WORDS]


def _entity_score(value: str) -> tuple[int, int]:
    return (len(_meaningful_tokens(value)), len(_compare_text(value)))


def _trim_trailing_punctuation(value: str) -> str:
    cleaned = _base.SPACE_RE.sub(" ", str(value or "")).strip(" ,;.-")
    cleaned = _TRAILING_EMPTY_PARENS_RE.sub("", cleaned).strip(" ,;.-")
    return cleaned


def _clean_party_candidate(value: str) -> str:
    cleaned = _base.SPACE_RE.sub(" ", str(value or "")).strip(" ,;.-")
    if not cleaned:
        return ""
    cleaned = _ROLE_NOISE_RE.split(cleaned)[0]
    cleaned = _OTHER_SUFFIX_RE.sub("", cleaned)
    if _OTHER_ONLY_RE.fullmatch(cleaned.strip()):
        return ""
    cleaned = _OAB_PAREN_RE.sub("", cleaned)
    cleaned = _OAB_INLINE_RE.sub("", cleaned)
    cleaned = _UF_REG_TRAILING_RE.sub("", cleaned)
    cleaned = _base.ADV_TITLES_PREFIX_RE.sub("", cleaned)
    return _trim_trailing_punctuation(cleaned)


def _clean_attorney_candidate(value: str) -> str:
    cleaned = _base.SPACE_RE.sub(" ", str(value or "")).strip(" ,;.-")
    if not cleaned:
        return ""
    cleaned = _OTHER_SUFFIX_RE.sub("", cleaned)
    if _OTHER_ONLY_RE.fullmatch(cleaned.strip()):
        return ""
    cleaned = _OAB_PAREN_RE.sub("", cleaned)
    cleaned = _OAB_INLINE_RE.sub("", cleaned)
    cleaned = _UF_REG_TRAILING_RE.sub("", cleaned)
    cleaned = _ROLE_NOISE_RE.split(cleaned)[0]
    cleaned = _base.ADV_TITLES_PREFIX_RE.sub("", cleaned)
    return _trim_trailing_punctuation(cleaned)


def _is_institutional_text(value: str) -> bool:
    return bool(_INSTITUTION_HINT_RE.search(value) or _BUSINESS_HINT_RE.search(value))


def _is_likely_person_name(value: str, *, min_tokens: int) -> bool:
    if _is_institutional_text(value):
        return False
    if len(_meaningful_tokens(value)) < min_tokens:
        return False
    return _base._is_valid_advogado_entity(value, include_institutional_entities=False)


def _split_on_safe_conjunction(value: str, *, min_person_tokens: int) -> list[str]:
    if " e " not in _compare_text(value):
        return [value]
    if _is_institutional_text(value) or "(" in value or ")" in value:
        return [value]
    parts = [part.strip(" ,;.-") for part in re.split(r"(?i)\s+e\s+", value) if part.strip(" ,;.-")]
    if len(parts) != 2:
        return [value]
    if all(_is_likely_person_name(part, min_tokens=min_person_tokens) for part in parts):
        return parts
    return [value]


def _is_valid_party_entity(entity: str, config: MetadataExtractionConfig) -> bool:
    cleaned = _clean_party_candidate(entity)
    if not cleaned:
        return False
    if _HEADER_DECISION_NOISE_RE.search(cleaned):
        return False
    if re.search(r"(?i)\badvogad|\brepresentantes?\s+(?:do|da|dos|das)\b", cleaned):
        return False
    if re.search(r"(?i)\bOAB\b", cleaned):
        return False
    if len(cleaned.split()) > max(2, int(config.max_entity_words)):
        return False

    meaningful_tokens = _meaningful_tokens(cleaned)
    if _is_institutional_text(cleaned):
        return config.include_institutional_entities and len(meaningful_tokens) >= 1 and len(cleaned.split()) >= 2
    return len(meaningful_tokens) >= 2


def _parse_party_block(block: str, config: MetadataExtractionConfig) -> list[str]:
    normalized = _base.SPACE_RE.sub(" ", block or "").strip(" ,;.-")
    if not normalized:
        return []

    entities: list[str] = []
    for chunk in re.split(r"\s*;\s*|\s*,\s*", normalized):
        candidate = _clean_party_candidate(chunk)
        if not candidate:
            continue
        for piece in _split_on_safe_conjunction(candidate, min_person_tokens=3):
            final_candidate = _clean_party_candidate(piece)
            if not _is_valid_party_entity(final_candidate, config):
                continue
            entities.append(final_candidate)
    return _semantic_dedupe_entities(entities, prefer_longer=True)


def _is_valid_attorney_entity(entity: str, config: MetadataExtractionConfig) -> bool:
    cleaned = _clean_attorney_candidate(entity)
    if not cleaned:
        return False
    if re.search(r"(?i)\b(?:partido|coliga[cç][aã]o|federa[cç][aã]o)\b", cleaned):
        return False
    if _is_institutional_text(cleaned):
        if not config.include_institutional_entities:
            return False
        if not re.search(r"(?i)\b(?:procuradoria|defensoria|advocacia|minist[eé]rio\s+p[úu]blico)\b", cleaned):
            return False
    return _base._is_valid_advogado_entity(
        cleaned,
        include_institutional_entities=config.include_institutional_entities,
    )


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            pieces.append(text[cursor:start])
        pieces.append(", ")
        cursor = max(cursor, end)
    if cursor < len(text):
        pieces.append(text[cursor:])
    return "".join(pieces)


def _parse_advogado_block(block: str, config: MetadataExtractionConfig) -> list[str]:
    normalized = _base.SPACE_RE.sub(" ", block or "").strip(" ,;.-")
    if not normalized:
        return []

    advogados: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in _ATTORNEY_WITH_REG_RE.finditer(normalized):
        candidate = _clean_attorney_candidate(match.group("name"))
        if not _is_valid_attorney_entity(candidate, config):
            continue
        advogados.append(candidate)
        spans.append(match.span())

    remainder = _remove_spans(normalized, spans)
    for chunk in re.split(r"\s*;\s*|\s*,\s*", remainder):
        candidate = _clean_attorney_candidate(chunk)
        if not candidate:
            continue
        for piece in _split_on_safe_conjunction(candidate, min_person_tokens=3):
            final_candidate = _clean_attorney_candidate(piece)
            if not _is_valid_attorney_entity(final_candidate, config):
                continue
            advogados.append(final_candidate)

    return _semantic_dedupe_entities(advogados, prefer_longer=True)


def _iter_header_blocks(text: str) -> list[tuple[str, str]]:
    labels = list(HEADER_LABEL_RE.finditer(text))
    if not labels:
        return []

    blocks: list[tuple[str, str]] = []
    for index, label_match in enumerate(labels):
        start = label_match.end()
        end = labels[index + 1].start() if index + 1 < len(labels) else len(text)
        label = _base.SPACE_RE.sub(" ", label_match.group("label")).strip()
        block = text[start:end]
        blocks.append((label, block))
    return blocks


def _is_attorney_label(label: str) -> bool:
    label_norm = _base.normalize_for_match(label)
    return label_norm.startswith("advogad") or label_norm.startswith("representante")


def _is_fragment_of(candidate: str, existing: str) -> bool:
    candidate_cmp = _compare_text(candidate)
    existing_cmp = _compare_text(existing)
    if not candidate_cmp or candidate_cmp == existing_cmp:
        return False
    if len(candidate_cmp) >= len(existing_cmp):
        return False

    candidate_tokens = _meaningful_tokens(candidate)
    existing_tokens = _meaningful_tokens(existing)
    if len(candidate_tokens) < 2 or len(existing_tokens) < len(candidate_tokens) + 1:
        return False

    if f" {candidate_cmp} " in f" {existing_cmp} ":
        token_gap = len(existing_tokens) - len(candidate_tokens)
        return token_gap >= 2 or _is_institutional_text(candidate) or _is_institutional_text(existing)
    return False


def _semantic_dedupe_entities(items: Iterable[str], *, prefer_longer: bool) -> list[str]:
    merged: list[str] = []
    for item in items:
        candidate = _trim_trailing_punctuation(item)
        if not candidate:
            continue
        candidate_key = _base.normalize_for_match(candidate)
        replaced = False
        skip = False

        for index, existing in enumerate(merged):
            existing_key = _base.normalize_for_match(existing)
            if candidate_key == existing_key:
                if prefer_longer and _entity_score(candidate) > _entity_score(existing):
                    merged[index] = candidate
                skip = True
                break
            if _is_fragment_of(candidate, existing):
                skip = True
                break
            if prefer_longer and _is_fragment_of(existing, candidate):
                merged[index] = candidate
                replaced = True
                skip = True
                break

        if replaced or skip:
            continue
        merged.append(candidate)
    return merged


def sanitize_advogados_multiselect(value: str, config: MetadataExtractionConfig) -> str:
    if not value:
        return ""
    cleaned_items = _parse_advogado_block(value, config)
    if not cleaned_items:
        fallback_items: list[str] = []
        for item in _base.split_multiselect_values(value):
            candidate = _clean_attorney_candidate(item)
            if not _is_valid_attorney_entity(candidate, config):
                continue
            fallback_items.append(candidate)
        cleaned_items = _semantic_dedupe_entities(fallback_items, prefer_longer=True)
    return ", ".join(cleaned_items)


def extract_header_metadata(
    *texts: str,
    config: Optional[MetadataExtractionConfig] = None,
) -> dict[str, list[str]]:
    metadata_config = config or MetadataExtractionConfig()
    partes: list[str] = []
    advogados: list[str] = []

    for text in texts:
        header_text = _base._prepare_metadata_header_text(text, max_chars=metadata_config.header_max_chars)
        if not header_text:
            continue
        for label, block in _iter_header_blocks(header_text):
            if _is_attorney_label(label):
                advogados.extend(_parse_advogado_block(block, metadata_config))
            else:
                partes.extend(_parse_party_block(block, metadata_config))

    return {
        "partes": _semantic_dedupe_entities(partes, prefer_longer=True),
        "advogados": _semantic_dedupe_entities(advogados, prefer_longer=True),
    }


def extract_partes_multiselect(*texts: str, config: Optional[MetadataExtractionConfig] = None) -> str:
    metadata = extract_header_metadata(*texts, config=config)
    return ", ".join(metadata.get("partes", []))


def extract_advogados_multiselect(*texts: str, config: Optional[MetadataExtractionConfig] = None) -> str:
    metadata = extract_header_metadata(*texts, config=config)
    return ", ".join(metadata.get("advogados", []))


def merge_multiselect_values(*values: str) -> str:
    merged: list[str] = []
    for value in values:
        candidates = _semantic_dedupe_entities(_base.split_multiselect_values(value), prefer_longer=True)
        for candidate in candidates:
            if any(
                _base.normalize_for_match(candidate) == _base.normalize_for_match(existing)
                or _is_fragment_of(candidate, existing)
                or _is_fragment_of(existing, candidate)
                for existing in merged
            ):
                continue
            merged.append(candidate)
    return ", ".join(merged)


def is_editorially_weak_news_url(url: str) -> bool:
    return _base.is_editorially_weak_news_url(url)


def official_news_url_live_status(url: str, *, timeout: int = 12) -> bool | None:
    return _base.official_news_url_live_status(url, timeout=timeout)


def _normalize_tse_news_url(value: object) -> str | None:
    return _base._normalize_tse_news_url(value)


def _normalize_tre_news_url(value: object) -> str | None:
    return _base._normalize_tre_news_url(value)


def _is_tse_news_url(url: str) -> bool:
    return _base._is_tse_news_url(url)


def _is_tre_news_url(url: str) -> bool:
    return _base._is_tre_news_url(url)


def _is_general_media_url(url: str) -> bool:
    return _base._is_general_media_url(url)


def resolve_artifacts_dir(*parts: str) -> Path:
    _sync_base_state()
    return _base.resolve_artifacts_dir(*parts)


def resolve_checkpoint_artifact_path(output_path: Path) -> Path:
    _sync_base_state()
    return _base.resolve_checkpoint_artifact_path(output_path)


def resolve_report_artifact_path(output_path: Path) -> Path:
    _sync_base_state()
    return _base.resolve_report_artifact_path(output_path)


def resolve_backup_artifacts_dir() -> Path:
    _sync_base_state()
    return _base.resolve_backup_artifacts_dir()


def resolve_web_lookup_cache_path(base_dir: Optional[Path] = None) -> Path:
    _sync_base_state()
    return _base.resolve_web_lookup_cache_path(base_dir=base_dir)


def resolve_intermediate_csv_dir() -> Path:
    _sync_base_state()
    return _base.resolve_intermediate_csv_dir()


def process_one_csv(*args, **kwargs):
    _sync_base_state()
    return _base.process_one_csv(*args, **kwargs)


def run_batch(*args, **kwargs):
    _sync_base_state()
    return _base.run_batch(*args, **kwargs)


def build_parser():
    _sync_base_state()
    return _base.build_parser()


def launch_gui():
    _sync_base_state()
    return _base.launch_gui()


def main() -> int:
    _sync_base_state()
    return _base.main()


_sync_base_state()


if __name__ == "__main__":
    raise SystemExit(main())
