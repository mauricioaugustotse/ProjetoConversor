#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoria + correção + enriquecimento de links para o ramo:
"Apuração de votos e eleições extraordinárias"

Saídas:
- CSV final com coluna `link_de_acesso` (arquivo novo, versionado por data)
- Relatórios de auditoria:
  - <prefix>_linhas_sem_match.csv
  - <prefix>_linhas_ambiguas.csv
  - <prefix>_omissoes_html_nao_no_csv.csv
  - <prefix>_csv_nao_no_html.csv
  - <prefix>_resumo.txt
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


TARGET_RAMO = "Apuração de votos e eleições extraordinárias"
MIN_CONTAINMENT_LEN = 40


# Padrão de siglas/tipos usados para reconhecer citação.
CIT_TOKEN = (
    r"(?:Ac\.|Res\.|EDcl|Embargos?|AgR(?:-[A-Za-z]+)?|MS|REspE?|REspe|REsp|"
    r"RO-?El|RMS|RCEd|AE|TutCautAnt|MC|AAg|Rec\.?)"
)

ANCHOR_TAG_RE = re.compile(
    r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL
)
HAS_CITATION_RE = re.compile(CIT_TOKEN, re.IGNORECASE)
DATE_IN_TEXT_RE = re.compile(r"\bde\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.IGNORECASE)
RELATOR_RE = re.compile(
    r"\brel\.\s*(?:Min\.|Des\.|Juiz(?:a)?|in\.)?\s*([^\).;]+)",
    re.IGNORECASE,
)

# Citações dentro de parênteses contendo "rel."
ROW_CITATION_PAREN_RE = re.compile(
    rf"\({CIT_TOKEN}[\s\S]{{0,360}}?rel\.[\s\S]{{0,180}}?\)",
    re.IGNORECASE,
)

# Fallback para citações truncadas/sem fechamento de parênteses
ROW_CITATION_FALLBACK_RE = re.compile(
    rf"{CIT_TOKEN}[\s\S]{{0,260}}?rel\.[^\n\.]{{0,140}}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HtmlCitation:
    href: str
    texto_citacao_raw: str
    texto_citacao_norm: str
    numero_norm: str
    relator_norm: str
    data_norm_mdy: str


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_anchor_inner_html(inner_html: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", inner_html or "")
    unescaped = html.unescape(without_tags)
    return normalize_ws(unescaped.replace("\xa0", " "))


def normalize_citation_text(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("−", "-")
    cleaned = strip_accents(cleaned.lower())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return normalize_ws(cleaned)


def normalize_process_number(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    digits = digits.lstrip("0")
    return digits or ""


def normalize_relator(text: str) -> str:
    s = strip_accents((text or "").lower())
    s = re.sub(
        r"\b(min\.|des\.|juiz(?:a)?|in\.|red\.?|designado|designada)\b",
        " ",
        s,
    )
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [t for t in normalize_ws(s).split() if t not in {"de", "da", "do", "dos", "das", "e"}]
    if len(tokens) > 3:
        tokens = tokens[-3:]
    return " ".join(tokens)


def parse_date_to_mdy(raw_date: str, prefer_day_first: bool) -> Tuple[str, str]:
    """
    Converte data textual para M/D/AAAA.
    Retorna (valor, status): status em {'parsed', 'empty', 'unparsed'}.
    """
    raw = normalize_ws((raw_date or "").replace("º", ""))
    if not raw:
        return "", "empty"

    m = re.fullmatch(r"(\d{1,2})([./-])(\d{1,2})\2(\d{2,4})", raw)
    if not m:
        return "", "unparsed"

    first_s, sep, second_s, year_s = m.groups()
    first = int(first_s)
    second = int(second_s)
    year = int(year_s)
    if len(year_s) == 2:
        year = 2000 + year if year <= 30 else 1900 + year

    # Regras conservadoras para distinguir D/M e M/D.
    if sep in ".-":
        day, month = first, second
    else:
        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            month, day = first, second
        elif first > 12 and second > 12:
            return "", "unparsed"
        else:
            # Ambígua (<=12). Preferência configurável.
            if prefer_day_first:
                day, month = first, second
            else:
                month, day = first, second

    try:
        dt.date(year, month, day)
    except ValueError:
        return "", "unparsed"

    return f"{month}/{day}/{year}", "parsed"


def extract_relator_from_text(citation_text: str) -> str:
    match = RELATOR_RE.search(citation_text or "")
    return normalize_ws(match.group(1)) if match else ""


def extract_process_from_text(citation_text: str) -> str:
    text = normalize_ws(citation_text or "")

    # 1) Preferência: número após "n." / "nº".
    n_matches = list(re.finditer(r"\bn[ºo°.]?\s*(\d[\d./-]*)", text, re.IGNORECASE))
    if n_matches:
        return normalize_process_number(n_matches[0].group(1))

    # 2) Fallback: trecho após "no/na/nos/nas".
    no_seg = re.search(r"\b(?:no|na|nos|nas)\b([^\)]{0,120})", text, re.IGNORECASE)
    if no_seg:
        tokens = re.findall(r"\d[\d./-]*", no_seg.group(1))
        tokens = [
            t
            for t in tokens
            if not re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", t)
        ]
        if tokens:
            return normalize_process_number(max(tokens, key=lambda x: len(re.sub(r"\D", "", x))))

    # 3) Fallback geral (antes de "rel."): maior token numérico não-data.
    before_rel = text.split("rel.", 1)[0]
    tokens = re.findall(r"\d[\d./-]*", before_rel)
    tokens = [
        t
        for t in tokens
        if not re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", t)
    ]
    if not tokens:
        return ""
    return normalize_process_number(max(tokens, key=lambda x: len(re.sub(r"\D", "", x))))


def extract_date_from_text(citation_text: str) -> str:
    match = DATE_IN_TEXT_RE.search(citation_text or "")
    if not match:
        return ""
    date_str, status = parse_date_to_mdy(match.group(1), prefer_day_first=True)
    return date_str if status == "parsed" else ""


def extract_html_citations(html_path: Path) -> List[HtmlCitation]:
    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    citations: List[HtmlCitation] = []
    seen: Set[Tuple[str, str]] = set()

    for href_raw, inner_html in ANCHOR_TAG_RE.findall(raw_html):
        href = normalize_ws(html.unescape(href_raw))
        if "sjur-servicos.tse.jus.br/sjur-servicos/rest/download/pdf/" not in href:
            continue

        citation_raw = clean_anchor_inner_html(inner_html)
        if not citation_raw:
            continue
        if not HAS_CITATION_RE.search(citation_raw):
            continue

        citation_norm = normalize_citation_text(citation_raw)
        if not citation_norm:
            continue

        dedupe_key = (citation_norm, href)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        citations.append(
            HtmlCitation(
                href=href,
                texto_citacao_raw=citation_raw,
                texto_citacao_norm=citation_norm,
                numero_norm=extract_process_from_text(citation_raw),
                relator_norm=normalize_relator(extract_relator_from_text(citation_raw)),
                data_norm_mdy=extract_date_from_text(citation_raw),
            )
        )
    return citations


def extract_row_citations(texto_original: str) -> List[Tuple[str, str]]:
    text = texto_original or ""
    raw_candidates = [normalize_ws(m.group(0)) for m in ROW_CITATION_PAREN_RE.finditer(text)]

    if not raw_candidates:
        raw_candidates = [normalize_ws(m.group(0)) for m in ROW_CITATION_FALLBACK_RE.finditer(text)]

    dedup: List[Tuple[str, str]] = []
    seen_norm: Set[str] = set()
    for raw in raw_candidates:
        norm = normalize_citation_text(raw)
        if len(norm) < 12:
            continue
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        dedup.append((raw, norm))
    return dedup


def summarize_text(value: str, max_len: int = 280) -> str:
    clean = normalize_ws(value or "")
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def build_html_indexes(
    html_citations: Sequence[HtmlCitation],
) -> Tuple[
    Dict[str, Set[str]],
    Dict[Tuple[str, str, str], Set[str]],
    Dict[Tuple[str, str], Set[str]],
    Dict[str, Set[str]],
    Dict[str, List[HtmlCitation]],
]:
    by_citation: Dict[str, Set[str]] = defaultdict(set)
    by_proc_rel_date: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    by_proc_rel: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    by_proc: Dict[str, Set[str]] = defaultdict(set)
    citation_to_records: Dict[str, List[HtmlCitation]] = defaultdict(list)

    for rec in html_citations:
        by_citation[rec.texto_citacao_norm].add(rec.href)
        citation_to_records[rec.texto_citacao_norm].append(rec)

        if rec.numero_norm:
            by_proc[rec.numero_norm].add(rec.href)
        if rec.numero_norm and rec.relator_norm:
            by_proc_rel[(rec.numero_norm, rec.relator_norm)].add(rec.href)
        if rec.numero_norm and rec.relator_norm and rec.data_norm_mdy:
            by_proc_rel_date[(rec.numero_norm, rec.relator_norm, rec.data_norm_mdy)].add(rec.href)

    return by_citation, by_proc_rel_date, by_proc_rel, by_proc, citation_to_records


def containment_candidates(
    row_citation_norms: Sequence[str],
    html_citation_keys: Sequence[str],
    html_by_citation: Dict[str, Set[str]],
) -> Set[str]:
    result: Set[str] = set()
    for rc in row_citation_norms:
        if len(rc) < MIN_CONTAINMENT_LEN:
            continue
        for hk in html_citation_keys:
            if len(hk) < MIN_CONTAINMENT_LEN:
                continue
            if rc in hk or hk in rc:
                result.update(html_by_citation[hk])
    return result


def metadata_candidates(
    numero_norm: str,
    relator_norm: str,
    data_norm_mdy: str,
    idx_proc_rel_date: Dict[Tuple[str, str, str], Set[str]],
    idx_proc_rel: Dict[Tuple[str, str], Set[str]],
    idx_proc: Dict[str, Set[str]],
) -> Tuple[Set[str], str]:
    if numero_norm and relator_norm and data_norm_mdy:
        cands = idx_proc_rel_date.get((numero_norm, relator_norm, data_norm_mdy), set())
        if cands:
            return set(cands), "C_proc_rel_data"
    if numero_norm and relator_norm:
        cands = idx_proc_rel.get((numero_norm, relator_norm), set())
        if cands:
            return set(cands), "C_proc_rel"
    if numero_norm:
        cands = idx_proc.get(numero_norm, set())
        if cands:
            return set(cands), "C_processo"
    return set(), "none"


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    today = dt.date.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(
        description="Audita e enriquece links do ramo 'Apuração de votos e eleições extraordinárias'."
    )
    parser.add_argument(
        "--csv-in",
        default="temas_selec_TSE_all.csv",
        help="CSV de entrada (default: temas_selec_TSE_all.csv)",
    )
    parser.add_argument(
        "--html-in",
        default="Apuração de votos e eleições extraordinárias — Temas Selecionados.html",
        help="HTML salvo localmente com os links de referência",
    )
    parser.add_argument(
        "--csv-out",
        default=f"temas_selec_TSE_all_apuracao_enriquecido_{today}.csv",
        help="CSV de saída enriquecido (default versionado por data)",
    )
    parser.add_argument(
        "--audit-prefix",
        default=f"auditoria_apuracao_{today}",
        help="Prefixo para arquivos de auditoria",
    )
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    args = parse_args()

    csv_in = Path(args.csv_in)
    html_in = Path(args.html_in)
    csv_out = Path(args.csv_out)
    audit_prefix = args.audit_prefix

    if not csv_in.exists():
        raise FileNotFoundError(f"CSV de entrada não encontrado: {csv_in}")
    if not html_in.exists():
        raise FileNotFoundError(f"HTML de entrada não encontrado: {html_in}")

    # 1) Carregar CSV completo
    with csv_in.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise RuntimeError("CSV de entrada está vazio.")

    has_link_col = "link_de_acesso" in original_fieldnames
    fieldnames = list(original_fieldnames)
    if not has_link_col:
        fieldnames.append("link_de_acesso")
        for row in rows:
            row["link_de_acesso"] = ""

    # Snapshot para checks de consistência de escopo
    original_rows = [dict(r) for r in rows]

    # 2) Carregar citações do HTML
    html_citations = extract_html_citations(html_in)
    (
        html_by_citation,
        html_by_proc_rel_date,
        html_by_proc_rel,
        html_by_proc,
        html_citation_records,
    ) = build_html_indexes(html_citations)
    html_citation_keys = list(html_by_citation.keys())

    # Relatórios
    linhas_sem_match: List[Dict[str, str]] = []
    linhas_ambiguas: List[Dict[str, str]] = []
    omissoes_html_nao_no_csv: List[Dict[str, str]] = []
    csv_nao_no_html: List[Dict[str, str]] = []

    # Métricas
    target_count = 0
    links_preenchidos = 0
    date_changed = 0
    date_unparsed = 0
    date_empty = 0

    # Para auditoria de omissões (conjunto de citações extraídas do CSV)
    csv_citation_entries: List[Dict[str, str]] = []

    # 3/4/5) Processar apenas o ramo-alvo
    for idx, row in enumerate(rows):
        if normalize_ws(row.get("ramo", "")) != TARGET_RAMO:
            # Se a coluna já existia, preserva valor fora do escopo.
            if not has_link_col:
                row["link_de_acesso"] = ""
            continue

        target_count += 1
        row_id = normalize_ws(row.get("ID", ""))

        # Normalização de data no ramo-alvo
        raw_date = normalize_ws(row.get("data_julgamento", ""))
        normalized_date, date_status = parse_date_to_mdy(raw_date, prefer_day_first=False)
        if date_status == "parsed":
            if raw_date != normalized_date:
                date_changed += 1
            row["data_julgamento"] = normalized_date
        elif date_status == "empty":
            date_empty += 1
            row["data_julgamento"] = ""
        else:
            date_unparsed += 1
            # Conservador: mantém valor original se não for possível interpretar.
            row["data_julgamento"] = raw_date

        # Extração de citações da linha
        citation_pairs = extract_row_citations(row.get("texto_original", ""))
        row_citation_norms = [norm for _, norm in citation_pairs]

        for raw_cit, norm_cit in citation_pairs:
            csv_citation_entries.append(
                {
                    "row_idx": str(idx),
                    "ID": row_id,
                    "numero_processo": normalize_ws(row.get("numero_processo", "")),
                    "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                    "relator": normalize_ws(row.get("relator", "")),
                    "data_julgamento": normalize_ws(row.get("data_julgamento", "")),
                    "citacao_csv_raw": raw_cit,
                    "citacao_csv_norm": norm_cit,
                    "texto_original_resumo": summarize_text(row.get("texto_original", "")),
                }
            )

        # Metadados normalizados para fallback
        numero_norm = normalize_process_number(row.get("numero_processo", ""))
        relator_norm = normalize_relator(row.get("relator", ""))
        data_norm, _ = parse_date_to_mdy(row.get("data_julgamento", ""), prefer_day_first=False)

        # Etapa A: match exato por citação normalizada
        candidates: Set[str] = set()
        match_stage = "none"
        for cit_norm in row_citation_norms:
            candidates.update(html_by_citation.get(cit_norm, set()))
        if candidates:
            match_stage = "A_exato"
        else:
            # Etapa B: match por contenção textual
            candidates = containment_candidates(row_citation_norms, html_citation_keys, html_by_citation)
            if candidates:
                match_stage = "B_contencao"
            else:
                # Etapa C: fallback por metadados
                candidates, match_stage = metadata_candidates(
                    numero_norm,
                    relator_norm,
                    data_norm,
                    html_by_proc_rel_date,
                    html_by_proc_rel,
                    html_by_proc,
                )

        if len(candidates) == 1:
            row["link_de_acesso"] = next(iter(candidates))
            links_preenchidos += 1
        elif len(candidates) == 0:
            row["link_de_acesso"] = ""
            linhas_sem_match.append(
                {
                    "ID": row_id,
                    "numero_processo": normalize_ws(row.get("numero_processo", "")),
                    "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                    "relator": normalize_ws(row.get("relator", "")),
                    "data_julgamento": normalize_ws(row.get("data_julgamento", "")),
                    "match_stage": match_stage,
                    "qtd_citacoes_extraidas": str(len(row_citation_norms)),
                    "citacao_principal": citation_pairs[0][0] if citation_pairs else "",
                    "texto_original_resumo": summarize_text(row.get("texto_original", "")),
                    "href_candidato": "",
                }
            )
        else:
            row["link_de_acesso"] = ""
            linhas_ambiguas.append(
                {
                    "ID": row_id,
                    "numero_processo": normalize_ws(row.get("numero_processo", "")),
                    "tipo_de_processo": normalize_ws(row.get("tipo_de_processo", "")),
                    "relator": normalize_ws(row.get("relator", "")),
                    "data_julgamento": normalize_ws(row.get("data_julgamento", "")),
                    "match_stage": match_stage,
                    "qtd_citacoes_extraidas": str(len(row_citation_norms)),
                    "citacao_principal": citation_pairs[0][0] if citation_pairs else "",
                    "qtd_hrefs_candidatos": str(len(candidates)),
                    "hrefs_candidatos": " | ".join(sorted(candidates)),
                    "texto_original_resumo": summarize_text(row.get("texto_original", "")),
                }
            )

    # 6) Auditoria de omissões (citações HTML x CSV)
    csv_citation_keys = {entry["citacao_csv_norm"] for entry in csv_citation_entries if entry["citacao_csv_norm"]}
    csv_citation_keys_list = list(csv_citation_keys)

    def html_key_is_represented_in_csv(html_key: str) -> bool:
        if html_key in csv_citation_keys:
            return True
        if len(html_key) < MIN_CONTAINMENT_LEN:
            return False
        for ck in csv_citation_keys_list:
            if len(ck) < MIN_CONTAINMENT_LEN:
                continue
            if ck in html_key or html_key in ck:
                return True
        return False

    for html_key in html_citation_keys:
        if html_key_is_represented_in_csv(html_key):
            continue
        for rec in html_citation_records.get(html_key, []):
            omissoes_html_nao_no_csv.append(
                {
                    "texto_citacao_raw": rec.texto_citacao_raw,
                    "texto_citacao_norm": rec.texto_citacao_norm,
                    "href": rec.href,
                    "numero_norm": rec.numero_norm,
                    "relator_norm": rec.relator_norm,
                    "data_norm_mdy": rec.data_norm_mdy,
                }
            )

    html_keys_set = set(html_citation_keys)
    html_keys_list = list(html_keys_set)
    seen_csv_nao_html: Set[Tuple[str, str]] = set()

    def csv_key_is_in_html(csv_key: str) -> bool:
        if csv_key in html_keys_set:
            return True
        if len(csv_key) < MIN_CONTAINMENT_LEN:
            return False
        for hk in html_keys_list:
            if len(hk) < MIN_CONTAINMENT_LEN:
                continue
            if csv_key in hk or hk in csv_key:
                return True
        return False

    for entry in csv_citation_entries:
        csv_key = entry["citacao_csv_norm"]
        if not csv_key:
            continue
        if csv_key_is_in_html(csv_key):
            continue

        dedupe_key = (entry["ID"], csv_key)
        if dedupe_key in seen_csv_nao_html:
            continue
        seen_csv_nao_html.add(dedupe_key)

        meta_cands, _ = metadata_candidates(
            normalize_process_number(entry["numero_processo"]),
            normalize_relator(entry["relator"]),
            parse_date_to_mdy(entry["data_julgamento"], prefer_day_first=False)[0],
            html_by_proc_rel_date,
            html_by_proc_rel,
            html_by_proc,
        )
        href_candidate = " | ".join(sorted(meta_cands)) if meta_cands else ""

        csv_nao_no_html.append(
            {
                "ID": entry["ID"],
                "numero_processo": entry["numero_processo"],
                "tipo_de_processo": entry["tipo_de_processo"],
                "relator": entry["relator"],
                "data_julgamento": entry["data_julgamento"],
                "citacao_csv_raw": entry["citacao_csv_raw"],
                "citacao_csv_norm": entry["citacao_csv_norm"],
                "href_candidato": href_candidate,
                "texto_original_resumo": entry["texto_original_resumo"],
            }
        )

    # 7) Salvar CSV final e relatórios
    write_csv(csv_out, rows, fieldnames)

    sem_match_path = Path(f"{audit_prefix}_linhas_sem_match.csv")
    ambiguas_path = Path(f"{audit_prefix}_linhas_ambiguas.csv")
    omissoes_path = Path(f"{audit_prefix}_omissoes_html_nao_no_csv.csv")
    csv_nao_html_path = Path(f"{audit_prefix}_csv_nao_no_html.csv")
    resumo_path = Path(f"{audit_prefix}_resumo.txt")

    write_csv(
        sem_match_path,
        linhas_sem_match,
        [
            "ID",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_julgamento",
            "match_stage",
            "qtd_citacoes_extraidas",
            "citacao_principal",
            "texto_original_resumo",
            "href_candidato",
        ],
    )
    write_csv(
        ambiguas_path,
        linhas_ambiguas,
        [
            "ID",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_julgamento",
            "match_stage",
            "qtd_citacoes_extraidas",
            "citacao_principal",
            "qtd_hrefs_candidatos",
            "hrefs_candidatos",
            "texto_original_resumo",
        ],
    )
    write_csv(
        omissoes_path,
        omissoes_html_nao_no_csv,
        [
            "texto_citacao_raw",
            "texto_citacao_norm",
            "href",
            "numero_norm",
            "relator_norm",
            "data_norm_mdy",
        ],
    )
    write_csv(
        csv_nao_html_path,
        csv_nao_no_html,
        [
            "ID",
            "numero_processo",
            "tipo_de_processo",
            "relator",
            "data_julgamento",
            "citacao_csv_raw",
            "citacao_csv_norm",
            "href_candidato",
            "texto_original_resumo",
        ],
    )

    # Checks básicos de consistência de escopo
    total_rows = len(rows)
    non_target_changes = 0
    for i, (before, after) in enumerate(zip(original_rows, rows)):
        if normalize_ws(after.get("ramo", "")) == TARGET_RAMO:
            continue
        before_date = normalize_ws(before.get("data_julgamento", ""))
        after_date = normalize_ws(after.get("data_julgamento", ""))
        if before_date != after_date:
            non_target_changes += 1

    without_link_target = target_count - links_preenchidos

    summary_lines = [
        f"CSV entrada: {csv_in}",
        f"HTML entrada: {html_in}",
        f"CSV saída: {csv_out}",
        f"Prefixo auditoria: {audit_prefix}",
        "",
        f"Linhas totais no CSV: {total_rows}",
        f"Linhas no ramo alvo: {target_count}",
        f"Linhas fora do ramo alvo: {total_rows - target_count}",
        "",
        f"Datas normalizadas (valor alterado): {date_changed}",
        f"Datas vazias no ramo alvo: {date_empty}",
        f"Datas não interpretáveis (mantidas): {date_unparsed}",
        "",
        f"Links preenchidos (match único): {links_preenchidos}",
        f"Linhas alvo sem link: {without_link_target}",
        f"Linhas sem match: {len(linhas_sem_match)}",
        f"Linhas ambíguas: {len(linhas_ambiguas)}",
        "",
        f"Omissões HTML não no CSV: {len(omissoes_html_nao_no_csv)}",
        f"Citações CSV não no HTML: {len(csv_nao_no_html)}",
        "",
        f"Alterações de data fora do ramo alvo (esperado 0): {non_target_changes}",
    ]
    resumo_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
