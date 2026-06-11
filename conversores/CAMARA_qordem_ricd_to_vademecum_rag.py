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
import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

import NOTION_import_codigo_eleitoral_2026_csvs as notion_import
import NOTION_import_vademecum_consolidado as vademecum


LOGGER = logging.getLogger("camara_qordem_ricd")
FETCH_ERRORS: Dict[str, str] = {}

DEFAULT_BASE_URL = "https://www.camara.leg.br/busca-qordem-api/qordem"
DEFAULT_SOURCE_PAGE = "https://www.camara.leg.br/v-busca-qordem"
DEFAULT_VADEMECUM_CSV = Path("vademecum_rag_consolidado.csv")
DEFAULT_OUTPUT_CSV = Path("camara_qordem_ricd_rag.csv")
DEFAULT_DETAILS_JSON = Path("Artefatos/checkpoints/camara_qordem_ricd_details.json")
DEFAULT_REPORT_JSON = Path("Artefatos/checkpoints/camara_qordem_ricd_report.json")
DEFAULT_CHECKPOINT = Path("Artefatos/checkpoints/vademecum_rag_consolidado_checkpoint.json")
RICD_NORMA_ID = "regimento_interno_camara_deputados"
RICD_NORMA_TITULO = "Regimento Interno da Camara dos Deputados"
RICD_NOME_POPULAR = "RICD"
RICD_NORMA_KEY = f"norma:{RICD_NORMA_ID}"

DEVICE_LISTS = {
    "questao": "dispositivosRegimentaisQO",
    "contradita": "dispositivosRegimentaisCD",
    "decisao": "dispositivosRegimentaisDE",
    "recurso_ccjr": "dispositivosRegimentaisRE",
}

OTHER_DEVICE_LISTS = {
    "constitucionais_questao": "dispositivosConstitucionaisQO",
    "constitucionais_contradita": "dispositivosConstitucionaisCD",
    "constitucionais_decisao": "dispositivosConstitucionaisDE",
    "constitucionais_recurso_ccjr": "dispositivosConstitucionaisRE",
    "outros_questao": "outrosDispositivosQO",
    "outros_contradita": "outrosDispositivosCD",
    "outros_decisao": "outrosDispositivosDE",
    "outros_recurso_ccjr": "outrosDispositivosRE",
}


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\ufeff", " ").replace("\xa0", " ")).strip()


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


def limit_text(text: Any, max_chars: int) -> str:
    value = normalize_text_block(text)
    if not value or len(value) <= max_chars:
        return value
    marker = "\n[TRUNCADO PARA RAG; consulte a fonte oficial pelo link da Questão de Ordem]"
    cut = value[: max(0, max_chars - len(marker))].rsplit(" ", 1)[0].rstrip()
    return cut + marker


def first_sentence(text: Any) -> str:
    value = normalize_ws(text)
    if not value:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-ZÁ-Ú0-9])", value)
    return normalize_ws(pieces[0] if pieces else value)


def legal_display(text: Any) -> str:
    value = normalize_ws(text)
    value = re.sub(r"Art\.\s*(\d+(?:-\w+)?[A-Z]?)\s*[ºo]?", r"Art. \1", value, flags=re.IGNORECASE)
    value = re.sub(r"§\s*(\d+)\s*[ºo]?", r"§\1º", value)
    value = re.sub(r"§\s*(\d+)\b", r"§\1º", value)
    value = re.sub(r"\s*,\s*", ", ", value)
    return value.strip()


def device_path_key(text: Any) -> str:
    value = legal_display(text)
    value = value.replace("Parágrafo único", "paragrafo unico").replace("parágrafo único", "paragrafo unico")
    value = value.replace("º", "o")
    value = re.sub(r"\bal[ií]nea\s+([a-z])\b", r"\1)", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return ascii_key(value)


def parse_article_number(raw: Any) -> str:
    value = normalize_ws(raw)
    match = re.search(r"(\d+(?:-\w+)?[A-Z]?)", value, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def format_article(article_number: str) -> str:
    return f"Art. {parse_article_number(article_number)}" if parse_article_number(article_number) else ""


def format_paragraph(token: str) -> str:
    key = ascii_key(token)
    if "paragrafo unico" in key:
        return "Parágrafo único"
    match = re.search(r"§\s*(\d+)", token)
    return f"§{match.group(1)}º" if match else normalize_ws(token)


def normalize_letter(token: str) -> str:
    match = re.search(r"[\"“']?([a-z])[\"”']?", token, flags=re.IGNORECASE)
    return f"{match.group(1).lower()})" if match else normalize_ws(token)


def extract_reference_targets(device: Mapping[str, Any]) -> List[str]:
    article = format_article(device.get("txtNumeroArtigo"))
    full = normalize_ws(device.get("txtNumeroArtigoCompleto"))
    if not article:
        match = re.search(r"Art\.?\s*(\d+(?:-\w+)?[A-Z]?)", full, flags=re.IGNORECASE)
        article = format_article(match.group(1)) if match else ""
    if not article:
        return []
    paren_match = re.search(r"\((.*)\)", full)
    if not paren_match:
        return [article]
    inside = normalize_ws(paren_match.group(1))
    if not inside or ascii_key(inside) == "caput":
        return [article]

    token_re = re.compile(
        r"par[aá]grafo\s+[úu]nico|§\s*\d+\s*[ºo]?|\b[IVXLCDM]+\b|[\"“'][a-z][\"”']|al[ií]nea\s+[a-z]",
        flags=re.IGNORECASE,
    )
    current: List[str] = []
    targets: List[str] = []
    for match in token_re.finditer(inside):
        token = match.group(0)
        token_key = ascii_key(token)
        if token_key.startswith("paragrafo") or token.strip().startswith("§"):
            current = [format_paragraph(token)]
            targets.append(", ".join([article] + current))
            continue
        if re.fullmatch(r"[IVXLCDM]+", token, flags=re.IGNORECASE):
            roman = token.upper()
            if current and (current[-1].startswith("§") or current[-1] == "Parágrafo único"):
                current = [current[-1], roman]
            else:
                current = [roman]
            targets.append(", ".join([article] + current))
            continue
        letter = normalize_letter(token)
        if current:
            targets.append(", ".join([article] + current + [letter]))
        else:
            targets.append(article)
    if not targets:
        return [article]

    deduped: List[str] = []
    for target in targets:
        if target not in deduped:
            deduped.append(target)
    # If a leaf target exists, remove its immediate generic parent from the same API reference.
    pruned: List[str] = []
    for target in deduped:
        key = device_path_key(target)
        is_parent = any(device_path_key(other).startswith(key + ",") for other in deduped if other != target)
        if not is_parent or len(deduped) == 1:
            pruned.append(target)
    return pruned or [article]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_ricd_anchor_maps(rows: Sequence[Mapping[str, str]]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    path_to_row_key: Dict[str, str] = {}
    article_to_row_key: Dict[str, str] = {}
    row_key_to_hierarchy: Dict[str, str] = {}
    for row in rows:
        if normalize_ws(row.get("norma_id")) != RICD_NORMA_ID:
            continue
        row_key = normalize_ws(row.get("row_key"))
        if not row_key:
            continue
        hierarchy = normalize_ws(row.get("hierarquia_normativa"))
        row_key_to_hierarchy[row_key] = hierarchy
        tail = normalize_ws(hierarchy.split(">")[-1] if hierarchy else row.get("dispositivo"))
        if tail:
            path_to_row_key[device_path_key(tail)] = row_key
        dispositivo = normalize_ws(row.get("dispositivo"))
        if normalize_ws(row.get("tipo_dispositivo")) == "artigo":
            number = parse_article_number(dispositivo)
            if number:
                article_to_row_key[parse_article_number(number)] = row_key
                path_to_row_key[device_path_key(format_article(number))] = row_key
    return path_to_row_key, article_to_row_key, row_key_to_hierarchy


def resolve_anchor(
    target: str,
    *,
    path_to_row_key: Mapping[str, str],
    article_to_row_key: Mapping[str, str],
) -> Tuple[str, str]:
    exact = path_to_row_key.get(device_path_key(target), "")
    if exact:
        return exact, "exact"
    article_match = re.search(r"Art\.\s*(\d+(?:-\w+)?[A-Z]?)", target, flags=re.IGNORECASE)
    if article_match:
        article_key = article_to_row_key.get(parse_article_number(article_match.group(1)), "")
        if article_key:
            return article_key, "article_fallback"
    return RICD_NORMA_KEY, "norma_fallback"


def read_details_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    if isinstance(payload, list):
        return {str(item.get("numInternoQOrdem")): item for item in payload if isinstance(item, dict)}
    return {}


def write_details_cache(path: Path, details: Mapping[str, Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    ordered = {str(key): details[str(key)] for key in sorted(details, key=lambda item: int(item))}
    tmp.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class QOrdemClient:
    def __init__(self, base_url: str, *, timeout_s: int, max_retries: int, rate_rps: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = max(5, int(timeout_s))
        self.max_retries = max(0, int(max_retries))
        self.min_interval_s = 1.0 / rate_rps if rate_rps > 0 else 0.0
        self.next_request_at = 0.0
        self.lock = threading.Lock()
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 Codex-RAG-QOrdem/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
            )
            self.local.session = session
        return session

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            with self.lock:
                now = time.monotonic()
                if now < self.next_request_at:
                    time.sleep(self.next_request_at - now)
                self.next_request_at = time.monotonic() + self.min_interval_s
            try:
                response = self.session().request(method, url, timeout=self.timeout_s, **kwargs)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(min(12.0, 0.8 * (2**attempt)))
                    continue
                response.raise_for_status()
                if not response.content:
                    return None
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(12.0, 0.8 * (2**attempt)))
        raise RuntimeError(f"Falha ao acessar {url}") from last_exc

    def search_page(self, page_index: int) -> Dict[str, Any]:
        return self.request(
            "POST",
            "/search",
            json={"filtro": {}, "numPagina": int(page_index), "ordem": "datSessaoQOrdem"},
        )

    def detail(self, qordem_id: int) -> Dict[str, Any]:
        payload = self.request("GET", f"/{int(qordem_id)}")
        return payload if isinstance(payload, dict) else {}


def fetch_all_details(
    client: QOrdemClient,
    *,
    cache_path: Path,
    workers: int,
    checkpoint_every: int,
    max_pages: int,
    max_details: int,
) -> Dict[str, Dict[str, Any]]:
    details = read_details_cache(cache_path)
    first = client.search_page(0)
    total_pages = int(first.get("qtdPaginas") or 0)
    total_results = int(first.get("resultadosCount") or 0)
    LOGGER.info("Busca geral qordem: %s resultados em %s paginas.", total_results, total_pages)
    if max_pages > 0:
        total_pages = min(total_pages, max_pages)

    ids: List[int] = []
    for page_index in range(total_pages):
        payload = first if page_index == 0 else client.search_page(page_index)
        for item in payload.get("resultadosList", []) or []:
            qid = item.get("numInternoQOrdem")
            if qid is not None:
                ids.append(int(qid))
        if page_index and page_index % 50 == 0:
            LOGGER.info("Paginas de busca lidas: %s/%s; ids acumulados: %s.", page_index + 1, total_pages, len(ids))
    ids = sorted(set(ids), reverse=True)
    if max_details > 0:
        ids = ids[:max_details]
    pending = [qid for qid in ids if str(qid) not in details]
    LOGGER.info("Detalhes qordem pendentes: %s de %s.", len(pending), len(ids))
    if not pending:
        return details

    submitted = 0
    completed = 0
    in_flight: Dict[Future[Dict[str, Any]], int] = {}
    max_in_flight = max(1, workers * 3)

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal submitted
        while submitted < len(pending) and len(in_flight) < max_in_flight:
            qid = pending[submitted]
            submitted += 1
            future = executor.submit(client.detail, qid)
            in_flight[future] = qid

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        submit_next(executor)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                qid = in_flight.pop(future)
                try:
                    detail = future.result()
                except Exception as exc:  # noqa: BLE001
                    FETCH_ERRORS[str(qid)] = str(exc)
                    LOGGER.warning("Detalhe qordem %s ignorado apos falha: %s", qid, exc)
                    detail = {}
                if detail:
                    details[str(qid)] = detail
                completed += 1
                if completed % checkpoint_every == 0 or completed == len(pending):
                    write_details_cache(cache_path, details)
                    LOGGER.info("Detalhes qordem salvos: %s/%s novos; cache=%s.", completed, len(pending), len(details))
            submit_next(executor)
    write_details_cache(cache_path, details)
    return details


def all_regimental_devices(detail: Mapping[str, Any]) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for bloco, key in DEVICE_LISTS.items():
        for device in detail.get(key, []) or []:
            if not isinstance(device, dict):
                continue
            full = normalize_ws(device.get("txtNumeroArtigoCompleto"))
            article = normalize_ws(device.get("txtNumeroArtigo"))
            dedupe_key = (bloco, article, full)
            if not full or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            item = dict(device)
            item["_bloco"] = bloco
            item["_source_key"] = key
            devices.append(item)
    return devices


def readable_devices(devices: Iterable[Mapping[str, Any]]) -> str:
    labels = []
    for device in devices:
        label = normalize_ws(device.get("txtNumeroArtigoCompleto"))
        bloco = normalize_ws(device.get("_bloco"))
        if label:
            labels.append(f"{label} [{bloco}]" if bloco else label)
    return "; ".join(labels)


def other_references(detail: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    refs: Dict[str, List[Mapping[str, Any]]] = {}
    for label, key in OTHER_DEVICE_LISTS.items():
        values = [item for item in detail.get(key, []) or [] if isinstance(item, dict)]
        if values:
            refs[label] = values
    return refs


def compact_detail_json(detail: Mapping[str, Any], selected_device: Mapping[str, Any], target: str, anchor_mode: str) -> str:
    payload = {
        "fonte": "Camara dos Deputados - busca-qordem-api",
        "url": f"{DEFAULT_SOURCE_PAGE}/{detail.get('numInternoQOrdem')}",
        "numInternoQOrdem": detail.get("numInternoQOrdem"),
        "numQOrdemComAno": detail.get("numQOrdemComAno"),
        "datSessaoQOrdem": detail.get("datSessaoQOrdem"),
        "datSessaoDecisao": detail.get("datSessaoDecisao"),
        "autor": detail.get("txtNomeAutorQOrdem"),
        "presidente_questao": detail.get("txtPresidenteQOrdem"),
        "presidente_decisao": detail.get("txtNomePresidenteDecisao"),
        "dispositivo_regimental_selecionado": selected_device,
        "alvo_ricd": target,
        "anchor_mode": anchor_mode,
        "dispositivos_regimentais": all_regimental_devices(detail),
        "outras_referencias": other_references(detail),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_text_rag(
    detail: Mapping[str, Any],
    *,
    selected_label: str,
    target: str,
    anchor_hierarchy: str,
    all_reg_devices: str,
    other_refs: Mapping[str, List[Mapping[str, Any]]],
    max_chars: int,
) -> str:
    other_ref_text = []
    for label, devices in other_refs.items():
        values = "; ".join(normalize_ws(item.get("txtNumeroArtigoCompleto")) for item in devices if normalize_ws(item.get("txtNumeroArtigoCompleto")))
        if values:
            other_ref_text.append(f"{label}: {values}")
    parts = [
        "Base: Vademecum",
        "Tipo da linha: questao_ordem",
        f"Norma interpretada: {RICD_NORMA_TITULO}",
        f"Nome popular: {RICD_NOME_POPULAR}",
        "Grupo: camara_deputados",
        f"Dispositivo regimental selecionado: {selected_label}",
        f"Ancora RICD: {target}",
        f"Hierarquia RICD: {anchor_hierarchy}",
        f"Questao de Ordem: {normalize_ws(detail.get('numQOrdemComAno'))}",
        f"ID interno Camara: {normalize_ws(detail.get('numInternoQOrdem'))}",
        f"Data da questao: {normalize_ws(detail.get('datSessaoQOrdem'))}",
        f"Data da decisao: {normalize_ws(detail.get('datSessaoDecisao'))}",
        f"Autor: {normalize_ws(detail.get('txtNomeAutorQOrdem'))}",
        f"Presidente da questao: {normalize_ws(detail.get('txtPresidenteQOrdem'))}",
        f"Presidente da decisao: {normalize_ws(detail.get('txtNomePresidenteDecisao'))}",
        f"Fonte: {DEFAULT_SOURCE_PAGE}/{normalize_ws(detail.get('numInternoQOrdem'))}",
        f"Dispositivos regimentais relacionados: {all_reg_devices}",
    ]
    if other_ref_text:
        parts.append("Outras referencias normativas: " + " | ".join(other_ref_text))
    sections = [
        ("Ementa da questao", detail.get("txtEmentaQOrdem"), 3500),
        ("Ementa da decisao", detail.get("txtEmentaDecisao"), 7000),
        ("Decisao", detail.get("txtDecisao"), 9000),
        ("Questao de ordem - inteiro teor", detail.get("txtQOrdem"), 8000),
        ("Contradita", detail.get("txtCDita"), 4000),
        ("Ementa do recurso", detail.get("txtEmentaRecurso"), 3500),
        ("Recurso", detail.get("txtRecurso"), 5000),
        ("Ementa CCJR", detail.get("txtEmentaCCJR"), 3500),
        ("Observacoes", "\n".join(filter(None, [normalize_text_block(detail.get("txtObservacaoQOrdem")), normalize_text_block(detail.get("txtObservacaoDecisao"))])), 2500),
        ("Indexacao", "\n".join(filter(None, [normalize_text_block(detail.get("txtIndexacaoQOrdem")), normalize_text_block(detail.get("txtIndexacaoDecisao"))])), 2500),
    ]
    for title, text, cap in sections:
        value = limit_text(text, cap)
        if value:
            parts.append(f"\n{title}:\n{value}")
    return limit_text("\n".join(part for part in parts if normalize_ws(part)), max_chars)


def build_qordem_rows(
    details: Mapping[str, Mapping[str, Any]],
    *,
    existing_rows: Sequence[Mapping[str, str]],
    max_text_rag_chars: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    path_to_row_key, article_to_row_key, row_key_to_hierarchy = load_ricd_anchor_maps(existing_rows)
    existing_keys = {normalize_ws(row.get("row_key")) for row in existing_rows if normalize_ws(row.get("row_key"))}
    max_id = 0
    max_ordem = 0
    for row in existing_rows:
        for key, current in (("id", max_id), ("ordem_global", max_ordem)):
            raw = normalize_ws(row.get(key))
            if raw.isdigit():
                if key == "id":
                    max_id = max(max_id, int(raw))
                else:
                    max_ordem = max(max_ordem, int(raw))

    rows: List[Dict[str, str]] = []
    seen_row_keys: set[str] = set()
    anchor_modes = Counter()
    skipped_no_regimental = 0
    generated_per_qo = Counter()

    for detail_id in sorted(details, key=lambda item: int(item)):
        detail = details[detail_id]
        reg_devices = all_regimental_devices(detail)
        if not reg_devices:
            skipped_no_regimental += 1
            continue
        all_reg_text = readable_devices(reg_devices)
        refs = other_references(detail)
        for device in reg_devices:
            selected_label = normalize_ws(device.get("txtNumeroArtigoCompleto"))
            targets = extract_reference_targets(device)
            if not targets:
                targets = [format_article(device.get("txtNumeroArtigo"))]
            for target in targets:
                parent_key, anchor_mode = resolve_anchor(target, path_to_row_key=path_to_row_key, article_to_row_key=article_to_row_key)
                anchor_modes[anchor_mode] += 1
                qid = normalize_ws(detail.get("numInternoQOrdem"))
                bloco = normalize_ws(device.get("_bloco")) or "questao"
                target_slug = slugify(target, max_len=48)
                row_key = f"qordem:{qid}:{bloco}:{target_slug}"
                if row_key in existing_keys or row_key in seen_row_keys:
                    continue
                seen_row_keys.add(row_key)
                generated_per_qo[qid] += 1
                max_id += 1
                max_ordem += 1
                title = f"QO {normalize_ws(detail.get('numQOrdemComAno'))} - {target}"
                source_url = f"{DEFAULT_SOURCE_PAGE}/{qid}"
                anchor_hierarchy = row_key_to_hierarchy.get(parent_key, RICD_NORMA_TITULO)
                text_rag = build_text_rag(
                    detail,
                    selected_label=selected_label,
                    target=target,
                    anchor_hierarchy=anchor_hierarchy,
                    all_reg_devices=all_reg_text,
                    other_refs=refs,
                    max_chars=max_text_rag_chars,
                )
                ementa_decisao = normalize_ws(detail.get("txtEmentaDecisao"))
                ementa_qo = normalize_ws(detail.get("txtEmentaQOrdem"))
                resumo = limit_text(ementa_decisao or ementa_qo or first_sentence(detail.get("txtQOrdem")), 650)
                notas_texto = normalize_text_block(
                    "\n".join(
                        part
                        for part in [
                            f"Fonte: {source_url}",
                            f"Dispositivo regimental selecionado: {selected_label}",
                            f"Ementa da questão: {ementa_qo}",
                            f"Ementa da decisão: {ementa_decisao}",
                            f"Decisão: {limit_text(detail.get('txtDecisao'), 9000)}",
                            f"Recurso: {limit_text(detail.get('txtRecurso'), 4000)}",
                            f"CCJR: {limit_text(detail.get('txtEmentaCCJR'), 2500)}",
                        ]
                        if normalize_ws(part.split(":", 1)[-1] if ":" in part else part)
                    )
                )
                referencias_normativas = json.dumps(
                    {
                        "regimentais": all_regimental_devices(detail),
                        "outras": refs,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                palavras = "; ".join(
                    item
                    for item in [
                        "questao de ordem",
                        "RICD",
                        normalize_ws(detail.get("numQOrdemComAno")),
                        normalize_ws(detail.get("txtNomeAutorQOrdem")),
                        normalize_ws(detail.get("txtNomePresidenteDecisao")),
                        normalize_ws(detail.get("txtIndexacaoQOrdem")),
                        normalize_ws(detail.get("txtIndexacaoDecisao")),
                    ]
                    if item
                )
                row: Dict[str, str] = {column: "" for column in vademecum.CSV_COLUMNS}
                row.update(
                    {
                        "id": str(max_id),
                        "ordem_global": str(max_ordem),
                        "row_key": row_key,
                        "titulo_linha": title,
                        "tipo_linha": "questao_ordem",
                        "tipo_base": "camara_qordem",
                        "linha_sintetica": "false",
                        "incluir_no_rag": "true",
                        "norma_id": RICD_NORMA_ID,
                        "norma_titulo": RICD_NORMA_TITULO,
                        "norma_numero": "17",
                        "norma_ano": "1989",
                        "norma_nome_popular": RICD_NOME_POPULAR,
                        "grupo": "camara_deputados",
                        "fonte_tipo": "camara_qordem_api",
                        "fonte_database_titulo": "Camara dos Deputados - Questoes de Ordem",
                        "fonte_database_id": "",
                        "fonte_data_source_id": "",
                        "fonte_page_id": source_url,
                        "id_fonte": qid,
                        "ordem_doc": normalize_ws(detail.get("numQOrdem")),
                        "tipo_dispositivo": "questao_ordem",
                        "dispositivo": title,
                        "dispositivo_pai": target,
                        "dispositivo_pai_row_key": parent_key,
                        "hierarquia_normativa": f"{anchor_hierarchy} > {title}" if anchor_hierarchy else title,
                        "documento_extraido": "Camara dos Deputados - Questao de Ordem",
                        "texto_dispositivo": normalize_text_block("\n".join(part for part in [ementa_qo, ementa_decisao] if part)),
                        "texto_em_vigor": normalize_text_block("\n".join(part for part in [ementa_qo, ementa_decisao] if part)),
                        "notas_texto": notas_texto,
                        "notas_json": compact_detail_json(detail, device, target, anchor_mode),
                        "qtd_notas": "1",
                        "referencias_normativas": referencias_normativas,
                        "referencias_jurisprudenciais": f"Questão de Ordem {normalize_ws(detail.get('numQOrdemComAno'))} - Câmara dos Deputados",
                        "resumo_curto": resumo,
                        "palavras_chave": palavras,
                        "alerta_qualidade": "" if anchor_mode != "norma_fallback" else "ancora_ricd_nao_encontrada",
                        "qualidade_texto": "ok" if anchor_mode != "norma_fallback" else "revisar",
                        "texto_rag": text_rag,
                        "chars": str(len(text_rag)),
                        "tokens_estimados": str(estimate_tokens(text_rag)),
                    }
                )
                rows.append(row)

    report = {
        "detalhes_total": len(details),
        "questoes_sem_dispositivo_regimental": skipped_no_regimental,
        "linhas_qordem_geradas": len(rows),
        "questoes_com_linhas": len(generated_per_qo),
        "anchor_modes": dict(anchor_modes),
        "linhas_com_alerta_qualidade": sum(1 for row in rows if row.get("alerta_qualidade")),
        "row_keys_duplicadas_geracao": len(rows) - len({row["row_key"] for row in rows}),
    }
    return rows, report


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=vademecum.CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_rows_to_vademecum(path: Path, rows: Sequence[Mapping[str, str]]) -> int:
    existing_keys = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            existing_keys = {normalize_ws(row.get("row_key")) for row in reader if normalize_ws(row.get("row_key"))}
    pending = [row for row in rows if normalize_ws(row.get("row_key")) not in existing_keys]
    if not pending:
        return 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=vademecum.CSV_COLUMNS, extrasaction="ignore")
        writer.writerows(pending)
    return len(pending)


def validate_qordem_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    keys = [normalize_ws(row.get("row_key")) for row in rows]
    return {
        "rows": len(rows),
        "duplicate_row_keys": len(keys) - len(set(keys)),
        "missing_parent_key": sum(1 for row in rows if not normalize_ws(row.get("dispositivo_pai_row_key"))),
        "missing_texto_rag": sum(1 for row in rows if not normalize_ws(row.get("texto_rag"))),
        "missing_source_url": sum(1 for row in rows if not normalize_ws(row.get("fonte_page_id"))),
        "quality_review": sum(1 for row in rows if normalize_ws(row.get("qualidade_texto")) != "ok"),
    }


def import_qordem_to_notion(
    rows: Sequence[Mapping[str, str]],
    *,
    checkpoint_path: Path,
    vademecum_page_url: str,
    database_title: str,
    workers: int,
    checkpoint_every: int,
    max_rich_text_chars: int,
    rate_rps: float,
    timeout_s: int,
    max_retries: int,
) -> Dict[str, Any]:
    token = notion_import.resolve_notion_token()
    if not token:
        raise RuntimeError("Token Notion nao encontrado. Configure NOTION_API_KEY ou Chave_Notion.txt.")
    target_page_id = notion_import.extract_notion_id_from_url(vademecum_page_url)
    client = notion_import.NotionClient(
        token,
        notion_version=notion_import.DEFAULT_NOTION_VERSION,
        timeout_s=timeout_s,
        max_retries=max_retries,
        rate_rps=rate_rps,
    )
    checkpoint = vademecum.import_pages(
        client,
        rows,
        target_page_id=target_page_id,
        database_title=database_title,
        checkpoint_path=checkpoint_path,
        workers=max(1, workers),
        checkpoint_every=max(1, checkpoint_every),
        max_rich_text_chars=max(2000, max_rich_text_chars),
    )
    relation_result = vademecum.update_parent_relations(
        client,
        rows,
        checkpoint_path=checkpoint_path,
        workers=max(1, workers),
        checkpoint_every=max(1, checkpoint_every),
    )
    checkpoint = vademecum.read_checkpoint(checkpoint_path)
    created_pages = checkpoint.get("created_pages") or {}
    relation_updates = checkpoint.get("relation_updates") or {}
    row_keys = {normalize_ws(row.get("row_key")) for row in rows}
    parent_keys = {normalize_ws(row.get("dispositivo_pai_row_key")) for row in rows if normalize_ws(row.get("dispositivo_pai_row_key"))}
    validation = {
        "rows": len(rows),
        "missing_pages": sum(1 for key in row_keys if key not in created_pages),
        "missing_parent_pages": sum(1 for key in parent_keys if key not in created_pages),
        "relations_marked": sum(1 for key in row_keys if key in relation_updates),
        "relation_result": relation_result,
        "database_url": checkpoint.get("database_url", ""),
        "data_source_id": checkpoint.get("data_source_id", ""),
    }
    checkpoint["qordem_import_validation"] = validation
    checkpoint["qordem_import_completed_at"] = notion_import.now_iso()
    vademecum.write_checkpoint(checkpoint_path, checkpoint)
    return validation


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera/importa linhas RAG de Questões de Ordem da Câmara ligadas ao RICD.")
    parser.add_argument("--api-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--vademecum-csv", type=Path, default=DEFAULT_VADEMECUM_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--details-json", type=Path, default=DEFAULT_DETAILS_JSON)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vademecum-page-url", default=vademecum.DEFAULT_VADEMECUM_PAGE_URL)
    parser.add_argument("--database-title", default=vademecum.DEFAULT_DATABASE_TITLE)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--csv-only", action="store_true")
    parser.add_argument("--append-vademecum", action="store_true")
    parser.add_argument("--import-notion", action="store_true")
    parser.add_argument("--exclude-review-from-append-import", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rate-rps", type=float, default=5.0)
    parser.add_argument("--notion-rate-rps", type=float, default=2.7)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--max-details", type=int, default=0)
    parser.add_argument("--max-text-rag-chars", type=int, default=28000)
    parser.add_argument("--max-rich-text-chars", type=int, default=30000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    existing_rows = read_csv_rows(args.vademecum_csv)
    if args.from_cache:
        details = read_details_cache(args.details_json)
    else:
        client = QOrdemClient(args.api_base_url, timeout_s=args.timeout_s, max_retries=args.max_retries, rate_rps=args.rate_rps)
        details = fetch_all_details(
            client,
            cache_path=args.details_json,
            workers=max(1, args.workers),
            checkpoint_every=max(1, args.checkpoint_every),
            max_pages=max(0, args.max_pages),
            max_details=max(0, args.max_details),
        )
    qordem_rows, report = build_qordem_rows(details, existing_rows=existing_rows, max_text_rag_chars=max(2000, args.max_text_rag_chars))
    validation = validate_qordem_rows(qordem_rows)
    report["validacao_local"] = validation
    write_rows_csv(args.output_csv, qordem_rows)
    LOGGER.info("CSV qordem escrito em %s com %s linhas.", args.output_csv, len(qordem_rows))
    sync_rows = [row for row in qordem_rows if normalize_ws(row.get("qualidade_texto")) == "ok"] if args.exclude_review_from_append_import else qordem_rows
    report["linhas_selecionadas_para_append_import"] = len(sync_rows)
    appended = 0
    if args.append_vademecum:
        appended = append_rows_to_vademecum(args.vademecum_csv, sync_rows)
        LOGGER.info("Linhas qordem anexadas ao CSV consolidado: %s.", appended)
    notion_validation: Dict[str, Any] = {}
    if args.import_notion and not args.csv_only:
        notion_validation = import_qordem_to_notion(
            sync_rows,
            checkpoint_path=args.checkpoint,
            vademecum_page_url=args.vademecum_page_url,
            database_title=args.database_title,
            workers=max(1, args.workers),
            checkpoint_every=max(1, args.checkpoint_every),
            max_rich_text_chars=max(2000, args.max_rich_text_chars),
            rate_rps=args.notion_rate_rps,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
        )
        LOGGER.info("Validacao import Notion qordem: %s", json.dumps(notion_validation, ensure_ascii=False, sort_keys=True))
    report["append_vademecum"] = {"enabled": bool(args.append_vademecum), "appended": appended}
    report["notion"] = notion_validation
    report["fetch_errors"] = dict(FETCH_ERRORS)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Relatorio escrito em %s: %s", args.report_json, json.dumps(report, ensure_ascii=False, sort_keys=True))
    if validation["duplicate_row_keys"] or validation["missing_parent_key"] or validation["missing_texto_rag"]:
        raise RuntimeError(f"Validacao local bloqueante falhou: {validation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
