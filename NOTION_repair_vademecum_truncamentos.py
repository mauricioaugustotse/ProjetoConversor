#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import threading
import unicodedata
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests
from lxml import html

import NOTION_import_codigo_eleitoral_2026_csvs as notion_import
import NOTION_import_vademecum_consolidado as vademecum


DEFAULT_CSV = Path("vademecum_rag_consolidado.csv")
DEFAULT_CHECKPOINT = Path("vademecum_rag_consolidado_checkpoint.json")
DEFAULT_REPAIR_REPORT = Path("vademecum_truncamentos_repair_report.json")

LOGGER = logging.getLogger("notion_vademecum_truncamentos_repair")

PLANALTO_URLS = {
    "constituicao_federal_1988": "https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm",
    "lei_4737_1965": "https://www.planalto.gov.br/ccivil_03/leis/l4737compilado.htm",
    "lei_9504_1997": "https://www.planalto.gov.br/ccivil_03/leis/l9504.htm",
    "lei_10406_2002": "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
    "decreto_lei_5452_1943": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm",
    "lei_9503_1997": "https://www.planalto.gov.br/ccivil_03/leis/l9503compilado.htm",
    "lei_15190": "https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2025/lei/l15190.htm",
}

MANUAL_REPAIRS = {
    "disp:decreto_lei_5452_1943:clt:94": (
        "Art. 29. O empregador terá o prazo de 5 (cinco) dias úteis para anotar na CTPS, em relação aos trabalhadores que admitir, "
        "a data de admissão, a remuneração e as condições especiais, se houver, facultada a adoção de sistema manual, mecânico ou eletrônico, "
        "conforme instruções a serem expedidas pelo Ministério da Economia. (Redação dada pela Lei nº 13.874, de 2019)\n"
        "§5º O descumprimento do disposto no §4º deste artigo submeterá o empregador ao pagamento de multa prevista no art. 52 deste Capítulo. "
        "(Incluído pela Lei nº 10.270, de 29.8.2001)"
    ),
    "disp:decreto_lei_5452_1943:clt:394": (
        "Art. 158 - Cabe aos empregados: (Redação dada pela Lei nº 6.514, de 22.12.1977)\n"
        "I - observar as normas de segurança e medicina do trabalho, inclusive as instruções de que trata o item II do artigo anterior; "
        "(Redação dada pela Lei nº 6.514, de 22.12.1977)\n"
        "II - colaborar com a empresa na aplicação dos dispositivos deste Capítulo. (Redação dada pela Lei nº 6.514, de 22.12.1977)"
    ),
    "disp:decreto_lei_5452_1943:clt:1187": (
        "Art. 453 - No tempo de serviço do empregado, quando readmitido, serão computados os períodos, ainda que não contínuos, "
        "em que tiver trabalhado anteriormente na empresa, salvo se houver sido despedido por falta grave, recebido indenização legal "
        "ou se aposentado espontaneamente. (Redação dada pela Lei nº 6.204, de 29.4.1975)\n"
        "§2º O ato de concessão de benefício de aposentadoria a empregado que não tiver completado 35 (trinta e cinco) anos de serviço, "
        "se homem, ou trinta, se mulher, importa em extinção do vínculo empregatício. (Incluído pela Lei nº 9.528, de 10.12.1997) "
        "(Vide ADIN 1.721) (Vide ADIN 1.770-3)"
    ),
    "disp:lei_9503_1997:ctb:860": (
        "Art. 162. Dirigir veículo:\n"
        "VII - sem possuir os cursos especializados ou específicos obrigatórios: (Incluído pela Lei nº 14.440, de 2022)\n"
        "Infração - gravíssima; (Incluído pela Lei nº 14.440, de 2022)\n"
        "Penalidade - multa; (Incluído pela Lei nº 14.440, de 2022)\n"
        "Medida administrativa - retenção do veículo até a apresentação de condutor habilitado. (Incluído pela Lei nº 14.440, de 2022)"
    ),
}


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\xa0", " ").replace("\ufeff", " ")).strip()


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


def comparison_key(text: Any) -> str:
    value = ascii_key(normalize_ws(text))
    value = re.sub(r"[^\w]+", " ", value)
    return normalize_ws(value)


def display_norm(text: Any) -> str:
    value = normalize_ws(text)
    value = re.sub(r"Art\.\s*(\d+(?:\.\d+)?[A-Z]?(?:-[A-Z])?)\s*[ºo]\b", r"Art. \1º", value)
    value = re.sub(r"§\s*(\d+)\s*[ºo]\b", r"§\1º", value)
    value = re.sub(r"§\s*(\d+)(?:º)?", r"§\1º", value)
    value = re.sub(r"Art\.\s*", "Art. ", value)
    value = re.sub(r"§\s+", "§", value)
    return value


def device_part_key(part: str) -> str:
    value = display_norm(part)
    return re.sub(r"^(Art\.\s*\d+(?:\.\d+)?[A-Z]?(?:-[A-Z])?)º$", r"\1", value)


def fix_annotation_parentheses(text: str) -> str:
    value = normalize_text_block(text)
    # Planalto pages sometimes omit the closing parenthesis inside annotation links.
    value = re.sub(
        r"(\((?:Reda[cç][aã]o dada|Inclu[ií]d[ao]|Renumerado|Par[aá]grafo renumerado|Revogado)[^()]*?\d{4})(\s+\((?:Vide|Vig[eê]ncia|Produ[cç][aã]o de efeitos))",
        r"\1) \2",
        value,
        flags=re.IGNORECASE,
    )
    value = value.replace("( (Vide", "(Vide").replace("( (", "(")
    diff = value.count("(") - value.count(")")
    if diff > 0:
        value += ")" * diff
    return value


def source_label_from_text(text: str) -> str:
    value = display_norm(text)
    match = re.match(r"^(Art\.\s*\d+(?:\.\d+)?[A-Z]?(?:-[A-Z])?)\s*[ºo]?\s*[\.-]?", value)
    if match:
        return device_part_key(match.group(1))
    if re.match(r"^Parágrafo único", value, flags=re.IGNORECASE):
        return "Parágrafo único"
    match = re.match(r"^(§\d+º?)\s*[\.-]?", value)
    if match:
        return device_part_key(match.group(1))
    match = re.match(r"^([IVXLCDM]+)\s*[-–]", value)
    if match:
        return match.group(1)
    match = re.match(r"^([a-z])\)", value)
    if match:
        return f"{match.group(1)})"
    return ""


class PlanaltoSource:
    def __init__(self, url: str) -> None:
        self.url = url
        self.by_article_and_label: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        response = requests.get(self.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
        response.raise_for_status()
        text = response.content.decode("latin1", errors="ignore")
        root = html.fromstring(text)
        current_article = ""
        for element in root.xpath("//p"):
            if element.xpath("ancestor-or-self::strike"):
                continue
            paragraph = display_norm(element.text_content())
            if len(paragraph) < 15:
                continue
            label = source_label_from_text(paragraph)
            if not label:
                continue
            if label.startswith("Art."):
                current_article = label
            if current_article:
                self.by_article_and_label[(current_article, label)].append(fix_annotation_parentheses(paragraph))

    def choose(self, article: str, label: str, current_text: str) -> str:
        candidates = self.by_article_and_label.get((article, label), [])
        if not candidates:
            return ""
        current_cmp = comparison_key(current_text)
        best = ""
        best_score = -1.0
        for candidate in candidates:
            candidate_cmp = comparison_key(candidate)
            score = SequenceMatcher(None, current_cmp[:1400], candidate_cmp[:1400]).ratio()
            if candidate_cmp[:80] and candidate_cmp[:80] in current_cmp:
                score += 1.0
            if current_cmp[:80] and current_cmp[:80] in candidate_cmp:
                score += 1.0
            score += min(len(candidate_cmp), 3000) / 100000.0
            if score > best_score:
                best = candidate
                best_score = score
        return best


def official_candidate(row: Mapping[str, str], sources: Dict[str, PlanaltoSource]) -> str:
    norma_id = normalize_ws(row.get("norma_id"))
    url = PLANALTO_URLS.get(norma_id)
    if not url:
        return ""
    if norma_id not in sources:
        LOGGER.info("Carregando fonte oficial: %s", url)
        sources[norma_id] = PlanaltoSource(url)
    source = sources[norma_id]
    parts = [device_part_key(part) for part in normalize_ws(row.get("hierarquia_normativa")).split(",") if normalize_ws(part)]
    if not parts:
        return ""
    article = parts[0]
    current = row.get("texto_em_vigor", "")
    components: List[str] = []
    article_text = source.choose(article, article, current)
    if article_text:
        components.append(article_text)
    for part in parts[1:]:
        part_text = source.choose(article, part, current)
        if part_text and part_text not in components:
            components.append(part_text)
    if not components and len(parts) == 1:
        components.append(source.choose(article, article, current))
    return fix_annotation_parentheses("\n".join(component for component in components if component))


def build_repaired_text(row: Mapping[str, str], sources: Dict[str, PlanaltoSource]) -> Tuple[str, str]:
    row_key = normalize_ws(row.get("row_key"))
    current = normalize_text_block(row.get("texto_em_vigor"))
    if row_key in MANUAL_REPAIRS:
        return normalize_text_block(MANUAL_REPAIRS[row_key]), "manual"
    if normalize_ws(row.get("alerta_qualidade")) == "possivel_truncamento_referencia_final":
        return current, "falso_positivo_referencia_final"
    candidate = official_candidate(row, sources)
    if candidate and len(candidate) >= len(current):
        return candidate, "fonte_oficial"
    repaired = fix_annotation_parentheses(current)
    if len(repaired) > len(current) or repaired != current:
        return repaired, "balanceamento_parenteses"
    return current, "sem_mudanca"


def recompute_row(row: Dict[str, str], new_text: str) -> Dict[str, str]:
    row["texto_dispositivo"] = new_text
    row["texto_em_vigor"] = new_text
    row["referencias_normativas"] = vademecum.extract_normative_refs(row.get("texto_em_vigor", ""), row.get("notas_texto", ""))
    row["referencias_jurisprudenciais"] = vademecum.extract_juris_refs(row.get("texto_em_vigor", ""), row.get("notas_texto", ""))
    alert, quality = vademecum.quality_alert(row.get("texto_em_vigor", ""))
    row["alerta_qualidade"] = alert
    row["qualidade_texto"] = quality
    row["resumo_curto"] = vademecum.make_summary(row)
    row["texto_rag"] = vademecum.build_text_rag(row)
    row["chars"] = str(len(row["texto_rag"]))
    row["tokens_estimados"] = str(vademecum.estimate_tokens(row["texto_rag"]))
    return row


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=vademecum.CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def rich_text_property(text: Any) -> Dict[str, Any]:
    return notion_import.rich_text_property(text, max_chars=30000)


def build_update_properties(row: Mapping[str, str]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for column in [
        "texto_dispositivo",
        "texto_em_vigor",
        "referencias_normativas",
        "referencias_jurisprudenciais",
        "resumo_curto",
        "alerta_qualidade",
        "texto_rag",
    ]:
        props[column] = rich_text_property(row.get(column, ""))
    props["qualidade_texto"] = notion_import.select_property(row.get("qualidade_texto", ""))
    props["chars"] = notion_import.number_property(row.get("chars", ""))
    props["tokens_estimados"] = notion_import.number_property(row.get("tokens_estimados", ""))
    return props


def update_page(client: notion_import.NotionClient, page_id: str, row: Mapping[str, str]) -> None:
    client.request("PATCH", f"/v1/pages/{page_id}", json_body={"properties": build_update_properties(row)})


def update_notion_rows(
    client: notion_import.NotionClient,
    rows_by_key: Mapping[str, Mapping[str, str]],
    changed_keys: Sequence[str],
    *,
    checkpoint_path: Path,
    workers: int,
    checkpoint_every: int,
) -> Dict[str, int]:
    checkpoint = vademecum.read_checkpoint(checkpoint_path)
    created_pages = checkpoint.get("created_pages") or {}
    repair_checkpoint = checkpoint.setdefault("truncation_repair_v1", {})
    updated = repair_checkpoint.setdefault("updated_row_keys", {})
    pending: List[Tuple[str, str]] = []
    for row_key in changed_keys:
        if updated.get(row_key):
            continue
        page_id = created_pages.get(row_key)
        if not page_id:
            raise RuntimeError(f"Pagina Notion nao encontrada no checkpoint para row_key={row_key}")
        pending.append((row_key, page_id))
    LOGGER.info("Paginas pendentes para atualizar no Notion: %s de %s.", len(pending), len(changed_keys))
    submitted = 0
    done_count = 0
    in_flight: Dict[Future[None], str] = {}
    lock = threading.Lock()
    max_in_flight = max(1, workers * 3)

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal submitted
        while submitted < len(pending) and len(in_flight) < max_in_flight:
            row_key, page_id = pending[submitted]
            submitted += 1
            future = executor.submit(update_page, client, page_id, rows_by_key[row_key])
            in_flight[future] = row_key

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_next(executor)
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                row_key = in_flight.pop(future)
                future.result()
                with lock:
                    updated[row_key] = notion_import.now_iso()
                    done_count += 1
                    if done_count % checkpoint_every == 0 or done_count == len(pending):
                        repair_checkpoint["updated_count"] = len(updated)
                        repair_checkpoint["updated_at"] = notion_import.now_iso()
                        vademecum.write_checkpoint(checkpoint_path, checkpoint)
                        LOGGER.info("Atualizacao Notion: %s/%s.", done_count, len(pending))
            submit_next(executor)
    repair_checkpoint["updated_count"] = len(updated)
    repair_checkpoint["updated_at"] = notion_import.now_iso()
    vademecum.write_checkpoint(checkpoint_path, checkpoint)
    return {"pending": len(pending), "updated": done_count, "total_marked": len(updated)}


def validate(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    return {
        "rows": len(rows),
        "quality_verify": sum(1 for row in rows if row.get("qualidade_texto") == "verificar"),
        "quality_critical": sum(1 for row in rows if row.get("qualidade_texto") == "critico"),
        "alert_nonblank": sum(1 for row in rows if normalize_ws(row.get("alerta_qualidade"))),
        "texto_rag_blank": sum(1 for row in rows if normalize_ws(row.get("incluir_no_rag")).lower() == "true" and not normalize_ws(row.get("texto_rag"))),
    }


def repair_csv(path: Path, *, report_path: Path) -> Tuple[List[Dict[str, str]], List[str], Dict[str, Any]]:
    rows = read_rows(path)
    sources: Dict[str, PlanaltoSource] = {}
    changed_keys: List[str] = []
    methods: Dict[str, int] = defaultdict(int)
    details: List[Dict[str, str]] = []
    for row in rows:
        if row.get("qualidade_texto") != "verificar":
            continue
        old_text = normalize_text_block(row.get("texto_em_vigor"))
        new_text, method = build_repaired_text(row, sources)
        methods[method] += 1
        old_alert = row.get("alerta_qualidade", "")
        recompute_row(row, new_text)
        if (
            new_text != old_text
            or row.get("qualidade_texto") != "verificar"
            or normalize_ws(row.get("alerta_qualidade")) != normalize_ws(old_alert)
        ):
            changed_keys.append(row["row_key"])
            details.append(
                {
                    "row_key": row["row_key"],
                    "norma_nome_popular": row.get("norma_nome_popular", ""),
                    "hierarquia_normativa": row.get("hierarquia_normativa", ""),
                    "metodo": method,
                    "old_chars": str(len(old_text)),
                    "new_chars": str(len(new_text)),
                    "old_alerta": old_alert,
                    "new_alerta": row.get("alerta_qualidade", ""),
                    "new_qualidade": row.get("qualidade_texto", ""),
                }
            )
    report = {
        "changed_rows": len(changed_keys),
        "methods": dict(sorted(methods.items())),
        "validation": validate(rows),
        "details": details,
        "official_sources": PLANALTO_URLS,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_rows(path, rows)
    return rows, changed_keys, report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repara truncamentos sinalizados na base Vademecum - RAG consolidado.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPAIR_REPORT)
    parser.add_argument("--csv-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-rps", type=float, default=2.7)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows, changed_keys, report = repair_csv(args.csv, report_path=args.report)
    LOGGER.info("Relatorio: %s", json.dumps({k: report[k] for k in ("changed_rows", "methods", "validation")}, ensure_ascii=False, sort_keys=True))
    if report["validation"]["quality_verify"] or report["validation"]["quality_critical"] or report["validation"]["alert_nonblank"] or report["validation"]["texto_rag_blank"]:
        raise RuntimeError("Validacao local ainda encontrou alertas ou campos RAG vazios.")
    if args.csv_only:
        return 0
    token = notion_import.resolve_notion_token()
    if not token:
        raise RuntimeError("Token Notion nao encontrado. Configure NOTION_API_KEY ou Chave_Notion.txt.")
    client = notion_import.NotionClient(
        token,
        notion_version=notion_import.DEFAULT_NOTION_VERSION,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        rate_rps=args.rate_rps,
    )
    rows_by_key = {row["row_key"]: row for row in rows}
    result = update_notion_rows(
        client,
        rows_by_key,
        changed_keys,
        checkpoint_path=args.checkpoint,
        workers=max(1, int(args.workers or 1)),
        checkpoint_every=max(1, int(args.checkpoint_every or 1)),
    )
    checkpoint = vademecum.read_checkpoint(args.checkpoint)
    checkpoint.setdefault("truncation_repair_v1", {})["result"] = result
    checkpoint["truncation_repair_v1"]["report_path"] = str(args.report)
    checkpoint["truncation_repair_v1"]["completed_at"] = notion_import.now_iso()
    vademecum.write_checkpoint(args.checkpoint, checkpoint)
    LOGGER.info("Resultado Notion: %s", json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
