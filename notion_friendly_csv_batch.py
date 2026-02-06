#!/usr/bin/env python3
"""Converte CSVs de jurisprudencia para um formato "Notion-friendly".

Visao geral do fluxo:
1. Le o CSV de entrada (encoding/delimitador detectados automaticamente).
2. Normaliza cabecalhos e celulas.
3. Aplica regras juridicas/estruturais por coluna.
4. (Opcional) Enriquecimento web via API da Perplexity para preencher URLs.
5. Gera um CSV *_notion.csv por arquivo e um compilado final.

Principais transformacoes:
- Remove colunas: numeroDecisao, numeroProtocolo.
- Limpa naturezaDocumento: "Sem anotacao" -> vazio.
- Formata numeroUnico no padrao CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO.
- Trunca apenas textoDecisao/textoEmenta (limite configuravel; 0 = sem truncamento).
- Partes e campos derivados em formato adequado para Notion:
  composicao (multi-select canonico), advogados (multi-select), resultado (multi-select).
- Normaliza referenciasLegislativas para formato simplificado.

Enriquecimento Perplexity (opcional):
- Preenche noticia_TSE, noticia_TRE e noticia_geral_1..9.
- Mantem URLs ja existentes (nao sobrescreve).
- Consulta apenas campos vazios.
- Precedencia da chave:
  --perplexity-api-key > PERPLEXITY_API_KEY > Chave_secreta_Perplexity.txt

Uso rapido (CLI):
- Basico:
  python3 notion_friendly_csv_batch.py arquivo1.csv arquivo2.csv --no-gui
- Com Perplexity:
  python3 notion_friendly_csv_batch.py arquivo.csv --no-gui --buscar-urls-perplexity
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
from urllib.parse import urlparse

import requests

ENCODINGS_TO_TRY = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
SPACE_RE = re.compile(r"\s+")
HEADER_ALLOWED_RE = re.compile(r"[^0-9A-Za-z_]+")
PARTES_SPLIT_RE = re.compile(r"[;,]")
RELATOR_PREFIX_RE = re.compile(r"^\s*relator(?:\(a\)|a|o)?\s*[:.\-]?\s*", re.IGNORECASE)
EXCLUDED_COLUMNS = {"numeroDecisao", "numeroProtocolo", "resuultado"}
URL_COLUMNS = ["noticia_TSE", "noticia_TRE"] + [f"noticia_geral_{i}" for i in range(1, 10)]
DEFAULT_PERPLEXITY_KEY_FILE = "Chave_secreta_Perplexity.txt"

CANONICAL_MINISTERS: list[tuple[str, tuple[str, ...]]] = [
    ("Min. Cármen Lúcia", ("carmen lucia",)),
    ("Min. Nunes Marques", ("nunes marques",)),
    ("Min. André Mendonça", ("andre mendonca",)),
    ("Min. Antonio Carlos Ferreira", ("antonio carlos ferreira",)),
    ("Min. Villas Bôas Cueva", ("villas boas cueva", "vilas boas cueva")),
    ("Min. Floriano de Azevedo Marques", ("floriano de azevedo marques",)),
    ("Min. Estela Aranha", ("estela aranha",)),
]

COMPOSICAO_TRIGGER_RE = re.compile(r"composi[çc][aã]o\s*:|acompanharam o relator", re.IGNORECASE)
COMPOSICAO_SEGMENT_RE = re.compile(
    r"composi[çc][aã]o\s*:[^.]{0,800}|acompanharam o relator[^.]{0,800}",
    re.IGNORECASE,
)
ADVOGADOS_BLOCK_RE = re.compile(
    r"(?is)advogad(?:o|a|os|as)\s*:\s*(.*?)(?=(?:\b(?:advogad(?:o|a|os|as)|"
    r"recorrente|recorrido|recorrida|agravante|agravado|agravada|representante|"
    r"relator(?:a)?|minist[eé]rio p[úu]blico eleitoral|decis[aã]o|ac[oó]rd[aã]o|"
    r"composi[çc][aã]o)\b\s*:)|$)"
)
ADVOGADO_NOME_OAB_RE = re.compile(
    r"([A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]+(?:\s+[A-ZÀ-ÖØ-Ýa-zà-öø-ÿ'`´^~.-]+){1,}?)"
    r"\s*(?:[-–—]\s*)?OAB\b",
    re.UNICODE,
)
LEG_SPLIT_RE = re.compile(r"\s*,\s*(?=LEG\.\s*:)", re.IGNORECASE)
LEGAL_ENTRY_RE = re.compile(
    r"(?i)(?P<tipo>LEI\s+COMPLEMENTAR|LEI\s+ORDIN[ÁA]RIA|LEI|DECRETO-LEI|"
    r"RESOLU[CÇ][AÃ]O\s+DO\s+TRIBUNAL\s+SUPERIOR\s+ELEITORAL|EMENDA\s+CONSTITUCIONAL|"
    r"CONSTITUI[CÇ][AÃ]O\s+FEDERAL)\s*N[ºo°]\.?\s*:\s*(?P<num>\d+)\s*Ano\s*:\s*(?P<ano>\d{4})"
)
ARTICLE_RE = re.compile(
    r"(?i)\bart\.?\s*\d+[º°]?(?:\s*,\s*(?:§\s*\d+[º°]?|inc\.?\s*[IVXLC]+|[IVXLC]+|al[íi]nea\s*[a-z]))*"
)
UNANIME_RE = re.compile(r"\bpor\s+unanimidade\b|\b[aà]\s+unanimidade\b", re.IGNORECASE)
MAIORIA_RE = re.compile(r"\bpor\s+maioria\b|\bmaioria\s+de\s+votos\b", re.IGNORECASE)
MONOCRATICA_RE = re.compile(r"\bdecis[aã]o\s+monocr[aá]tica\b", re.IGNORECASE)
NAO_CONHECIMENTO_NORM_RE = re.compile(
    r"\b(?:nao\s+conhec(?:imento|id[oa]s?|o|er(?:am|a|ei|emos|ia|iam)?|eu)|"
    r"nao-?conhec(?:imento|id[oa]s?)|nao\s+se\s+conhece|nao\s+foi\s+conhecid[oa]|"
    r"neg(?:o|ou|a(?:r)?|aram)\s+conhecimento)\b",
    re.IGNORECASE,
)
ADV_TITLES_PREFIX_RE = re.compile(r"^\s*(?:dr|dra|doutor|doutora)\.?\s+", re.IGNORECASE)
CNJ_DIGITS_RE = re.compile(r"\D+")

VEICULOS_GERAIS = [
    "folha de s.paulo",
    "estadao",
    "cnn brasil",
    "g1",
    "conjur",
    "migalhas",
    "uol",
    "poder360",
    "metropoles",
    "agencia brasil",
    "jota",
    "valor economico",
]

VEICULOS_DOMINIOS = [
    "folha.uol.com.br",
    "estadao.com.br",
    "cnnbrasil.com.br",
    "g1.globo.com",
    "conjur.com.br",
    "migalhas.com.br",
    "uol.com.br",
    "poder360.com.br",
    "metropoles.com",
    "agenciabrasil.ebc.com.br",
    "jota.info",
    "valor.globo.com",
]


@dataclass
class ProcessSummary:
    input_path: Path
    output_path: Path
    rows: int
    truncated_cells: int
    encoding: str
    delimiter: str
    fields: list[str]


@dataclass
class WebLookupConfig:
    enabled: bool = False
    api_key: str = ""
    model: str = "sonar"
    timeout_seconds: int = 15
    max_workers: int = 4
    batch_size: int = 20
    delay_between_batches: float = 0.3


def detect_csv_format(path: Path) -> tuple[str, str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS_TO_TRY:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(8192)
            delimiter = ","
            if sample:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ","
            return encoding, delimiter
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"Nao foi possivel detectar encoding de {path}: {last_error}")


def sanitize_header(name: str | None, used: set[str]) -> str:
    text = (name or "").replace("\ufeff", "").strip()
    text = CONTROL_CHAR_RE.sub("", text)
    text = SPACE_RE.sub("_", text)
    text = HEADER_ALLOWED_RE.sub("", text)

    if not text:
        text = "coluna"

    base = text
    counter = 2
    while text in used:
        text = f"{base}_{counter}"
        counter += 1

    used.add(text)
    return text


def sanitize_cell(value: object, max_chars: int, replace_newlines: bool) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "")
    text = text.replace("¿", "\"")
    text = CONTROL_CHAR_RE.sub(" ", text)

    if replace_newlines:
        text = (
            text.replace("\r\n", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("\t", " ")
        )

    text = SPACE_RE.sub(" ", text).strip()
    truncated = False
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    return text, truncated


def read_secret_from_file(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not raw:
        return ""

    first_line = raw.splitlines()[0].strip()
    if "=" in first_line:
        first_line = first_line.split("=", 1)[1].strip()
    first_line = first_line.strip("\"' ")
    return first_line


def resolve_perplexity_api_key(cli_value: str, key_file_path: str) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    env_value = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if env_value:
        return env_value
    file_value = read_secret_from_file(key_file_path)
    if file_value:
        return file_value
    return ""


class GerenciadorRequisicoes:
    def __init__(self, api_key: str, model: str, max_workers: int = 4) -> None:
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.sessao = requests.Session()
        self.sessao.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    async def call_perplexity(self, prompt: str, timeout: int = 15) -> Optional[dict[str, object]]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
        }
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.sessao.post(
                    "https://api.perplexity.ai/chat/completions",
                    json=payload,
                    timeout=timeout,
                ),
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            citations = result.get("citations") or []
            if not isinstance(citations, list):
                citations = []
            return {"content": content, "citations": citations}
        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None

    def close(self) -> None:
        self.sessao.close()
        self.executor.shutdown(wait=True)


def _limpar_url_bruta(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip().strip(").,;:!?]}>\"'")
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return None


def limpar_url(texto: str) -> Optional[str]:
    if not texto:
        return None
    match = re.search(r"https?://\S+", texto)
    if not match:
        return None
    return _limpar_url_bruta(match.group(0))


def extrair_urls_multiplas(texto: str) -> Optional[list[str]]:
    if not texto:
        return None
    encontrados = re.findall(r"https?://\S+", texto)
    if not encontrados:
        return None
    urls: list[str] = []
    vistos: set[str] = set()
    for url in encontrados:
        limpa = _limpar_url_bruta(url)
        if limpa and limpa not in vistos:
            vistos.add(limpa)
            urls.append(limpa)
    return urls if urls else None


def extrair_urls_de_citations(citations: object) -> list[str]:
    if not isinstance(citations, list):
        return []
    urls: list[str] = []
    vistos: set[str] = set()
    for item in citations:
        if not isinstance(item, str):
            continue
        limpa = _limpar_url_bruta(item)
        if limpa and limpa not in vistos:
            vistos.add(limpa)
            urls.append(limpa)
    return urls


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _is_tse_url(url: str) -> bool:
    return _host(url).endswith("tse.jus.br")


def _is_tse_news_url(url: str) -> bool:
    host = _host(url)
    return host.endswith("tse.jus.br") and "/comunicacao/noticias/" in url.lower()


def _is_tre_url(url: str) -> bool:
    host = _host(url)
    return ".jus.br" in host and "tre-" in host


def _is_tre_news_url(url: str) -> bool:
    if not _is_tre_url(url):
        return False
    lower = url.lower()
    return "/comunicacao/noticias/" in lower or "/noticias/" in lower


def _is_general_media_url(url: str) -> bool:
    host = _host(url)
    return any(host == domain or host.endswith("." + domain) for domain in VEICULOS_DOMINIOS)


def _pick_first(urls: Iterable[str], predicate: Callable[[str], bool]) -> Optional[str]:
    for url in urls:
        if predicate(url):
            return url
    return None


def compact_text(value: Optional[str], max_len: int) -> str:
    if not value:
        return ""
    texto = " ".join(value.split())
    if len(texto) > max_len:
        return texto[: max_len - 3] + "..."
    return texto


def build_lookup_payload(row: dict[str, str]) -> dict[str, str]:
    tribunal = " ".join(
        part for part in (row.get("siglaTribunalJE", ""), row.get("origemDecisao", "")) if part
    ).strip()
    origem = " ".join(part for part in (row.get("siglaUF", ""), row.get("nomeMunicipio", "")) if part).strip()
    ementa = row.get("textoEmenta") or ""
    decisao = row.get("textoDecisao") or ""
    punchline = ementa or decisao or (row.get("assuntos") or "")
    tema = row.get("assuntos") or ementa or ""
    return {
        "numero_processo": row.get("numeroProcesso") or row.get("numeroUnico") or "",
        "classe_processo": row.get("descricaoClasse") or row.get("siglaClasse") or "",
        "tema": tema,
        "partes": row.get("partes") or "",
        "tribunal": tribunal,
        "data_sessao": row.get("dataDecisao") or "",
        "origem": origem,
        "punchline": punchline,
    }


def gerar_prompt_tse(page_data: dict[str, str]) -> str:
    numero = compact_text(page_data.get("numero_processo"), 120)
    classe = compact_text(page_data.get("classe_processo"), 80)
    tema = compact_text(page_data.get("tema"), 200)
    tribunal = compact_text(page_data.get("tribunal"), 120)
    data = compact_text(page_data.get("data_sessao"), 80)
    punchline = compact_text(page_data.get("punchline"), 200)

    return f"""Encontre uma notícia OFICIAL da Justiça Eleitoral APENAS no domínio tse.jus.br sobre:

Processo: {numero}
Classe: {classe}
Tema: {tema}
Tribunal: {tribunal}
Data da Sessão: {data}
Punchline: {punchline}

RETORNE APENAS o URL direto da notícia encontrada (começando com http), sem explicações.
Se NÃO encontrar, retorne: VAZIO"""


def gerar_prompt_tre(page_data: dict[str, str]) -> str:
    numero = compact_text(page_data.get("numero_processo"), 120)
    tema = compact_text(page_data.get("tema"), 200)
    origem = compact_text(page_data.get("origem"), 100)
    punchline = compact_text(page_data.get("punchline"), 200)

    return f"""Encontre uma notícia OFICIAL de um Tribunal Regional Eleitoral (TRE estadual) em domínios tre-XX.jus.br sobre:

Processo: {numero}
Tema: {tema}
Estado de Origem: {origem}
Punchline: {punchline}

RETORNE APENAS o URL direto da notícia (começando com http e contendo tre-), sem explicações.
Se NÃO encontrar, retorne: VAZIO"""


def gerar_prompt_gerais(page_data: dict[str, str]) -> str:
    numero = compact_text(page_data.get("numero_processo"), 120)
    tema = compact_text(page_data.get("tema"), 200)
    partes = compact_text(page_data.get("partes"), 200)
    punchline = compact_text(page_data.get("punchline"), 200)
    veiculos_str = ", ".join(VEICULOS_GERAIS)

    return f"""Encontre noticias sobre este caso eleitoral em grandes veículos de mídia brasileira.

Processo: {numero}
Tema: {tema}
Partes principais: {partes}
Resumo da decisão: {punchline}

Procure apenas nos seguintes veículos: {veiculos_str}.

REGRAS DE RESPOSTA (OBRIGATÓRIAS):
- Responda SOMENTE com uma lista de até 9 URLs.
- Cada URL deve aparecer sozinha em uma linha.
- Não escreva texto explicativo, títulos, comentários ou bullets.
- Se não encontrar nenhuma noticia, responda exatamente: VAZIO
"""


def gerar_prompt_gerais_com_sites(page_data: dict[str, str]) -> str:
    numero = compact_text(page_data.get("numero_processo"), 120)
    tema = compact_text(page_data.get("tema"), 200)
    partes = compact_text(page_data.get("partes"), 200)
    punchline = compact_text(page_data.get("punchline"), 200)
    sites_str = " ".join([f"site:{domain}" for domain in VEICULOS_DOMINIOS])

    return f"""Encontre noticias sobre este caso eleitoral nos domínios listados.

Processo: {numero}
Tema: {tema}
Partes principais: {partes}
Resumo da decisão: {punchline}

Restrição de domínios (obrigatória): {sites_str}

REGRAS DE RESPOSTA (OBRIGATÓRIAS):
- Responda SOMENTE com uma lista de até 9 URLs.
- Cada URL deve aparecer sozinha em uma linha.
- Não escreva texto explicativo, títulos, comentários ou bullets.
- Se não encontrar nenhuma noticia, responda exatamente: VAZIO
"""


async def buscar_todas_noticias_async(
    gerenciador: GerenciadorRequisicoes,
    page_data: dict[str, str],
    precisa_tse: bool,
    precisa_tre: bool,
    precisa_gerais: bool,
    timeout_seconds: int,
) -> tuple[Optional[str], Optional[str], Optional[list[str]]]:
    prompt_tse = gerar_prompt_tse(page_data) if precisa_tse else None
    prompt_tre = gerar_prompt_tre(page_data) if precisa_tre else None
    prompt_gerais = gerar_prompt_gerais(page_data) if precisa_gerais else None
    prompt_gerais_site = gerar_prompt_gerais_com_sites(page_data) if precisa_gerais else None

    async def _none() -> None:
        return None

    try:
        resposta_tse, resposta_tre, resposta_gerais, resposta_gerais_site = await asyncio.gather(
            gerenciador.call_perplexity(prompt_tse, timeout=timeout_seconds) if prompt_tse else _none(),
            gerenciador.call_perplexity(prompt_tre, timeout=timeout_seconds) if prompt_tre else _none(),
            gerenciador.call_perplexity(prompt_gerais, timeout=timeout_seconds) if prompt_gerais else _none(),
            gerenciador.call_perplexity(prompt_gerais_site, timeout=timeout_seconds) if prompt_gerais_site else _none(),
        )
    except Exception:
        return None, None, None

    def content_of(resp: object) -> str:
        if isinstance(resp, dict):
            return str(resp.get("content", "") or "")
        return ""

    def citations_of(resp: object) -> list[str]:
        if isinstance(resp, dict):
            return extrair_urls_de_citations(resp.get("citations"))
        return []

    # TSE: prioriza noticias oficiais e usa fallback para dominio geral do TSE apenas se necessario.
    tse_content = content_of(resposta_tse)
    tse_content_url = limpar_url(tse_content) if tse_content else None
    tse_citations = citations_of(resposta_tse)
    url_tse = None
    if tse_content_url and _is_tse_news_url(tse_content_url):
        url_tse = tse_content_url
    if not url_tse:
        url_tse = _pick_first(tse_citations, _is_tse_news_url)
    if not url_tse and tse_content_url and _is_tse_url(tse_content_url):
        url_tse = tse_content_url
    if not url_tse:
        url_tse = _pick_first(tse_citations, _is_tse_url)

    # TRE: prioriza noticias em dominios TRE, com fallback para paginas gerais do TRE.
    tre_content = content_of(resposta_tre)
    tre_content_url = limpar_url(tre_content) if tre_content else None
    tre_citations = citations_of(resposta_tre)
    url_tre = None
    if tre_content_url and _is_tre_news_url(tre_content_url):
        url_tre = tre_content_url
    if not url_tre:
        url_tre = _pick_first(tre_citations, _is_tre_news_url)
    if not url_tre and tre_content_url and _is_tre_url(tre_content_url):
        url_tre = tre_content_url
    if not url_tre:
        url_tre = _pick_first(tre_citations, _is_tre_url)

    urls_gerais: list[str] = []
    vistos: set[str] = set()
    for resposta in (resposta_gerais, resposta_gerais_site):
        texto = content_of(resposta)
        extraidas = None
        if texto:
            extraidas = extrair_urls_multiplas(texto)
            if extraidas:
                extraidas = [url for url in extraidas if _is_general_media_url(url)]
        if not extraidas:
            citations_media = [url for url in citations_of(resposta) if _is_general_media_url(url)]
            extraidas = citations_media if citations_media else None
        if not extraidas:
            continue
        for url in extraidas:
            if url not in vistos:
                vistos.add(url)
                urls_gerais.append(url)

    return url_tse, url_tre, (urls_gerais if urls_gerais else None)


def _aplicar_urls_no_row(row: dict[str, str], url_tse: Optional[str], url_tre: Optional[str], urls_gerais: Optional[list[str]]) -> None:
    if url_tse and not row.get("noticia_TSE"):
        row["noticia_TSE"] = url_tse
    if url_tre and not row.get("noticia_TRE"):
        row["noticia_TRE"] = url_tre

    gerais = urls_gerais or []
    if not gerais:
        return

    idx_url = 0
    for idx in range(1, 10):
        col = f"noticia_geral_{idx}"
        if row.get(col):
            continue
        if idx_url >= len(gerais):
            break
        row[col] = gerais[idx_url]
        idx_url += 1


async def enriquecer_rows_com_urls_async(
    rows: list[dict[str, str]],
    logger: Callable[[str], None],
    config: WebLookupConfig,
) -> None:
    if not rows:
        return
    gerenciador = GerenciadorRequisicoes(
        api_key=config.api_key,
        model=config.model,
        max_workers=config.max_workers,
    )
    try:
        total = len(rows)
        for start in range(0, total, config.batch_size):
            end = min(start + config.batch_size, total)
            lote = rows[start:end]
            logger(f"[Perplexity] Processando lote {start + 1}-{end} de {total}")
            tarefas = []
            for row in lote:
                precisa_tse = not bool((row.get("noticia_TSE") or "").strip())
                precisa_tre = not bool((row.get("noticia_TRE") or "").strip())
                precisa_gerais = any(not bool((row.get(f"noticia_geral_{i}") or "").strip()) for i in range(1, 10))
                tarefas.append(
                    buscar_todas_noticias_async(
                        gerenciador,
                        build_lookup_payload(row),
                        precisa_tse=precisa_tse,
                        precisa_tre=precisa_tre,
                        precisa_gerais=precisa_gerais,
                        timeout_seconds=config.timeout_seconds,
                    )
                )
            resultados = await asyncio.gather(*tarefas)
            for row, (url_tse, url_tre, urls_gerais) in zip(lote, resultados):
                _aplicar_urls_no_row(row, url_tse, url_tre, urls_gerais)
            if end < total and config.delay_between_batches > 0:
                await asyncio.sleep(config.delay_between_batches)
    finally:
        gerenciador.close()


def format_partes_as_multiselect(value: str) -> str:
    if not value:
        return ""

    seen: set[str] = set()
    ordered: list[str] = []
    for part in PARTES_SPLIT_RE.split(value):
        item = SPACE_RE.sub(" ", part).strip(" ,;")
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ",".join(ordered)


def clean_relator_prefix(value: str) -> str:
    if not value:
        return ""
    cleaned = RELATOR_PREFIX_RE.sub("", value)
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def clean_natureza_documento(value: str) -> str:
    if not value:
        return ""
    normalized = normalize_for_match(value)
    if normalized == "sem anotacao":
        return ""
    return value


def format_numero_unico_cnj(value: str) -> str:
    if not value:
        return ""
    digits = CNJ_DIGITS_RE.sub("", value)
    if len(digits) != 20:
        return value
    return (
        f"{digits[0:7]}-{digits[7:9]}.{digits[9:13]}."
        f"{digits[13]}.{digits[14:16]}.{digits[16:20]}"
    )


def normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized


def dedupe_preserve(items: Iterable[str], key_func: Callable[[str], str] | None = None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    key_callable = key_func or (lambda item: item)
    for item in items:
        key = key_callable(item)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def format_br_number(number: str) -> str:
    digits = re.sub(r"\D", "", number or "")
    if not digits:
        return ""
    return f"{int(digits):,}".replace(",", ".")


def format_law_reference(law_type: str, number: str, year: str) -> str:
    law_type_norm = normalize_for_match(law_type)
    number_fmt = format_br_number(number)
    year_short = year[-2:] if year else ""

    if "lei complementar" in law_type_norm:
        return f"LC n° {number_fmt}/{year_short}"
    if "decreto-lei" in law_type_norm:
        return f"Decreto-Lei n° {number_fmt}/{year_short}"
    if "resolucao do tribunal superior eleitoral" in law_type_norm:
        return f"Res. TSE n° {number_fmt}/{year_short}"
    if "emenda constitucional" in law_type_norm:
        return f"EC n° {number_fmt}/{year_short}"
    if "constituicao federal" in law_type_norm:
        return "CF/88"
    return f"Lei n° {number_fmt}/{year_short}"


def normalize_article(article_text: str) -> str:
    text = SPACE_RE.sub(" ", article_text).strip(" ,;")
    text = re.sub(r"(?i)^art\.?", "art.", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text


def clean_referencias_legislativas(value: str) -> str:
    if not value:
        return ""

    text = SPACE_RE.sub(" ", value).strip()
    entries = LEG_SPLIT_RE.split(text)
    laws: list[str] = []
    for entry in entries:
        match = LEGAL_ENTRY_RE.search(entry)
        if not match:
            continue
        law = format_law_reference(
            law_type=match.group("tipo"),
            number=match.group("num"),
            year=match.group("ano"),
        )
        if law:
            laws.append(law)
    laws = dedupe_preserve(laws, key_func=lambda item: item.casefold())

    articles = [normalize_article(m.group(0)) for m in ARTICLE_RE.finditer(text)]
    articles = dedupe_preserve(articles, key_func=lambda item: normalize_for_match(item))

    if not laws:
        return ""
    if articles and len(laws) == 1:
        combined = [f"{article}, {laws[0]}" for article in articles]
        return "; ".join(combined)
    if articles:
        return "; ".join(laws + articles)
    return "; ".join(laws)


def extract_composicao_multiselect(*texts: str) -> str:
    source = " ".join(item for item in texts if item)
    if not source:
        return ""
    if not COMPOSICAO_TRIGGER_RE.search(source):
        return ""

    segments = [match.group(0) for match in COMPOSICAO_SEGMENT_RE.finditer(source)]
    if not segments:
        segments = [source]

    search_space = normalize_for_match(" ".join(segments))
    found: list[str] = []
    for canonical_name, aliases in CANONICAL_MINISTERS:
        if any(alias in search_space for alias in aliases):
            found.append(canonical_name)

    return ",".join(found)


def extract_advogados_multiselect(*texts: str) -> str:
    source = " ".join(item for item in texts if item)
    if not source:
        return ""

    names: list[str] = []
    for block_match in ADVOGADOS_BLOCK_RE.finditer(source):
        block = SPACE_RE.sub(" ", block_match.group(1)).strip()
        if not block:
            continue
        for name_match in ADVOGADO_NOME_OAB_RE.finditer(block):
            name = SPACE_RE.sub(" ", name_match.group(1)).strip(" ,;.-")
            if name:
                cleaned_name = ADV_TITLES_PREFIX_RE.sub("", name).strip()
                if cleaned_name:
                    names.append(cleaned_name)

    names = dedupe_preserve(names, key_func=lambda item: normalize_for_match(item))
    return ",".join(names)


def classify_resultado(texto_decisao: str, descricao_tipo_decisao: str = "") -> str:
    scope = texto_decisao[:5000] if texto_decisao else ""
    labels: list[str] = []

    if UNANIME_RE.search(scope):
        labels.append("unânime")
    elif MAIORIA_RE.search(scope):
        labels.append("por maioria")
    elif MONOCRATICA_RE.search(scope):
        labels.append("monocrática")
    else:
        descricao_norm = normalize_for_match(descricao_tipo_decisao)
        if "monocratica" in descricao_norm:
            labels.append("monocrática")
        else:
            labels.append("monocrática")

    scope_norm = normalize_for_match(scope)
    if NAO_CONHECIMENTO_NORM_RE.search(scope_norm):
        labels.append("não conhecido")

    return ",".join(dedupe_preserve(labels, key_func=lambda item: item.casefold()))


def process_one_csv(
    input_path: Path,
    out_dir: Path,
    max_texto_chars: int,
    replace_newlines: bool,
    web_lookup_config: WebLookupConfig,
    logger: Callable[[str], None],
) -> ProcessSummary:
    encoding, delimiter = detect_csv_format(input_path)
    logger(f"[Lendo] {input_path.name} (encoding={encoding}, delimiter='{delimiter}')")

    with input_path.open("r", encoding=encoding, newline="") as src:
        reader = csv.DictReader(src, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{input_path.name}: sem cabecalho valido")

        used_headers: set[str] = set()
        clean_headers = [sanitize_header(name, used_headers) for name in reader.fieldnames]
        mapping = list(zip(reader.fieldnames, clean_headers))
        source_key_by_clean = {clean: original for original, clean in mapping}

        output_path = out_dir / f"{input_path.stem}_notion.csv"
        output_fields = [header for header in clean_headers if header not in EXCLUDED_COLUMNS]
        for derived in ("composicao", "advogados", "resultado"):
            if derived not in output_fields:
                output_fields.append(derived)
        for url_column in URL_COLUMNS:
            if url_column not in output_fields:
                output_fields.append(url_column)

        processed_rows: list[dict[str, str]] = []
        truncated_cells = 0

        for source_row in reader:
            clean_row: dict[str, str] = {}
            for original_key, clean_key in mapping:
                if clean_key in EXCLUDED_COLUMNS:
                    continue

                per_cell_limit = max_texto_chars if clean_key in {"textoEmenta", "textoDecisao"} else 0
                clean_value, was_truncated = sanitize_cell(
                    source_row.get(original_key, ""),
                    max_chars=per_cell_limit,
                    replace_newlines=replace_newlines,
                )
                if clean_key == "partes":
                    clean_value = format_partes_as_multiselect(clean_value)
                elif clean_key == "relatores":
                    clean_value = clean_relator_prefix(clean_value)
                elif clean_key == "referenciasLegislativas":
                    clean_value = clean_referencias_legislativas(clean_value)
                elif clean_key == "naturezaDocumento":
                    clean_value = clean_natureza_documento(clean_value)
                elif clean_key == "numeroUnico":
                    clean_value = format_numero_unico_cnj(clean_value)
                clean_row[clean_key] = clean_value
                if was_truncated:
                    truncated_cells += 1

            texto_decisao_key = source_key_by_clean.get("textoDecisao")
            texto_ementa_key = source_key_by_clean.get("textoEmenta")
            descricao_tipo_decisao_key = source_key_by_clean.get("descricaoTipoDecisao")
            texto_decisao_raw = source_row.get(texto_decisao_key, "") if texto_decisao_key else ""
            texto_ementa_raw = source_row.get(texto_ementa_key, "") if texto_ementa_key else ""
            descricao_tipo_decisao_raw = (
                source_row.get(descricao_tipo_decisao_key, "") if descricao_tipo_decisao_key else ""
            )
            texto_decisao_full, _ = sanitize_cell(texto_decisao_raw, max_chars=0, replace_newlines=replace_newlines)
            texto_ementa_full, _ = sanitize_cell(texto_ementa_raw, max_chars=0, replace_newlines=replace_newlines)
            descricao_tipo_decisao_full, _ = sanitize_cell(
                descricao_tipo_decisao_raw,
                max_chars=0,
                replace_newlines=replace_newlines,
            )

            clean_row["composicao"] = extract_composicao_multiselect(texto_decisao_full, texto_ementa_full)
            clean_row["advogados"] = extract_advogados_multiselect(texto_decisao_full, texto_ementa_full)
            clean_row["resultado"] = classify_resultado(
                texto_decisao=texto_decisao_full,
                descricao_tipo_decisao=descricao_tipo_decisao_full,
            )

            for url_column in URL_COLUMNS:
                clean_row[url_column] = clean_row.get(url_column, "")

            processed_rows.append(clean_row)

        if web_lookup_config.enabled:
            logger(f"[Perplexity] Iniciando busca de URLs para {len(processed_rows)} linhas...")
            asyncio.run(enriquecer_rows_com_urls_async(processed_rows, logger, web_lookup_config))

        with output_path.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(processed_rows)

        rows = len(processed_rows)

    logger(f"[Gerado] {output_path.name} | linhas={rows} | celulas_truncadas={truncated_cells}")
    return ProcessSummary(
        input_path=input_path,
        output_path=output_path,
        rows=rows,
        truncated_cells=truncated_cells,
        encoding=encoding,
        delimiter=delimiter,
        fields=output_fields,
    )


def compile_csvs(processed_paths: Sequence[Path], combined_output: Path, logger: Callable[[str], None]) -> int:
    if not processed_paths:
        raise ValueError("Nenhum arquivo processado para compilar.")

    ordered_headers: list[str] = []
    for path in processed_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            for header in headers:
                if header not in ordered_headers:
                    ordered_headers.append(header)

    total_rows = 0
    with combined_output.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=ordered_headers)
        writer.writeheader()

        for path in processed_paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    writer.writerow({header: row.get(header, "") for header in ordered_headers})
                    total_rows += 1

    logger(f"[Compilado] {combined_output.name} | linhas_totais={total_rows}")
    return total_rows


def normalize_input_paths(file_paths: Iterable[str]) -> list[Path]:
    normalized: list[Path] = []
    seen: set[Path] = set()
    for item in file_paths:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {item}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Arquivo ignorado (nao e .csv): {path.name}")
        if path not in seen:
            normalized.append(path)
            seen.add(path)
    return normalized


def run_batch(
    files: Sequence[str],
    out_dir: str,
    max_texto_chars: int,
    combined_name: str,
    replace_newlines: bool,
    web_lookup_config: WebLookupConfig,
    logger: Callable[[str], None],
) -> tuple[list[ProcessSummary], Path, int]:
    if not files:
        raise ValueError("Nenhum CSV informado para processamento.")

    if max_texto_chars < 0:
        raise ValueError("max_texto_chars nao pode ser negativo. Use 0 para sem truncamento.")
    if web_lookup_config.enabled and not web_lookup_config.api_key:
        raise ValueError(
            "Busca de URLs ativada, mas sem API key da Perplexity. "
            "Use --perplexity-api-key, a variavel de ambiente PERPLEXITY_API_KEY "
            f"ou o arquivo {DEFAULT_PERPLEXITY_KEY_FILE}."
        )

    input_paths = normalize_input_paths(files)
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not combined_name.lower().endswith(".csv"):
        combined_name = f"{combined_name}.csv"
    combined_path = output_dir / combined_name

    summaries: list[ProcessSummary] = []
    for input_path in input_paths:
        summary = process_one_csv(
            input_path=input_path,
            out_dir=output_dir,
            max_texto_chars=max_texto_chars,
            replace_newlines=replace_newlines,
            web_lookup_config=web_lookup_config,
            logger=logger,
        )
        summaries.append(summary)

    compiled_rows = compile_csvs(
        processed_paths=[summary.output_path for summary in summaries],
        combined_output=combined_path,
        logger=logger,
    )
    return summaries, combined_path, compiled_rows


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class App:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("CSV -> Notion Friendly (Lote)")
            self.root.geometry("900x620")

            self.file_vars: dict[str, tk.BooleanVar] = {}

            self.output_dir_var = tk.StringVar(value=str(Path.cwd()))
            self.max_texto_chars_var = tk.StringVar(value="9000")
            self.combined_name_var = tk.StringVar(value="jurisprudencia_compilado_notion.csv")
            self.replace_newlines_var = tk.BooleanVar(value=True)
            self.buscar_urls_var = tk.BooleanVar(value=False)
            self.perplexity_key_file_var = tk.StringVar(value=str(Path.cwd() / DEFAULT_PERPLEXITY_KEY_FILE))
            self.perplexity_api_key_var = tk.StringVar(
                value=resolve_perplexity_api_key("", self.perplexity_key_file_var.get())
            )
            self.perplexity_model_var = tk.StringVar(value="sonar")
            self.perplexity_workers_var = tk.StringVar(value="4")
            self.perplexity_timeout_var = tk.StringVar(value="15")
            self.perplexity_batch_size_var = tk.StringVar(value="20")
            self.perplexity_delay_var = tk.StringVar(value="0.3")

            self._build_ui()

        def _build_ui(self) -> None:
            main = ttk.Frame(self.root, padding=12)
            main.pack(fill="both", expand=True)

            top_controls = ttk.Frame(main)
            top_controls.pack(fill="x")

            ttk.Button(top_controls, text="Processar selecionados", command=self.process_selected).pack(
                side="right", padx=(8, 0)
            )
            ttk.Button(top_controls, text="Selecionar CSVs", command=self.add_files).pack(side="left", padx=(0, 8))
            ttk.Button(top_controls, text="Marcar todos", command=self.check_all).pack(side="left", padx=(0, 8))
            ttk.Button(top_controls, text="Desmarcar todos", command=self.uncheck_all).pack(side="left", padx=(0, 8))
            ttk.Button(top_controls, text="Limpar lista", command=self.clear_files).pack(side="left")

            list_box = ttk.LabelFrame(main, text="Arquivos selecionados (checkbox)", padding=8)
            list_box.pack(fill="both", expand=False, pady=(10, 10))

            self.canvas = tk.Canvas(list_box, height=230)
            scrollbar = ttk.Scrollbar(list_box, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=scrollbar.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            self.files_frame = ttk.Frame(self.canvas)
            self.canvas.create_window((0, 0), window=self.files_frame, anchor="nw")
            self.files_frame.bind(
                "<Configure>",
                lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
            )

            options = ttk.LabelFrame(main, text="Opcoes", padding=10)
            options.pack(fill="x", pady=(0, 10))

            ttk.Label(options, text="Pasta de saida:").grid(row=0, column=0, sticky="w")
            ttk.Entry(options, textvariable=self.output_dir_var, width=70).grid(row=0, column=1, sticky="ew", padx=8)
            ttk.Button(options, text="Selecionar...", command=self.choose_output_dir).grid(row=0, column=2, sticky="e")

            ttk.Label(options, text="Limite em textoDecisao/textoEmenta (0 = sem truncamento):").grid(row=1, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.max_texto_chars_var, width=12).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))

            ttk.Label(options, text="Nome do compilado:").grid(row=2, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.combined_name_var, width=35).grid(
                row=2,
                column=1,
                sticky="w",
                padx=8,
                pady=(8, 0),
            )

            ttk.Checkbutton(
                options,
                text="Substituir quebras de linha/tabs por espaco",
                variable=self.replace_newlines_var,
            ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

            ttk.Checkbutton(
                options,
                text="Buscar URLs com Perplexity (TSE/TRE/Gerais)",
                variable=self.buscar_urls_var,
            ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

            ttk.Label(options, text="Perplexity API key:").grid(row=5, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.perplexity_api_key_var, width=70, show="*").grid(
                row=5,
                column=1,
                sticky="ew",
                padx=8,
                pady=(8, 0),
            )

            ttk.Label(options, text="Arquivo da chave:").grid(row=6, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.perplexity_key_file_var, width=70).grid(
                row=6,
                column=1,
                sticky="ew",
                padx=8,
                pady=(8, 0),
            )

            perf_frame = ttk.Frame(options)
            perf_frame.grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
            ttk.Label(perf_frame, text="Modelo").grid(row=0, column=0, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_model_var, width=10).grid(row=0, column=1, padx=(4, 14))
            ttk.Label(perf_frame, text="Workers").grid(row=0, column=2, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_workers_var, width=5).grid(row=0, column=3, padx=(4, 14))
            ttk.Label(perf_frame, text="Timeout(s)").grid(row=0, column=4, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_timeout_var, width=6).grid(row=0, column=5, padx=(4, 14))
            ttk.Label(perf_frame, text="Batch").grid(row=0, column=6, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_batch_size_var, width=6).grid(row=0, column=7, padx=(4, 14))
            ttk.Label(perf_frame, text="Delay(s)").grid(row=0, column=8, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_delay_var, width=6).grid(row=0, column=9, padx=(4, 0))

            options.columnconfigure(1, weight=1)

            ttk.Button(main, text="Processar selecionados", command=self.process_selected).pack(anchor="w")

            log_box = ttk.LabelFrame(main, text="Log", padding=8)
            log_box.pack(fill="both", expand=True, pady=(10, 0))
            self.log_widget = tk.Text(log_box, height=12, wrap="word")
            self.log_widget.pack(fill="both", expand=True)
            self.log_widget.configure(state="disabled")

        def log(self, message: str) -> None:
            self.log_widget.configure(state="normal")
            self.log_widget.insert("end", f"{message}\n")
            self.log_widget.see("end")
            self.log_widget.configure(state="disabled")
            self.root.update_idletasks()

        def add_files(self) -> None:
            file_paths = filedialog.askopenfilenames(
                title="Selecione os arquivos CSV",
                filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
            )
            for file_path in file_paths:
                if file_path not in self.file_vars:
                    self.file_vars[file_path] = tk.BooleanVar(value=True)
            self.refresh_file_list()

        def clear_files(self) -> None:
            self.file_vars.clear()
            self.refresh_file_list()

        def check_all(self) -> None:
            for variable in self.file_vars.values():
                variable.set(True)

        def uncheck_all(self) -> None:
            for variable in self.file_vars.values():
                variable.set(False)

        def choose_output_dir(self) -> None:
            selected_dir = filedialog.askdirectory(title="Selecione a pasta de saida")
            if selected_dir:
                self.output_dir_var.set(selected_dir)

        def refresh_file_list(self) -> None:
            for child in self.files_frame.winfo_children():
                child.destroy()

            for row_index, file_path in enumerate(sorted(self.file_vars)):
                label = Path(file_path).name
                ttk.Checkbutton(self.files_frame, text=label, variable=self.file_vars[file_path]).grid(
                    row=row_index,
                    column=0,
                    sticky="w",
                    padx=4,
                    pady=2,
                )

        def process_selected(self) -> None:
            selected_files = [path for path, var in self.file_vars.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("Aviso", "Selecione ao menos um arquivo CSV.")
                return

            try:
                max_texto_chars = int(self.max_texto_chars_var.get().strip())
            except ValueError:
                messagebox.showerror("Erro", "Limite em textoDecisao/textoEmenta deve ser um numero inteiro.")
                return
            if max_texto_chars < 0:
                messagebox.showerror("Erro", "Limite em textoDecisao/textoEmenta nao pode ser negativo. Use 0 para sem truncamento.")
                return

            try:
                perplexity_workers = int(self.perplexity_workers_var.get().strip())
                perplexity_timeout = int(self.perplexity_timeout_var.get().strip())
                perplexity_batch_size = int(self.perplexity_batch_size_var.get().strip())
                perplexity_delay = float(self.perplexity_delay_var.get().strip())
            except ValueError:
                messagebox.showerror("Erro", "Parametros da Perplexity invalidos (workers/timeout/batch/delay).")
                return

            if min(perplexity_workers, perplexity_timeout, perplexity_batch_size) <= 0:
                messagebox.showerror("Erro", "Workers, timeout e batch precisam ser maiores que zero.")
                return
            if perplexity_delay < 0:
                messagebox.showerror("Erro", "Delay nao pode ser negativo.")
                return

            web_lookup_config = WebLookupConfig(
                enabled=self.buscar_urls_var.get(),
                api_key=resolve_perplexity_api_key(
                    self.perplexity_api_key_var.get().strip(),
                    self.perplexity_key_file_var.get().strip(),
                ),
                model=self.perplexity_model_var.get().strip() or "sonar",
                timeout_seconds=perplexity_timeout,
                max_workers=perplexity_workers,
                batch_size=perplexity_batch_size,
                delay_between_batches=perplexity_delay,
            )

            try:
                summaries, combined_path, compiled_rows = run_batch(
                    files=selected_files,
                    out_dir=self.output_dir_var.get().strip(),
                    max_texto_chars=max_texto_chars,
                    combined_name=self.combined_name_var.get().strip() or "jurisprudencia_compilado_notion.csv",
                    replace_newlines=self.replace_newlines_var.get(),
                    web_lookup_config=web_lookup_config,
                    logger=self.log,
                )
            except Exception as exc:  # pylint: disable=broad-except
                messagebox.showerror("Erro no processamento", str(exc))
                return

            total_rows = sum(item.rows for item in summaries)
            total_trunc = sum(item.truncated_cells for item in summaries)
            messagebox.showinfo(
                "Concluido",
                (
                    f"Arquivos processados: {len(summaries)}\n"
                    f"Linhas processadas: {total_rows}\n"
                    f"Celulas truncadas: {total_trunc}\n"
                    f"Linhas no compilado: {compiled_rows}\n"
                    f"Compilado: {combined_path}"
                ),
            )

    root = tk.Tk()
    App(root)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Converte CSVs para versao notion-friendly e gera compilado final."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Arquivos CSV para processar (se informado, roda em modo CLI).",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Pasta de saida (padrao: pasta atual).",
    )
    parser.add_argument(
        "--max-texto-chars",
        type=int,
        default=9000,
        help="Limite de caracteres para textoDecisao e textoEmenta (0 = sem truncamento).",
    )
    parser.add_argument(
        "--max-ementa-chars",
        dest="max_texto_chars",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-chars",
        dest="max_texto_chars",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--combined-name",
        default="jurisprudencia_compilado_notion.csv",
        help="Nome do arquivo compilado final.",
    )
    parser.add_argument(
        "--keep-newlines",
        action="store_true",
        help="Preserva quebras de linha e tabs nas celulas.",
    )
    parser.add_argument(
        "--buscar-urls-perplexity",
        action="store_true",
        help="Ativa busca de URLs (TSE/TRE/Gerais) por linha via API da Perplexity.",
    )
    parser.add_argument(
        "--perplexity-api-key",
        default="",
        help="API key da Perplexity (precedencia maior; evita depender de arquivo/env).",
    )
    parser.add_argument(
        "--perplexity-key-file",
        default=DEFAULT_PERPLEXITY_KEY_FILE,
        help=(
            "Arquivo local com a chave da Perplexity (padrao: "
            f"{DEFAULT_PERPLEXITY_KEY_FILE}; usado quando --perplexity-api-key/env nao forem informados)."
        ),
    )
    parser.add_argument(
        "--perplexity-model",
        default="sonar",
        help="Modelo da Perplexity (padrao: sonar).",
    )
    parser.add_argument(
        "--perplexity-max-workers",
        type=int,
        default=4,
        help="Numero maximo de workers para chamadas HTTP da Perplexity.",
    )
    parser.add_argument(
        "--perplexity-timeout",
        type=int,
        default=15,
        help="Timeout (segundos) de cada chamada Perplexity.",
    )
    parser.add_argument(
        "--perplexity-batch-size",
        type=int,
        default=20,
        help="Quantidade de linhas por lote de consultas Perplexity.",
    )
    parser.add_argument(
        "--perplexity-delay",
        type=float,
        default=0.3,
        help="Pausa (segundos) entre lotes de consultas Perplexity.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Forca modo CLI mesmo sem arquivos posicionais.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_cli = bool(args.files) or args.no_gui
    if run_cli:
        if not args.files:
            parser.error("No modo --no-gui, informe ao menos um arquivo CSV.")
        if args.perplexity_max_workers <= 0:
            parser.error("--perplexity-max-workers deve ser maior que zero.")
        if args.perplexity_timeout <= 0:
            parser.error("--perplexity-timeout deve ser maior que zero.")
        if args.perplexity_batch_size <= 0:
            parser.error("--perplexity-batch-size deve ser maior que zero.")
        if args.perplexity_delay < 0:
            parser.error("--perplexity-delay nao pode ser negativo.")

        def cli_logger(message: str) -> None:
            print(message)

        web_lookup_config = WebLookupConfig(
            enabled=args.buscar_urls_perplexity,
            api_key=resolve_perplexity_api_key(args.perplexity_api_key.strip(), args.perplexity_key_file),
            model=args.perplexity_model.strip() or "sonar",
            timeout_seconds=args.perplexity_timeout,
            max_workers=args.perplexity_max_workers,
            batch_size=args.perplexity_batch_size,
            delay_between_batches=args.perplexity_delay,
        )

        try:
            summaries, combined_path, compiled_rows = run_batch(
                files=args.files,
                out_dir=args.out_dir,
                max_texto_chars=args.max_texto_chars,
                combined_name=args.combined_name,
                replace_newlines=not args.keep_newlines,
                web_lookup_config=web_lookup_config,
                logger=cli_logger,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Erro: {exc}", file=sys.stderr)
            return 1

        total_rows = sum(item.rows for item in summaries)
        total_trunc = sum(item.truncated_cells for item in summaries)
        print("\nResumo:")
        print(f"- Arquivos processados: {len(summaries)}")
        print(f"- Linhas processadas: {total_rows}")
        print(f"- Celulas truncadas: {total_trunc}")
        print(f"- Linhas no compilado: {compiled_rows}")
        print(f"- Compilado: {combined_path}")
        return 0

    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
