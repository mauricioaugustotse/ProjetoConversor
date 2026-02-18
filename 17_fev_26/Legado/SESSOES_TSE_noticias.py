# -*- coding: utf-8 -*-
"""
Enrich sessoes with news links using an API call.

Defaults:
  Model: gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
except ImportError as exc:
    raise SystemExit("ERRO: openai nao encontrado. Execute: pip install openai") from exc

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None


DEFAULT_MODEL = "gpt-4o-mini"

NEWS_KEYS = ["noticia_TSE", "noticia_TRE", "noticia_geral"]
GENERAL_COLUMN_PREFIX = "noticia_geral"
GENERAL_COLUMN_RE = re.compile(r"^noticia_geral_(\d+)$")
DEFAULT_GENERAL_MAX = 9

CONTEXT_FIELDS = [
    "tema",
    "punchline",
    "resultado",
    "votacao",
    "numero_processo",
    "classe_processo",
    "tribunal",
    "origem",
    "data_sessao",
    "relator",
    "partes",
    "eleicao",
]

SYSTEM_PROMPT = (
    "Voce e um assistente de pesquisa especializado em noticias sobre casos julgados pela "
    "Justica Eleitoral. Use pesquisa aprofundada na web (pro) e verifique o permalink da noticia. "
    "Responda somente com JSON valido, sem markdown ou texto extra."
)

USER_PROMPT_TEMPLATE = (
    "Encontre noticias reais e atuais relacionadas ao item de sessao/decisao abaixo. "
    "Use numero do processo, classe, partes, relator, tema e data como base da busca. "
    "Inclua links somente se o artigo tratar claramente do mesmo caso/decisao/sessao. "
    "Se nao houver noticias relevantes, retorne arrays vazios.\n\n"
    "Responda APENAS com um objeto JSON valido com estas chaves e arrays de URLs:\n"
    "- noticia_TSE: URLs de dominios terminados em tse.jus.br (apenas oficiais relevantes)\n"
    "- noticia_TRE: URLs de dominios tre-XX.jus.br (ex: tre-sp.jus.br)\n"
    "- noticia_geral: URLs de grandes veiculos brasileiros (Folha, Estadao, G1, CNN Brasil, "
    "ConJur, Migalhas, UOL, etc.)\n\n"
    "Nao inclua titulos ou texto. Se houver duvida sobre o permalink, prefira omitir o link.\n\n"
    "Contexto:\n{context}\n"
)

URL_RE = re.compile(r"https?://[^\s\]\)>,;\"']+", re.IGNORECASE)
TRE_DOMAIN_RE = re.compile(r"(?:^|\.)tre-[a-z]{2}\.jus\.br$", re.IGNORECASE)

GENERAL_DOMAINS = [
    "folha.uol.com.br",
    "uol.com.br",
    "conjur.com.br",
    "migalhas.com.br",
    "gazetadopovo.com.br",
    "estadao.com.br",
    "g1.globo.com",
    "oglobo.globo.com",
    "metropoles.com",
    "terra.com.br",
    "cnnbrasil.com.br",
    "cnn.com",
]


def _build_context(row: Dict[str, str], max_len: int = 240) -> str:
    lines: List[str] = []
    for field in CONTEXT_FIELDS:
        raw = (row.get(field) or "").strip()
        if not raw:
            continue
        if len(raw) > max_len:
            raw = raw[:max_len].rstrip() + "..."
        lines.append(f"{field}: {raw}")
    return "\n".join(lines).strip()


def _output_text_from_response(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    # Fallback for older response shapes
    try:
        parts: List[str] = []
        for item in response.output or []:
            if getattr(item, "type", "") != "message":
                continue
            for content in item.content or []:
                if getattr(content, "type", "") == "output_text":
                    parts.append(content.text)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _call_openai_with_web_search(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                tools=[{"type": "web_search_preview"}],
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = _output_text_from_response(response)
            if not text:
                raise ValueError("Resposta vazia da API.")
            return text
        except (APIError, RateLimitError, APITimeoutError, APIConnectionError, ValueError) as exc:
            last_err = exc
            wait = 2 ** attempt
            logging.warning("Erro na API (tentativa %d/%d): %s", attempt + 1, max_retries, exc)
            time.sleep(wait)
        except Exception as exc:
            last_err = exc
            logging.warning("Erro inesperado na API (tentativa %d/%d): %s", attempt + 1, max_retries, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Falha na chamada da API apos {max_retries} tentativas: {last_err}")


def _clean_response_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"\s*\[web:\d+\]", "", cleaned)
    return cleaned.strip()


def _extract_json(text: str) -> object:
    cleaned = _clean_response_text(text)
    if not cleaned:
        return {}
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _normalize_url(url: str) -> str:
    cleaned = (url or "").strip().strip(".,;)]}>\"'")
    if not cleaned:
        return ""
    if not re.match(r"^https?://", cleaned, re.IGNORECASE):
        cleaned = "https://" + cleaned
    return cleaned


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_tse_domain(domain: str) -> bool:
    return domain == "tse.jus.br" or domain.endswith(".tse.jus.br")


def _is_tre_domain(domain: str) -> bool:
    return bool(TRE_DOMAIN_RE.search(domain))


def _is_general_domain(domain: str) -> bool:
    for base in GENERAL_DOMAINS:
        if domain == base or domain.endswith("." + base):
            return True
    return False


def _urls_from_value(value: object) -> List[str]:
    urls: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict):
                for key in ("url", "link", "href"):
                    candidate = item.get(key)
                    if isinstance(candidate, str):
                        urls.append(candidate)
                        break
    elif isinstance(value, dict):
        for key in ("url", "link", "href"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                urls.append(candidate)
    elif isinstance(value, str):
        urls.extend(URL_RE.findall(value))
    return urls


def _classify_urls(urls: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    tse: List[str] = []
    tre: List[str] = []
    geral: List[str] = []
    seen = set()
    for raw in urls:
        normalized = _normalize_url(raw)
        if not normalized or normalized in seen:
            continue
        domain = _domain_from_url(normalized)
        if not domain:
            continue
        if _is_tse_domain(domain):
            tse.append(normalized)
        elif _is_tre_domain(domain):
            tre.append(normalized)
        elif _is_general_domain(domain):
            geral.append(normalized)
        else:
            continue
        seen.add(normalized)
    return tse, tre, geral


def _combine_urls_from_response(text: str) -> Tuple[List[str], List[str], List[str]]:
    data = _extract_json(text)
    urls: List[str] = []
    if isinstance(data, dict):
        for key in NEWS_KEYS:
            urls.extend(_urls_from_value(data.get(key)))
        for key, value in data.items():
            if key.startswith(f"{GENERAL_COLUMN_PREFIX}_"):
                urls.extend(_urls_from_value(value))
    elif isinstance(data, list):
        urls.extend(_urls_from_value(data))
    if not urls:
        cleaned = _clean_response_text(text)
        urls = URL_RE.findall(cleaned)
    return _classify_urls(urls)


def _join_urls(urls: List[str]) -> str:
    return ", ".join(urls) if urls else ""


def _general_columns_from_fields(fieldnames: Iterable[str], max_geral: int) -> List[str]:
    existing: List[Tuple[int, str]] = []
    for name in fieldnames:
        match = GENERAL_COLUMN_RE.match(name or "")
        if match:
            existing.append((int(match.group(1)), name))
    if existing:
        return [name for _, name in sorted(existing)]
    if max_geral <= 0:
        return []
    return [f"{GENERAL_COLUMN_PREFIX}_{i}" for i in range(1, max_geral + 1)]


def _build_output_fields(fieldnames: List[str], general_columns: List[str]) -> List[str]:
    output_fields = fieldnames[:]
    for col in NEWS_KEYS:
        if col not in output_fields:
            output_fields.append(col)
    for col in general_columns:
        if col not in output_fields:
            output_fields.append(col)
    return output_fields


def _derive_output_path(input_path: str) -> str:
    directory, filename = os.path.split(input_path)
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".csv"
    output_name = f"{base} - com notícias{ext}"
    return os.path.join(directory, output_name)


def _select_input_csv() -> str:
    if tk is None or filedialog is None:
        raise SystemExit("ERRO: tkinter nao disponivel para selecao de arquivo.")
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Selecione o CSV de entrada",
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        root.destroy()
    except Exception as exc:
        raise SystemExit(f"ERRO: nao foi possivel abrir a caixa de selecao: {exc}") from exc
    if not selected:
        raise SystemExit("ERRO: nenhum arquivo selecionado.")
    return selected


def _read_existing_output(path: str) -> Tuple[List[str], int]:
    if not os.path.exists(path):
        return [], 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        count = 0
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                logging.warning("Linha incompleta detectada no checkpoint. Ignorando o restante.")
                break
            count += 1
    return header, count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Busca noticias por linha usando API e adiciona colunas ao CSV."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Caminho do CSV de entrada (se omitido, abre seletor).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Ignorado; a saida segue o nome do input no mesmo diretorio.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Modelo OpenAI com web search (padrao: gpt-4o-mini).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Processa apenas as primeiras N linhas.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pausa entre chamadas da API (segundos).")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximo de tentativas da API.")
    parser.add_argument(
        "--max-geral",
        type=int,
        default=DEFAULT_GENERAL_MAX,
        help="Numero maximo de colunas noticia_geral_N (default: 9).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Salva checkpoint a cada N linhas (default: 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma a partir do CSV de saida existente.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora CSV de saida existente e reprocessa tudo.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log detalhado.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    checkpoint_every = args.checkpoint_every
    if checkpoint_every <= 0:
        logging.warning("checkpoint-every <= 0; usando 1.")
        checkpoint_every = 1

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERRO: OPENAI_API_KEY nao definido no ambiente.")
    model = (args.model or "").strip()
    if model != "gpt-4o-mini":
        raise SystemExit("ERRO: modelo fixo: gpt-4o-mini.")

    client = OpenAI(api_key=api_key)

    input_path = (args.input or "").strip()
    if input_path and not os.path.exists(input_path):
        logging.warning("Arquivo de entrada nao encontrado: %s. Abrindo seletor.", input_path)
        input_path = ""
    if not input_path:
        input_path = _select_input_csv()
    output_path = _derive_output_path(input_path)
    if args.output:
        logging.warning("Ignorando --output; saida segue o nome do input.")

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        logging.warning("Nenhuma linha encontrada no CSV de entrada.")
        return

    general_columns = _general_columns_from_fields(fieldnames, args.max_geral)
    output_fields = _build_output_fields(fieldnames, general_columns)

    cache: Dict[str, Tuple[List[str], List[str], List[str]]] = {}

    write_mode = "w"
    if args.resume and args.no_resume:
        raise SystemExit("ERRO: use apenas --resume ou --no-resume.")

    should_resume = args.resume or (not args.no_resume and os.path.exists(output_path))
    processed_rows = 0
    if should_resume and os.path.exists(output_path):
        existing_header, processed_rows = _read_existing_output(output_path)
        if existing_header:
            missing_cols = [col for col in fieldnames if col not in existing_header]
            if missing_cols:
                raise SystemExit(
                    "ERRO: o CSV de saida nao contem todas as colunas do input. "
                    "Use --no-resume para gerar um novo arquivo ou remova o existente."
                )
            output_fields = existing_header
            general_columns = _general_columns_from_fields(existing_header, 0)
            write_mode = "a"
        if processed_rows:
            logging.info("Retomando a partir da linha %d do CSV de saida.", processed_rows + 1)

    with open(output_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        if write_mode == "w":
            writer.writeheader()

        output_field_set = set(output_fields)
        total = len(rows) if args.limit <= 0 else min(len(rows), args.limit)
        if processed_rows >= total:
            logging.info("Nada a fazer: %d linhas ja processadas.", processed_rows)
            return

        for idx, row in enumerate(rows[:total], start=1):
            if idx <= processed_rows:
                continue
            context = _build_context(row)
            if not context:
                tse_urls, tre_urls, geral_urls = [], [], []
            else:
                cache_key = context
                if cache_key in cache:
                    tse_urls, tre_urls, geral_urls = cache[cache_key]
                else:
                    prompt = USER_PROMPT_TEMPLATE.format(context=context)
                    try:
                        raw_text = _call_openai_with_web_search(
                            client=client,
                            model=model,
                            prompt=prompt,
                            max_retries=args.max_retries,
                        )
                        tse_urls, tre_urls, geral_urls = _combine_urls_from_response(raw_text)
                        cache[cache_key] = (tse_urls, tre_urls, geral_urls)
                    except Exception as exc:
                        logging.error("Falha ao consultar API na linha %d: %s", idx, exc)
                        tse_urls, tre_urls, geral_urls = [], [], []

            if "noticia_TSE" in output_field_set:
                row["noticia_TSE"] = _join_urls(tse_urls)
            if "noticia_TRE" in output_field_set:
                row["noticia_TRE"] = _join_urls(tre_urls)
            if "noticia_geral" in output_field_set:
                row["noticia_geral"] = _join_urls(geral_urls)
            for pos, col in enumerate(general_columns):
                if col not in output_field_set:
                    continue
                row[col] = geral_urls[pos] if pos < len(geral_urls) else ""
            if general_columns and len(geral_urls) > len(general_columns):
                logging.info(
                    "Mais links gerais (%d) do que colunas (%d) na linha %d; truncando.",
                    len(geral_urls),
                    len(general_columns),
                    idx,
                )
            writer.writerow(row)

            if checkpoint_every and (idx % checkpoint_every == 0 or idx == total):
                f.flush()
                os.fsync(f.fileno())
            if idx % 10 == 0 or idx == total:
                logging.info("Processado %d/%d", idx, total)
            if args.sleep:
                time.sleep(args.sleep)

    logging.info("Arquivo gerado: %s", output_path)


if __name__ == "__main__":
    main()
