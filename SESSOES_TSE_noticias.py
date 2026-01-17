# -*- coding: utf-8 -*-
"""
Enrich sessoes_all_2024_2025.csv with news links using an API call.

Usage:
  python3 SESSOES_TSE_noticias_viaAPI.py
  python3 SESSOES_TSE_noticias_viaAPI.py --input /caminho/para/arquivo.csv

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

NEWS_COLUMNS = ["noticia_TSE", "noticia_TRE", "noticia_geral"]

CONTEXT_FIELDS = [
    "tema",
    "punchline",
    "numero_processo",
    "classe_processo",
    "tribunal",
    "origem",
    "data_sessao",
    "relator",
    "partes",
]

SYSTEM_PROMPT = (
    "You are a research assistant. Use web search. "
    "Return JSON only, without any extra text."
)

USER_PROMPT_TEMPLATE = (
    "Find news articles related to the following Brazilian electoral court session item. "
    "Only include links if the article is clearly about the same case/decision/session. "
    "If no relevant news exists, return empty arrays.\n\n"
    "Return ONLY a JSON object with keys:\n"
    "- noticia_TSE: list of URLs from domains that end with tse.jus.br\n"
    "- noticia_TRE: list of URLs from domains that match tre-XX.jus.br (any subdomain)\n"
    "- noticia_geral: list of URLs from Folha (folha.uol.com.br), ConJur (conjur.com.br), "
    "Migalhas (migalhas.com.br), Gazeta do Povo (gazetadopovo.com.br), "
    "CNN (cnnbrasil.com.br or cnn.com)\n\n"
    "Context:\n{context}\n"
)

URL_RE = re.compile(r"https?://[^\s\]\)>,;\"']+", re.IGNORECASE)
TRE_DOMAIN_RE = re.compile(r"(?:^|\.)tre-[a-z]{2}\.jus\.br$", re.IGNORECASE)

GENERAL_DOMAINS = [
    "folha.uol.com.br",
    "conjur.com.br",
    "migalhas.com.br",
    "gazetadopovo.com.br",
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


def _extract_json(text: str) -> Dict[str, object]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
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
    for key in NEWS_COLUMNS:
        urls.extend(_urls_from_value(data.get(key)))
    if not urls:
        urls = URL_RE.findall(text)
    return _classify_urls(urls)


def _join_urls(urls: List[str]) -> str:
    return ", ".join(urls) if urls else ""


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


def _read_existing_output(path: str, expected_header: List[str]) -> Tuple[bool, int]:
    if not os.path.exists(path):
        return False, 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False, 0
        if expected_header and header != expected_header:
            raise SystemExit(
                "ERRO: o cabecalho do CSV de saida nao corresponde ao esperado. "
                "Use --output para um novo arquivo ou remova o arquivo existente."
            )
        count = 0
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                logging.warning("Linha incompleta detectada no checkpoint. Ignorando o restante.")
                break
            count += 1
    return True, count


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

    output_fields = fieldnames[:]
    for col in NEWS_COLUMNS:
        if col not in output_fields:
            output_fields.append(col)

    cache: Dict[str, Tuple[List[str], List[str], List[str]]] = {}

    write_mode = "w"
    if args.resume and args.no_resume:
        raise SystemExit("ERRO: use apenas --resume ou --no-resume.")

    should_resume = args.resume or (not args.no_resume and os.path.exists(output_path))
    processed_rows = 0
    if should_resume and os.path.exists(output_path):
        has_header, processed_rows = _read_existing_output(output_path, output_fields)
        if has_header:
            write_mode = "a"
            if processed_rows:
                logging.info("Retomando a partir da linha %d do CSV de saida.", processed_rows + 1)

    with open(output_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        if write_mode == "w":
            writer.writeheader()

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

            row["noticia_TSE"] = _join_urls(tse_urls)
            row["noticia_TRE"] = _join_urls(tre_urls)
            row["noticia_geral"] = _join_urls(geral_urls)
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
