# -*- coding: utf-8 -*-
"""Cliente mínimo da API do Notion: busca página + blocos recursivamente."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from . import config

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

# Exceções de rede que merecem nova tentativa (a rede do Notion sofre resets
# intermitentes no handshake TLS — WinError 10054). ConnectionError do requests
# já cobre SSLError (herda dela); não listar separado.
_RETRIABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
MAX_TENTATIVAS = 6


def normalize_page_id(url_or_id: str) -> str:
    """Extrai e formata o ID da página (UUID 8-4-4-4-12) a partir de URL ou id."""
    import re

    s = (url_or_id or "").strip().split("?")[0].split("#")[0]
    # 1) UUID já formatado (com hifens) em qualquer parte da string
    m = re.search(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", s
    )
    if m:
        raw = m.group(0).replace("-", "").lower()
    else:
        # 2) bloco contíguo de 32 hex (o ID no fim da URL, após o último hífen do slug)
        blocos = re.findall(r"[0-9a-fA-F]{32}", s)
        if not blocos:
            raise ValueError(f"Não foi possível extrair o ID da página de: {url_or_id!r}")
        raw = blocos[-1].lower()
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"


class NotionClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or config.load_notion_token()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path: str, **kw) -> Dict[str, Any]:
        url = f"{API}{path}"
        last = None
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_TENTATIVAS):
            try:
                resp = self.session.request(method, url, timeout=60, **kw)
            except _RETRIABLE as exc:
                last_exc = exc
                time.sleep(min(1.5 * (attempt + 1), 8.0))
                continue
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", "2")))
                continue
            if resp.status_code >= 400:
                last = resp
                if resp.status_code in (500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Notion {method} {path} -> {resp.status_code}: {resp.text[:400]}"
                )
            return resp.json()
        if last is None and last_exc is not None:
            raise RuntimeError(
                f"Falha de conexão com o Notion após {MAX_TENTATIVAS} tentativas "
                f"({type(last_exc).__name__}). Verifique a rede e tente novamente."
            )
        raise RuntimeError(
            f"Notion {method} {path} falhou: {last.status_code if last else '??'} "
            f"{last.text[:300] if last else ''}"
        )

    def get_page(self, page_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/pages/{page_id}")

    def get_block_children(self, block_id: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            data = self._request("GET", f"/blocks/{block_id}/children", params=params)
            for item in data.get("results", []) or []:
                if item.get("has_children") and item.get("type") not in {"child_page", "child_database"}:
                    item["_children"] = self.get_block_children(item["id"])
                results.append(item)
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break
        return results


def page_title(page_meta: Dict[str, Any]) -> str:
    for prop in (page_meta.get("properties") or {}).values():
        if prop.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in prop.get("title", []))
    return ""


def url_publica_da_pagina(client: NotionClient, page_id: str) -> Optional[str]:
    """Primeira propriedade tipo 'url' com valor PÚBLICO (não-Notion) da página,
    em ordem alfabética de nome — na base de julgados do TSE, link_1 traz o PDF
    do acórdão no sjur. None se a página não tem URL pública ou a chamada falha
    (rede/permissão): a mention degrada para o comportamento antigo."""
    try:
        meta = client.get_page(page_id)
    except Exception:
        return None
    urls = []
    for nome, prop in (meta.get("properties") or {}).items():
        if prop.get("type") == "url" and prop.get("url"):
            u = str(prop["url"]).strip()
            if u.startswith("http") and "notion" not in u.lower():
                urls.append((nome.lower(), u))
    return min(urls)[1] if urls else None


def resolver_mentions_publicas(blocks, client: Optional[NotionClient] = None) -> int:
    """Pré-resolução das mentions internas que NÃO casam com uma norma de
    config.FONTES_OFICIAIS (ex. julgados do TSE citados como "(cf. @página)"):
    consulta a própria página mencionada e troca o href interno pela URL
    pública dela (prop 'url', ex. o PDF do sjur) — assim a referência externa
    sobrevive no .docx em vez de virar texto morto. Mentions de norma ficam
    como estão (richtext as resolve por texto, sem rede). Cache por página;
    sem URL pública ou com falha, o href interno permanece (vira texto puro).
    Muta blocks in place (rich e células de tabela); retorna o nº de mentions
    resolvidas."""
    from dataclasses import replace as _replace

    from .richtext import _link_interno_notion, resolver_fonte_publica

    client = client or NotionClient()
    cache: Dict[str, Optional[str]] = {}
    resolvidas = 0

    def _url_da_mention(href: str) -> Optional[str]:
        try:
            pid = normalize_page_id(href)
        except ValueError:
            return None
        if pid not in cache:
            cache[pid] = url_publica_da_pagina(client, pid)
        return cache[pid]

    def _varre(rich_list) -> None:
        nonlocal resolvidas
        for i, r in enumerate(rich_list or []):
            h = (r.href or "").strip()
            if not h or not _link_interno_notion(h):
                continue
            if resolver_fonte_publica(r.text):
                continue
            url = _url_da_mention(h)
            if url:
                rich_list[i] = _replace(r, href=url)
                resolvidas += 1

    for b in blocks or []:
        _varre(b.rich)
        for row in (b.extra or {}).get("rows") or []:
            for cell in row:
                _varre(cell)
    return resolvidas


def fetch_page(url_or_id: str, token: Optional[str] = None):
    """Retorna (page_id, titulo, blocos)."""
    client = NotionClient(token)
    page_id = normalize_page_id(url_or_id)
    meta = client.get_page(page_id)
    title = page_title(meta)
    blocks = client.get_block_children(page_id)
    return page_id, title, blocks
