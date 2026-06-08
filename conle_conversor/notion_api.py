# -*- coding: utf-8 -*-
"""Cliente mínimo da API do Notion: busca página + blocos recursivamente."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from . import config

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


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
        for attempt in range(5):
            resp = self.session.request(method, url, timeout=60, **kw)
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


def fetch_page(url_or_id: str, token: Optional[str] = None):
    """Retorna (page_id, titulo, blocos)."""
    client = NotionClient(token)
    page_id = normalize_page_id(url_or_id)
    meta = client.get_page(page_id)
    title = page_title(meta)
    blocks = client.get_block_children(page_id)
    return page_id, title, blocks
