# -*- coding: utf-8 -*-
"""Montagem e escrita de blocos no Notion (markdown inline -> rich_text de escrita).

Generaliza a lógica validada em _aplicar_edicoes.py. Suporta `**bold**`, `*itálico*`,
`[texto](url)`, com chunking de 1900 chars. Escreve numa página EM BRANCO por append
em lotes (a ordem é preservada)."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

import requests

from . import config_gerador as cfg

API = "https://api.notion.com/v1"


def _url_valida(href) -> bool:
    """URL aceitável para um link do Notion: http(s) sem espaços. Evita o 400
    'Invalid URL for link' quando o redator gera um link com placeholder/sentinela
    (ex.: [ADC nº 29](LINK?)) ou URL malformada. URLs INTERNAS do Notion também
    são rejeitadas (o link vira texto puro): o Notion as renderia como mention
    da página — e página de base sem título aparece como "Untitled" no texto
    (achado do usuário, 06/07/2026, Estudo Resoluções TSE 2026); a regra 5 das
    REGRAS_CITACAO já proíbe citá-las, isto é o cinto de segurança."""
    h = str(href or "").strip()
    if not h or re.fullmatch(r"https?://\S+", h) is None:
        return False
    return not re.search(r"//(?:www\.)?(?:notion\.so|app\.notion\.com)\b", h)


# ===================== markdown inline -> rich_text =====================
def _emit(text: str, ann: dict, href, segs: list) -> None:
    while text:
        cands = []
        if href is None:
            m = re.search(r"\[([^\]]+)\]\(([^)]+)\)", text)
            if m:
                cands.append((m.start(), "link", m))
        if not ann["bold"]:
            m = re.search(r"\*\*(.+?)\*\*", text)
            if m:
                cands.append((m.start(), "bold", m))
        if not ann["italic"]:
            m = re.search(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)", text)
            if m:
                cands.append((m.start(), "italic", m))
        if not cands:
            segs.append((text, dict(ann), href))
            return
        cands.sort(key=lambda x: x[0])
        pos, kind, m = cands[0]
        if pos > 0:
            segs.append((text[:pos], dict(ann), href))
        if kind == "link":
            _emit(m.group(1), ann, m.group(2), segs)
        else:
            na = dict(ann)
            na[kind] = True
            _emit(m.group(1), na, href, segs)
        text = text[m.end():]


def _chunks(s: str, size: int = 1900) -> List[str]:
    return [s[i:i + size] for i in range(0, len(s), size)] or [""]


def md_to_rich(md: str) -> List[dict]:
    md = md or ""
    segs: list = []
    _emit(md, {"bold": False, "italic": False}, None, segs)
    out: List[dict] = []
    for text, ann, href in segs:
        if text == "":
            continue
        for chunk in _chunks(text):
            rt: Dict[str, Any] = {"type": "text", "text": {"content": chunk}}
            if href and _url_valida(href):
                rt["text"]["link"] = {"url": href.strip()}
            rt["annotations"] = {
                "bold": ann["bold"], "italic": ann["italic"],
                "strikethrough": False, "underline": False, "code": False, "color": "default",
            }
            out.append(rt)
    return out or [{"type": "text", "text": {"content": ""}}]


# ===================== helpers de bloco =====================
def _blk(tipo: str, payload: dict) -> dict:
    return {"object": "block", "type": tipo, tipo: payload}


def bloco_paragraph(md: str) -> dict:
    return _blk("paragraph", {"rich_text": md_to_rich(md)})


def bloco_heading(nivel: int, md: str) -> dict:
    t = f"heading_{max(1, min(3, nivel))}"
    return _blk(t, {"rich_text": md_to_rich(md)})


def bloco_quote(md: str) -> dict:
    return _blk("quote", {"rich_text": md_to_rich(md)})


def bloco_bullet(md: str) -> dict:
    return _blk("bulleted_list_item", {"rich_text": md_to_rich(md)})


def bloco_callout(md: str, *, emoji: str = "📌", cor: str = "gray_background") -> dict:
    return _blk("callout", {
        "rich_text": md_to_rich(md),
        "icon": {"type": "emoji", "emoji": emoji},
        "color": cor,
    })


def bloco_divider() -> dict:
    return _blk("divider", {})


def bloco_tabela(linhas: List[List[str]], *, header: bool = True) -> dict:
    """linhas: lista de linhas; cada linha é lista de células em markdown."""
    largura = max((len(ln) for ln in linhas), default=1)
    children = []
    for ln in linhas:
        cells = [md_to_rich(c) for c in ln]
        while len(cells) < largura:
            cells.append(md_to_rich(""))
        children.append({"object": "block", "type": "table_row", "table_row": {"cells": cells}})
    return _blk("table", {
        "table_width": largura,
        "has_column_header": header,
        "has_row_header": False,
        "children": children,
    })


# ===================== escrita na página =====================
def _headers() -> dict:
    return {
        "Authorization": f"Bearer {cfg.load_notion_token()}",
        "Notion-Version": cfg.NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _sanitizar_links(obj) -> int:
    """Remove, recursivamente, links com URL inválida de QUALQUER rich_text da estrutura
    (parágrafos, células de tabela, filhos). Rede de segurança final contra o 400 do Notion —
    funciona inclusive em blocos já montados (ex.: regravação do último resultado)."""
    n = 0
    if isinstance(obj, dict):
        t = obj.get("text")
        if isinstance(t, dict) and isinstance(t.get("link"), dict) and not _url_valida(t["link"].get("url")):
            t.pop("link", None)
            n += 1
        for v in obj.values():
            n += _sanitizar_links(v)
    elif isinstance(obj, list):
        for it in obj:
            n += _sanitizar_links(it)
    return n


def escrever_pagina(page_id: str, blocos: List[dict], *, progress=None) -> int:
    """Append dos blocos na página (lotes de 90). Página deve estar vazia.
    Retorna o total de blocos escritos."""
    log = progress or (lambda _m: None)
    removidos = _sanitizar_links(blocos)
    if removidos:
        log(f"   {removidos} link(s) com URL inválida convertido(s) em texto (evita erro do Notion).")
    sess = requests.Session()
    sess.headers.update(_headers())
    total = 0
    for i in range(0, len(blocos), 90):
        lote = blocos[i:i + 90]
        body = json.dumps({"children": lote})
        for tentativa in range(4):
            r = sess.patch(f"{API}/blocks/{page_id}/children", data=body, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After", "2")))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Notion append -> {r.status_code}: {r.text[:400]}")
            break
        else:
            # as 4 tentativas se esgotaram em 429/5xx — NÃO mascarar como sucesso
            raise RuntimeError(
                f"Notion append falhou após 4 tentativas no lote {i // 90 + 1} "
                f"(blocos {i}..{i + len(lote)}): último status {r.status_code}: {r.text[:300]}")
        total += len(lote)
        log(f"   gravados {total}/{len(blocos)} blocos...")
        time.sleep(0.34)
    return total


def page_id_from_url(url_or_id: str) -> str:
    """Reusa a normalização do conversor."""
    from conle_conversor.notion_api import normalize_page_id

    return normalize_page_id(url_or_id)


def verificar_acesso_pagina(page_id: str) -> None:
    """Confere se a integração consegue ler a página ANTES de gastar chamadas de IA.
    Levanta erro claro se a página não estiver compartilhada com a integração."""
    r = requests.get(f"{API}/blocks/{page_id}", headers=_headers(), timeout=30)
    if r.status_code == 404:
        raise RuntimeError(
            "Página não encontrada ou NÃO compartilhada com a integração. No Notion, abra a "
            "página -> menu \"...\" (canto superior direito) -> Conexões -> adicione a integração; "
            "ou crie a página dentro de uma área que já esteja compartilhada.")
    if r.status_code >= 400:
        raise RuntimeError(f"Não foi possível acessar a página ({r.status_code}): {r.text[:200]}")
