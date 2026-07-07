# -*- coding: utf-8 -*-
"""Normaliza os blocos brutos do Notion em uma estrutura intermediária (IR)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass
class RichText:
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    href: Optional[str] = None
    # Tipo do rich_text no Notion: "text" | "mention" | "equation". Campo FINAL
    # com default: construções posicionais de 5 args continuam válidas (degrada
    # para "text"; a decisão de hyperlink é primariamente pelo domínio do href).
    kind: str = "text"


@dataclass
class Block:
    type: str
    rich: List[RichText] = field(default_factory=list)
    children: List["Block"] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


TEXTUAL = {
    "paragraph",
    "heading_1",
    "heading_2",
    "heading_3",
    # A API do Notion passou a devolver heading_4 (observado 06/07/2026 na
    # página "Auditoria e verificabilidade"); títulos de 4º nível existem.
    "heading_4",
    "bulleted_list_item",
    "numbered_list_item",
    "quote",
    "callout",
    "toggle",
}


def extract_rich(rt_list: Optional[List[Dict[str, Any]]]) -> List[RichText]:
    out: List[RichText] = []
    for t in rt_list or []:
        ann = t.get("annotations", {}) or {}
        out.append(
            RichText(
                text=t.get("plain_text", ""),
                bold=bool(ann.get("bold")),
                italic=bool(ann.get("italic")),
                underline=bool(ann.get("underline")),
                href=t.get("href"),
                kind=t.get("type", "text"),
            )
        )
    return out


def plain(rich: List[RichText]) -> str:
    return "".join(r.text for r in rich)


def parse_blocks(raw_blocks: List[Dict[str, Any]]) -> List[Block]:
    out: List[Block] = []
    for b in raw_blocks or []:
        t = b.get("type")
        data = b.get(t) or {} if isinstance(b.get(t), dict) else {}
        raw_children = b.get("_children", [])

        if t == "table":
            rows: List[List[List[RichText]]] = []
            for rb in raw_children:
                if rb.get("type") == "table_row":
                    cells = [extract_rich(c) for c in (rb["table_row"].get("cells") or [])]
                    rows.append(cells)
            out.append(
                Block(
                    type="table",
                    extra={
                        "rows": rows,
                        "has_column_header": bool(data.get("has_column_header")),
                        "has_row_header": bool(data.get("has_row_header")),
                        "table_width": data.get("table_width", 0),
                    },
                )
            )
        elif t == "divider":
            out.append(Block(type="divider"))
        elif t == "image":
            out.append(Block(type="image", extra={"image": data}))
        elif t == "equation":
            out.append(Block(type="equation", extra={"expression": data.get("expression", "")}))
        elif t in TEXTUAL:
            out.append(
                Block(
                    type=t,
                    rich=extract_rich(data.get("rich_text")),
                    children=parse_blocks(raw_children),
                    extra={"icon": data.get("icon")} if t == "callout" else {},
                )
            )
        else:
            # bloco não previsto: preserva texto se houver
            out.append(
                Block(
                    type=t or "unknown",
                    rich=extract_rich(data.get("rich_text")),
                    children=parse_blocks(raw_children),
                )
            )
    return out


def flatten_blocks(blocks: List[Block]) -> List[Block]:
    """Lineariza a árvore de blocos em pré-ordem (pai antes dos filhos).

    O conversor trata a página como uma sequência linear de blocos, mas o Notion
    permite aninhar (indentar) blocos sob outro — e seções inteiras (incl. a
    minuta) podem acabar como `children` de um parágrafo. Sem achatar, tanto o
    splitter quanto o renderizador percorreriam só o nível raiz e perderiam todo
    o conteúdo aninhado. Tabelas guardam suas linhas em `extra` (não têm children
    no IR) e, portanto, não são afetadas.
    """
    out: List[Block] = []
    for b in blocks:
        if b.children:
            out.append(replace(b, children=[]))
            out.extend(flatten_blocks(b.children))
        else:
            out.append(b)
    return out
