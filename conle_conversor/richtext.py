# -*- coding: utf-8 -*-
"""Conversão de rich text (Notion) em runs do Word."""
from __future__ import annotations

from typing import List, Optional

from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from .notion_parser import RichText

# Cor azul padrão de hyperlink do Word (igual ao estilo "Hyperlink").
_COR_LINK = "0563C1"


def _add_hyperlink(paragraph, text: str, url: str, *, bold: bool = False, italic: bool = False):
    """Insere um hyperlink REAL (clicável) no parágrafo. O python-docx não expõe API
    para isso, então monta-se o elemento w:hyperlink no XML, com a relationship externa
    apontando para a URL. Mantém o link do Notion preservado no .docx (azul + sublinhado)."""
    r_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), _COR_LINK)
    rpr.append(color)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(u)
    if bold:
        rpr.append(OxmlElement("w:b"))
    if italic:
        rpr.append(OxmlElement("w:i"))
    run.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    return hyperlink


def add_runs(paragraph, rich_list: List[RichText], *, force_bold: Optional[bool] = None):
    """Adiciona runs ao parágrafo preservando bold/italic/underline e os HYPERLINKS
    do Notion (referências viram links clicáveis no .docx). '\n' dentro de um run vira
    quebra de linha (soft break)."""
    for r in rich_list or []:
        if not r.text:
            continue
        partes = r.text.split("\n")
        for i, parte in enumerate(partes):
            if i > 0:
                paragraph.add_run().add_break()
            if parte == "":
                continue
            b = r.bold if force_bold is None else force_bold
            href = (r.href or "").strip()
            if href.startswith("http"):
                # preserva o link do Notion como hyperlink real e clicável
                _add_hyperlink(paragraph, parte, href, bold=bool(b), italic=bool(r.italic))
                continue
            run = paragraph.add_run(parte)
            if b:
                run.bold = True
            if r.italic:
                run.italic = True
            if r.underline:
                run.underline = True
    return paragraph


def split_rich_lines(rich_list: List[RichText]) -> List[List[RichText]]:
    """Quebra uma lista de rich text em linhas (separadas por '\n'),
    preservando a formatação de cada trecho."""
    linhas: List[List[RichText]] = [[]]
    for r in rich_list or []:
        partes = r.text.split("\n")
        for i, parte in enumerate(partes):
            if i > 0:
                linhas.append([])
            if parte:
                linhas[-1].append(RichText(parte, r.bold, r.italic, r.underline, r.href))
    return [ln for ln in linhas if any(x.text.strip() for x in ln)]
