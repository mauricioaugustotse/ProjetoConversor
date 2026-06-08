# -*- coding: utf-8 -*-
"""Conversão de rich text (Notion) em runs do Word."""
from __future__ import annotations

from typing import List, Optional

from .notion_parser import RichText


def add_runs(paragraph, rich_list: List[RichText], *, force_bold: Optional[bool] = None):
    """Adiciona runs ao parágrafo preservando bold/italic/underline.
    '\n' dentro de um run vira quebra de linha (soft break). Ignora hyperlinks
    (os modelos da casa exibem referências como texto normal)."""
    for r in rich_list or []:
        if not r.text:
            continue
        partes = r.text.split("\n")
        for i, parte in enumerate(partes):
            if i > 0:
                paragraph.add_run().add_break()
            if parte == "":
                continue
            run = paragraph.add_run(parte)
            b = r.bold if force_bold is None else force_bold
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
