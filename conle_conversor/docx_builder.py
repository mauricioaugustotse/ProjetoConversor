# -*- coding: utf-8 -*-
"""Constrói os .docx (Informação Técnica e minuta de proposição) a partir dos
templates, aplicando os estilos nomeados da casa."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from . import config
from .config import S
from .classifier import TipoProposicao
from .meta import MetaDocumento
from .notion_parser import Block, RichText, plain
from .richtext import add_runs, split_rich_lines
from .splitter import PaginaSeparada
from .harmonizer import Abertura


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _style_name(doc, name: str) -> str:
    try:
        _ = doc.styles[name]
        return name
    except KeyError:
        return "Normal"


def _p(doc, style: str, *, text: Optional[str] = None, rich: Optional[List[RichText]] = None,
       force_bold: Optional[bool] = None):
    para = doc.add_paragraph(style=_style_name(doc, style))
    if text is not None:
        run = para.add_run(text)
        if force_bold:
            run.bold = True
    elif rich is not None:
        add_runs(para, rich, force_bold=force_bold)
    return para


def _add_bullet(doc, rich: List[RichText]):
    para = doc.add_paragraph(style=_style_name(doc, S.CORPO))
    pf = para.paragraph_format
    pf.first_line_indent = Pt(-14.2)
    pf.left_indent = Pt(28.4)
    para.add_run("•\t")
    add_runs(para, rich)
    return para


def _add_quote_as_transcricao(doc, rich: List[RichText]):
    for linha in split_rich_lines(rich):
        _p(doc, S.TRANSCRICAO, rich=linha)


def _add_table(doc, block: Block):
    rows = block.extra.get("rows", [])
    if not rows:
        return
    ncols = block.extra.get("table_width") or max((len(r) for r in rows), default=1)
    has_header = block.extra.get("has_column_header", False) or _looks_like_header(rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = doc.styles[_style_name(doc, "Plain Table 1")]
    table.autofit = True
    for ri, row in enumerate(rows):
        for ci in range(ncols):
            cell = table.cell(ri, ci)
            para = cell.paragraphs[0]
            for r in list(para.runs):
                r.text = ""
            cell_rich = row[ci] if ci < len(row) else []
            add_runs(para, cell_rich, force_bold=True if (ri == 0 and has_header) else None)


def _looks_like_header(rows) -> bool:
    if not rows:
        return False
    first = rows[0]
    return all(any(rt.bold for rt in cell) for cell in first if cell)


# ---------------------------------------------------------------------------
# corpo da IT
# ---------------------------------------------------------------------------
def _render_it_blocks(doc, blocks: List[Block]):
    for b in blocks:
        if b.type == "heading_1":
            _p(doc, S.TITULO_ITEM, rich=b.rich)
        elif b.type == "heading_2":
            _p(doc, S.TITULO_ITEM, rich=b.rich)
        elif b.type == "heading_3":
            _p(doc, S.TITULO_SUB, rich=b.rich)
        elif b.type == "paragraph":
            if plain(b.rich).strip():
                _p(doc, S.CORPO, rich=b.rich)
        elif b.type == "bulleted_list_item":
            _add_bullet(doc, b.rich)
        elif b.type == "numbered_list_item":
            _add_bullet(doc, b.rich)
        elif b.type == "quote":
            _add_quote_as_transcricao(doc, b.rich)
        elif b.type == "callout":
            if plain(b.rich).strip():
                _p(doc, S.CORPO, rich=b.rich)
        elif b.type == "table":
            _add_table(doc, b)
        elif b.type == "divider":
            continue
        else:
            if plain(b.rich).strip():
                _p(doc, S.CORPO, rich=b.rich)


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii").upper()


PRETO = RGBColor(0x00, 0x00, 0x00)


def _run_label(para, texto: str):
    """Rótulo da capa: herda o verde/negrito do estilo CAPA ESTUDO - TÍTULO."""
    r = para.add_run(texto)
    r.font.size = Pt(12)
    r.font.name = "Arial"
    return r


def _run_valor(para, texto: str, *, bold=False):
    """Valor da capa: preto, 12pt, Arial. bold=False reproduz o `b=0` do modelo;
    bold=None herda o negrito do estilo (caso do valor do TEOR)."""
    r = para.add_run(texto)
    r.font.size = Pt(12)
    r.font.name = "Arial"
    r.font.color.rgb = PRETO
    if bold is not None:
        r.bold = bold
    return r


def _espacar(p, before: float, after: float):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.15


def _preencher_capa(doc, teor: str, solicitante: str, autor_linhas):
    """Reconstrói o campo da capa (TEOR / SOLICITANTE / AUTOR) reproduzindo o
    padrão do modelo: rótulo verde-negrito, valor preto; valor do TEOR em
    negrito e os demais sem negrito; espaçamentos próprios por linha."""
    if not doc.tables:
        return
    table = doc.tables[0]
    alvo = None
    for row in table.rows:
        for cell in row.cells:
            if "TEOR DA SOLICITACAO" in _fold(cell.text):
                alvo = cell
                break
        if alvo is not None:
            break
    if alvo is None:
        return

    estilo = _style_name(doc, S.CAPA_TITULO)
    for p in list(alvo.paragraphs):
        p._p.getparent().remove(p._p)

    # TEOR DA SOLICITAÇÃO — valor em negrito (herda), justificado
    p = alvo.add_paragraph(style=estilo)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _espacar(p, 36, 10)
    _run_label(p, "TEOR DA SOLICITAÇÃO: ")
    _run_valor(p, teor, bold=None)

    # SOLICITANTE — valor sem negrito
    p = alvo.add_paragraph(style=estilo)
    _espacar(p, 24, 12)
    _run_label(p, "SOLICITANTE: ")
    _run_valor(p, solicitante, bold=False)

    # AUTOR — valor (consultor + cargo + área) sem negrito, em três linhas
    p = alvo.add_paragraph(style=estilo)
    _espacar(p, 24, 36)
    _run_label(p, "AUTOR: ")
    for i, linha in enumerate(autor_linhas):
        if i > 0:
            _run_valor(p, "", bold=False).add_break()
        _run_valor(p, linha, bold=False)


# ---------------------------------------------------------------------------
# build IT
# ---------------------------------------------------------------------------
def build_it(sep: PaginaSeparada, abertura: Abertura, meta: MetaDocumento) -> Document:
    doc = Document(str(config.TEMPLATE_IT))
    _preencher_capa(doc, abertura.teor, meta.solicitante_capa, meta.autor_linhas)

    _p(doc, S.CORPO, text=meta.vocativo)
    _p(doc, S.CORPO, text=abertura.encaminhamento)
    _p(doc, S.CORPO, text=config.BLOCO_RESOLUCAO_INTRO)
    for linha in config.BLOCO_RESOLUCAO_TRANSCRICAO:
        _p(doc, S.TRANSCRICAO, text=linha)
    _p(doc, S.CORPO, text=abertura.transicao)

    _render_it_blocks(doc, sep.it_blocks)

    _p(doc, S.FECHO, text=meta.fecho_it_txt)
    _p(doc, S.ASSINATURA, text=meta.consultor)
    _p(doc, S.ASSINATURA, text=meta.consultor_cargo)
    _p(doc, S.SISCONLE, text=meta.sisconle_txt)
    return doc


# ---------------------------------------------------------------------------
# build proposição
# ---------------------------------------------------------------------------
_PREAMBULO_RE = re.compile(r"\b(decreta|resolve|promulga)", re.IGNORECASE)


def _epigrafe_text(sep: PaginaSeparada, meta: MetaDocumento) -> str:
    """Normaliza a epígrafe para o padrão da casa, preservando o tipo detectado."""
    nome = sep.tipo.nome_extenso
    return f"{nome} Nº          , DE {meta.ano}"


def _render_articulado(doc, blocks: List[Block], tipo: TipoProposicao):
    for b in blocks:
        if b.type == "paragraph":
            t = plain(b.rich).strip()
            if not t:
                continue
            if _PREAMBULO_RE.search(t) and len(t) <= 120:
                _p(doc, S.PREAMBULO, text=tipo.preambulo)
            else:
                _p(doc, S.CORPO, rich=b.rich)
        elif b.type == "quote":
            _add_quote_as_transcricao(doc, b.rich)
        elif b.type in ("heading_1", "heading_2", "heading_3"):
            # eventual subtítulo dentro do articulado -> corpo
            _p(doc, S.CORPO, rich=b.rich)
        elif b.type == "bulleted_list_item" or b.type == "numbered_list_item":
            _p(doc, S.CORPO, rich=b.rich)
        elif b.type == "table":
            _add_table(doc, b)


def _render_justificativa(doc, blocks: List[Block]):
    for b in blocks:
        if b.type == "quote":
            _add_quote_as_transcricao(doc, b.rich)
        elif b.type in ("heading_1", "heading_2", "heading_3"):
            if plain(b.rich).strip():
                _p(doc, S.CORPO, rich=b.rich)
        elif b.type in ("bulleted_list_item", "numbered_list_item"):
            _add_bullet(doc, b.rich)
        elif b.type == "table":
            _add_table(doc, b)
        elif plain(b.rich).strip():
            _p(doc, S.CORPO, rich=b.rich)


def build_proposicao(sep: PaginaSeparada, meta: MetaDocumento) -> Document:
    doc = Document(str(config.TEMPLATE_PROPOSICAO))
    tipo = sep.tipo

    _p(doc, S.EPIGRAFE, text=_epigrafe_text(sep, meta))
    _p(doc, S.AUTORIA, text=meta.autoria_prop)
    _p(doc, S.EMENTA, rich=sep.ementa or [RichText(plain(sep.ementa))])

    _render_articulado(doc, sep.articulado_blocks, tipo)

    _p(doc, S.JUSTIFICACAO, text="Justificação")
    _render_justificativa(doc, sep.justificativa_blocks)

    _p(doc, S.FECHO, text=meta.fecho_prop_txt(tipo.local_fecho))
    _p(doc, S.ASSINATURA, text=meta.assinatura_prop)
    _p(doc, S.SISCONLE, text=meta.sisconle_txt)
    return doc
