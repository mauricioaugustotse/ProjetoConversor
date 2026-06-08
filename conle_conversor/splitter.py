# -*- coding: utf-8 -*-
"""Separa a página do Notion em: cabeçalho, corpo da IT, minuta e justificativa."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from . import classifier
from .notion_parser import Block, RichText, plain


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).upper().strip()


def _is_heading(b: Block) -> bool:
    return b.type in ("heading_1", "heading_2", "heading_3")


def _heading_text(b: Block) -> str:
    return _norm(plain(b.rich))


@dataclass
class PaginaSeparada:
    titulo: str
    objeto: str                                  # texto após "Objeto:" do callout-cabeçalho
    introducao_texto: str                        # texto da seção introdutória (p/ harmonização)
    it_blocks: List[Block] = field(default_factory=list)
    epigrafe: List[RichText] = field(default_factory=list)
    ementa: List[RichText] = field(default_factory=list)
    articulado_blocks: List[Block] = field(default_factory=list)
    justificativa_blocks: List[Block] = field(default_factory=list)
    tipo: classifier.TipoProposicao = classifier.PL


def _extrai_objeto(callout: Block) -> str:
    full = plain(callout.rich)
    m = re.search(r"Objeto\s*:\s*(.+)", full, re.S)
    return (m.group(1).strip() if m else full.strip())


def _eh_callout_cabecalho(b: Block) -> bool:
    if b.type != "callout":
        return False
    t = _norm(plain(b.rich))
    return ("OBJETO" in t) or ("CONSULTORIA LEGISLATIVA" in t) or ("ESTUDO" in t)


def _eh_callout_epigrafe(b: Block) -> bool:
    if b.type != "callout":
        return False
    t = _norm(plain(b.rich))
    return any(
        k in t
        for k in (
            "PROJETO DE LEI",
            "PROPOSTA DE EMENDA",
            "PROJETO DE RESOLUCAO",
            "PROJETO DE DECRETO",
        )
    )


def _split_callout_epigrafe(b: Block):
    """Separa a 1ª linha (epígrafe) do restante (ementa) dentro do callout."""
    linhas_rich: List[List[RichText]] = [[]]
    for r in b.rich:
        partes = r.text.split("\n")
        for i, p in enumerate(partes):
            if i > 0:
                linhas_rich.append([])
            if p:
                linhas_rich[-1].append(RichText(p, r.bold, r.italic, r.underline, r.href))
    linhas_rich = [ln for ln in linhas_rich if any(x.text.strip() for x in ln)]
    epigrafe = linhas_rich[0] if linhas_rich else []
    ementa: List[RichText] = []
    for ln in linhas_rich[1:]:
        if ementa:
            ementa.append(RichText("\n"))
        ementa.extend(ln)
    return epigrafe, ementa


def split_page(blocks: List[Block], titulo: str) -> PaginaSeparada:
    # 1) cabeçalho (callout com Objeto)
    objeto = ""
    idx_apos_cabecalho = 0
    for i, b in enumerate(blocks):
        if _eh_callout_cabecalho(b):
            objeto = _extrai_objeto(b)
            idx_apos_cabecalho = i + 1
            break
    # pula dividers/imagens imediatamente após o cabeçalho
    while idx_apos_cabecalho < len(blocks) and blocks[idx_apos_cabecalho].type in ("divider", "image"):
        idx_apos_cabecalho += 1

    # 2) marcos: início da minuta (callout de epígrafe) e justificativa
    idx_epigrafe: Optional[int] = None
    for i in range(idx_apos_cabecalho, len(blocks)):
        if _eh_callout_epigrafe(blocks[i]):
            idx_epigrafe = i
            break

    idx_justif: Optional[int] = None
    if idx_epigrafe is not None:
        for i in range(idx_epigrafe + 1, len(blocks)):
            if _is_heading(blocks[i]) and ("JUSTIFIC" in _heading_text(blocks[i])):
                idx_justif = i
                break

    # 3) fim da IT = heading "MINUTA" imediatamente antes da epígrafe, ou a própria epígrafe
    if idx_epigrafe is not None:
        it_end = idx_epigrafe
        # recua sobre headings/divisores órfãos que anunciam a minuta
        j = idx_epigrafe - 1
        while j >= idx_apos_cabecalho and (
            blocks[j].type in ("divider",)
            or (_is_heading(blocks[j]) and "MINUTA" in _heading_text(blocks[j]))
        ):
            it_end = j
            j -= 1
    else:
        it_end = len(blocks)

    it_blocks = blocks[idx_apos_cabecalho:it_end]

    # 4) minuta
    epigrafe: List[RichText] = []
    ementa: List[RichText] = []
    articulado_blocks: List[Block] = []
    justificativa_blocks: List[Block] = []
    if idx_epigrafe is not None:
        epigrafe, ementa = _split_callout_epigrafe(blocks[idx_epigrafe])
        fim_articulado = idx_justif if idx_justif is not None else len(blocks)
        articulado_blocks = blocks[idx_epigrafe + 1 : fim_articulado]
        if idx_justif is not None:
            justificativa_blocks = blocks[idx_justif + 1 :]

    # 5) texto introdutório (1ª seção) para harmonização
    introducao_texto = _coleta_introducao(it_blocks)

    tipo = classifier.detectar_tipo(titulo, plain(epigrafe), _heading_text_da_minuta(blocks, idx_epigrafe))

    return PaginaSeparada(
        titulo=titulo,
        objeto=objeto,
        introducao_texto=introducao_texto,
        it_blocks=it_blocks,
        epigrafe=epigrafe,
        ementa=ementa,
        articulado_blocks=articulado_blocks,
        justificativa_blocks=justificativa_blocks,
        tipo=tipo,
    )


def _heading_text_da_minuta(blocks: List[Block], idx_epigrafe: Optional[int]) -> str:
    if idx_epigrafe is None:
        return ""
    for j in range(idx_epigrafe - 1, -1, -1):
        if _is_heading(blocks[j]):
            return _heading_text(blocks[j])
        if j < idx_epigrafe - 3:
            break
    return ""


def _coleta_introducao(it_blocks: List[Block]) -> str:
    """Concatena os parágrafos da primeira seção (até o 2º heading) para dar
    contexto à harmonização da abertura."""
    textos: List[str] = []
    headings_vistos = 0
    for b in it_blocks:
        if _is_heading(b):
            headings_vistos += 1
            if headings_vistos >= 2:
                break
            continue
        if b.type == "paragraph":
            t = plain(b.rich).strip()
            if t:
                textos.append(t)
        if sum(len(x) for x in textos) > 2500:
            break
    return "\n\n".join(textos)
