# -*- coding: utf-8 -*-
"""Conversão de rich text (Notion) em runs do Word."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Optional

from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from . import config
from .notion_parser import RichText

# Cor azul padrão de hyperlink do Word (igual ao estilo "Hyperlink").
_COR_LINK = "0563C1"

# Domínios internos do Notion: nunca viram hyperlink no .docx (o público externo
# não acessa as bases). notion.site (páginas publicadas) fica de fora de propósito.
_DOMINIOS_NOTION = re.compile(
    r"^(notion://|https?://(www\.)?notion\.so\b|https?://app\.notion\.com\b)", re.I
)


def _link_interno_notion(href: str) -> bool:
    return bool(href) and (href.startswith("/") or _DOMINIOS_NOTION.match(href) is not None)


def resolver_fonte_publica(texto: str) -> Optional[str]:
    """Mapeia o texto de uma mention interna ("RICD - Art. 54") para a URL da
    fonte pública oficial, via config.FONTES_OFICIAIS. None se não mapeada."""
    for padrao, template, _aliases in config.FONTES_OFICIAIS:
        m = re.search(padrao, texto or "")
        if m:
            return m.expand(template)
    return None


def _aliases_da_norma(texto: str) -> List[str]:
    """Aliases (nome por extenso/sigla) da norma citada em `texto`, se
    reconhecida em config.FONTES_OFICIAIS — usados para podar o eco redundante
    que às vezes segue a mention (ex. "RICD" repetido depois de "RICD - Art. 54")."""
    for padrao, _template, aliases in config.FONTES_OFICIAIS:
        if re.search(padrao, texto or ""):
            return list(aliases)
    return []


# Janela (em caracteres) em que se procura o eco logo após a mention — curta o
# bastante para não alcançar uma citação legítima e distante da MESMA norma
# mais adiante (ex. outro dispositivo) nem uma definição por extenso da sigla
# (ex. "do Regimento Interno da Câmara dos Deputados (RICD)", ~47 chars);
# os ecos redundantes observados ficam a até ~26 chars do início do trecho.
_JANELA_ECO = 32


_RE_ESPACOS_DUPLOS = re.compile(r"[ \t]{2,}")
_RE_ESPACO_ANTES_PONTUACAO = re.compile(r"[ \t]+([,.;:)])")


def _encontrar_eco(texto: str, aliases: List[str]) -> Optional[tuple]:
    """Localiza, dentro de uma janela curta a partir do início de `texto`, o eco
    redundante do nome/sigla de uma norma já citada pela mention anterior.
    Dois padrões: (1) fecha parênteses logo em seguida — o eco vai até (mas sem
    incluir) o ")" (ex. ", RICD)" -> ")"); (2) "do/da <norma>" no meio da frase
    (ex. ", caput e inciso VI, da CF permanecem" -> ", caput e inciso VI, permanecem").
    Retorna (início, fim) do trecho a remover, ou None."""
    if not aliases or not texto:
        return None
    alt = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    padrao = re.compile(
        rf"[,;]?\s*(?:\bdo\b\s+|\bda\b\s+)?(?:{alt})\b\s*(?=\))"
        rf"|\b(?:do|da)\b\s+(?:{alt})\b"
    )
    m = padrao.search(texto[:_JANELA_ECO])
    return (m.start(), m.end()) if m else None


def _remover_eco(texto: str, aliases: List[str]) -> str:
    """Variante de conveniência de _encontrar_eco que já retorna o texto podado
    e com espaços/pontuação normalizados (usada quando o eco cabe num único
    run; limpar_ecos_redundantes trata o caso geral, de vários runs)."""
    span = _encontrar_eco(texto, aliases)
    if not span:
        return texto
    novo = texto[: span[0]] + texto[span[1] :]
    novo = _RE_ESPACOS_DUPLOS.sub(" ", novo)
    novo = _RE_ESPACO_ANTES_PONTUACAO.sub(r"\1", novo)
    return novo


def limpar_ecos_redundantes(rich_list: List[RichText]) -> List[RichText]:
    """Varre a lista de rich text de um parágrafo e, após cada trecho que cita
    uma norma reconhecida (mention já resolvida por extenso), poda o eco
    redundante que segue. O eco pode cair num run diferente do imediatamente
    seguinte (ex.: um trecho em itálico separa a mention do eco em runs
    distintos: ", " + "caput" + " e inciso VI, da CF...") — por isso os runs
    seguintes são concatenados (até a janela) para a busca, e a remoção é
    redistribuída para os runs originais conforme a sobreposição de índices."""
    out: List[RichText] = list(rich_list or [])
    i = 0
    while i < len(out):
        aliases = _aliases_da_norma(out[i].text)
        if not aliases:
            i += 1
            continue
        limites = []  # (índice do run, início, fim) no texto concatenado
        acumulado = ""
        j = i + 1
        while j < len(out) and len(acumulado) < _JANELA_ECO:
            limites.append((j, len(acumulado), len(acumulado) + len(out[j].text)))
            acumulado += out[j].text
            j += 1
        span = _encontrar_eco(acumulado, aliases)
        if span:
            ini, fim = span
            for idx, r_ini, r_fim in limites:
                corte_ini, corte_fim = max(ini, r_ini), min(fim, r_fim)
                if corte_ini < corte_fim:
                    texto_r = out[idx].text
                    novo = texto_r[: corte_ini - r_ini] + texto_r[corte_fim - r_ini :]
                    novo = _RE_ESPACOS_DUPLOS.sub(" ", novo)
                    novo = _RE_ESPACO_ANTES_PONTUACAO.sub(r"\1", novo)
                    out[idx] = replace(out[idx], text=novo)
        i += 1
    return out


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
    """Adiciona runs ao parágrafo preservando bold/italic/underline e os hyperlinks
    EXTERNOS (viram links clicáveis no .docx). Links/mentions internos do Notion
    nunca viram hyperlink: quando o texto casa com config.FONTES_OFICIAIS, ganham
    link para a fonte pública oficial; senão, saem como texto puro. Também poda o
    eco redundante do nome da norma que às vezes segue a mention (ver
    limpar_ecos_redundantes). '\n' dentro de um run vira quebra de linha (soft break)."""
    rich_list = limpar_ecos_redundantes(rich_list)
    for r in rich_list or []:
        if not r.text:
            continue
        href = (r.href or "").strip()
        if _link_interno_notion(href):
            # o match usa o texto completo da mention, não cada parte pós-split
            href = resolver_fonte_publica(r.text) or ""
        partes = r.text.split("\n")
        for i, parte in enumerate(partes):
            if i > 0:
                paragraph.add_run().add_break()
            if parte == "":
                continue
            b = r.bold if force_bold is None else force_bold
            if href.startswith("http"):
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
