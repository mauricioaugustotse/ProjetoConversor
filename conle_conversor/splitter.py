# -*- coding: utf-8 -*-
"""Separa a página do Notion em: cabeçalho, corpo da IT, minuta e justificativa.
Também reconhece o layout de PARECER DE COMISSÃO (Relatório/Voto/Substitutivo)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import List, Optional

from . import classifier
from .notion_parser import Block, RichText, plain


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).upper().strip()


def _is_heading(b: Block) -> bool:
    return b.type in ("heading_1", "heading_2", "heading_3", "heading_4")


def _heading_text(b: Block) -> str:
    return _norm(plain(b.rich))


@dataclass
class Minuta:
    """Uma minuta de proposição da página (a página pode trazer mais de uma,
    ex. "PL e PLP": cada epígrafe com articulado e justificativa próprios)."""
    epigrafe: List[RichText] = field(default_factory=list)
    ementa: List[RichText] = field(default_factory=list)
    articulado_blocks: List[Block] = field(default_factory=list)
    justificativa_blocks: List[Block] = field(default_factory=list)
    tipo: classifier.TipoProposicao = classifier.PL


@dataclass
class PaginaSeparada:
    titulo: str
    objeto: str                                  # texto após "Objeto:" do callout-cabeçalho
    introducao_texto: str                        # texto da seção introdutória (p/ harmonização)
    it_blocks: List[Block] = field(default_factory=list)
    # Campos da 1ª minuta (compatibilidade: build_proposicao e chamadores
    # antigos leem daqui); a lista completa fica em `minutas`.
    epigrafe: List[RichText] = field(default_factory=list)
    ementa: List[RichText] = field(default_factory=list)
    articulado_blocks: List[Block] = field(default_factory=list)
    justificativa_blocks: List[Block] = field(default_factory=list)
    tipo: classifier.TipoProposicao = classifier.PL
    minutas: List[Minuta] = field(default_factory=list)


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

    # 2) marcos: epígrafes das minutas (a página pode ter mais de uma, ex. "PL e PLP")
    idx_epigrafes = [
        i for i in range(idx_apos_cabecalho, len(blocks)) if _eh_callout_epigrafe(blocks[i])
    ]

    def _inicio_minuta(idx_epi: int) -> int:
        # recua sobre headings/divisores órfãos que anunciam a minuta
        inicio = idx_epi
        j = idx_epi - 1
        while j >= idx_apos_cabecalho and (
            blocks[j].type in ("divider",)
            or (_is_heading(blocks[j]) and "MINUTA" in _heading_text(blocks[j]))
        ):
            inicio = j
            j -= 1
        return inicio

    inicios = [_inicio_minuta(i) for i in idx_epigrafes]

    # 3) fim da IT = início da 1ª minuta (heading "MINUTA"/divisores antes da epígrafe)
    it_end = inicios[0] if idx_epigrafes else len(blocks)
    it_blocks = blocks[idx_apos_cabecalho:it_end]

    # 4) minutas: cada epígrafe delimita articulado + justificativa até a próxima
    minutas: List[Minuta] = []
    for k, idx_epi in enumerate(idx_epigrafes):
        fim = inicios[k + 1] if k + 1 < len(idx_epigrafes) else len(blocks)
        idx_justif: Optional[int] = None
        for i in range(idx_epi + 1, fim):
            if _is_heading(blocks[i]) and ("JUSTIFIC" in _heading_text(blocks[i])):
                idx_justif = i
                break
        epigrafe, ementa = _split_callout_epigrafe(blocks[idx_epi])
        fim_articulado = idx_justif if idx_justif is not None else fim
        justificativa = blocks[idx_justif + 1 : fim] if idx_justif is not None else []
        # Tipo: com uma única minuta o título da página ajuda na detecção; com
        # várias ele é ambíguo por definição ("PL e PLP") e só a epígrafe e o
        # heading da própria minuta decidem.
        textos_tipo = ([titulo] if len(idx_epigrafes) == 1 else []) + [
            plain(epigrafe), _heading_text_da_minuta(blocks, idx_epi)
        ]
        minutas.append(Minuta(
            epigrafe=epigrafe,
            ementa=ementa,
            articulado_blocks=blocks[idx_epi + 1 : fim_articulado],
            justificativa_blocks=justificativa,
            tipo=classifier.detectar_tipo(*textos_tipo),
        ))

    # 5) texto introdutório (1ª seção) para harmonização
    introducao_texto = _coleta_introducao(it_blocks)

    primeira = minutas[0] if minutas else Minuta(tipo=classifier.detectar_tipo(titulo, "", ""))

    return PaginaSeparada(
        titulo=titulo,
        objeto=objeto,
        introducao_texto=introducao_texto,
        it_blocks=it_blocks,
        epigrafe=primeira.epigrafe,
        ementa=primeira.ementa,
        articulado_blocks=primeira.articulado_blocks,
        justificativa_blocks=primeira.justificativa_blocks,
        tipo=primeira.tipo,
        minutas=minutas,
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


# ---------------------------------------------------------------------------
# Parecer de comissão (Relatório / Voto / Substitutivo)
# ---------------------------------------------------------------------------
# Obs.: _norm descarta o travessão (– e — não são ASCII), então "I – Relatório"
# normaliza para "I RELATORIO"; só o hífen ASCII sobrevive — daí o "-?".
_RE_HEAD_RELATORIO = re.compile(r"^I\s*-?\s*RELATORIO")
_RE_HEAD_VOTO = re.compile(r"^II\s*-?\s*VOTO")
_RE_SIGLA_COMISSAO = re.compile(r"\(([A-Z]{2,6})\)")
_RE_PREFIXO_CAMPO = re.compile(
    r"^\s*(Proposi[çc][ãa]o|Autor(?:ia|a)?|Relator(?:a)?|Ementa|"
    r"Objeto(?:\s+do\s+parecer)?|Apensad[oa]s?|Apensos?)\s*:\s*",
    re.IGNORECASE,
)
# Designação de proposição sem rótulo ("Projeto de Lei nº 3.263, de 2023",
# "PL 3.263/2023"), comparada sobre _norm ("nº" vira "NO"); usada quando a
# página traz o cabeçalho no formato do documento final, sem "Proposição:" etc.
_RE_DESIGNACAO_PROP = re.compile(
    r"^(?:PROJETO DE LEI(?: COMPLEMENTAR)?|PROPOSTA DE EMENDA(?: A CONSTITUICAO)?|"
    r"PROJETO DE DECRETO(?: LEGISLATIVO)?|PROJETO DE RESOLUCAO|MEDIDA PROVISORIA|"
    r"PL|PLP|PEC|PDL|PDC|PRC)\s+(?:N[O.]?\s*)?[\d.]+"
)
# Separador de campos na mesma linha: "Autor: X · Relatora: Y"
_RE_SEP_CAMPOS = re.compile(r"\s*[·•]\s*")
_RE_ASSINATURA_RELATOR = re.compile(r"^(?P<nome>.+?)\s*[-–—]\s*Relator[a]?\.?\s*$", re.IGNORECASE)
_RE_FECHO_COMISSAO = re.compile(r"^\s*Sala da Comiss[ãa]o", re.IGNORECASE)


@dataclass
class ParecerSeparado:
    titulo: str
    comissao: str = ""            # "Comissão de Constituição e Justiça e de Cidadania (CCJC)"
    comissao_sigla: str = ""      # "CCJC" ("" se ausente)
    proposicao: str = ""          # "Projeto de Lei nº 3.031, de 2025"
    apensados: str = ""           # "" se não houver
    autoria: str = ""             # "Deputada Simone Marquetto (MDB/SP)"
    relator: str = ""             # "" => fallback (assinatura da página / GUI / placeholder)
    ementa: List[RichText] = field(default_factory=list)
    objeto: str = ""              # "Objeto do parecer: ..." (metadado; não renderizado)
    relatorio_heading: str = ""   # texto do heading como veio ("I – Relatório")
    voto_heading: str = ""        # "II – Voto do Relator"
    relatorio_blocks: List[Block] = field(default_factory=list)
    voto_blocks: List[Block] = field(default_factory=list)
    tem_substitutivo: bool = False
    sub_epigrafe: str = ""        # texto do heading "Substitutivo ao Projeto de Lei nº ..."
    sub_ementa: List[RichText] = field(default_factory=list)
    sub_articulado_blocks: List[Block] = field(default_factory=list)
    tipo: classifier.TipoProposicao = classifier.PL


def _eh_callout_comissao(b: Block) -> bool:
    if b.type != "callout":
        return False
    t = _norm(plain(b.rich))
    return "COMISSAO" in t or _RE_SIGLA_COMISSAO.search(t) is not None


def detectar_layout(blocks: List[Block]) -> str:
    """Classifica a página: "parecer" (callout de comissão + headings I-Relatório
    e II-Voto) ou "it" (layout IT+minuta tratado por split_page)."""
    tem_comissao = False
    nao_vazios = 0
    for b in blocks:
        txt = plain(b.rich).strip()
        if not txt:
            continue
        nao_vazios += 1
        if nao_vazios > 12:
            break
        t = _norm(txt)
        if _eh_callout_comissao(b):
            tem_comissao = True
        elif b.type == "paragraph" and (
            t.startswith("OBJETO DO PARECER") or t.startswith("RELATOR:") or t.startswith("RELATORA:")
        ):
            tem_comissao = True  # reforço quando o callout é atípico
    tem_relatorio = any(_is_heading(b) and _RE_HEAD_RELATORIO.match(_heading_text(b)) for b in blocks)
    tem_voto = any(_is_heading(b) and _RE_HEAD_VOTO.match(_heading_text(b)) for b in blocks)
    return "parecer" if (tem_comissao and tem_relatorio and tem_voto) else "it"


def _strip_prefixo_rich(rich: List[RichText], n: int) -> List[RichText]:
    """Remove os primeiros n caracteres atravessando os runs (preserva a
    formatação/links do restante)."""
    out: List[RichText] = []
    restante = n
    for r in rich:
        if restante >= len(r.text):
            restante -= len(r.text)
            continue
        out.append(replace(r, text=r.text[restante:]) if restante > 0 else r)
        restante = 0
    return out


def _campo_chave(rotulo: str) -> str:
    """Prefixo capturado → nome do campo do ParecerSeparado."""
    r = _norm(rotulo)
    if r.startswith("PROPOSI"):
        return "proposicao"
    if r.startswith("AUTOR"):
        return "autoria"
    if r.startswith("RELATOR"):
        return "relator"
    if r.startswith("EMENTA"):
        return "ementa"
    if r.startswith("OBJETO"):
        return "objeto"
    return "apensados"  # APENSADOS / APENSOS


def _consumir_campo_cabecalho(par: ParecerSeparado, b: Block, txt: str, ultimo_campo: str) -> str:
    """Atribui um parágrafo entre o callout e o I-Relatório aos campos do
    parecer. Aceita o formato rotulado ("Proposição: …", "Ementa: …") e o
    formato "cru" do documento final (designação sem rótulo, parágrafo-ementa,
    "Autor: X · Relatora: Y" na mesma linha). Devolve o último campo tratado
    (para continuação multi-parágrafo)."""
    # linha multi-campo: divide em "·"/"•" quando os segmentos são rotulados
    partes = _RE_SEP_CAMPOS.split(txt)
    if len(partes) > 1 and all(_RE_PREFIXO_CAMPO.match(p) for p in partes[1:]):
        for parte in partes:
            m = _RE_PREFIXO_CAMPO.match(parte)
            if m:
                ultimo_campo = _campo_chave(m.group(1))
                valor = parte[m.end():].strip()
                if ultimo_campo == "ementa":
                    par.ementa = [RichText(valor)]
                else:
                    setattr(par, ultimo_campo, valor)
            elif not par.proposicao and _RE_DESIGNACAO_PROP.match(_norm(parte)):
                par.proposicao = parte.strip()
                ultimo_campo = ""
        return ultimo_campo

    m = _RE_PREFIXO_CAMPO.match(txt)
    if m:
        campo = _campo_chave(m.group(1))
        if campo == "ementa":
            par.ementa = _strip_prefixo_rich(b.rich, m.end())
        else:
            setattr(par, campo, txt[m.end():].strip())
        return campo
    # sem rótulo: 1º candidato é a designação da proposição (linha única)…
    if not par.proposicao and _RE_DESIGNACAO_PROP.match(_norm(txt)):
        par.proposicao = txt
        return ""
    if ultimo_campo == "ementa":  # continuação da ementa (multi-parágrafo)
        par.ementa = par.ementa + [RichText("\n")] + list(b.rich)
        return ultimo_campo
    if ultimo_campo:  # continuação de campo texto
        atual = getattr(par, ultimo_campo)
        setattr(par, ultimo_campo, f"{atual} {txt}".strip())
        return ultimo_campo
    # …e, já identificada a proposição, o parágrafo sem rótulo é a ementa
    if par.proposicao and not par.ementa:
        par.ementa = list(b.rich)
        return "ementa"
    return ultimo_campo


def split_parecer(blocks: List[Block], titulo: str) -> ParecerSeparado:
    par = ParecerSeparado(titulo=titulo)

    # 1) callout da comissão + campos nos parágrafos seguintes
    idx_apos_cabecalho = 0
    for i, b in enumerate(blocks):
        if _eh_callout_comissao(b):
            linha = plain(b.rich).strip().splitlines()[0].strip()
            m = _RE_SIGLA_COMISSAO.search(linha)
            # sufixo após a sigla (ex.: "… (CCJC) — Parecer de Admissibilidade")
            # fica fora do parágrafo COMISSÃO, como no modelo da casa
            par.comissao = linha[: m.end()].strip() if m else linha
            par.comissao_sigla = m.group(1) if m else ""
            j = i + 1
            ultimo_campo = ""
            while j < len(blocks) and blocks[j].type in ("paragraph", "divider"):
                txt = plain(blocks[j].rich).strip()
                if txt:
                    ultimo_campo = _consumir_campo_cabecalho(par, blocks[j], txt, ultimo_campo)
                j += 1
            idx_apos_cabecalho = j
            break

    # 2) seções Relatório / Voto
    idx_rel: Optional[int] = None
    idx_voto: Optional[int] = None
    for i in range(idx_apos_cabecalho, len(blocks)):
        if not _is_heading(blocks[i]):
            continue
        t = _heading_text(blocks[i])
        if idx_rel is None and _RE_HEAD_RELATORIO.match(t):
            idx_rel = i
        elif idx_rel is not None and idx_voto is None and _RE_HEAD_VOTO.match(t):
            idx_voto = i
            break

    # 3) substitutivo: primeiro heading após o Voto iniciando com "SUBSTITUTIVO"
    idx_sub: Optional[int] = None
    inicio_busca_sub = (idx_voto + 1) if idx_voto is not None else idx_apos_cabecalho
    for i in range(inicio_busca_sub, len(blocks)):
        if _is_heading(blocks[i]) and _heading_text(blocks[i]).startswith("SUBSTITUTIVO"):
            idx_sub = i
            break

    if idx_rel is not None:
        par.relatorio_heading = plain(blocks[idx_rel].rich).strip()
        fim_rel = idx_voto if idx_voto is not None else (idx_sub if idx_sub is not None else len(blocks))
        par.relatorio_blocks = blocks[idx_rel + 1 : fim_rel]
    if idx_voto is not None:
        par.voto_heading = plain(blocks[idx_voto].rich).strip()
        fim_voto = idx_sub if idx_sub is not None else len(blocks)
        par.voto_blocks = blocks[idx_voto + 1 : fim_voto]

    # 4) conteúdo do substitutivo: ementa + articulado (fecho/assinatura do
    #    Notion são descartados — o builder gera fecho e assinatura padronizados)
    if idx_sub is not None:
        par.tem_substitutivo = True
        par.sub_epigrafe = plain(blocks[idx_sub].rich).strip()
        corpo = blocks[idx_sub + 1 :]
        k = 0
        while k < len(corpo) and not plain(corpo[k].rich).strip():
            k += 1
        if k < len(corpo) and corpo[k].type == "paragraph":
            par.sub_ementa = list(corpo[k].rich)
            k += 1
        articulado: List[Block] = []
        for b in corpo[k:]:
            txt = plain(b.rich).strip()
            if _RE_FECHO_COMISSAO.match(txt):
                break
            articulado.append(b)
        par.sub_articulado_blocks = articulado
        # assinatura ao final da página como fallback do relator
        if not par.relator:
            for b in reversed(corpo):
                m = _RE_ASSINATURA_RELATOR.match(plain(b.rich).strip())
                if m:
                    par.relator = m.group("nome").strip()
                    break

    par.tipo = classifier.detectar_tipo(par.proposicao, par.sub_epigrafe, titulo)
    return par


# Caput de artigo dentro de quote do articulado: '"Art. 165-A. Quando a…"' /
# '"Art. 161. (…)' / '"Art. 36-A. ......'.
_RE_CAPUT_QUOTE = re.compile(
    r"^\s*[\"'“”‘’]?\s*Art\.?\s*(\d+)(?:[ºo°])?(?:\s*[-–‑]\s*([A-Za-z])\b)?\s*\.?\s*(?P<resto>.*)$"
)
# Caput preservado: "(…)"/"(...)" ou linha de pontilhado de supressão — o
# artigo EXISTE na norma alterada (só recebe §/inciso ou mantém o caput).
_RE_CAPUT_PRESERVADO = re.compile(r"^(?:\(\s*(?:…|\.{3,})\s*\)|[.…\s]{3,}$)")
_RE_VERBO_ACRESCIMO = re.compile(
    r"acrescid\w+|acrescent\w+|inclu[ií]\w+|inserid\w+|insere\w*|introduz\w*", re.I
)


def dispositivos_criados(articulado_blocks: List[Block]) -> frozenset:
    """Artigos CRIADOS pelo articulado da minuta — conjunto de (num, LETRA)
    usado para NÃO linkar esses dispositivos quando citados na justificação/
    IT/parecer (achado do usuário, 06/07/2026: "art. 66-A da Lei das Eleições"
    proposto saía linkado para uma âncora que não existe; a decisão de
    05/07/2026 já vedava o link a dispositivo em deliberação).

    Heurística (calibrada nas minutas reais de 05-06/07/2026): caput com
    "(…)"/pontilhado ⇒ artigo EXISTENTE; caput de texto pleno ⇒ criado quando
    tem sufixo de letra ("Art. 165-A. Quando…") ou quando o comando de
    alteração imediatamente anterior ACRESCE aquele artigo — mas NÃO quando o
    comando altera o próprio artigo ("O art. 36-A … passa a vigorar…", que
    prova que ele existe) nem quando é reescrita sem letra ("Art. 22.
    Qualquer partido…" sob "passa a vigorar com as seguintes alterações")."""
    criados = set()
    comando = ""
    for b in articulado_blocks or []:
        if b.type == "paragraph":
            t = plain(b.rich).strip()
            if t:
                comando = t
            continue
        if b.type != "quote":
            continue
        for linha in plain(b.rich).splitlines():
            m = _RE_CAPUT_QUOTE.match(linha)
            if not m:
                continue
            num, letra = int(m.group(1)), (m.group(2) or "").upper()
            resto = m.group("resto").strip()
            if _RE_CAPUT_PRESERVADO.match(resto or "…"):
                continue  # "(…)" ou "……": artigo existente
            ref = rf"art(?:igo)?s?\.?\s*{num}(?:[ºo°])?" + (
                rf"\s*[-–‑]\s*{letra}\b" if letra else r"(?!\s*[-–‑]\s*[A-Za-z])"
            )
            if re.search(rf"\b{ref}[^.;]{{0,80}}?passa(?:m|r[áã]o?)?\s+a\s+vigorar", comando, re.I):
                continue  # o comando altera o PRÓPRIO artigo: ele existe
            if letra:
                criados.add((num, letra))
            elif _RE_VERBO_ACRESCIMO.search(comando) and re.search(rf"\b{ref}", comando, re.I):
                criados.add((num, letra))  # acréscimo explícito de artigo sem letra
    return frozenset(criados)


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
