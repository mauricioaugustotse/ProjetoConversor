# -*- coding: utf-8 -*-
"""Orquestra a conversão: URL do Notion -> .docx de IT e/ou minuta de proposição."""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import config
from .config import data_extenso
from .docx_builder import build_it, build_parecer, build_proposicao
from .harmonizer import gerar_abertura
from .meta import MetaDocumento
from .notion_api import fetch_page, resolver_mentions_publicas
from .notion_parser import Block, parse_blocks, flatten_blocks
from .residual import polir_residuos_pagina
from .splitter import detectar_layout, split_page, split_parecer, PaginaSeparada, ParecerSeparado


_INVALIDOS = re.compile(r'[\\/:*?"<>|]+')
# Prefixo de sigla de proposição no início do título da página (ex.: "PL - ",
# "PLP – "): redundante porque o nome do arquivo da minuta já traz a sigla
# ("Minuta de PLP 2026 - …"). Removê-lo evita "Minuta de PLP 2026 - PLP - …".
_SIGLA_PREFIXO = re.compile(
    r"^(PLP|PEC|PRC|PDL|PDC|PLV|MPV|PL)\s*[-–—:]\s*", re.IGNORECASE
)
# Título no layout invertido "Minuta — PEC: <tema>" / "Minuta de PEC — <tema>"
# (a palavra "minuta" vem ANTES do tema): o prefixo é descartado; a sigla que
# sobrar cai no _SIGLA_PREFIXO. Sem isso o tema virava literalmente "Minuta".
_MINUTA_PREFIXO = re.compile(
    r"^minuta(?:\s+de)?\s*(?:(?:PLP|PEC|PRC|PDL|PDC|PLV|MPV|PL)\b\s*)?[-–—:]+\s*",
    re.IGNORECASE,
)


def _sanitize(nome: str, limite: int = 150) -> str:
    nome = _INVALIDOS.sub("", nome)
    nome = re.sub(r"\s+", " ", nome).strip(" .")
    return nome[:limite].strip()


def _tema(titulo: str) -> str:
    base = _MINUTA_PREFIXO.sub("", (titulo or "").strip())
    base = re.split(r"[—–-]\s*minuta", base, flags=re.IGNORECASE)[0]
    base = base.split("—")[0].split("–")[0]
    base = _SIGLA_PREFIXO.sub("", base.strip(" -–—"))
    return _sanitize(base.strip(" -–—"))


# Solicitante e composição dos nomes de arquivo, no padrão das pastas-padrão:
#   IT:     "Informação Técnica - Dep. <Nome> - <tema>.docx"
#   minuta: "Minuta de <SIGLA> <ano> - <tema> (Dep. <Nome>).docx"
_DEP_PREFIXO = re.compile(r"^\s*dep(?:utad[oa])?\.?\s+", re.IGNORECASE)
# Solicitante que não é parlamentar individual não recebe o prefixo "Dep."
# (ex.: "Secretaria da Mulher - …").
_ORGAO = re.compile(
    r"\b(secretaria|comiss[ãa]o|mesa|bancada|lideran[çc]a|frente|n[úu]cleo|"
    r"procuradoria|ouvidoria|presid[êe]ncia|gabinete|colegiado)\b",
    re.IGNORECASE,
)
# Teto do nome do arquivo (sem extensão): as pastas-padrão são profundas e o
# Windows limita o caminho total; quando preciso, só o tema é encurtado.
_MAX_STEM = 160


def _solicitante_rotulo(nome: str) -> str:
    """Solicitante no padrão dos nomes de arquivo: "Dep. <Nome>" para parlamentar,
    o próprio nome para órgãos/colegiados, "" se não informado. Remove um prefixo
    "Dep."/"Deputado(a)" já digitado, para não duplicar."""
    nome = _DEP_PREFIXO.sub("", (nome or "").strip()).strip(" -–—")
    if not nome:
        return ""
    return nome if _ORGAO.search(nome) else f"Dep. {nome}"


def _compor_nome(fixas: str, tema: str, sufixo: str = "") -> str:
    """Monta "<fixas><tema><sufixo>.docx" sanitizado, encurtando SÓ o tema para
    caber em _MAX_STEM (preserva os componentes estruturais e o sufixo)."""
    folga = _MAX_STEM - len(fixas) - len(sufixo)
    if folga > 0 and len(tema) > folga:
        tema = tema[:folga].rstrip(" -–—.,;")
    return _sanitize(fixas + tema + sufixo, limite=400) + ".docx"


def _caminho_unico(pasta: Path, nome: str) -> Path:
    alvo = pasta / nome
    if not alvo.exists():
        return alvo
    stem, suf = alvo.stem, alvo.suffix
    i = 2
    while True:
        cand = pasta / f"{stem} ({i}){suf}"
        if not cand.exists():
            return cand
        i += 1


@dataclass
class ResultadoConversao:
    page_id: str
    titulo: str
    tipo_sigla: str
    tipo_extenso: str
    abertura_via_ia: bool
    caminhos: List[Path] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)


def _hoje() -> datetime.date:
    return datetime.date.today()


# Nome de arquivo do parecer, no padrão da pasta Pareceres:
#   "Parecer <SIGLA COMISSÃO> - <SIGLA> nº <num>-<ano> - <tema> (Rel. <Nome>).docx"
_RE_NUM_ANO = re.compile(r"n[ºo°.]?\s*([\d.]+)\s*,?\s*de\s*(\d{4})", re.IGNORECASE)
_RE_SIGLA_TITULO = re.compile(
    r"\b(PL|PLP|PEC|PDL|PDC|PRC)\s*n?[ºo°.]*\s*([\d.]+)\s*/\s*(\d{4})", re.IGNORECASE
)
_RE_TEMA_POS_SIGLA = re.compile(
    r"^(?:PL|PLP|PEC|PDL|PDC|PRC)[\s\d.,/º°-]*[—–-]\s*(.+)$", re.IGNORECASE
)


def _ref_proposicao(par: ParecerSeparado) -> str:
    m = _RE_NUM_ANO.search(par.proposicao or "")
    if m:
        return f"{par.tipo.sigla} nº {m.group(1)}-{m.group(2)}"
    m = _RE_SIGLA_TITULO.search(par.titulo or "")
    if m:
        return f"{m.group(1).upper()} nº {m.group(2)}-{m.group(3)}"
    return f"{par.tipo.sigla} s-n"


def _tema_parecer(titulo: str) -> str:
    """Tema do título de página-parecer ("PL 3.031/2025 — Rota…" → "Rota…").
    O _tema comum pegaria o lado ERRADO do travessão (a referência do PL)."""
    m = _RE_TEMA_POS_SIGLA.match((titulo or "").strip())
    if m:
        return _sanitize(m.group(1).strip(" -–—"))
    return _tema(titulo)


def _rel_curto(relator: str) -> str:
    """"Deputado Hildo Rocha (MDB/MA)" → "Hildo Rocha" (para o nome do arquivo)."""
    r = re.sub(r"^\s*Deputad[oa](?:\(a\))?\s+", "", (relator or "").strip())
    r = re.sub(r"\s*\([^)]*\)\s*$", "", r).strip()
    return "" if (not r or "[" in r) else r


def build_meta(sep: Optional[PaginaSeparada], overrides: Optional[Dict] = None) -> MetaDocumento:
    # `sep` não é usado (mantido na assinatura por compatibilidade); no fluxo de
    # parecer é passado None.
    overrides = overrides or {}
    hoje = _hoje()
    ano = int(overrides.get("ano") or hoje.year)
    data_it = overrides.get("data_fecho_it")
    if data_it is None:
        data_it = data_extenso(hoje.day, hoje.month, ano)
    meta = MetaDocumento(
        ano=ano,
        vocativo=overrides.get("vocativo", config.VOCATIVO_DEFAULT),
        consultor=overrides.get("consultor", config.CONSULTOR_NOME),
        consultor_cargo=overrides.get("consultor_cargo", config.CONSULTOR_CARGO),
        sisconle=overrides.get("sisconle", ""),
        deputado_nome=overrides.get("deputado_nome", ""),
        data_fecho_it=data_it,
        data_fecho_prop=overrides.get("data_fecho_prop", ""),
    )
    return meta


def converter(
    url: str,
    *,
    usar_ia: bool = True,
    gerar_it: bool = True,
    gerar_proposicao: bool = True,
    overrides: Optional[Dict] = None,
    out_it_dir: Optional[Path] = None,
    out_prop_dir: Optional[Path] = None,
    out_parecer_dir: Optional[Path] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> ResultadoConversao:
    log = progress or (lambda _m: None)

    log("Conectando ao Notion e baixando a página…")
    page_id, titulo, raw = fetch_page(url)
    log(f"Página: “{titulo}”. Processando blocos…")
    blocks = flatten_blocks(parse_blocks(raw))

    # Mentions internas sem norma mapeada (ex. julgados do TSE) ganham a URL
    # pública da própria página mencionada — a referência não se perde no docx.
    n_pub = resolver_mentions_publicas(blocks)
    if n_pub:
        log(f"Referências de mentions internas resolvidas para fonte pública: {n_pub}.")

    # O tipo de documento é decidido pela ESTRUTURA da página (auto-detecção):
    # layout de parecer de comissão gera o parecer; senão, IT/minuta.
    if detectar_layout(blocks) == "parecer":
        return _converter_parecer(
            page_id, titulo, blocks, overrides, out_parecer_dir,
            checkboxes_marcados=(gerar_it or gerar_proposicao), log=log,
        )

    if not gerar_it and not gerar_proposicao:
        raise RuntimeError(
            "Selecione ao menos um documento: Informação Técnica ou minuta de proposição."
        )

    sep = split_page(blocks, titulo)
    log(f"Tipo de proposição identificado: {sep.tipo.sigla} ({sep.tipo.nome_extenso}).")

    if usar_ia:
        # Rede de segurança: resíduos de citação que as regras determinísticas
        # não resolveram (raro) são polidos pontualmente pela IA, com validação
        # anti-perda (ver conle_conversor.residual).
        log("Verificando resíduos de citação…")
        resultado_polimento = polir_residuos_pagina(sep, log)
    else:
        resultado_polimento = []

    meta = build_meta(sep, overrides)

    log("Harmonizando a abertura da Informação Técnica…")
    abertura = gerar_abertura(
        sep.objeto, sep.introducao_texto, titulo, sep.tipo.nome_extenso, usar_ia=usar_ia
    )
    log("Abertura gerada por IA." if abertura.via_ia else "Abertura gerada por modelo-padrão (sem IA).")

    resultado = ResultadoConversao(
        page_id=page_id,
        titulo=titulo,
        tipo_sigla=sep.tipo.sigla,
        tipo_extenso=sep.tipo.nome_extenso,
        abertura_via_ia=abertura.via_ia,
    )
    if not abertura.via_ia and usar_ia:
        resultado.avisos.append("A abertura foi gerada por modelo-padrão (IA indisponível).")
    resultado.avisos.extend(resultado_polimento)

    tema = _tema(titulo)

    it_dir = Path(out_it_dir or config.OUTPUT_IT_DIR)
    prop_dir = Path(out_prop_dir or config.OUTPUT_PROPOSICAO_DIR)

    if gerar_it:
        it_dir.mkdir(parents=True, exist_ok=True)
        log("Montando o documento da Informação Técnica…")
        doc_it = build_it(sep, abertura, meta)
        rot = _solicitante_rotulo(meta.deputado_nome)
        prefixo = "Informação Técnica - " + (f"{rot} - " if rot else "")
        caminho = _caminho_unico(it_dir, _compor_nome(prefixo, tema))
        doc_it.save(str(caminho))
        resultado.caminhos.append(caminho)
        log(f"IT salva em: {caminho}")

    if gerar_proposicao:
        prop_dir.mkdir(parents=True, exist_ok=True)
        log("Montando a minuta de proposição…")
        doc_prop = build_proposicao(sep, meta)
        rot = _solicitante_rotulo(meta.deputado_nome)
        sufixo = f" ({rot})" if rot else ""
        nome = _compor_nome(f"Minuta de {sep.tipo.sigla} {meta.ano} - ", tema, sufixo)
        caminho = _caminho_unico(prop_dir, nome)
        doc_prop.save(str(caminho))
        resultado.caminhos.append(caminho)
        log(f"Minuta salva em: {caminho}")

    log("Concluído.")
    return resultado


def _converter_parecer(
    page_id: str,
    titulo: str,
    blocks: List[Block],
    overrides: Optional[Dict],
    out_parecer_dir: Optional[Path],
    *,
    checkboxes_marcados: bool,
    log: Callable[[str], None],
) -> ResultadoConversao:
    log("Layout de PARECER DE COMISSÃO detectado — gerando o parecer "
        "(as opções de IT/minuta/IA não se aplicam a este layout).")
    par = split_parecer(blocks, titulo)
    log(f"Comissão: {par.comissao_sigla or par.comissao or '—'}; "
        f"proposição: {par.proposicao or titulo}; tipo {par.tipo.sigla}.")

    meta = build_meta(None, overrides)
    resultado = ResultadoConversao(
        page_id=page_id,
        titulo=titulo,
        tipo_sigla=par.tipo.sigla,
        tipo_extenso=par.tipo.nome_extenso,
        abertura_via_ia=False,
    )
    if checkboxes_marcados:
        resultado.avisos.append(
            "Página com layout de parecer de comissão: as opções de IT/minuta foram ignoradas."
        )
    if not par.tem_substitutivo:
        resultado.avisos.append(
            "Não foi encontrado Substitutivo na página — o parecer foi gerado sem essa parte."
        )

    pasta = Path(out_parecer_dir or config.OUTPUT_PARECER_DIR)
    pasta.mkdir(parents=True, exist_ok=True)
    log("Montando o documento do parecer…")
    doc = build_parecer(par, meta)

    rel = _rel_curto(par.relator)
    sufixo = f" (Rel. {rel})" if rel else ""
    prefixo = f"Parecer {par.comissao_sigla or 'Comissão'} - {_ref_proposicao(par)} - "
    caminho = _caminho_unico(pasta, _compor_nome(prefixo, _tema_parecer(titulo), sufixo))
    doc.save(str(caminho))
    resultado.caminhos.append(caminho)
    log(f"Parecer salvo em: {caminho}")
    log("Concluído.")
    return resultado
