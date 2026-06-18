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
from .docx_builder import build_it, build_proposicao
from .harmonizer import gerar_abertura
from .meta import MetaDocumento
from .notion_api import fetch_page
from .notion_parser import parse_blocks, flatten_blocks
from .splitter import split_page, PaginaSeparada


_INVALIDOS = re.compile(r'[\\/:*?"<>|]+')
# Prefixo de sigla de proposição no início do título da página (ex.: "PL - ",
# "PLP – "): redundante porque o conversor já prefixa "IT - "/"PL - " no nome do
# arquivo. Removê-lo evita nomes como "PL - PL - …" e "IT - PL - …".
_SIGLA_PREFIXO = re.compile(
    r"^(PLP|PEC|PRC|PDL|PDC|PLV|MPV|PL)\s*[-–—:]\s*", re.IGNORECASE
)


def _sanitize(nome: str, limite: int = 150) -> str:
    nome = _INVALIDOS.sub("", nome)
    nome = re.sub(r"\s+", " ", nome).strip(" .")
    return nome[:limite].strip()


def _tema(titulo: str) -> str:
    base = re.split(r"[—–-]\s*minuta", titulo, flags=re.IGNORECASE)[0]
    base = base.split("—")[0].split("–")[0]
    base = _SIGLA_PREFIXO.sub("", base.strip(" -–—"))
    return _sanitize(base.strip(" -–—"))


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


def build_meta(sep: PaginaSeparada, overrides: Optional[Dict] = None) -> MetaDocumento:
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
    progress: Optional[Callable[[str], None]] = None,
) -> ResultadoConversao:
    log = progress or (lambda _m: None)

    log("Conectando ao Notion e baixando a página…")
    page_id, titulo, raw = fetch_page(url)
    log(f"Página: “{titulo}”. Processando blocos…")
    blocks = flatten_blocks(parse_blocks(raw))
    sep = split_page(blocks, titulo)
    log(f"Tipo de proposição identificado: {sep.tipo.sigla} ({sep.tipo.nome_extenso}).")

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

    tema = _tema(titulo)

    it_dir = Path(out_it_dir or config.OUTPUT_IT_DIR)
    prop_dir = Path(out_prop_dir or config.OUTPUT_PROPOSICAO_DIR)

    if gerar_it:
        it_dir.mkdir(parents=True, exist_ok=True)
        log("Montando o documento da Informação Técnica…")
        doc_it = build_it(sep, abertura, meta)
        caminho = _caminho_unico(it_dir, _sanitize(f"IT - {tema}") + ".docx")
        doc_it.save(str(caminho))
        resultado.caminhos.append(caminho)
        log(f"IT salva em: {caminho}")

    if gerar_proposicao:
        prop_dir.mkdir(parents=True, exist_ok=True)
        log("Montando a minuta de proposição…")
        doc_prop = build_proposicao(sep, meta)
        nome = _sanitize(f"{sep.tipo.sigla} - {tema}") + ".docx"
        caminho = _caminho_unico(prop_dir, nome)
        doc_prop.save(str(caminho))
        resultado.caminhos.append(caminho)
        log(f"Minuta salva em: {caminho}")

    log("Concluído.")
    return resultado
