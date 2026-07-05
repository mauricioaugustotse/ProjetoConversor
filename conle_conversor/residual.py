# -*- coding: utf-8 -*-
"""Polimento residual (IA) das citações que as regras determinísticas não resolveram.

Rede de segurança OPCIONAL (gated pela flag "usar IA" da GUI, a mesma do
harmonizer): depois que richtext.preparar_rich aplicou a limpeza de colchetes,
a linkificação e a supressão do "(cf.)" redundante, este módulo procura
ARTEFATOS TIPOGRÁFICOS remanescentes (colchete colado em citação linkada,
parênteses vazios, pontuação órfã) e pede à OpenAI uma reescrita MÍNIMA do
parágrafo, restrita ao artefato. Validação anti-perda obrigatória: todos os
links e todos os números do parágrafo original precisam sobreviver — falhou,
o parágrafo original é mantido intacto. Em páginas cobertas pelas regras
determinísticas este módulo não é acionado (detectar_residuos devolve vazio).

O parecer de comissão fica fora do escopo (a GUI já avisa que as opções de IA
não se aplicam àquele layout).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

from . import config
from .notion_parser import Block, RichText
from .richtext import preparar_rich

# Blocos que podem ser polidos: prosa da IT/justificação. Quotes (transcrição
# de lei, linkificar=False) e tabelas ficam de fora de propósito.
_TIPOS_POLIVEIS = {
    "paragraph", "bulleted_list_item", "numbered_list_item", "callout",
    "heading_1", "heading_2", "heading_3",
}

# Marcador interno para "run linkado" na varredura do texto concatenado
# (caractere de area privada Unicode - nunca ocorre em texto real).
_MARCA_LINK = ""

_RE_CF_MORTO = re.compile(rf"\(\s*cf\.?\s*\)", re.I)
_RE_PARENTESE_VAZIO = re.compile(r"\(\s*\)")
# Pontuação órfã (espaço antes de , . ; :) ou duplicação anômala. FORA do
# padrão, por serem legítimos: reticências/pontilhados de supressão ("...",
# "Art. 29. ....") e abreviação seguida de vírgula ("cf., no TSE, …").
_RE_PONTUACAO_ORFA = re.compile(r"\s+[,.;:](?:\s|$)|,{2,}|;{2,}|,\.|[,;]\s+[,.;]")

SYSTEM_PROMPT = """Você revisa PONTUALMENTE um parágrafo de documento jurídico da \
Consultoria Legislativa da Câmara dos Deputados. O parágrafo contém pequenos artefatos \
tipográficos apontados pelo revisor (ex.: colchete solto colado numa citação, parênteses \
vazios, pontuação duplicada ou órfã). Reescreva o parágrafo APENAS o suficiente para \
remover os artefatos apontados. É PROIBIDO: alterar, acrescentar ou remover palavras, \
citações, números ou referências; mudar links (a notação [texto](url) deve ser preservada \
com o mesmo texto e a mesma url); resumir ou reordenar. Responda SOMENTE com o parágrafo \
corrigido, sem comentários."""


def detectar_residuos(rich_prep: List[RichText]) -> List[str]:
    """Artefatos tipográficos que sobraram APÓS preparar_rich. Lista vazia na
    imensa maioria dos parágrafos — as regras determinísticas cobrem os casos
    conhecidos. Função pura (sem rede)."""
    problemas: List[str] = []
    n = len(rich_prep or [])
    for i, r in enumerate(rich_prep or []):
        t = r.text or ""
        if not (r.href or "").startswith("http"):
            continue
        if t.startswith("[") or t.endswith("]"):
            problemas.append(f"colchete colado na citação linkada {t!r}")
        ant = rich_prep[i - 1].text if i > 0 else ""
        seg = rich_prep[i + 1].text if i + 1 < n else ""
        if (ant or "").endswith("["):
            problemas.append(f"colchete órfão antes da citação linkada {t!r}")
        if (seg or "").startswith("]"):
            problemas.append(f"colchete órfão depois da citação linkada {t!r}")
    texto = "".join(
        _MARCA_LINK if (r.href or "").startswith("http") else (r.text or "")
        for r in rich_prep or []
    )
    if _RE_CF_MORTO.search(texto):
        problemas.append("parêntese remissivo '(cf.)' vazio")
    if _RE_PARENTESE_VAZIO.search(texto):
        problemas.append("parênteses vazios")
    m = _RE_PONTUACAO_ORFA.search(texto.replace(_MARCA_LINK, "X"))
    if m:
        problemas.append(f"pontuação órfã/duplicada em {m.group(0)!r}")
    return problemas


def _serializar(rich_list: List[RichText]) -> str:
    """Parágrafo em texto com links na notação [texto](url)."""
    partes = []
    for r in rich_list or []:
        if (r.href or "").startswith("http"):
            partes.append(f"[{r.text}]({r.href})")
        else:
            partes.append(r.text or "")
    return "".join(partes)


_RE_LINK_MD = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _desserializar(texto: str, modelo: RichText) -> List[RichText]:
    """Texto com [texto](url) de volta a runs (formatação neutra do modelo)."""
    out: List[RichText] = []
    pos = 0
    for m in _RE_LINK_MD.finditer(texto):
        if m.start() > pos:
            out.append(replace(modelo, text=texto[pos:m.start()], href=None))
        out.append(replace(modelo, text=m.group(1), href=m.group(2)))
        pos = m.end()
    if pos < len(texto):
        out.append(replace(modelo, text=texto[pos:], href=None))
    return [r for r in out if r.text]


_RE_NUMERO = re.compile(r"\d[\d./º°‑\-]*")


def _validar_anti_perda(orig: List[RichText], novo: List[RichText]) -> bool:
    """Nada de conteúdo pode se perder na reescrita: mesmo multiconjunto de
    URLs, todos os números do original presentes, comprimento próximo."""
    urls_o = Counter((r.href or "").strip() for r in orig if (r.href or "").startswith("http"))
    urls_n = Counter((r.href or "").strip() for r in novo if (r.href or "").startswith("http"))
    if urls_o != urls_n:
        return False
    txt_o = "".join(r.text or "" for r in orig)
    txt_n = "".join(r.text or "" for r in novo)
    nums_o = Counter(_RE_NUMERO.findall(txt_o))
    nums_n = Counter(_RE_NUMERO.findall(txt_n))
    if nums_o - nums_n:  # número que sumiu
        return False
    if not (0.6 <= (len(txt_n) or 1) / (len(txt_o) or 1) <= 1.4):
        return False
    return True


def polir_bloco(
    rich_prep: List[RichText], problemas: List[str], client
) -> Tuple[List[RichText], bool]:
    """Reescrita mínima do parágrafo (já preparado) via OpenAI, restrita aos
    artefatos apontados. Retorna (rich_final, True) se a reescrita passou na
    validação anti-perda; senão (original intacto, False). Parágrafo com
    formatação própria (bold/itálico/sublinhado) não é polido — a
    reserialização a perderia."""
    if any(r.bold or r.italic or r.underline for r in rich_prep or []):
        return rich_prep, False
    user = (
        "Artefatos apontados pelo revisor:\n- " + "\n- ".join(problemas) +
        "\n\nParágrafo:\n" + _serializar(rich_prep)
    )
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    texto = (resp.choices[0].message.content or "").strip()
    if not texto:
        return rich_prep, False
    modelo = replace(rich_prep[0], href=None) if rich_prep else RichText("")
    novo = _desserializar(texto, modelo)
    if not _validar_anti_perda(rich_prep, novo):
        return rich_prep, False
    return novo, True


def polir_residuos_pagina(sep, log: Optional[Callable[[str], None]] = None) -> List[str]:
    """Varre a prosa da página separada (IT + justificação) atrás de resíduos
    e aplica o polimento IA bloco a bloco, substituindo b.rich in place quando
    a validação passa. Retorna avisos para o resultado da conversão. Sem chave
    da OpenAI ou com erro de rede, nada é alterado."""
    log = log or (lambda _m: None)
    avisos: List[str] = []
    client = None

    def _cliente():
        nonlocal client
        if client is None:
            api_key = config.load_openai_key()
            if not api_key:
                return None
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        return client

    # Só a prosa que LINKA no .docx (IT + justificação). Ementa/articulado da
    # minuta saem sem hyperlinks (texto oficial) e ficam fora do polimento.
    alvos: List[Block] = [
        b for b in (list(sep.it_blocks) + list(sep.justificativa_blocks))
        if b.type in _TIPOS_POLIVEIS and b.rich
    ]

    for b in alvos:
        prep = preparar_rich(list(b.rich))
        problemas = detectar_residuos(prep)
        if not problemas:
            continue
        trecho = "".join(r.text or "" for r in prep)[:60]
        cli = _cliente()
        if cli is None:
            avisos.append(
                f"Resíduo de citação mantido (IA sem chave): {'; '.join(problemas)} em “{trecho}…”."
            )
            continue
        try:
            novo, ok = polir_bloco(prep, problemas, cli)
        except Exception as exc:  # rede, cota, parsing…
            avisos.append(f"Resíduo de citação mantido (IA indisponível: {exc}) em “{trecho}…”.")
            continue
        if ok:
            b.rich[:] = novo
            log(f"IA poliu resíduo de citação em “{trecho}…”.")
            avisos.append(f"IA poliu resíduo de citação ({'; '.join(problemas)}).")
        else:
            avisos.append(
                f"Resíduo de citação mantido (reescrita reprovada na validação) em “{trecho}…”."
            )
    return avisos
