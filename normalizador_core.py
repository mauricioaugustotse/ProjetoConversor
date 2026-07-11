# -*- coding: utf-8 -*-
"""Normalizador de páginas do Notion importadas de PDF (OCR ruidoso).

Retira ruídos (LaTeX de OCR, artigos colados, headings duplicados, folha de rosto
processual, certidões/assinaturas e imagens residuais), normaliza a hierarquia e
reescreve a página in-place com backup. Genérico, mas calibrado para o primeiro
alvo: as 30 subpáginas de estatutos partidários da página do Ofício (Consulta TSE
0601908-68.2022.6.00.0000).

Reusa: _doc_lib (dump/render/substituir_inplace/imagens), _lia_build
(md_doc_to_blocks), _lia_notion (_req), conle_gerador.llm (IA opcional).

Uso:
  py normalizador_core.py --descobrir <url|id>
  py normalizador_core.py --diagnosticar [--pagina ID | --todas]
  py normalizador_core.py --previa --pagina ID [--com-ia] [--manter-imagens] [--sem-recorte]
  py normalizador_core.py --aplicar --pagina ID [--sem-renomear] [--forcar]
  py normalizador_core.py --restaurar --pagina ID
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import _doc_lib as dl
import _lia_build as build
import _lia_notion as ln
from conle_conversor.notion_api import normalize_page_id, page_title

WORK = Path(__file__).resolve().parent / "_normalizador_work"
PAGINAS_JSON = WORK / "paginas.json"
MANIFEST_JSON = WORK / "manifest.json"
DIAG_JSON = WORK / "diagnostico.json"
DIAG_CSV = WORK / "diagnostico.csv"

MODELO_IA = "gpt-5.6-luna"

# ============================================================ partidos (30)
# ordem importa: nomes mais específicos primeiro (PCdoB antes de PCB; PSDB antes
# de PSD/PSB). Nomes de uma palavra só NÃO entram como "por extenso" (ambíguos:
# cidadania, missão, agir...) — para eles vale a sigla em parênteses/isolada.
PARTIDOS: List[Tuple[str, str]] = [
    ("PCdoB", r"Partido\s+Comunista\s+do\s+Brasil"),
    ("PCB", r"Partido\s+Comunista\s+Brasileiro"),
    ("PCO", r"Partido\s+da\s+Causa\s+Oper[áa]ria"),
    ("PSDB", r"Partido\s+da\s+Social\s+Democracia\s+Brasileira"),
    ("PSD", r"Partido\s+Social\s+Democr[áa]tico"),
    ("PSB", r"Partido\s+Socialista\s+Brasileiro"),
    ("PSTU", r"Partido\s+Socialista\s+dos\s+Trabalhadores\s+Unificado"),
    ("PSOL", r"Partido\s+Socialismo\s+e\s+Liberdade"),
    ("PRTB", r"Partido\s+Renovador\s+Trabalhista\s+Brasileiro"),
    ("PRD", r"Partido\s+Renova[çc][ãa]o\s+Democr[áa]tica"),
    ("PDT", r"Partido\s+Democr[áa]tico\s+Trabalhista"),
    ("PT", r"Partido\s+dos\s+Trabalhadores"),
    ("PV", r"Partido\s+Verde"),
    ("PL", r"Partido\s+Liberal"),
    ("PP", r"Progressistas|Partido\s+Progressista"),
    ("PODE", r"Podemos"),
    ("MDB", r"Movimento\s+Democr[áa]tico\s+Brasileiro"),
    ("NOVO", r"Partido\s+Novo"),
    ("REDE", r"Rede\s+Sustentabilidade"),
    ("UNIÃO", r"Uni[ãa]o\s+Brasil"),
    ("UP", r"Unidade\s+Popular"),
    ("DC", r"Democracia\s+Crist[ãa]"),
    ("REPUBLICANOS", ""),
    ("CIDADANIA", ""),
    ("SOLIDARIEDADE", ""),
    ("AVANTE", ""),
    ("AGIR", ""),
    ("MOBILIZA", ""),
    ("MISSÃO", ""),
    ("DEMOCRATA", ""),
]
SIGLAS = [s for s, _ in PARTIDOS]
# variantes que aparecem nos textos -> sigla canônica
ALIAS = {"PODEMOS": "PODE", "UNIAO": "UNIÃO", "UNIÃO BRASIL": "UNIÃO",
         "UNIAO BRASIL": "UNIÃO", "MISSAO": "MISSÃO", "SD": "SOLIDARIEDADE"}
# siglas femininas p/ título ("Estatuto da REDE")
_FEM = {"REDE", "UP", "UNIÃO", "MISSÃO", "DC"}

# ============================================================ regex do pipeline
_TBL_RE = re.compile(r"^\s*(\||\[TABELA\])")
_QUOTE_RE = re.compile(r"^\s*>")
_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

_CHAR_MAP = {
    "№": "nº",  # №
    " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "",    # soft hyphen
    "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'",
    "﻿": "",
    "ㅇ": "º",   # OCR usa o jamo coreano ㅇ como ordinal ("Art. 2ㅇ")
    "－": "-", "，": ",", "：": ":", "｜": "|", "／": "/",  # formas fullwidth
    "．": ".", "（": " (", "）": ") ", "＂": '"', "；": ";", "！": "!", "？": "?",
    # homoglifos gregos que o OCR usa no lugar de latinos ("Τίτυιο ι" = TÍTULO I)
    "Τ": "T", "τ": "t", "ί": "í", "ι": "i", "υ": "u", "ο": "o",
    "Α": "A", "Ε": "E", "Ι": "I", "Ο": "O", "α": "a", "ε": "e",
}

_SPAN_RE = re.compile(r"\$\s*`?(.*?)`?\s*\$", re.DOTALL)
_EQ_RE = re.compile(r"^(\s*)\[EQUATION\]\s*(.*)$")

_ROTULO_SOLO = re.compile(
    r"^(#{0,3}\s*)(Artigo|Art\.?|ARTIGO|ART\.?|Par[áa]grafo|PAR[ÁA]GRAFO|§|"
    r"Cap[íi]tulo|CAP[ÍI]TULO|T[íi]tulo|T[ÍI]TULO|Se[çc][ãa]o|SE[ÇC][ÃA]O)\s*$")
_NUM_SOLO = re.compile(
    r"^\s*(?:\d{1,3}\s*[ºo°]?(?:\s*-?\s*[A-Z])?|[IVXLCDM]+)\s*[-–—.:]?\s*$")
_ART_NUM_SOLO = re.compile(
    r"^(#{0,3}\s*)((?:Artigo|Art\.?|ARTIGO|ART\.?|§|Par[áa]grafo)\s*\d{1,3}(?:\s*[ºo°])?"
    r"(?:\s*-?\s*[A-Z])?)\s*[-–—]?\s*$")

# split de dispositivos colados na MESMA linha: exige fim de sentença antes
_SPLIT_DISP = re.compile(
    r"(?<=[.;:!?)])\s+(?="
    r"(?:Artigo|Art\.|ARTIGO|ART\.)\s*\d{1,3}\s*(?:[ºo°]\s*)?(?:[-–—.]|[A-ZÀ-Ü(])"
    r"|§\s*\d{1,2}"
    r"|Par[áa]grafo\s+(?:[ÚU]nico|Primeiro|Segundo|Terceiro|Quarto|Quinto|Sexto|"
    r"S[ée]timo|Oitavo|Nono|D[ée]cimo)"
    r")")

_H_ROTULO = re.compile(
    r"^(#{1,3})\s*((?:Cap[íi]tulo|CAP[ÍI]TULO|T[íi]tulo|T[ÍI]TULO|Se[çc][ãa]o|"
    r"SE[ÇC][ÃA]O|Subse[çc][ãa]o|SUBSE[ÇC][ÃA]O|Parte|PARTE)\s+[IVXLCDM\d]+[ºo°]?)"
    r"\s*[-–—:.]?\s*$")

_PROC_RE = re.compile(
    r"(?i)(REGISTRO DE PARTIDO POL[ÍI]TICO|Relator[a]?\s*:|Requerente\s*:|"
    r"Advogad[oa]s?\s*:|RES\.\s*-?\s*TSE|sess[ãa]o(?:\s+\w+){0,2}\s+realizada|DEFERIMENTO|"
    r"anota[çc][ãa]o das (?:altera[çc][õo]es|modifica[çc][õo]es)|"
    r"TRIBUNAL SUPERIOR ELEITORAL|altera[çc][õo]es estatut[áa]rias)")

_FIM_RE = re.compile(
    r"(?i)(certifi|certid[ãa]o|\bOAB\b|assinado\s+(?:eletronicamente|digitalmente)|"
    r"averba[çc]|protocol|cart[óo]rio|registro civil|of[íi]cio de registro|"
    r"p[áa]gina \d+ de \d+|tabeli|autentic|firma reconhecida|selo digital|"
    r"documento assinado|assinatura|carimbo|SECRET[ÁA]RI[OA][ -]GERAL|"
    r"registro de t[íi]tulos|advogad[oa]|digitally signed|assinado digitalmente|"
    r"presidente da executiva|documento foi gerado|t[íi]tulo:\s*\d)")
_ALLCAPS_RE = re.compile(r"^[A-ZÀ-Ü0-9 .ºª\-–—/§'\",()nº]{6,90}$")
_LOCALDATA_RE = re.compile(
    r"^[A-ZÀ-Ü][\wÀ-ü. ]{2,40},\s*(?:aos?\s+)?\d{1,2}º?\s+de\s+[a-zç]+\s+de\s+\d{4}")

_IMG_RE = re.compile(r"^(\s*)\[IMAGEM n=(\d+)[^\]]*\]\s*(.*)$")
_ART_COUNT_RE = re.compile(r"(?im)(?:^|[\s>(])(?:artigo|art\.?)\s*\d{1,3}(?!\d)")
_DISP_RE = re.compile(r"^\s*(?:Artigo|Art\.?|ARTIGO|ART\.?|§|Par[áa]grafo)\b")

_MESES = {"janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
          "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
          "outubro": 10, "novembro": 11, "dezembro": 12}


def _e_estrutural(l: str) -> bool:
    """Linha que as passadas estruturais devem PULAR (tabela/quote)."""
    return bool(_TBL_RE.match(l) or _QUOTE_RE.match(l))


# ============================================================ passadas p01..p11
def p01_normalizar_chars(md: str) -> str:
    for k, v in _CHAR_MAP.items():
        md = md.replace(k, v)
    # colapsa espaços múltiplos internos (preserva indentação inicial da linha)
    md = re.sub(r"(?<=\S)[ \t]{2,}", " ", md)
    # aspas espaçadas de OCR: alínea " b " -> alínea "b". O lookbehind impede o
    # colapso do conector entre duas citações ('"f" e "g"'), onde o candidato a
    # abre-aspas é na verdade o fecha-aspas precedido de letra
    md = re.sub(r'(?<![\w"])"\s+([^"\n]{1,3}?)\s+"', r'"\1"', md)
    # fração com espaços: 2 / 3 -> 2/3
    md = re.sub(r"(?<=\d)\s+/\s+(?=\d)", "/", md)
    # percentual partido: "5 0%" -> "50%"
    md = re.sub(r"(?<=\d)\s+(\d)\s*%", r"\1%", md)
    # espaço após vírgula/ponto colados em letra ("dias,contados" / "vigor.Artigo")
    md = re.sub(r",(?=[A-Za-zÀ-ü])", ", ", md)
    md = re.sub(r"(?<=[a-zà-ü])\.(?=[A-ZÀ-Ü][a-zà-ü])", ". ", md)
    md = re.sub(r"(?<=\S) (?=[)\].,;:])", "", md)   # espaço antes de pontuação de fechamento
    md = re.sub(r"(?<=\S)[ \t]{2,}", " ", md)
    return md


def _latex_seguro(s: str) -> str:
    """Substituições LaTeX->texto seguras para aplicar em linha inteira
    (só padrões inequívocos; não remove chaves/backslashes soltos)."""
    for _ in range(10):
        prev = s
        s = re.sub(r"\{\s*\}", "", s)
        s = re.sub(r"\\(?:mathbf|boldsymbol|boldmath|mathrm|mathit|operatorname|"
                   r"text|textbf|textit|bf|it|rm|underline)\s*\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\(?:begin|end)\s*\{\w+\}", "", s)
        s = re.sub(r"\^\{?\s*\\?circ\s*\}?", "º", s)
        s = re.sub(r"\^\{?\s*\\?prime\s*\}?", "'", s)
        s = re.sub(r"\^\{?\s*o\s*\}?", "º", s)
        s = re.sub(r"\^\{?\s*a\s*\}?", "ª", s)
        # ordinal mal-OCRizado em sobrescrito após dígito/N: ^{0} ^{2} ^{9} ^{®} ^{e} ^{O} ^{P}
        s = re.sub(r"(?<=[\dNn])\s*\^\{?\s*[®0O29eEºªP]\s*\}?", "º", s)
        s = re.sub(r"\\S\b", "§", s)
        s = re.sub(r"\\+%", "%", s)
        s = re.sub(r"\\+§", "§", s)
        s = re.sub(r"\bR\s*\\+\$?\s*(?=\d)", "R$ ", s)   # "R \$ 20" / "R \20" -> "R$ 20"
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", s)
        s = re.sub(r"\\[,;!:|]|\\ ", " ", s)
        if s == prev:
            break
    # "\1º -" no início (o OCR comeu o §)
    s = re.sub(r"^\s*\\(\d{1,2}[ºo°])", r"§ \1", s)
    return s


def _latex_inline(expr: str) -> str:
    """Conversão agressiva do MIOLO de um span $...$/[EQUATION] (era só LaTeX)."""
    s = _latex_seguro(expr)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = re.sub(r"\s*/\s*", "/", s)          # 1 / 3 -> 1/3
    s = re.sub(r"(\d)\s+([ºª])", r"\1\2", s)  # 9 º -> 9º
    return re.sub(r"\s+", " ", s).strip()


def p02_latex_para_texto(md: str, relatorio: dict) -> str:
    md = _SPAN_RE.sub(lambda m: _latex_inline(m.group(1)), md)
    linhas = md.split("\n")
    out = []
    for l in linhas:
        m = _EQ_RE.match(l)
        if m:
            l = f"{m.group(1)}{_latex_inline(m.group(2))}"
        elif "\\" in l or "^{" in l:
            # equação inline da API vem como plain_text SEM delimitadores $:
            # aplica as conversões seguras na linha inteira
            l = _latex_seguro(l)
            # fração nua "1 / 3 (um terço)" só quando isolada entre dígitos
            l = re.sub(r"(?<=\d)\s*/\s*(?=\d)", "/", l)
        l = re.sub(r"(?<=\d)\s+(?=%)", "", l)  # "50 %" -> "50%"
        # arruma ordinais fora de span ("n º", "5 º")
        l = re.sub(r"\b(n)\s+([ºª])", r"\1\2", l)
        l = re.sub(r"(\d)\s+([ºª])(?![\wºª])", r"\1\2", l)
        if re.search(r"\\|\^\{|(?<![Rr])\$", l) and not _TBL_RE.match(l):
            relatorio.setdefault("suspeitas", []).append(l.strip()[:160])
        out.append(l)
    return "\n".join(out)


_LIXO_PJE_RE = re.compile(
    r"(?i)(Assinado eletronicamente por\s*:|"
    r"Num\.?\s*\d{6,}\s*[-–—]\s*P[áa]g\.?\s*\d+|"
    r"https?://\S*pje\S*|listView\.seam|"
    r"N[úu]mero do documento\s*:\s*\d{10,}|"
    r"^\s*\d{20,}\s*$|"
    # carimbo de cartório mal-OCRizado ("1º Officio de Brastia-DF Nº de Protocalo e Registro")
    r"\bOff?[íi]?ci?o de [BH]ra\w+|Protoc[ao]lo e Registro|"
    r"^\s*#{0,3}\s*N[ºª°]?\.?\s*de Protoco?lo\b.*$|"
    r"^\s*#{0,3}\s*de Protocolo\s*$|"
    r"^\s*Registro (?:Civil )?de Pessoas Jur[íi]d\w*\s*$|"
    r"^\s*Pessoas Jur[íi]dicas\s*$|"
    r"^\s*\d{5,}\s*$)")


def p015_remover_lixo_pje(md: str, relatorio: dict) -> str:
    """Remove linhas de carimbo do PJe/cartório (assinatura eletrônica, paginação,
    URLs de conferência, protocolo) que o OCR espalha pelo MEIO do documento.
    Roda APÓS o p02 (o carimbo costuma vir embrulhado em LaTeX). Linhas que
    começam com dispositivo/inciso/alínea nunca são removidas (o carimbo pode
    estar incrustado num texto legítimo)."""
    out, removidas = [], 0
    protegida = re.compile(r"^\s*(?:[IVXLC]+\s*[-–—.]|[a-z]\s*\))")
    for l in md.split("\n"):
        if (not _e_estrutural(l) and _LIXO_PJE_RE.search(l)
                and not _DISP_RE.match(l.strip()) and not protegida.match(l)):
            removidas += 1
            continue
        out.append(l)
    if removidas:
        relatorio["lixo_pje_linhas"] = removidas
    return "\n".join(out)


_RODAPE_RE = re.compile(
    r"(?i)(te[lt]l?\s*[/f]?\s*fax|tel\.?\s*[:.]|fax\s*[:.]|www\.|\.org\.?\s*\.?br|"
    r"\.com\.?\s*\.?br|sede (?:operacional|nacional|administrativa)\s*[:.]|"
    r"cep[\s:.-]*\d|@\w+\.)")


def p016_rodapes_repetidos(md: str, relatorio: dict) -> str:
    """Rodapés/cabeçalhos de página repetidos pelo documento (endereço, Tel/Fax e
    variantes de OCR, site): toda linha cujo conteúdo normalizado se repete 2+ vezes
    E casa padrão de rodapé é removida em TODAS as ocorrências (mesmo em citação)."""
    linhas = md.split("\n")

    def _norm(l: str) -> str:
        return re.sub(r"\s+", " ", l.strip().lstrip(">").strip().lower())

    freq: Dict[str, int] = {}
    for l in linhas:
        n = _norm(l)
        if n:
            freq[n] = freq.get(n, 0) + 1
    out, removidas = [], 0
    for l in linhas:
        n = _norm(l)
        if n and freq.get(n, 0) >= 2 and _RODAPE_RE.search(n) and not _TBL_RE.match(l):
            removidas += 1
            continue
        out.append(l)
    if removidas:
        relatorio["rodapes_removidos"] = removidas
    return "\n".join(out)


def p046_numeros_de_pagina(md: str, relatorio: dict) -> str:
    """Após as fusões: linha composta só de 1-4 dígitos (sem ordinal) é número de
    página/carimbo — remove e REUNIFICA o parágrafo que ela partiu ao meio."""
    linhas = md.split("\n")
    out: List[str] = []
    removidos = reunificados = 0
    i = 0
    while i < len(linhas):
        s = linhas[i].strip()
        if re.fullmatch(r"\d{1,4}", s) and not _e_estrutural(linhas[i]):
            removidos += 1
            # reunifica: anterior termina sem pontuação final + próxima começa
            # minúscula (e não é alínea/inciso) -> mesmo parágrafo partido
            j = i + 1
            while j < len(linhas) and not linhas[j].strip():
                j += 1
            prox = linhas[j].strip() if j < len(linhas) else ""
            ant = out[-1] if out and out[-1].strip() else ""
            if (ant and prox and re.search(r"[a-zà-ü,]$", ant.strip())
                    and re.match(r"^[a-zà-ü]", prox)
                    and not re.match(r"^[a-z]\s*\)", prox)
                    and not _e_estrutural(linhas[j])
                    and not _HEAD_RE.match(ant.strip())):
                out[-1] = ant.rstrip() + " " + prox
                reunificados += 1
                i = j + 1
                continue
            i += 1
            continue
        out.append(linhas[i])
        i += 1
    if removidos:
        relatorio["numeros_pagina_removidos"] = removidos
        relatorio["paragrafos_reunificados"] = reunificados
    return "\n".join(out)


def p03_deshifenizar(md: str) -> str:
    # junta apenas quando a continuação começa com minúscula (quebra de linha de OCR)
    return re.sub(r"(\w)-\n([a-zà-ü])", r"\1\2", md)


def p04_fundir_blocos_partidos(md: str) -> str:
    linhas = md.split("\n")
    out: List[str] = []
    i = 0
    while i < len(linhas):
        l = linhas[i]
        if _e_estrutural(l):
            out.append(l)
            i += 1
            continue
        # próxima linha não-vazia
        j = i + 1
        while j < len(linhas) and not linhas[j].strip():
            j += 1
        prox = linhas[j] if j < len(linhas) else ""
        m_rot = _ROTULO_SOLO.match(l.strip())
        if m_rot and prox and not _e_estrutural(prox) and (
                _NUM_SOLO.match(prox.strip())
                or re.match(r"^\s*\d{1,3}\s*[ºo°]?\s*[-–—]\s+\S", prox)):
            linhas[j] = f"{l.strip()} {prox.strip()}"
            i = j
            continue
        m_art = _ART_NUM_SOLO.match(l.strip())
        if m_art and prox and not _e_estrutural(prox) and \
                re.match(r"^\s*(?:[-–—(]|[a-zà-ü])", prox):
            pref = m_art.group(1) or ""
            corpo = m_art.group(2)
            sep = "" if re.match(r"^\s*[-–—]", prox) else " -"
            linhas[j] = f"{pref}{corpo}{sep} {prox.strip()}"
            i = j
            continue
        out.append(l)
        i += 1
    return "\n".join(out)


_FRAG_RE = re.compile(r"^\s*[\d.,/%ºª°()\-– ]{1,15}\s*$")
_ORDINAL_ORFA = re.compile(r"^\s*[ºª°]\s*$")


def p045_fundir_fragmentos(md: str, relatorio: dict) -> str:
    """Reintegra fragmentos que o OCR deixou em bloco próprio no MEIO da frase:
    '...de no mínimo' / '20%' / '(vinte por cento)...'  e  '...da Lei n.' / 'º' /
    '9096/95, os demais...'."""
    linhas = md.split("\n")
    out: List[str] = []
    fundidos = 0
    i = 0
    while i < len(linhas):
        l = linhas[i]
        s = l.strip()
        anterior = out[-1] if out else ""
        ok_anterior = (anterior.strip() and not _e_estrutural(anterior)
                       and not _HEAD_RE.match(anterior.strip()))
        if s and ok_anterior and not _e_estrutural(l):
            # próxima linha não-vazia
            j = i + 1
            while j < len(linhas) and not linhas[j].strip():
                j += 1
            prox = linhas[j].strip() if j < len(linhas) else ""
            if _ORDINAL_ORFA.match(s):
                # 'º' órfão: cola SEM espaço na anterior ("Lei n." + "º" -> "Lei n.º");
                # se a próxima começa com dígito, é o número da norma: cola também
                out[-1] = anterior.rstrip() + s
                fundidos += 1
                if prox and re.match(r"^\d", prox) and not _e_estrutural(linhas[j]):
                    out[-1] += " " + prox
                    i = j + 1
                    continue
                i += 1
                continue
            if (_FRAG_RE.match(s) and not _NUM_SOLO.match(s)
                    and re.search(r"[a-zà-ü,]$", anterior.strip())):
                # fragmento numérico ("20%") no meio da frase: a anterior termina
                # sem pontuação final -> reintegra; idem a continuação minúscula/(
                out[-1] = anterior.rstrip() + " " + s
                fundidos += 1
                if prox and re.match(r"^[a-zà-ü(]", prox) and not _e_estrutural(linhas[j]):
                    out[-1] += " " + prox
                    i = j + 1
                    continue
                i += 1
                continue
        out.append(l)
        i += 1
    if fundidos:
        relatorio["fragmentos_fundidos"] = fundidos
    return "\n".join(out)


def p05_separar_artigos_colados(md: str) -> str:
    out = []
    for l in md.split("\n"):
        if _e_estrutural(l) or _HEAD_RE.match(l):
            out.append(l)
            continue
        # "Artigo 12-O militante..." / "Artigo 65-Na Casa..." (traço colado do OCR)
        # -> "Artigo 12 - O militante". Dispositivos com sufixo real ("Art. 37-A -
        # ...", "art. 9º-B, § 3º") têm UMA letra + separador depois e não casam.
        l = re.sub(r"^((?:Artigo|Art\.?|ARTIGO|ART\.?)\s*\d{1,3})-([OAE])\s+(?=[\wÀ-ü])",
                   r"\1 - \2 ", l)
        l = re.sub(r"^((?:Artigo|Art\.?|ARTIGO|ART\.?)\s*\d{1,3})-(?=[A-ZÀ-Ü][a-zà-ü])",
                   r"\1 - ", l)
        out.append(_SPLIT_DISP.sub("\n", l))
    return "\n".join(out)


def p055_ordinais_dispositivos(md: str) -> str:
    """Estilo dos dispositivos: 1º-9º sempre com ordinal; de 10 em diante sem
    ordinal (o separador nativo do documento é preservado). Junta dígitos
    partidos ('Art. 7 0º' -> 'Art. 70') e uniformiza enumerações ('I- ' -> 'I - ')."""
    out = []
    for l in md.split("\n"):
        if _e_estrutural(l):
            out.append(l)
            continue
        # dígitos partidos no rótulo: "Art. 7 0º -" -> "Art. 70 -"
        l = re.sub(r"^((?:Artigo|Art\.?|ARTIGO|ART\.?|§|Par[áa]grafo)\s*)(\d)\s+(\d)\s*[ºo°]?(?=\s*[-–—.:])",
                   r"\1\2\3", l)
        # ordinal indevido em >=10, em QUALQUER posição (referências internas incluídas)
        l = re.sub(r"\b((?:[Aa]rt\.?|[Aa]rtigo|ART\.?|ARTIGO|§|[Pp]ar[áa]grafo)\s*)(\d{2,})\s*[ºo°]\b",
                   r"\1\2", l)
        # ordinal faltando em 1-9, só no RÓTULO (início de linha, com separador)
        l = re.sub(r"^((?:Artigo|Art\.?|ARTIGO|ART\.?)\s*)([1-9])(\s*[-–—.:])",
                   r"\1\2º\3", l)
        l = re.sub(r"^(§\s*)([1-9])(\s*[-–—.:])", r"\1\2º\3", l)
        # enumerações com espaçamento uniforme: "I- texto" -> "I - texto"; "a)texto" -> "a) texto"
        l = re.sub(r"^([IVXLCDM]+)\s*[-–—]\s*(?=\S)", r"\1 - ", l)
        l = re.sub(r"^([a-z])\s*\)\s*(?=\S)", r"\1) ", l)
        out.append(l)
    return "\n".join(out)


def p06_fundir_headings_duplicados(md: str) -> str:
    linhas = md.split("\n")
    out: List[str] = []
    i = 0
    while i < len(linhas):
        l = linhas[i]
        m = _H_ROTULO.match(l.strip())
        if m:
            j = i + 1
            while j < len(linhas) and not linhas[j].strip():
                j += 1
            if j < len(linhas) and not _e_estrutural(linhas[j]):
                prox = linhas[j].strip()
                mh = _HEAD_RE.match(prox)
                nome = mh.group(2).strip() if mh else prox
                if (nome and len(nome) <= 90 and not _DISP_RE.match(nome)
                        and not _H_ROTULO.match(prox)):
                    nome = re.sub(r"[:;.,]+$", "", nome).strip()
                    out.append(f"{m.group(1)} {m.group(2)} — {nome}")
                    i = j + 1
                    continue
        out.append(l)
        i += 1
    return "\n".join(out)


def p07_pontuacao_headings(md: str) -> str:
    out = []
    for l in md.split("\n"):
        m = _HEAD_RE.match(l)
        if m:
            txt = re.sub(r"[:;.,]+$", "", m.group(2)).strip()
            out.append(f"{m.group(1)} {txt}")
        else:
            out.append(l)
    return "\n".join(out)


def _acha_inicio_estatuto(linhas: List[str]) -> int:
    siglas_alt = "|".join(re.escape(s) for s in SIGLAS + list(ALIAS))
    par_re = re.compile(r"\((?:%s)\)" % siglas_alt)

    def _estende_para_tras(i: int) -> int:
        # headings contíguos acima da âncora fazem parte do título do documento
        # (ex.: "# UNIDADE POPULAR ESTATUTO" logo antes de "## Capítulo I")
        j = i
        while j > 0:
            k = j - 1
            while k >= 0 and not linhas[k].strip():
                k -= 1
            if k >= 0 and _HEAD_RE.match(linhas[k].strip()) \
                    and not _PROC_RE.search(linhas[k]):
                j = k
                continue
            break
        return j

    achado = -1
    # 1ª passada: âncoras fortes (título do estatuto / preâmbulo)
    for i, l in enumerate(linhas):
        s = l.strip()
        m = _HEAD_RE.match(s)
        txt = m.group(2) if m else s
        if _PROC_RE.search(txt):
            continue
        if re.match(r"(?i)^estatutos?\b", txt):
            achado = i
            break
        if m and re.search(r"(?i)\bestatutos?\b", txt):
            achado = i
            break
        if m and re.match(r"(?i)^pre[âa]mbulo\b", txt):
            achado = i
            break
        # títulos de outros tipos de documento (heading): regimentos, resoluções,
        # relatórios, pareceres, manuais... — o normalizador é genérico
        if m and re.match(r"(?i)^(regimento|regulamento|resolu[çc][ãa]o|portaria|"
                          r"c[óo]digo\s|lei\s+n|relat[óo]rio|parecer|nota\s+t[ée]cnica|"
                          r"manual\b|cartilha|edital|contrato|conv[êe]nio|"
                          r"sum[áa]rio\s+executivo|introdu[çc][ãa]o\b|apresenta[çc][ãa]o\b)",
                          txt):
            achado = i
            break
        if m and par_re.search(txt) and "REQUERENTE" not in txt.upper():
            achado = i
            break
        if m and re.match(r"(?i)^partido\s+\S", txt):
            achado = i
            break
    if achado < 0:
        # 2ª passada: estatutos sem título próprio — começam direto em "TÍTULO I"
        # / "CAPÍTULO I" (tolerante a OCR: "Títuio i", "CAPÍTULOI") ou no "Art. 1º"
        for i, l in enumerate(linhas):
            s = l.strip()
            m = _HEAD_RE.match(s)
            txt = m.group(2) if m else s
            if re.match(r"(?i)^(t[íi]tu[a-zà-ü]{0,2}|cap[íi]tulo)\s*(i|1|primeiro)\b", txt):
                achado = i
                break
            if re.match(r"(?:Artigo|Art\.?|ARTIGO|ART\.?)\s*1\s*[ºo°]?\s*[-–—.]", txt):
                achado = i
                break
    return _estende_para_tras(achado) if achado > 0 else achado


def _linha_cortavel_fim(l: str) -> bool:
    s = l.strip()
    if not s or len(s) < 4:
        return True
    if _IMG_RE.match(l):
        return True
    if _FIM_RE.search(s):
        return True
    if _LOCALDATA_RE.match(s):
        return True
    if _ALLCAPS_RE.match(s) and re.search(r"[A-ZÀ-Ü]{2}", s):
        return True
    return False


def p08_recortar_escopo(md: str, relatorio: dict) -> Tuple[str, List[str], List[str]]:
    linhas = md.split("\n")
    corte_ini: List[str] = []
    corte_fim: List[str] = []

    # ---- início
    idx = _acha_inicio_estatuto(linhas)
    if idx > 0:
        prefixo = linhas[:idx]
        hits = sum(1 for l in prefixo for _ in _PROC_RE.finditer(l))
        n_disp = sum(1 for l in prefixo if _DISP_RE.match(l.strip()))
        if n_disp >= 3:
            # o "prefixo" contém texto normativo em sequência: âncora achada tarde
            # demais — cortar destruiria artigos. Não corta nada.
            relatorio["recorte_inicio"] = (
                f"prefixo contém {n_disp} linhas-dispositivo — corte de início "
                "SUSPENSO (revisar manualmente)")
        elif hits >= 2:
            # metadados úteis antes de descartar
            ptxt = "\n".join(prefixo)
            m = re.search(r"(?i)sess[ãa]o(?:\s+\w+){0,2}\s+realizada\s+em\s+"
                          r"(\d{1,2})[ºo°]?\s+de\s+(\w+)\s+de\s+(\d{4})", ptxt)
            if m and m.group(2).lower() in _MESES:
                relatorio["data_aprovacao"] = f"{int(m.group(1))}.{_MESES[m.group(2).lower()]}.{m.group(3)}"
            m = re.search(r"(?i)REGISTRO DE PARTIDO POL[ÍI]TICO\s*(?:nº|n\.|№)?\s*([\d.\-/]{5,30})", ptxt)
            if m:
                relatorio["rpp"] = m.group(1).strip(" .")
            corte_ini = prefixo
            linhas = linhas[idx:]
        else:
            relatorio["recorte_inicio"] = f"prefixo com só {hits} marcador(es) processual(is) — mantido"
    elif idx < 0:
        relatorio["recorte_inicio"] = "início do estatuto não identificado — nada cortado"

    # ---- fim
    ult = -1
    for i, l in enumerate(linhas):
        if _e_estrutural(l):
            continue
        if _DISP_RE.match(l.strip()) or re.search(r"(?i)entra(?:r[áa])? em vigor", l):
            ult = i
    if ult >= 0:
        # estende a zona normativa: alíneas/incisos/parágrafos do último artigo
        k = ult + 1
        while k < len(linhas):
            s = linhas[k].strip()
            if not s:
                k += 1
                continue
            if (_DISP_RE.match(s)
                    or re.match(r"^(?:[a-z]\s*\)|[IVXLCDM]+\s*[-–—.:])", s)):
                ult = k
                k += 1
                continue
            break
        sufixo = [l for l in linhas[ult + 1:] if l.strip()]
        tem_tabela = any(_TBL_RE.match(l) for l in linhas[ult + 1:])
        # tabela no sufixo pode ser anexo legítimo (bloqueia) OU a caixinha de
        # assinatura digital do PDF (corta) — decide pela presença de marcadores
        # inequívocos de fecho (OAB, advogado, "Digitally signed"...)
        e_fecho = any(_FIM_RE.search(l) for l in sufixo)
        if sufixo and len(sufixo) <= 60 and (not tem_tabela or e_fecho):
            # após o último dispositivo só restam assinaturas/certidões/carimbos:
            # corta o fecho inteiro (registrado em cortes.md)
            corte_fim = sufixo
            linhas = linhas[:ult + 1]
        elif sufixo:
            # sufixo atípico (muito grande ou com tabela): heurística conservadora,
            # de trás para frente, parando na primeira linha não reconhecida
            k = len(linhas) - 1
            while k > ult and _linha_cortavel_fim(linhas[k]):
                k -= 1
            if k + 1 < len(linhas):
                corte_fim = [l for l in linhas[k + 1:] if l.strip()]
                linhas = linhas[:k + 1]
            relatorio["recorte_fim"] = (
                f"sufixo atípico ({len(sufixo)} linhas/tabela) — corte parcial conservador")
    else:
        # documento NÃO articulado (relatório, parecer, ofício...): sem "último
        # dispositivo" como fronteira, usa só a heurística conservadora — corta o
        # bloco final contíguo de linhas inequívocas de fecho/assinatura
        k = len(linhas) - 1
        while k >= 0 and _linha_cortavel_fim(linhas[k]):
            k -= 1
        if k + 1 < len(linhas):
            corte_fim = [l for l in linhas[k + 1:] if l.strip()]
            linhas = linhas[:k + 1]
        if not corte_fim:
            relatorio["recorte_fim"] = "documento não articulado — só fecho inequívoco seria cortado (nada encontrado)"

    relatorio["corte_inicio_linhas"] = len([l for l in corte_ini if l.strip()])
    relatorio["corte_fim_linhas"] = len(corte_fim)
    return "\n".join(linhas), corte_ini, corte_fim


def p09_remover_imagens(md: str, relatorio: dict, *, manter: bool = False) -> str:
    out = []
    removidas = 0
    for l in md.split("\n"):
        m = _IMG_RE.match(l)
        if not m:
            out.append(l)
            continue
        if manter:
            leg = m.group(3).strip() or "imagem do documento original"
            out.append(f"> 🖼️ Figura {m.group(2)} — {leg}")
        else:
            removidas += 1
    relatorio["imagens_removidas"] = removidas
    return "\n".join(out)


def p10_hierarquia(md: str, relatorio: dict) -> str:
    linhas = md.split("\n")
    tem_titulo_agrupador = any(
        re.match(r"^#{1,3}\s*(T[ÍI]TULO|PARTE)\b", l.strip()) for l in linhas
        if not _e_estrutural(l))

    # funde preâmbulo "# Nome do Partido (SIGLA)" + "## ESTATUTO" num H1 único
    idx_heads = [i for i, l in enumerate(linhas) if _HEAD_RE.match(l.strip())]
    if len(idx_heads) >= 2:
        h1 = _HEAD_RE.match(linhas[idx_heads[0]].strip())
        h2 = _HEAD_RE.match(linhas[idx_heads[1]].strip())
        t1, t2 = h1.group(2).strip(), h2.group(2).strip()
        soh_estatuto = re.compile(r"(?i)^estatuto$")
        estrutural = re.compile(r"(?i)^(t[íi]tu[a-zà-ü]{0,2}|cap[íi]tulo|parte|"
                                r"se[çc][ãa]o|pre[âa]mbulo)\b")
        if soh_estatuto.match(t2) and not soh_estatuto.match(t1) and not estrutural.match(t1):
            linhas[idx_heads[0]] = f"# Estatuto — {t1}"
            linhas[idx_heads[1]] = ""
            relatorio["preambulo_fundido"] = f"Estatuto — {t1}"
        elif soh_estatuto.match(t1) and not soh_estatuto.match(t2) and not estrutural.match(t2):
            linhas[idx_heads[0]] = f"# Estatuto — {t2}"
            linhas[idx_heads[1]] = ""
            relatorio["preambulo_fundido"] = f"Estatuto — {t2}"

    out = []
    primeiro = True
    for l in linhas:
        if _e_estrutural(l):
            out.append(l)
            continue
        m = _HEAD_RE.match(l.strip())
        if not m:
            out.append(l)
            continue
        txt = m.group(2).strip()
        if _DISP_RE.match(txt):
            out.append(txt)          # "heading" que é artigo -> parágrafo
            continue
        if re.match(r"(?i)^(t[íi]tulo|parte)\b", txt):
            nivel = 2
        elif re.match(r"(?i)^cap[íi]tulo\b", txt):
            nivel = 3 if tem_titulo_agrupador else 2
        elif re.match(r"(?i)^(se[çc][ãa]o|subse[çc][ãa]o)\b", txt):
            nivel = 3
        elif primeiro:
            nivel = 1
        else:
            nivel = min(max(len(m.group(1)), 2), 3)
        primeiro = False
        out.append("#" * nivel + " " + txt)
    return "\n".join(out)


def p11_limpeza_final(md: str) -> str:
    out = []
    for l in md.split("\n"):
        l = re.sub(r"^(\s*)\[(?:CALLOUT|TOGGLE)\]\s*", r"\1", l)
        out.append(l.rstrip())
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ============================================================ métricas / gate
def medir_ruido(md: str) -> dict:
    return {
        "latex": len(_SPAN_RE.findall(md)) + len(re.findall(r"\\(?:mathbf|circ|mathrm|boldsymbol)", md)),
        "numero_sign": md.count("№"),
        "imagens": len(re.findall(r"(?m)^\s*\[IMAGEM n=", md)),
        "headings_rotulo": len([1 for l in md.split("\n") if _H_ROTULO.match(l.strip())]),
        "chars": len(md),
    }


def contar_artigos(md: str) -> int:
    return len(_ART_COUNT_RE.findall(md))


def limpar_md(md: str, *, recortar: bool = True, manter_imagens: bool = False
              ) -> Tuple[str, dict, List[str], List[str]]:
    """Pipeline completo. Retorna (md_limpo, relatorio, corte_ini, corte_fim)."""
    relatorio: dict = {"ruido_bruto": medir_ruido(md), "artigos_bruto": contar_artigos(md)}
    md = p01_normalizar_chars(md)
    md = p02_latex_para_texto(md, relatorio)
    md = p015_remover_lixo_pje(md, relatorio)
    md = p016_rodapes_repetidos(md, relatorio)
    md = p03_deshifenizar(md)
    md = p04_fundir_blocos_partidos(md)
    md = p045_fundir_fragmentos(md, relatorio)
    md = p046_numeros_de_pagina(md, relatorio)
    md = p05_separar_artigos_colados(md)
    md = p055_ordinais_dispositivos(md)
    md = p06_fundir_headings_duplicados(md)
    md = p07_pontuacao_headings(md)
    corte_ini: List[str] = []
    corte_fim: List[str] = []
    if recortar:
        md, corte_ini, corte_fim = p08_recortar_escopo(md, relatorio)
    relatorio["artigos_pos_recorte"] = contar_artigos(md)
    md = p09_remover_imagens(md, relatorio, manter=manter_imagens)
    md = p10_hierarquia(md, relatorio)
    md = p11_limpeza_final(md)
    relatorio["artigos_final"] = contar_artigos(md)
    relatorio["ruido_final"] = medir_ruido(md)
    return md, relatorio, corte_ini, corte_fim


# ============================================================ tipo de documento
def detectar_tipo_documento(md: str) -> str:
    """'normativo' (estatutos, regimentos, resoluções, leis — texto articulado) ou
    'generico' (relatórios, pareceres, ofícios, manuais). Decide os prompts da IA."""
    return "normativo" if contar_artigos(md) >= 8 else "generico"


_CONTEXTO_DOC = {
    "normativo": "documentos NORMATIVOS brasileiros (estatutos, regimentos, resoluções, leis)",
    "generico": "documentos institucionais e jurídicos brasileiros (relatórios, pareceres, "
                "ofícios, manuais, notas técnicas)",
}


# ============================================================ IA opcional
_SYSTEM_IA = """Você corrige parágrafos isolados de {contexto} importados de PDF \
que vieram de OCR com ruído.
REGRAS ABSOLUTAS:
- NUNCA resuma, reescreva, acrescente ou remova conteúdo; corrija apenas ruído tipográfico de OCR.
- Preserve EXATAMENTE números de artigos, parágrafos, incisos, leis, resoluções, datas, \
percentuais e nomes próprios.
- Converta resíduos de notação matemática/LaTeX para texto corrido (ex.: "9^{circ}" -> "9º"; \
"S 2º" com S de parágrafo -> "§ 2º"; "1 / 3" -> "1/3").
- Corrija apenas erros INEQUÍVOCOS de OCR (letra no lugar de dígito, acento óbvio, hifenização \
de quebra de linha).
- Na dúvida, devolva o texto EXATAMENTE como veio.
Responda em JSON: {"itens": [{"i": <número do item>, "texto": "<texto corrigido>"}]} \
contendo TODOS os itens recebidos, na mesma ordem."""


def _tokens_norm(s: str) -> List[str]:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]{4,}", s)]


def gate_fidelidade(antes: str, depois: str) -> bool:
    if re.findall(r"\d+", antes) != re.findall(r"\d+", depois):
        return False
    razao = len(depois) / max(1, len(antes))
    if not (0.9 <= razao <= 1.1):
        return False
    a, b = _tokens_norm(antes), _tokens_norm(depois)
    novos = [t for t in b if t not in set(a)]
    if any(re.search(r"\d", t) for t in novos):
        return False
    if a and difflib.SequenceMatcher(a=a, b=b).ratio() < 0.85:
        return False
    return True


def selecionar_suspeitos(linhas: List[str]) -> List[int]:
    idxs = []
    for i, l in enumerate(linhas):
        s = l.strip()
        if not s or _e_estrutural(l):
            continue
        if re.search(r"[\\$]|\^\{|№", s):
            idxs.append(i)
            continue
        if re.search(r"\b\d+[OIl]\d+\b|\b[OIl]\d{2,}\b", s):  # 2O22, l2.034
            idxs.append(i)
    return idxs


def curar_com_ia(md: str, relatorio: dict, log: Callable[[str], None] = print,
                 tipo: str = "normativo") -> str:
    from conle_gerador import llm
    system = _SYSTEM_IA.replace("{contexto}", _CONTEXTO_DOC.get(tipo, _CONTEXTO_DOC["generico"]))
    linhas = md.split("\n")
    idxs = selecionar_suspeitos(linhas)
    relatorio["ia_suspeitos"] = len(idxs)
    if not idxs:
        log("  IA: nenhum parágrafo suspeito — nada a fazer.")
        return md
    aceitos = rejeitados = chamadas = 0
    tokens_est = 0
    for lote_ini in range(0, len(idxs), 20):
        lote = idxs[lote_ini:lote_ini + 20]
        itens = [{"i": k, "texto": linhas[k].strip()} for k in lote]
        user = json.dumps({"itens": itens}, ensure_ascii=False)
        tokens_est += (len(user) + len(system)) // 4
        chamadas += 1
        try:
            resp = llm.chat(system, user, json_mode=True, model=MODELO_IA)
        except Exception as exc:  # noqa: BLE001
            log(f"  IA: lote falhou ({exc}) — mantendo texto determinístico.")
            continue
        por_i = {it.get("i"): it.get("texto", "") for it in (resp or {}).get("itens", [])
                 if isinstance(it, dict)}
        for k in lote:
            novo = (por_i.get(k) or "").strip()
            antes = linhas[k].strip()
            if not novo or novo == antes:
                continue
            if gate_fidelidade(antes, novo):
                ind = linhas[k][:len(linhas[k]) - len(linhas[k].lstrip())]
                linhas[k] = ind + novo
                aceitos += 1
            else:
                rejeitados += 1
    relatorio["ia_chamadas"] = chamadas
    relatorio["ia_aceitos"] = aceitos
    relatorio["ia_rejeitados"] = rejeitados
    relatorio["ia_tokens_estimados"] = tokens_est
    log(f"  IA: {chamadas} chamada(s), {aceitos} correção(ões) aceitas, "
        f"{rejeitados} rejeitadas pelo gate, ~{tokens_est} tokens de entrada.")
    return "\n".join(linhas)


# ============================================================ partido / título
# ============================================================ revisão final IA
_SYSTEM_REVISAO = """Você faz a REVISÃO FINAL de parágrafos de {contexto} importados de PDF, \
já limpos de OCR. Sua única função é remover ruído residual de digitalização e \
corrigir espaçamento.
PODE:
- corrigir espaçamento e hifenização ("Secre taria" -> "Secretaria"; espaço antes de vírgula);
- REMOVER ruído incrustado que não faz parte do texto legal: endereços de cartório/sede soltos \
no meio de um dispositivo, telefone/fax e variantes de OCR ("Tet/Fax", "TelfFax", "Tell/Fax"), \
sites, números de protocolo/selo/código, datas de carimbo, fragmentos de rodapé ou cabeçalho, \
letras órfãs no fim de frases, símbolos espúrios ("©" antes de inciso, "U-" grudado em numeração), \
sequências sem sentido;
- corrigir APENAS grafias impossíveis em português, sem mudar a palavra pretendida \
("Paptido" -> "Partido", "Düretório" -> "Diretório", "votouo" -> "votou o", "grau de-recurso" \
-> "grau de recurso");
- devolver "" (string vazia) quando o item INTEIRO for ruído (número solto, código, resto de carimbo).
NÃO PODE (regras absolutas):
- reescrever, parafrasear, resumir, reordenar ou completar texto;
- adicionar QUALQUER palavra, letra ou número que não esteja no item;
- alterar números de artigos, parágrafos, incisos, leis, resoluções, datas, percentuais ou valores;
- remover marcadores de dispositivo mesmo sozinhos na linha ("§ 11º", "Artigo 5º", "IV -"): \
são rótulos legítimos, não ruído;
- remover texto normativo (na dúvida sobre ser ruído, MANTENHA como está).
Responda em JSON: {"itens": [{"i": <número>, "texto": "<revisado>"}]} contendo APENAS os itens \
que você alterou (omita os inalterados)."""


def _cat_alnum(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(re.findall(r"[a-z0-9]+", s.lower()))


def gate_delecao(antes: str, depois: str) -> bool:
    """Aceita a revisão apenas se a saída for a entrada MENOS deleções (respaçamento
    é livre porque a comparação ignora espaços/pontuação): nenhuma inserção ou
    substituição de conteúdo, e no máximo 30% do texto removido."""
    cat_a, cat_b = _cat_alnum(antes), _cat_alnum(depois)
    if not cat_b:
        return len(antes.strip()) <= 80  # remoção total: só para linhas curtas
    if cat_b == cat_a:
        return True  # só respaçamento/pontuação
    ops = difflib.SequenceMatcher(a=cat_a, b=cat_b, autojunk=False).get_opcodes()
    # micro-correções de grafia impossível ("Paptido"->"Partido") são as ÚNICAS
    # substituições toleradas: <=2 chars de cada lado, sem dígitos, máx. 3 por item
    replaces = 0
    for op, i1, i2, j1, j2 in ops:
        if op == "insert":
            return False
        if op == "replace":
            if (i2 - i1) > 2 or (j2 - j1) > 2:
                return False
            if re.search(r"\d", cat_a[i1:i2]) or re.search(r"\d", cat_b[j1:j2]):
                return False
            replaces += 1
            if replaces > 3:
                return False
    # anti-montagem: um "equal" minúsculo espremido entre duas deleções permitiria
    # compor palavra nova com letras esparsas do trecho removido
    for idx, (op, i1, i2, _j1, _j2) in enumerate(ops):
        if (op == "equal" and (i2 - i1) <= 2
                and 0 < idx < len(ops) - 1
                and ops[idx - 1][0] == "delete" and ops[idx + 1][0] == "delete"):
            return False
    removido = sum(op[2] - op[1] for op in ops if op[0] == "delete")
    return removido <= 0.30 * len(cat_a)


def revisar_com_ia(md: str, relatorio: dict, log: Callable[[str], None] = print,
                   tipo: str = "normativo") -> str:
    """Passada FINAL com a OpenAI sobre TODO o texto (lotes de linhas), com o
    gate_delecao garantindo que só saem ruídos — nada entra nem muda."""
    from conle_gerador import llm
    system = _SYSTEM_REVISAO.replace("{contexto}",
                                     _CONTEXTO_DOC.get(tipo, _CONTEXTO_DOC["generico"]))
    linhas = md.split("\n")
    idxs = [i for i, l in enumerate(linhas)
            if l.strip() and not _e_estrutural(l)]
    aceitos = rejeitados = removidas = chamadas = 0
    tokens_est = 0
    for lote_ini in range(0, len(idxs), 40):
        lote = idxs[lote_ini:lote_ini + 40]
        itens = [{"i": k, "texto": linhas[k].strip()} for k in lote]
        user = json.dumps({"itens": itens}, ensure_ascii=False)
        tokens_est += (len(user) + len(system)) // 4
        chamadas += 1
        try:
            resp = llm.chat(system, user, json_mode=True, model=MODELO_IA)
        except Exception as exc:  # noqa: BLE001
            log(f"  revisão IA: lote falhou ({exc}) — mantendo texto como está.")
            continue
        por_i = {it.get("i"): it.get("texto", None) for it in (resp or {}).get("itens", [])
                 if isinstance(it, dict)}
        for k in lote:
            if k not in por_i or por_i[k] is None:
                continue
            novo = str(por_i[k]).strip()
            antes = linhas[k].strip()
            if novo == antes:
                continue
            m = _HEAD_RE.match(antes)
            if m and novo and not novo.startswith("#"):
                novo = m.group(1) + " " + novo  # preserva o nível do heading
            if gate_delecao(antes, novo):
                if novo:
                    ind = linhas[k][:len(linhas[k]) - len(linhas[k].lstrip())]
                    linhas[k] = ind + novo
                else:
                    linhas[k] = ""
                    removidas += 1
                aceitos += 1
            else:
                rejeitados += 1
    relatorio["revisao_ia_chamadas"] = chamadas
    relatorio["revisao_ia_aceitos"] = aceitos
    relatorio["revisao_ia_linhas_removidas"] = removidas
    relatorio["revisao_ia_rejeitados"] = rejeitados
    relatorio["revisao_ia_tokens_estimados"] = tokens_est
    log(f"  revisão IA: {chamadas} chamada(s), {aceitos} ajustes aceitos "
        f"({removidas} linhas-ruído removidas), {rejeitados} rejeitados pelo gate, "
        f"~{tokens_est} tokens de entrada.")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(linhas))


def detectar_partido(md: str) -> Tuple[Optional[str], int]:
    """Retorna (sigla, confianca). confianca: 3=parênteses, 2=nome por extenso,
    1=sigla isolada, 0=não detectado."""
    terco = md[: max(2000, len(md) // 3)]
    pontos: Dict[str, int] = {}

    todas = SIGLAS + list(ALIAS)
    par_re = re.compile(r"\((%s)\)" % "|".join(re.escape(s) for s in todas))
    for m in par_re.finditer(terco):
        sig = ALIAS.get(m.group(1), m.group(1))
        pontos[sig] = pontos.get(sig, 0) + 3

    for sig, nome_re in PARTIDOS:
        if nome_re:
            n = len(re.findall(nome_re, terco))
            if n:
                pontos[sig] = pontos.get(sig, 0) + 2 * min(n, 5)

    # frequência da sigla isolada no doc inteiro decide casos como o DEMOCRATA
    # (31 menções) contra 1 citação incidental de outro partido
    for sig in todas:
        n = len(re.findall(r"\b%s\b" % re.escape(sig), md))
        if n >= 3:
            canon = ALIAS.get(sig, sig)
            pontos[canon] = pontos.get(canon, 0) + min(n, 10)

    if not pontos:
        return None, 0
    rank = sorted(pontos.items(), key=lambda kv: -kv[1])
    if len(rank) > 1 and rank[1][1] >= rank[0][1]:
        return None, 0  # empate -> ambíguo
    sig, pts = rank[0]
    return sig, min(pts, 3)


def detectar_data(titulo_antigo: str, relatorio: dict) -> Tuple[str, str]:
    """Retorna (rotulo, data) p/ o título novo: ('aprovado em', d) ou ('de', d) ou ('','')."""
    if relatorio.get("data_aprovacao"):
        return "aprovado em", relatorio["data_aprovacao"]
    m = re.search(r"aprovado_em_(\d{1,2})_(\d{1,2})_(\d{4})", titulo_antigo or "")
    if m:
        return "aprovado em", f"{int(m.group(1))}.{int(m.group(2))}.{m.group(3)}"
    m = re.search(r"de_(\d{1,2})_(\d{1,2})_(\d{4})", titulo_antigo or "")
    if m:
        return "de", f"{int(m.group(1))}.{int(m.group(2))}.{m.group(3)}"
    return "", ""


def montar_titulo(sigla: str, rotulo: str, data: str) -> str:
    art = "da" if sigla in _FEM else "do"
    base = f"Estatuto {art} {sigla}"
    return f"{base} — {rotulo} {data}" if data else base


def renomear_pagina(pid: str, novo_titulo: str) -> None:
    ln._req("PATCH", f"/pages/{pid}", {
        "properties": {"title": {"title": [
            {"type": "text", "text": {"content": novo_titulo[:1990]}}]}}})


# ============================================================ workdir / manifest
def _id8(pid: str) -> str:
    # últimos 12 hex: os UUIDs do workspace compartilham o prefixo temporal,
    # só o final distingue as páginas
    return normalize_page_id(pid).replace("-", "")[-12:]


def _dir_pagina(pid: str) -> Path:
    d = WORK / _id8(pid)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _carrega(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _grava(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _manifest() -> dict:
    return _carrega(MANIFEST_JSON, {})


def _atualiza_manifest(pid: str, **campos) -> dict:
    man = _manifest()
    ent = man.setdefault(normalize_page_id(pid), {})
    ent.update(campos)
    _grava(MANIFEST_JSON, man)
    return ent


def _sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


# ============================================================ operações
def op_descobrir(url_ou_id: str) -> List[dict]:
    pid = normalize_page_id(url_ou_id)
    out, cursor = [], None
    while True:
        q = "?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        d = ln._req("GET", f"/blocks/{pid}/children{q}")
        for b in d.get("results", []):
            if b.get("type") == "child_page":
                out.append({"page_id": b["id"],
                            "titulo": (b.get("child_page") or {}).get("title", ""),
                            "id8": _id8(b["id"])})
        if not d.get("has_more"):
            break
        cursor = d.get("next_cursor")
    _grava(PAGINAS_JSON, {"mae": pid, "paginas": out})
    print(f"{len(out)} subpágina(s) encontrada(s) em {pid}:")
    for p in out:
        print(f"  {p['id8']}  {p['titulo']}")
    return out


def _processa(pid: str, *, recortar: bool, manter_imagens: bool, com_ia: bool,
              revisao_ia: bool = False, log: Callable[[str], None] = print):
    """dump -> render -> limpeza. Retorna dict com tudo (sem tocar o Notion)."""
    pid = normalize_page_id(pid)
    blocks = dl.dump_blocos(pid)
    images: List[dict] = []
    bruto = dl.render_md(blocks, images)
    limpo, rel, corte_ini, corte_fim = limpar_md(
        bruto, recortar=recortar, manter_imagens=manter_imagens)
    tipo = detectar_tipo_documento(bruto)
    rel["tipo_documento"] = tipo
    if com_ia:
        limpo = curar_com_ia(limpo, rel, log, tipo=tipo)
    if revisao_ia:
        limpo = revisar_com_ia(limpo, rel, log, tipo=tipo)
        rel["artigos_final"] = contar_artigos(limpo)
    sigla, conf = detectar_partido(bruto)
    rel["partido"] = sigla or ""
    rel["partido_confianca"] = conf
    return {"pid": pid, "blocks": blocks, "images": images, "bruto": bruto,
            "limpo": limpo, "rel": rel, "corte_ini": corte_ini, "corte_fim": corte_fim,
            "sigla": sigla, "conf": conf}


def _titulo_atual(pid: str) -> str:
    try:
        return page_title(dl.client().get_page(normalize_page_id(pid)))
    except Exception:
        return ""


def op_diagnosticar(pids: List[str]) -> None:
    diag = _carrega(DIAG_JSON, {})
    for pid in pids:
        pid = normalize_page_id(pid)
        titulo = _titulo_atual(pid)
        print(f"— diagnosticando {_id8(pid)} ({titulo[:60]})")
        r = _processa(pid, recortar=True, manter_imagens=False, com_ia=False)
        rel = r["rel"]
        linha = {
            "pagina": pid, "id8": _id8(pid), "titulo": titulo,
            "partido": r["sigla"] or "", "partido_confianca": r["conf"],
            "chars": rel["ruido_bruto"]["chars"],
            "artigos": rel["artigos_final"],
            "latex": rel["ruido_bruto"]["latex"],
            "imagens": rel["ruido_bruto"]["imagens"],
            "corte_inicio_linhas": rel.get("corte_inicio_linhas", 0),
            "corte_fim_linhas": rel.get("corte_fim_linhas", 0),
            "avisos": "; ".join(str(rel[k]) for k in ("recorte_inicio", "recorte_fim")
                                if k in rel),
        }
        diag[pid] = linha
        _atualiza_manifest(pid, partido=linha["partido"], titulo_antigo=titulo,
                           conferir_partido=(r["conf"] < 2))
        print(f"   partido={linha['partido'] or '?'} artigos={linha['artigos']} "
              f"latex={linha['latex']} imagens={linha['imagens']} "
              f"cortes={linha['corte_inicio_linhas']}+{linha['corte_fim_linhas']}"
              + (f"  AVISO: {linha['avisos']}" if linha["avisos"] else ""))
    _grava(DIAG_JSON, diag)
    with DIAG_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        campos = ["id8", "titulo", "partido", "partido_confianca", "chars", "artigos",
                  "latex", "imagens", "corte_inicio_linhas", "corte_fim_linhas", "avisos"]
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for v in diag.values():
            w.writerow(v)
    print(f"diagnóstico: {len(diag)} página(s) -> {DIAG_CSV}")


def op_previa(pid: str, *, com_ia: bool, manter_imagens: bool, recortar: bool,
              revisao_ia: bool = False) -> None:
    pid = normalize_page_id(pid)
    d = _dir_pagina(pid)
    titulo = _titulo_atual(pid)
    print(f"— prévia de {_id8(pid)} ({titulo[:70]})")
    r = _processa(pid, recortar=recortar, manter_imagens=manter_imagens, com_ia=com_ia,
                  revisao_ia=revisao_ia)
    (d / "bruto.md").write_text(r["bruto"], encoding="utf-8")
    (d / "limpo.md").write_text(r["limpo"], encoding="utf-8")
    cortes = ["# CORTES PROPOSTOS", "", "## INÍCIO REMOVIDO", ""]
    cortes += [l for l in r["corte_ini"] if l.strip()] or ["(nada)"]
    cortes += ["", "## FIM REMOVIDO", ""]
    cortes += r["corte_fim"] or ["(nada)"]
    (d / "cortes.md").write_text("\n".join(cortes), encoding="utf-8")
    _grava(d / "relatorio.json", r["rel"])
    if manter_imagens and r["images"]:
        mapa = dl.baixar_imagens(r["images"], d / "img")
        _grava(d / "images.json", mapa)
        print(f"   {len(mapa)} imagem(ns) baixadas para reanexar.")
    m_tit = re.match(r"^Estatuto d[ao] (\S+)", titulo)
    if m_tit:
        titulo_novo = titulo  # página já renomeada numa rodada anterior: preserva
        if not r["sigla"] and m_tit.group(1) in SIGLAS:
            # o título atual é a pista mais confiável quando o corpo já perdeu a
            # folha de rosto (caso UP): herda a sigla e não marca "conferir"
            r["sigla"], r["conf"] = m_tit.group(1), 2
    else:
        rotulo, data = detectar_data(titulo, r["rel"])
        titulo_novo = montar_titulo(r["sigla"], rotulo, data) if r["sigla"] else ""
    _atualiza_manifest(
        pid, status="previa", hash_limpo=_sha(r["limpo"]), titulo_antigo=titulo,
        titulo_novo_proposto=titulo_novo, partido=r["sigla"] or "",
        conferir_partido=(r["conf"] < 2), com_ia=com_ia,
        manter_imagens=manter_imagens, recortar=recortar)
    rel = r["rel"]
    ini, fim = rel.get("corte_inicio_linhas", 0), rel.get("corte_fim_linhas", 0)
    print(f"   artigos: bruto={rel['artigos_bruto']} pós-recorte={rel['artigos_pos_recorte']} "
          f"final={rel['artigos_final']}")
    print(f"   ruído: latex {rel['ruido_bruto']['latex']}->{rel['ruido_final']['latex']} | "
          f"imagens {rel['ruido_bruto']['imagens']} removidas={rel.get('imagens_removidas', 0)}")
    print(f"   CORTE INÍCIO: {ini} linha(s); CORTE FIM: {fim} linha(s) -> ver {d / 'cortes.md'}")
    for k in ("recorte_inicio", "recorte_fim"):
        if k in rel:
            print(f"   AVISO: {rel[k]}")
    if titulo_novo:
        print(f"   título proposto: {titulo_novo}")
    else:
        print("   partido não detectado com segurança — página NÃO será renomeada.")
    print(f"   prévia em: {d}")


def op_aplicar(pid: str, *, renomear: bool, forcar: bool) -> int:
    pid = normalize_page_id(pid)
    d = _dir_pagina(pid)
    man = _manifest().get(pid, {})
    if man.get("status") == "aplicada" and not forcar:
        print(f"— {_id8(pid)}: já aplicada (use --forcar para reaplicar). Pulando.")
        return 0
    limpo_path = d / "limpo.md"
    if not limpo_path.exists():
        print(f"ERRO: prévia inexistente para {_id8(pid)} — rode --previa antes.")
        return 2
    limpo = limpo_path.read_text(encoding="utf-8")
    if man.get("hash_limpo") and _sha(limpo) != man["hash_limpo"]:
        print(f"ERRO: limpo.md de {_id8(pid)} difere da prévia registrada "
              "(hash divergente). Regere a prévia com --previa.")
        return 2
    rel = _carrega(d / "relatorio.json", {})
    a_rec, a_fim = rel.get("artigos_pos_recorte"), rel.get("artigos_final")
    if a_rec is not None and a_fim is not None and a_fim < a_rec:
        print(f"ERRO: gate de artigos falhou ({a_fim} < {a_rec}) — aplicação abortada.")
        return 2
    n_imgs = rel.get("ruido_bruto", {}).get("imagens", 0)
    if a_fim is not None and a_fim < 20 and n_imgs > 5 and not man.get("manter_imagens"):
        print(f"ERRO: {_id8(pid)} tem só {a_fim} artigo(s) em texto e {n_imgs} imagens — "
              "página majoritariamente ESCANEADA (o conteúdo está nas imagens). "
              "Aplicar removeria o estatuto. Gere a prévia com --manter-imagens ou "
              "trate esta página manualmente.")
        return 2
    print(f"— aplicando {_id8(pid)} ({man.get('titulo_antigo', '')[:60]})")
    blocos = build.md_doc_to_blocks(limpo)
    print(f"   {len(blocos)} bloco(s) novos.")
    # nunca sobrescreve o backup do conteúdo ORIGINAL numa reaplicação; e cada
    # reaplicação/retentativa ganha arquivo numerado (uma tentativa interrompida
    # no delete deixaria o backup seguinte parcial se fosse sobrescrito)
    bkp = d / "backup_original.json"
    if bkp.exists():
        n = 1
        while (d / f"backup_reaplicacao_{n}.json").exists():
            n += 1
        bkp = d / f"backup_reaplicacao_{n}.json"
    dl.substituir_inplace(pid, blocos, backup_path=bkp,
                          progress=lambda m: print(m))
    if man.get("manter_imagens") and (d / "images.json").exists():
        mapa = _carrega(d / "images.json", [])
        feitos, alvos = dl.reanexar_imagens(pid, mapa)
        print(f"   imagens reanexadas: {feitos}/{alvos}")
    titulo_novo = (man.get("titulo_novo_proposto") or "").strip()
    if renomear and titulo_novo and not man.get("conferir_partido"):
        renomear_pagina(pid, titulo_novo)
        print(f"   renomeada para: {titulo_novo}")
    elif renomear and not titulo_novo:
        print("   sem título proposto — página não renomeada.")
    _atualiza_manifest(pid, status="aplicada")
    print(f"   OK. Backup em {d / 'backup_original.json'}")
    return 0


# ---------- restauração a partir do backup
_RT_KEEP = {"type", "text", "equation", "mention", "annotations"}


def _sanitiza_rt(rt_list) -> List[dict]:
    out = []
    for t in rt_list or []:
        novo = {k: v for k, v in t.items() if k in _RT_KEEP}
        txt = novo.get("text")
        if isinstance(txt, dict):
            novo["text"] = {k: v for k, v in txt.items() if k in ("content", "link")}
        out.append(novo)
    return out


def _bloco_escrevivel(b: dict, perdas: List[str]) -> Optional[dict]:
    t = b.get("type")
    data = b.get(t)
    if not t or not isinstance(data, dict):
        return None
    if t in ("child_page", "child_database", "unsupported"):
        perdas.append(f"{t} ignorado")
        return None
    if t == "image":
        if data.get("type") == "external":
            novo = {"type": "external", "external": data.get("external", {})}
            if data.get("caption"):
                novo["caption"] = _sanitiza_rt(data["caption"])
            return {"object": "block", "type": "image", "image": novo}
        perdas.append("imagem interna (URL S3 expirada) não restaurável")
        return None
    novo_data = dict(data)
    for k in ("rich_text", "caption"):
        if k in novo_data:
            novo_data[k] = _sanitiza_rt(novo_data[k])
    if t == "table_row":
        novo_data["cells"] = [_sanitiza_rt(c) for c in data.get("cells", [])]
    filhos = []
    for fb in b.get("_children") or []:
        conv = _bloco_escrevivel(fb, perdas)
        if conv:
            filhos.append(conv)
    if t == "table":
        novo_data = {"table_width": data.get("table_width", 1),
                     "has_column_header": data.get("has_column_header", False),
                     "has_row_header": data.get("has_row_header", False),
                     "children": filhos}
    elif filhos:
        novo_data["children"] = filhos
    return {"object": "block", "type": t, t: novo_data}


def op_restaurar(pid: str) -> int:
    pid = normalize_page_id(pid)
    d = _dir_pagina(pid)
    bkp_path = d / "backup_original.json"
    if not bkp_path.exists():
        print(f"ERRO: não há backup em {bkp_path}.")
        return 2
    backup = _carrega(bkp_path, [])
    perdas: List[str] = []
    blocos = [bl for b in backup if (bl := _bloco_escrevivel(b, perdas))]
    print(f"— restaurando {_id8(pid)}: {len(blocos)} bloco(s) do backup"
          + (f" ({len(perdas)} perda(s): {', '.join(sorted(set(perdas)))})" if perdas else ""))
    dl.substituir_inplace(pid, blocos, backup_path=d / "backup_pos_aplicacao.json",
                          progress=lambda m: print(m))
    man = _manifest().get(pid, {})
    antigo = (man.get("titulo_antigo") or "").strip()
    if antigo:
        renomear_pagina(pid, antigo)
        print(f"   título restaurado: {antigo}")
    _atualiza_manifest(pid, status="restaurada")
    print("   OK.")
    return 0


# ============================================================ CLI
def _pids_de(args) -> List[str]:
    if args.pagina:
        return [args.pagina]
    if args.todas:
        pags = _carrega(PAGINAS_JSON, {}).get("paginas", [])
        if not pags:
            print("ERRO: rode --descobrir antes (paginas.json vazio).")
            sys.exit(2)
        return [p["page_id"] for p in pags]
    print("ERRO: informe --pagina ID ou --todas.")
    sys.exit(2)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Normalizador de páginas do Notion importadas de PDF")
    ap.add_argument("--descobrir", metavar="URL", help="lista subpáginas da página-mãe")
    ap.add_argument("--diagnosticar", action="store_true")
    ap.add_argument("--previa", action="store_true")
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--restaurar", action="store_true")
    ap.add_argument("--pagina", metavar="ID")
    ap.add_argument("--todas", action="store_true")
    ap.add_argument("--com-ia", action="store_true", dest="com_ia")
    ap.add_argument("--revisao-ia", action="store_true", dest="revisao_ia",
                    help="passada FINAL da OpenAI no texto todo (remove ruído, corrige espaços)")
    ap.add_argument("--manter-imagens", action="store_true", dest="manter_imagens")
    ap.add_argument("--sem-recorte", action="store_true", dest="sem_recorte")
    ap.add_argument("--sem-renomear", action="store_true", dest="sem_renomear")
    ap.add_argument("--forcar", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(exist_ok=True)
    if args.descobrir:
        op_descobrir(args.descobrir)
        return
    if args.diagnosticar:
        op_diagnosticar(_pids_de(args))
        return
    if args.previa:
        falhas = []
        for pid in _pids_de(args):
            for tentativa in range(3):
                try:
                    op_previa(pid, com_ia=args.com_ia, manter_imagens=args.manter_imagens,
                              recortar=not args.sem_recorte, revisao_ia=args.revisao_ia)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"ERRO transitório em {_id8(pid)} ({exc}) — "
                          f"tentativa {tentativa + 1}/3")
                    time.sleep(5 * (tentativa + 1))
            else:
                falhas.append(_id8(pid))
        if falhas:
            print(f"ERRO: prévia falhou para: {', '.join(falhas)}")
            sys.exit(1)
        return
    if args.aplicar:
        rc = 0
        for pid in _pids_de(args):
            # a rede do Notion sofre resets intermitentes (WinError 10054): a página
            # é retomável (backup + manifest), então tenta de novo antes de desistir
            for tentativa in range(3):
                try:
                    rc = max(rc, op_aplicar(pid, renomear=not args.sem_renomear,
                                            forcar=args.forcar))
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"ERRO transitório em {_id8(pid)} ({exc}) — "
                          f"tentativa {tentativa + 1}/3")
                    time.sleep(5 * (tentativa + 1))
            else:
                print(f"ERRO: {_id8(pid)} falhou após 3 tentativas — seguindo o lote.")
                rc = 2
        sys.exit(rc)
    if args.restaurar:
        if not args.pagina:
            print("ERRO: --restaurar exige --pagina ID.")
            sys.exit(2)
        sys.exit(op_restaurar(args.pagina))
    ap.print_help()


if __name__ == "__main__":
    main()
