# -*- coding: utf-8 -*-
"""Descobridor semi-automático de fontes oficiais — o "passo 2" do detector de
lacunas (aprovado em conceito em 05/07/2026, implementado em 06/07/2026).

Quando a página cita uma norma FORA dos mapas (o aviso "Normas citadas sem
fonte mapeada"), este módulo:
  1. deriva CANDIDATAS de URL pelas convenções conhecidas (Planalto para
     leis/LCs/ECs; página compilada do TSE para resoluções);
  2. VALIDA cada candidata por GET real (status 200 + o número da norma no
     conteúdo) e afere se a página tem âncoras de artigo (#artN);
  3. só grava em `normas_extras.json` com a CONFIRMAÇÃO do usuário (CLI
     interativa ou botão "Verificar fontes" da GUI).

O `config` mescla o arquivo nos mapas no import — a conversão seguinte já
linka. POLÍTICA: nunca chutar URL em tempo de conversão; toda entrada nasce
de candidata validada + confirmação humana.

Uso:  py -m conle_conversor.fontes <url-da-pagina-notion> [--auto]
      py -m conle_conversor.fontes --norma "Lei nº 12.034/2009" [--auto]
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import requests

from . import config

ARQ_EXTRAS = Path(__file__).resolve().parent / "normas_extras.json"
_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_TIMEOUT = 25
_PLANALTO = "https://www.planalto.gov.br/ccivil_03"

_MESES = ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho",
          "agosto", "setembro", "outubro", "novembro", "dezembro"]


@dataclass
class Candidata:
    url: str
    valida: bool = False
    tem_ancoras: bool = False
    obs: str = ""


@dataclass
class Achado:
    designacao: str                     # como citada no texto
    classe: str                         # lei | lcp | emenda | resolucao_tse
    chave: str                          # dígitos ("12034") ou "NN.NNN"
    ano: Optional[int] = None
    candidatas: List[Candidata] = field(default_factory=list)

    @property
    def melhor(self) -> Optional[Candidata]:
        validas = [c for c in self.candidatas if c.valida]
        validas.sort(key=lambda c: not c.tem_ancoras)  # com âncoras primeiro
        return validas[0] if validas else None


# ---------------------------------------------------------------------------
# parse da designação (as strings que o detector de lacunas devolve)
# ---------------------------------------------------------------------------
_RE_LEI = re.compile(
    r"^Lei\s*n[ºo°.]*\s*(?P<num>[\d.]+)\s*"
    r"(?:/\s*(?P<ano1>\d{2,4})|,?\s+de(?:\s+(?P<data>[^,;()]*?))?\s*(?P<ano2>\d{4}))?$",
    re.I,
)
_RE_LCP = re.compile(
    r"^(?:LC|Lei\s+Complementar)\s*n?[ºo°.]*\s*(?P<num>[\d.]+)\s*"
    r"(?:/\s*(?P<ano1>\d{2,4})|,?\s+de[^;()]*?(?P<ano2>\d{4}))?$",
    re.I,
)
_RE_EC = re.compile(
    r"^(?:EC|Emenda\s+Constitucional)\s*n?[ºo°.]*\s*(?P<num>\d{1,3})\s*"
    r"(?:/\s*(?P<ano1>\d{4})|,?\s+de[^;()]*?(?P<ano2>\d{4}))?$",
    re.I,
)
_RE_RES_TSE = re.compile(
    r"^Res(?:\.|olu[çc][ãa]o)?\s*[-–.]?\s*(?:d[oa]\s+)?(?:TSE\s*)?n?[ºo°.]*\s*"
    r"(?P<num>\d{2}\.?\d{3})\s*"
    r"(?:/\s*(?P<ano1>\d{4})|,?\s+de\s+(?P<dia>\d{1,2})[ºo°]?\s+de\s+"
    r"(?P<mes>[a-zç]+)\s+de\s+(?P<ano2>\d{4}))?$",
    re.I,
)


def _digitos(s: str) -> str:
    return re.sub(r"\D", "", s or "")


_RE_RES_CAMARA = re.compile(
    r"^Resolu[çc][ãa]o(?:\s+da\s+C[âa]mara(?:\s+dos\s+Deputados)?)?\s*"
    r"n[ºo°.]*\s*(?P<num>\d{1,3})\s*"
    r"(?:/\s*(?P<ano1>\d{4})|,?\s+de(?:\s+\d{1,2}[ºo°]?\s+de\s+[a-zç]+\s+de)?\s+(?P<ano2>\d{4}))$",
    re.I,
)


def parsear_designacao(txt: str) -> Optional[Achado]:
    t = " ".join((txt or "").split()).strip(" .,;")
    m = _RE_RES_TSE.match(t)
    if m and len(_digitos(m.group("num"))) == 5:
        num = _digitos(m.group("num"))
        ach = Achado(t, "resolucao_tse", f"{num[:2]}.{num[2:]}",
                     int(m.group("ano1") or m.group("ano2") or 0) or None)
        ach._detalhe = m.groupdict()  # dia/mês p/ montar o slug
        return ach
    m = _RE_RES_CAMARA.match(t)
    if m:
        ano = int(m.group("ano1") or m.group("ano2"))
        return Achado(t, "resolucao_camara", f"{int(m.group('num'))}/{ano}", ano)
    m = _RE_EC.match(t)
    if m:
        return Achado(t, "emenda", _digitos(m.group("num")),
                      int(m.group("ano1") or m.group("ano2") or 0) or None)
    m = _RE_LCP.match(t)
    if m:
        return Achado(t, "lcp", _digitos(m.group("num")),
                      int(m.group("ano1") or m.group("ano2") or 0) or None)
    m = _RE_LEI.match(t)
    if m:
        ano = m.group("ano1") or m.group("ano2")
        if ano and len(ano) == 2:
            ano = ("19" if int(ano) > 26 else "20") + ano
        return Achado(t, "lei", _digitos(m.group("num")),
                      int(ano) if ano else None)
    return None


# ---------------------------------------------------------------------------
# candidatas por convenção
# ---------------------------------------------------------------------------
def _faixa_ato(ano: int) -> Optional[str]:
    for ini in (2004, 2007, 2011, 2015, 2019, 2023):
        fim = ini + (2 if ini in (2004, 2023) else 3)
        if ini <= ano <= fim:
            return f"_ato{ini}-{fim}"
    return None


def _com_ponto(digs: str) -> str:
    return f"{digs[:-3]}.{digs[-3:]}" if len(digs) > 3 else digs


def candidatas_de(ach: Achado) -> List[str]:
    n = ach.chave
    if ach.classe == "emenda":
        return [f"{_PLANALTO}/constituicao/emendas/emc/emc{n}.htm"]
    if ach.classe == "lcp":
        return [f"{_PLANALTO}/leis/lcp/lcp{n}.htm",
                f"{_PLANALTO}/leis/lcp/Lcp{n}.htm"]
    if ach.classe == "lei":
        digs = n
        urls = []
        if int(digs) <= 9999:
            urls += [f"{_PLANALTO}/leis/l{digs}.htm", f"{_PLANALTO}/leis/L{digs}.htm"]
        ano = ach.ano
        if ano:
            faixa = _faixa_ato(ano)
            for forma in (digs, _com_ponto(digs), _com_ponto(digs).replace(".", "")):
                if faixa:
                    urls += [f"{_PLANALTO}/{faixa}/{ano}/lei/l{forma}.htm",
                             f"{_PLANALTO}/{faixa}/{ano}/lei/L{forma}.htm"]
                urls += [f"{_PLANALTO}/leis/{ano}/l{forma}.htm",
                         f"{_PLANALTO}/leis/{ano}/L{forma}.htm"]
        # dedup preservando a ordem
        return list(dict.fromkeys(urls))
    if ach.classe == "resolucao_tse":
        det = getattr(ach, "_detalhe", {}) or {}
        dia, mes, ano = det.get("dia"), det.get("mes"), det.get("ano2") or ach.ano
        slug_num = n.replace(".", "-")
        if dia and mes and ano:
            mes_slug = _sem_acento(mes.lower())  # slugs usam "marco" sem cedilha
            base = f"https://www.tse.jus.br/legislacao/compilada/res/{ano}"
            return [f"{base}/resolucao-no-{slug_num}-de-{int(dia)}-de-{mes_slug}-de-{ano}",
                    f"{base}/resolucao-no-{slug_num}-{int(dia)}-de-{mes_slug}-de-{ano}"]
        return _buscar_res_tse_no_site(n, ach.ano)
    if ach.classe == "resolucao_camara":
        # o slug do LEGIN tem id interno não-derivável e o site bloqueia
        # busca programática — as candidatas vêm do APRENDIZADO dos links do
        # autor na própria página (analisar_pagina) ou do --url manual
        return list(getattr(ach, "_aprendidas", []) or [])
    return []


def _sem_acento(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _buscar_res_tse_no_site(num: str, ano: Optional[int]) -> List[str]:
    """Sem a data por extenso não dá para montar o slug — busca no próprio
    site do TSE (busca Plone) por links de página compilada."""
    try:
        r = requests.get(
            "https://www.tse.jus.br/@@busca",
            params={"SearchableText": f"Resolução nº {num}"},
            headers=_HDR, timeout=_TIMEOUT,
        )
        achadas = re.findall(
            rf"https://www\.tse\.jus\.br/legislacao/compilada/res/\d{{4}}/"
            rf"resolucao-no-{num.replace('.', '-')}[a-z0-9-]*",
            r.text,
        )
        return list(dict.fromkeys(achadas))[:3]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# validação por GET
# ---------------------------------------------------------------------------
def validar(cand: Candidata, chave: str) -> Candidata:
    try:
        r = requests.get(cand.url, headers=_HDR, timeout=_TIMEOUT)
    except Exception as exc:
        cand.obs = f"erro de rede: {exc}"
        return cand
    if r.status_code != 200:
        cand.obs = f"HTTP {r.status_code}"
        return cand
    r.encoding = r.apparent_encoding
    digs = _digitos(chave.split("/")[0])  # "10/2009" (Res. Câmara) -> "10"
    padrao_num = re.escape(_com_ponto(digs)) + "|" + re.escape(digs)
    if not re.search(padrao_num, r.text):
        cand.obs = "página não menciona o número da norma"
        return cand
    cand.valida = True
    # páginas do LEGIN não precisam de âncora nativa: o conversor usa text
    # fragment (#:~:text=Art.%20N), que desce ao artigo mesmo sem elas
    cand.tem_ancoras = bool(re.search(r'(?:name|id)="art\d', r.text)) or \
        "camara.leg.br/legin/" in cand.url
    cand.obs = "com âncoras #artN" if cand.tem_ancoras else "SEM âncoras de artigo"
    return cand


def descobrir(designacao: str, log: Callable[[str], None] = lambda m: None,
              aprendidas: Optional[dict] = None) -> Optional[Achado]:
    """Parse + candidatas + validação de uma designação de norma. `aprendidas`
    = {chave: url} de links do AUTOR na própria página (resoluções da Câmara
    no LEGIN, cujo slug não é derivável)."""
    ach = parsear_designacao(designacao)
    if ach is None:
        log(f"  ? não sei derivar URL para: {designacao!r} (classe não suportada)")
        return None
    if ach.classe == "resolucao_camara" and aprendidas and ach.chave in aprendidas:
        ach._aprendidas = [aprendidas[ach.chave]]
        log("  (candidata aprendida de link do autor na própria página)")
    for url in candidatas_de(ach):
        c = validar(Candidata(url), ach.chave)
        ach.candidatas.append(c)
        log(f"  {'✓' if c.valida else '✗'} {url}  ({c.obs})")
        if c.valida and c.tem_ancoras:
            break  # a melhor possível já apareceu
    return ach


# ---------------------------------------------------------------------------
# persistência (normas_extras.json) + merge em memória
# ---------------------------------------------------------------------------
_CLASSE_JSON = {"lei": "leis", "lcp": "lcps", "emenda": "emendas",
                "resolucao_tse": "resolucoes_tse", "resolucao_camara": "resolucoes_camara"}


def carregar_extras() -> dict:
    if ARQ_EXTRAS.exists():
        try:
            return json.loads(ARQ_EXTRAS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def gravar_achado(ach: Achado) -> str:
    """Grava a melhor candidata validada no arquivo de dados e atualiza os
    mapas do config JÁ CARREGADOS (a conversão na mesma sessão linka)."""
    melhor = ach.melhor
    if melhor is None:
        raise ValueError("nenhuma candidata validada para gravar")
    dados = carregar_extras()
    dados.setdefault(_CLASSE_JSON[ach.classe], {})[ach.chave] = melhor.url
    ARQ_EXTRAS.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # merge em memória
    if ach.classe == "resolucao_tse":
        config.RESOLUCOES_TSE[ach.chave] = melhor.url
    elif ach.classe == "resolucao_camara":
        config.RESOLUCOES_CAMARA[ach.chave] = melhor.url
    elif ach.classe == "emenda":
        config.EMENDAS_CONSTITUCIONAIS[ach.chave] = melhor.url
    elif ach.classe == "lei":
        config.LEIS_EXTRAS[ach.chave] = melhor.url
    elif ach.classe == "lcp":
        config.LCPS_EXTRAS[ach.chave] = melhor.url
    return melhor.url


# ---------------------------------------------------------------------------
# análise de uma página do Notion (lacunas -> achados validados)
# ---------------------------------------------------------------------------
def analisar_pagina(url_notion: str, log: Callable[[str], None] = lambda m: None) -> List[Achado]:
    from .notion_api import fetch_page
    from .notion_parser import parse_blocks, flatten_blocks
    from .richtext import detectar_normas_sem_fonte

    log("Baixando a página e procurando normas sem fonte mapeada…")
    _pid, titulo, raw = fetch_page(url_notion)
    blocks = flatten_blocks(parse_blocks(raw))
    rich_lists = [b.rich for b in blocks if b.type != "quote" and b.rich]
    for b in blocks:
        for row in b.extra.get("rows", []):
            rich_lists.extend(row)
    lacunas = detectar_normas_sem_fonte(rich_lists)
    log(f"Página “{titulo}”: {len(lacunas)} norma(s) sem fonte." if lacunas
        else f"Página “{titulo}”: nenhuma lacuna — todas as normas citadas já linkam.")
    # aprendizado: links do AUTOR para resoluções da Câmara no LEGIN — a mesma
    # resolução citada COM link num trecho e SEM link em outro
    aprendidas: dict = {}
    for rich in rich_lists:
        for r in rich:
            m = re.search(
                r"legin/fed/rescad/(\d{4})/resolucaodacamaradosdeputados-(\d+)-",
                r.href or "",
            )
            if m:
                aprendidas[f"{int(m.group(2))}/{m.group(1)}"] = (r.href or "").split("#")[0]
    achados = []
    for lac in lacunas:
        log(f"— {lac}")
        ach = descobrir(lac, log, aprendidas)
        if ach is not None:
            achados.append(ach)
    return achados


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = list(argv if argv is not None else sys.argv[1:])
    auto = "--auto" in args
    if auto:
        args.remove("--auto")
    url_manual = None
    if "--url" in args:
        i = args.index("--url")
        url_manual = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--norma":
        ach = parsear_designacao(args[1]) if url_manual else None
        if url_manual:
            # URL informada manualmente: só valida e confirma (caminho para as
            # resoluções da Câmara, cujo slug do LEGIN não é derivável)
            if ach is None:
                print(f"designação não reconhecida: {args[1]!r}")
                return 2
            c = validar(Candidata(url_manual), ach.chave)
            ach.candidatas.append(c)
            print(f"  {'✓' if c.valida else '✗'} {url_manual}  ({c.obs})")
            achados = [ach]
        else:
            achados = [a for a in [descobrir(args[1], print)] if a is not None]
    else:
        achados = analisar_pagina(args[0], print)
    gravadas = 0
    for ach in achados:
        melhor = ach.melhor
        if melhor is None:
            print(f"✗ {ach.designacao}: nenhuma candidata validada — mapeie manualmente "
                  f"(config.py ou normas_extras.json).")
            continue
        aviso = "" if melhor.tem_ancoras else "  [ATENÇÃO: sem âncoras — o link abrirá no topo]"
        print(f"\n{ach.designacao}\n  → {melhor.url}{aviso}")
        if auto or input("  Aprovar e gravar? [s/N] ").strip().lower() in ("s", "sim"):
            gravar_achado(ach)
            gravadas += 1
            print("  ✓ gravada em normas_extras.json")
    print(f"\n{gravadas} fonte(s) aprovada(s). A próxima conversão já gera os links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
