# -*- coding: utf-8 -*-
"""Cliente da API de Dados Abertos da Câmara dos Deputados (sem token).

Alimenta a Seção 6 (proposições correlatas) e o risco de apensação regimental.
Busca por `keywords` (filtro AND); varredura por espécie com filtro local de
ementa (norma-alvo por marcadores; PEC por artigos da Constituição); detalha a
tramitação atual via `statusProposicao` (situação, órgão, regime, despacho —
do despacho extrai-se a proposição-mãe da apensação) e a árvore real de
apensadas/relacionadas via /proposicoes/{id}/relacionadas nas top correlatas.
"""
from __future__ import annotations

import datetime
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE = "https://dadosabertos.camara.leg.br/api/v2"
HEADERS = {"Accept": "application/json", "User-Agent": "CONLE-gerador/1.0"}
_CACHE_DIR = Path(__file__).resolve().parent / ".cache_camara"
_CACHE_HORAS = 12  # a varredura por espécie é cara (~1,5k itens); reusa por meio dia


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    return s.lower()


def _get(path: str, params: Optional[dict] = None) -> dict:
    for tentativa in range(4):
        try:
            r = requests.get(BASE + path, params=params or {}, headers=HEADERS, timeout=40)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.5 * (tentativa + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if tentativa == 3:
                raise
            time.sleep(1.2 * (tentativa + 1))
    return {}


def buscar_proposicoes(
    keywords: str,
    *,
    sigla: Optional[str] = None,
    n: int = 10,
    ano_min: Optional[int] = None,
) -> List[dict]:
    params: Dict[str, Any] = {
        "keywords": keywords,
        "itens": n,
        "ordem": "DESC",
        "ordenarPor": "id",
    }
    if sigla:
        params["siglaTipo"] = sigla
    if ano_min:
        params["ano"] = list(range(ano_min, datetime.date.today().year + 1))
    try:
        return _get("/proposicoes", params).get("dados", []) or []
    except Exception:
        return []


def _cache_path(sigla: str, anos: List[int], tag: str):
    return _CACHE_DIR / f"{sigla}_{min(anos)}_{max(anos)}_{tag}.json"


def _cache_ler(sigla: str, anos: List[int], tag: str) -> Optional[List[dict]]:
    cp = _cache_path(sigla, anos, tag)
    try:
        if cp.exists():
            d = json.loads(cp.read_text(encoding="utf-8"))
            if time.time() - d.get("ts", 0) < _CACHE_HORAS * 3600:
                return d.get("dados", [])
    except Exception:
        pass
    return None


def _cache_gravar(sigla: str, anos: List[int], dados: List[dict], tag: str) -> None:
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        _cache_path(sigla, anos, tag).write_text(
            json.dumps({"ts": time.time(), "dados": dados}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def listar_por_periodo(
    sigla: str, anos: List[int], *, marc: Optional[List[str]] = None,
    artigos: Optional[List[str]] = None,
    cap_por_ano: int = 6000, cap_total: int = 25000, por_pagina: int = 100, usar_cache: bool = True,
) -> List[dict]:
    """Lista as proposições de uma espécie (siglaTipo) no período, paginando.
    É a base da busca por veículo normativo: o `keywords` da API não faz busca textual de ementa
    (índice de termos controlados, incompleto), então recuperamos a espécie e filtramos a ementa
    localmente. Quando `marc` (marcadores da norma-alvo, AND) é dado, o filtro roda DENTRO da
    varredura, antes de acumular — assim só as proposições que alteram a norma entram em `out` e o
    teto não é gasto com páginas irrelevantes; sem `marc`, `artigos` (números de artigo, OR — o
    caso PEC) faz o mesmo papel. O teto é POR ANO (cap_por_ano), garantindo que cada ano do
    lookback seja visitado (antes, um cap global esgotava nos anos recentes e perdia o histórico).
    O resultado é cacheado por meio dia, com chave que inclui o filtro de norma/artigos."""
    anos = list(anos)
    tag = ("n" + "-".join(marc)) if marc else (("a" + "-".join(sorted(artigos))) if artigos else "all")
    if usar_cache:
        cached = _cache_ler(sigla, anos, tag)
        if cached is not None:
            return cached

    def _passa(it: dict) -> bool:
        e = it.get("ementa") or ""
        if marc:
            return _casa_marcadores(e, marc)
        if artigos:
            return _casa_artigos(e, artigos)
        return True

    out: List[dict] = []
    baixadas_total = 0  # teto GLOBAL de segurança: anos são DESC, então corta só os mais antigos
    for ano in anos:
        pagina, baixadas_ano = 1, 0
        while True:
            params = {"siglaTipo": sigla, "ano": ano, "itens": por_pagina,
                      "pagina": pagina, "ordem": "DESC", "ordenarPor": "id"}
            try:
                bruto = _get("/proposicoes", params).get("dados", []) or []
            except Exception:
                bruto = []
            baixadas_ano += len(bruto)
            baixadas_total += len(bruto)
            out += [it for it in bruto if _passa(it)]
            if len(bruto) < por_pagina or baixadas_ano >= cap_por_ano:
                break
            pagina += 1
        if baixadas_total >= cap_total:  # protege contra explosão em espécies volumosas (PL)
            break
    # grava no cache MESMO se vazio (cache negativo): norma sem correlatas não deve re-varrer a
    # espécie inteira a cada execução; o TTL de 12h limita o risco de cachear um vazio transitório.
    if usar_cache:
        _cache_gravar(sigla, anos, out, tag)
    return out


def detalhar(prop_id: int) -> dict:
    try:
        return _get(f"/proposicoes/{prop_id}").get("dados", {}) or {}
    except Exception:
        return {}


def status_resumo(detalhe: dict) -> dict:
    """Resumo da tramitação ATUAL a partir do statusProposicao já retornado por
    detalhar() — 0 chamadas extras: situação, órgão, regime, forma de apreciação,
    despacho (de onde se extrai a apensação) e data (AAAA-MM-DD)."""
    st = (detalhe or {}).get("statusProposicao") or {}
    return {
        "situacao": (st.get("descricaoSituacao") or "").strip(),
        "orgao": (st.get("siglaOrgao") or "").strip(),
        "regime": (st.get("regime") or "").strip(),
        "apreciacao": (st.get("apreciacao") or "").strip(),
        "despacho": (st.get("despacho") or "").strip(),
        "data": (st.get("dataHora") or "")[:10],
    }


# Formas reais de despacho de apensação (calibradas na API em 05/07/2026):
#   "Apense-se à(ao) PL-1234/2020."  |  "Apensação desta proposição ao PL 4250/2021."
#   "apensado ao PL X/AAAA"          |  NÃO casa "apense-se a este(a)" (movimento inverso)
#   e NÃO casa "Desapensação..." (guarda (?<!des)).
_RE_APENSE = re.compile(
    r"(?<!des)apens(?:e(?:m)?-se(?!\s+a\s+est)|a[cç][aã]o\s+desta\s+proposi[cç][aã]o|ad[oa])"
    r"[^.;]{0,40}?\b(PEC|PLP|PDL|PDC|PRC|PL)\s*[-.\s]*n?[ºo°.]*\s*"
    r"([\d.]+)\s*[/\s]\s*((?:19|20)\d{2})",
    re.IGNORECASE)


def _apensada_a(despacho: str) -> str:
    """Extrai do despacho a proposição a que esta foi apensada ('' se não houver)."""
    m = _RE_APENSE.search(despacho or "")
    if not m:
        return ""
    return f"{m.group(1).upper()} {re.sub(r'[^0-9]', '', m.group(2))}/{m.group(3)}"


def autores(prop_id: int) -> List[dict]:
    try:
        return _get(f"/proposicoes/{prop_id}/autores").get("dados", []) or []
    except Exception:
        return []


def relacionadas(prop_id: int) -> List[dict]:
    try:
        return _get(f"/proposicoes/{prop_id}/relacionadas").get("dados", []) or []
    except Exception:
        return []


def _ementa_norm(e: str) -> str:
    """Ementa sem acento, minúscula e com números compactados (9.504 -> 9504)."""
    return re.sub(r"(?<=\d)\.(?=\d)", "", _norm(e))


def _marcadores_norma(norma_alvo: str) -> List[str]:
    """Marcadores que identificam, na ementa, uma proposição que altera a NORMA-ALVO.
    Ex.: "Lei Complementar nº 64/1990" -> ["complementar", "64"]; "Lei nº 9.504/1997" ->
    ["9504"] (número distintivo dispensa o tipo). Constituição é fraca (toda PEC a altera),
    então não vira marcador — para PEC a proximidade vem dos termos temáticos."""
    s = _norm(norma_alvo or "")
    if not s or "constituic" in s:
        return []
    tipo = "complementar" if "complementar" in s else ("lei" if "lei" in s else None)
    numero = None
    # 1) o número costuma PRECEDER o ano ("<número>/<ano>" ou "<número>-<ano>"): isso desambigua
    #    leis cujo número cai na faixa de ano (ex.: Lei nº 2.000/2010 -> número 2000, não ano)
    m = re.search(r"(\d[\d.]*)\s*[/-]\s*(?:19|20)\d{2}\b", s)
    if m:
        numero = re.sub(r"\D", "", m.group(1))
    else:  # 2) fallback: primeiro token que não pareça um ano isolado
        for n in re.findall(r"\d[\d.]*", s):
            d = re.sub(r"\D", "", n)
            if not (len(d) == 4 and d[:2] in ("19", "20")):
                numero = d
                break
    if not numero:
        return []
    if int(numero) >= 1000:  # número distintivo basta (Lei 9.504, 8.429...)
        return [numero]
    return [x for x in (tipo, numero) if x]  # número baixo (LC 64) precisa do tipo


def _casa_marcadores(ementa: str, marc: List[str]) -> bool:
    e = _ementa_norm(ementa)
    return bool(marc) and all(re.search(rf"\b{re.escape(m)}\b", e) for m in marc)


def _artigos_alvo(dispositivos_alvo: Optional[List[str]]) -> List[str]:
    """Números de artigo dos dispositivos-alvo: ['art. 14, §9º', 'art. 77'] -> ['14', '77'].
    É o filtro do caso PEC: a Constituição não gera marcador (toda PEC a altera), mas a
    ementa de PEC costuma nomear o artigo alterado ('Altera o art. 14 da Constituição...')."""
    nums: List[str] = []
    for d in dispositivos_alvo or []:
        for m in re.finditer(r"art\w*\.?\s*(\d+)", _norm(str(d))):
            if m.group(1) not in nums:
                nums.append(m.group(1))
    return nums


def _casa_artigos(ementa: str, artigos: List[str]) -> bool:
    """OR (diferente do AND dos marcadores): tocar QUALQUER artigo-alvo já é correlata."""
    if not artigos:
        return False
    e = _ementa_norm(ementa)
    return any(re.search(rf"\bart(?:igo)?s?\.?\s*{a}\b", e) for a in artigos)


def proposicoes_correlatas(
    palavras_chave: List[str],
    *,
    sigla: Optional[str] = None,
    norma_alvo: str = "",
    tema: str = "",
    dispositivos_alvo: Optional[List[str]] = None,
    max_total: int = 20,
    detalhar_situacao: bool = True,
    termos_relevancia: Optional[List[str]] = None,
    anos_lookback: int = 8,
    relacionadas_top: int = 5,
    workers: int = 4,
    log: Optional[Any] = None,
) -> Dict[str, Any]:
    """Levanta as proposições CORRELATAS para a Seção 6 e o risco de apensação.

    Estratégia (espelha a pesquisa manual de um consultor): o risco de apensação mede-se
    pelo VEÍCULO NORMATIVO — proposições da MESMA espécie que alteram a MESMA norma-alvo —,
    não pela proximidade do recorte. Por isso recupera-se a espécie inteira no período e
    filtra-se a ementa localmente pelos marcadores da norma-alvo (para PEC, pelos números
    de artigo da Constituição em `dispositivos_alvo`); a proximidade temática (termos do
    recorte) serve para ORDENAR (mais próximas no topo), nunca para descartar.

    -> {"itens": [...], "resumo": {...}}. Cada item traz a tramitação atual (situação,
    órgão, regime, data), a apensação extraída do despacho e, nas `relacionadas_top` de
    maior score, o tamanho da árvore real de relacionadas. O `resumo` agrega contagens
    para a abertura quantificada da Seção 6 — ATENÇÃO: `por_situacao_listadas` cobre
    APENAS as proposições listadas/detalhadas, não o universo.
    """
    _log = log if callable(log) else (lambda _m: None)
    termos_tema = [_norm(t) for t in (termos_relevancia or palavras_chave or [])
                   if t and len(t.strip()) >= 3]
    marc = _marcadores_norma(norma_alvo)
    # caso PEC: a Constituição não gera marcador — o veículo é identificado pelos artigos-alvo
    artigos: List[str] = []
    if not marc and (sigla == "PEC" or "constituic" in _norm(norma_alvo or "")):
        artigos = _artigos_alvo(dispositivos_alvo)
        if sigla == "PEC" and not artigos:
            _log("Câmara: PEC sem dispositivos-alvo — varredura por espécie pulada (só keywords).")
    ano_atual = datetime.date.today().year
    anos = list(range(ano_atual, ano_atual - anos_lookback, -1))

    candidatos: Dict[int, dict] = {}
    # 1) veículo normativo: a espécie inteira no período, filtrando ementa pela norma/artigos-alvo
    if sigla and (marc or artigos):
        alvo_txt = "norma-alvo" if marc else f"artigos-alvo ({', '.join(artigos)})"
        _log(f"Câmara: varrendo {sigla} ({anos[-1]}–{anos[0]}) para alterações da {alvo_txt}…")
        for it in listar_por_periodo(sigla, anos, marc=marc or None, artigos=artigos or None):
            pid = it.get("id")
            if pid and pid not in candidatos:
                candidatos[pid] = it
    # 2) complemento por palavras-chave indexadas (pega o que a espécie/período não cobriu)
    for termo in [t for t in (palavras_chave or []) if t and t.strip()][:6]:
        for it in buscar_proposicoes(termo, sigla=sigla, n=15):
            pid = it.get("id")
            if pid and pid not in candidatos:
                candidatos[pid] = it

    def _casa_alvo(ementa: str) -> bool:
        return _casa_marcadores(ementa, marc) or _casa_artigos(ementa, artigos)

    # termos de PROXIMIDADE = recorte temático que DIFERENCIA. Descarta os termos "do instituto"
    # (presentes em quase toda ementa que altera a norma-alvo, ex.: "inelegibilidade"), pois não
    # distinguem as candidatas — sem isso, o desempate por ano expulsaria as do recorte (ambiental).
    casam = [it for it in candidatos.values() if _casa_alvo(it.get("ementa") or "")]
    n_casam = max(1, len(casam))
    # indexa por id da proposição (pid), não por id() do objeto (à prova de refatoração/GC)
    _ements = {it.get("id"): _ementa_norm(it.get("ementa") or "") for it in candidatos.values()}

    def _freq(t: str) -> int:
        return sum(1 for it in casam if re.search(rf"\b{re.escape(t)}\b", _ements[it.get("id")]))

    termos_prox = [t for t in set(termos_tema) if t and _freq(t) <= 0.45 * n_casam]

    scored = []
    for it in candidatos.values():
        e = _ements[it.get("id")]
        casa_norma = _casa_alvo(it.get("ementa") or "")
        nt = sum(1 for t in termos_prox if re.search(rf"\b{re.escape(t)}\b", e))
        score = (5.0 if casa_norma else 0.0) + nt
        if score > 0:  # correlata se altera a norma-alvo OU casa algum termo do recorte
            scored.append((score, it.get("ano") or 0, it))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    itens = [it for _, _, it in scored[:max_total]]
    _log(f"Câmara: {len(candidatos)} candidatas → {len(itens)} correlatas (norma-alvo + recorte).")

    # detalhamento PARALELO (2-3 chamadas/item; a cortesia vem do teto de workers)
    top_rel = {it.get("id") for it in itens[:relacionadas_top]}

    def _detalhe(it: dict) -> Dict[str, Any]:
        pid = it.get("id")
        item: Dict[str, Any] = {
            "id": pid,
            "sigla": it.get("siglaTipo"),
            "numero": it.get("numero"),
            "ano": it.get("ano"),
            "ementa": (it.get("ementa") or "").strip(),
            "situacao": "", "autor": "", "n_autores": 0,
            "orgao": "", "regime": "", "apreciacao": "", "data_situacao": "",
            "apensada_a": "", "n_relacionadas": None,
            "url": f"https://www.camara.leg.br/proposicoesWeb/fichadetramitacao?idProposicao={pid}",
        }
        if detalhar_situacao:
            st = status_resumo(detalhar(pid))
            item.update({"situacao": st["situacao"], "orgao": st["orgao"],
                         "regime": st["regime"], "apreciacao": st["apreciacao"],
                         "data_situacao": st["data"], "apensada_a": _apensada_a(st["despacho"])})
            aut = autores(pid)
            if aut:
                item["autor"] = (aut[0].get("nome") or "").strip()
                item["n_autores"] = len(aut)
            if pid in top_rel:  # árvore real de apensadas/relacionadas só nas top (economia)
                item["n_relacionadas"] = len(relacionadas(pid))
        return item

    if itens:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_detalhe, itens))  # preserva a ordem por score
    else:
        out = []

    # resumo agregado (dados observados p/ a abertura quantificada da Seção 6)
    por_ano: Dict[int, int] = {}
    for it in casam:
        a = int(it.get("ano") or 0)
        por_ano[a] = por_ano.get(a, 0) + 1
    por_situacao: Dict[str, int] = {}
    apensadas = 0
    if detalhar_situacao:
        for item in out:
            s = item["situacao"] or "(sem situação informada)"
            por_situacao[s] = por_situacao.get(s, 0) + 1
            if item["apensada_a"]:
                apensadas += 1
    resumo = {
        "periodo": f"{anos[-1]}–{anos[0]}",
        "total_candidatas": len(candidatos),
        "total_norma_alvo": len(casam),
        "por_ano_norma_alvo": {a: n for a, n in sorted(por_ano.items()) if a},
        "listadas": len(out),
        "por_situacao_listadas": por_situacao,
        "apensadas_listadas": apensadas,
    }
    return {"itens": out, "resumo": resumo}
