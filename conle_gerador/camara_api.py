# -*- coding: utf-8 -*-
"""Cliente da API de Dados Abertos da Câmara dos Deputados (sem token).

Alimenta a Seção 6 (proposições correlatas) e o risco de apensação regimental.
Busca por `keywords` (filtro AND); detalha situação via `statusProposicao`.
"""
from __future__ import annotations

import datetime
import json
import re
import time
import unicodedata
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
        params["ano"] = list(range(ano_min, 2027))
    try:
        return _get("/proposicoes", params).get("dados", []) or []
    except Exception:
        return []


def _cache_ler(sigla: str, anos: List[int]) -> Optional[List[dict]]:
    cp = _CACHE_DIR / f"{sigla}_{min(anos)}_{max(anos)}.json"
    try:
        if cp.exists():
            d = json.loads(cp.read_text(encoding="utf-8"))
            if time.time() - d.get("ts", 0) < _CACHE_HORAS * 3600:
                return d.get("dados", [])
    except Exception:
        pass
    return None


def _cache_gravar(sigla: str, anos: List[int], dados: List[dict]) -> None:
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        cp = _CACHE_DIR / f"{sigla}_{min(anos)}_{max(anos)}.json"
        cp.write_text(json.dumps({"ts": time.time(), "dados": dados}, ensure_ascii=False),
                      encoding="utf-8")
    except Exception:
        pass


def listar_por_periodo(
    sigla: str, anos: List[int], *, cap: int = 4000, por_pagina: int = 100, usar_cache: bool = True,
) -> List[dict]:
    """Lista TODAS as proposições de uma espécie (siglaTipo) no período, paginando.
    É a base da busca por veículo normativo: o `keywords` da API não faz busca textual
    de ementa (índice de termos controlados, incompleto), então recuperamos a espécie
    inteira e filtramos a ementa localmente (estratégia confiável p/ risco de apensação).
    O resultado é cacheado em disco por meio dia (a varredura é o gargalo da Seção 6)."""
    if usar_cache:
        cached = _cache_ler(sigla, list(anos))
        if cached is not None:
            return cached
    out: List[dict] = []
    for ano in anos:
        pagina = 1
        while True:
            params = {"siglaTipo": sigla, "ano": ano, "itens": por_pagina,
                      "pagina": pagina, "ordem": "DESC", "ordenarPor": "id"}
            try:
                dd = _get("/proposicoes", params).get("dados", []) or []
            except Exception:
                dd = []
            out += dd
            if len(dd) < por_pagina or len(out) >= cap:
                break
            pagina += 1
        if len(out) >= cap:
            break
    if usar_cache and out:
        _cache_gravar(sigla, list(anos), out)
    return out


def detalhar(prop_id: int) -> dict:
    try:
        return _get(f"/proposicoes/{prop_id}").get("dados", {}) or {}
    except Exception:
        return {}


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
    for n in re.findall(r"\d[\d.]*", s):
        d = re.sub(r"\D", "", n)
        if not (len(d) == 4 and d[:2] in ("19", "20")):  # descarta o ano
            numero = d
            break
    if not numero:
        return []
    if int(numero) >= 1000:  # número distintivo basta (Lei 9.504, 8.429...)
        return [numero]
    return [m for m in (tipo, numero) if m]  # número baixo (LC 64) precisa do tipo


def _casa_marcadores(ementa: str, marc: List[str]) -> bool:
    e = _ementa_norm(ementa)
    return bool(marc) and all(re.search(rf"\b{re.escape(m)}\b", e) for m in marc)


def proposicoes_correlatas(
    palavras_chave: List[str],
    *,
    sigla: Optional[str] = None,
    norma_alvo: str = "",
    tema: str = "",
    max_total: int = 12,
    detalhar_situacao: bool = True,
    termos_relevancia: Optional[List[str]] = None,
    anos_lookback: int = 8,
    log: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Levanta as proposições CORRELATAS para a Seção 6 e o risco de apensação.

    Estratégia (espelha a pesquisa manual de um consultor): o risco de apensação mede-se
    pelo VEÍCULO NORMATIVO — proposições da MESMA espécie que alteram a MESMA norma-alvo —,
    não pela proximidade do recorte. Por isso recupera-se a espécie inteira no período e
    filtra-se a ementa localmente pelos marcadores da norma-alvo; a proximidade temática
    (termos do recorte) serve para ORDENAR (mais próximas no topo), nunca para descartar.
    """
    _log = log if callable(log) else (lambda _m: None)
    termos_tema = [_norm(t) for t in (termos_relevancia or palavras_chave or [])
                   if t and len(t.strip()) >= 3]
    marc = _marcadores_norma(norma_alvo)
    ano_atual = datetime.date.today().year
    anos = list(range(ano_atual, ano_atual - anos_lookback, -1))

    candidatos: Dict[int, dict] = {}
    # 1) veículo normativo: a espécie inteira no período, filtrando ementa pela norma-alvo
    if sigla and marc:
        _log(f"Câmara: varrendo {sigla} ({anos[-1]}–{anos[0]}) para alterações da norma-alvo…")
        for it in listar_por_periodo(sigla, anos):
            pid = it.get("id")
            if pid and pid not in candidatos:
                candidatos[pid] = it
    # 2) complemento por palavras-chave indexadas (pega o que a espécie/período não cobriu)
    for termo in [t for t in (palavras_chave or []) if t and t.strip()][:6]:
        for it in buscar_proposicoes(termo, sigla=sigla, n=15):
            pid = it.get("id")
            if pid and pid not in candidatos:
                candidatos[pid] = it

    # termos de PROXIMIDADE = recorte temático que DIFERENCIA. Descarta os termos "do instituto"
    # (presentes em quase toda ementa que altera a norma-alvo, ex.: "inelegibilidade"), pois não
    # distinguem as candidatas — sem isso, o desempate por ano expulsaria as do recorte (ambiental).
    casam = [it for it in candidatos.values() if _casa_marcadores(it.get("ementa") or "", marc)]
    n_casam = max(1, len(casam))
    _ements = {id(it): _ementa_norm(it.get("ementa") or "") for it in candidatos.values()}

    def _freq(t: str) -> int:
        return sum(1 for it in casam if re.search(rf"\b{re.escape(t)}\b", _ements[id(it)]))

    termos_prox = [t for t in set(termos_tema) if t and _freq(t) <= 0.45 * n_casam]

    scored = []
    for it in candidatos.values():
        e = _ements[id(it)]
        casa_norma = _casa_marcadores(it.get("ementa") or "", marc)
        nt = sum(1 for t in termos_prox if re.search(rf"\b{re.escape(t)}\b", e))
        score = (5.0 if casa_norma else 0.0) + nt
        if score > 0:  # correlata se altera a norma-alvo OU casa algum termo do recorte
            scored.append((score, it.get("ano") or 0, it))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    itens = [it for _, _, it in scored[:max_total]]
    _log(f"Câmara: {len(candidatos)} candidatas → {len(itens)} correlatas (norma-alvo + recorte).")

    out: List[Dict[str, Any]] = []
    for it in itens:
        pid = it.get("id")
        situacao, autor = "", ""
        if detalhar_situacao:
            d = detalhar(pid)
            situacao = ((d.get("statusProposicao") or {}).get("descricaoSituacao") or "").strip()
            aut = autores(pid)
            if aut:
                autor = (aut[0].get("nome") or "").strip()
            time.sleep(0.12)
        out.append({
            "id": pid,
            "sigla": it.get("siglaTipo"),
            "numero": it.get("numero"),
            "ano": it.get("ano"),
            "ementa": (it.get("ementa") or "").strip(),
            "situacao": situacao,
            "autor": autor,
            "url": f"https://www.camara.leg.br/proposicoesWeb/fichadetramitacao?idProposicao={pid}",
        })
    return out
