# -*- coding: utf-8 -*-
"""Cliente da API de Dados Abertos da Câmara dos Deputados (sem token).

Alimenta a Seção 6 (proposições correlatas) e o risco de apensação regimental.
Busca por `keywords` (filtro AND); detalha situação via `statusProposicao`.
"""
from __future__ import annotations

import time
import unicodedata
from typing import Any, Dict, List, Optional

import requests

BASE = "https://dadosabertos.camara.leg.br/api/v2"
HEADERS = {"Accept": "application/json", "User-Agent": "CONLE-gerador/1.0"}


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


def proposicoes_correlatas(
    palavras_chave: List[str],
    *,
    sigla: Optional[str] = None,
    max_total: int = 12,
    detalhar_situacao: bool = True,
    termos_relevancia: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Busca por vários termos ATÔMICOS (a API trata `keywords` como AND; termos de
    uma palavra rendem melhor), deduplica por id, filtra por relevância de ementa,
    ordena por ano desc e (opcional) detalha a situação de tramitação.

    Retorna dicts: id, sigla, numero, ano, ementa, situacao, autor, url.
    """
    vistos: Dict[int, dict] = {}
    termos = [t.strip() for t in palavras_chave if t and t.strip()][:8]
    # 1) busca FOCADA na espécie da proposta (ex.: só PEC) — melhora o risco de apensação
    for termo in termos:
        for it in buscar_proposicoes(termo, sigla=sigla, n=12):
            pid = it.get("id")
            if pid and pid not in vistos:
                vistos[pid] = it
    # 2) se a espécie rendeu pouco, amplia sem filtro de tipo (sem custo — API gratuita)
    if sigla and len(vistos) < 6:
        for termo in termos:
            for it in buscar_proposicoes(termo, sigla=None, n=12):
                pid = it.get("id")
                if pid and pid not in vistos:
                    vistos[pid] = it

    # filtro de relevância: ementa deve conter algum termo (reduz ruído da busca AND)
    rel = [_norm(t) for t in (termos_relevancia or palavras_chave) if t and t.strip()]
    if rel:
        # exige relevância na ementa; se nada casar, retorna VAZIO (melhor que listar ruído —
        # a Seção 6 dirá que não há correlatas) em vez de despejar tudo.
        base = [it for it in vistos.values()
                if any(t in _norm(it.get("ementa") or "") for t in rel)]
    else:
        base = list(vistos.values())
    itens = sorted(base, key=lambda x: (x.get("ano") or 0), reverse=True)[:max_total]

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
            time.sleep(0.15)
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
