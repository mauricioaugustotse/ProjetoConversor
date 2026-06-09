# -*- coding: utf-8 -*-
"""Pesquisa web ECONÔMICA via Gemini com grounding (Google Search).

Réplica enxuta do padrão usado no projeto JULES-IA
(tse_youtube_notion_core.call_gemini_generate_content_rest, use_google_search=True):
1 chamada grounded por consulta, modelo flash-lite, temperatura baixa, prompt curto.
Só `requests` — sem dependência nova. É a ÚNICA fonte de pesquisa web do gerador
(o GPT-5.5 nunca navega)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from . import config_gerador as cfg

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

SYSTEM_DEFAULT = (
    "Você é um pesquisador jurídico-legislativo. Use a busca para localizar legislação e "
    "jurisprudência ATUAIS e confiáveis, priorizando fontes oficiais (planalto.gov.br, "
    "stf.jus.br, tse.jus.br, camara.leg.br, senado.leg.br). Responda em português do Brasil, "
    "de forma OBJETIVA e CURTA: fatos verificáveis, números de normas/decisões e datas. "
    "Cite as fontes. NÃO invente; se não encontrar, diga que não encontrou."
)


def _extract_text(payload: Dict[str, Any]) -> str:
    partes: List[str] = []
    for cand in payload.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            t = part.get("text")
            if t:
                partes.append(str(t))
    return "\n".join(partes).strip()


def _extract_urls(payload: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    for cand in payload.get("candidates") or []:
        meta = cand.get("groundingMetadata") or {}
        for chunk in meta.get("groundingChunks") or []:
            uri = (chunk.get("web") or {}).get("uri")
            if uri and uri not in urls:
                urls.append(uri)
    return urls


def pesquisar(
    query: str,
    *,
    foco: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Retorna {"texto", "fontes", "ok", "erro"}. Degrada sem quebrar o fluxo."""
    key = cfg.load_gemini_key()
    if not key:
        return {"texto": "", "fontes": [], "ok": False,
                "erro": "Chave Gemini ausente (Chave_Gemini.txt / GEMINI_API_KEY)."}
    model = model or cfg.MODEL_GEMINI
    system = SYSTEM_DEFAULT + (f"\nFoco desta consulta: {foco}" if foco else "")
    payload = {
        "contents": [{"parts": [{"text": query}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        # NÃO definir responseMimeType: o grounding é desabilitado quando se força JSON.
        "generationConfig": {"temperature": 0.1},
        "tools": [{"googleSearch": {}}],
    }
    url = f"{GEMINI_BASE}/models/{model}:generateContent?key={key}"
    try:
        r = requests.post(url, json=payload, timeout=(10, timeout))
    except Exception as exc:  # noqa: BLE001
        return {"texto": "", "fontes": [], "ok": False, "erro": str(exc)}
    if r.status_code >= 400:
        return {"texto": "", "fontes": [], "ok": False,
                "erro": f"Gemini {r.status_code}: {r.text[:300]}"}
    data = r.json()
    texto = _extract_text(data)
    return {"texto": texto, "fontes": _extract_urls(data), "ok": bool(texto), "erro": ""}


def pesquisar_lote(
    perguntas: List[str], *, foco: Optional[str] = None, model: Optional[str] = None,
    timeout: int = 80,
) -> Dict[str, Any]:
    """Agrupa VÁRIAS perguntas numa ÚNICA chamada grounded. ECONOMIA: o grounding (Google
    Search) é cobrado por REQUISIÇÃO, não por pergunta — uma chamada com N perguntas custa
    como uma, não como N. Retorna {"texto", "fontes", "ok", "erro"}."""
    perguntas = [p.strip() for p in (perguntas or []) if p and p.strip()]
    if not perguntas:
        return {"texto": "", "fontes": [], "ok": False, "erro": "sem perguntas"}
    bloco = "\n".join(f"{i}. {p}" for i, p in enumerate(perguntas, 1))
    query = (
        "Responda às perguntas abaixo de forma OBJETIVA e CURTA, uma resposta NUMERADA por "
        "pergunta (use a busca para fatos atuais), citando normas/decisões com números e datas "
        "e as fontes oficiais. Se não encontrar algo, diga que não encontrou.\n\n" + bloco
    )
    return pesquisar(query, foco=foco, model=model, timeout=timeout)
