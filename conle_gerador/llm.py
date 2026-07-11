# -*- coding: utf-8 -*-
"""Wrapper OpenAI para REDAÇÃO/ANÁLISE e embeddings. NUNCA faz pesquisa web
(isso é papel exclusivo do gemini_web). Modelo padrão: cfg.MODEL_REDACAO (gpt-5.6-sol)."""
from __future__ import annotations

import json
import re
import time
from typing import List, Optional, Union

from . import config_gerador as cfg

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        key = cfg.load_openai_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY ausente (.env ou env). Necessária para a redação.")
        _client = OpenAI(api_key=key)
    return _client


def _parse_json(texto: str) -> dict:
    texto = (texto or "").strip()
    try:
        return json.loads(texto)
    except Exception:
        m = re.search(r"\{.*\}", texto, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def chat(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: int = 4,
    max_output_tokens: int = 16000,
) -> Union[str, dict]:
    """Uma chamada de redação via Responses API (suporta gpt-5.5 E gpt-5.5-pro;
    o chat/completions não aceita os modelos 'pro'). Robusto: retira temperature
    se o modelo não aceitar e tenta novamente em erros transitórios."""
    client = _get_client()
    model = model or cfg.MODEL_REDACAO
    # system vai como mensagem do input (e não em 'instructions') porque o modo
    # json_object da Responses API exige a palavra 'json' DENTRO do input.
    base = {"model": model,
            "input": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]}
    if json_mode:
        base["text"] = {"format": {"type": "json_object"}}

    temp = temperature
    tokens = max_output_tokens
    bump_feito = False
    last = None
    for i in range(max_retries):
        try:
            kwargs = dict(base)
            kwargs["max_output_tokens"] = tokens
            if temp is not None:
                kwargs["temperature"] = temp
            resp = client.responses.create(**kwargs)
            # truncamento por limite de tokens: a resposta vem 'incomplete' e o JSON sai cortado
            # (viraria seção vazia em silêncio) — re-chama JÁ com o dobro de tokens, sem gastar uma
            # iteração de retry (senão um bump na última iteração cairia fora do loop e lançaria erro).
            det = getattr(resp, "incomplete_details", None)
            reason = (getattr(det, "reason", "") or "") if det is not None else ""
            if getattr(resp, "status", "") == "incomplete" and "max_output" in reason and not bump_feito:
                bump_feito = True
                tokens = min(tokens * 2, 32000)
                kwargs["max_output_tokens"] = tokens
                resp = client.responses.create(**kwargs)
            content = getattr(resp, "output_text", "") or ""
            return _parse_json(content) if json_mode else content.strip()
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            # modelo recusa 'temperature' custom -> tenta sem
            if "temperature" in msg and temp is not None:
                temp = None
                continue
            # erros transitórios -> backoff
            if any(s in msg for s in ("rate limit", "429", "timeout", "overloaded",
                                      "503", "502", "500", "connection")) and i < max_retries - 1:
                time.sleep(2.0 * (i + 1))
                continue
            if i < max_retries - 1:
                time.sleep(1.5 * (i + 1))
                continue
            raise
    raise RuntimeError(f"Falha no chat OpenAI (responses): {last}")


def chat_visao(
    system: str,
    user: str,
    imagens_png: List[bytes],
    *,
    model: Optional[str] = None,
    max_retries: int = 3,
    max_output_tokens: int = 16000,
) -> str:
    """Chamada com IMAGENS (Responses API) — transcrição de PDFs escaneados e fotos
    de documentos anexados à demanda. Usa o modelo de redação (multimodal); o modelo
    econômico dos anexos serve só para condensação de texto."""
    import base64

    client = _get_client()
    model = model or cfg.MODEL_REDACAO
    content: List[dict] = [{"type": "input_text", "text": user}]
    for img in imagens_png:
        b64 = base64.b64encode(img).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
    base = {"model": model,
            "input": [{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens}
    last = None
    for i in range(max_retries):
        try:
            resp = client.responses.create(**base)
            return (getattr(resp, "output_text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < max_retries - 1:
                time.sleep(2.0 * (i + 1))
                continue
            raise
    raise RuntimeError(f"Falha no chat de visão OpenAI: {last}")


def embed(texts: List[str], *, model: Optional[str] = None, batch: int = 96) -> List[List[float]]:
    """Embeddings (text-embedding-3-small por padrão), em lotes."""
    client = _get_client()
    model = model or cfg.MODEL_EMBED
    out: List[List[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] if t else " " for t in texts[i:i + batch]]
        for tentativa in range(4):
            try:
                resp = client.embeddings.create(model=model, input=chunk)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as exc:  # noqa: BLE001
                if tentativa == 3:
                    raise
                time.sleep(2.0 * (tentativa + 1))
    return out
