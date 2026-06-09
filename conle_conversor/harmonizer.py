# -*- coding: utf-8 -*-
"""Harmoniza a abertura da IT (após o vocativo) via OpenAI, com fallback robusto.

Gera três peças textuais que NÃO existem na página do Notion, mas são exigidas
pelo padrão da Informação Técnica da casa:
  - teor:          síntese curta da demanda (campo TEOR DA SOLICITAÇÃO no cabeçalho);
  - encaminhamento: 1º parágrafo ("Com relação à sua solicitação ...");
  - transicao:     parágrafo de transição após a transcrição da Resolução
                   ("No caso desta solicitação, ...").
O corpo técnico permanece fiel ao Notion; a IA atua só nesta abertura.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import config


@dataclass
class Abertura:
    teor: str
    encaminhamento: str
    transicao: str
    via_ia: bool


SYSTEM_PROMPT = """Você é Consultor Legislativo da Câmara dos Deputados e redige a ABERTURA \
de uma Informação Técnica (IT) da Consultoria Legislativa, no padrão exato da casa.

A IT abre com o vocativo "Senhor(a) Deputado(a)," e, em seguida, traz:
1) ENCAMINHAMENTO: um único parágrafo que começa por "Com relação à sua solicitação ..." e \
descreve a solicitação DE FORMA CONCRETA E ESPECÍFICA — nomeando o TIPO EXATO da proposição em \
anexo (use exatamente o "Tipo de proposição em anexo" informado: ex. "Projeto de Lei \
Complementar", "Proposta de Emenda à Constituição") e o OBJETO REAL do pedido (o que se cria ou \
altera, em qual norma, e com que finalidade). É PROIBIDO usar formulações genéricas como \
"elaboração de projeto de lei/proposta", "a presente proposta" ou "etc": adeque SEMPRE ao caso \
concreto desta IT. O parágrafo termina, obrigatoriamente, com a fórmula: "prestamos os seguintes \
esclarecimentos, em estrita observância às determinações da Resolução da Câmara dos Deputados \
nº 48, de 1993, que regula os trabalhos desta Consultoria Legislativa." Pode haver 1 ou 2 frases \
adicionais, após essa fórmula, detalhando a iniciativa.
2) TRANSIÇÃO: um único parágrafo, iniciado por "No caso desta solicitação," que anuncia, de \
forma objetiva e CONCRETA, o que a minuta em anexo propõe e o que a IT examinará.

Também produza o TEOR: uma síntese de 1 a 2 frases da DEMANDA ORIGINAL do parlamentar, CONCRETA \
(o tipo de proposição e o objeto específico do pedido, em linguagem direta), para o cabeçalho do \
documento — nunca genérica.

Estilo: português formal, técnico-jurídico, impessoal (1ª pessoa do plural: "prestamos", \
"examinamos"), coeso e conciso. NÃO use markdown. NÃO invente nomes de deputado, números de \
proposição ou datas. Responda SOMENTE com um objeto JSON válido com as chaves \
"teor", "encaminhamento" e "transicao"."""

FEWSHOT = """EXEMPLOS DE REFERÊNCIA (estilo a imitar):

[encaminhamento] Com relação à sua solicitação de elaboração de Projeto de Lei para instituir a \
chamada Lei de Responsabilidade Gerencial — proposta que visa condicionar repasses do Fundo \
Partidário e do Fundo Eleitoral a indicadores de desempenho e critérios objetivos de avaliação —, \
prestamos os seguintes esclarecimentos, em estrita observância às determinações da Resolução da \
Câmara dos Deputados nº 48, de 1993, que regula os trabalhos desta Consultoria Legislativa.

[transicao] No caso desta solicitação, o projeto de lei em anexo institui a Lei de \
Responsabilidade Gerencial para condicionar parcelas do Fundo Partidário e do Fundo Especial de \
Financiamento de Campanha ao desempenho de prefeituras geridas pelos partidos."""


def _fallback(objeto: str, introducao: str, tipo_extenso: str) -> Abertura:
    obj = " ".join((objeto or "").split())
    teor = obj[:400].rstrip(" .,;—-") + ("…" if len(obj) > 400 else "")
    # encaminhamento CONCRETO mesmo sem IA: nomeia o tipo exato e o objeto da solicitação
    tipo_low = (tipo_extenso or "proposição").lower()
    # usa o 1º eixo do objeto (antes do 1º travessão) e nunca corta no meio de uma palavra
    primeiro = re.split(r"\s+[—–]\s+", obj, maxsplit=1)[0]
    obj_curto = primeiro[:240]
    if len(primeiro) > 240:
        obj_curto = obj_curto.rsplit(" ", 1)[0]
    obj_curto = obj_curto.rstrip(" .,;—–-")
    if obj_curto:
        encaminhamento = (
            f"Com relação à sua solicitação de elaboração de {tipo_low} — {obj_curto} —, "
            "prestamos os seguintes esclarecimentos, em estrita observância às determinações da "
            "Resolução da Câmara dos Deputados nº 48, de 1993, que regula os trabalhos desta "
            "Consultoria Legislativa."
        )
    else:
        encaminhamento = (
            f"Com relação à sua solicitação de elaboração de {tipo_low}, prestamos os seguintes "
            "esclarecimentos, em estrita observância às determinações da Resolução da Câmara dos "
            "Deputados nº 48, de 1993, que regula os trabalhos desta Consultoria Legislativa."
        )
    primeira = ""
    if introducao:
        primeira = introducao.strip().split("\n")[0].split(". ")[0].strip()
    transicao = (
        f"No caso desta solicitação, examina-se a viabilidade jurídica da demanda e apresenta-se, "
        f"em anexo, a minuta de {tipo_extenso.lower()} pertinente."
    )
    if primeira:
        transicao = f"No caso desta solicitação, {primeira[0].lower() + primeira[1:]}."
    return Abertura(teor=teor or "—", encaminhamento=encaminhamento, transicao=transicao, via_ia=False)


def gerar_abertura(
    objeto: str,
    introducao: str,
    titulo: str,
    tipo_extenso: str,
    *,
    usar_ia: bool = True,
    model: str | None = None,
) -> Abertura:
    if not usar_ia:
        return _fallback(objeto, introducao, tipo_extenso)

    api_key = config.load_openai_key()
    if not api_key:
        return _fallback(objeto, introducao, tipo_extenso)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        user = (
            f"{FEWSHOT}\n\n"
            f"=== DADOS DESTA IT ===\n"
            f"Título da página: {titulo}\n\n"
            f"Tipo de proposição em anexo: {tipo_extenso}\n\n"
            f"Objeto/escopo do estudo:\n{objeto}\n\n"
            f"Trecho da introdução (contexto):\n{introducao[:2200]}\n\n"
            f"Gere o JSON com 'teor', 'encaminhamento' e 'transicao'."
        )
        resp = client.chat.completions.create(
            model=model or config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        teor = " ".join(str(data.get("teor", "")).split()).strip()
        enc = " ".join(str(data.get("encaminhamento", "")).split()).strip()
        tra = " ".join(str(data.get("transicao", "")).split()).strip()
        if not (enc and tra):
            return _fallback(objeto, introducao, tipo_extenso)
        return Abertura(teor=teor or objeto[:300], encaminhamento=enc, transicao=tra, via_ia=True)
    except Exception as exc:  # rede, cota, parsing...
        fb = _fallback(objeto, introducao, tipo_extenso)
        print(f"[harmonizer] IA indisponível ({exc}); usando fallback.")
        return fb
