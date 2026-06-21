# -*- coding: utf-8 -*-
"""Configuração do gerador: modelos, bases RAG e chaves (reusa conle_conversor.config)."""
from __future__ import annotations

import os
from pathlib import Path

from conle_conversor import config as _cfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
RAG_CACHE_DIR = PACKAGE_DIR / ".rag_cache"

NOTION_VERSION = "2022-06-28"

# ---------------------------------------------------------------- modelos
MODEL_REDACAO = "gpt-5.5"            # redação/análise (OpenAI) — nunca faz web
MODEL_REDACAO_PRO = "gpt-5.5-pro"   # opção de alta qualidade
MODEL_GEMINI = "gemini-3.1-flash-lite"   # pesquisa web grounded (econômico)
MODEL_EMBED = "text-embedding-3-small"   # RAG

# ---------------------------------------------------------------- chaves
# Reusa os loaders do conversor para Notion e OpenAI.
load_notion_token = _cfg.load_notion_token
load_openai_key = _cfg.load_openai_key

# Caminho do projeto-irmão que já guarda a chave do Gemini.
JULES_DIR = Path(r"C:\Users\mauri\JULES-IA")


def load_gemini_key() -> str:
    """Env GEMINI_API_KEY/GOOGLE_API_KEY → Chave_Gemini.txt local → JULES-IA."""
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    candidatos = [
        PROJECT_ROOT / "Chave_Gemini.txt",
        PROJECT_ROOT / "Chave_Google_API.txt",
        JULES_DIR / "Chave_Gemini.txt",
        JULES_DIR / "Chave_Google_API.txt",
    ]
    for p in candidatos:
        if p.is_file():
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    s = p.read_text(encoding=enc).strip()
                    if s:
                        return s
                except UnicodeDecodeError:
                    continue
    return ""


# ---------------------------------------------------------------- bases RAG
# id=None => o notion_rag resolve pelo título exato (via /v1/search), excluindo BACKUP.
# props_texto=None => auto-detecta (todas as propriedades title/rich_text do database).
BASES_RAG = {
    # Em todas, `texto_rag` é a prop consolidada para retrieval (preparada pelo usuário);
    # o notion_rag a prioriza e só cai nas demais se ela estiver vazia.
    "vademecum": {
        "id": "43dc0feb-e1ea-447f-a5fa-8c9b21670ce7",
        "titulo": "Vademecum - RAG consolidado",
        "label": "Vademécum",
        "categoria": "normativo",
        # "conteúdo primeiro": o texto distintivo do dispositivo vai ao início do embedding;
        # contexto (norma/hierarquia) ao fim. props_texto é fallback se faltar conteúdo.
        "props_conteudo": ["dispositivo", "texto_dispositivo", "resumo_curto", "notas_texto", "palavras_chave"],
        "props_contexto": ["norma_nome_popular", "hierarquia_normativa"],
        "props_texto": ["texto_rag", "norma_titulo", "hierarquia_normativa",
                         "texto_dispositivo", "palavras_chave"],
        "flag_rag": "incluir_no_rag",
    },
    "resolucoes_tse": {
        "id": "6eff337f-d025-4513-b5a3-003ff34b419d",
        "titulo": "Resolucoes TSE - RAG consolidado",
        "label": "Resoluções TSE",
        "categoria": "normativo",
        "props_conteudo": ["dispositivo", "texto_em_vigor", "texto_dispositivo", "resumo_curto", "palavras_chave"],
        "props_contexto": ["norma_nome_popular", "hierarquia_normativa"],
        "props_texto": ["texto_rag", "norma_titulo", "hierarquia_normativa", "texto_dispositivo"],
        "flag_rag": "incluir_no_rag",
    },
    "codigo_eleitoral": {
        "id": "66da171f-0acd-42af-8875-0860265f5d23",
        "titulo": "Codigo Eleitoral 2026 anotado - dispositivos",
        "label": "Código Eleitoral 2026 anotado",
        "categoria": "normativo",
        "props_conteudo": ["dispositivo", "texto_dispositivo", "resumo_curto", "palavras_chave"],
        "props_contexto": ["norma_nome_popular", "hierarquia_normativa"],
        "props_texto": ["texto_rag", "norma_titulo", "hierarquia_normativa", "texto_dispositivo"],
        "flag_rag": "incluir_no_rag",
    },
    # Jurisprudência do TSE estruturada: cada linha tem link oficial (link_1) + análise rica.
    "temas": {
        "id": "30a72195-5c64-80a9-b5f1-c958107d9d0e",
        "titulo": "temas",
        "label": "Jurisprudência TSE (temas)",
        "categoria": "jurisprudencia",
        "props_conteudo": ["tema", "tese", "punchline", "texto_original", "contexto"],
        "props_contexto": ["relator", "ramo", "subramo"],
        "props_texto": ["tema", "tese", "punchline", "contexto", "texto_original", "relator"],
        "flag_rag": None,
    },
    # Plenário do TSE (deliberações analisadas) — só os campos JURÍDICOS vão ao embedding;
    # notícias/advogados/partes ficam de fora (ruído).
    "sess_es": {
        "id": None,                 # resolvido por título "sessões" (evita ambiguidade data_source x database id)
        "titulo": "sessões",
        "label": "Sessões do plenário (TSE)",
        "categoria": "jurisprudencia",
        "props_conteudo": ["tema", "punchline", "analise_do_conteudo_juridico", "raciocinio_juridico",
                            "fundamentacao_normativa", "precedentes_citados"],
        "props_contexto": ["resultado", "votacao", "relator", "classe_processo", "numero_processo"],
        "props_texto": None,
        "flag_rag": None,
    },
    "dje": {
        "id": None,                 # resolvido por título
        "titulo": "DJe",
        "label": "DJe (jurisprudência)",
        "categoria": "jurisprudencia",
        "props_conteudo": ["tema", "punchline", "textoEmenta", "referenciasLegislativas"],
        "props_contexto": ["relator", "siglaClasse", "numeroProcesso"],
        "props_texto": None,
        "flag_rag": None,
    },
    # Biblioteca CONLE "Livros e artigos" — obras limpas/estruturadas (interseccionalidade,
    # Cartilha AGU 2024, TSE perfil racial, NT 30 anos de cotas, Eleições 2026/Senado, Entre Nós
    # e o relatório IDP/LIA — todos CONSOLIDADOS aqui), 1 linha por chunk + coluna "obra".
    # (O database separado "LIA Integridade - RAG" foi arquivado; a base "lia" foi removida.)
    "livros": {
        "id": "38572195-5c64-819a-80aa-c7d58de8bef2",
        "titulo": "Livros e artigos - RAG",
        "label": "Livros e artigos (biblioteca)",
        "categoria": "doutrina",
        "props_conteudo": ["subsecao", "texto_rag"],
        "props_contexto": ["obra", "secao", "artigos", "jurisdicoes"],
        "props_texto": ["texto_rag", "obra", "secao", "subsecao", "artigos"],
        "flag_rag": "incluir_no_rag",
    },
}

# Bases ligadas por padrão na GUI (DJe fica opcional; sessões agora incluída, curada).
BASES_PADRAO = ["vademecum", "resolucoes_tse", "codigo_eleitoral", "temas", "sess_es", "livros"]

# Nunca indexar/consultar bases cujo título contenha estes termos.
EXCLUIR_TITULOS = ("BACKUP",)

# Databases de RESOLUÇÃO individual ("RESOLUÇÃO Nº 23.xxx ...") já estão consolidadas
# na base "Resoluções TSE - RAG consolidado" — escondê-las da descoberta (redundantes).
# O padrão singular não casa "Resolucoes TSE - RAG consolidado" (plural).
import re as _re_excl
EXCLUIR_REGEX = _re_excl.compile(r"^\s*resolu[çc][ãa]o\s+n", _re_excl.IGNORECASE)


# ---------------------------------------------------------------- descoberta dinâmica
import re as _re

_BASES_CACHE = None


def _slug(titulo: str) -> str:
    s = _re.sub(r"[^a-z0-9]+", "_", (titulo or "").lower()).strip("_")
    return s[:40] or "base"


def descobrir_bases(forcar: bool = False) -> dict:
    """Lista TODAS as databases acessíveis (exceto BACKUP) e combina com as bases
    conhecidas (que têm props boas). Resolve o id das conhecidas sem id (ex.: DJe).
    Bases novas entram com props_texto=None (auto-detecção no notion_rag)."""
    global _BASES_CACHE
    if _BASES_CACHE is not None and not forcar:
        return _BASES_CACHE
    import json as _json
    import requests as _rq

    reg = {k: dict(v) for k, v in BASES_RAG.items()}
    try:
        H = {"Authorization": f"Bearer {load_notion_token()}",
             "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}
        r = _rq.post("https://api.notion.com/v1/search", headers=H,
                     data=_json.dumps({"filter": {"property": "object", "value": "database"},
                                       "page_size": 100}), timeout=40)
        dbs = r.json().get("results", [])
    except Exception:  # noqa: BLE001
        _BASES_CACHE = reg
        return reg

    known_ids = {v["id"] for v in reg.values() if v.get("id")}
    for o in dbs:
        titulo = "".join(x.get("plain_text", "") for x in o.get("title", []) or []) or "(sem título)"
        if any(ex.lower() in titulo.lower() for ex in EXCLUIR_TITULOS):
            continue
        if EXCLUIR_REGEX.match(titulo):  # resoluções individuais já consolidadas
            continue
        dbid = o["id"]
        if dbid in known_ids:
            continue
        # resolve o id de uma base conhecida sem id (ex.: DJe por título)
        casou = False
        for b in reg.values():
            if not b.get("id") and b.get("titulo") and b["titulo"].strip().lower() == titulo.strip().lower():
                b["id"] = dbid
                known_ids.add(dbid)
                casou = True
                break
        if casou:
            continue
        chave = _slug(titulo)
        if chave in reg:
            chave = chave + "_" + dbid[:4]
        # base NOVA/desconhecida: opt-in manual — não entra com flag automática nem no padrão,
        # e fica fora da allowlist jurídica (não polui o top-k do redator sem decisão explícita).
        reg[chave] = {"id": dbid, "titulo": titulo, "label": titulo[:42],
                      "categoria": "desconhecida", "props_texto": None, "flag_rag": None}
    _BASES_CACHE = reg
    return reg


def bases_registry() -> dict:
    return descobrir_bases()
