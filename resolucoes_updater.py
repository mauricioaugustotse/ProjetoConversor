# -*- coding: utf-8 -*-
"""Atualizador da base Notion 'resoluções' (Resolucoes TSE - RAG consolidado).

Fluxo: raspa os índices anuais de https://www.tse.jus.br/legislacao/compilada/res
→ CATÁLOGO: 1 linha sintética por resolução (número, data, ementa, URL) para todo
o acervo → TEXTO INTEGRAL: para as resoluções monitoradas (resolucoes_work/
monitoradas.json), baixa a página compilada, parseia dispositivo a dispositivo
(_res_parser: redações tachadas descartadas, notas "(Redação dada/Incluído/
Revogado pela Resolução nº X)" viram status_alteracao) e sincroniza por row_key
com hash local (create/patch/archive) → enriquece via gpt-5.6-luna (nível norma +
dispositivos alterados) → opcionalmente reindexa o RAG.

Idempotente: manifest resolucoes_work/inseridos.jsonl + hash-store hashes.json.
Dry-run gera novidades_<data>.csv / alteracoes_<data>.csv sem tocar no Notion.

  python resolucoes_updater.py                          # dry-run (ano atual + anterior)
  python resolucoes_updater.py --varredura-completa     # dry-run, todos os anos
  python resolucoes_updater.py --aplicar --varredura-completa   # 1ª carga do catálogo
  python resolucoes_updater.py --aplicar --escopo integrais     # ingere monitoradas
  python resolucoes_updater.py --aplicar --limite 20            # teste fim-a-fim
  python resolucoes_updater.py --arquivar-antigas               # fase única (migração)
  python resolucoes_updater.py --relacionar                     # relation pai<-filho
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import _res_lib as L
import _res_parser as P
import _res_scraper as S

MONITORADAS_PATH = L.WORK_DIR / "monitoradas.json"
HASHES_PATH = L.WORK_DIR / "hashes.json"
MANIFEST = L.WORK_DIR / "inseridos.jsonl"
ARQUIVADAS = L.WORK_DIR / "arquivadas.jsonl"
CACHE_IA = L.WORK_DIR / "updater_cache.jsonl"
MODELO_PADRAO = "gpt-5.6-luna"
PROMPT_VERSION = "v1"

# seed: as 12 resoluções estruturantes já mantidas na base
MONITORADAS_SEED = {
    "23444": "2015", "23605": "2019", "23607": "2019", "23608": "2019",
    "23609": "2019", "23610": "2019", "23673": "2021", "23677": "2021",
    "23735": "2024", "23750": "2026", "23751": "2026", "23760": "2026",
}
NOMES_POPULARES = {
    "23444": "Teste Público de Segurança (TPS)",
    "23605": "FEFC",
    "23607": "Prestação de contas",
    "23608": "Representações, reclamações e direito de resposta",
    "23609": "Registro de candidatura",
    "23610": "Propaganda eleitoral",
    "23673": "Fiscalização e auditoria do sistema eletrônico de votação",
    "23677": "Sistemas eleitorais majoritário e proporcional",
    "23735": "Ilícitos eleitorais",
    "23750": "Cronograma operacional do Cadastro Eleitoral 2026",
    "23751": "Atos gerais do processo eleitoral 2026",
    "23760": "Calendário Eleitoral 2026",
}
ALTERA_RE = re.compile(r"Altera\s+a\s+Resolu[çc][ãa]o(?:-TSE)?\s+n[ºo°.]*\s*([\d.]+)",
                       re.IGNORECASE)


def fmt_numero(num_compacto: str) -> str:
    """'23669' -> '23.669'; '9235' -> '9.235'."""
    return f"{num_compacto[:-3]}.{num_compacto[-3:]}" if len(num_compacto) > 3 else num_compacto


def parse_ref_resolucao(texto: str) -> tuple:
    """Referência de resolução -> (numero_compacto, ano).

    Aceita 'nº/ano' (23.669/2022) OU o link da página compilada no site do TSE
    (https://www.tse.jus.br/legislacao/compilada/res/{ano}/{slug}). Levanta
    ValueError com mensagem amigável se não reconhecer.
    """
    bruto = (texto or "").strip()
    m = re.search(r"tse\.jus\.br/legislacao/compilada/res/((?:19|20)\d{2})/([^/?#\s]+)",
                  bruto, re.IGNORECASE)
    if m:
        ano, slug = m.group(1), m.group(2).strip("/")
        num = re.sub(r"\D", "", S._numero("", slug))
        if not 4 <= len(num) <= 5:
            raise ValueError(f"não consegui extrair o número da resolução do link "
                             f"(slug: {slug}) — informe no formato nº/ano, ex.: 23.669/2022")
        return num, ano
    m = re.fullmatch(r"(\d{1,2}\.?\d{3})\s*/\s*((?:19|20)\d{2})", bruto)
    if m:
        return re.sub(r"\D", "", m.group(1)), m.group(2)
    raise ValueError("informe nº/ano (ex.: 23.669/2022) ou o link da resolução em "
                     "https://www.tse.jus.br/legislacao/compilada/res")


def localizar_no_indice(num: str, ano: str, progress=print) -> Optional[dict]:
    """Procura a resolução no índice anual do site (cache diário) -> item ou None."""
    try:
        itens = S.republicacoes(S.listar_ano(int(ano), progress=progress), progress=progress)
    except Exception as e:  # noqa: BLE001
        progress(f"  AVISO: falha ao consultar o índice {ano}: {str(e)[:120]}")
        return None
    return next((it for it in itens if it["numero_compacto"] == num), None)


# ------------------------------------------------------------------ estado local
def carregar_monitoradas() -> Dict[str, str]:
    if MONITORADAS_PATH.exists():
        return json.loads(MONITORADAS_PATH.read_text(encoding="utf-8"))
    L.WORK_DIR.mkdir(parents=True, exist_ok=True)
    MONITORADAS_PATH.write_text(json.dumps(MONITORADAS_SEED, indent=2), encoding="utf-8")
    return dict(MONITORADAS_SEED)


def salvar_monitoradas(mon: Dict[str, str]) -> None:
    MONITORADAS_PATH.write_text(json.dumps(mon, indent=2, sort_keys=True), encoding="utf-8")


def _load_hashes() -> Dict[str, dict]:
    if HASHES_PATH.exists():
        return json.loads(HASHES_PATH.read_text(encoding="utf-8"))
    return {}


_hash_lock = threading.Lock()


def _save_hashes(hashes: Dict[str, dict]) -> None:
    tmp = HASHES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(hashes, ensure_ascii=False), encoding="utf-8")
    tmp.replace(HASHES_PATH)


def _hash_disp(d: dict) -> str:
    base = (d.get("texto_proprio") or "") + "|" + (d.get("notas_texto") or "")
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ estado Notion
def carregar_existentes(progress=print) -> dict:
    progress("Baixando o estado atual da base 'resoluções' no Notion (props leves)...")
    props = ["row_key", "norma_id", "tipo_base", "grupo_resolucao", "url_fonte"]
    pages = L.query_all(only_props=props, progress=progress)
    por_rk: Dict[str, dict] = {}
    max_id = 0
    for p in pages:
        pr = p.get("properties") or {}
        rk = str(L.plain(pr.get("row_key")) or "")
        v = L.plain(pr.get("id"))
        if isinstance(v, (int, float)):
            max_id = max(max_id, int(v))
        if rk:
            por_rk[rk] = {
                "page_id": p["id"],
                "tipo_base": L.plain(pr.get("tipo_base")),
                "grupo": L.plain(pr.get("grupo_resolucao")),
                "url_fonte": L.plain(pr.get("url_fonte")),
                "norma_id": L.plain(pr.get("norma_id")),
            }
    progress(f"Base atual: {len(pages)} páginas, {len(por_rk)} com row_key, id máx = {max_id}.")
    return {"n": len(pages), "por_rk": por_rk, "max_id": max_id}


# ------------------------------------------------------------------ montagem de props
def _titulo_catalogo(it: dict) -> str:
    em = P.limit_text(it.get("ementa", ""), 80)
    base = f"Res.-TSE nº {it['numero']}/{it['ano']}"
    return f"{base} — {em}" if em else base


def _texto_rag_catalogo(it: dict, norma_id: str) -> str:
    partes = [
        "Base: Resolucoes TSE",
        f"Norma: Resolução-TSE nº {it['numero']}/{it['ano']}",
        "Tipo da linha: norma",
        f"Fonte: {it['url']}",
    ]
    if it.get("data_iso"):
        partes.append(f"Data: {it['data_iso']}")
    if it.get("ementa"):
        partes.extend(["", "Ementa:", it["ementa"]])
    return "\n".join(partes)


def props_catalogo(it: dict, seq_id: int, monitorada: bool) -> dict:
    norma_id = P.norm_id(it["numero"], it["ano"])
    texto_rag = _texto_rag_catalogo(it, norma_id)
    props = {
        "Nome": {"title": L.rt(_titulo_catalogo(it))},
        "id": {"number": seq_id},
        "row_key": {"rich_text": L.rt(f"norma:{norma_id}")},
        "titulo_linha": {"rich_text": L.rt(_titulo_catalogo(it))},
        "tipo_linha": {"select": {"name": "norma"}},
        "tipo_base": {"select": {"name": "sintetico" if monitorada else "catalogo"}},
        "tipo_dispositivo": {"select": {"name": "norma"}},
        "status_alteracao": {"select": {"name": "norma"}},
        "classe_vigencia_texto": {"select": {"name": "norma"}},
        "grupo_resolucao": {"select": {"name": "texto_integral" if monitorada else "catalogo"}},
        "prioridade_rag": {"select": {"name": "contexto"}},
        "usar_como_texto_vigente": {"checkbox": False},
        "incluir_no_rag": {"checkbox": True},
        "linha_sintetica": {"checkbox": True},
        "norma_id": {"rich_text": L.rt(norma_id)},
        "norma_numero": {"rich_text": L.rt(it["numero"])},
        "norma_ano": {"number": int(it["ano"])},
        "norma_titulo": {"rich_text": L.rt(f"Res.-TSE nº {it['numero']}/{it['ano']}")},
        "texto_dispositivo": {"rich_text": L.rt_multi(it.get("ementa", ""))},
        "resumo_curto": {"rich_text": L.rt(P.limit_text(it.get("ementa", "")))},
        "url_fonte": {"url": it["url"]},
        "fonte_csv": {"rich_text": L.rt(it["url"])},
        "texto_rag": {"rich_text": L.rt_multi(texto_rag)},
        "chars": {"number": len(texto_rag)},
        "tokens_estimados": {"number": P.estimate_tokens(texto_rag)},
    }
    if it.get("data_iso"):
        props["data_publicacao"] = {"date": {"start": it["data_iso"]}}
    pop = NOMES_POPULARES.get(it["numero_compacto"])
    if pop:
        props["norma_nome_popular"] = {"rich_text": L.rt(pop)}
    return props


def _montar_texto_rag_disp(d: dict, meta: dict, ia: dict) -> str:
    labels = [
        ("Base", "Resolucoes TSE"),
        ("Norma", f"Res.-TSE nº {meta['numero']}/{meta['ano']}"),
        ("Nome popular", NOMES_POPULARES.get(meta["numero_compacto"], "")),
        ("Tipo da linha", "estrutura" if d["kind"] == "estrutura" else "dispositivo"),
        ("Tipo do dispositivo", d["tipo_dispositivo"]),
        ("Dispositivo", d["dispositivo"]),
        ("Hierarquia", d["hierarquia"]),
        ("Dispositivo pai", d["dispositivo_pai"]),
        ("Status", d["status_alteracao"]),
        ("Fonte", meta["url"]),
    ]
    parts = [f"{l}: {P.normalize_ws(v)}" for l, v in labels if P.normalize_ws(v)]
    if ia.get("resumo_curto"):
        parts.extend(["", "Resumo:", ia["resumo_curto"]])
    if d["notas_texto"]:
        parts.extend(["", "Notas:", d["notas_texto"]])
    texto = d["texto_proprio"] if d["status_alteracao"] != "revogado" else ""
    if texto:
        parts.extend(["", "Texto do dispositivo:", d["texto_dispositivo"] or texto])
    return "\n".join(parts).strip()


def props_dispositivo(d: dict, meta: dict, seq_id: int, ia: Optional[dict] = None,
                      progress=print) -> dict:
    ia = ia or {}
    norma_id = P.norm_id(meta["numero"], meta["ano"])
    revogado = d["status_alteracao"] == "revogado"
    estrutura = d["kind"] == "estrutura"
    titulo = f"Res. {meta['numero']}/{meta['ano']}, {P.limit_text(d['hierarquia'], 140)}"
    texto_rag = _montar_texto_rag_disp(d, meta, ia)
    if estrutura:
        classe, vigente, prio = "estrutura", False, "contexto"
        tipo_linha, tipo_base = "estrutura", "sintetico"
    elif revogado:
        classe, vigente, prio = "revogado", False, "contexto"
        tipo_linha, tipo_base = "dispositivo", "operacional"
    else:
        classe, vigente, prio = "oficial_em_vigor", True, "principal"
        tipo_linha, tipo_base = "dispositivo", "operacional"
    resumo = ia.get("resumo_curto") or P.limit_text(P.first_sentence(d["texto_proprio"]))
    props = {
        "Nome": {"title": L.rt(titulo)},
        "id": {"number": seq_id},
        "row_key": {"rich_text": L.rt(d["row_key"])},
        "titulo_linha": {"rich_text": L.rt(titulo)},
        "tipo_linha": {"select": {"name": tipo_linha}},
        "tipo_base": {"select": {"name": tipo_base}},
        "tipo_dispositivo": {"select": {"name": d["tipo_dispositivo"]}},
        "status_alteracao": {"select": {"name": d["status_alteracao"]}},
        "classe_vigencia_texto": {"select": {"name": classe}},
        "grupo_resolucao": {"select": {"name": "texto_integral"}},
        "prioridade_rag": {"select": {"name": prio}},
        "usar_como_texto_vigente": {"checkbox": vigente},
        "incluir_no_rag": {"checkbox": True},
        "linha_sintetica": {"checkbox": estrutura},
        "norma_id": {"rich_text": L.rt(norma_id)},
        "norma_numero": {"rich_text": L.rt(meta["numero"])},
        "norma_ano": {"number": int(meta["ano"])},
        "norma_titulo": {"rich_text": L.rt(f"Res.-TSE nº {meta['numero']}/{meta['ano']}")},
        "dispositivo": {"rich_text": L.rt(d["dispositivo"])},
        "dispositivo_pai": {"rich_text": L.rt(d["dispositivo_pai"])},
        "dispositivo_pai_row_key": {"rich_text": L.rt(d["dispositivo_pai_row_key"])},
        "hierarquia_normativa": {"rich_text": L.rt(d["hierarquia"])},
        "ordem_doc": {"number": d["ordem_doc"]},
        "texto_dispositivo": {"rich_text": L.rt_multi(d["texto_dispositivo"], progress, d["row_key"])},
        "texto_em_vigor": {"rich_text": L.rt_multi("" if revogado else d["texto_proprio"],
                                                   progress, d["row_key"])},
        "notas_texto": {"rich_text": L.rt_multi(d["notas_texto"])},
        "notas_json": {"rich_text": L.rt_multi(json.dumps(d["notas"], ensure_ascii=False)
                                               if d["notas"] else "")},
        "qtd_notas": {"number": len(d["notas"])},
        "resumo_curto": {"rich_text": L.rt(resumo)},
        "url_fonte": {"url": meta["url"]},
        "fonte_csv": {"rich_text": L.rt(meta["url"])},
        "texto_rag": {"rich_text": L.rt_multi(texto_rag, progress, d["row_key"])},
        "chars": {"number": len(texto_rag)},
        "tokens_estimados": {"number": P.estimate_tokens(texto_rag)},
    }
    if d["texto_proprio_marcado"] != d["texto_proprio"]:
        props["texto_marcado"] = {"rich_text": L.rt_multi(d["texto_proprio_marcado"],
                                                          progress, d["row_key"])}
    if d["resolucao_alteradora"]:
        props["resolucao_alteradora"] = {"rich_text": L.rt(d["resolucao_alteradora"])}
        props["resolucao_alterada"] = {"rich_text": L.rt(f"{meta['numero']}/{meta['ano']}")}
    if revogado:
        props["alerta_vigencia"] = {"rich_text": L.rt(
            f"Dispositivo revogado{' pela Resolução nº ' + d['resolucao_alteradora'] if d['resolucao_alteradora'] else ''}.")}
    refs = P.extract_normative_refs(d["texto_proprio"], d["notas_texto"])
    if refs:
        props["referencias_normativas"] = {"rich_text": L.rt_multi(refs)}
    if ia.get("palavras_chave"):
        props["palavras_chave"] = {"rich_text": L.rt(str(ia["palavras_chave"]))}
    if meta.get("data_iso"):
        props["data_publicacao"] = {"date": {"start": meta["data_iso"]}}
    return props


# ------------------------------------------------------------------ IA (nano)
def _load_cache_ia() -> dict:
    cache = {}
    if CACHE_IA.exists():
        for ln in open(CACHE_IA, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                cache[r["k"]] = r
    return cache


def enriquecer(itens: List[dict], model: str, progress=print) -> Dict[str, dict]:
    """itens: [{'row_key', 'prompt_user'}] -> row_key -> {resumo_curto, palavras_chave}.
    Cache idempotente por sha1; geração paralela (8 threads) como temas_updater."""
    from conle_gerador import llm

    cache = _load_cache_ia()
    out: Dict[str, dict] = {}
    pend: List[tuple] = []
    for it in itens:
        k = hashlib.sha1((PROMPT_VERSION + "|" + it["row_key"] + "|" +
                          it["prompt_user"]).encode("utf-8")).hexdigest()
        if k in cache:
            out[it["row_key"]] = cache[k]["campos"]
        else:
            pend.append((it, k))
    progress(f"  IA: {len(out)} em cache, {len(pend)} a gerar")
    lock = threading.Lock()
    feito = [0]

    def _gerar(item):
        it, k = item
        try:
            data = llm.chat(
                "Você é um assistente jurídico especializado em direito eleitoral "
                "brasileiro. Responda apenas em JSON válido.",
                it["prompt_user"], json_mode=True, model=model, max_output_tokens=600)
            if not isinstance(data, dict):
                data = {}
            campos = {
                "resumo_curto": P.limit_text(str(data.get("resumo_curto") or "").strip(), 420),
                "palavras_chave": P.limit_text(
                    "; ".join(data["palavras_chave"]) if isinstance(data.get("palavras_chave"), list)
                    else str(data.get("palavras_chave") or "").strip(), 800),
            }
            with lock:
                with open(CACHE_IA, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "row_key": it["row_key"], "campos": campos},
                                       ensure_ascii=False) + "\n")
                out[it["row_key"]] = campos
                feito[0] += 1
                if feito[0] % 100 == 0 or feito[0] == len(pend):
                    progress(f"  IA {feito[0]}/{len(pend)}")
        except Exception as e:  # noqa: BLE001
            with lock:
                progress(f"  IA ERRO {it['row_key'][-40:]}: {str(e)[:120]}")

    if pend:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for item in pend:
                ex.submit(_gerar, item)
    return out


def _prompt_norma(meta: dict, estrutura_resumo: str) -> str:
    return (f"Resolução-TSE nº {meta['numero']}/{meta['ano']}"
            f" ({NOMES_POPULARES.get(meta['numero_compacto'], '')}).\n"
            f"Ementa: {meta.get('ementa', '')}\n"
            f"Estrutura (capítulos/seções): {estrutura_resumo[:1200]}\n\n"
            "Gere JSON com: resumo_curto (até 300 caracteres, o que a resolução "
            "disciplina) e palavras_chave (lista de 6 a 12 termos de busca).")


def _prompt_disp(d: dict, meta: dict) -> str:
    return (f"Dispositivo \"{d['hierarquia']}\" da Resolução-TSE nº "
            f"{meta['numero']}/{meta['ano']} "
            f"({NOMES_POPULARES.get(meta['numero_compacto'], 'norma eleitoral')}).\n"
            f"Status: {d['status_alteracao']}. Notas: {d['notas_texto'] or '-'}\n"
            f"Texto vigente: {(d['texto_proprio'] or '(revogado, sem texto)')[:1500]}\n\n"
            "Gere JSON com: resumo_curto (até 300 caracteres, efeito prático do "
            "dispositivo e o que mudou) e palavras_chave (lista de 5 a 10 termos).")


# ------------------------------------------------------------------ inserção/sync
def _ja_inseridos() -> Dict[str, str]:
    out = {}
    if MANIFEST.exists():
        for ln in open(MANIFEST, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                out[r["row_key"]] = r["page_id"]
    return out


def _inserir_lote(planos: List[tuple], hashes: Dict[str, dict], progress=print) -> int:
    """planos: [(row_key, props, hash|None, norma_compacta)] -> cria páginas (4 threads),
    manifest + hash-store para retomada/diffs futuros."""
    feitos = _ja_inseridos()
    pend = [pl for pl in planos if pl[0] not in feitos]
    if len(pend) < len(planos):
        progress(f"  manifest: {len(planos) - len(pend)} já inseridas (retomada)")
    lock = threading.Lock()
    cont = {"ok": 0, "erro": 0}

    def _criar(pl):
        rk, props, h, norma = pl
        try:
            pg = L.create_page(props)
            with lock:
                with open(MANIFEST, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"row_key": rk, "page_id": pg["id"],
                                        "quando": datetime.now().isoformat()}) + "\n")
                if h is not None:
                    hashes[rk] = {"h": h, "pid": pg["id"], "norma": norma}
                cont["ok"] += 1
                if cont["ok"] % 100 == 0 or cont["ok"] == len(pend):
                    progress(f"  criadas {cont['ok']}/{len(pend)}")
                    _save_hashes(hashes)
        except Exception as e:  # noqa: BLE001
            with lock:
                cont["erro"] += 1
                progress(f"  ERRO criar {rk[-50:]}: {str(e)[:150]}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for pl in pend:
            ex.submit(_criar, pl)
    _save_hashes(hashes)
    if cont["erro"]:
        progress(f"  erros: {cont['erro']} (re-rodar --aplicar retoma pelo manifest)")
    return cont["ok"]


# ------------------------------------------------------------------ catálogo
def coletar_indice(anos: List[int], progress=print) -> List[dict]:
    itens: List[dict] = []
    for ano in anos:
        lote = S.republicacoes(S.listar_ano(ano, progress=progress), progress=progress)
        progress(f"  índice {ano}: {len(lote)} resoluções")
        itens.extend(lote)
    return itens


def atualizar_catalogo(itens: List[dict], estado: dict, monitoradas: Dict[str, str],
                       dry_run: bool, limite: Optional[int], hashes: Dict[str, dict],
                       progress=print) -> dict:
    novos, presentes, avisos_alteradora = [], 0, []
    for it in itens:
        norma_id = P.norm_id(it["numero"], it["ano"])
        rk = f"norma:{norma_id}"
        m = ALTERA_RE.search(it.get("ementa", ""))
        if m:
            alvo = re.sub(r"\D", "", m.group(1))
            if alvo not in monitoradas:
                avisos_alteradora.append((it["numero"], it["ano"], m.group(1)))
        if rk in estado["por_rk"]:
            presentes += 1
            continue
        novos.append(it)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    nov_csv = L.WORK_DIR / f"novidades_{stamp}.csv"
    with open(nov_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["numero", "ano", "data", "ementa", "url"])
        for it in novos:
            w.writerow([it["numero"], it["ano"], it["data_iso"], it["ementa"], it["url"]])
    progress(f"Catálogo: {presentes} já na base, {len(novos)} novas. Relatório: {nov_csv.name}")
    for num, ano, alvo in avisos_alteradora:
        progress(f"  AVISO: Res. {num}/{ano} altera a Resolução nº {alvo}, que NÃO é monitorada")

    criadas = 0
    if not dry_run and novos:
        alvo = novos[:limite] if limite else novos
        seq = estado["max_id"]
        planos = []
        for it in alvo:
            seq += 1
            mon = it["numero_compacto"] in monitoradas and monitoradas.get(
                it["numero_compacto"]) == it["ano"]
            planos.append((f"norma:{P.norm_id(it['numero'], it['ano'])}",
                           props_catalogo(it, seq, mon), None, it["numero_compacto"]))
        estado["max_id"] = seq
        progress(f"Criando {len(planos)} linhas de catálogo no Notion...")
        criadas = _inserir_lote(planos, hashes, progress=progress)
    return {"novas": len(novos), "criadas": criadas, "csv": str(nov_csv),
            "avisos": len(avisos_alteradora)}


# ------------------------------------------------------------------ texto integral
def sincronizar_integral(meta: dict, estado: dict, hashes: Dict[str, dict],
                         dry_run: bool, model: str, com_ia: bool = True,
                         force_download: bool = True, progress=print) -> dict:
    """Baixa+parseia a compilada da resolução e sincroniza os dispositivos
    (create/patch/archive por row_key com hash local)."""
    num = meta["numero_compacto"]
    progress(f"\n=== Res. {meta['numero']}/{meta['ano']} — texto integral ===")
    html = S.baixar_compilada(meta["url"], num, meta["ano"], force=force_download,
                              progress=progress)
    out = P.parse_compilada(html, meta, progress=progress)
    disps = out["dispositivos"]
    s = out["stats"]
    progress(f"  parser: {s['dispositivos']} dispositivos, {s['estruturas']} estruturas, "
             f"{s['anexos']} itens de anexo | alterados={s['alterados']} "
             f"incluidos={s['incluidos']} revogados={s['revogados']}")

    meus_hashes = {rk: v for rk, v in hashes.items() if v.get("norma") == num}
    novos, mudados, iguais = [], [], 0
    site_keys = set()
    for d in disps:
        h = _hash_disp(d)
        site_keys.add(d["row_key"])
        reg = meus_hashes.get(d["row_key"])
        if reg is None:
            novos.append((d, h))
        elif reg["h"] != h:
            mudados.append((d, h, reg["pid"]))
        else:
            iguais += 1
    sumidos = [(rk, v["pid"]) for rk, v in meus_hashes.items() if rk not in site_keys]
    progress(f"  diff: {len(novos)} novos, {len(mudados)} alterados, {iguais} iguais, "
             f"{len(sumidos)} sumiram do site")

    if dry_run:
        return {"norma": meta["numero"], "novos": len(novos), "mudados": len(mudados),
                "sumidos": len(sumidos), "disps": disps}

    # IA: nível dispositivo (status != inalterado) apenas para novos/mudados
    ia_map: Dict[str, dict] = {}
    if com_ia:
        pedidos = [{"row_key": d["row_key"], "prompt_user": _prompt_disp(d, meta)}
                   for d, *_ in (novos + mudados)
                   if d["kind"] != "estrutura" and d["status_alteracao"] != "inalterado"]
        if pedidos:
            progress(f"  IA nano em {len(pedidos)} dispositivos alterados/incluídos/revogados...")
            ia_map = enriquecer(pedidos, model, progress=progress)

    seq = estado["max_id"]
    planos = []
    for d, h in novos:
        seq += 1
        planos.append((d["row_key"], props_dispositivo(d, meta, seq, ia_map.get(d["row_key"]),
                                                       progress), h, num))
    estado["max_id"] = seq
    criadas = _inserir_lote(planos, hashes, progress=progress) if planos else 0

    atualizadas = 0
    for d, h, pid in mudados:
        props = props_dispositivo(d, meta, 0, ia_map.get(d["row_key"]), progress)
        props.pop("id", None)  # preserva o id numérico existente
        try:
            L.patch_page_props(pid, props)
            hashes[d["row_key"]] = {"h": h, "pid": pid, "norma": num}
            atualizadas += 1
        except Exception as e:  # noqa: BLE001
            progress(f"  ERRO patch {d['row_key'][-50:]}: {str(e)[:150]}")
    if mudados:
        _save_hashes(hashes)

    arquivadas = 0
    for rk, pid in sumidos:
        try:
            L.archive_page(pid)
            with open(ARQUIVADAS, "a", encoding="utf-8") as f:
                f.write(json.dumps({"row_key": rk, "page_id": pid, "motivo": "sumiu_do_site",
                                    "quando": datetime.now().isoformat()}) + "\n")
            hashes.pop(rk, None)
            arquivadas += 1
            progress(f"  ARQUIVADA (sumiu do site): {rk}")
        except Exception as e:  # noqa: BLE001
            progress(f"  ERRO archive {rk[-50:]}: {str(e)[:150]}")
    if sumidos:
        _save_hashes(hashes)

    # linha raiz (norma:) — promove/atualiza a ficha da monitorada
    norma_id = P.norm_id(meta["numero"], meta["ano"])
    rk_raiz = f"norma:{norma_id}"
    raiz = estado["por_rk"].get(rk_raiz)
    estrutura_resumo = "; ".join(d["hierarquia"] for d in disps if d["kind"] == "estrutura")[:1500]
    ia_raiz: Dict[str, dict] = {}
    if com_ia:
        ia_raiz = enriquecer([{"row_key": rk_raiz,
                               "prompt_user": _prompt_norma(meta, estrutura_resumo)}],
                             model, progress=progress)
    campos = ia_raiz.get(rk_raiz) or {}
    props_raiz = {
        "tipo_base": {"select": {"name": "sintetico"}},
        "grupo_resolucao": {"select": {"name": "texto_integral"}},
        "url_fonte": {"url": meta["url"]},
        "fonte_csv": {"rich_text": L.rt(meta["url"])},
        "texto_dispositivo": {"rich_text": L.rt_multi(meta.get("ementa", ""))},
        "incluir_no_rag": {"checkbox": True},
    }
    if campos.get("resumo_curto"):
        props_raiz["resumo_curto"] = {"rich_text": L.rt(campos["resumo_curto"])}
    if campos.get("palavras_chave"):
        props_raiz["palavras_chave"] = {"rich_text": L.rt(campos["palavras_chave"])}
    if meta.get("data_iso"):
        props_raiz["data_publicacao"] = {"date": {"start": meta["data_iso"]}}
    if raiz:
        L.patch_page_props(raiz["page_id"], props_raiz)
        progress("  linha raiz atualizada (ficha da norma)")
    else:
        seq = estado["max_id"] = estado["max_id"] + 1
        base = props_catalogo(meta, seq, True)
        base.update(props_raiz)
        _inserir_lote([(rk_raiz, base, None, num)], hashes, progress=progress)
        progress("  linha raiz criada (ficha da norma)")

    return {"norma": meta["numero"], "novos": len(novos), "criadas": criadas,
            "mudados": len(mudados), "atualizadas": atualizadas,
            "sumidos": len(sumidos), "arquivadas": arquivadas}


# ------------------------------------------------------------------ migração (fase única)
def arquivar_antigas(progress=print) -> int:
    """Arquiva as linhas 'textos_operacionais' antigas (importação manual pré-site),
    substituídas pela ingestão texto_integral. Manifest para rollback."""
    progress("Buscando linhas grupo_resolucao='textos_operacionais'...")
    pages = L.query_all(only_props=["row_key", "grupo_resolucao", "norma_id"],
                        progress=progress)
    alvo = []
    for p in pages:
        pr = p.get("properties") or {}
        if str(L.plain(pr.get("grupo_resolucao"))) == "textos_operacionais":
            alvo.append((p["id"], str(L.plain(pr.get("row_key")))))
    progress(f"{len(alvo)} linhas antigas a arquivar.")
    ja = set()
    if ARQUIVADAS.exists():
        for ln in open(ARQUIVADAS, encoding="utf-8"):
            if ln.strip():
                ja.add(json.loads(ln)["page_id"])
    pend = [(pid, rk) for pid, rk in alvo if pid not in ja]
    lock = threading.Lock()
    cont = {"ok": 0}

    def _arq(item):
        pid, rk = item
        try:
            L.archive_page(pid)
            with lock:
                with open(ARQUIVADAS, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"row_key": rk, "page_id": pid,
                                        "motivo": "migracao_texto_integral",
                                        "quando": datetime.now().isoformat()}) + "\n")
                cont["ok"] += 1
                if cont["ok"] % 200 == 0 or cont["ok"] == len(pend):
                    progress(f"  arquivadas {cont['ok']}/{len(pend)}")
        except Exception as e:  # noqa: BLE001
            progress(f"  ERRO archive {rk[-50:]}: {str(e)[:120]}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for item in pend:
            ex.submit(_arq, item)
    return cont["ok"]


def relacionar(progress=print) -> int:
    """2ª passada: preenche dispositivo_pai_relacao com o page_id do pai
    (via hash-store, sem re-consultar o Notion)."""
    hashes = _load_hashes()
    manif = _ja_inseridos()
    pid_por_rk = {rk: v["pid"] for rk, v in hashes.items()}
    pid_por_rk.update(manif)
    # precisa do dispositivo_pai_row_key: relê do cache de parse mais recente? Não —
    # deriva do próprio row_key não é possível; consulta as páginas texto_integral.
    progress("Baixando dispositivo_pai_row_key das linhas texto_integral...")
    pages = L.query_all(only_props=["row_key", "dispositivo_pai_row_key", "grupo_resolucao"],
                        progress=progress)
    cont = 0
    pend = []
    for p in pages:
        pr = p.get("properties") or {}
        if str(L.plain(pr.get("grupo_resolucao"))) != "texto_integral":
            continue
        pai_rk = str(L.plain(pr.get("dispositivo_pai_row_key")) or "")
        if not pai_rk:
            continue
        pai_pid = pid_por_rk.get(pai_rk)
        if pai_pid:
            pend.append((p["id"], pai_pid))
    progress(f"{len(pend)} relações a preencher.")
    lock = threading.Lock()

    def _rel(item):
        nonlocal cont
        pid, pai_pid = item
        try:
            L.patch_page_props(pid, {"dispositivo_pai_relacao": {"relation": [{"id": pai_pid}]}})
            with lock:
                cont += 1
                if cont % 200 == 0 or cont == len(pend):
                    progress(f"  relacionadas {cont}/{len(pend)}")
        except Exception as e:  # noqa: BLE001
            progress(f"  ERRO relation: {str(e)[:120]}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for item in pend:
            ex.submit(_rel, item)
    return cont


# ------------------------------------------------------------------ orquestração
def executar(dry_run: bool = True, escopo: str = "tudo", limite: Optional[int] = None,
             reindexar: bool = False, varredura_completa: bool = False,
             reingerir: Optional[List[str]] = None, model: str = MODELO_PADRAO,
             com_ia: bool = True, progress=print) -> dict:
    L.WORK_DIR.mkdir(parents=True, exist_ok=True)
    monitoradas = carregar_monitoradas()
    hashes = _load_hashes()
    resumo: Dict[str, Any] = {}

    ano_atual = datetime.now().year
    if varredura_completa:
        anos = S.listar_anos(progress=progress)
        progress(f"Varredura completa: {len(anos)} anos ({anos[0]}–{anos[-1]})")
    else:
        anos = [a for a in (ano_atual - 1, ano_atual) if a >= 1994]
        progress(f"Varredura padrão: anos {anos} (use --varredura-completa p/ acervo inteiro)")
    # anos das monitoradas sempre entram (metas p/ texto integral)
    anos_mon = sorted({int(a) for a in monitoradas.values()})
    anos = sorted(set(anos) | set(anos_mon))

    itens = coletar_indice(anos, progress=progress)
    estado = carregar_existentes(progress=progress)

    if escopo in ("tudo", "catalogo"):
        resumo["catalogo"] = atualizar_catalogo(itens, estado, monitoradas, dry_run,
                                                limite, hashes, progress=progress)

    if escopo in ("tudo", "integrais"):
        por_num = {it["numero_compacto"]: it for it in itens}
        alvos = sorted(monitoradas) if not reingerir else \
            [re.sub(r"\D", "", x) for x in reingerir]
        res_int = []
        for num in alvos:
            meta = por_num.get(num)
            if not meta:
                progress(f"  AVISO: monitorada {num} não encontrada nos índices raspados")
                continue
            res_int.append(sincronizar_integral(meta, estado, hashes, dry_run, model,
                                                com_ia=com_ia, progress=progress))
        resumo["integrais"] = res_int
        # relatório de alterações (dry-run)
        if dry_run and res_int:
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            alt_csv = L.WORK_DIR / f"alteracoes_{stamp}.csv"
            with open(alt_csv, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["norma", "novos", "mudados", "sumidos"])
                for r in res_int:
                    w.writerow([r["norma"], r["novos"], r["mudados"], r["sumidos"]])
            progress(f"\nRelatório de alterações: {alt_csv}")

    if not dry_run and reindexar:
        from conle_gerador import notion_rag
        progress("\nReindexando RAG (base resolucoes_tse)...")
        notion_rag.indexar(["resolucoes_tse"], progress=progress)

    progress("\nConcluído." if not dry_run else "\nDry-run concluído (nada gravado).")
    return resumo


def main() -> None:
    ap = argparse.ArgumentParser(description="Atualizador da base 'resoluções' (site TSE -> Notion)")
    ap.add_argument("--aplicar", action="store_true", help="grava no Notion (senão, dry-run)")
    ap.add_argument("--escopo", choices=["tudo", "catalogo", "integrais"], default="tudo")
    ap.add_argument("--limite", type=int, help="máximo de criações de catálogo (teste)")
    ap.add_argument("--reindexar", action="store_true", help="reindexa o RAG ao final")
    ap.add_argument("--varredura-completa", action="store_true",
                    help="raspa todos os anos (1994-hoje); padrão: ano atual + anterior")
    ap.add_argument("--reingerir", nargs="*", help="força re-ingestão dessas resoluções (ex.: 23444)")
    ap.add_argument("--sem-ia", action="store_true", help="não chama o gpt-5.6-luna")
    ap.add_argument("--model", default=MODELO_PADRAO)
    ap.add_argument("--arquivar-antigas", action="store_true",
                    help="MIGRAÇÃO: arquiva as linhas textos_operacionais antigas")
    ap.add_argument("--relacionar", action="store_true",
                    help="2ª passada: preenche dispositivo_pai_relacao")
    ap.add_argument("--monitorar", metavar="REF",
                    help="adiciona resolução às monitoradas (nº/ano ou link do TSE)")
    args = ap.parse_args()
    if args.monitorar:
        num, ano = parse_ref_resolucao(args.monitorar)
        item = localizar_no_indice(num, ano)
        if item:
            print(f"Encontrada no índice {ano}: Res. {item['numero']}/{ano} — "
                  f"{item['ementa'][:200]}")
        else:
            print(f"AVISO: Res. {fmt_numero(num)}/{ano} não localizada no índice {ano} "
                  "(monitorando mesmo assim; confira o número).")
        mon = carregar_monitoradas()
        mon[num] = ano
        salvar_monitoradas(mon)
        print(f"Monitorada registrada: Res. {fmt_numero(num)}/{ano}. Texto integral na "
              f"próxima execução: python resolucoes_updater.py --aplicar --escopo integrais")
        return
    if args.arquivar_antigas:
        arquivar_antigas()
        return
    if args.relacionar:
        relacionar()
        return
    executar(dry_run=not args.aplicar, escopo=args.escopo, limite=args.limite,
             reindexar=args.reindexar, varredura_completa=args.varredura_completa,
             reingerir=args.reingerir, model=args.model, com_ia=not args.sem_ia)


if __name__ == "__main__":
    main()
