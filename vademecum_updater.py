# -*- coding: utf-8 -*-
"""Atualizador da base Notion 'vademecum' (Vademecum - RAG consolidado).

Sincroniza as normas da base com as FONTES OFICIAIS, dispositivo a dispositivo:
- Planalto (normas federais compiladas; autoritativo — divergência vira patch);
- Câmara dos Deputados: RICD (DOCX oficial compilado; o nome do arquivo é o
  detector de versão) e Código de Ética (LEGIN norma atualizada);
- Questões de ordem: API pública da Câmara, modo incremental (só QOs novas).

Matching por HIERARQUIA normalizada (row_keys da base são posicionais) e diff
pela PARTE PRÓPRIA do texto via comparison_key (_vade_match). Linhas novas
criadas com row_key disp:{norma_id}:planalto|camara:{slug}. Linhas da base sem
correspondente na fonte NUNCA são arquivadas (vão para nao_encontrados_*.csv).

Idempotente: manifest vademecum_work/inseridos.jsonl + hash-store hashes.json
(sha1 do texto próprio + notas por row_key -> diffs futuros instantâneos).

  python vademecum_updater.py                                # dry-run tudo
  python vademecum_updater.py --escopo planalto --normas lei_9504_1997
  python vademecum_updater.py --aplicar --escopo planalto --normas ...
  python vademecum_updater.py --aplicar --escopo qordem      # QOs novas
  python vademecum_updater.py --aplicar --reindexar
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

import _vade_fontes as F
import _vade_lib as L
import _vade_match as VM
import _vade_parser as VP

HASHES_PATH = L.WORK_DIR / "hashes.json"
MANIFEST = L.WORK_DIR / "inseridos.jsonl"
CACHE_IA = L.WORK_DIR / "updater_cache.jsonl"
MODELO_PADRAO = "gpt-5.6-luna"
PROMPT_VERSION = "v1"
HOJE = datetime.now().strftime("%Y-%m-%d")

RICD_ID = "regimento_interno_camara_deputados"
ETICA_ID = "codigo_etica_decoro_camara_deputados"

# metadata das normas NOVAS (ingestão integral do Planalto)
NORMAS_NOVAS = {
    "lei_9096_1995": {
        "norma_id": "lei_9096_1995",
        "norma_titulo": "Lei n. 9.096, de 19 de setembro de 1995",
        "norma_numero": "9096", "norma_ano": "1995",
        "norma_nome_popular": "Lei dos Partidos Politicos",
        "grupo": "eleitoral",
    },
    "lei_6091_1974": {
        "norma_id": "lei_6091_1974",
        "norma_titulo": "Lei n. 6.091, de 15 de agosto de 1974",
        "norma_numero": "6091", "norma_ano": "1974",
        "norma_nome_popular": "Transporte gratuito de eleitores (zonas rurais)",
        "grupo": "eleitoral",
    },
}


def _vc():
    """conversores.NOTION_import_vademecum_consolidado (import tardio: os.chdir raiz)."""
    import conversores.NOTION_import_vademecum_consolidado as VC
    return VC


# ------------------------------------------------------------------ estado local
def _load_hashes() -> Dict[str, dict]:
    if HASHES_PATH.exists():
        return json.loads(HASHES_PATH.read_text(encoding="utf-8"))
    return {}


def _save_hashes(hashes: Dict[str, dict]) -> None:
    tmp = HASHES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(hashes, ensure_ascii=False), encoding="utf-8")
    tmp.replace(HASHES_PATH)


def _hash_disp(d: dict) -> str:
    base = (d.get("texto_proprio") or "") + "|" + (d.get("notas_texto") or "")
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _ja_inseridos() -> Dict[str, str]:
    out = {}
    if MANIFEST.exists():
        for ln in open(MANIFEST, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                out[r["row_key"]] = r["page_id"]
    return out


# ------------------------------------------------------------------ estado da base
def carregar_base(refresh: bool = False, progress=print) -> dict:
    """Estado da base a partir do dump local (refresh=True re-baixa tudo)."""
    if refresh or not L.DUMP_PATH.exists():
        progress("Atualizando dump da base vademecum (35k linhas, alguns minutos)...")
        L.dump(progress=progress)
    linhas = L.load_dump()
    por_norma: Dict[str, List[dict]] = {}
    raizes: Dict[str, dict] = {}
    max_id = 0
    for d in linhas:
        pr = dict(d["props"])
        pr["_pid"] = d["id"]
        nid = pr.get("norma_id") or ""
        por_norma.setdefault(nid, []).append(pr)
        v = pr.get("id")
        if isinstance(v, (int, float)):
            max_id = max(max_id, int(v))
        rk = pr.get("row_key") or ""
        if rk.startswith("norma:"):
            raizes[nid] = pr
    progress(f"Base (dump): {len(linhas)} linhas, {len(por_norma)} normas, id máx = {max_id}.")
    return {"n": len(linhas), "por_norma": por_norma, "raizes": raizes, "max_id": max_id}


def metadata_norma(norma_id: str, estado: dict) -> dict:
    raiz = estado["raizes"].get(norma_id)
    if raiz:
        return {
            "norma_id": norma_id,
            "norma_titulo": raiz.get("norma_titulo") or "",
            "norma_numero": str(raiz.get("norma_numero") or ""),
            "norma_ano": str(raiz.get("norma_ano") or ""),
            "norma_nome_popular": raiz.get("norma_nome_popular") or "",
            "grupo": raiz.get("grupo") or "vademecum",
        }
    if norma_id in NORMAS_NOVAS:
        return dict(NORMAS_NOVAS[norma_id])
    extras = F.carregar_normas_extras()
    if norma_id in extras:
        e = extras[norma_id]
        return {"norma_id": norma_id,
                "norma_titulo": e.get("norma_titulo") or norma_id,
                "norma_numero": str(e.get("norma_numero") or ""),
                "norma_ano": str(e.get("norma_ano") or ""),
                "norma_nome_popular": e.get("norma_nome_popular") or "",
                "grupo": e.get("grupo") or "vademecum"}
    return {"norma_id": norma_id, "norma_titulo": norma_id, "norma_numero": "",
            "norma_ano": "", "norma_nome_popular": norma_id, "grupo": "vademecum"}


# ------------------------------------------------------------------ props Notion
def _governanca(d: dict) -> dict:
    """Props de vigência conforme o caso (só linhas tocadas/criadas)."""
    if d["kind"] == "estrutura":
        st, cl, vig, pr = "estrutura", "estrutura", False, "contexto"
    elif d["status_alteracao"] == "revogado":
        st, cl, vig, pr = "revogado", "revogado", False, "contexto"
    else:
        st = d["status_alteracao"] or "inalterado"
        cl, vig, pr = "oficial_em_vigor", True, "principal"
    props = {
        "status_alteracao": {"select": {"name": st}},
        "classe_vigencia_texto": {"select": {"name": cl}},
        "usar_como_texto_vigente": {"checkbox": vig},
        "prioridade_rag": {"select": {"name": pr}},
        "data_ultima_sincronizacao": {"date": {"start": HOJE}},
    }
    if d.get("norma_alteradora"):
        props["norma_alteradora"] = {"rich_text": L.rt(d["norma_alteradora"])}
    return props


def montar_props(d: dict, meta: dict, url_fonte: str, ia: Optional[dict] = None,
                 novo: bool = False, seq_id: Optional[int] = None,
                 progress=print) -> dict:
    """Props Notion completas de um dispositivo (create) ou de patch (novo=False:
    sem Nome/id/row_key/ordem, preservando os da linha existente)."""
    VC = _vc()
    ia = ia or {}
    revogado = d["status_alteracao"] == "revogado"
    estrutura = d["kind"] == "estrutura"
    texto_em_vigor = "" if revogado else d["texto_dispositivo"]
    row_csv = {
        "norma_titulo": meta["norma_titulo"], "norma_nome_popular": meta["norma_nome_popular"],
        "grupo": meta["grupo"],
        "tipo_linha": "estrutura" if estrutura else "dispositivo",
        "tipo_dispositivo": d["tipo_dispositivo"],
        "dispositivo": d["dispositivo"], "hierarquia_normativa": d["hierarquia"],
        "dispositivo_pai": d["dispositivo_pai"],
        "dispositivo_pai_row_key": d.get("dispositivo_pai_row_key", ""),
        "fonte_database_titulo": "Fonte oficial (site)", "fonte_page_id": url_fonte,
        "texto_em_vigor": texto_em_vigor, "texto_dispositivo": d["texto_dispositivo"],
        "notas_texto": d["notas_texto"],
        "resumo_curto": ia.get("resumo_curto") or "",
        "referencias_normativas": VC.extract_normative_refs(d["texto_dispositivo"], d["notas_texto"]),
        "referencias_jurisprudenciais": VC.extract_juris_refs(d["texto_dispositivo"], d["notas_texto"]),
        "palavras_chave": ia.get("palavras_chave") or "",
        "qualidade_texto": "", "alerta_qualidade": "",
    }
    if not row_csv["resumo_curto"]:
        # não usar VC.make_summary: o first_sentence legado degenera em "…: Art."
        nucleo = VM.primeira_sentenca_robusta(d["texto_proprio"])
        nome = meta["norma_nome_popular"] or meta["norma_titulo"]
        row_csv["resumo_curto"] = VM.normalize_ws(
            f"{nome}, {d['dispositivo']}: {nucleo}" if nucleo
            else f"{nome}, {d['dispositivo']}.")[:420]
    if revogado and not row_csv["resumo_curto"].startswith("[REVOGADO"):
        row_csv["resumo_curto"] = f"{VM.MARCADOR_REVOGADO} {row_csv['resumo_curto']}"[:420]
    if revogado or estrutura:
        alerta, qual = "", "ok"
    else:
        alerta, qual = VC.quality_alert(d["texto_proprio"])
    row_csv["alerta_qualidade"], row_csv["qualidade_texto"] = alerta, qual
    texto_rag = VC.build_text_rag(row_csv)
    if revogado:
        texto_rag = f"{VM.MARCADOR_REVOGADO}\n{texto_rag}"

    props: Dict[str, Any] = {
        "tipo_linha": {"select": {"name": row_csv["tipo_linha"]}},
        "tipo_base": {"select": {"name": "fonte_oficial"}},
        "tipo_dispositivo": {"select": {"name": d["tipo_dispositivo"]}},
        "grupo": {"select": {"name": meta["grupo"]}},
        "fonte_tipo": {"select": {"name": "fonte_oficial_site"}},
        "dispositivo": {"rich_text": L.rt(d["dispositivo"])},
        "dispositivo_pai": {"rich_text": L.rt(d["dispositivo_pai"])},
        "hierarquia_normativa": {"rich_text": L.rt_multi(d["hierarquia"])},
        "texto_dispositivo": {"rich_text": L.rt_multi(d["texto_dispositivo"], progress, d["row_key"])},
        "texto_em_vigor": {"rich_text": L.rt_multi(texto_em_vigor, progress, d["row_key"])},
        "notas_texto": {"rich_text": L.rt_multi(d["notas_texto"])},
        "notas_json": {"rich_text": L.rt_multi(json.dumps(d["notas"], ensure_ascii=False)
                                               if d["notas"] else "")},
        "qtd_notas": {"number": len(d["notas"])},
        "referencias_normativas": {"rich_text": L.rt_multi(row_csv["referencias_normativas"])},
        "referencias_jurisprudenciais": {"rich_text": L.rt_multi(row_csv["referencias_jurisprudenciais"])},
        "resumo_curto": {"rich_text": L.rt(row_csv["resumo_curto"])},
        "qualidade_texto": {"select": {"name": qual}},
        "alerta_qualidade": {"rich_text": L.rt(alerta)},
        "texto_rag": {"rich_text": L.rt_multi(texto_rag, progress, d["row_key"])},
        "chars": {"number": len(texto_rag)},
        "tokens_estimados": {"number": VP.RP.estimate_tokens(texto_rag)},
        "url_fonte": {"url": url_fonte},
        "fonte_page_id": {"rich_text": L.rt(url_fonte)},
        "incluir_no_rag": {"checkbox": True},
        "linha_sintetica": {"checkbox": estrutura},
    }
    if row_csv["palavras_chave"]:
        props["palavras_chave"] = {"rich_text": L.rt(str(row_csv["palavras_chave"]))}
    props.update(_governanca(d))
    if novo:
        titulo = f"{meta['norma_nome_popular'] or meta['norma_titulo']}, " \
                 f"{VM.normalize_ws(d['hierarquia'])[:130]}"
        props["Nome"] = {"title": L.rt(titulo)}
        props["titulo_linha"] = {"rich_text": L.rt(titulo)}
        props["row_key"] = {"rich_text": L.rt(d["row_key"])}
        props["dispositivo_pai_row_key"] = {"rich_text": L.rt(d.get("dispositivo_pai_row_key", ""))}
        props["norma_id"] = {"rich_text": L.rt(meta["norma_id"])}
        props["norma_titulo"] = {"rich_text": L.rt(meta["norma_titulo"])}
        if meta.get("norma_numero"):
            props["norma_numero"] = {"rich_text": L.rt(meta["norma_numero"])}
        if str(meta.get("norma_ano") or "").isdigit():
            props["norma_ano"] = {"number": int(meta["norma_ano"])}
        props["norma_nome_popular"] = {"rich_text": L.rt(meta["norma_nome_popular"])}
        if seq_id is not None:
            props["id"] = {"number": seq_id}
            props["ordem_global"] = {"number": seq_id}
        props["ordem_doc"] = {"number": d["ordem_doc"]}
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
    """itens: [{'row_key','prompt_user'}] -> row_key -> {resumo_curto, palavras_chave}."""
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
                "Você é um assistente jurídico especializado em direito brasileiro "
                "(eleitoral, constitucional e legislativo). Responda apenas em JSON válido.",
                it["prompt_user"], json_mode=True, model=model, max_output_tokens=600)
            if not isinstance(data, dict):
                data = {}
            campos = {
                "resumo_curto": VM.normalize_ws(str(data.get("resumo_curto") or ""))[:420],
                "palavras_chave": VM.normalize_ws(
                    "; ".join(data["palavras_chave"]) if isinstance(data.get("palavras_chave"), list)
                    else str(data.get("palavras_chave") or ""))[:800],
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


def _prompt_disp(d: dict, meta: dict) -> str:
    return (f"Dispositivo \"{d['hierarquia']}\" — {meta['norma_titulo']} "
            f"({meta['norma_nome_popular']}).\n"
            f"Status: {d['status_alteracao']}. Notas: {d['notas_texto'] or '-'}\n"
            f"Texto vigente: {(d['texto_proprio'] or '(revogado, sem texto)')[:1500]}\n\n"
            "Gere JSON com: resumo_curto (até 300 caracteres, efeito prático do "
            "dispositivo e o que mudou, se mudou) e palavras_chave (lista de 5 a 10 termos).")


def _prompt_norma(meta: dict, n_disp: int) -> str:
    return (f"{meta['norma_titulo']} ({meta['norma_nome_popular']}), "
            f"grupo {meta['grupo']}, ~{n_disp} dispositivos.\n\n"
            "Gere JSON com: resumo_curto (até 300 caracteres, o que a norma "
            "disciplina) e palavras_chave (lista de 6 a 12 termos de busca).")


# ------------------------------------------------------------------ inserção
def _inserir_lote(planos: List[tuple], hashes: Dict[str, dict], progress=print) -> int:
    """planos: [(row_key, props, hash|None, norma_id)] -> create (4 threads) c/ manifest."""
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
                progress(f"  ERRO criar {rk[-60:]}: {str(e)[:150]}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for pl in pend:
            ex.submit(_criar, pl)
    _save_hashes(hashes)
    if cont["erro"]:
        progress(f"  erros: {cont['erro']} (re-rodar --aplicar retoma pelo manifest)")
    return cont["ok"]


# ------------------------------------------------------------------ sync por norma
def _parse_fonte(norma_id: str, force: bool, progress=print):
    """-> (out do parser, url_fonte) ou (None, msg_skip)."""
    if norma_id == RICD_ID:
        info = F.descobrir_docx_ricd(progress=progress)
        registrada = F.versao_ricd_registrada()
        if registrada == info["versao"] and not force:
            return None, f"RICD sem mudança de versão ({info['versao']})"
        progress(f"  RICD versão no site: {info['versao']} (registrada: {registrada or '-'})")
        path = F.baixar_ricd_docx(info["url"], force=True, progress=progress)
        out = VP.parse_ricd_docx(path, progress=progress)
        out["_ricd_info"] = info
        return out, info["url"]
    if norma_id == ETICA_ID:
        html = F.baixar_etica(force=force, progress=progress)
        return VP.parse_etica_html(html, progress=progress), F.ETICA_URL
    html, url = F.baixar_planalto(norma_id, force=force, progress=progress)
    return VP.parse_planalto(html, norma_id, progress=progress), url


def sincronizar_norma(norma_id: str, estado: dict, hashes: Dict[str, dict],
                      dry_run: bool, model: str, com_ia: bool = True,
                      force_download: bool = True, forcar: bool = False,
                      progress=print) -> dict:
    progress(f"\n=== {norma_id} ===")
    try:
        out, url = _parse_fonte(norma_id, force_download, progress=progress)
    except Exception as e:  # noqa: BLE001
        progress(f"  ERRO na fonte: {str(e)[:200]}")
        return {"norma": norma_id, "erro": str(e)[:200]}
    if out is None:
        progress(f"  {url}")
        return {"norma": norma_id, "skip": url}
    meta = metadata_norma(norma_id, estado)
    s = out["stats"]
    progress(f"  parser: {s['dispositivos']} dispositivos, {s['estruturas']} estruturas, "
             f"{s['anexos']} anexos | alt={s['alterados']} incl={s['incluidos']} rev={s['revogados']}")

    linhas_norma = estado["por_norma"].get(norma_id, [])
    base_disp = [ln for ln in linhas_norma
                 if ln.get("tipo_linha") == "dispositivo"
                 and ln.get("tipo_base") in ("matriz_notion", "fonte_oficial")]
    norma_nova = not linhas_norma

    disps_fonte = out["dispositivos"]
    if norma_nova:
        alvo_fonte = disps_fonte  # ingestão integral: estruturas e anexos também
        res = {"pares": [], "novos": [d for d in alvo_fonte],
               "nao_encontrados": [], "duplicatas": [],
               "stats": {"pares": 0, "iguais": 0, "formatacao": 0, "substantivos": 0,
                         "novos": len(alvo_fonte), "nao_encontrados": 0, "duplicatas": 0}}
        progress(f"  norma NOVA: ingestão integral de {len(alvo_fonte)} linhas")
    else:
        idx = VM.indexar_base(base_disp)
        res = VM.casar_dispositivos(disps_fonte, idx, progress=progress)
        n_anexos_skip = sum(1 for d in disps_fonte if d["kind"] == "anexo")
        st = res["stats"]
        progress(f"  diff: casados={st['pares']} ({st['pares']/max(1,len(base_disp))*100:.1f}% da base) "
                 f"iguais={st['iguais']} formatacao={st['formatacao']} "
                 f"SUBSTANTIVOS={st['substantivos']} novos={st['novos']} "
                 f"nao_encontrados={st['nao_encontrados']} dup={st['duplicatas']}"
                 + (f" | anexos ignorados no sync: {n_anexos_skip}" if n_anexos_skip else ""))

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # CSV de auditoria
    aud_csv = L.WORK_DIR / f"sync_{norma_id}_{stamp}.csv"
    with open(aud_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["acao", "hierarquia", "status", "alteradora", "texto_base", "texto_fonte"])
        for d, ln, c in res["pares"]:
            if c == "igual":
                continue
            bp = VM.parte_propria(ln.get("texto_em_vigor") or ln.get("texto_dispositivo", ""))
            w.writerow([c, d["hierarquia"], d["status_alteracao"], d["norma_alteradora"],
                        bp[:300], d["texto_proprio"][:300]])
        for d in res["novos"]:
            w.writerow(["novo", d["hierarquia"], d["status_alteracao"], d["norma_alteradora"],
                        "", d["texto_proprio"][:300]])
        for ln in res["nao_encontrados"]:
            w.writerow(["nao_encontrado", ln.get("hierarquia_normativa", ""), "", "",
                        VM.parte_propria(ln.get("texto_em_vigor", ""))[:300], ""])
    progress(f"  auditoria: {aud_csv.name}")

    resumo = {"norma": norma_id, **res["stats"], "criadas": 0, "atualizadas": 0}
    if dry_run:
        return resumo

    # gate de segurança contra regressão de parser
    st = res["stats"]
    if not norma_nova and st["pares"] and st["substantivos"] / st["pares"] > 0.30 and not forcar:
        progress(f"  GATE: {st['substantivos']}/{st['pares']} substantivos (>30%) — "
                 f"revise {aud_csv.name} e rode com --forcar para aplicar")
        resumo["gate"] = True
        return resumo

    # IA: dispositivos substantivos + novos (exceto estruturas/inalterados de norma nova)
    ia_map: Dict[str, dict] = {}
    if com_ia:
        pedidos = []
        for d, _ln, c in res["pares"]:
            if c == "substantivo" and d["kind"] == "dispositivo":
                pedidos.append({"row_key": d["row_key"], "prompt_user": _prompt_disp(d, meta)})
        for d in res["novos"]:
            if d["kind"] == "dispositivo" and (norma_nova is False or d["status_alteracao"] != "inalterado"):
                pedidos.append({"row_key": d["row_key"], "prompt_user": _prompt_disp(d, meta)})
        if pedidos:
            progress(f"  IA nano em {len(pedidos)} dispositivos...")
            ia_map = enriquecer(pedidos, model, progress=progress)

    # patches (pares substantivos)
    atualizadas = 0
    for d, ln, c in res["pares"]:
        h = _hash_disp(d)
        if c != "substantivo":
            hashes[VM.normalize_ws(ln.get("row_key", "")) or d["row_key"]] = {
                "h": h, "pid": ln["_pid"], "norma": norma_id}
            continue
        props = montar_props(d, meta, url, ia_map.get(d["row_key"]), novo=False, progress=progress)
        try:
            L.patch_page_props(ln["_pid"], props)
            hashes[VM.normalize_ws(ln.get("row_key", "")) or d["row_key"]] = {
                "h": h, "pid": ln["_pid"], "norma": norma_id}
            atualizadas += 1
            if atualizadas % 50 == 0:
                progress(f"  atualizadas {atualizadas}...")
                _save_hashes(hashes)
        except Exception as e:  # noqa: BLE001
            progress(f"  ERRO patch {d['row_key'][-60:]}: {str(e)[:150]}")
    _save_hashes(hashes)
    progress(f"  atualizadas: {atualizadas}")

    # creates (novos)
    pid_por_rk = {VM.normalize_ws(ln.get("row_key", "")): ln["_pid"] for ln in linhas_norma}
    hier_para_rk = {VM.match_key(ln.get("hierarquia_normativa", "")): VM.normalize_ws(ln.get("row_key", ""))
                    for ln in base_disp}
    seq = estado["max_id"]
    planos = []
    novos_finais: List[dict] = []
    for d in res["novos"]:
        seq += 1
        # pai preferencialmente apontando para linha já existente na base
        pai_rk_base = hier_para_rk.get(VM.match_key(d["dispositivo_pai"]))
        d = dict(d)
        if pai_rk_base:
            d["dispositivo_pai_row_key"] = pai_rk_base
        novos_finais.append(d)
        planos.append((d["row_key"], montar_props(d, meta, url, ia_map.get(d["row_key"]),
                                                  novo=True, seq_id=seq, progress=progress),
                       _hash_disp(d), norma_id))
    estado["max_id"] = seq
    criadas = _inserir_lote(planos, hashes, progress=progress) if planos else 0

    # relation do pai para as criadas (quando o pai é conhecido)
    manif = _ja_inseridos()
    pid_por_rk.update(manif)
    rel = 0
    for d in novos_finais:
        pid = manif.get(d["row_key"]) or (hashes.get(d["row_key"]) or {}).get("pid")
        pai_pid = pid_por_rk.get(d.get("dispositivo_pai_row_key", ""))
        if pid and pai_pid:
            try:
                L.patch_page_props(pid, {"dispositivo_pai_relacao": {"relation": [{"id": pai_pid}]}})
                rel += 1
            except Exception:  # noqa: BLE001
                pass
    if rel:
        progress(f"  relações pai preenchidas: {rel}")

    # raiz da norma (cria p/ norma nova; patch nas demais)
    raiz = estado["raizes"].get(norma_id)
    ia_raiz = {}
    if com_ia:
        ia_raiz = enriquecer([{"row_key": f"norma:{norma_id}",
                               "prompt_user": _prompt_norma(meta, s["dispositivos"])}],
                             model, progress=progress).get(f"norma:{norma_id}", {})
    props_raiz = {
        "url_fonte": {"url": url},
        "data_ultima_sincronizacao": {"date": {"start": HOJE}},
        "status_alteracao": {"select": {"name": "norma"}},
        "classe_vigencia_texto": {"select": {"name": "norma"}},
        "prioridade_rag": {"select": {"name": "contexto"}},
        "usar_como_texto_vigente": {"checkbox": False},
        "incluir_no_rag": {"checkbox": True},
    }
    if ia_raiz.get("resumo_curto"):
        props_raiz["resumo_curto"] = {"rich_text": L.rt(ia_raiz["resumo_curto"])}
    if ia_raiz.get("palavras_chave"):
        props_raiz["palavras_chave"] = {"rich_text": L.rt(ia_raiz["palavras_chave"])}
    # texto_rag da ficha da norma (as 2 primeiras raízes criadas nasceram sem)
    row_raiz = {
        "norma_titulo": meta["norma_titulo"], "norma_nome_popular": meta["norma_nome_popular"],
        "grupo": meta["grupo"], "tipo_linha": "norma", "tipo_dispositivo": "norma",
        "dispositivo": meta["norma_titulo"], "hierarquia_normativa": meta["norma_titulo"],
        "fonte_page_id": url, "qualidade_texto": "ok",
        "resumo_curto": (ia_raiz.get("resumo_curto") or ""),
        "palavras_chave": (ia_raiz.get("palavras_chave") or ""),
    }
    tr_raiz = _vc().build_text_rag(row_raiz)
    props_raiz["texto_rag"] = {"rich_text": L.rt_multi(tr_raiz)}
    props_raiz["chars"] = {"number": len(tr_raiz)}
    props_raiz["tokens_estimados"] = {"number": max(1, round(len(tr_raiz) / 4))}
    if raiz:
        L.patch_page_props(raiz["_pid"], props_raiz)
        progress("  raiz atualizada")
    else:
        seq = estado["max_id"] = estado["max_id"] + 1
        titulo = f"{meta['norma_titulo']} — {meta['norma_nome_popular']}"
        props_raiz.update({
            "Nome": {"title": L.rt(titulo)},
            "titulo_linha": {"rich_text": L.rt(titulo)},
            "row_key": {"rich_text": L.rt(f"norma:{norma_id}")},
            "tipo_linha": {"select": {"name": "norma"}},
            "tipo_base": {"select": {"name": "fonte_oficial"}},
            "tipo_dispositivo": {"select": {"name": "norma"}},
            "grupo": {"select": {"name": meta["grupo"]}},
            "fonte_tipo": {"select": {"name": "fonte_oficial_site"}},
            "linha_sintetica": {"checkbox": True},
            "norma_id": {"rich_text": L.rt(norma_id)},
            "norma_titulo": {"rich_text": L.rt(meta["norma_titulo"])},
            "norma_nome_popular": {"rich_text": L.rt(meta["norma_nome_popular"])},
            "id": {"number": seq},
            "ordem_global": {"number": seq},
        })
        if meta.get("norma_numero"):
            props_raiz["norma_numero"] = {"rich_text": L.rt(meta["norma_numero"])}
        if str(meta.get("norma_ano") or "").isdigit():
            props_raiz["norma_ano"] = {"number": int(meta["norma_ano"])}
        _inserir_lote([(f"norma:{norma_id}", props_raiz, None, norma_id)], hashes,
                      progress=progress)
        progress("  raiz criada")

    if norma_id == RICD_ID and out.get("_ricd_info"):
        F.registrar_versao_ricd(out["_ricd_info"])
    resumo["criadas"] = criadas
    resumo["atualizadas"] = atualizadas
    return resumo


# ------------------------------------------------------------------ questões de ordem
def sincronizar_qordem(estado: dict, hashes: Dict[str, dict], dry_run: bool,
                       progress=print) -> dict:
    import conversores.CAMARA_qordem_ricd_to_vademecum_rag as CQ

    detalhes = CQ.read_details_cache(CQ.DEFAULT_DETAILS_JSON)
    progress(f"QO: cache com {len(detalhes)} detalhes")
    client = CQ.QOrdemClient(CQ.DEFAULT_BASE_URL, timeout_s=40, max_retries=4, rate_rps=2.0)

    novos_ids: List[str] = []
    page = 0
    paginas_conhecidas = 0
    while True:
        payload = client.search_page(page)
        itens = payload.get("resultadosList", []) or []
        if not itens:
            break
        ids_pagina = [str(it.get("numInternoQOrdem")) for it in itens if it.get("numInternoQOrdem")]
        desconhecidos = [i for i in ids_pagina if i not in detalhes]
        novos_ids.extend(desconhecidos)
        if not desconhecidos:
            paginas_conhecidas += 1
            if paginas_conhecidas >= 2:  # 2 páginas seguidas 100% conhecidas -> fim
                break
        else:
            paginas_conhecidas = 0
        qtd = int(payload.get("qtdPaginas") or 0)
        page += 1
        if qtd and page >= qtd:
            break
        if page > 400:
            break
    novos_ids = list(dict.fromkeys(novos_ids))
    progress(f"QO: {len(novos_ids)} questões novas na API")
    if dry_run or not novos_ids:
        return {"norma": "qordem", "novos": len(novos_ids), "criadas": 0}

    for i, qid in enumerate(novos_ids, 1):
        try:
            det = client.detail(int(qid))
            if det:
                detalhes[qid] = det
        except Exception as e:  # noqa: BLE001
            progress(f"  ERRO detalhe QO {qid}: {str(e)[:100]}")
        if i % 25 == 0:
            progress(f"  detalhes {i}/{len(novos_ids)}")
    CQ.write_details_cache(CQ.DEFAULT_DETAILS_JSON, detalhes)

    # existing_rows no formato CSV-like (âncoras RICD + dedupe de row_keys)
    linhas = estado["por_norma"].get(RICD_ID, [])
    existing_rows = [{k: str(v) for k, v in ln.items() if not k.startswith("_")}
                     for ln in linhas]
    detalhes_novos = {qid: detalhes[qid] for qid in novos_ids if qid in detalhes}
    # 4500: chunks de QO acima de ~1.500 tokens diluíam a similaridade semântica
    rows, report = CQ.build_qordem_rows(detalhes_novos, existing_rows=existing_rows,
                                        max_text_rag_chars=4500)
    progress(f"QO: {len(rows)} linhas novas geradas ({report.get('anchor_modes')})")

    pid_por_rk = {VM.normalize_ws(ln.get("row_key", "")): ln["_pid"] for ln in linhas}
    planos = []
    seq = estado["max_id"]
    for row in rows:
        seq += 1
        props = _props_qordem(row, seq)
        planos.append((row["row_key"], props, None, "qordem"))
    estado["max_id"] = seq
    criadas = _inserir_lote(planos, hashes, progress=progress) if planos else 0

    manif = _ja_inseridos()
    rel = 0
    for row in rows:
        pid = manif.get(row["row_key"])
        pai_pid = pid_por_rk.get(VM.normalize_ws(row.get("dispositivo_pai_row_key", "")))
        if pid and pai_pid:
            try:
                L.patch_page_props(pid, {"dispositivo_pai_relacao": {"relation": [{"id": pai_pid}]}})
                rel += 1
            except Exception:  # noqa: BLE001
                pass
    progress(f"QO: criadas {criadas}, relações {rel}")
    return {"norma": "qordem", "novos": len(novos_ids), "criadas": criadas}


_QO_NUM = {"id", "ordem_global", "norma_ano", "id_fonte", "ordem_doc", "qtd_notas",
           "chars", "tokens_estimados"}
_QO_CHECK = {"linha_sintetica", "incluir_no_rag"}
_QO_SELECT = {"tipo_linha", "tipo_base", "grupo", "fonte_tipo", "tipo_dispositivo",
              "qualidade_texto"}


def _props_qordem(row: Dict[str, str], seq_id: int) -> dict:
    props: Dict[str, Any] = {"Nome": {"title": L.rt(row.get("titulo_linha") or row["row_key"])}}
    for col, val in row.items():
        val = str(val or "")
        if not val or col in ("ordem_doc",):
            if col == "ordem_doc" and val and re.sub(r"\D", "", val):
                props["ordem_doc"] = {"number": int(re.sub(r"\D", "", val)[:9])}
            continue
        if col == "id":
            props["id"] = {"number": seq_id}
        elif col == "ordem_global":
            props["ordem_global"] = {"number": seq_id}
        elif col in _QO_NUM:
            num = re.sub(r"[^\d]", "", val)
            if num:
                props[col] = {"number": int(num[:9])}
        elif col in _QO_CHECK:
            props[col] = {"checkbox": val.lower() == "true"}
        elif col in _QO_SELECT:
            props[col] = {"select": {"name": val}}
        else:
            props[col] = {"rich_text": L.rt_multi(val)}
    # governança das QOs
    props.update({
        "status_alteracao": {"select": {"name": "questao_ordem"}},
        "classe_vigencia_texto": {"select": {"name": "interpretacao"}},
        "usar_como_texto_vigente": {"checkbox": False},
        "prioridade_rag": {"select": {"name": "contexto"}},
        "data_ultima_sincronizacao": {"date": {"start": HOJE}},
        "url_fonte": {"url": row.get("fonte_page_id") or None},
    })
    return props


# ------------------------------------------------------------------ orquestração
def executar(dry_run: bool = True, escopo: str = "tudo",
             normas: Optional[List[str]] = None, limite: Optional[int] = None,
             reindexar: bool = False, refresh_dump: bool = False,
             model: str = MODELO_PADRAO, com_ia: bool = True, forcar: bool = False,
             progress=print) -> dict:
    L.WORK_DIR.mkdir(parents=True, exist_ok=True)
    estado = carregar_base(refresh=refresh_dump, progress=progress)
    hashes = _load_hashes()
    resumo: Dict[str, Any] = {"normas": []}

    if escopo in ("tudo", "planalto", "camara"):
        alvos: List[str] = []
        if normas:
            alvos = [n for n in normas]
        else:
            if escopo in ("tudo", "planalto"):
                extras = F.carregar_normas_extras()
                alvos.extend(n for n in F.urls_planalto() if n in estado["por_norma"]
                             or n in NORMAS_NOVAS or n in extras)
            if escopo in ("tudo", "camara"):
                alvos.extend([RICD_ID, ETICA_ID])
        if limite:
            alvos = alvos[:limite]
        for norma_id in alvos:
            r = sincronizar_norma(norma_id, estado, hashes, dry_run, model,
                                  com_ia=com_ia, forcar=forcar, progress=progress)
            resumo["normas"].append(r)

    if escopo in ("tudo", "qordem"):
        resumo["qordem"] = sincronizar_qordem(estado, hashes, dry_run, progress=progress)

    if not dry_run and reindexar:
        from conle_gerador import notion_rag
        progress("\nReindexando RAG (base vademecum)...")
        notion_rag.indexar(["vademecum"], progress=progress)

    tot_sub = sum(r.get("substantivos", 0) for r in resumo["normas"])
    tot_nov = sum(r.get("novos", 0) for r in resumo["normas"])
    tot_cri = sum(r.get("criadas", 0) for r in resumo["normas"])
    tot_atu = sum(r.get("atualizadas", 0) for r in resumo["normas"])
    progress(f"\nTOTAIS: substantivos={tot_sub} novos={tot_nov} "
             f"criadas={tot_cri} atualizadas={tot_atu}"
             + (f" | QO novas={resumo.get('qordem', {}).get('novos', 0)}" if "qordem" in resumo else ""))
    progress("Dry-run concluído (nada gravado)." if dry_run else "Concluído.")
    return resumo


def main() -> None:
    ap = argparse.ArgumentParser(description="Atualizador da base 'vademecum' (fontes oficiais -> Notion)")
    ap.add_argument("--aplicar", action="store_true", help="grava no Notion (senão, dry-run)")
    ap.add_argument("--escopo", choices=["tudo", "planalto", "camara", "qordem"], default="tudo")
    ap.add_argument("--normas", nargs="*", help="norma_ids específicos")
    ap.add_argument("--limite", type=int, help="máximo de normas neste ciclo")
    ap.add_argument("--reindexar", action="store_true")
    ap.add_argument("--refresh-dump", action="store_true", help="re-baixa o dump da base antes")
    ap.add_argument("--sem-ia", action="store_true")
    ap.add_argument("--forcar", action="store_true", help="ignora o gate de 30%% de divergência")
    ap.add_argument("--model", default=MODELO_PADRAO)
    ap.add_argument("--adicionar-url", metavar="URL",
                    help="registra norma extra pelo link do Planalto (ingestão integral "
                         "no próximo --aplicar --escopo planalto)")
    ap.add_argument("--nome-popular", default="", help="nome popular da norma (c/ --adicionar-url)")
    ap.add_argument("--grupo", default="vademecum", help="grupo da norma (c/ --adicionar-url)")
    args = ap.parse_args()
    if args.adicionar_url:
        meta = F.detectar_norma_planalto(args.adicionar_url)
        meta["norma_nome_popular"] = args.nome_popular
        meta["grupo"] = args.grupo or "vademecum"
        F.adicionar_norma_extra(meta)
        print(f"Registrada: {meta['norma_titulo']} (norma_id={meta['norma_id']})")
        if meta.get("ementa"):
            print(f"Ementa: {meta['ementa'][:300]}")
        print("Ingestão integral na próxima execução: "
              f"python vademecum_updater.py --aplicar --escopo planalto --normas {meta['norma_id']}")
        return
    executar(dry_run=not args.aplicar, escopo=args.escopo, normas=args.normas,
             limite=args.limite, reindexar=args.reindexar, refresh_dump=args.refresh_dump,
             model=args.model, com_ia=not args.sem_ia, forcar=args.forcar)


if __name__ == "__main__":
    main()
