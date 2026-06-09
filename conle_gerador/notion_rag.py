# -*- coding: utf-8 -*-
"""RAG sobre as bases do Notion do usuário (Vademécum, Resoluções TSE, Código
Eleitoral, DJe). Indexa por embeddings (text-embedding-3-small) com cache
incremental em .rag_cache/<base>.jsonl e busca por similaridade de cosseno.

CLI:
  py -m conle_gerador.notion_rag --indexar               # bases padrão
  py -m conle_gerador.notion_rag --indexar --bases dje   # base específica
  py -m conle_gerador.notion_rag --buscar "reserva de lei complementar"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from . import config_gerador as cfg
from . import llm

try:
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None

API = "https://api.notion.com/v1"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {cfg.load_notion_token()}",
        "Notion-Version": cfg.NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _req(method: str, path: str, body: Optional[dict] = None, params=None) -> dict:
    for tentativa in range(5):
        r = requests.request(method, f"{API}{path}", headers=_headers(), params=params,
                             data=json.dumps(body) if body is not None else None, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("Retry-After", "2")))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Notion {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"Notion {method} {path} falhou após retries")


# ---------------------------------------------------------------- extração de texto
def _plain_prop(p: dict) -> str:
    t = p.get("type")
    if t in ("title", "rich_text"):
        return "".join(x.get("plain_text", "") for x in p.get(t, []) or [])
    if t == "select":
        return (p.get("select") or {}).get("name", "") or ""
    if t == "multi_select":
        return ", ".join(o.get("name", "") for o in p.get("multi_select", []) or [])
    if t == "number":
        v = p.get("number")
        return "" if v is None else str(v)
    if t == "url":
        return p.get("url") or ""
    if t == "date":
        return ((p.get("date") or {}) or {}).get("start", "") or ""
    if t == "status":
        return (p.get("status") or {}).get("name", "") or ""
    return ""


def _titulo_pagina(page: dict) -> str:
    for p in (page.get("properties") or {}).values():
        if p.get("type") == "title":
            txt = _plain_prop(p)
            if txt:
                return txt
    return "(sem título)"


# ---------------------------------------------------------------- resolução da base
def _resolver(base: dict):
    db_id = base.get("id")
    if not db_id:
        data = _req("POST", "/search", {"filter": {"property": "object", "value": "database"},
                                        "page_size": 100})
        alvo = (base.get("titulo") or "").strip().lower()
        for o in data.get("results", []):
            titulo = "".join(x.get("plain_text", "") for x in o.get("title", []) or [])
            tl = titulo.strip().lower()
            if any(ex.lower() in tl for ex in cfg.EXCLUIR_TITULOS):
                continue
            if tl == alvo or alvo in tl:
                db_id = o["id"]
                break
        if not db_id:
            raise RuntimeError(f"Base não encontrada por título: {base.get('titulo')!r}")
    meta = _req("GET", f"/databases/{db_id}")
    all_props = meta.get("properties", {})
    props = base.get("props_texto")
    if not props:
        props = [n for n, p in all_props.items() if p.get("type") in ("title", "rich_text")]
    else:
        props = [n for n in props if n in all_props]
        if not props:
            props = [n for n, p in all_props.items() if p.get("type") in ("title", "rich_text")]
    flag = base.get("flag_rag")
    if flag and flag not in all_props:
        flag = None
    # ids das propriedades a baixar (filter_properties) — só o necessário, p/ acelerar.
    prop_ids: List[str] = []
    for nome in props:
        pid = (all_props.get(nome) or {}).get("id")
        if pid and pid not in prop_ids:
            prop_ids.append(pid)
    # colunas de conteúdo/contexto (montagem "conteúdo primeiro")
    for nome in (base.get("props_conteudo") or []) + (base.get("props_contexto") or []):
        pid = (all_props.get(nome) or {}).get("id")
        if pid and pid not in prop_ids:
            prop_ids.append(pid)
    for nome, p in all_props.items():
        if p.get("type") == "title" and p.get("id") and p["id"] not in prop_ids:
            prop_ids.append(p["id"])
    # também baixa as colunas que carregam URLs (link_1/2/3, fonte_page_id, props tipo url),
    # senão o filter_properties as descartaria e perderíamos os links oficiais.
    for nome, p in all_props.items():
        eh_url = p.get("type") == "url" or any(k in nome.lower() for k in ("link", "url", "fonte_page", "href"))
        eh_meta = nome.lower() in _META_COLS
        if (eh_url or eh_meta) and p.get("id") and p["id"] not in prop_ids:
            prop_ids.append(p["id"])
    return db_id, props, flag, prop_ids


def _query_all(db_id: str, flag: Optional[str], *, prop_ids=None, desde: Optional[str] = None,
               limite: Optional[int] = None, progress=None) -> List[dict]:
    results: List[dict] = []
    cursor = None
    # filter_properties reduz MUITO o payload (baixa ~5 props em vez de 40-55).
    params = [("filter_properties", pid) for pid in (prop_ids or [])]
    filtros: List[dict] = []
    if flag:
        filtros.append({"property": flag, "checkbox": {"equals": True}})
    if desde:
        filtros.append({"timestamp": "last_edited_time", "last_edited_time": {"on_or_after": desde}})
    while True:
        body: Dict[str, Any] = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        if len(filtros) == 1:
            body["filter"] = filtros[0]
        elif len(filtros) > 1:
            body["filter"] = {"and": filtros}
        d = _req("POST", f"/databases/{db_id}/query", body, params=params or None)
        results.extend(d.get("results", []))
        if progress and len(results) % 500 == 0:
            progress(f"   baixados {len(results)} registros...")
        if limite and len(results) >= limite:
            return results[:limite]
        if not d.get("has_more"):
            break
        cursor = d.get("next_cursor")
        if not cursor:
            break
    return results


_DOMINIOS_OFICIAIS = ("planalto.gov.br", "stf.jus.br", "tse.jus.br", "stj.jus.br",
                      "camara.leg.br", "senado.leg.br", "jus.br", "gov.br")

# Colunas de governança a baixar (além das de texto/url) para filtro/rerank/recência.
_META_COLS = {
    "usar_como_texto_vigente", "classe_vigencia_texto", "status_alteracao", "alerta_vigencia",
    "prioridade_rag", "qualidade_texto", "alerta_qualidade",
    "data_sessao", "data_julgamento", "data", "datadecisao", "anoeleicao", "norma_ano",
}


def _meta_da_pagina(page: dict, titulo: str, texto: str) -> dict:
    """Extrai sinais de governança da linha: vigência (bool), prioridade, qualidade e data.
    Combina colunas estruturadas (quando existem) com heurística textual (cancelada/revogado)."""
    pr = page.get("properties") or {}

    def g(nome: str) -> str:
        p = pr.get(nome)
        return _plain_prop(p).strip() if p else ""

    def gb(nome: str):
        p = pr.get(nome)
        if p and p.get("type") == "checkbox":
            return bool(p.get("checkbox"))
        v = g(nome).lower()
        return True if v in ("true", "sim") else (False if v in ("false", "nao", "não") else None)

    vigente = True
    if gb("usar_como_texto_vigente") is False:
        vigente = False
    cls = g("classe_vigencia_texto").lower()
    if cls and cls not in ("oficial_em_vigor", "estrutura", "norma"):
        vigente = False
    st = g("status_alteracao").lower()
    if any(k in st for k in ("revogad", "alterado_substitu", "substituid")):
        vigente = False
    # marcadores FORMAIS de revogação/cancelamento (com parêntese) em qualquer posição do texto
    blob = (titulo + " " + (texto or "")).lower()
    if "(cancelad" in blob or re.search(r"\(\s*revogad", blob):
        vigente = False

    prioridade = (g("prioridade_rag") or "principal").lower()
    qualidade = (g("qualidade_texto") or "ok").lower()
    if g("alerta_qualidade").strip():
        qualidade = "alerta"

    data = ""
    for c in ("data_sessao", "data_julgamento", "data", "dataDecisao"):
        v = g(c)
        if v:
            data = v[:10]
            break
    if not data:
        ano = g("anoEleicao") or g("norma_ano")
        if ano.strip():
            data = ano.strip()[:4]
    return {"vigente": vigente, "prioridade": prioridade, "qualidade": qualidade, "data": data}


def _urls_da_pagina(page: dict) -> str:
    """Extrai a melhor URL das colunas da linha (props type=url + hrefs em rich_text),
    priorizando domínios oficiais. Devolve a 1ª (ou '')."""
    urls: List[str] = []
    for _nome, p in (page.get("properties") or {}).items():
        t = p.get("type")
        if t == "url" and p.get("url"):
            urls.append(p["url"])
        elif t in ("rich_text", "title"):
            for x in p.get(t, []) or []:
                if x.get("href"):
                    urls.append(x["href"])
            # colunas tipo link_1/link_2 guardam a URL como TEXTO puro
            txt = "".join(x.get("plain_text", "") for x in p.get(t, []) or [])
            urls.extend(re.findall(r"https?://[^\s)\]\"']+", txt))
    _IGNORAR = ("youtube.com", "youtu.be", "facebook.", "instagram.", "twitter.", "//x.com")
    seen: List[str] = []
    for u in urls:
        if u and u not in seen and not any(d in u.lower() for d in _IGNORAR):
            seen.append(u)
    seen.sort(key=lambda u: 0 if any(d in u for d in _DOMINIOS_OFICIAIS) else 1)
    return seen[0] if seen else ""


def _texto_da_pagina(page: dict, base: dict) -> str:
    """Monta o texto a EMBEDDAR. Se a base define `props_conteudo`, usa a estratégia
    'conteúdo primeiro' (texto distintivo no início; metadados de contexto ao fim),
    o que melhora a densidade do embedding. Senão, cai no `texto_rag`/`props_texto`."""
    pr = page.get("properties") or {}

    def val(nome: str) -> str:
        p = pr.get(nome)
        return _plain_prop(p).strip() if p else ""

    conteudo = base.get("props_conteudo")
    if conteudo:
        partes = [v for v in (val(c) for c in conteudo) if v]
        texto = "\n".join(partes).strip()
        ctx = [f"{c}: {val(c)}" for c in (base.get("props_contexto") or []) if val(c)]
        if ctx:
            texto += "\n[" + " | ".join(ctx) + "]"
        if len(texto) >= 20:
            return texto[:8000]

    props = base.get("props_texto")
    if not props:
        props = [n for n, p in pr.items() if p.get("type") in ("title", "rich_text")]
    if "texto_rag" in props:
        p = pr.get("texto_rag")
        if p:
            t = _plain_prop(p).strip()
            if len(t) >= 20:
                return t[:8000]
    partes = [val(n) for n in props if n != "texto_rag"]
    return "\n".join(v for v in partes if v).strip()[:8000]


# ---------------------------------------------------------------- indexação
def _meta_load() -> dict:
    p = cfg.RAG_CACHE_DIR / "_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _meta_save(meta: dict) -> None:
    (cfg.RAG_CACHE_DIR / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def indexar(bases: Optional[List[str]] = None, *, limite: Optional[int] = None,
            full: bool = False, progress=None) -> dict:
    """Indexa as bases por embeddings, com cache INCREMENTAL: nas reexecuções só
    baixa/reembeda o que foi editado desde a última indexação (last_edited_time).
    Use full=True para reconstruir do zero."""
    log = progress or (lambda _m: None)
    bases = bases or cfg.BASES_PADRAO
    cfg.RAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = _meta_load()
    resumo = {}
    for chave in bases:
        base = cfg.bases_registry().get(chave)
        if not base:
            log(f"[{chave}] base desconhecida; pulando.")
            continue
        log(f"[{base['label']}] resolvendo base...")
        db_id, props, flag, prop_ids = _resolver(base)

        cache_path = cfg.RAG_CACHE_DIR / f"{chave}.jsonl"
        existing: Dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    existing[r["page_id"]] = r
                except Exception:  # noqa: BLE001
                    pass

        desde = None if (full or not existing) else (meta.get(chave) or {}).get("last_indexed")
        t_inicio = datetime.now(timezone.utc).isoformat()
        modo = "completa" if not desde else f"incremental (desde {desde[:10]})"
        log(f"[{base['label']}] indexação {modo}; baixando registros...")
        pages = _query_all(db_id, flag, prop_ids=prop_ids, desde=desde, limite=limite, progress=log)
        log(f"[{base['label']}] {len(pages)} registros a processar (cache atual: {len(existing)}).")

        registros: Dict[str, dict] = dict(existing)  # mantém os não modificados
        novos_textos: List[str] = []
        novos_keys: List[str] = []
        for pg in pages:
            txt = _texto_da_pagina(pg, base)
            if not txt:
                registros.pop(pg["id"], None)
                continue
            titulo = _titulo_pagina(pg)
            url = _urls_da_pagina(pg)
            # NÃO reaproveitar o nome `meta` (é o dict global do _meta.json): usar meta_pg
            meta_pg = _meta_da_pagina(pg, titulo, txt)
            h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
            old = existing.get(pg["id"])
            if old and old.get("hash") == h and old.get("vetor"):
                rec = dict(old)
                rec["url"] = url  # recaptura url + metadados de governança sem custo de embedding
                rec.update(meta_pg)
                registros[pg["id"]] = rec
            else:
                registros[pg["id"]] = {"page_id": pg["id"], "fonte": base["label"],
                                        "titulo": titulo, "texto": txt[:8000],
                                        "url": url, "hash": h, "vetor": None, **meta_pg}
                novos_textos.append(txt[:8000])
                novos_keys.append(pg["id"])

        if novos_textos:
            log(f"[{base['label']}] gerando embeddings de {len(novos_textos)} novos/alterados (reaproveita {len(registros) - len(novos_textos)})...")
            B = 96
            for i in range(0, len(novos_textos), B):
                vetores = llm.embed(novos_textos[i:i + B])
                for k, v in zip(novos_keys[i:i + B], vetores):
                    registros[k]["vetor"] = v
                log(f"   embeddings {min(i + B, len(novos_textos))}/{len(novos_textos)}")
        else:
            log(f"[{base['label']}] nada novo desde a última indexação.")

        regs = [r for r in registros.values() if r.get("vetor")]
        with cache_path.open("w", encoding="utf-8") as f:
            for r in regs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta[chave] = {"last_indexed": t_inicio, "n": len(regs), "label": base["label"]}
        _meta_save(meta)
        resumo[chave] = len(regs)
        log(f"[{base['label']}] indexado: {len(regs)} trechos -> {cache_path.name}")
    return resumo


# ---------------------------------------------------------------- busca
@dataclass
class Trecho:
    fonte: str
    titulo: str
    texto: str
    score: float
    url: str = ""
    vigente: bool = True
    prioridade: str = "principal"
    data: str = ""


# Pesos de re-ranking por prioridade_rag (curadoria já existente nas bases).
_PESO_PRIORIDADE = {"principal": 1.0, "secundario": 0.9, "comparativo": 0.8, "contexto": 0.8,
                    "auditoria": 0.45, "revisar": 0.5, "": 0.9}


def _ajuste_score(r: dict) -> float:
    p = _PESO_PRIORIDADE.get((r.get("prioridade") or "principal"), 0.9)
    if (r.get("qualidade") or "ok") != "ok":
        p *= 0.85
    return p


def _load_cache(bases: List[str]) -> List[dict]:
    recs: List[dict] = []
    for chave in bases:
        p = cfg.RAG_CACHE_DIR / f"{chave}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("vetor"):
                    recs.append(r)
            except Exception:  # noqa: BLE001
                pass
    return recs


def buscar(query: str, *, k: int = 8, bases: Optional[List[str]] = None,
           somente_vigente: bool = True) -> List[Trecho]:
    bases = bases or cfg.BASES_PADRAO
    recs = _load_cache(bases)
    if somente_vigente:
        recs = [r for r in recs if r.get("vigente", True)]  # exclui revogados/cancelados/redação anterior
    if not recs:
        return []
    qv = llm.embed([query])[0]
    if _np is not None:
        M = _np.array([r["vetor"] for r in recs], dtype="float32")
        q = _np.array(qv, dtype="float32")
        M /= (_np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        q /= (_np.linalg.norm(q) + 1e-9)
        sims = (M @ q).tolist()
    else:
        qn = math.sqrt(sum(x * x for x in qv)) + 1e-9
        sims = []
        for r in recs:
            v = r["vetor"]
            dot = sum(a * b for a, b in zip(qv, v))
            vn = math.sqrt(sum(b * b for b in v)) + 1e-9
            sims.append(dot / (qn * vn))
    # re-ranking: score_final = cosseno x peso(prioridade_rag/qualidade)
    scored = sorted(((sims[i] * _ajuste_score(recs[i]), sims[i], recs[i]) for i in range(len(recs))),
                    key=lambda x: x[0], reverse=True)
    return [Trecho(r["fonte"], r["titulo"], r["texto"], float(cos), r.get("url", ""),
                   r.get("vigente", True), r.get("prioridade", "principal"), r.get("data", ""))
            for _sc, cos, r in scored[:k]]


def _main() -> int:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexar", action="store_true")
    ap.add_argument("--full", action="store_true", help="reconstrói o índice do zero")
    ap.add_argument("--buscar", type=str, default="")
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    if args.indexar:
        r = indexar(args.bases, limite=args.limite, full=args.full, progress=print)
        print("RESUMO:", r)
    if args.buscar:
        for t in buscar(args.buscar, k=args.k, bases=args.bases):
            print(f"[{t.score:.3f}] {t.fonte} · {t.titulo[:60]}")
            print("   ", t.texto[:160].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
