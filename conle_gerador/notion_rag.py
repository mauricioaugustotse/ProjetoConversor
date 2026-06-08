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
    for nome, p in all_props.items():
        if p.get("type") == "title" and p.get("id") and p["id"] not in prop_ids:
            prop_ids.append(p["id"])
    # também baixa as colunas que carregam URLs (link_1/2/3, fonte_page_id, props tipo url),
    # senão o filter_properties as descartaria e perderíamos os links oficiais.
    for nome, p in all_props.items():
        eh_url = p.get("type") == "url" or any(k in nome.lower() for k in ("link", "url", "fonte_page", "href"))
        if eh_url and p.get("id") and p["id"] not in prop_ids:
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
    seen: List[str] = []
    for u in urls:
        if u and u not in seen:
            seen.append(u)
    seen.sort(key=lambda u: 0 if any(d in u for d in _DOMINIOS_OFICIAIS) else 1)
    return seen[0] if seen else ""


def _texto_da_pagina(page: dict, props: List[str]) -> str:
    pr = page.get("properties") or {}
    # `texto_rag` (quando presente) já consolida documento/norma/dispositivo p/ retrieval.
    if "texto_rag" in props:
        p = pr.get("texto_rag")
        if p:
            t = _plain_prop(p).strip()
            if len(t) >= 20:
                return t[:8000]
    partes: List[str] = []
    for nome in props:
        if nome == "texto_rag":
            continue
        p = pr.get(nome)
        if not p:
            continue
        txt = _plain_prop(p).strip()
        if txt:
            partes.append(txt)
    return "\n".join(partes).strip()


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
            txt = _texto_da_pagina(pg, props)
            if not txt:
                registros.pop(pg["id"], None)
                continue
            url = _urls_da_pagina(pg)
            h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
            old = existing.get(pg["id"])
            if old and old.get("hash") == h and old.get("vetor"):
                rec = dict(old)
                rec["url"] = url  # captura/atualiza a URL reaproveitando o vetor (sem custo)
                registros[pg["id"]] = rec
            else:
                registros[pg["id"]] = {"page_id": pg["id"], "fonte": base["label"],
                                        "titulo": _titulo_pagina(pg), "texto": txt[:8000],
                                        "url": url, "hash": h, "vetor": None}
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


def buscar(query: str, *, k: int = 8, bases: Optional[List[str]] = None) -> List[Trecho]:
    bases = bases or cfg.BASES_PADRAO
    recs = _load_cache(bases)
    if not recs:
        return []
    qv = llm.embed([query])[0]
    if _np is not None:
        M = _np.array([r["vetor"] for r in recs], dtype="float32")
        q = _np.array(qv, dtype="float32")
        M /= (_np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        q /= (_np.linalg.norm(q) + 1e-9)
        sims = M @ q
        ordem = sims.argsort()[::-1][:k]
        return [Trecho(recs[i]["fonte"], recs[i]["titulo"], recs[i]["texto"], float(sims[i]),
                       recs[i].get("url", "")) for i in ordem]
    # fallback python puro
    qn = math.sqrt(sum(x * x for x in qv)) + 1e-9
    scored = []
    for r in recs:
        v = r["vetor"]
        dot = sum(a * b for a, b in zip(qv, v))
        vn = math.sqrt(sum(b * b for b in v)) + 1e-9
        scored.append((dot / (qn * vn), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [Trecho(r["fonte"], r["titulo"], r["texto"], float(s), r.get("url", "")) for s, r in scored[:k]]


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
