# -*- coding: utf-8 -*-
"""Atualizador da base Notion 'temas' (Jurisprudência TSE por assunto).

Fluxo: baixa os 22 ramos live de https://temasselecionados.tse.jus.br/ → extrai os
julgados (parser validado de conversores/TEMAS_SELC_csv_to_csv_completadorHTML.py) →
diff contra a base Notion (chave: ramo+processo+data+tipo e pelos links já gravados)
→ enriquece os NOVOS via OpenAI (PROMPT_EXTRACAO, gpt-5.6-luna) → cria as páginas
direto no Notion (relator normalizado pelo de-para de _temas_normalizar_relator,
incluir_no_rag=True) → opcionalmente reindexa o RAG.

Diferença para o fluxo antigo: não para no CSV — escreve no Notion (idempotente por
manifest temas_work/inseridos.jsonl). Auditorias em temas_work/novidades_<data>.csv.

  python temas_updater.py                      # dry-run: relatório de novidades
  python temas_updater.py --aplicar            # enriquece + insere no Notion
  python temas_updater.py --aplicar --limite 1 # insere só 1 (teste fim-a-fim)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import _temas_lib as L
import _temas_normalizar_relator as NR
import conversores.TEMAS_SELC_csv_to_csv_completadorHTML as TS
from conle_gerador import llm

HOME_URL = "https://temasselecionados.tse.jus.br/"
HTML_CACHE = L.WORK_DIR / "html_cache"
CACHE_IA = L.WORK_DIR / "updater_cache.jsonl"
MANIFEST = L.WORK_DIR / "inseridos.jsonl"
PROMPT_PATH = L.ROOT / "conversores" / "TEMAS_SELC_txt_to_csv.py"
MODELO_PADRAO = "gpt-5.6-luna"
PROMPT_VERSION = "v1"

RAMO_LINK_RE = re.compile(
    r"href=[\"'](https://temasselecionados\.tse\.jus\.br/temas-selecionados/[^\"'#]+)[\"']",
    re.IGNORECASE,
)
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)

# Slugs cujo nome no site não bate com a option 'ramo' da base (mapeamento manual)
RAMO_POR_SLUG = {
    "repositorio-decisoes-sobre-enfrentamento-desinformacao-eleitoral":
        "Enfrentamento à desinformação eleitoral",
    "comprovacao-de-receitas-e-gastos-partidarios":
        "Comprovação de receitas e gastos partidários",
}


def _fetch(url: str, tries: int = 5, progress=print) -> str:
    last = None
    for i in range(tries):
        try:
            return TS.fetch_url_text(url)
        except Exception as e:  # noqa: BLE001
            last = e
            progress(f"  ... retry {i + 1} em {url.split('/')[-1] or url}: {str(e)[:80]}")
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Falha ao baixar {url}: {last}")


def baixar_ramos(progress=print) -> List[Tuple[str, str, str]]:
    """Baixa a home + páginas dos ramos. Retorna [(nome_ramo, url, html)].
    HTMLs ficam em temas_work/html_cache/<AAAAMMDD>/ para auditoria/reprocesso."""
    cache_dir = HTML_CACHE / datetime.now().strftime("%Y%m%d")
    cache_dir.mkdir(parents=True, exist_ok=True)
    home = _fetch(HOME_URL, progress=progress)
    urls = sorted(set(RAMO_LINK_RE.findall(home)))
    progress(f"Home: {len(urls)} ramos encontrados.")
    out: List[Tuple[str, str, str]] = []
    for url in urls:
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        html_txt = _fetch(url, progress=progress)
        (cache_dir / f"{slug}.html").write_text(html_txt, encoding="utf-8")
        nome = RAMO_POR_SLUG.get(slug, "")
        if not nome:
            m = H1_RE.search(html_txt)
            nome = TS.clean_html_text(m.group(1)) if m else ""
        if not nome:
            m = TITLE_RE.search(html_txt)
            if m:
                nome = TS.clean_html_text(m.group(1)).split("—")[0].strip()
        nome = nome or slug
        out.append((nome, url, html_txt))
        progress(f"  baixado: {nome}")
    return out


def _mdy_to_iso(mdy: str) -> str:
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", (mdy or "").strip())
    if not m:
        return ""
    mm, d, y = map(int, m.groups())
    try:
        return f"{y:04d}-{mm:02d}-{d:02d}"
    except ValueError:
        return ""


def _chave_pagina(props: dict) -> Tuple[str, str, str, str]:
    data = str(L.plain(props.get("data_julgamento")) or "")[:10]
    return (
        TS.normalize_key(str(L.plain(props.get("ramo")) or "")),
        TS.normalize_process_number(str(L.plain(props.get("numero_processo")) or "")),
        data,
        TS.normalize_tipo_key(str(L.plain(props.get("tipo_de_processo")) or "")),
    )


def _chave_candidato(c: "TS.Candidate") -> Tuple[str, str, str, str]:
    return (
        TS.normalize_key(c.ramo),
        TS.normalize_process_number(c.numero_norm),
        _mdy_to_iso(c.data_mdy),
        TS.normalize_tipo_key(c.tipo),
    )


def carregar_existentes(progress=print) -> dict:
    progress("Baixando o estado atual da base 'temas' no Notion...")
    props_nomes = ["tema", "ramo", "numero_processo", "data_julgamento",
                   "tipo_de_processo", "numero_no_ramo", "link_1", "link_2", "link_3"]
    pages = L.query_all(only_props=props_nomes, progress=progress)
    chaves, chaves_frouxas, hrefs = set(), set(), set()
    max_id = 0
    max_no_ramo: Dict[str, float] = {}
    for p in pages:
        pr = p.get("properties") or {}
        ch = _chave_pagina(pr)
        chaves.add(ch)
        chaves_frouxas.add(ch[:3])  # sem o tipo: site e base divergem na classe às vezes
        for lk in ("link_1", "link_2", "link_3"):
            u = L.plain(pr.get(lk))
            if u:
                hrefs.add(str(u).strip())
        v = L.plain(pr.get("ID"))
        if isinstance(v, (int, float)):
            max_id = max(max_id, int(v))
        rk = TS.normalize_key(str(L.plain(pr.get("ramo")) or ""))
        n = L.plain(pr.get("numero_no_ramo"))
        if isinstance(n, (int, float)):
            max_no_ramo[rk] = max(max_no_ramo.get(rk, 0), int(n))
    progress(f"Base atual: {len(pages)} páginas, {len(hrefs)} links, ID máx = {max_id}.")
    return {"n": len(pages), "chaves": chaves, "chaves_frouxas": chaves_frouxas,
            "hrefs": hrefs, "max_id": max_id, "max_no_ramo": max_no_ramo}


def _schema() -> dict:
    return L.req("GET", f"/databases/{L.DB_ID}")


def _mapear_ramo(nome_live: str, ramo_options: List[str]) -> str:
    alvo = TS.normalize_key(nome_live)
    for opt in ramo_options:
        if TS.normalize_key(opt) == alvo:
            return opt
    return nome_live


def _depara_relator() -> dict:
    """chave normalizada -> canônico (do CSV do de-para, se existir)."""
    out = {}
    if NR.DEPARA_CSV.exists():
        for g, c in NR.carregar_depara().items():
            out[NR.chave(g)] = c
    return out


DESCONHECIDOS_CSV = L.WORK_DIR / "relator_desconhecidos.csv"
_desconhecidos_vistos: set = set()


def _normalizar_relator(grafia: str, depara: dict, progress=None) -> str:
    """Saneia sufixos de função (', red', ', rel', 'e o'...) e aplica o de-para.
    GATE: grafia fora do de-para é logada em relator_desconhecidos.csv e avisada —
    é assim que variações novas são capturadas em vez de sujar a base em silêncio."""
    limpo = NR.sanear(grafia)
    if not limpo:
        return ""
    canon = depara.get(NR.chave(limpo))
    if canon:
        return canon
    forma = NR._sem_prefixo(limpo)
    if forma and forma not in _desconhecidos_vistos:
        _desconhecidos_vistos.add(forma)
        with open(DESCONHECIDOS_CSV, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()};{grafia};{forma}\n")
        if progress:
            progress(f"  AVISO: relator fora do de-para: {grafia!r} -> gravando {forma!r} "
                     f"(revisar em {DESCONHECIDOS_CSV.name})")
    return forma


# ------------------------------------------------------------------ IA
def _load_cache_ia() -> dict:
    cache = {}
    if CACHE_IA.exists():
        for ln in open(CACHE_IA, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                cache[r["k"]] = r
    return cache


def enriquecer(cands: List["TS.Candidate"], model: str, progress=print) -> Dict[str, dict]:
    """href -> {tema, contexto, tese, punchline, bullet_points} (cache idempotente;
    geração em paralelo — 8 threads — como nos _*_enriquecer)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    prompt_base = TS.load_prompt_from_temas_script(PROMPT_PATH)
    cache = _load_cache_ia()
    out: Dict[str, dict] = {}
    pend: List[tuple] = []
    for c in cands:
        texto = c.texto_original or ""
        if c.citacao_raw:
            texto = f"{texto}\n({c.citacao_raw})"
        k = hashlib.sha1((PROMPT_VERSION + "|" + c.href + "|" + texto).encode("utf-8")).hexdigest()
        if k in cache:
            out[c.href] = cache[k]["campos"]
        else:
            pend.append((c, texto, k))
    progress(f"  IA: {len(out)} em cache, {len(pend)} a gerar")
    lock = threading.Lock()
    feito = [0]

    def _gerar(item):
        try:
            _gerar_impl(item)
        except Exception as e:  # noqa: BLE001
            with lock:
                progress(f"  IA ERRO {item[0].href[-30:]}: {str(e)[:120]}")

    def _gerar_impl(item):
        c, texto, k = item
        user = prompt_base.format(texto_julgado=texto)
        data = llm.chat("Você é um assistente de jurimetria especializado em jurisprudência "
                        "eleitoral brasileira. Responda apenas em JSON válido.",
                        user, json_mode=True, model=model, max_output_tokens=2000)
        if not isinstance(data, dict):
            data = {}
        data = {str(kk).strip().lower(): vv for kk, vv in data.items()}
        # o nano às vezes aninha os campos analíticos dentro de casos[0]
        casos = data.get("casos")
        if isinstance(casos, list) and casos and isinstance(casos[0], dict):
            aninhado = {str(kk).strip().lower(): vv for kk, vv in casos[0].items()}
            for kk in ("tema", "contexto", "tese", "punchline", "bullet_points"):
                if not data.get(kk) and aninhado.get(kk):
                    data[kk] = aninhado[kk]
        campos = TS.parse_api_analytics(data)
        with lock:
            with open(CACHE_IA, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "href": c.href, "campos": campos},
                                   ensure_ascii=False) + "\n")
            out[c.href] = campos
            feito[0] += 1
            if feito[0] % 100 == 0 or feito[0] == len(pend):
                progress(f"  IA {feito[0]}/{len(pend)}: {campos.get('tema', '')[:70]}")

    if pend:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for item in pend:
                ex.submit(_gerar, item)
    return out


# ------------------------------------------------------------------ inserção
def _ja_inseridos() -> set:
    out = set()
    if MANIFEST.exists():
        for ln in open(MANIFEST, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                out.add(json.loads(ln)["href"])
    return out


def inserir(novos: List["TS.Candidate"], campos_ia: Dict[str, dict], estado: dict,
            schema: dict, progress=print) -> int:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    depara = _depara_relator()
    tipo_relator = schema["properties"].get("relator", {}).get("type", "rich_text")
    ramo_opts = [o["name"] for o in
                 schema["properties"].get("ramo", {}).get("select", {}).get("options", [])]
    feitos = _ja_inseridos()
    pend = [c for c in novos if c.href not in feitos]

    # numeração pré-atribuída (determinística) antes do pool
    seq_id = estado["max_id"]
    planos = []
    for c in pend:
        ia = campos_ia.get(c.href) or {}
        ramo = _mapear_ramo(c.ramo, ramo_opts)
        rk = TS.normalize_key(ramo)
        seq_id += 1
        estado["max_no_ramo"][rk] = estado["max_no_ramo"].get(rk, 0) + 1
        props: dict = {
            "tema": {"title": L.rt(ia.get("tema") or c.citacao_raw[:200])},
            "ID": {"number": seq_id},
            "numero_no_ramo": {"number": estado["max_no_ramo"][rk]},
            "ramo": {"select": {"name": ramo}},
            "incluir_no_rag": {"checkbox": True},
            "link_1": {"url": c.href},
        }
        rel = _normalizar_relator(c.relator, depara) if c.relator else ""
        if rel:
            props["relator"] = ({"select": {"name": rel}} if tipo_relator == "select"
                                else {"rich_text": L.rt(rel)})
        num = TS.normalize_process_number(c.numero_norm)
        if num.isdigit():
            props["numero_processo"] = {"number": int(num)}
        iso = _mdy_to_iso(c.data_mdy)
        if iso:
            props["data_julgamento"] = {"date": {"start": iso}}
        for campo, valor in (("tipo_de_processo", c.tipo), ("subramo", c.subramo),
                             ("texto_original", c.texto_original),
                             ("contexto", ia.get("contexto")), ("tese", ia.get("tese")),
                             ("punchline", ia.get("punchline")),
                             ("bullet_points", ia.get("bullet_points"))):
            if valor:
                props[campo] = {"rich_text": L.rt(str(valor))}
        planos.append((c, props))

    lock = threading.Lock()
    cont = {"ok": 0, "erro": 0}

    def _criar(item):
        c, props = item
        try:
            pg = L.create_page(props)
            with lock:
                with open(MANIFEST, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"href": c.href, "page_id": pg["id"],
                                        "quando": datetime.now().isoformat()}) + "\n")
                cont["ok"] += 1
                if cont["ok"] % 100 == 0 or cont["ok"] == len(planos):
                    progress(f"  criadas {cont['ok']}/{len(planos)}")
        except Exception as e:  # noqa: BLE001
            with lock:
                cont["erro"] += 1
                progress(f"  ERRO criar {c.href[-30:]}: {str(e)[:120]}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for item in planos:
            ex.submit(_criar, item)
    if cont["erro"]:
        progress(f"  erros de criação: {cont['erro']} (re-rodar --aplicar retoma pelo manifest)")
    return cont["ok"]


# ------------------------------------------------------------------ orquestração
def executar(dry_run: bool = True, limite: Optional[int] = None, reindexar: bool = False,
             model: str = MODELO_PADRAO, progress=print) -> dict:
    L.WORK_DIR.mkdir(parents=True, exist_ok=True)
    ramos = baixar_ramos(progress=progress)

    candidatos: List[TS.Candidate] = []
    for nome, _url, html_txt in ramos:
        cs = TS.extract_candidates_from_html(nome, html_txt, "live")
        candidatos.extend(cs)
        progress(f"  {nome}: {len(cs)} julgados no site")

    # dedupe interna por href (fica o candidato mais completo)
    por_href: Dict[str, TS.Candidate] = {}
    for c in candidatos:
        atual = por_href.get(c.href)
        if atual is None or TS.candidate_rank(c) > TS.candidate_rank(atual):
            por_href[c.href] = c
    candidatos = list(por_href.values())

    estado = carregar_existentes(progress=progress)
    novos, incompletos = [], []
    for c in candidatos:
        ch = _chave_candidato(c)
        if (c.href in estado["hrefs"] or ch in estado["chaves"]
                or ch[:3] in estado["chaves_frouxas"]):
            continue
        ok, faltando = TS.candidate_has_full_key(c)
        (novos if ok else incompletos).append(c)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    cols = ["ramo", "subramo", "tipo", "numero_processo", "data", "relator", "href",
            "citacao", "texto_original"]

    def _linha(c: TS.Candidate) -> dict:
        return {"ramo": c.ramo, "subramo": c.subramo, "tipo": c.tipo,
                "numero_processo": c.numero_norm, "data": _mdy_to_iso(c.data_mdy),
                "relator": c.relator, "href": c.href, "citacao": c.citacao_raw,
                "texto_original": c.texto_original[:500]}

    nov_csv = L.WORK_DIR / f"novidades_{stamp}.csv"
    with open(nov_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=";")
        w.writeheader()
        w.writerows(_linha(c) for c in novos)
    if incompletos:
        with open(L.WORK_DIR / f"incompletos_{stamp}.csv", "w", encoding="utf-8-sig",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter=";")
            w.writeheader()
            w.writerows(_linha(c) for c in incompletos)

    resumo = {"site": len(candidatos), "base": estado["n"], "novos": len(novos),
              "incompletos": len(incompletos), "criadas": 0, "csv": str(nov_csv)}
    progress(f"\nSite: {resumo['site']} julgados | Base: {resumo['base']} | "
             f"NOVOS: {resumo['novos']} | incompletos: {resumo['incompletos']}")
    progress(f"Relatório: {nov_csv}")

    if dry_run or not novos:
        if not dry_run and reindexar:
            from conle_gerador import notion_rag
            notion_rag.indexar(["temas"], progress=progress)
        return resumo

    alvo = novos[:limite] if limite else novos
    progress(f"\nEnriquecendo {len(alvo)} novos via {model}...")
    campos_ia = enriquecer(alvo, model, progress=progress)
    schema = _schema()
    progress("Criando páginas no Notion...")
    resumo["criadas"] = inserir(alvo, campos_ia, estado, schema, progress=progress)
    progress(f"Criadas: {resumo['criadas']}.")

    if reindexar:
        from conle_gerador import notion_rag
        progress("\nReindexando RAG (base temas)...")
        notion_rag.indexar(["temas"], progress=progress)
    return resumo


def main() -> None:
    ap = argparse.ArgumentParser(description="Atualizador da base 'temas' (site TSE -> Notion)")
    ap.add_argument("--aplicar", action="store_true", help="enriquece e insere (senão, dry-run)")
    ap.add_argument("--limite", type=int, help="insere no máximo N novos (teste)")
    ap.add_argument("--reindexar", action="store_true", help="reindexa o RAG ao final")
    ap.add_argument("--model", default=MODELO_PADRAO)
    args = ap.parse_args()
    executar(dry_run=not args.aplicar, limite=args.limite,
             reindexar=args.reindexar, model=args.model)


if __name__ == "__main__":
    main()
