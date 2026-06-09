# -*- coding: utf-8 -*-
"""Pipeline do gerador: demanda -> (RAG + Câmara + Gemini) -> IT + minuta -> Notion.

A página é montada na MESMA anatomia que conle_conversor/splitter.py reconhece, de
modo que o conversor existente gera os .docx a partir dela.
"""
from __future__ import annotations

import datetime
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from conle_conversor import classifier, config as cconf

from . import camara_api, config_gerador as cfg, gemini_web, llm, notion_rag, prompts
from . import notion_writer as nw


@dataclass
class Resultado:
    page_id: str
    titulo: str
    tipo_sigla: str
    n_blocos: int
    fontes_web: List[str] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)


# ============================ etapas ============================
def analisar_demanda(demanda: str, *, model: Optional[str] = None, log: Callable = print) -> dict:
    log("Analisando a demanda (espécie, tema, dispositivos, palavras-chave)…")
    data = llm.chat(prompts.SYS_ANALISE, f"DEMANDA:\n{demanda}", json_mode=True, model=model)
    if not isinstance(data, dict) or not data.get("tema"):
        data = {"tema": demanda[:80], "tipo_sigla": "PL", "objeto": demanda,
                "palavras_chave_camara": [], "consultas_rag": [demanda[:120]], "consultas_web": []}
    return data


def _fmt_rag(trechos) -> str:
    linhas = []
    for t in trechos:
        meta = []
        if getattr(t, "data", ""):
            meta.append(f"data {t.data}")
        if getattr(t, "url", ""):
            meta.append(f"LINK OFICIAL: {t.url}")
        tag = ("  [" + " | ".join(meta) + "]") if meta else ""
        linhas.append(f"- ({t.fonte} · {t.titulo}){tag}\n  {t.texto[:800]}")
    return "\n".join(linhas) if linhas else "(sem trechos relevantes nas bases internas)"


def _fmt_camara(props: List[dict]) -> str:
    linhas = []
    for p in props:
        sit = f" — situação: {p['situacao']}" if p.get("situacao") else ""
        aut = f" (autor: {p['autor']})" if p.get("autor") else ""
        linhas.append(f"- {p['sigla']} {p['numero']}/{p['ano']}{aut}: {p['ementa'][:240]}{sit} | {p['url']}")
    return "\n".join(linhas) if linhas else "(nenhuma proposição correlata localizada)"


def _fmt_web(web: List[dict]) -> str:
    blocos = []
    for w in web:
        if w.get("texto"):
            fontes = "; ".join(w.get("fontes", [])[:5])
            blocos.append(f"{w['texto']}\nFontes: {fontes}")
    return "\n\n".join(blocos) if blocos else "(sem resultados de pesquisa web)"


_FONTE_CAT = None


def _cat_fonte(label: str) -> str:
    global _FONTE_CAT
    if _FONTE_CAT is None:
        _FONTE_CAT = {b.get("label"): b.get("categoria", "normativo")
                     for b in cfg.bases_registry().values()}
    return _FONTE_CAT.get(label, "normativo")


def _cota_por_categoria(trechos, *, n_total: int = 14, max_juris: int = 6):
    """Garante convivência: jurisprudência (temas/sessões/DJe) não pode engolir o top-k —
    no máximo `max_juris` trechos; o restante é normativo; completa se sobrar espaço."""
    juris = [t for t in trechos if _cat_fonte(t.fonte) == "jurisprudencia"][:max_juris]
    norm = [t for t in trechos if _cat_fonte(t.fonte) != "jurisprudencia"]
    out = norm[: n_total - len(juris)] + juris
    if len(out) < n_total:
        out += [t for t in trechos if t not in out][: n_total - len(out)]
    return out[:n_total]


def coletar_contexto(
    analise: dict,
    *,
    usar_rag: bool = True,
    usar_camara: bool = True,
    usar_web: bool = True,
    bases_rag: Optional[List[str]] = None,
    tipo_sigla: Optional[str] = None,
    somente_vigente: bool = True,
    log: Callable = print,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"rag": [], "camara": [], "web": [], "links": None}

    if usar_rag:
        # consultas: as gerais + dirigidas aos dispositivos (legislação) e julgados (jurisprudência),
        # para que a fundamentação cubra os DOIS eixos com profundidade ("valorizar as tintas").
        consultas = list(analise.get("consultas_rag") or [])
        consultas += [f"dispositivo legal: {d}" for d in (analise.get("dispositivos_chave") or [])[:3]]
        consultas += [f"jurisprudência/tese: {j}" for j in (analise.get("julgados_chave") or [])[:3]]
        if not consultas:
            consultas = [analise.get("tema", "")]
        vistos = set()
        trechos = []
        for q in consultas[:7]:
            log(f"RAG: consultando bases internas — “{q[:60]}”")
            for t in notion_rag.buscar(q, k=6, bases=bases_rag, somente_vigente=somente_vigente):
                chave = (t.titulo, t.texto[:40])
                if chave not in vistos:
                    vistos.add(chave)
                    trechos.append(t)
        ctx["rag"] = _cota_por_categoria(trechos, n_total=14, max_juris=6)
        nj = sum(1 for t in ctx["rag"] if _cat_fonte(t.fonte) == "jurisprudencia")
        log(f"RAG: {len(ctx['rag'])} trechos ({len(ctx['rag']) - nj} normativos, {nj} jurisprudência).")

    if usar_camara:
        kws = analise.get("palavras_chave_camara") or []
        rel = kws + [analise.get("tema", "")]
        log(f"Câmara: buscando proposições correlatas ({', '.join(kws[:6])})…")
        ctx["camara"] = camara_api.proposicoes_correlatas(
            kws, sigla=tipo_sigla, norma_alvo=analise.get("norma_alvo", ""),
            tema=analise.get("tema", ""), termos_relevancia=rel, max_total=12, log=log)
        log(f"Câmara: {len(ctx['camara'])} proposições correlatas.")

    if usar_web:
        # ECONOMIA: 1 ÚNICA chamada grounded agrupando as perguntas (o grounding é cobrado por
        # requisição, não por pergunta). A coleta de links é feita SÓ na 2ª passada dirigida
        # (_enriquecer_links), que sabe o que o redator efetivamente citou — sem a antiga "qlinks"
        # preventiva (redundante). Total Gemini por geração: 2 (esta + o enriquecimento).
        perguntas = [q for q in (analise.get("consultas_web") or []) if q and q.strip()][:4]
        if perguntas:
            log(f"Web (Gemini): 1 consulta agrupada — {len(perguntas)} perguntas (economia de grounding)…")
            r = gemini_web.pesquisar_lote(perguntas, foco=analise.get("tema"))
            if r.get("ok"):
                ctx["web"].append(r)
            elif r.get("erro"):
                log(f"   web indisponível: {r['erro'][:80]}")
    return ctx


def _contexto_txt(ctx: dict) -> str:
    links = ctx.get("links") or {}
    if links.get("texto"):
        links_txt = links["texto"] + "\nFontes: " + "; ".join(links.get("fontes", [])[:8])
    else:
        links_txt = "(sem coleta dedicada de links — use a TABELA DE LINKS oficiais e o que houver na pesquisa web)"
    return (
        "=== TRECHOS DAS BASES INTERNAS (RAG) ===\n" + _fmt_rag(ctx.get("rag", [])) +
        "\n\n=== PROPOSIÇÕES NA CÂMARA (API oficial) ===\n" + _fmt_camara(ctx.get("camara", [])) +
        "\n\n=== PESQUISA WEB (Gemini grounded) ===\n" + _fmt_web(ctx.get("web", [])) +
        "\n\n=== REFERÊNCIAS COM LINKS OFICIAIS (use estas URLs para linkar a 1ª menção de cada "
        "precedente e norma; não invente URLs) ===\n" + links_txt
    )


def redigir_it(demanda: str, analise: dict, ctx: dict, *, model=None, log: Callable = print) -> dict:
    log("Redigindo o corpo da Informação Técnica (seções 1–5 e 7)…")
    user = (
        f"DEMANDA:\n{demanda}\n\nANÁLISE:\n{json.dumps(analise, ensure_ascii=False)}\n\n"
        f"CONTEXTO:\n{_contexto_txt(ctx)}"
    )
    it = llm.chat(prompts.SYS_IT, user, json_mode=True, model=model)
    return it if isinstance(it, dict) else {}


def redigir_secao6(analise: dict, camara: List[dict], *, model=None, log: Callable = print) -> dict:
    # gera a Seção 6 mesmo sem correlatas (dirá que não foram localizadas proposições semelhantes)
    log("Redigindo a Seção 6 (proposições correlatas / risco de apensação)…")
    user = (
        f"TEMA: {analise.get('tema')}\nESPÉCIE: {analise.get('tipo_sigla')}\n\n"
        f"PROPOSIÇÕES (dados reais da Câmara):\n{_fmt_camara(camara)}"
    )
    s6 = llm.chat(prompts.SYS_SECAO6, user, json_mode=True, model=model)
    return s6 if isinstance(s6, dict) else {}


def redigir_minuta(demanda: str, analise: dict, ctx: dict, tipo, it: Optional[dict] = None,
                   *, model=None, log: Callable = print) -> dict:
    log("Redigindo a minuta (ementa, articulado, justificativa)…")
    # passa o corpo já redigido da IT para a minuta REFLETIR exatamente os mesmos
    # dispositivos (coerência IT <-> minuta <-> justificativa).
    it_resumo = ""
    if it:
        it_resumo = (
            "CORPO DA IT JÁ REDIGIDO (a minuta DEVE alterar os MESMOS dispositivos citados aqui, e a "
            "justificativa DEVE refleti-los):\n"
            f"- Mapeamento (Seção 4): {json.dumps(it.get('mapeamento', {}), ensure_ascii=False)}\n"
            f"- Análise (Seção 5): {json.dumps(it.get('analise_solicitacao', {}), ensure_ascii=False)}\n"
            f"- Conclusão (Seção 7): {json.dumps(it.get('conclusao', []), ensure_ascii=False)}\n\n"
        )
    user = (
        f"DEMANDA:\n{demanda}\n\nESPÉCIE: {tipo.sigla} — {tipo.nome_extenso}\n"
        f"ANÁLISE:\n{json.dumps(analise, ensure_ascii=False)}\n\n{it_resumo}CONTEXTO:\n{_contexto_txt(ctx)}"
    )
    m = llm.chat(prompts.SYS_MINUTA, user, json_mode=True, model=model)
    return m if isinstance(m, dict) else {}


# ============================ enriquecimento de links (2ª passada) ============================
# Precedentes/súmulas/temas citáveis — para detectar referências SEM link no texto redigido.
# \b evita casar "RE" dentro de "Supremo"/"sobre"; o tipo exige número logo após.
_RE_REF = re.compile(
    r"\b(?:ADI|ADC|ADPF|ADO|ARE|RE|REsp|AREsp|HC|MS|MI|PET|Rcl|MC)\s*n?[ºo°.]*\s*\d[\d.]*(?:/[A-Z]{2})?"
    r"|\bTema\s*n?[ºo°.]*\s*\d[\d.]*"
    r"|\bS[úu]mula(?:\s*Vinculante)?\s*n?[ºo°.]*\s*\d+(?:\s*d[oa]\s*(?:STF|STJ|TSE))?",
    re.IGNORECASE,
)
# "NOME [LINK?]" — captura o nome do precedente que o redator marcou (tipo + número até o sentinela).
_RE_LINKQ = re.compile(
    r"(\b(?:ADI|ADC|ADPF|ADO|ARE|RE|REsp|AREsp|HC|MS|MI|PET|Rcl|MC|Tema|S[úu]mula(?:\s+Vinculante)?)"
    r"\s*n?[ºo°.]*\s*\d[^\[\]\n]*?)\s*\[LINK\?\]",
    re.IGNORECASE,
)


def _norm_ref(s: str) -> str:
    """Chave canônica de uma referência: 'ADC nº 29' -> 'adc29'; 'Súmula 41 do TSE' -> 'sumula41tse'."""
    s = (s or "").lower()
    s = re.sub(r"n[ºo°.]+", " ", s)
    s = re.sub(r"\b(?:de|da|do)\b", " ", s)
    return re.sub(r"[^\w]", "", s)


def _walk_strings(obj, fn):
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_walk_strings(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk_strings(v, fn) for k, v in obj.items()}
    return obj


def _texto_plano(obj) -> str:
    buf: List[str] = []
    _walk_strings(obj, lambda s: (buf.append(s), s)[1])
    return "\n".join(buf)


def _sub_fora_de_links(s: str, repl) -> str:
    """Aplica _RE_REF.sub(repl) apenas nas partes de `s` que NÃO estão dentro de [..](..)."""
    out, i = [], 0
    for m in re.finditer(r"\[[^\]]+\]\([^)]+\)", s):
        out.append(_RE_REF.sub(repl, s[i:m.start()]))
        out.append(m.group(0))
        i = m.end()
    out.append(_RE_REF.sub(repl, s[i:]))
    return "".join(out)


def _refs_pendentes(texto: str) -> List[str]:
    """Referências citadas SEM link (cruas) ou marcadas [LINK?]."""
    pend, vistos = [], set()
    for m in _RE_LINKQ.finditer(texto):
        ids = _RE_REF.findall(m.group(1))
        alvo = (ids[0] if ids else m.group(1)).strip()
        ch = _norm_ref(alvo)
        if ch and ch not in vistos:
            vistos.add(ch); pend.append(alvo)
    cru = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", texto).replace("[LINK?]", " ")
    for m in _RE_REF.finditer(cru):
        alvo = re.sub(r"\s+", " ", m.group(0)).strip()
        ch = _norm_ref(alvo)
        if ch and ch not in vistos:
            vistos.add(ch); pend.append(alvo)
    return pend


# Cache em disco dos links que o Gemini já resolveu (precedentes recorrentes — Ficha Limpa etc.
# reaparecem em muitas ITs): evita pagar grounding pela mesma URL de novo.
_CACHE_LINKS = cfg.PACKAGE_DIR / ".cache_links.json"
_CACHE_LINKS_DIAS = 30


def _links_cache_ler() -> Dict[str, str]:
    try:
        if _CACHE_LINKS.exists():
            d = json.loads(_CACHE_LINKS.read_text(encoding="utf-8"))
            cutoff = time.time() - _CACHE_LINKS_DIAS * 86400
            return {k: v["url"] for k, v in d.items()
                    if isinstance(v, dict) and v.get("url") and v.get("ts", 0) > cutoff}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _links_cache_gravar(novos: Dict[str, str]) -> None:
    if not novos:
        return
    try:
        d = json.loads(_CACHE_LINKS.read_text(encoding="utf-8")) if _CACHE_LINKS.exists() else {}
        for k, url in novos.items():
            d[k] = {"url": url, "ts": time.time()}
        _CACHE_LINKS.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _buscar_links_oficiais(refs: List[str], tema: str, log: Callable) -> Dict[str, str]:
    """Resolve referências -> URL oficial, reusando o cache em disco e só consultando o Gemini
    (1 chamada grounded) para o que ainda falta. ECONOMIA: precedentes recorrentes não repagam."""
    if not refs:
        return {}
    cache = _links_cache_ler()
    mapa: Dict[str, str] = {}
    faltam: List[str] = []
    for r in refs:
        ch = _norm_ref(r)
        if ch in cache:
            mapa[ch] = cache[ch]
        else:
            faltam.append(r)
    if not faltam:
        log(f"Links: {len(mapa)} resolvido(s) pelo cache (0 chamadas Gemini).")
        return mapa
    lista = "; ".join(faltam[:24])
    q = (
        f"Tema: {tema}. Para CADA uma destas referências jurídicas, forneça a URL OFICIAL de inteiro "
        f"teor/ficha (Supremo Tribunal Federal em portal.stf.jus.br ou redir.stf.jus.br; Tribunal "
        f"Superior Eleitoral em tse.jus.br; Superior Tribunal de Justiça em scon.stj.jus.br ou "
        f"processo.stj.jus.br): {lista}.\nResponda APENAS uma por linha, no formato exato "
        f"\"Referência => URL\". Se não tiver certeza da URL oficial de alguma, OMITA a linha "
        f"(nunca invente)."
    )
    log(f"Links: {len(mapa)} do cache; buscando {len(faltam)} via Gemini dirigido (1 chamada)…")
    r = gemini_web.pesquisar(q, foco="links oficiais de precedentes")
    novos: Dict[str, str] = {}
    if r.get("ok") and r.get("texto"):
        for ln in r["texto"].splitlines():
            if "=>" in ln and "http" in ln:
                nome, url = ln.split("=>", 1)
                u = re.search(r"https?://\S+", url)
                if u:
                    novos[_norm_ref(nome)] = u.group(0).rstrip(".,;)]")
    _links_cache_gravar(novos)
    mapa.update(novos)
    return mapa


def _enriquecer_links(dicts: List[dict], ctx: dict, analise: dict,
                      *, usar_web: bool = True, log: Callable = print) -> None:
    """2ª passada (in-place): linka precedentes citados sem link/marcados [LINK?] usando URLs já no
    contexto (RAG) + uma consulta Gemini dirigida pela lista EXATA do que o redator efetivamente citou.
    Respeita a 1ª menção (links só na primeira); refs sem URL ficam por extenso (sentinela removido)."""
    dicts = [d for d in dicts if d]
    texto = "\n".join(_texto_plano(d) for d in dicts)
    cru = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", texto)
    if "[LINK?]" not in texto and not _RE_REF.search(cru):
        return

    # mapa base: URLs de precedentes vindas do RAG (referência verificada da casa)
    mapa: Dict[str, str] = {}
    for t in ctx.get("rag", []):
        u = getattr(t, "url", "") or ""
        if any(d in u for d in ("tse.jus.br", "stf.jus.br", "stj.jus.br")):
            blob = (t.titulo or "") + " " + (t.texto[:200] or "")
            for mm in _RE_REF.finditer(blob):
                mapa.setdefault(_norm_ref(mm.group(0)), u)

    pend = [r for r in _refs_pendentes(texto) if _norm_ref(r) not in mapa]
    if usar_web and pend:
        mapa.update(_buscar_links_oficiais(pend, analise.get("tema", ""), log))

    # refs JÁ linkadas pelo modelo não devem ser re-linkadas (preserva a 1ª menção real)
    linkados = {_norm_ref(mm.group(0))
                for m in re.finditer(r"\[([^\]]+)\]\([^)]+\)", texto)
                for mm in _RE_REF.finditer(m.group(1))}
    n_ok = [0]

    def fix(s: str) -> str:
        def repl_q(m):  # resolve "NOME [LINK?]"
            nome = m.group(1).strip()
            ids = _RE_REF.findall(nome)
            ch = _norm_ref(ids[0] if ids else nome)
            url = mapa.get(ch)
            if url and ch not in linkados:
                linkados.add(ch); n_ok[0] += 1
                return f"[{nome}]({url})"
            return nome  # sem URL (ou já linkado antes): remove o sentinela
        s = _RE_LINKQ.sub(repl_q, s)

        def repl_raw(m):  # linka referência crua na 1ª menção
            alvo = m.group(0)
            ch = _norm_ref(alvo)
            url = mapa.get(ch)
            if url and ch not in linkados:
                linkados.add(ch); n_ok[0] += 1
                return f"[{alvo}]({url})"
            return alvo
        return _sub_fora_de_links(s, repl_raw)

    for d in dicts:
        novo = _walk_strings(d, fix)
        d.clear(); d.update(novo)
    faltam = sum(1 for r in pend if _norm_ref(r) not in linkados)
    log(f"Links: {n_ok[0]} precedente(s) linkado(s) na 1ª menção; {faltam} sem URL oficial (por extenso).")


# ============================ montagem dos blocos ============================
def _paras(lst) -> List[dict]:
    return [nw.bloco_paragraph(p) for p in (lst or []) if str(p).strip()]


def _tit(t, default: str) -> str:
    """Título de seção (heading_2) padronizado em CAIXA ALTA, como nos modelos da casa."""
    return (t or default).strip().upper()


def montar_blocos(analise: dict, it: dict, sec6: dict, minuta: dict, tipo, ano: int) -> List[dict]:
    B: List[dict] = []
    area = cconf.CONSULTOR_AREA

    # ---- cabeçalho (callout com Objeto:) ----
    objeto = (analise.get("objeto") or analise.get("tema") or "").strip()
    objeto = re.sub(r"^\s*objeto\s*:\s*", "", objeto, flags=re.IGNORECASE)  # evita "Objeto: Objeto:"
    cab = (f"**CÂMARA DOS DEPUTADOS — CONSULTORIA LEGISLATIVA**\n"
           f"**Área de {area}**\n**ESTUDO**\n**Objeto:** {objeto}")
    B.append(nw.bloco_callout(cab, emoji="📋", cor="green_background"))
    B.append(nw.bloco_divider())

    # ---- 1. Introdução ----
    B.append(nw.bloco_heading(2, "1. INTRODUÇÃO"))
    B += _paras(it.get("introducao"))

    # ---- 2. Marco constitucional ----
    mc = it.get("marco_constitucional") or {}
    B.append(nw.bloco_heading(2, _tit(mc.get("titulo"), "2. MARCO CONSTITUCIONAL")))
    B += _paras(mc.get("paragrafos"))

    # ---- 3. Quadro normativo ----
    qn = it.get("quadro_normativo") or {}
    B.append(nw.bloco_heading(2, _tit(qn.get("titulo"), "3. QUADRO NORMATIVO ATUAL")))
    B += _paras(qn.get("intro"))
    for sub in qn.get("subsecoes") or []:
        B.append(nw.bloco_heading(3, sub.get("titulo") or "3.x"))
        B += _paras(sub.get("paragrafos"))

    # ---- 4. Mapeamento + tabela ----
    mp = it.get("mapeamento") or {}
    B.append(nw.bloco_heading(2, _tit(mp.get("titulo"), "4. MAPEAMENTO DOS DISPOSITIVOS AFETADOS")))
    B += _paras(mp.get("intro"))
    tab = mp.get("tabela") or {}
    colunas = tab.get("colunas") or ["Dispositivo", "O que diz hoje", "Dificuldade atual", "Solução na minuta"]
    linhas = tab.get("linhas") or []
    if linhas:
        B.append(nw.bloco_heading(3, "4.1 — Tabela consolidada"))
        header = [f"**{c}**" for c in colunas]
        B.append(nw.bloco_tabela([header] + [[str(c) for c in ln] for ln in linhas], header=True))

    # ---- 5. Análise da solicitação ----
    an = it.get("analise_solicitacao") or {}
    B.append(nw.bloco_heading(2, _tit(an.get("titulo"), "5. ANÁLISE ESPECÍFICA DA SOLICITAÇÃO")))
    B += _paras(an.get("intro"))
    for sub in an.get("subsecoes") or []:
        B.append(nw.bloco_heading(3, sub.get("titulo") or "5.x"))
        B += _paras(sub.get("paragrafos"))

    # ---- 6. Proposições correlatas ----
    if sec6:
        B.append(nw.bloco_heading(2, "6. PROPOSIÇÕES LEGISLATIVAS SEMELHANTES NA CÂMARA DOS DEPUTADOS"))
        B += _paras(sec6.get("abertura"))
        for item in sec6.get("itens") or []:
            if str(item).strip():
                B.append(nw.bloco_bullet(str(item)))
        B += _paras(sec6.get("fecho_risco"))

    # ---- 7. Conclusão + fecho da IT ----
    B.append(nw.bloco_heading(2, "7. CONCLUSÃO"))
    B += _paras(it.get("conclusao"))
    B.append(nw.bloco_paragraph(cconf.FECHO_IT_LINHA_1))
    B.append(nw.bloco_paragraph(cconf.FECHO_IT_LINHA_2))

    # ---- 8. Minuta ----
    B.append(nw.bloco_heading(2, f"8. MINUTA DE {tipo.nome_extenso}"))
    ementa = (minuta.get("ementa") or "").strip()
    epigrafe = f"**{tipo.nome_extenso} Nº ___, DE {ano}**" + (f"\n{ementa}" if ementa else "")
    B.append(nw.bloco_callout(epigrafe, emoji="📜", cor="gray_background"))
    B.append(nw.bloco_paragraph(tipo.preambulo))
    for item in minuta.get("articulado") or []:
        texto = str(item.get("texto") or "").strip()
        if not texto:
            continue
        if item.get("tipo") == "quote":
            B.append(nw.bloco_quote(texto))
        else:
            B.append(nw.bloco_paragraph(texto))

    # ---- 9. Justificativa ----
    B.append(nw.bloco_heading(2, "9. JUSTIFICATIVA"))
    B += _paras(minuta.get("justificativa"))
    return B


# ============================ título da página ============================
def _set_titulo(page_id: str, titulo: str) -> None:
    body = {"properties": {"title": {"title": [{"type": "text", "text": {"content": titulo[:190]}}]}}}
    try:
        requests.patch(
            f"https://api.notion.com/v1/pages/{page_id}",
            headers={"Authorization": f"Bearer {cfg.load_notion_token()}",
                     "Notion-Version": cfg.NOTION_VERSION, "Content-Type": "application/json"},
            data=json.dumps(body), timeout=40,
        )
    except Exception:  # noqa: BLE001
        pass


# ============================ persistência do resultado ============================
ULTIMO_RESULTADO = cfg.PACKAGE_DIR / ".ultimo_resultado.json"


def _salvar_resultado(page_id: str, titulo: str, blocos: List[dict]) -> None:
    try:
        ULTIMO_RESULTADO.write_text(
            json.dumps({"page_id": page_id, "titulo": titulo, "blocos": blocos}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def regravar_ultimo(page_url: Optional[str] = None, *, progress=None) -> dict:
    """Regrava o ÚLTIMO resultado gerado (salvo em disco) numa página — sem
    reprocessar nada. Útil quando a gravação falhou (ex.: página não compartilhada)."""
    log = progress or (lambda _m: None)
    if not ULTIMO_RESULTADO.exists():
        raise RuntimeError("Não há resultado salvo para regravar.")
    data = json.loads(ULTIMO_RESULTADO.read_text(encoding="utf-8"))
    page_id = nw.page_id_from_url(page_url) if page_url else data["page_id"]
    nw.verificar_acesso_pagina(page_id)
    _set_titulo(page_id, data.get("titulo", ""))
    n = nw.escrever_pagina(page_id, data["blocos"], progress=log)
    return {"page_id": page_id, "titulo": data.get("titulo"), "n_blocos": n}


# ============================ orquestração ============================
def gerar(
    demanda: str,
    page_url: str,
    *,
    usar_rag: bool = True,
    usar_camara: bool = True,
    usar_web: bool = True,
    bases_rag: Optional[List[str]] = None,
    somente_vigente: bool = True,
    ano: Optional[int] = None,
    model: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Resultado:
    log = progress or (lambda _m: None)
    ano = ano or datetime.date.today().year
    avisos: List[str] = []

    # valida o acesso à página ANTES de gastar qualquer chamada de IA
    page_id = nw.page_id_from_url(page_url)
    log("Verificando acesso à página do Notion…")
    nw.verificar_acesso_pagina(page_id)

    analise = analisar_demanda(demanda, model=model, log=log)
    tipo = classifier.detectar_tipo(analise.get("tipo_sigla", ""), demanda)
    log(f"Espécie identificada: {tipo.sigla} — {tipo.nome_extenso}.")

    ctx = coletar_contexto(analise, usar_rag=usar_rag, usar_camara=usar_camara,
                           usar_web=usar_web, bases_rag=bases_rag, tipo_sigla=tipo.sigla,
                           somente_vigente=somente_vigente, log=log)
    if usar_rag and not ctx["rag"]:
        avisos.append("Nenhum trecho recuperado do RAG — as bases foram indexadas? (py -m conle_gerador.notion_rag --indexar)")

    # corpo da IT e Seção 6 são independentes — redige em paralelo (ambos I/O em chamadas de API);
    # a minuta vem depois, pois depende do corpo da IT (coerência de dispositivos).
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_it = ex.submit(redigir_it, demanda, analise, ctx, model=model, log=log)
        fut_s6 = (ex.submit(redigir_secao6, analise, ctx.get("camara", []), model=model, log=log)
                  if usar_camara else None)
        it = fut_it.result()
        sec6 = fut_s6.result() if fut_s6 else {}
    minuta = redigir_minuta(demanda, analise, ctx, tipo, it=it, model=model, log=log)

    # 2ª passada de links: linka precedentes citados sem link/marcados [LINK?] no corpo e na Seção 6
    # (a minuta/justificativa NÃO leva links, por regra). Dirigida pelo que o redator efetivamente citou.
    try:
        _enriquecer_links([it, sec6], ctx, analise, usar_web=usar_web, log=log)
    except Exception as e:  # noqa: BLE001
        log(f"   enriquecimento de links falhou (seguindo sem): {e}")

    blocos = montar_blocos(analise, it, sec6, minuta, tipo, ano)
    log(f"Montados {len(blocos)} blocos na anatomia da IT.")

    titulo = f"{analise.get('tema', 'Estudo')} — minuta de {tipo.sigla}"
    # salva o resultado em disco ANTES de gravar — se a gravação falhar, nada se perde
    _salvar_resultado(page_id, titulo, blocos)
    _set_titulo(page_id, titulo)
    log("Gravando na página do Notion…")
    n = nw.escrever_pagina(page_id, blocos, progress=log)

    fontes_web = [u for w in ctx.get("web", []) for u in w.get("fontes", [])]
    avisos.append("Confira a numeração de dispositivos da minuta (alíneas/parágrafos) antes de protocolar.")
    return Resultado(page_id=page_id, titulo=titulo, tipo_sigla=tipo.sigla,
                     n_blocos=n, fontes_web=fontes_web[:12], avisos=avisos)
