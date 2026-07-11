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

from . import anexos as anexos_mod
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
    gerou_it: bool = True
    gerou_minuta: bool = True


# ============================ etapas ============================
def analisar_demanda(demanda: str, *, docs: Optional[List[anexos_mod.Anexo]] = None,
                     model: Optional[str] = None, log: Callable = print) -> dict:
    log("Analisando a demanda (espécie, tema, dispositivos, palavras-chave)…")
    user = f"DEMANDA:\n{demanda}"
    docs_txt = anexos_mod.resumo_para_analise(docs or [])
    if docs_txt:
        user += f"\n\nDOCUMENTOS ENCAMINHADOS PELO PARLAMENTAR (parte integrante da demanda):\n{docs_txt}"
    data = llm.chat(prompts.SYS_ANALISE, user, json_mode=True, model=model)
    if not isinstance(data, dict) or not data.get("tema"):
        data = {"tema": demanda[:80], "tipo_sigla": "PL", "objeto": demanda,
                "palavras_chave_camara": [], "consultas_rag": [demanda[:120]], "consultas_web": []}
    return data


def _fmt_rag(trechos) -> str:
    linhas = []
    for t in trechos:
        meta = []
        if getattr(t, "ancora", ""):
            meta.append(t.ancora)  # jurisprudência: Rel. X | resultado Y | proc. Z
        if getattr(t, "data", ""):
            meta.append(f"data {t.data}")
        url = getattr(t, "url", "")
        # URL interna do Notion não é fonte pública — não oferecer como LINK OFICIAL
        if url and not re.search(r"(?:^notion://|//(?:www\.)?notion\.so\b|//app\.notion\.com\b)", url):
            meta.append(f"LINK OFICIAL: {url}")
        tag = ("  [" + " | ".join(meta) + "]") if meta else ""
        # ficha sem título vira rótulo derivado do conteúdo — "Untitled"/"(sem
        # título)" no cabeçalho induzia o redator a citá-lo literalmente
        titulo = (getattr(t, "titulo", "") or "").strip()
        if not titulo or titulo.lower() in ("untitled", "(sem título)", "sem título"):
            titulo = (getattr(t, "texto", "") or "").strip().splitlines()[0][:60] or "trecho"
        linhas.append(f"- ({t.fonte} · {titulo}){tag}\n  {t.texto[:800]}")
    return "\n".join(linhas) if linhas else "(sem trechos relevantes nas bases internas)"


def _fmt_camara(props: List[dict]) -> str:
    linhas = []
    for p in props:
        aut = ""
        if p.get("autor"):
            extra = f" +{p['n_autores'] - 1}" if (p.get("n_autores") or 0) > 1 else ""
            aut = f" (autor: {p['autor']}{extra})"
        # situação com a tramitação real: órgão, regime e data (quando detalhados)
        sit = ""
        if p.get("situacao"):
            det = [d for d in (p.get("orgao"), p.get("regime"),
                               f"em {p['data_situacao']}" if p.get("data_situacao") else "") if d]
            sit = f" — situação: {p['situacao']}" + (f" ({', '.join(det)})" if det else "")
        apensada = f"; apensada ao {p['apensada_a']}" if p.get("apensada_a") else ""
        rel = (f"; {p['n_relacionadas']} relacionada(s)/apensada(s) na árvore"
               if p.get("n_relacionadas") else "")
        linhas.append(f"- {p['sigla']} {p['numero']}/{p['ano']}{aut}: "
                      f"{p['ementa'][:240]}{sit}{apensada}{rel} | {p['url']}")
    return "\n".join(linhas) if linhas else "(nenhuma proposição correlata localizada)"


def _fmt_camara_resumo(resumo: dict) -> str:
    """Bloco quantitativo agregado da busca na Câmara (dados reais p/ a abertura da Seção 6)."""
    if not resumo:
        return ""
    partes = [f"RESUMO QUANTITATIVO (dados reais da API da Câmara; varredura da espécie no período "
              f"{resumo.get('periodo', '?')} — a busca complementar por palavras-chave pode "
              f"acrescentar proposições de anos anteriores):"]
    if resumo.get("total_norma_alvo") is not None:
        partes.append(f"- {resumo['total_norma_alvo']} proposição(ões) da espécie alteram a "
                      f"norma/artigos-alvo (de {resumo.get('total_candidatas', '?')} candidatas examinadas).")
    por_ano = resumo.get("por_ano_norma_alvo") or {}
    if por_ano:
        serie = "; ".join(f"{a}: {n}" for a, n in sorted(por_ano.items(), reverse=True))
        partes.append(f"- Distribuição por ano (das que alteram a norma/artigos-alvo): {serie}.")
    por_sit = resumo.get("por_situacao_listadas") or {}
    if por_sit:
        dist = "; ".join(f"{s}: {n}" for s, n in sorted(por_sit.items(), key=lambda x: -x[1]))
        partes.append(f"- Das {resumo.get('listadas', '?')} LISTADAS abaixo (as mais correlatas), "
                      f"situação atual: {dist}. Esta distribuição cobre SOMENTE as listadas.")
    if resumo.get("apensadas_listadas"):
        partes.append(f"- {resumo['apensadas_listadas']} da(s) listada(s) já tramita(m) apensada(s) a outra proposição.")
    return "\n".join(partes)


def _fmt_web(web: List[dict]) -> str:
    blocos = []
    for w in web:
        if w.get("texto"):
            fontes = "; ".join(w.get("fontes", [])[:5])
            # teto por resposta: era o único bloco do contexto sem limite de tamanho
            blocos.append(f"{w['texto'][:6000]}\nFontes: {fontes}")
    return "\n\n".join(blocos) if blocos else "(sem resultados de pesquisa web)"


_FONTE_CAT = None


def _cat_fonte(label: str) -> str:
    global _FONTE_CAT
    if _FONTE_CAT is None:
        _FONTE_CAT = {b.get("label"): b.get("categoria", "normativo")
                     for b in cfg.bases_registry().values()}
    return _FONTE_CAT.get(label, "normativo")


def _cota_por_categoria(trechos, *, n_total: int = 14, max_juris: int = 6):
    """Seleção do top-k por RELEVÂNCIA (rerank global por score), com a cota como TETO: a
    jurisprudência (temas/sessões/DJe) entra no máximo `max_juris` vezes para não engolir o
    normativo. Antes a seleção seguia a ORDEM das consultas (um trecho fraco da 1ª consulta
    excluía um forte da última); agora ordena pelo score final do RAG."""
    ordenados = sorted(trechos, key=lambda t: getattr(t, "score", 0.0), reverse=True)
    out, n_juris = [], 0
    for t in ordenados:
        if len(out) >= n_total:
            break
        if _cat_fonte(t.fonte) == "jurisprudencia":
            if n_juris >= max_juris:
                continue
            n_juris += 1
        out.append(t)
    if len(out) < n_total:  # faltou normativo — completa com a jurisprudência além da cota
        vistos = {id(t) for t in out}
        for t in ordenados:
            if len(out) >= n_total:
                break
            if id(t) not in vistos:
                out.append(t)
                vistos.add(id(t))
    return out


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
    ctx: Dict[str, Any] = {"rag": [], "camara": [], "web": []}

    if usar_rag:
        # consultas: as gerais + dirigidas aos dispositivos (legislação) e julgados (jurisprudência),
        # para que a fundamentação cubra os DOIS eixos com profundidade ("valorizar as tintas").
        # reserva slots para as consultas DIRIGIDAS (dispositivos/julgados): elas vêm depois das
        # gerais e, sem reserva, o corte da lista as eliminava — justo os dois eixos a cobrir.
        gerais = [q for q in (analise.get("consultas_rag") or []) if q and str(q).strip()][:4]
        dirigidas = ([f"dispositivo legal: {d}" for d in (analise.get("dispositivos_chave") or [])[:2]]
                     + [f"jurisprudência/tese: {j}" for j in (analise.get("julgados_chave") or [])[:2]])
        consultas = gerais + dirigidas
        if not consultas:
            consultas = [analise.get("tema", "")]
        vistos = set()
        trechos = []
        for q in consultas:
            log(f"RAG: consultando bases internas — “{q[:60]}”")
            for t in notion_rag.buscar(q, k=6, bases=bases_rag, somente_vigente=somente_vigente,
                                       progress=log):
                chave = t.page_id or (t.titulo, t.texto[:40])  # dedup por identidade real (page_id)
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
        r = camara_api.proposicoes_correlatas(
            kws, sigla=tipo_sigla, norma_alvo=analise.get("norma_alvo", ""),
            tema=analise.get("tema", ""), dispositivos_alvo=analise.get("dispositivos_alvo"),
            termos_relevancia=rel, max_total=20, log=log)
        ctx["camara"], ctx["camara_resumo"] = r["itens"], r["resumo"]
        log(f"Câmara: {len(ctx['camara'])} proposições correlatas "
            f"({ctx['camara_resumo'].get('total_norma_alvo', 0)} alteram a norma/artigos-alvo).")

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
    # IT/minuta recebem o resumo + as 12 mais correlatas (controle de tokens);
    # a Seção 6 (redigir_secao6) recebe a lista completa.
    camara_txt = _fmt_camara(ctx.get("camara", [])[:12])
    resumo_txt = _fmt_camara_resumo(ctx.get("camara_resumo", {}))
    if resumo_txt:
        camara_txt = resumo_txt + "\n\n" + camara_txt
    # documentos do parlamentar vêm ANTES das fontes de pesquisa: são parte da demanda
    docs_txt = anexos_mod.formatar_contexto(ctx.get("anexos") or [])
    bloco_docs = ("=== DOCUMENTOS ENCAMINHADOS PELO PARLAMENTAR (parte integrante da demanda) ===\n"
                  + prompts.NOTA_ANEXOS + "\n\n" + docs_txt + "\n\n") if docs_txt else ""
    return (
        bloco_docs +
        "=== TRECHOS DAS BASES INTERNAS (RAG) ===\n" + _fmt_rag(ctx.get("rag", [])) +
        "\n\n=== PROPOSIÇÕES NA CÂMARA (API oficial) ===\n" + camara_txt +
        "\n\n=== PESQUISA WEB (Gemini grounded) ===\n" + _fmt_web(ctx.get("web", [])) +
        "\n\n=== LINKS OFICIAIS — para linkar a 1ª menção de cada precedente/norma (nunca invente "
        "URL) ===\nUse os \"[LINK OFICIAL: ...]\" dos trechos do RAG, a TABELA DE LINKS do Planalto e "
        "as URLs da pesquisa web acima; quando faltar a URL de um precedente, marque-o com [LINK?] — "
        "é resolvido automaticamente na etapa de enriquecimento."
    )


def redigir_it(demanda: str, analise: dict, ctx: dict, *, com_minuta: bool = True,
               model=None, log: Callable = print) -> dict:
    log("Redigindo o corpo da Informação Técnica (seções 1–5 e 7)…")
    user = (
        f"DEMANDA:\n{demanda}\n\nANÁLISE:\n{json.dumps(analise, ensure_ascii=False)}\n\n"
        f"CONTEXTO:\n{_contexto_txt(ctx)}"
    )
    it = llm.chat(prompts.sys_it(com_minuta), user, json_mode=True, model=model)
    return it if isinstance(it, dict) else {}


def redigir_secao6(analise: dict, camara: List[dict], resumo: Optional[dict] = None,
                   *, model=None, log: Callable = print) -> dict:
    # gera a Seção 6 mesmo sem correlatas (dirá que não foram localizadas proposições semelhantes)
    log("Redigindo a Seção 6 (proposições correlatas / risco de apensação)…")
    resumo_txt = _fmt_camara_resumo(resumo or {})
    user = (
        f"TEMA: {analise.get('tema')}\nESPÉCIE: {analise.get('tipo_sigla')}\n\n"
        + (resumo_txt + "\n\n" if resumo_txt else "")
        + f"PROPOSIÇÕES (dados reais da Câmara):\n{_fmt_camara(camara)}"
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
    m = llm.chat(prompts.sys_minuta(com_it=bool(it)), user, json_mode=True, model=model)
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
    """Chave canônica de uma referência: 'ADC nº 29' -> 'adc29'; 'Súmula nº 41 do TSE' -> 'sumula41tse'.
    MANTÉM o sufixo do tribunal: 'Súmula nº 41 do TSE' e 'Súmula nº 41 do STF' são precedentes DISTINTOS
    (com URLs distintas) e não podem colapsar — o modelo cita súmula sempre com o tribunal (regra de
    citação), então o ganho de remover o sufixo não compensa o risco de link cruzado entre tribunais."""
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
                ids = _RE_REF.findall(nome)  # extrai a referência canônica (descarta numeração/prefixo do modelo)
                ch = _norm_ref(ids[0]) if ids else ""
                u = re.search(r"https?://\S+", url)
                if ch and u:  # só grava chaves de referência REAL — não envenena o cache
                    novos[ch] = u.group(0).rstrip(".,;)]")
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
            alvo = ids[0] if ids else nome  # a referência reconhecível (a 1ª, p/ refs compostas)
            ch = _norm_ref(alvo)
            url = mapa.get(ch)
            if url and ch not in linkados:
                linkados.add(ch); n_ok[0] += 1
                # linka SÓ a referência reconhecida; em "ADC nº 29 e nº 30", o "e nº 30" (sem tipo,
                # não casável) fica fora da âncora — evita o link único errado englobando ambas
                return nome.replace(alvo, f"[{alvo}]({url})", 1) if alvo != nome else f"[{nome}]({url})"
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


def montar_blocos(analise: dict, it: dict, sec6: dict, minuta: dict, tipo, ano: int, *,
                  gerar_it: bool = True, gerar_minuta: bool = True) -> List[dict]:
    B: List[dict] = []
    area = cconf.CONSULTOR_AREA

    # ---- cabeçalho (callout com Objeto:) ----
    # SEMPRE presente, em qualquer escopo: é a âncora que o splitter do conversor
    # usa para reconhecer a página (OBJETO/CONSULTORIA LEGISLATIVA/ESTUDO).
    objeto = (analise.get("objeto") or analise.get("tema") or "").strip()
    objeto = re.sub(r"^\s*objeto\s*:\s*", "", objeto, flags=re.IGNORECASE)  # evita "Objeto: Objeto:"
    cab = (f"**CÂMARA DOS DEPUTADOS — CONSULTORIA LEGISLATIVA**\n"
           f"**Área de {area}**\n**ESTUDO**\n**Objeto:** {objeto}")
    B.append(nw.bloco_callout(cab, emoji="📋", cor="green_background"))
    B.append(nw.bloco_divider())

    if gerar_it:
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
        colunas = tab.get("colunas") or ["Dispositivo", "O que diz hoje", "Dificuldade atual",
                                         "Solução na minuta" if gerar_minuta else "Solução proposta"]
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

    if gerar_minuta:
        # ---- 8. Minuta (sem o "8."/"9." quando a página não traz a IT) ----
        # o splitter exige apenas "MINUTA" no heading e o callout de epígrafe intocado
        B.append(nw.bloco_heading(2, (f"8. MINUTA DE {tipo.nome_extenso}" if gerar_it
                                      else f"MINUTA DE {tipo.nome_extenso}")))
        ementa = (minuta.get("ementa") or "").strip()
        epigrafe = f"**{tipo.nome_extenso} Nº ___, DE {ano}**" + (f"\n{ementa}" if ementa else "")
        B.append(nw.bloco_callout(epigrafe, emoji="📜", cor="gray_background"))
        B.append(nw.bloco_paragraph(tipo.preambulo))
        for item in minuta.get("articulado") or []:
            # tolera o LLM desviando do formato {tipo, texto} e emitindo strings cruas no articulado
            # (senão item.get quebraria com AttributeError e perderia toda a geração já paga)
            if isinstance(item, str):
                texto, tipo_item = item.strip(), "paragraph"
            elif isinstance(item, dict):
                texto, tipo_item = str(item.get("texto") or "").strip(), item.get("tipo")
            else:
                continue
            if not texto:
                continue
            B.append(nw.bloco_quote(texto) if tipo_item == "quote" else nw.bloco_paragraph(texto))

        # ---- 9. Justificativa ----
        B.append(nw.bloco_heading(2, "9. JUSTIFICATIVA" if gerar_it else "JUSTIFICATIVA"))
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
    anexos: Optional[List[str]] = None,
    usar_rag: bool = True,
    usar_camara: bool = True,
    usar_web: bool = True,
    bases_rag: Optional[List[str]] = None,
    somente_vigente: bool = True,
    gerar_it: bool = True,
    gerar_minuta: bool = True,
    ano: Optional[int] = None,
    model: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Resultado:
    log = progress or (lambda _m: None)
    ano = ano or datetime.date.today().year
    avisos: List[str] = []
    if not gerar_it and not gerar_minuta:
        raise RuntimeError("Selecione ao menos um documento: Informação Técnica ou minuta de proposição.")

    # valida o acesso à página ANTES de gastar qualquer chamada de IA
    page_id = nw.page_id_from_url(page_url)
    log("Verificando acesso à página do Notion…")
    nw.verificar_acesso_pagina(page_id)
    if gerar_it and gerar_minuta:
        log("Escopo: Informação Técnica + minuta de proposição.")
    else:
        log("Escopo: apenas " + ("a Informação Técnica." if gerar_it else "a minuta de proposição."))

    # documentos encaminhados pelo parlamentar: extraídos ANTES da análise (a alimentam)
    docs: List[anexos_mod.Anexo] = []
    if anexos:
        log(f"Anexos: extraindo o texto de {len(anexos)} documento(s) encaminhado(s)…")
        docs = anexos_mod.processar(list(anexos), log=log)
        avisos += [f"Anexo “{a.nome}”: {a.aviso}" for a in docs if a.aviso]
        if not any(a.texto for a in docs):
            avisos.append("Nenhum anexo rendeu texto aproveitável — a geração seguiu só com a demanda digitada.")

    analise = analisar_demanda(demanda, docs=docs, model=model, log=log)
    tipo = classifier.detectar_tipo(analise.get("tipo_sigla", ""), demanda)
    log(f"Espécie identificada: {tipo.sigla} — {tipo.nome_extenso}.")

    # a Câmara alimenta a Seção 6 (parte da IT): no modo só-minuta é a etapa mais lenta sem valor
    if usar_camara and not gerar_it:
        log("Câmara: pulada (a Seção 6 pertence à IT, fora do escopo).")
    ctx = coletar_contexto(analise, usar_rag=usar_rag, usar_camara=usar_camara and gerar_it,
                           usar_web=usar_web, bases_rag=bases_rag, tipo_sigla=tipo.sigla,
                           somente_vigente=somente_vigente, log=log)
    ctx["anexos"] = docs  # entram no _contexto_txt da redação (IT e minuta)
    if usar_rag and not ctx["rag"]:
        avisos.append("Nenhum trecho recuperado do RAG — as bases foram indexadas? (py -m conle_gerador.notion_rag --indexar)")

    # corpo da IT e Seção 6 são independentes — redige em paralelo (ambos I/O em chamadas de API);
    # a minuta vem depois, pois depende do corpo da IT (coerência de dispositivos).
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_it = (ex.submit(redigir_it, demanda, analise, ctx, com_minuta=gerar_minuta,
                            model=model, log=log)
                  if gerar_it else None)
        fut_s6 = (ex.submit(redigir_secao6, analise, ctx.get("camara", []),
                            ctx.get("camara_resumo"), model=model, log=log)
                  if (usar_camara and gerar_it) else None)
        # não deixar a falha de uma redação abortar tudo e perder as chamadas já pagas:
        # captura, segue com o que houver e sinaliza no resultado
        try:
            it = fut_it.result() if fut_it else {}
        except Exception as e:  # noqa: BLE001
            log(f"   redação do corpo da IT falhou: {e}")
            it, avisos = {}, avisos + ["A redação do corpo da IT falhou — a página pode estar incompleta."]
        try:
            sec6 = fut_s6.result() if fut_s6 else {}
        except Exception as e:  # noqa: BLE001
            log(f"   redação da Seção 6 falhou: {e}")
            sec6 = {}
    minuta = (redigir_minuta(demanda, analise, ctx, tipo, it=it, model=model, log=log)
              if gerar_minuta else {})
    # alerta de seção vazia (parse/truncamento do modelo): evita gravar em silêncio
    if gerar_it and not it:
        avisos.append("Corpo da IT vazio (possível truncamento/JSON inválido do modelo) — revise a página.")
    if gerar_minuta and not minuta:
        avisos.append("Minuta vazia (possível truncamento/JSON inválido do modelo) — revise a página.")

    # 2ª passada de links: linka precedentes citados sem link/marcados [LINK?] no corpo e na Seção 6
    # (a minuta/justificativa NÃO leva links, por regra). Dirigida pelo que o redator efetivamente citou.
    if gerar_it:
        try:
            _enriquecer_links([it, sec6], ctx, analise, usar_web=usar_web, log=log)
        except Exception as e:  # noqa: BLE001
            log(f"   enriquecimento de links falhou (seguindo sem): {e}")

    blocos = montar_blocos(analise, it, sec6, minuta, tipo, ano,
                           gerar_it=gerar_it, gerar_minuta=gerar_minuta)
    log(f"Montados {len(blocos)} blocos na anatomia da IT.")

    tema = analise.get("tema", "Estudo")
    if gerar_it and not gerar_minuta:
        titulo = f"{tema} — Informação Técnica"
    else:  # ambas ou só-minuta: o sufixo "— minuta" orienta o conversor
        titulo = f"{tema} — minuta de {tipo.sigla}"
    # salva o resultado em disco ANTES de gravar — se a gravação falhar, nada se perde
    _salvar_resultado(page_id, titulo, blocos)
    _set_titulo(page_id, titulo)
    log("Gravando na página do Notion…")
    n = nw.escrever_pagina(page_id, blocos, progress=log)

    fontes_web = [u for w in ctx.get("web", []) for u in w.get("fontes", [])]
    if gerar_minuta:
        avisos.append("Confira a numeração de dispositivos da minuta (alíneas/parágrafos) antes de protocolar.")
    if gerar_it and not gerar_minuta:
        avisos.append("No conversor, marque apenas 'Gerar Informação Técnica' (não há minuta na página).")
    elif gerar_minuta and not gerar_it:
        avisos.append("No conversor, marque apenas 'Gerar minuta de proposição' (não há IT na página).")
    return Resultado(page_id=page_id, titulo=titulo, tipo_sigla=tipo.sigla,
                     n_blocos=n, fontes_web=fontes_web[:12], avisos=avisos,
                     gerou_it=gerar_it, gerou_minuta=gerar_minuta)
