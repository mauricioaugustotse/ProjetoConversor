# -*- coding: utf-8 -*-
"""Pipeline do gerador: demanda -> (RAG + Câmara + Gemini) -> IT + minuta -> Notion.

A página é montada na MESMA anatomia que conle_conversor/splitter.py reconhece, de
modo que o conversor existente gera os .docx a partir dela.
"""
from __future__ import annotations

import datetime
import json
import re
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
        link = f"  [LINK OFICIAL: {t.url}]" if getattr(t, "url", "") else ""
        linhas.append(f"- ({t.fonte} · {t.titulo}){link}\n  {t.texto[:800]}")
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


def coletar_contexto(
    analise: dict,
    *,
    usar_rag: bool = True,
    usar_camara: bool = True,
    usar_web: bool = True,
    bases_rag: Optional[List[str]] = None,
    tipo_sigla: Optional[str] = None,
    log: Callable = print,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"rag": [], "camara": [], "web": [], "links": None}

    if usar_rag:
        consultas = analise.get("consultas_rag") or [analise.get("tema", "")]
        vistos = set()
        trechos = []
        for q in consultas[:5]:
            log(f"RAG: consultando bases internas — “{q[:60]}”")
            for t in notion_rag.buscar(q, k=6, bases=bases_rag):
                chave = (t.titulo, t.texto[:40])
                if chave not in vistos:
                    vistos.add(chave)
                    trechos.append(t)
        ctx["rag"] = trechos[:14]
        log(f"RAG: {len(ctx['rag'])} trechos recuperados.")

    if usar_camara:
        kws = analise.get("palavras_chave_camara") or []
        rel = kws + [analise.get("tema", "")]
        log(f"Câmara: buscando proposições correlatas ({', '.join(kws[:6])})…")
        ctx["camara"] = camara_api.proposicoes_correlatas(
            kws, sigla=tipo_sigla, termos_relevancia=rel, max_total=12)
        log(f"Câmara: {len(ctx['camara'])} proposições correlatas.")

    if usar_web:
        for q in (analise.get("consultas_web") or [])[:4]:
            log(f"Web (Gemini): {q[:70]}")
            r = gemini_web.pesquisar(q, foco=analise.get("tema"))
            if r.get("ok"):
                ctx["web"].append(r)
            elif r.get("erro"):
                log(f"   web indisponível: {r['erro'][:80]}")
        # +1 consulta dedicada: URLs oficiais de precedentes e normas (p/ linkar a 1ª menção)
        tema = analise.get("tema", "")
        disp = ", ".join(analise.get("dispositivos_alvo") or [])
        qlinks = (
            f"Para o tema \"{tema}\" (dispositivos: {disp}), liste os PRINCIPAIS precedentes do "
            "Supremo Tribunal Federal e do Tribunal Superior Eleitoral e as normas (leis, leis "
            "complementares, emendas constitucionais) pertinentes, cada um seguido de sua URL "
            "OFICIAL — inteiro teor em stf.jus.br/tse.jus.br para julgados; planalto.gov.br para "
            "normas. Uma por linha, no formato: Nome (ex.: ADI nº 5.617; LC nº 64/1990) — URL."
        )
        log("Web (Gemini): coletando links oficiais de precedentes e normas…")
        rl = gemini_web.pesquisar(qlinks, foco="links oficiais")
        if rl.get("ok"):
            ctx["links"] = rl
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
                           usar_web=usar_web, bases_rag=bases_rag, tipo_sigla=tipo.sigla, log=log)
    if usar_rag and not ctx["rag"]:
        avisos.append("Nenhum trecho recuperado do RAG — as bases foram indexadas? (py -m conle_gerador.notion_rag --indexar)")

    it = redigir_it(demanda, analise, ctx, model=model, log=log)
    sec6 = redigir_secao6(analise, ctx.get("camara", []), model=model, log=log) if usar_camara else {}
    minuta = redigir_minuta(demanda, analise, ctx, tipo, it=it, model=model, log=log)

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
