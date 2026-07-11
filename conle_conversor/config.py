# -*- coding: utf-8 -*-
"""Configurações, caminhos e blocos de texto fixos do conversor."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, NamedTuple, Optional

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"

TEMPLATE_IT = TEMPLATES_DIR / "template_IT.docx"
TEMPLATE_PROPOSICAO = TEMPLATES_DIR / "template_proposicao.docx"
TEMPLATE_PARECER = TEMPLATES_DIR / "template_parecer.docx"

# Base das pastas-padrão da casa (Consultoria Legislativa)
STS_DIR = Path(r"C:\Users\mauri\OneDrive\Documentos\12 - Consultoria Legislativa\STs")

# Saída dos documentos gerados (.docx)
OUTPUT_IT_DIR = STS_DIR / "Info Técnica"
OUTPUT_PROPOSICAO_DIR = STS_DIR / "Proposições"
OUTPUT_PARECER_DIR = STS_DIR / "Pareceres"

# Pastas-modelo usadas para (re)gerar os templates (build_templates).
# Documentos reais equivalentes aos modelos originais (renomeados na
# reorganização das pastas); usados apenas ao regenerar templates/.
MODELO_IT = OUTPUT_IT_DIR / (
    "Informação Técnica - Dep. Julio Lopes - "
    "condicionando Fundo Partidário e Fundo Eleitoral a KPIs_1.docx"
)
MODELO_PROPOSICAO = OUTPUT_PROPOSICAO_DIR / (
    "Minuta de PLP 2026 - inelegibilidade por vínculo com "
    "organizações criminosas (Dep. Coronel Assis).docx"
)
MODELO_PARECER = OUTPUT_PARECER_DIR / (
    "Parecer CCJC - PL nº 7.684-2017 - loterias e prevenção "
    "à lavagem de dinheiro_2.docx"
)

# ---------------------------------------------------------------------------
# Credenciais
# ---------------------------------------------------------------------------
NOTION_KEY_FILE = PROJECT_ROOT / "Chave_Notion.txt"
ENV_FILE = PROJECT_ROOT / ".env"


def load_notion_token() -> str:
    token = os.environ.get("NOTION_TOKEN", "").strip()
    if token:
        return token
    if NOTION_KEY_FILE.exists():
        return NOTION_KEY_FILE.read_text(encoding="utf-8").strip()
    raise RuntimeError("Token do Notion não encontrado (Chave_Notion.txt ou env NOTION_TOKEN).")


def load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"')
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            m = re.match(r'\s*OPENAI_API_KEY\s*=\s*"?([^"\n]+)"?', line)
            if m:
                return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Defaults de identificação
# ---------------------------------------------------------------------------
CONSULTOR_NOME = "Maurício Augusto Chiaramonte Vieira"
CONSULTOR_CARGO = "Consultor Legislativo"
# Linhas do campo AUTOR na capa (cargo detalhado + área de atuação)
CONSULTOR_CARGO_CAPA = "Consultor Legislativo da Área I"
CONSULTOR_AREA = (
    "Direito Constitucional, Eleitoral, Municipal, Regimento Interno, "
    "Processo Legislativo e Poder Judiciário"
)
LOCAL_FECHO_IT = "Consultoria Legislativa"
LOCAL_FECHO_PROPOSICAO = "Sala das Sessões"
LOCAL_FECHO_PARECER = "Sala da Comissão"

OPENAI_MODEL = "gpt-5.6-luna"

# Estilos (nomes exatos definidos nos templates Word) -----------------------
class S:
    NORMAL = "Normal"
    CORPO = "CORPO PADRÃO"
    TITULO_ITEM = "TÍTULO ITEM (1.)"
    TITULO_SUB = "TÍTULO SUBITEM (1.1)"
    TITULO_SUB2 = "TÍTULO SUBITEM-NÍVEL2 (1.1.1)"
    TRANSCRICAO = "TRANSCRIÇÃO LEI"
    FECHO = "FECHO (local e data)"
    ASSINATURA = "ASSINATURA"
    SISCONLE = "Nº DO TRABALHO SISCONLE"
    CAPA_TITULO = "CAPA ESTUDO - TÍTULO"
    # Proposição
    EPIGRAFE = "EPÍGRAFE (nome da proposição)"
    AUTORIA = "AUTORIA DE PROPOSIÇÃO"
    EMENTA = "EMENTA"
    PREAMBULO = "PREÂMBULO"
    JUSTIFICACAO = "JUSTIFICAÇÃO"
    # Parecer de comissão
    COMISSAO = "COMISSÃO"
    APENSO = "APENSO"
    AUTOR_RELATOR = "AUTOR/RELATOR EMENTA"
    RELATORIO_VOTO = "RELATÓRIO/VOTO"


# ---------------------------------------------------------------------------
# Fontes públicas oficiais: normas reconhecidas por texto (tabela única)
# ---------------------------------------------------------------------------
# A tabela NORMAS_OFICIAIS alimenta DOIS mecanismos do richtext:
#   1. mentions internas do Notion (@página das bases Vademecum, ex.
#      "Lei das Eleições - Art. 36-A") -> hyperlink para a fonte pública
#      oficial (o link interno do Notion nunca é preservado no .docx);
#   2. citações TEXTUAIS digitadas no corpo ("art. 22, XVI, da LC nº 64/1990")
#      -> mesmo hyperlink, montado pelos padrões de richtext.linkificar_citacoes.
# Cada norma declara: o prefixo da mention (mention_re), as designações usadas
# no corpo do texto após "da/do" (citacao_re), a designação NUMERADA que pode
# ser linkada sozinha, sem artigo (citacao_isolada_re; None = não linkar solta,
# p/ evitar over-linking de "Constituição Federal"/"RICD" soltos), o callable
# que monta a URL a partir do Match (grupos nomeados num/letra/res) e os
# aliases do nome — usados também para podar o "eco" redundante que o texto ao
# redor às vezes repete, ex. "(RICD - Art. 151, RICD)" -> "(RICD - Art. 151)".

# Constituição Federal no Planalto (âncora nativa #artN por artigo).
CF_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm"
# LC 64/1990 (Lei das Inelegibilidades) e Lei 9.504/1997 (Lei das Eleições) no
# Planalto — mesmas URLs que os autores das páginas já usam nos links manuais.
LC64_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp64.htm"
L9504_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/leis/l9504.htm"
# Código Eleitoral (Lei 4.737/1965): a versão ORIGINAL tem âncoras name="artN"
# (aferido: art242/art299 existem); a "compilado" NÃO tem âncora nenhuma.
L4737_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/leis/l4737.htm"
# Lei 5.889/1973 (trabalho rural) — âncoras nativas #artN aferidas (inclusive
# letra minúscula: name="art14a"), padrão ccivil_03.
L5889_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/leis/l5889.htm"
# LC 95/1998 (elaboração/redação das leis) — citada em todo exame de técnica
# legislativa; mesma URL que os autores já usam nos links manuais.
LCP95_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp95.htm"
# NR-31 (segurança e saúde no trabalho rural) — página oficial vigente no
# portal do MTE (aferida 05/07/2026; NR não tem artigos, link sem âncora).
NR31_MTE_URL = (
    "https://www.gov.br/trabalho-e-emprego/pt-br/acesso-a-informacao/"
    "participacao-social/conselhos-e-orgaos-colegiados/"
    "comissao-tripartite-partitaria-permanente/normas-regulamentadora/"
    "normas-regulamentadoras-vigentes/norma-regulamentadora-no-31-nr-31"
)
# Lei 1.079/1950 (crimes de responsabilidade) — âncoras nativas #artN aferidas
# (06/07/2026; inclusive por inciso: art4i), padrão ccivil_03.
L1079_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/leis/l1079.htm"
# Lei 14.192/2021 (violência política contra a mulher) e Lei 14.211/2021 —
# aferidas por GET 06/07/2026 (páginas do Estudo Resoluções TSE 2026).
L14192_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14192.htm"
L14211_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14211.htm"
# Lei 9.096/1995 (Lei dos Partidos Políticos) — âncora nativa #art15 aferida.
L9096_PLANALTO_URL = "https://www.planalto.gov.br/ccivil_03/leis/l9096.htm"


def ancora_artigo_cf(num: int) -> str:
    return f"#art{num}"


def ancora_planalto(num: int, letra: str = "") -> str:
    """Âncora nativa do Planalto: #artN, letra de artigo acrescido em minúscula
    (art. 36-A -> #art36a) — convenção já usada em richtext._ancorar_dispositivo."""
    return f"#art{num}{(letra or '').lower()}"


# Resolução da Câmara nº 17/1989 (RICD), texto consolidado no LEGIN — NÃO a
# página institucional genérica (troca pedida pelo usuário).
RICD_LEGIN_URL = (
    "https://www2.camara.leg.br/legin/fed/rescad/1989/"
    "resolucaodacamaradosdeputados-17-21-setembro-1989-320110-normaatualizada-pl.html"
)


def ancora_artigo_ricd(num: int, letra: str = "") -> str:
    """Fragmento de destaque de texto (#:~:text=) que abre a página do LEGIN
    já no caput do artigo. A página consolidada não tem âncoras HTML próprias
    (só bookmarks residuais de Word), então usa-se o text fragment do browser —
    mesmo recurso já presente nos links dos documentos-modelo da casa. O alvo
    reproduz a grafia do caput no LEGIN: "Art. 5º" (1º-9º) / "Art. 54." (10+) /
    "Art. 13-A." (artigo acrescido — aferido 06/07/2026: a 1ª ocorrência de
    "Art. 13-A"/"Art. 20-A" na página é o próprio caput); aferido contra o
    HTML real: em 264 dos 278 artigos sem letra a 1ª ocorrência é o caput;
    nos demais o link apenas destaca uma remissão anterior."""
    if letra:
        base = f"Art.%20{num}%C2%BA" if num < 10 else f"Art.%20{num}"
        return f"#:~:text={base}-{letra.upper()}."
    alvo = f"Art.%20{num}%C2%BA" if num < 10 else f"Art.%20{num}."
    return f"#:~:text={alvo}"


# Resoluções do TSE com página compilada conhecida (a URL traz a data por
# extenso e não é derivável do número — mapa extensível, chave normalizada
# "NN.NNN"). Fora do mapa -> sem link (mention vira texto puro; citação
# textual não é linkada).
_TSE_COMPILADA = "https://www.tse.jus.br/legislacao/compilada/res/"
RESOLUCOES_TSE = {
    "23.444": _TSE_COMPILADA + "2015/resolucao-no-23-444-de-30-de-abril-de-2015",
    "23.600": _TSE_COMPILADA + "2019/resolucao-no-23-600-de-12-de-dezembro-de-2019",
    "23.605": _TSE_COMPILADA + "2019/resolucao-no-23-605-de-17-de-dezembro-de-2019",
    "23.607": _TSE_COMPILADA + "2019/resolucao-no-23-607-de-17-de-dezembro-de-2019",
    "23.608": _TSE_COMPILADA + "2019/resolucao-no-23-608-de-18-de-dezembro-de-2019",
    "23.609": _TSE_COMPILADA + "2019/resolucao-no-23-609-de-18-de-dezembro-de-2019",
    "23.610": _TSE_COMPILADA + "2019/resolucao-no-23-610-de-18-de-dezembro-de-2019",
    "23.659": _TSE_COMPILADA + "2021/resolucao-no-23-659-de-26-de-outubro-de-2021",
    "23.664": _TSE_COMPILADA + "2021/resolucao-no-23-664-de-9-de-dezembro-de-2021",
    "23.665": _TSE_COMPILADA + "2021/resolucao-no-23-665-de-9-de-dezembro-de-2021",
    "23.670": _TSE_COMPILADA + "2021/resolucao-no-23-670-de-14-de-dezembro-de-2021",
    "23.671": _TSE_COMPILADA + "2021/resolucao-no-23-671-de-14-de-dezembro-de-2021",
    # CUIDADO: o slug da 23.673 NÃO tem o "de" antes do dia (irregularidade do site)
    "23.673": _TSE_COMPILADA + "2021/resolucao-no-23-673-14-de-dezembro-de-2021",
    "23.675": _TSE_COMPILADA + "2021/resolucao-no-23-675-de-16-de-dezembro-de-2021",
    "23.677": _TSE_COMPILADA + "2021/resolucao-no-23-677-de-16-de-dezembro-de-2021",
    "23.679": _TSE_COMPILADA + "2022/resolucao-no-23-679-de-8-de-fevereiro-de-2022",
    "23.729": _TSE_COMPILADA + "2024/resolucao-no-23-729-de-27-de-fevereiro-de-2024",
    "23.731": _TSE_COMPILADA + "2024/resolucao-no-23-731-de-27-de-fevereiro-de-2024",
    "23.732": _TSE_COMPILADA + "2024/resolucao-no-23-732-de-27-de-fevereiro-de-2024",
    "23.734": _TSE_COMPILADA + "2024/resolucao-no-23-734-de-27-de-fevereiro-de-2024",
    "23.735": _TSE_COMPILADA + "2024/resolucao-no-23-735-de-27-de-fevereiro-de-2024",
    "23.746": _TSE_COMPILADA + "2025/resolucao-no-23-746-de-11-de-marco-de-2025",
    "23.749": _TSE_COMPILADA + "2026/resolucao-no-23-749-de-26-de-fevereiro-de-2026",
    "23.750": _TSE_COMPILADA + "2026/resolucao-no-23-750-de-26-de-fevereiro-de-2026",
    "23.751": _TSE_COMPILADA + "2026/resolucao-no-23-751-de-26-de-fevereiro-de-2026",
    "23.752": _TSE_COMPILADA + "2026/resolucao-no-23-752-de-26-de-fevereiro-de-2026",
    "23.754": _TSE_COMPILADA + "2026/resolucao-no-23-754-de-2-de-marco-de-2026",
    "23.755": _TSE_COMPILADA + "2026/resolucao-no-23-755-de-2-de-marco-de-2026",
    "23.756": _TSE_COMPILADA + "2026/resolucao-no-23-756-de-2-de-marco-de-2026",
    "23.757": _TSE_COMPILADA + "2026/resolucao-no-23-757-de-2-de-marco-de-2026",
    "23.758": _TSE_COMPILADA + "2026/resolucao-no-23-758-de-2-de-marco-de-2026",
    "23.759": _TSE_COMPILADA + "2026/resolucao-no-23-759-de-26-de-fevereiro-de-2026",
    "23.760": _TSE_COMPILADA + "2026/resolucao-no-23-760-de-2-de-marco-de-2026",
}

# Emendas Constitucionais no Planalto (mapa fechado, verificado por GET
# 06/07/2026; a convenção emc/emcNNN.htm não é chutada para números novos).
EMENDAS_CONSTITUCIONAIS = {
    "117": "https://www.planalto.gov.br/ccivil_03/constituicao/emendas/emc/emc117.htm",
    "133": "https://www.planalto.gov.br/ccivil_03/constituicao/emendas/emc/emc133.htm",
}


def _fonte_emenda(m: Optional["re.Match"]) -> Optional[str]:
    d = m.groupdict() if m else {}
    base = EMENDAS_CONSTITUCIONAIS.get(re.sub(r"\D", "", d.get("emc") or ""))
    if not base:
        return None
    if not d.get("num"):
        return base
    return base + ancora_planalto(int(d["num"]), d.get("letra") or "")


# ---------------------------------------------------------------------------
# Normas descobertas/aprovadas pelo usuário (conle_conversor/normas_extras.json,
# gravado pela ferramenta `py -m conle_conversor.fontes` ou pelo botão
# "Verificar fontes" da GUI). O merge acontece aqui, no import: leis e LCs
# aprovadas linkam pela NormaOficial GENÉRICA no fim da tabela; resoluções do
# TSE e ECs entram nos mapas respectivos.
# ---------------------------------------------------------------------------
def _carregar_normas_extras() -> dict:
    try:
        import json as _json
        from pathlib import Path as _Path
        p = _Path(__file__).resolve().parent / "normas_extras.json"
        if p.exists():
            return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_NORMAS_EXTRAS = _carregar_normas_extras()
RESOLUCOES_TSE.update(_NORMAS_EXTRAS.get("resolucoes_tse", {}))
EMENDAS_CONSTITUCIONAIS.update(_NORMAS_EXTRAS.get("emendas", {}))
LEIS_EXTRAS = dict(_NORMAS_EXTRAS.get("leis", {}))    # {"12034": url}
LCPS_EXTRAS = dict(_NORMAS_EXTRAS.get("lcps", {}))    # {"101": url}
# Resoluções da CÂMARA (chave "N/AAAA" — número pequeno exige o ano): o slug
# do LEGIN tem id interno não-derivável, então as entradas nascem no
# descobridor — aprendidas dos links do AUTOR na própria página ou informadas
# manualmente com --url. Fora do mapa -> sem link, como sempre.
RESOLUCOES_CAMARA = dict(_NORMAS_EXTRAS.get("resolucoes_camara", {}))


def _fonte_res_camara(m: Optional["re.Match"]) -> Optional[str]:
    d = m.groupdict() if m else {}
    if not (d.get("rcd") and d.get("rcd_ano")):
        return None
    base = RESOLUCOES_CAMARA.get(f"{int(d['rcd'])}/{d['rcd_ano']}")
    if not base:
        return None
    if not d.get("num"):
        return base
    # páginas do LEGIN: mesmo text fragment do RICD
    return base + ancora_artigo_ricd(int(d["num"]), d.get("letra") or "")


def _fonte_lei_extra(m: Optional["re.Match"]) -> Optional[str]:
    d = m.groupdict() if m else {}
    base = LEIS_EXTRAS.get(re.sub(r"\D", "", d.get("leinum") or ""))
    if not base:
        return None
    if not d.get("num"):
        return base
    return base + ancora_planalto(int(d["num"]), d.get("letra") or "")


def _fonte_lcp_extra(m: Optional["re.Match"]) -> Optional[str]:
    d = m.groupdict() if m else {}
    base = LCPS_EXTRAS.get(re.sub(r"\D", "", d.get("lcpnum") or ""))
    if not base:
        return None
    if not d.get("num"):
        return base
    return base + ancora_planalto(int(d["num"]), d.get("letra") or "")


def url_resolucao_tse(numero: str) -> Optional[str]:
    """URL da página compilada da resolução ("23735"/"23.735" -> mapa), ou None."""
    n = re.sub(r"[^\d]", "", numero or "")
    if len(n) == 5:
        n = f"{n[:2]}.{n[2:]}"
    return RESOLUCOES_TSE.get(n)


def ancora_artigo_tse(num: int, letra: str = "") -> str:
    """Âncora de artigo nas páginas compiladas do TSE: id="artN" nativo existe
    só em PARTE dos artigos (aferido: 23.735 tem art1/2/6/11/15; 23.610 tem 51
    ids) — por isso combina-se o fragmento nativo com um text fragment de
    fallback na grafia do caput ("Art. 6º" / "Art. 10."), no espírito de
    ancora_artigo_ricd. Nenhum dos dois existindo, degrada para o topo."""
    letra = (letra or "").lower()
    if letra:
        alvo = f"Art.%20{num}-{letra.upper()}."
    else:
        alvo = f"Art.%20{num}%C2%BA" if num < 10 else f"Art.%20{num}."
    return f"#art{num}{letra}:~:text={alvo}"


# Sufixo comum dos prefixos de mention: "… - Art. 36-A" (nº + letra opcional).
_ART_MENTION = r"Art\.?\s*(?P<num>\d+)(?:[ºo°])?(?:\s*[-–‑]\s*(?P<letra>[A-Za-z])\b)?"


def _fonte_planalto(base_url: str) -> Callable[[Optional["re.Match"]], Optional[str]]:
    """Callable de URL p/ normas no Planalto: com grupo num -> âncora #artN
    (letra entra minúscula); sem artigo -> topo da página."""
    def montar(m: Optional["re.Match"]) -> Optional[str]:
        d = m.groupdict() if m else {}
        if not d.get("num"):
            return base_url
        return base_url + ancora_planalto(int(d["num"]), d.get("letra") or "")
    return montar


def _fonte_ricd(m: Optional["re.Match"]) -> Optional[str]:
    d = m.groupdict() if m else {}
    if not d.get("num"):
        return RICD_LEGIN_URL
    return RICD_LEGIN_URL + ancora_artigo_ricd(int(d["num"]), d.get("letra") or "")


def _fonte_res_tse(m: Optional["re.Match"]) -> Optional[str]:
    """Resolução do TSE: URL do mapa (None se não mapeada) + âncora de artigo."""
    d = m.groupdict() if m else {}
    base = url_resolucao_tse(d.get("res") or "")
    if not base:
        return None
    if not d.get("num"):
        return base
    return base + ancora_artigo_tse(int(d["num"]), d.get("letra") or "")


class NormaOficial(NamedTuple):
    mention_re: str                      # prefixo do texto da mention interna
    citacao_re: str                      # designações no corpo, após "da/do"
    citacao_isolada_re: Optional[str]    # designação numerada linkável sozinha
    montar_url: Callable[[Optional["re.Match"]], Optional[str]]
    aliases: tuple                       # p/ poda de eco redundante


NORMAS_OFICIAIS = [
    NormaOficial(
        mention_re=rf"^Constitui[çc][ãa]o Federal\s*[-–—]\s*{_ART_MENTION}",
        citacao_re=(r"CF(?:/88)?\b|Constitui[çc][ãa]o(?:\s+Federal|\s+da\s+"
                    r"Rep[úu]blica(?:\s+Federativa\s+do\s+Brasil)?)?(?:\s+de\s+1988)?"),
        citacao_isolada_re=None,
        montar_url=_fonte_planalto(CF_PLANALTO_URL),
        aliases=("Constituição Federal", "CF"),
    ),
    NormaOficial(
        mention_re=rf"^RICD\s*[-–—]\s*{_ART_MENTION}",
        # "Resolução nº 17, de 1989" é a designação FORMAL do RICD — específica
        # aqui para não cair na genérica de resolução da Câmara (mapa fechado)
        citacao_re=(r"RICD\b|Regimento\s+Interno(?:\s+da\s+C[âa]mara(?:\s+dos\s+"
                    r"Deputados)?)?"
                    r"|Resolu[çc][ãa]o(?:\s+da\s+C[âa]mara(?:\s+dos\s+Deputados)?)?"
                    r"\s*n[ºo°.]*\s*17,?\s+de\s+(?:21\s+de\s+setembro\s+de\s+)?1989"),
        citacao_isolada_re=(r"Resolu[çc][ãa]o(?:\s+da\s+C[âa]mara(?:\s+dos\s+Deputados)?)?"
                            r"\s*n[ºo°.]*\s*17,?\s+de\s+(?:21\s+de\s+setembro\s+de\s+)?1989"),
        montar_url=_fonte_ricd,
        aliases=("RICD",),
    ),
    NormaOficial(
        # Base Vademecum "Lei de Inelegibilidade - Art. 22"
        mention_re=rf"^Lei d[ae]s? Inelegibilidades?\s*[-–—]\s*{_ART_MENTION}",
        citacao_re=(r"(?:LC|Lei\s+Complementar)\s*n?[ºo°.]*\s*64"
                    r"(?:\s*/\s*(?:19)?90|,?\s+de(?:\s+18\s+de\s+maio\s+de)?\s+1990)?"
                    r"|Lei\s+d[ae]s?\s+Inelegibilidades?"),
        citacao_isolada_re=(r"(?:LC|Lei\s+Complementar)\s*n?[ºo°.]*\s*64"
                            r"(?:\s*/\s*(?:19)?90|,?\s+de(?:\s+18\s+de\s+maio\s+de)?\s+1990)"),
        montar_url=_fonte_planalto(LC64_PLANALTO_URL),
        aliases=("Lei de Inelegibilidade", "Lei de Inelegibilidades",
                 "Lei das Inelegibilidades", "LC nº 64/1990", "LC 64/1990"),
    ),
    NormaOficial(
        mention_re=rf"^Lei das Elei[çc][õo]es\s*[-–—]\s*{_ART_MENTION}",
        citacao_re=(r"Lei\s*n?[ºo°.]*\s*9\.?504"
                    r"(?:\s*/\s*(?:19)?97|,?\s+de(?:\s+30\s+de\s+setembro\s+de)?\s+1997)?"
                    r"|Lei\s+das\s+Elei[çc][õo]es|Lei\s+Eleitoral"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*9\.?504"
                            r"(?:\s*/\s*(?:19)?97|,?\s+de(?:\s+30\s+de\s+setembro\s+de)?\s+1997)"),
        montar_url=_fonte_planalto(L9504_PLANALTO_URL),
        aliases=("Lei das Eleições", "Lei nº 9.504/1997", "Lei 9.504/1997"),
    ),
    NormaOficial(
        mention_re=rf"^C[óo]digo Eleitoral\s*[-–—]\s*{_ART_MENTION}",
        # "CE" só casa como designação após "art(s). N … do" (contexto do padrão
        # de citação) — não há risco com a sigla do estado do Ceará.
        citacao_re=(r"C[óo]digo\s+Eleitoral|\bCE\b"
                    r"|Lei\s*n?[ºo°.]*\s*4\.?737"
                    r"(?:\s*/\s*(?:19)?65|,?\s+de(?:\s+15\s+de\s+julho\s+de)?\s+1965)?"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*4\.?737"
                            r"(?:\s*/\s*(?:19)?65|,?\s+de(?:\s+15\s+de\s+julho\s+de)?\s+1965)"),
        montar_url=_fonte_planalto(L4737_PLANALTO_URL),
        aliases=("Código Eleitoral",),
    ),
    NormaOficial(
        mention_re=rf"^Lei\s*n?[ºo°.]*\s*5\.?889(?:\s*/\s*(?:19)?73)?\s*[-–—]\s*{_ART_MENTION}",
        citacao_re=(r"Lei\s*n?[ºo°.]*\s*5\.?889"
                    r"(?:\s*/\s*(?:19)?73|,?\s+de(?:\s+8\s+de\s+junho\s+de)?\s+1973)?"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*5\.?889"
                            r"(?:\s*/\s*(?:19)?73|,?\s+de(?:\s+8\s+de\s+junho\s+de)?\s+1973)"),
        montar_url=_fonte_planalto(L5889_PLANALTO_URL),
        aliases=("Lei nº 5.889/1973", "Lei 5.889/1973"),
    ),
    NormaOficial(
        mention_re=rf"^(?:LC|Lei Complementar)\s*n?[ºo°.]*\s*95(?:\s*/\s*(?:19)?98)?\s*[-–—]\s*{_ART_MENTION}",
        citacao_re=(r"(?:LC|Lei\s+Complementar)\s*n?[ºo°.]*\s*95"
                    r"(?:\s*/\s*(?:19)?98|,?\s+de(?:\s+26\s+de\s+fevereiro\s+de)?\s+1998)?"),
        citacao_isolada_re=(r"(?:LC|Lei\s+Complementar)\s*n?[ºo°.]*\s*95"
                            r"(?:\s*/\s*(?:19)?98|,?\s+de(?:\s+26\s+de\s+fevereiro\s+de)?\s+1998)"),
        montar_url=_fonte_planalto(LCP95_PLANALTO_URL),
        aliases=("Lei Complementar nº 95/1998", "LC nº 95/1998", "LC 95/1998"),
    ),
    NormaOficial(
        # NR não tem artigos (itens 31.x.x) — link sempre para o topo da página.
        mention_re=r"^(?:NR[-\s]?31|Norma\s+Regulamentadora\s*n?[ºo°.]*\s*31)\b",
        citacao_re=r"NR[-\s]?31\b|Norma\s+Regulamentadora\s*n?[ºo°.]*\s*31\b",
        citacao_isolada_re=(r"Norma\s+Regulamentadora\s*n?[ºo°.]*\s*31"
                            r"(?:\s*\(NR[-\s]?31\))?|NR[-\s]?31\b"),
        montar_url=lambda m: NR31_MTE_URL,
        aliases=("NR-31", "Norma Regulamentadora nº 31"),
    ),
    NormaOficial(
        mention_re=rf"^Lei\s*n?[ºo°.]*\s*1\.?079(?:\s*/\s*(?:19)?50)?\s*[-–—]\s*{_ART_MENTION}",
        citacao_re=(r"Lei\s*n?[ºo°.]*\s*1\.?079"
                    r"(?:\s*/\s*(?:19)?50|,?\s+de(?:\s+10\s+de\s+abril\s+de)?\s+1950)?"
                    r"|Lei\s+dos\s+Crimes\s+de\s+Responsabilidade"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*1\.?079"
                            r"(?:\s*/\s*(?:19)?50|,?\s+de(?:\s+10\s+de\s+abril\s+de)?\s+1950)"),
        montar_url=_fonte_planalto(L1079_PLANALTO_URL),
        aliases=("Lei nº 1.079/1950", "Lei 1.079/1950"),
    ),
    NormaOficial(
        mention_re=(r"^(?:EC|Emenda Constitucional)\s*n?[ºo°.]*\s*(?P<emc>\d{1,3})"
                    rf"(?:\s*/\s*\d{{4}})?\s*[-–—]\s*{_ART_MENTION}"),
        citacao_re=(r"(?:EC|Emenda\s+Constitucional)\s*n?[ºo°.]*\s*(?P<emc>\d{1,3})"
                    r"(?:\s*/\s*\d{4}|,?\s+de\s+\d{4})?"),
        citacao_isolada_re=(r"(?:EC|Emenda\s+Constitucional)\s*n?[ºo°.]*\s*(?P<emc>\d{1,3})"
                            r"(?:\s*/\s*\d{4}|,?\s+de\s+\d{4})"),
        montar_url=_fonte_emenda,
        aliases=(),
    ),
    NormaOficial(
        # Base Vademecum "Lei dos Partidos Políticos - Art. 15"
        mention_re=(r"^(?:Lei dos Partidos Pol[ií]ticos|Lei\s*n?[ºo°.]*\s*9\.?096"
                    rf"(?:\s*/\s*(?:19)?95)?)\s*[-–—,]\s*{_ART_MENTION}"),
        citacao_re=(r"Lei\s*n?[ºo°.]*\s*9\.?096"
                    r"(?:\s*/\s*(?:19)?95|,?\s+de(?:\s+19\s+de\s+setembro\s+de)?\s+1995)?"
                    r"|Lei\s+dos\s+Partidos\s+Pol[ií]ticos"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*9\.?096"
                            r"(?:\s*/\s*(?:19)?95|,?\s+de(?:\s+19\s+de\s+setembro\s+de)?\s+1995)"),
        montar_url=_fonte_planalto(L9096_PLANALTO_URL),
        aliases=("Lei dos Partidos Políticos", "Lei nº 9.096/1995", "Lei 9.096/1995"),
    ),
    NormaOficial(
        mention_re=rf"^Lei\s*n?[ºo°.]*\s*14\.?192(?:\s*/\s*(?:20)?21)?\s*[-–—]\s*{_ART_MENTION}",
        citacao_re=(r"Lei\s*n?[ºo°.]*\s*14\.?192"
                    r"(?:\s*/\s*(?:20)?21|,?\s+de(?:\s+4\s+de\s+agosto\s+de)?\s+2021)?"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*14\.?192"
                            r"(?:\s*/\s*(?:20)?21|,?\s+de(?:\s+4\s+de\s+agosto\s+de)?\s+2021)"),
        montar_url=_fonte_planalto(L14192_PLANALTO_URL),
        aliases=("Lei nº 14.192/2021", "Lei 14.192/2021"),
    ),
    NormaOficial(
        mention_re=rf"^Lei\s*n?[ºo°.]*\s*14\.?211(?:\s*/\s*(?:20)?21)?\s*[-–—]\s*{_ART_MENTION}",
        citacao_re=(r"Lei\s*n?[ºo°.]*\s*14\.?211"
                    r"(?:\s*/\s*(?:20)?21|,?\s+de(?:\s+27\s+de\s+setembro\s+de)?\s+2021)?"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*14\.?211"
                            r"(?:\s*/\s*(?:20)?21|,?\s+de(?:\s+27\s+de\s+setembro\s+de)?\s+2021)"),
        montar_url=_fonte_planalto(L14211_PLANALTO_URL),
        aliases=("Lei nº 14.211/2021", "Lei 14.211/2021"),
    ),
    NormaOficial(
        # Base Vademecum "Res.-TSE n. 23.735/2024 - Ilicitos eleitorais".
        # Nas citações textuais o "TSE" e o "nº" são opcionais ("Resolução nº
        # 23.751/2026", "Res. 23.673/2021") — número 2X.XXX de 5 dígitos é
        # assinatura das resoluções do TSE; fora do mapa continua sem link.
        # Mentions também no formato "Res. 23.677/2021, Art. 7º" (TSE e nº
        # opcionais; dispositivo após vírgula/traço vira âncora) — base de
        # resoluções por dispositivo vista na Consulta distrital (06/07/2026).
        mention_re=(r"^Res(?:\.|olu[çc][ãa]o)?\s*[-–.]?\s*(?:TSE\s*)?n?[ºo°.]*\s*"
                    rf"(?P<res>\d{{2}}\.?\d{{3}})(?:\s*/\s*\d{{4}})?(?:\s*[,;–—-]\s*{_ART_MENTION})?"),
        # O ano também é opcional ("art. 187 da Res. 23.751" — páginas de
        # 06/07/2026): o número de 5 dígitos é chave unívoca do mapa fechado.
        citacao_re=(r"Res(?:\.|olu[çc][ãa]o)\s*[-–.]?\s*(?:do\s+)?(?:TSE\s*)?(?:n[ºo°.]*\s*)?"
                    r"(?P<res>\d{2}\.?\d{3})(?:\s*(?:/\s*\d{4}|,?\s+de\s+[^,;()]{4,40}?\d{4}))?"),
        citacao_isolada_re=(r"Res(?:\.|olu[çc][ãa]o)\s*[-–.]?\s*(?:do\s+)?(?:TSE\s*)?(?:n[ºo°.]*\s*)?"
                            r"(?P<res>\d{2}\.?\d{3})(?:\s*(?:/\s*\d{4}|,?\s+de\s+[^,;()]{4,40}?\d{4}))?"),
        montar_url=_fonte_res_tse,
        aliases=(),
    ),
    # --- genéricas das normas APROVADAS em normas_extras.json (sempre por
    # ÚLTIMO: as específicas acima vencem no linkificar por ordem da tabela;
    # número fora do mapa -> montar_url None -> sem link, como sempre) ------
    NormaOficial(
        mention_re=(r"^Lei\s*n?[ºo°.]*\s*(?P<leinum>\d{1,2}\.?\d{3})"
                    rf"(?:\s*/\s*\d{{2,4}})?\s*[-–—,]\s*{_ART_MENTION}"),
        citacao_re=(r"Lei\s*n?[ºo°.]*\s*(?P<leinum>\d{1,2}\.?\d{3})"
                    r"(?:\s*/\s*\d{2,4}|,?\s+de\s+[^,;()]{0,30}?\d{4})?"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*(?P<leinum>\d{1,2}\.?\d{3})"
                            r"(?:\s*/\s*\d{2,4}|,?\s+de\s+[^,;()]{0,30}?\d{4})"),
        montar_url=_fonte_lei_extra,
        aliases=(),
    ),
    NormaOficial(
        mention_re=(r"^(?:LC|Lei Complementar)\s*n?[ºo°.]*\s*(?P<lcpnum>\d{1,3})"
                    rf"(?:\s*/\s*\d{{2,4}})?\s*[-–—,]\s*{_ART_MENTION}"),
        citacao_re=(r"(?:LC|Lei\s+Complementar)\s*n?[ºo°.]*\s*(?P<lcpnum>\d{1,3})"
                    r"(?:\s*/\s*\d{2,4}|,?\s+de\s+[^,;()]{0,30}?\d{4})?"),
        citacao_isolada_re=(r"(?:LC|Lei\s+Complementar)\s*n?[ºo°.]*\s*(?P<lcpnum>\d{1,3})"
                            r"(?:\s*/\s*\d{2,4}|,?\s+de\s+[^,;()]{0,30}?\d{4})"),
        montar_url=_fonte_lcp_extra,
        aliases=(),
    ),
    NormaOficial(
        # Resolução da Câmara aprovada no mapa (qualificador opcional: "a
        # Resolução nº 6, de 2023" no contexto CONLE é da Câmara; só linka o
        # que estiver em RESOLUCOES_CAMARA, então não há risco de acertar a
        # casa errada). O ano é OBRIGATÓRIO (número pequeno é ambíguo sem ele).
        mention_re=(r"^Resolu[çc][ãa]o(?:\s+da\s+C[âa]mara(?:\s+dos\s+Deputados)?)?"
                    r"\s*n[ºo°.]*\s*(?P<rcd>\d{1,3})\s*"
                    r"(?:/\s*|,?\s+de\s+(?:\d{1,2}[ºo°]?\s+de\s+[a-zA-Zç]+\s+de\s+)?)"
                    rf"(?P<rcd_ano>\d{{4}})\s*[-–—,]\s*{_ART_MENTION}"),
        citacao_re=(r"Resolu[çc][ãa]o(?:\s+da\s+C[âa]mara(?:\s+dos\s+Deputados)?)?"
                    r"\s*n[ºo°.]*\s*(?P<rcd>\d{1,3})\s*"
                    r"(?:/\s*|,?\s+de\s+(?:\d{1,2}[ºo°]?\s+de\s+[a-zA-Zç]+\s+de\s+)?)"
                    r"(?P<rcd_ano>\d{4})"),
        citacao_isolada_re=(r"Resolu[çc][ãa]o(?:\s+da\s+C[âa]mara(?:\s+dos\s+Deputados)?)?"
                            r"\s*n[ºo°.]*\s*(?P<rcd>\d{1,3})\s*"
                            r"(?:/\s*|,?\s+de\s+(?:\d{1,2}[ºo°]?\s+de\s+[a-zA-Zç]+\s+de\s+)?)"
                            r"(?P<rcd_ano>\d{4})"),
        montar_url=_fonte_res_camara,
        aliases=(),
    ),
]

# Formato consumido por richtext.resolver_fonte_publica/_aliases_da_norma:
# (regex do texto da mention, callable(Match) -> URL ou None, aliases).
FONTES_OFICIAIS = [(n.mention_re, n.montar_url, n.aliases) for n in NORMAS_OFICIAIS]


# ---------------------------------------------------------------------------
# URLs "equivalentes" que os autores costumam linkar no Notion, canonizadas
# para a fonte oficial do padrão da casa (mapa FECHADO, cada par verificado —
# nunca chutar URL). Ex.: a página institucional do RICD vira o texto
# consolidado no LEGIN; os "compilado" do Planalto viram a versão com âncoras
# #artN (o compilado não tem âncora nenhuma). Comparação por URL normalizada
# (sem esquema/www, path em minúsculas, sem "/" final).
# ---------------------------------------------------------------------------
def _norm_url(u: str) -> str:
    u = re.sub(r"^https?://(?:www\d?\.)?", "", (u or "").strip(), flags=re.I)
    return u.split("#")[0].rstrip("/").lower()


_URLS_CANONICAS = {
    _norm_url("https://www2.camara.leg.br/atividade-legislativa/legislacao/"
              "regimento-interno-da-camara-dos-deputados"): RICD_LEGIN_URL,
    # download .docx do RICD atualizado (visto na página da PRC Bancada Negra)
    _norm_url("https://www2.camara.leg.br/atividade-legislativa/legislacao/"
              "regimento-interno-da-camara-dos-deputados/arquivos-1/"
              "copy_of_RICDatualizadoatRCD342026.docx"): RICD_LEGIN_URL,
    # publicação ORIGINAL da Resolução 17/1989 -> texto consolidado
    _norm_url("https://www2.camara.leg.br/legin/fed/rescad/1989/"
              "resolucaodacamaradosdeputados-17-21-setembro-1989-320110-"
              "publicacaooriginal-1-pl.html"): RICD_LEGIN_URL,
    _norm_url("https://www.planalto.gov.br/ccivil_03/constituicao/"
              "ConstituicaoCompilado.htm"): CF_PLANALTO_URL,
    _norm_url("https://www.planalto.gov.br/ccivil_03/leis/l4737compilado.htm"): L4737_PLANALTO_URL,
    _norm_url("https://www.planalto.gov.br/ccivil_03/leis/L9504compilado.htm"): L9504_PLANALTO_URL,
    # compilado da Lei 9.096 NÃO tem âncora nenhuma (aferido 06/07/2026: o
    # "art. 44" abria no topo) — o original tem #artN nativo
    _norm_url("https://www.planalto.gov.br/ccivil_03/leis/L9096compilado.htm"): L9096_PLANALTO_URL,
}


def url_canonica(href: str) -> Optional[str]:
    """URL oficial canônica para um href conhecido como equivalente (None se o
    href não está no mapa). Fragment existente é preservado pelo chamador."""
    return _URLS_CANONICAS.get(_norm_url(href))


# ---------------------------------------------------------------------------
# Blocos de texto fixos (idênticos em todas as ITs da casa)
# ---------------------------------------------------------------------------
PONTILHADO = "." * 102

# Transcrição da Resolução nº 48/1993 (competência dos consultores)
BLOCO_RESOLUCAO_INTRO = (
    "O inciso IV do art. 6º e o § 4º do art. 12 da Resolução determinam:"
)
_RES_ART6 = [
    "Art. 6º Aos Consultores e Assessores Legislativos compete realizar os fins "
    "institucionais da Assessoria, previstos no art. 2º, incisos I a VII, e, destacadamente:",
    PONTILHADO,
    "IV – informar, preliminarmente, o solicitante, quando for o caso, da inviabilidade "
    "constitucional, jurídica, legal ou regimental, técnica, financeira ou orçamentária de "
    "proposição que lhes tenha sido distribuída para relatar ou elaborar.",
]
_RES_ART12_P4 = [
    PONTILHADO,
    "[Art. 12] § 4º Recebido pedido de elaboração de minuta de projeto coincidente com o "
    "conteúdo de outro em tramitação, deve o Consultor ou Assessor, preliminarmente, informar "
    "a existência deste ao interessado e encaminhar-lhe exemplar ou cópia do respectivo avulso.",
]
# Transcrição completa (com o § 4º do art. 12) — usada quando HÁ risco de apensação.
BLOCO_RESOLUCAO_TRANSCRICAO = _RES_ART6 + _RES_ART12_P4
# Sem o § 4º do art. 12 — usada quando NÃO há proposições correlatas (sem risco de apensação).
BLOCO_RESOLUCAO_TRANSCRICAO_SEM_APENSACAO = list(_RES_ART6)
# Intro do bloco, nas duas variantes (concorda em número com o que é transcrito).
BLOCO_RESOLUCAO_INTRO_SEM_APENSACAO = "O inciso IV do art. 6º da Resolução determina:"

# Fecho padrão da IT (também costuma constar na própria página do Notion)
FECHO_IT_LINHA_1 = "Era o que tínhamos a informar."
FECHO_IT_LINHA_2 = (
    "Outrossim, colocamo-nos à disposição para prestar qualquer esclarecimento ou "
    "para tomar providências adicionais que se façam necessárias."
)

# Vocativos aceitos
VOCATIVO_DEFAULT = "Senhor(a) Deputado(a),"

MESES = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]


def data_extenso(dia: int, mes: int, ano: int) -> str:
    d = "1º" if dia == 1 else str(dia)
    return f"{d} de {MESES[mes - 1]} de {ano}"
