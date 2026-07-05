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

OPENAI_MODEL = "gpt-4o"

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


def ancora_artigo_ricd(num: int) -> str:
    """Fragmento de destaque de texto (#:~:text=) que abre a página do LEGIN
    já no caput do artigo. A página consolidada não tem âncoras HTML próprias
    (só bookmarks residuais de Word), então usa-se o text fragment do browser —
    mesmo recurso já presente nos links dos documentos-modelo da casa. O alvo
    reproduz a grafia do caput no LEGIN: "Art. 5º" (1º-9º) / "Art. 54." (10+);
    aferido contra o HTML real: em 264 dos 278 artigos a 1ª ocorrência é o
    próprio caput; nos demais o link apenas destaca uma remissão anterior."""
    alvo = f"Art.%20{num}%C2%BA" if num < 10 else f"Art.%20{num}."
    return f"#:~:text={alvo}"


# Resoluções do TSE com página compilada conhecida (a URL traz a data por
# extenso e não é derivável do número — mapa extensível, chave normalizada
# "NN.NNN"). Fora do mapa -> sem link (mention vira texto puro; citação
# textual não é linkada).
RESOLUCOES_TSE = {
    "23.610": "https://www.tse.jus.br/legislacao/compilada/res/2019/"
              "resolucao-no-23-610-de-18-de-dezembro-de-2019",
    "23.607": "https://www.tse.jus.br/legislacao/compilada/res/2019/"
              "resolucao-no-23-607-de-17-de-dezembro-de-2019",
    "23.735": "https://www.tse.jus.br/legislacao/compilada/res/2024/"
              "resolucao-no-23-735-de-27-de-fevereiro-de-2024",
}


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
    return RICD_LEGIN_URL + ancora_artigo_ricd(int(d["num"]))


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
        citacao_re=(r"RICD\b|Regimento\s+Interno(?:\s+da\s+C[âa]mara(?:\s+dos\s+"
                    r"Deputados)?)?"),
        citacao_isolada_re=None,
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
        citacao_re=(r"C[óo]digo\s+Eleitoral"
                    r"|Lei\s*n?[ºo°.]*\s*4\.?737"
                    r"(?:\s*/\s*(?:19)?65|,?\s+de(?:\s+15\s+de\s+julho\s+de)?\s+1965)?"),
        citacao_isolada_re=(r"Lei\s*n?[ºo°.]*\s*4\.?737"
                            r"(?:\s*/\s*(?:19)?65|,?\s+de(?:\s+15\s+de\s+julho\s+de)?\s+1965)"),
        montar_url=_fonte_planalto(L4737_PLANALTO_URL),
        aliases=("Código Eleitoral",),
    ),
    NormaOficial(
        # Base Vademecum "Res.-TSE n. 23.735/2024 - Ilicitos eleitorais"
        mention_re=(r"^Res(?:\.|olu[çc][ãa]o)?\s*[-–.]?\s*TSE\s*n[ºo°.]*\s*"
                    r"(?P<res>\d{2}\.?\d{3})(?:\s*/\s*\d{4})?"),
        citacao_re=(r"Resolu[çc][ãa]o\s*(?:do\s+)?TSE\s*n[ºo°.]*\s*"
                    r"(?P<res>\d{2}\.?\d{3})\s*(?:/\s*\d{4}|,?\s+de\s+[^,;()]{4,40}?\d{4})"),
        citacao_isolada_re=(r"Resolu[çc][ãa]o\s*(?:do\s+)?TSE\s*n[ºo°.]*\s*"
                            r"(?P<res>\d{2}\.?\d{3})\s*(?:/\s*\d{4}|,?\s+de\s+[^,;()]{4,40}?\d{4})"),
        montar_url=_fonte_res_tse,
        aliases=(),
    ),
]

# Formato consumido por richtext.resolver_fonte_publica/_aliases_da_norma:
# (regex do texto da mention, callable(Match) -> URL ou None, aliases).
FONTES_OFICIAIS = [(n.mention_re, n.montar_url, n.aliases) for n in NORMAS_OFICIAIS]


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
