# -*- coding: utf-8 -*-
"""Configurações, caminhos e blocos de texto fixos do conversor."""
from __future__ import annotations

import os
import re
from pathlib import Path

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
# Fontes públicas oficiais para mentions internas do Notion
# ---------------------------------------------------------------------------
# O texto das mentions (@página das bases internas, ex. Vademecum) já cita o
# dispositivo de origem ("RICD - Art. 54", "Constituição Federal - Art. 24, VII").
# No .docx o link interno do Notion nunca é preservado (o público não acessa as
# bases); quando o padrão abaixo casa, o texto ganha hyperlink para a fonte
# pública oficial. Cada entrada é (regex, template de URL com \1 etc., aliases
# do nome da norma — usados também para podar o "eco" redundante que o texto ao
# redor às vezes repete, ex. "(RICD - Art. 151, RICD)" -> "(RICD - Art. 151)").
# Extensível: novas normas = novas tuplas. Resolver via propriedades da página
# mencionada (norma_id) exigiria chamadas extras à API — fora de escopo.
FONTES_OFICIAIS = [
    (r"^Constitui[çc][ãa]o Federal\s*[-–—]\s*Art\.?\s*(\d+)",
     r"https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm#art\1",
     ("Constituição Federal", "CF")),
    # Resolução da Câmara nº 17/1989 (RICD), texto consolidado no LEGIN.
    (r"^RICD\s*[-–—]\s*Art\.?\s*\d+",
     r"https://www2.camara.leg.br/legin/fed/rescad/1989/"
     r"resolucaodacamaradosdeputados-17-21-setembro-1989-320110-normaatualizada-pl.html",
     ("RICD",)),
]


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
    "Outrossim, aproveitamos o ensejo para renovar votos de estima e consideração e "
    "colocarmo-nos ao dispor para prestar qualquer esclarecimento ou para tomar providências "
    "adicionais que se façam necessárias."
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
