# -*- coding: utf-8 -*-
"""
Este script converte TODOS os relatórios .txt da pasta atual em arquivos CSV,
utilizando a API da OpenAI para extrair dados de forma estruturada.

*** MELHORIAS NESTA VERSÃO ***
- MODO DETALHADO POR PADRÃO (AMIGÁVEL): Execute o script normalmente para acompanhar
  o progresso com mensagens simples. Use --quiet para reduzir os logs e --debug
  para ver detalhes técnicos.
- PROMPTS REFINADOS: As instruções para a IA foram melhoradas para aumentar a
  precisão e o preenchimento correto dos campos do JSON.
- MODELO ATUALIZADO: Utiliza o 'gpt-5-mini', um modelo mais moderno e eficiente
  para este tipo de tarefa.
- ROBUSTEZ: Melhor tratamento de erros e retentativas de conexão com a API.

*** COMO USAR ***
1. Crie um arquivo .env na mesma pasta e adicione sua chave da OpenAI:
   OPENAI_API_KEY="sua_chave_aqui"
2. Instale as dependências: pip install -r requirements.txt
3. Execute no terminal:
   - Execução padrão (detalhada e amigável): python SESSÕES_TSE.py
   - Execução silenciosa: python SESSÕES_TSE.py --quiet
   - Execução técnica (com detalhes completos): python SESSÕES_TSE.py --debug
   - Para informar arquivos direto no comando: python SESSÕES_TSE.py caminho\\arquivo1.txt caminho\\arquivo2.txt
4. Se nenhum arquivo for informado, o Windows Explorer será aberto para seleção.
5. Os arquivos CSV convertidos aparecerão na mesma pasta do .txt original, com o mesmo nome.
"""

import os
import json
import time
import logging
import argparse
import threading
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError

# --- 1. Configuração e Prompts ---

# Carrega as variáveis de ambiente (sua chave da API)
load_dotenv()

class FriendlyFormatter(logging.Formatter):
    LEVEL_MAP = {
        "DEBUG": "DETALHE",
        "INFO": "INFO",
        "WARNING": "AVISO",
        "ERROR": "ERRO",
        "CRITICAL": "ERRO",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = self.LEVEL_MAP.get(record.levelname, record.levelname)
        return super().format(record)

def configure_logging(args: argparse.Namespace) -> None:
    global SHOW_TECHNICAL_LOGS
    log_level = logging.INFO if args.quiet else logging.DEBUG
    if args.verbose:
        log_level = logging.DEBUG
    if args.debug:
        log_level = logging.DEBUG
        SHOW_TECHNICAL_LOGS = True

    handler = logging.StreamHandler()
    formatter = FriendlyFormatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(handler)

    # Evita logs técnicos de bibliotecas HTTP.
    for noisy_logger in ("openai", "openai._base_client", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Inicializa o cliente da OpenAI
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("A variável OPENAI_API_KEY não foi encontrada no arquivo .env.")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível inicializar o cliente da OpenAI. Verifique seu arquivo .env. Detalhes: {e}")
    exit(1)

# Instrução principal para a IA, focada em precisão e no contexto jurídico.
SYSTEM_PROMPT = """
Você é um assistente de IA especializado em análise de documentos jurídicos, focado na extração de dados de relatórios de sessões do TSE.
Sua tarefa é analisar meticulosamente o relatório fornecido e preencher CADA CAMPO do esquema JSON com a informação correspondente de forma detalhada e completa.
Geralmente são julgados de 1 a 3 processos por sessão. Por isso, a quantidade de linhas no CSV final varia de acordo com cada julgado apreciado no TSE.

REGRAS IMPORTANTES:
1.  **EXTRAÇÃO COMPLETA**: Faça o seu melhor para preencher todos os campos. Se uma informação não estiver, deixe o campo como uma string vazia `""`.
2.  **REGRAS DE FOCO**: Foco na extração de "data_sessao", "numero_processo", "partes", "advogados", "tema", "analise_do_conteudo_juridico" e "raciocinio_juridico", pois são PONTOS CRÍTICOS DE ALTA RELEVÂNCIA.
3.  **PADRÃO DE PREENCHIMENTO**: Para campos que esperam múltiplos valores (como 'composicao', 'partes', 'advogados'), extraia TODOS os itens e separe-os usando vírgula seguido de um espaço (`, `).
4.  **SAÍDA ESTRITA**: O resultado final DEVE SER um único objeto JSON válido, sem nenhum texto, comentário, explicação ou formatação `markdown` (como ```json) antes ou depois.
5.  **ITEM ÚNICO**: Quando o texto fornecido for uma seção individual de um processo, retorne APENAS 1 item no array "itens".
6.  **JULGAMENTO CONJUNTO**: Se houver mais de um número de processo no texto, retorne um item por processo no array "itens". Se algum dado for comum, repita-o nos itens.
7.  **PADRÕES ESPECÍFICOS**:
    - `numero_processo`: extraia SEMPRE o número completo do CNJ (ex: `0000000-00.0000.0.00.0000`). Só use o formato curto se o CNJ completo não existir no texto.
    - `TRE`: use o formato `TRE-UF` quando a UF estiver indicada.
    - `composicao`, `relator` e `pedido_vista`: sempre use o prefixo `Min.` antes do nome.
8.  **RACIOCÍNIO JURÍDICO DETALHADO**: Explique claramente a tese vencedora, a tese divergente (se houver), e indique ministros vencedores e vencidos quando o texto citar. Se a decisão for unânime, deixe isso explícito.
9.  **ADVOGADOS**: Sempre prefixe os nomes com `Dr.` (masculino) ou `Dra.` (feminino).
"""

# Esquema que define a estrutura dos dados a serem extraídos.
JSON_SCHEMA = {
    "sessao": {
        "data_sessao": "string (DD/MM/AAAA)",
        "composicao": "string (nomes dos ministros presentes, separados por ', 'Sempre usar 'Min.' antes do nome",
    },
    "itens": [
        {
            "numero_processo": "string (ex: 0600264-60.2024.6.00.0000, 0600758-23.2022.6.00.0000, etc.)",
            "eleicao": "string (ex: 2024, 2022, 2020, etc.)",
            "classe_processo": "string (ex: REspe, AgRg-REspe, RO, AREspe, etc.). Adote SEMPRE a sigla",
            "origem": "string (Município/UF).",
            "TRE": "string (ex: TRE-SP)",
            "partes": "string (NOMES dos candidatos, dos partidos políticos, das coligações envolvidas e dos indicados em lista tríplice, separados por ', ')",
            "advogados": "string (NOMES de todos os advogados, separados por ', ', sempre prefixar 'Dr.' ou 'Dra.' conforme o gênero)",
            "relator": "string (o nome do relator ou relatora da decisão. Sempre usar 'Min.' antes do nome)",
            "tipo_registro": "string (ex: 'Julgamento 1' | 'Julgamento 2' | 'Julgamento 3'",
            "analise_do_conteudo_juridico": "string (resumo robusto do contexto fático julgado e do tema eleitoral)",
            "tema": "string (o título principal que resume o assunto tratado em até 20 palavras)",
            "punchline": "string (Crie uma frase de efeito, curta e de alto impacto, resumindo o julgado.)",
            "resoluções_citadas": "string (resoluções do TSE citadas, separadas por ', ' com número, ano e artigo.",
            "fundamentacao_normativa": "string (principais artigos, leis e súmulas citados, separados por ', '",
            "precedentes_citados": "string (resumo dos precedentes citados, separados por ', ')",
            "raciocinio_juridico": "string (descrição robusta da tese vencedora, a tese divergente se houver, os argumentos, a conclusão final do colegiado e os ministros vencedores/vencidos quando citados.)",
            "pedido_vista": "string (nome do ministro que pediu vista, se houver. Sempre usar 'Min.' antes do nome. Se não houver, deixar vazio)",
            "resultado_final": "string (somente essas: Provido, Desprovido, Não conhecido, Suspenso por vista, Aprovada, Rejeitada, Prejudicado, Anulado, Parcialmente acolhido - com/sem efeitos modificativos. Se houver mais de um resultado, separar por ', ')",
            "votacao": "string (somente essas: Unânime, Por maioria, Suspenso)",
            "youtube_link": "string (links do vídeo da sessão no YouTube, separados por ', ' se houver mais de um. UTILIZAR SEMPRE links similares a esse: 'http://www.youtube.com/watch?v=K1QFVHrtzqg)",
        }
    ]
}

# JSON Schema para forçar saída estruturada via response_format.
JSON_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sessao": {
            "type": "object",
            "properties": {
                "data_sessao": {"type": "string"},
                "composicao": {"type": "string"},
            },
            "required": ["data_sessao", "composicao"],
            "additionalProperties": False,
        },
        "itens": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "numero_processo": {"type": "string"},
                    "eleicao": {"type": "string"},
                    "classe_processo": {"type": "string"},
                    "origem": {"type": "string"},
                    "TRE": {"type": "string"},
                    "partes": {"type": "string"},
                    "advogados": {"type": "string"},
                    "relator": {"type": "string"},
                    "tipo_registro": {"type": "string"},
                    "analise_do_conteudo_juridico": {"type": "string"},
                    "tema": {"type": "string"},
                    "punchline": {"type": "string"},
                    "resoluções_citadas": {"type": "string"},
                    "fundamentacao_normativa": {"type": "string"},
                    "precedentes_citados": {"type": "string"},
                    "raciocinio_juridico": {"type": "string"},
                    "pedido_vista": {"type": "string"},
                    "resultado_final": {"type": "string"},
                    "votacao": {"type": "string"},
                    "youtube_link": {"type": "string"},
                },
                "required": [
                    "numero_processo",
                    "eleicao",
                    "classe_processo",
                    "origem",
                    "TRE",
                    "partes",
                    "advogados",
                    "relator",
                    "tipo_registro",
                    "analise_do_conteudo_juridico",
                    "tema",
                    "punchline",
                    "resoluções_citadas",
                    "fundamentacao_normativa",
                    "precedentes_citados",
                    "raciocinio_juridico",
                    "pedido_vista",
                    "resultado_final",
                    "votacao",
                    "youtube_link",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sessao", "itens"],
    "additionalProperties": False,
}

CSV_COLUMN_ORDER = [
    "data_sessao",
    "TRE",
    "advogados",
    "analise_do_conteudo_juridico",
    "classe_processo",
    "composicao",
    "eleicao",
    "fundamentacao_normativa",
    "numero_processo",
    "origem",
    "partes",
    "pedido_vista",
    "precedentes_citados",
    "punchline",
    "raciocinio_juridico",
    "relator",
    "resoluções_citadas",
    "resultado_final",
    "tema",
    "tipo_registro",
    "votacao",
    "youtube_link",
]

SHOW_TECHNICAL_LOGS = False

CLASSE_PROCESSO_MAP = [
    (r"agravo regimental no agravo em recurso especial eleitoral", "AgRg-AREspe"),
    (r"agravo regimental no recurso especial eleitoral", "AgRg-REspe"),
    (r"agravo em recurso especial eleitoral", "AREspe"),
    (r"recurso especial eleitoral", "REspe"),
    (r"agravo regimental", "AgRg"),
]

CNJ_REGEX = r"\b\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b"
SHORT_PROCESSO_REGEX = r"\b\d{6,7}-\d{2}\b"

STATE_UF = {
    "acre": "AC",
    "alagoas": "AL",
    "amapá": "AP",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceará": "CE",
    "ceara": "CE",
    "distrito federal": "DF",
    "espírito santo": "ES",
    "espirito santo": "ES",
    "goiás": "GO",
    "goias": "GO",
    "maranhão": "MA",
    "maranhao": "MA",
    "mato grosso": "MT",
    "mato grosso do sul": "MS",
    "minas gerais": "MG",
    "pará": "PA",
    "para": "PA",
    "paraíba": "PB",
    "paraiba": "PB",
    "paraná": "PR",
    "parana": "PR",
    "pernambuco": "PE",
    "piauí": "PI",
    "piaui": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondônia": "RO",
    "rondonia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "são paulo": "SP",
    "sao paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}

def get_user_prompt(
    report_text: str,
    base_context: Optional[Dict[str, Any]] = None,
    prefilled_fields: Optional[Dict[str, Any]] = None,
    single_item: bool = False,
    process_numbers: Optional[List[str]] = None
) -> str:
    """Cria o prompt do usuário com as instruções, o esquema e o texto do relatório."""
    base_context = base_context or {}
    prefilled_fields = prefilled_fields or {}

    context_block = ""
    if base_context:
        context_block += "\nCONTEXTO BASE (use se estiver coerente com o texto):\n"
        context_block += json.dumps(base_context, indent=2, ensure_ascii=False)
    if prefilled_fields:
        context_block += "\n\nCAMPOS PRE-PREENCHIDOS (use como base e corrija se necessário):\n"
        context_block += json.dumps(prefilled_fields, indent=2, ensure_ascii=False)

    single_item_note = "\nRETORNE APENAS 1 item no array \"itens\".\n" if single_item else ""
    joint_note = ""
    if process_numbers and len(process_numbers) > 1 and not single_item:
        joined = ", ".join(process_numbers)
        joint_note = f"\nPROCESSOS IDENTIFICADOS (retorne um item por processo):\n{joined}\n"

    return f"""
Por favor, extraia os dados do relatório de sessão do TSE abaixo.
Siga RIGOROSAMENTE o esquema JSON fornecido para estruturar sua resposta.
{single_item_note}
{joint_note}
{context_block}

ESQUEMA JSON DE SAÍDA:
{json.dumps(JSON_SCHEMA, indent=2, ensure_ascii=False)}

RELATÓRIO PARA ANÁLISE:
---
{report_text}
---
"""

# --- 2. Extração e Processamento de Dados ---

def normalize_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00ad": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text

def extract_full_cnj(text: str) -> str:
    match = re.search(CNJ_REGEX, text)
    return match.group(0) if match else ""

def extract_short_processo(text: str) -> str:
    match = re.search(SHORT_PROCESSO_REGEX, text)
    return match.group(0) if match else ""

def clean_label_value(value: str) -> str:
    value = value.replace("**", "").strip()
    value = re.sub(r"^\*+\s*", "", value).strip()
    value = re.sub(r"\s*\*+$", "", value).strip()
    return value.rstrip(".").strip()

def normalize_token(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")

FEMALE_NAME_HINTS = {
    "ana", "maria", "mariana", "marina", "carla", "claudia", "clara", "camila",
    "carolina", "caroline", "beatriz", "bianca", "renata", "fernanda", "patricia",
    "luciana", "lucia", "marcia", "sandra", "silvana", "viviane", "oneida",
    "andreia", "adriana", "tatiana", "vanessa", "aline", "leticia", "raquel",
    "cristina", "cristiane", "gisele", "giovana", "gabriela", "isabel", "marta",
    "claudia", "carmen", "estela", "juliana", "julia", "amanda", "bruna",
    "daniela", "elaine", "eliane", "joana", "laura", "luana", "marcela", "patricia",
}

def infer_advogado_prefix(name: str, label_hint: str = "") -> str:
    hint = label_hint.lower()
    if "advogada" in hint:
        return "Dra."
    if "advogado" in hint:
        return "Dr."
    first = name.split()[0] if name.split() else ""
    if normalize_token(first) in FEMALE_NAME_HINTS:
        return "Dra."
    return "Dr."

def normalize_advogado_name(name: str, label_hint: str = "") -> str:
    name = name.strip()
    if not name:
        return ""
    suffix = ""
    if "(" in name:
        base, extra = name.split("(", 1)
        name = base.strip()
        suffix = " (" + extra.strip()
    name = name.rstrip(".;:,").strip()
    if not name:
        return ""
    prefix = ""
    if re.match(r"(?i)^dra\.?\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^dra\.?\s+", "", name).strip()
    elif re.match(r"(?i)^dr\.?\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^dr\.?\s+", "", name).strip()
    elif re.match(r"(?i)^doutora\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^doutora\s+", "", name).strip()
    elif re.match(r"(?i)^doutor\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^doutor\s+", "", name).strip()
    elif re.match(r"(?i)^(sra|srta|senhora)\.?\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^(sra|srta|senhora)\.?\s+", "", name).strip()
    elif re.match(r"(?i)^(sr|senhor)\.?\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^(sr|senhor)\.?\s+", "", name).strip()
    if not name:
        return ""
    if not prefix:
        prefix = infer_advogado_prefix(name, label_hint)
    return f"{prefix} {name}{suffix}".strip()

def normalize_advogados_list(value: str, label_hint: str = "") -> str:
    if not value:
        return ""
    if re.search(r"não\s+mencionad|não\s+há|não\s+informad", value, flags=re.IGNORECASE):
        return ""
    text = normalize_text(value)
    parts = split_advogados_entries(text)
    normalized = []
    for part in parts:
        part = re.sub(r"(?i)^(advogad[oa]s?|defensor[oa]s?)\s*:?\s*", "", part).strip()
        if not part:
            continue
        normalized_name = normalize_advogado_name(part, label_hint)
        if normalized_name:
            normalized.append(normalized_name)
    return ", ".join(dedupe_preserve_order(normalized))

def split_advogados_entries(text: str) -> List[str]:
    parts = []
    buffer = ""
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if depth == 0:
            if text[index:index + 3].lower() == " e ":
                if buffer.strip():
                    parts.append(buffer.strip())
                buffer = ""
                index += 3
                continue
            if char in ",;":
                if buffer.strip():
                    parts.append(buffer.strip())
                buffer = ""
                index += 1
                continue
        buffer += char
        index += 1
    if buffer.strip():
        parts.append(buffer.strip())
    return parts

def dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique

def extract_process_numbers_from_line(line: str) -> List[str]:
    line = normalize_text(line)
    fulls = re.findall(CNJ_REGEX, line)
    if fulls:
        return dedupe_preserve_order(fulls)
    shorts = re.findall(SHORT_PROCESSO_REGEX, line)
    return dedupe_preserve_order(shorts)

def extract_ministro_names_from_line(line: str) -> List[str]:
    line = normalize_text(line).strip()
    if not line:
        return []
    line = re.sub(r"^\s*[*\-•]+\s*", "", line)
    line = re.sub(r"^\s*\d+\.\s*", "", line)
    line = re.sub(r"^\s*\d+\)\s*", "", line)
    line = line.replace("**", "").strip()
    line = re.sub(r"^\*{1,2}|\*{1,2}$", "", line).strip()
    line = re.split(r"(?i)\b(?:nota|obs|observa[cç][aã]o|não votam|nao votam)\s*:", line)[0].strip()
    line = re.sub(r"\s+e\s+", ", ", line)
    lower_line = line.lower()
    if (
        ("procurador" in lower_line or "pge" in lower_line or "mpe" in lower_line
         or "ministério público" in lower_line or "ministerio publico" in lower_line)
        and not re.search(r"(?i)\bmin\.|\bministro\b|\bministra\b", line)
    ):
        return []
    parts = re.split(r"\s*[;,]\s*", line)
    names = []
    label_prefix = (
        r"(ministros?\s+presentes?|"
        r"ministers?\s+present(?:es)?|"
        r"ministros?\s+que\s+comp[oó]em\s+a\s+sess[aã]o|"
        r"composi[cç][aã]o\s+da\s+(?:mesa(?:\s+julgadora)?|"
        r"corte(?:\s+na\s+sess[aã]o)?|plen[aá]rio)|"
        r"presid[êe]ncia(?:\s*\(.*?\))?|"
        r"presidente(?:\s+da\s+sess[aã]o|\s+do\s+julgamento)?|"
        r"vice-?presidente|"
        r"ministro\s+substituto|"
        r"ministra\s+substituta|"
        r"aus[eê]nci\w*|"
        r"ausent\w*)"
    )
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = re.sub(rf"(?i)^{label_prefix}\s*:?\s*", "", part).strip()
        part = part.lstrip(":- ").strip()
        part = re.sub(r"\s*\(.*?\)\s*", "", part).strip()
        part = part.lstrip(":- ").strip()
        if not part:
            continue
        lower = part.lower()
        if "composi" in lower and ("completa" in lower or "mesma" in lower or "vide item" in lower):
            continue
        if "presidente da república" in lower or "presidente da republica" in lower:
            continue
        if "ministério" in lower or "ministerio" in lower:
            continue
        if "secretário" in lower or "secretario" in lower:
            continue
        if not re.search(r"[A-Za-zÁÉÍÓÚÂÊÔÃÕÇ]", part):
            continue
        names.append(normalize_ministro_name(part))
    return [name for name in names if name]

def is_ministros_block_terminator(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("---"):
        return True
    if re.match(r"^#{2,4}\s+", stripped):
        return True
    if re.search(r"(?i)^\s*\*{0,2}(MINIST[ÉE]RIO|SECRET[ÁA]RIO|PRESID[ÊE]NCIA|DATA DA SESS[ÃA]O)\b", stripped):
        return True
    if re.search(r"(?i)^\s*\*{0,2}[A-ZÁÉÍÓÚÃÕÇ][A-ZÁÉÍÓÚÃÕÇ\s\-]+:\s*", stripped):
        return True
    return False

def extract_dados_processo_block(text: str) -> str:
    match = re.search(
        r"(?im)^\s*.*\bDADOS\s+DO(?:S)?\s+(?:PROCESSO(?:S)?|CASO)\b.*$",
        text
    )
    if not match:
        return ""
    start = match.end()
    next_match = re.search(
        r"(?im)^\s*(?:#{2,4}\s+|"
        r"\*{0,2}(?:ANÁLISE DO CONTEÚDO JURÍDICO|ANÁLISE|"
        r"RACIOCÍNIO DESENVOLVIDO|RACIOCÍNIO|CONCLUSÃO|"
        r"PONTOS PROCESSUAIS|EFEITOS|OBS|OBSERVA)\b.*)$",
        text[start:]
    )
    end = start + next_match.start() if next_match else len(text)
    return text[start:end].strip()

def extract_process_numbers_from_section(text: str) -> Tuple[List[str], bool]:
    numbers = []
    plural_label = False
    retificacao_present = False
    header_lines = text.splitlines()[:5]
    for line in header_lines:
        if re.search(r"retific", line, flags=re.IGNORECASE):
            retificacao_present = True
        line_numbers = extract_process_numbers_from_line(line)
        if line_numbers:
            numbers.extend(line_numbers)
    for match in re.finditer(
        r"(?im)^\s*[*-]?\s*\*{0,2}Processos?\*{0,2}\s*:\s*([^\n]+)",
        text
    ):
        label = match.group(0)
        if re.search(r"Processos", label, flags=re.IGNORECASE):
            plural_label = True
        line = clean_label_value(match.group(1))
        line = normalize_text(line)
        if re.search(r"retific", line, flags=re.IGNORECASE):
            retificacao_present = True
        line_numbers = extract_process_numbers_from_line(line)
        if line_numbers:
            numbers.extend(line_numbers)

    dados_block = extract_dados_processo_block(text)
    if dados_block:
        skip_labels = (
            "jurisprud",
            "precedent",
            "legisla",
            "fundamenta",
            "resolu",
            "súmula",
            "sumula",
        )
        for raw_line in dados_block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if any(label in line.lower() for label in skip_labels):
                continue
            line_numbers = extract_process_numbers_from_line(line)
            if line_numbers:
                numbers.extend(line_numbers)

    numbers = dedupe_preserve_order(numbers)
    has_joint_keyword = bool(
        re.search(
            r"(julgament[oa]s?\s+conjunt[oa]s?|"
            r"julgad[oa]s?\s+em\s+conjunto|"
            r"julgamento\s+simult[aâ]neo|"
            r"processos?\s+conexos|"
            r"apensad[oa]s?|"
            r"reunid[oa]s?\s+para\s+julgamento|"
            r"conjunto\s+de\s+agravos)",
            text,
            flags=re.IGNORECASE
        )
    )

    if retificacao_present and not plural_label and not has_joint_keyword and len(numbers) > 1:
        numbers = [numbers[-1]]

    is_joint = (len(numbers) > 1) or has_joint_keyword or plural_label
    return numbers, is_joint

def extract_process_numbers_from_value(value: str) -> List[str]:
    if not value:
        return []
    value = normalize_text(value)
    fulls = re.findall(CNJ_REGEX, value)
    if fulls:
        return dedupe_preserve_order(fulls)
    shorts = re.findall(SHORT_PROCESSO_REGEX, value)
    return dedupe_preserve_order(shorts)

def extract_process_numbers_from_items(items: List[Dict[str, Any]]) -> Tuple[List[str], bool]:
    numbers = []
    multi_found = False
    for item in items:
        item_numbers = extract_process_numbers_from_value(item.get("numero_processo", ""))
        if len(item_numbers) > 1:
            multi_found = True
        numbers.extend(item_numbers)
    return dedupe_preserve_order(numbers), multi_found

def align_items_to_process_numbers(
    items: List[Dict[str, Any]],
    process_numbers: List[str],
    is_joint: bool
) -> List[Dict[str, Any]]:
    if not is_joint:
        return items

    expanded = []
    for item in items:
        item_numbers = extract_process_numbers_from_value(item.get("numero_processo", ""))
        if len(item_numbers) <= 1:
            expanded.append(item)
            continue
        for number in item_numbers:
            clone = item.copy()
            clone["numero_processo"] = number
            expanded.append(clone)

    if not process_numbers or len(process_numbers) <= 1:
        return expanded

    if not expanded:
        return items

    if len(expanded) > len(process_numbers):
        used = set()
        for item in expanded:
            for number in extract_process_numbers_from_value(item.get("numero_processo", "")):
                used.add(number)
        missing = [num for num in process_numbers if num not in used]
        if missing:
            for item in expanded:
                if not item.get("numero_processo") and missing:
                    item["numero_processo"] = missing.pop(0)
            if missing:
                template = expanded[-1]
                for number in missing:
                    clone = template.copy()
                    clone["numero_processo"] = number
                    expanded.append(clone)
        return expanded

    aligned = []
    unused = expanded[:]
    for number in process_numbers:
        match_idx = None
        for idx, item in enumerate(unused):
            item_numbers = extract_process_numbers_from_value(item.get("numero_processo", ""))
            if number in item_numbers:
                match_idx = idx
                break
        if match_idx is not None:
            item = unused.pop(match_idx)
        elif unused:
            item = unused.pop(0)
        else:
            item = expanded[-1].copy()
        item["numero_processo"] = number
        aligned.append(item)
    return aligned

def parse_date_from_text(text: str) -> str:
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if match:
        day, month, year = match.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"

    months = {
        "janeiro": "01",
        "fevereiro": "02",
        "março": "03",
        "marco": "03",
        "abril": "04",
        "maio": "05",
        "junho": "06",
        "julho": "07",
        "agosto": "08",
        "setembro": "09",
        "outubro": "10",
        "novembro": "11",
        "dezembro": "12",
    }
    match = re.search(
        r"\b(\d{1,2})\s+de\s+([A-Za-zçÇãõéíóúÁÉÍÓÚ]+)\s+de\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        day, month_name, year = match.groups()
        month = months.get(month_name.lower())
        if month:
            return f"{int(day):02d}/{month}/{year}"
    return ""

YOUTUBE_LINK_REGEX = (
    r"https?://(?:www\.)?"
    r"(?:youtube\.com/watch\?v=[\w-]+(?:&\S*)?|youtu\.be/[\w-]+(?:\?\S*)?)"
)

def extract_youtube_links(text: str) -> List[str]:
    cleaned = []
    for match in re.finditer(YOUTUBE_LINK_REGEX, text):
        link = match.group(0).rstrip(").,;\"']")
        if link not in cleaned:
            cleaned.append(link)
    return cleaned

def extract_first_youtube_link(text: str) -> str:
    for match in re.finditer(YOUTUBE_LINK_REGEX, text):
        return match.group(0).rstrip(").,;\"']")
    return ""

def normalize_youtube_link(value: str) -> str:
    if not value:
        return ""
    match = re.search(YOUTUBE_LINK_REGEX, value)
    if match:
        return match.group(0).rstrip(").,;\"']")
    if "," in value:
        return value.split(",", 1)[0].strip()
    return value.strip()

def find_label_value(text: str, labels: List[str]) -> str:
    for label in labels:
        pattern = rf"(?im)^\s*[*-]?\s*\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*:\s*([^\n]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_label_value(match.group(1))
        pattern = rf"{re.escape(label)}\s*:\s*([^\n]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_label_value(match.group(1))
    return ""

def extract_uf_from_text(text: str) -> str:
    match = re.search(r"\bUF\s*:\s*([A-Z]{2})\b", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"/([A-Z]{2})\b", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"\(([A-Z]{2})\)", text)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"Tribunal Regional Eleitoral d[oa]\s+([A-Za-zçÇãõéíóúÁÉÍÓÚ ]+)",
        text,
        flags=re.IGNORECASE
    )
    if match:
        state_name = match.group(1).strip().lower()
        uf = STATE_UF.get(state_name)
        if uf:
            return uf
    lower = text.lower()
    for state_name in sorted(STATE_UF.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(state_name)}\b", lower):
            return STATE_UF[state_name]
    return ""

def normalize_tre(value: str, uf: str) -> str:
    if value:
        match = re.search(r"\bTRE-([A-Z]{2})\b", value)
        if match:
            return f"TRE-{match.group(1).upper()}"
    if uf:
        return f"TRE-{uf}"
    return value

def normalize_ministro_name(name: str) -> str:
    name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
    name = name.rstrip(".;:,").strip()
    if not name:
        return ""
    name = re.sub(r"^(e|E)\s+", "", name).strip()
    name = re.sub(r"^Ministr[oa]s?\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^Ministra\s+", "Min. ", name, flags=re.IGNORECASE)
    name = re.sub(r"^Ministro\s+", "Min. ", name, flags=re.IGNORECASE)
    if not name.startswith("Min."):
        name = f"Min. {name}"
    return name

def normalize_composicao(value: str) -> str:
    if not value:
        return ""
    value = value.replace(";", ",")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    normalized = []
    seen = set()
    for part in parts:
        name = normalize_ministro_name(part)
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    return ", ".join(normalized)

def parse_composicao(text: str, base_composicao: str) -> str:
    value = find_label_value(
        text,
        [
            "Ministros presentes",
            "Ministros que compõem a sessão",
            "Ministros que compoem a sessao",
            "Ministros",
            "Composição da Mesa Julgadora",
            "Composição da Mesa",
            "Composição do Plenário",
            "Composição da Corte na sessão",
            "Composição da Corte na sessao",
            "Composição da Corte",
        ]
    )
    if not value:
        extracted = extract_composicao_from_text(text, max_lines=None, include_leadership=False)
        return extracted or base_composicao
    if re.search(r"composi[cç][aã]o completa|vide item anterior|mesma compos", value, flags=re.IGNORECASE):
        return base_composicao
    normalized = normalize_composicao(value)
    if base_composicao and normalized and len(normalized.split(", ")) < 2:
        return base_composicao
    return normalized

def parse_relator(text: str) -> str:
    value = find_label_value(text, ["Ministro Relator", "Ministra Relatora", "Relator", "Relatora"])
    return normalize_ministro_name(value) if value else ""

def extract_advogados_block(text: str) -> Tuple[str, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(
            r"(?im)^\s*[*-]?\s*\*{0,2}\d*\.?\)?\s*(Advogados?|Advogada)\*{0,2}"
            r"(?:\s*\([^)]*\))?\s*:\s*(.*)$",
            line
        )
        if not match:
            continue
        label = match.group(1)
        remainder = clean_label_value(match.group(2))
        names = []
        if remainder:
            names.append(remainder)
        for next_line in lines[index + 1:]:
            if not next_line.strip():
                if names:
                    break
                continue
            if is_composicao_block_terminator(next_line):
                break
            if re.search(r"(?i)^\s*(?:#{1,4}\s+|\d+\.|\d+\))", next_line.strip()):
                break
            cleaned = clean_label_value(next_line)
            cleaned = re.sub(r"^\s*[*-]\s*", "", cleaned)
            cleaned = re.sub(r"(?i)^advogad[oa]s?\s*:?\s*", "", cleaned).strip()
            if cleaned:
                names.append(cleaned)
        if names:
            return ", ".join(names), label
    return "", ""

def parse_advogados(text: str) -> str:
    value, label = extract_advogados_block(text)
    if value:
        return normalize_advogados_list(value, label)
    return ""

def extract_partes(text: str) -> str:
    parties = []
    base = find_label_value(text, ["Nome das partes/interessados", "Nome das partes", "Partes"])
    if base:
        parties.append(base)
    for match in re.finditer(
        r"(?im)^\s*\*?\s*\*\*(?:Parties|Partes)[^:]*\*\*\s*:\s*(.+)$",
        text
    ):
        value = clean_label_value(match.group(1))
        if value:
            parties.append(value)
    unique = []
    seen = set()
    for party in parties:
        if party not in seen:
            seen.add(party)
            unique.append(party)
    return ", ".join(unique)

def parse_eleicao(text: str) -> str:
    value = find_label_value(text, ["Eleição a que se refere", "Eleição"])
    if not value:
        return ""
    if re.search(r"não se aplica", value, flags=re.IGNORECASE):
        return ""
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return match.group(0) if match else value

def extract_section_block(text: str, titles: List[str]) -> str:
    for title in titles:
        pattern = re.compile(
            rf"(?im)^\s*(?:#{{1,4}}\s*)?\*{{0,2}}(?:\d+(?:\.\d+)*\.)?\s*{re.escape(title)}\*{{0,2}}:?\s*$",
            flags=re.IGNORECASE | re.MULTILINE
        )
        match = pattern.search(text)
        if not match:
            continue
        start = match.end()
        next_match = re.search(
            r"(?im)^\s*(?:#{1,4}\s+|\*{0,2}\d+(?:\.\d+)*\.\s*[A-ZÁÉÍÓÚ][A-ZÁÉÍÓÚ\sÇÃÕÂÊÔÍÓÚ\-]*:?\s*$|\*\*Obs:|OBS:|OBSERVA)",
            text[start:]
        )
        end = start + next_match.start() if next_match else len(text)
        return text[start:end].strip()
    return ""

def collapse_block(block: str, separator: str = ", ") -> str:
    if not block:
        return ""
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\*+\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        line = line.strip("-•")
        line = clean_label_value(line)
        if line:
            lines.append(line)
    unique = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return separator.join(unique)

def parse_origem(text: str) -> str:
    city = find_label_value(text, ["Cidade de origem", "Origem"])
    if not city:
        return ""
    city = normalize_text(city)
    city = city.strip().rstrip(".")
    if "/" in city:
        match = re.match(r"^(.*?)/\s*([A-Za-z]{2})$", city.strip())
        if match:
            return f"{match.group(1).strip()}/{match.group(2).upper()}"
        return city
    uf = extract_uf_from_text(city) or extract_uf_from_text(text)
    city = re.sub(r"\s*\(.*?\)\s*", "", city).strip()
    if re.search(r"\s[-–—]\s", city):
        city = re.split(r"\s[-–—]\s", city, maxsplit=1)[0].strip()
    if city and uf:
        return f"{city}/{uf}"
    return city

def extract_processo_e_classe(text: str) -> Dict[str, str]:
    result = {"numero_processo": "", "classe_processo": ""}
    line = find_label_value(text, ["Processo", "Processos"])
    if line:
        line = normalize_text(line)
        line = line.split("(")[0].strip()
        full_cnj = extract_full_cnj(line)
        if full_cnj:
            result["numero_processo"] = full_cnj
        else:
            short_num = extract_short_processo(line)
            if short_num:
                result["numero_processo"] = short_num
        splitter = re.compile(r"\bn[ºo°]\.?\b", flags=re.IGNORECASE)
        parts = splitter.split(line, maxsplit=1)
        result["classe_processo"] = parts[0].strip()
    else:
        result["numero_processo"] = extract_full_cnj(text) or extract_short_processo(text)
    return result

def normalize_classe_processo(value: str) -> str:
    if not value:
        return ""
    lowered = value.lower()
    for pattern, normalized in CLASSE_PROCESSO_MAP:
        if re.search(pattern, lowered):
            return normalized
    return value.strip()

def extract_resolucoes(text: str) -> str:
    matches = re.findall(
        r"Resolu[cç][aã]o(?:-TSE)?\s*n[ºo]?\s*[\d\.]+(?:/\d{4})?(?:\s*\([^)]+\))?",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = []
    for match in matches:
        value = match.strip().rstrip(".")
        if value not in cleaned:
            cleaned.append(value)
    return ", ".join(cleaned)

def extract_resultado_votacao(text: str) -> Dict[str, str]:
    lower = text.lower()
    resultado = ""
    votacao = ""

    if re.search(r"por unanimidade|decis[aã]o un[aâ]nime|aprova[cç][aã]o un[aâ]nime|un[aâ]nime|unanimemente", lower):
        votacao = "Unânime"
    if "por maioria" in lower:
        votacao = "Por maioria"
    if "suspenso" in lower:
        votacao = "Suspenso"
        if "vista" in lower:
            resultado = "Suspenso por vista"

    if "parcial provimento" in lower or "parcialmente" in lower:
        resultado = "Parcialmente acolhido"
    elif (
        "negado provimento" in lower
        or "negou provimento" in lower
        or "nego provimento" in lower
        or "nega provimento" in lower
        or "desprovido" in lower
        or "não provido" in lower
        or "nao provido" in lower
    ):
        resultado = "Desprovido"
    elif re.search(r"\bprovido\b|\bprovimento\b", lower):
        resultado = "Provido"
    elif (
        "não conhecido" in lower
        or "nao conhecido" in lower
        or "não conhecimento" in lower
        or "nao conhecimento" in lower
        or "não conhecer" in lower
        or "nao conhecer" in lower
    ):
        resultado = "Não conhecido"
    elif "aprov" in lower and not resultado:
        resultado = "Aprovada"
    elif "rejeitad" in lower:
        resultado = "Rejeitada"
    elif "prejudicado" in lower:
        resultado = "Prejudicado"
    elif "anulado" in lower:
        resultado = "Anulado"

    return {"resultado_final": resultado, "votacao": votacao}

def extract_pedido_vista(text: str, relator: str) -> str:
    if not re.search(r"pedido de vista|vista|retirou o processo|retirou de pauta", text, flags=re.IGNORECASE):
        return ""
    match = re.search(r"(Min\.\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^,.;\n]+)", text)
    if match:
        return match.group(1).strip()
    return relator

def merge_composicao_values(values: List[str]) -> str:
    names = []
    seen = set()
    for value in values:
        if not value:
            continue
        normalized = normalize_composicao(value)
        for name in [part.strip() for part in normalized.split(",") if part.strip()]:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return ", ".join(names)

COMPOSICAO_SCAN_LIMIT = 240

def find_composicao_header_limit(lines: List[str]) -> int:
    for index, line in enumerate(lines):
        if re.search(r"(?i)relat[óo]rio de julgamentos", line):
            return index
        if re.search(r"(?i)^\s*#{1,4}\s*\d*\.?\s*(processo|julgamento)", line):
            return index
        if re.search(r"(?i)^\s*\d+\)\s*(processo|julgamento)", line):
            return index
        if re.search(r"(?i)^\s*\d+\.\s*(processo|julgamento)", line):
            return index
        if re.search(r"(?i)^\s*dados do processo\b", line):
            return index
    return len(lines)

def detect_composicao_label(line: str) -> str:
    if re.search(r"(?i)\bministros?\s+presentes?\b", line):
        return "presentes"
    if re.search(r"(?i)\bministers?\s+present(?:es)?\b", line):
        return "presentes"
    if re.search(r"(?i)\bministros?\s+que\s+comp[oó]em\s+a\s+sess[aã]o\b", line):
        return "presentes"
    if re.search(r"(?i)\bcomposi[cç][aã]o\s+da\s+(mesa|corte|mesa\s+julgadora|plen[aá]rio)\b", line):
        return "composicao"
    if re.match(r"(?i)^\s*[*-]?\s*\*{0,2}\d*\.?\s*ministros?\s*:", line):
        return "ministros"
    return ""

def is_composicao_block_terminator(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("---"):
        return True
    if re.match(r"^#{2,4}\s+", stripped):
        return True
    if re.search(
        r"(?i)^\s*\*{0,2}\d*\.?\)?\s*\*{0,2}(data da sess[aã]o|sess[aã]o|"
        r"relat[oó]rio|processo|ministro relator|relator|elei[cç][aã]o|"
        r"classe|origem|partes|advogad|jurisprud|fundamenta|racioc[ií]nio|"
        r"conclus[aã]o|resolu[cç][aã]o|legisla[cç][aã]o|resultado|vota[cç][aã]o)\b",
        stripped
    ):
        return True
    if re.search(r"(?i)minist[ée]rio p[uú]blico|vice-procurador|procurador|pge|mpe", stripped):
        return True
    return False

def extract_composicao_block(lines: List[str], start_index: int) -> List[str]:
    block_names = []
    for line in lines[start_index + 1:]:
        if not line.strip():
            continue
        names = extract_ministro_names_from_line(line)
        if names:
            block_names.extend(names)
            continue
        if is_composicao_block_terminator(line):
            break
    return dedupe_preserve_order(block_names)

def extract_composicao_candidates(
    text: str,
    max_lines: Optional[int] = None
) -> List[Tuple[List[str], int, str]]:
    lines = text.splitlines()
    if max_lines:
        lines = lines[:max_lines]
    candidates = []
    for idx, line in enumerate(lines):
        label_type = detect_composicao_label(line)
        if not label_type:
            continue
        inline_names = extract_ministro_names_from_line(line)
        if inline_names:
            candidates.append((dedupe_preserve_order(inline_names), idx, label_type))
            continue
        block_names = extract_composicao_block(lines, idx)
        if block_names:
            candidates.append((block_names, idx, label_type))
    return candidates

def score_composicao_candidate(names: List[str], line_index: int, label_type: str) -> int:
    size = len(names)
    score = 0
    if 6 <= size <= 7:
        score += 6
    elif size == 5 or size == 8:
        score += 3
    elif size == 4:
        score += 1
    if size > 7:
        score -= (size - 7)
    label_bonus = {"presentes": 3, "composicao": 2, "ministros": 1}
    score += label_bonus.get(label_type, 0)
    score += max(0, 3 - (line_index // 40))
    return score

def select_best_composicao_candidate(
    candidates: List[Tuple[List[str], int, str]]
) -> List[str]:
    preferred = [
        (names, line_index, label_type)
        for names, line_index, label_type in candidates
        if label_type == "composicao" and len(names) >= 5
    ]
    if preferred:
        candidates = preferred
    best = None
    best_score = None
    best_index = None
    for names, line_index, label_type in candidates:
        score = score_composicao_candidate(names, line_index, label_type)
        if best is None or score > best_score:
            best = names
            best_score = score
            best_index = line_index
            continue
        if score == best_score:
            if abs(7 - len(names)) < abs(7 - len(best)):
                best = names
                best_index = line_index
            elif abs(7 - len(names)) == abs(7 - len(best)) and line_index < best_index:
                best = names
                best_index = line_index
    return best or []

def extract_absent_ministros(text: str) -> List[str]:
    names = []
    for line in text.splitlines():
        match = re.search(r"(?i)aus[eê]nci\w*.*", line)
        if match:
            names.extend(extract_ministro_names_from_line(match.group(0)))
            continue
        if re.match(r"(?i)^\s*[*-]?\s*\*{0,2}ausent", line):
            match = re.search(r"(?i)ausent\w*.*", line)
            if match:
                names.extend(extract_ministro_names_from_line(match.group(0)))
    return dedupe_preserve_order(names)

def extract_presidencia_name(text: str) -> str:
    value = find_label_value(text, ["Presidência", "Presidente", "Presidente da Sessão"])
    return normalize_ministro_name(value) if value else ""

def extract_leadership_names(lines: List[str]) -> List[str]:
    leaders = []
    for line in lines:
        if re.search(r"(?i)presidente\s+do\s+julgamento", line):
            continue
        if re.match(
            r"(?i)^\s*[*-]?\s*\*{0,2}(presid[êe]ncia(?:\s*\(.*?\))?|"
            r"presidente(?:\s+da\s+sess[aã]o)?|vice-?presidente)\b",
            line
        ):
            leaders.extend(extract_ministro_names_from_line(line))
    return dedupe_preserve_order(leaders)

def extract_composicao_from_text(
    text: str,
    max_lines: Optional[int] = COMPOSICAO_SCAN_LIMIT,
    include_leadership: bool = True
) -> str:
    lines = text.splitlines()
    if max_lines:
        lines = lines[:max_lines]
    header_limit = find_composicao_header_limit(lines)
    header_text = "\n".join(lines[:header_limit])
    candidates = extract_composicao_candidates(header_text)
    if not candidates and max_lines:
        candidates = extract_composicao_candidates(text, None)
    if not candidates:
        return ""
    selected = select_best_composicao_candidate(candidates)
    absent = extract_absent_ministros(text)
    names = [name for name in selected if name not in absent]
    presidencia = extract_presidencia_name(header_text)
    if presidencia and presidencia not in names and presidencia not in absent:
        names.insert(0, presidencia)
    if include_leadership and len(names) < 7:
        lines = text.splitlines()
        if max_lines:
            lines = lines[:max_lines]
        leaders = extract_leadership_names(lines)
        for leader in leaders:
            if leader not in names and leader not in absent:
                names.append(leader)
                if len(names) >= 7:
                    break
    return ", ".join(dedupe_preserve_order(names))

def extract_composicao_from_header(text: str) -> str:
    return extract_composicao_from_text(text, max_lines=COMPOSICAO_SCAN_LIMIT, include_leadership=False)

def extract_global_context(text: str) -> Dict[str, Any]:
    base = {}
    base_date = find_label_value(text, ["Data da Sessão", "Data da sessão"])
    base["data_sessao"] = parse_date_from_text(base_date) or parse_date_from_text(text)
    base["composicao"] = extract_composicao_from_header(text) or parse_composicao(text, "")
    first_youtube_link = extract_first_youtube_link(text)
    if first_youtube_link:
        base["youtube_link"] = first_youtube_link
    return base

def extract_raw_fields(section_text: str, base_context: Dict[str, Any]) -> Dict[str, Any]:
    raw = {}
    raw["tema"] = extract_section_title(section_text)
    raw["data_sessao"] = parse_date_from_text(section_text) or base_context.get("data_sessao", "")
    raw["composicao"] = parse_composicao(section_text, base_context.get("composicao", ""))
    raw["relator"] = parse_relator(section_text)
    raw["advogados"] = parse_advogados(section_text)
    raw["partes"] = extract_partes(section_text)
    raw["precedentes_citados"] = find_label_value(
        section_text,
        ["Jurisprudência citada", "Jurisprudência do TSE mencionada", "Jurisprudência mencionada"]
    )
    raw["fundamentacao_normativa"] = find_label_value(section_text, ["Legislação", "Fundamentação Normativa"])
    if not raw["fundamentacao_normativa"]:
        block = extract_section_block(section_text, ["FUNDAMENTAÇÃO NORMATIVA", "FUNDAMENTAÇÃO"])
        raw["fundamentacao_normativa"] = collapse_block(block, separator=", ")
    if not raw["precedentes_citados"]:
        block = extract_section_block(section_text, ["PRECEDENTES CITADOS", "PRECEDENTES"])
        raw["precedentes_citados"] = collapse_block(block, separator="; ")

    raw["analise_do_conteudo_juridico"] = collapse_block(
        extract_section_block(section_text, ["ANÁLISE DO CONTEÚDO JURÍDICO", "ANÁLISE"]),
        separator=" "
    )
    raciocinio_base = collapse_block(
        extract_section_block(section_text, ["RACIOCÍNIO DESENVOLVIDO", "RACIOCÍNIO"]),
        separator=" "
    )
    pontos_processuais = collapse_block(
        extract_section_block(section_text, ["PONTOS PROCESSUAIS RELEVANTES", "PONTOS PROCESSUAIS"]),
        separator=" "
    )
    efeitos_providencias = collapse_block(
        extract_section_block(section_text, ["EFEITOS E PROVIDÊNCIAS PRÁTICAS", "CONCLUSÃO E EFEITOS", "EFEITOS"]),
        separator=" "
    )
    extras = []
    if pontos_processuais:
        extras.append(f"Pontos processuais: {pontos_processuais}.")
    if efeitos_providencias:
        extras.append(f"Efeitos e providências: {efeitos_providencias}.")
    raw["raciocinio_juridico"] = " ".join([raciocinio_base, *extras]).strip()

    processo_info = extract_processo_e_classe(section_text)
    raw.update(processo_info)
    raw["numero_processo"] = normalize_processo_num(raw.get("numero_processo", ""))
    raw["classe_processo"] = normalize_classe_processo(raw.get("classe_processo", ""))

    raw["eleicao"] = parse_eleicao(section_text)
    raw["origem"] = parse_origem(section_text)

    uf = extract_uf_from_text(section_text)
    if not uf and raw.get("origem"):
        uf = extract_uf_from_text(raw["origem"])
    tre_from_text = normalize_tre("", uf)
    if tre_from_text:
        raw["TRE"] = tre_from_text

    raw["resoluções_citadas"] = extract_resolucoes(section_text)
    resultado_votacao = extract_resultado_votacao(section_text)
    raw.update(resultado_votacao)
    raw["pedido_vista"] = extract_pedido_vista(section_text, raw.get("relator", ""))

    first_youtube_link = extract_first_youtube_link(section_text)
    if first_youtube_link:
        raw["youtube_link"] = first_youtube_link

    return raw

def split_report_into_sections(text: str) -> List[str]:
    def section_has_process_marker(section: str) -> bool:
        if re.search(r"DADOS DO(?:S)? PROCESSO(?:S)?|DADOS DO CASO", section, flags=re.IGNORECASE):
            return True
        if re.search(r"(?im)^\s*\*{0,2}PROCESSOS?\b", section):
            return True
        if re.search(CNJ_REGEX, section) or re.search(SHORT_PROCESSO_REGEX, section):
            return True
        return False

    headings = list(re.finditer(r"(?m)^(#{2,4})\s+.+$", text))
    sections = []
    if headings:
        for index, heading in enumerate(headings):
            level = len(heading.group(1))
            start = heading.start()
            end = len(text)
            for next_heading in headings[index + 1:]:
                next_level = len(next_heading.group(1))
                if next_level <= level:
                    end = next_heading.start()
                    break
            section = text[start:end].strip()
            obs_match = re.search(r"(?im)^\s*(?:\*\*)?OBSERVA", section)
            if obs_match:
                section = section[:obs_match.start()].rstrip()
            if section_has_process_marker(section):
                sections.append(section)
    return sections

def extract_section_title(section_text: str) -> str:
    match = re.search(r"(?m)^#{2,4}\s+\*{0,2}(.+?)\*{0,2}\s*$", section_text)
    if match:
        title = clean_label_value(match.group(1))
        title = re.sub(r"^\s*\d+(?:\.\d+)*\s*[\)\.\-–—]*\s*", "", title)
        title = title.strip()
        return title or "Seção"
    return "Seção"

def merge_session_info(
    extracted_session: Dict[str, Any],
    base_context: Dict[str, Any],
    raw_fields: Dict[str, Any]
) -> Dict[str, Any]:
    data_sessao = (
        extracted_session.get("data_sessao")
        or raw_fields.get("data_sessao")
        or base_context.get("data_sessao")
        or ""
    )
    composicao = (
        extracted_session.get("composicao")
        or raw_fields.get("composicao")
        or base_context.get("composicao")
        or ""
    )
    return {
        "data_sessao": data_sessao,
        "composicao": normalize_composicao(composicao),
    }

def normalize_processo_num(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value)
    full_cnj = extract_full_cnj(value)
    if full_cnj:
        return full_cnj
    short_num = extract_short_processo(value)
    return short_num if short_num else value.strip()

def apply_item_fallbacks(
    item: Dict[str, Any],
    raw_fields: Dict[str, Any],
    base_context: Dict[str, Any],
    registro_index: int
) -> Dict[str, Any]:
    for key, value in raw_fields.items():
        if key in item and not item.get(key) and value:
            item[key] = value

    if not item.get("youtube_link") and base_context.get("youtube_link"):
        item["youtube_link"] = base_context["youtube_link"]

    item["tipo_registro"] = f"Julgamento {registro_index}"

    raw_full_cnj = extract_full_cnj(raw_fields.get("numero_processo", ""))
    item_full_cnj = extract_full_cnj(item.get("numero_processo", ""))
    if raw_full_cnj and not item_full_cnj:
        item["numero_processo"] = raw_full_cnj
    item["numero_processo"] = normalize_processo_num(item.get("numero_processo", ""))

    uf = extract_uf_from_text(item.get("origem", "")) or extract_uf_from_text(raw_fields.get("origem", ""))
    item["TRE"] = normalize_tre(item.get("TRE", ""), uf)

    if item.get("classe_processo"):
        item["classe_processo"] = normalize_classe_processo(item["classe_processo"])
    elif raw_fields.get("classe_processo"):
        item["classe_processo"] = normalize_classe_processo(raw_fields["classe_processo"])

    if item.get("relator"):
        item["relator"] = normalize_ministro_name(item["relator"])
    if item.get("pedido_vista"):
        item["pedido_vista"] = normalize_ministro_name(item["pedido_vista"])

    return item

def apply_sequential_tipo_registro(rows: List[Dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        row["tipo_registro"] = f"Julgamento {index}"

def process_report_text(report_text: str, model: str) -> List[Dict[str, Any]]:
    normalized_text = normalize_text(report_text)
    base_context = extract_global_context(normalized_text)
    sections = split_report_into_sections(normalized_text)

    if not sections:
        process_numbers, is_joint = extract_process_numbers_from_section(normalized_text)
        structured_data = extract_structured_data(
            normalized_text,
            model,
            process_numbers=process_numbers if is_joint else None
        )
        items = structured_data.get("itens", []) or []
        derived_numbers, multi_found = extract_process_numbers_from_items(items)
        if not process_numbers and derived_numbers:
            process_numbers = derived_numbers
        if multi_found:
            is_joint = True
        if is_joint:
            items = align_items_to_process_numbers(items, process_numbers, is_joint)
            structured_data = {**structured_data, "itens": items}
        rows = flatten_data_for_csv(structured_data)
        apply_sequential_tipo_registro(rows)
        return rows

    logging.info(f"Encontrei {len(sections)} processo(s). Vou analisar um por um.")

    rows = []
    registro_index = 1
    for section in sections:
        section_title = extract_section_title(section)
        logging.info(f"Analisando: {section_title}")
        process_numbers, is_joint = extract_process_numbers_from_section(section)
        if is_joint and process_numbers:
            logging.info(
                "Detectei julgamento conjunto com %d processo(s).",
                len(process_numbers)
            )
        raw_fields = extract_raw_fields(section, base_context)
        if is_joint and len(process_numbers) > 1:
            item_prefill = {}
            fallback_fields = {
                k: v for k, v in raw_fields.items()
                if k not in {
                    "numero_processo",
                    "classe_processo",
                    "partes",
                    "advogados",
                    "origem",
                    "eleicao",
                    "TRE",
                }
            }
        else:
            item_prefill = {k: v for k, v in raw_fields.items() if k not in ("data_sessao", "composicao")}
            fallback_fields = raw_fields
        structured_data = extract_structured_data(
            section,
            model,
            base_context=base_context,
            prefilled_fields=item_prefill,
            single_item=not is_joint,
            process_numbers=process_numbers if (is_joint and process_numbers) else None
        )
        items = structured_data.get("itens", []) or []
        derived_numbers, multi_found = extract_process_numbers_from_items(items)
        if not process_numbers and derived_numbers:
            process_numbers = derived_numbers
        if multi_found:
            is_joint = True
        if is_joint:
            items = align_items_to_process_numbers(items, process_numbers, is_joint)
        if not items:
            logging.warning("Esta parte não gerou dados. Vou seguir para a próxima.")
            continue

        sessao_info = merge_session_info(structured_data.get("sessao", {}), base_context, raw_fields)
        for item in items:
            item = apply_item_fallbacks(item, fallback_fields, base_context, registro_index)
            rows.extend(flatten_data_for_csv({"sessao": sessao_info, "itens": [item]}))
            registro_index += 1

    apply_sequential_tipo_registro(rows)
    return rows

def select_input_files() -> List[str]:
    """Abre o Windows Explorer para selecionar arquivos de relatório (.txt)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        logging.error("Não consegui abrir a janela de seleção de arquivos.")
        logging.debug(f"Detalhes do erro: {e}")
        return []

    logging.info("Abrindo a janela do Windows para selecionar arquivos...")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilenames(
        title="Selecione os relatórios do TSE (.txt)",
        filetypes=[("Relatórios .txt", "*.txt"), ("Todos os arquivos", "*.*")]
    )
    root.destroy()
    return list(selected)

def normalize_input_files(file_paths: List[str]) -> List[str]:
    """Filtra e normaliza a lista de arquivos selecionados."""
    normalized = []
    for path in file_paths:
        if not path.lower().endswith(".txt"):
            logging.warning(f"Ignorando arquivo que não é .txt: {path}")
            continue
        if not os.path.isfile(path):
            logging.warning(f"Arquivo não encontrado no caminho informado: {path}")
            continue
        normalized.append(os.path.abspath(path))
    return normalized

def extract_structured_data(
    report_text: str,
    model: str,
    max_retries: int = 3,
    base_context: Optional[Dict[str, Any]] = None,
    prefilled_fields: Optional[Dict[str, Any]] = None,
    single_item: bool = False,
    process_numbers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Envia o texto do relatório para a API da OpenAI e retorna os dados estruturados.
    Implementa uma estratégia de retentativa com backoff exponencial para lidar com
    erros de API.
    """
    prompt = get_user_prompt(
        report_text,
        base_context=base_context,
        prefilled_fields=prefilled_fields,
        single_item=single_item,
        process_numbers=process_numbers
    )
    logging.debug("Texto preparado para envio à IA.")
    logging.debug(f"Tamanho do relatório: {len(report_text)} caracteres. Tamanho do pedido à IA: {len(prompt)} caracteres.")

    for attempt in range(max_retries):
        try:
            logging.info(f"Enviando o relatório para a IA (tentativa {attempt + 1}/{max_retries})...")
            start_time = time.perf_counter()
            stop_event = threading.Event()
            progress_thread = None
            progress_interval = 5 if logging.getLogger().isEnabledFor(logging.DEBUG) else 10

            def log_api_progress():
                elapsed = 0
                interval = progress_interval
                while not stop_event.wait(interval):
                    elapsed += interval
                    logging.info(f"A IA ainda está analisando... {elapsed}s")

            progress_thread = threading.Thread(target=log_api_progress, daemon=True)
            progress_thread.start()

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "tse_relatorio_sessao",
                            "schema": JSON_OUTPUT_SCHEMA,
                            "strict": True,
                        },
                    }
                )
            finally:
                stop_event.set()
                if progress_thread is not None:
                    progress_thread.join(timeout=1)

            elapsed = time.perf_counter() - start_time
            usage = getattr(response, "usage", None)
            logging.info("Resposta da IA recebida em %.2fs.", elapsed)
            if usage:
                logging.debug(
                    "Detalhes de uso: tokens prompt=%s, completion=%s, total=%s",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                )

            content = response.choices[0].message.content
            if SHOW_TECHNICAL_LOGS:
                logging.debug(f"Resposta bruta da IA (JSON, início): {content[:400]}...")

            # Validação final para garantir que é um JSON válido
            parsed_json = json.loads(content)
            logging.info("Resposta validada. Vamos preparar o CSV.")
            return parsed_json

        except json.JSONDecodeError as e:
            logging.warning(
                f"A resposta da IA não veio no formato esperado (tentativa {attempt + 1}). Vamos tentar novamente."
            )
            if SHOW_TECHNICAL_LOGS:
                logging.debug(f"Detalhes do erro: {e}. Conteúdo problemático: {content}")

        except APIError as e:
            logging.warning(
                f"A comunicação com a IA falhou nesta tentativa ({attempt + 1}). Vamos tentar novamente."
            )
            if SHOW_TECHNICAL_LOGS:
                logging.debug(f"Detalhes do erro da API: {e}")

        except Exception as e:
            logging.error(
                f"Ocorreu um problema inesperado nesta tentativa ({attempt + 1}).",
                exc_info=True
            )

        # Lógica de espera (backoff exponencial) antes da próxima tentativa
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logging.info(f"Aguardando {wait_time}s para tentar novamente.")
            time.sleep(wait_time)

    raise ConnectionError(f"Não foi possível obter uma resposta JSON válida da API após {max_retries} tentativas.")

def flatten_data_for_csv(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converte o JSON aninhado em uma lista de dicionários planos para o CSV."""
    sessao_info = json_data.get("sessao", {})
    itens_julgamento = json_data.get("itens", [])

    if not itens_julgamento:
        logging.warning("Não encontrei julgamentos nesta resposta da IA.")
        return []

    logging.debug(f"Transformando {len(itens_julgamento)} julgamento(s) em linhas de CSV.")
    
    csv_rows = []
    for item in itens_julgamento:
        row = {
            "data_sessao": sessao_info.get("data_sessao", ""),
            "composicao": sessao_info.get("composicao", ""),
            **item
        }
        
        # Limpa e padroniza os valores para o CSV
        for key, value in row.items():
            if value is None:
                row[key] = ""
            elif isinstance(value, list):
                # Converte listas em string, embora o prompt já peça para a IA fazer isso
                row[key] = ", ".join(map(str, value))
            else:
                row[key] = str(value).strip()

        row["data_sessao"] = parse_date_from_text(row.get("data_sessao", "")) or row.get("data_sessao", "")
        row["numero_processo"] = normalize_processo_num(row.get("numero_processo", ""))
        row["TRE"] = normalize_tre(row.get("TRE", ""), extract_uf_from_text(row.get("origem", "")))
        row["advogados"] = normalize_advogados_list(row.get("advogados", ""))
        row["classe_processo"] = normalize_classe_processo(row.get("classe_processo", ""))
        row["composicao"] = normalize_composicao(row.get("composicao", ""))
        row["youtube_link"] = normalize_youtube_link(row.get("youtube_link", ""))
        if row.get("relator"):
            row["relator"] = normalize_ministro_name(row["relator"])
        if row.get("pedido_vista"):
            row["pedido_vista"] = normalize_ministro_name(row["pedido_vista"])
        
        csv_rows.append(row)
    
    return csv_rows

# --- 3. Execução Principal ---

def main():
    """
    Função principal que orquestra a leitura, processamento e gravação dos dados.
    """
    parser = argparse.ArgumentParser(description="Converte relatórios .txt do TSE para .csv usando a API da OpenAI.")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Força modo detalhado (padrão quando não há --quiet)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostra detalhes técnicos (conteúdo bruto e erros detalhados)."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduz logs para o nível INFO."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Caminhos para arquivos .txt. Se omitido, abre o Windows Explorer para seleção."
    )
    args = parser.parse_args()

    configure_logging(args)

    if args.quiet:
        logging.info("Modo silencioso ativado. Logs reduzidos.")
    else:
        logging.info("Modo detalhado ativado (padrão).")

    # Modelo recomendado: gpt-5-mini é rápido, barato e muito bom em seguir instruções de formato.
    MODELO_OPENAI = "gpt-5-mini"

    if args.files:
        selected_files = args.files
    else:
        selected_files = select_input_files()

    files_to_process = normalize_input_files(selected_files)

    if not files_to_process:
        logging.warning("Nenhum arquivo selecionado. Encerrando.")
        return

    logging.info(f"Vamos processar {len(files_to_process)} arquivo(s).")

    for input_file_path in files_to_process:
        filename = os.path.basename(input_file_path)
        logging.info(f"Iniciando análise do arquivo: {filename}")
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
            
            if not report_text.strip():
                logging.warning(f"O arquivo '{filename}' está vazio. Não há nada para processar.")
                continue
            
            logging.debug("Texto do relatório carregado com sucesso.")

            flattened_rows = process_report_text(report_text, MODELO_OPENAI)
            
            if not flattened_rows:
                logging.warning(
                    f"Não consegui extrair dados de '{filename}'. Por isso, o CSV não será criado."
                )
                continue

            df = pd.DataFrame(flattened_rows)
            
            # Garante que todas as colunas do esquema existam no DataFrame final
            final_columns = CSV_COLUMN_ORDER
            for col in final_columns:
                if col not in df.columns:
                    df[col] = "" # Adiciona colunas faltantes com valores vazios
            df = df[final_columns] # Reordena as colunas para seguir o padrão

            base_name = os.path.splitext(filename)[0]
            output_dir = os.path.dirname(input_file_path) or "."
            output_csv_path = os.path.abspath(os.path.join(output_dir, f"{base_name}.csv"))

            logging.info(f"Salvando o CSV em: {output_csv_path}")
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            if os.path.isfile(output_csv_path):
                logging.info(f"Pronto! CSV gerado com {len(df)} linha(s).")
            else:
                logging.warning("Tentei salvar o CSV, mas não encontrei o arquivo no disco.")

        except Exception as e:
            logging.error(
                f"Ocorreu um problema ao processar o arquivo {filename}.",
                exc_info=not args.quiet
            )
    
    logging.info("Processamento finalizado.")

if __name__ == "__main__":
    main()
