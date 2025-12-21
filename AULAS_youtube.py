# -*- coding: utf-8 -*-
"""
Este script converte TODOS os relatórios .txt da pasta atual em arquivos CSV,
utilizando a API da OpenAI para extrair dados de forma estruturada.

*** MELHORIAS NESTA VERSÃO ***
- MODO VERBOSE: Execute o script com a flag -v (python seu_script.py -v)
  para acompanhar em detalhes o que está acontecendo, incluindo a comunicação
  com a API.
- PROMPTS REFINADOS: As instruções para a IA foram melhoradas para aumentar a
  precisão e o preenchimento correto dos campos do JSON.
- MODELO ATUALIZADO: Utiliza o 'gpt-5-mini', um modelo mais moderno e eficiente
  para este tipo de tarefa.
- ROBUSTEZ: Melhor tratamento de erros e retentativas de conexão com a API.

*** COMO USAR ***
1. Crie um arquivo .env na mesma pasta e adicione sua chave da OpenAI:
   OPENAI_API_KEY="sua_chave_aqui"
2. Instale as dependências: pip install -r requirements.txt
3. Coloque seus arquivos de relatório (.txt) na MESMA PASTA deste script.
4. Execute no terminal:
   - Para execução normal: python tse_extractor_refinado.py
   - Para execução detalhada: python tse_extractor_refinado.py --verbose
5. Os arquivos CSV convertidos aparecerão ao lado dos arquivos .txt originais.
"""

import os
import json
import time
import logging
import argparse
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError

# --- 1. Configuração e Prompts ---

# Carrega as variáveis de ambiente (sua chave da API)
load_dotenv()

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
Você é um assistente de IA especializado em análise de documentos jurídicos, focado na extração de dados aulas de direito.
Sua tarefa é analisar meticulosamente o relatório fornecido e preencher CADA CAMPO do esquema JSON com a informação correspondente de forma detalhada e completa.
Geralmente são tratados vários assuntos diferentes em cada aula. Por isso, a quantidade de linhas no CSV final varia de acordo com cada tenma da aula de direito.

REGRAS IMPORTANTES:
1.  **EXTRAÇÃO COMPLETA**: Faça o seu melhor para preencher todos os campos. Se uma informação não estiver, deixe o campo como uma string vazia `""`.
2.  **REGRAS DE FOCO**: Foco na extração de "tema", "fundamentacao_normativa", "precedentes_citados", pois são PONTOS CRÍTICOS DE ALTA RELEVÂNCIA.
3.  **PADRÃO DE PREENCHIMENTO**: Para campos que esperam múltiplos valores (como 'legislação' e 'jurisprudência'), extraia TODOS os itens e separe-os usando vírgula seguido de um espaço (`, `).
4.  **SAÍDA ESTRITA**: O resultado final DEVE SER um único objeto JSON válido, sem nenhum texto, comentário, explicação ou formatação `markdown` (como ```json) antes ou depois.
"""

# Esquema que define a estrutura dos dados a serem extraídos.
JSON_SCHEMA = {
    "sessao": {
        "tema": "string (o título principal que resume o assunto tratado em até 20 palavras)",
    },
    "itens": [
        {
            "conteudo": "string (ex: 'Conteúdo 1' | 'Conteúdo 2' | 'Conteúdo 3'",
            "analise_do_conteudo_fatico": "string (resumo robusto do contexto fático e da controversia jurídica tratada, com detalhes relevantes, se for um caso concreto)",
            "punchline": "string (Crie uma frase de efeito, curta e de alto impacto, resumindo o assunto tratado em cada conteúdo.)",
            "fundamentacao_normativa": "string (artigos, leis e súmulas citados, separados por ', '",
            "precedentes_citados": "string (precedentes citados, separados por ', ')",
            "raciocinio_juridico": "string (descrição robusta do assunto explicado, dos elementos contrários, se houver, dos argumentos e dos fundamentos expostos no tópico.)",
            "youtube_link": "string (links do vídeo da aula do YouTube)",
        }
    ]
}

def get_user_prompt(report_text: str) -> str:
    """Cria o prompt do usuário com as instruções, o esquema e o texto do relatório."""
    return f"""
Por favor, extraia os dados do relatório de sessão do TSE abaixo.
Siga RIGOROSAMENTE o esquema JSON fornecido para estruturar sua resposta.

ESQUEMA JSON DE SAÍDA:
{json.dumps(JSON_SCHEMA, indent=2, ensure_ascii=False)}

RELATÓRIO PARA ANÁLISE:
---
{report_text}
---
"""

# --- 2. Extração e Processamento de Dados ---

def extract_structured_data(report_text: str, model: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Envia o texto do relatório para a API da OpenAI e retorna os dados estruturados.
    Implementa uma estratégia de retentativa com backoff exponencial para lidar com
    erros de API.
    """
    prompt = get_user_prompt(report_text)
    logging.debug(f"Prompt enviado para a API (primeiros 500 caracteres): {prompt[:500]}...")

    for attempt in range(max_retries):
        try:
            logging.info(f"Iniciando chamada à API OpenAI (tentativa {attempt + 1}/{max_retries})...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            logging.debug(f"Resposta JSON recebida (bruta): {content}")

            # Validação final para garantir que é um JSON válido
            parsed_json = json.loads(content)
            logging.info("Resposta da API recebida e validada como JSON com sucesso.")
            return parsed_json

        except json.JSONDecodeError as e:
            logging.error(f"Erro de decodificação JSON na tentativa {attempt + 1}: {e}. A API não retornou um JSON válido.")
            logging.debug(f"Conteúdo problemático: {content}")

        except APIError as e:
            logging.warning(f"Erro na API OpenAI (tentativa {attempt + 1}): {e}. Tentando novamente...")

        except Exception as e:
            logging.error(f"Ocorreu um erro inesperado (tentativa {attempt + 1}): {e}", exc_info=True)

        # Lógica de espera (backoff exponencial) antes da próxima tentativa
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logging.info(f"Aguardando {wait_time}s antes da próxima tentativa.")
            time.sleep(wait_time)

    raise ConnectionError(f"Não foi possível obter uma resposta JSON válida da API após {max_retries} tentativas.")

def flatten_data_for_csv(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converte o JSON aninhado em uma lista de dicionários planos para o CSV."""
    sessao_info = json_data.get("sessao", {})
    itens_aula = json_data.get("itens", [])

    if not itens_aula:
        logging.warning("O JSON extraído não contém 'itens'.")
        return []

    logging.debug(f"Normalizando {len(itens_aula)} item(ns) para formato CSV.")
    
    csv_rows = []
    for item in itens_aula:
        # **ALTERAÇÃO AQUI**: Combina o 'tema' da sessão com os dados de cada item.
        row = {
            "tema": sessao_info.get("tema", ""),
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
        
        csv_rows.append(row)
    
    return csv_rows

# --- 3. Execução Principal ---

def main():
    """
    Função principal que orquestra a leitura, processamento e gravação dos dados.
    """
    parser = argparse.ArgumentParser(description="Converte relatórios .txt para .csv usando a API da OpenAI.")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Ativa o modo detalhado (verbose) para exibir logs de depuração."
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if args.verbose:
        logging.debug("Modo verbose ativado.")

    WORK_DIR = "."
    # Modelo recomendado: gpt-5-mini é rápido, barato e muito bom em seguir instruções de formato.
    MODELO_OPENAI = "gpt-5-mini"

    files_to_process = [f for f in os.listdir(WORK_DIR) if f.lower().endswith('.txt')]
    
    if not files_to_process:
        logging.warning("Nenhum arquivo .txt encontrado na pasta atual.")
        return

    logging.info(f"Encontrado(s) {len(files_to_process)} relatório(s) para processar: {', '.join(files_to_process)}")

    for filename in files_to_process:
        input_file_path = os.path.join(WORK_DIR, filename)
        logging.info(f"--- Processando relatório: {filename} ---")
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
            
            if not report_text.strip():
                logging.warning(f"O arquivo '{filename}' está vazio. Pulando.")
                continue
            
            logging.debug("Arquivo de relatório lido com sucesso.")

            structured_data = extract_structured_data(report_text, MODELO_OPENAI)
            flattened_rows = flatten_data_for_csv(structured_data)
            
            if not flattened_rows:
                logging.warning(f"Nenhum dado foi extraído de '{filename}'. O CSV não será gerado para este arquivo.")
                continue

            df = pd.DataFrame(flattened_rows)
            
            # **ALTERAÇÃO AQUI**: Garante que as colunas do CSV correspondam ao novo schema.
            final_columns = [
                "tema", # Coluna da sessão
                *JSON_SCHEMA['itens'][0].keys() # Colunas dos itens
            ]
            for col in final_columns:
                if col not in df.columns:
                    df[col] = "" # Adiciona colunas faltantes com valores vazios
            df = df[final_columns] # Reordena as colunas para seguir o padrão

            base_name = os.path.splitext(filename)[0]
            output_csv_path = os.path.join(WORK_DIR, f"{base_name}.csv")

            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logging.info(f"SUCESSO! Arquivo CSV gerado com {len(df)} linha(s) em: {output_csv_path}")

        except Exception as e:
            logging.error(f"Ocorreu um erro CRÍTICO ao processar o arquivo {filename}: {e}", exc_info=args.verbose)
    
    logging.info("--- Processamento de todos os arquivos concluído. ---")

if __name__ == "__main__":
    main()