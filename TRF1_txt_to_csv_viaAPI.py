import os
import re
import json
import pandas as pd
import openai
import time
from dotenv import load_dotenv
import glob

# --- CONFIGURAÇÃO ---

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
# Define a chave da API da OpenAI a partir das variáveis de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verifica se a chave da API foi carregada corretamente
if not openai.api_key:
    print("ERRO: A chave de API da OpenAI não foi encontrada. Verifique seu arquivo .env")
    exit()

# Expressão regular para identificar o final de cada julgado no texto
PADRAO_FINAL_JULGADO = re.compile(r'\([^)]*– PJe, rel\..*?\)\.?', re.DOTALL)

# Template do prompt que será enviado para a API da OpenAI
PROMPT_TEMPLATE = """
# PAPEL E OBJETIVO
Você é um assistente de jurimetria altamente treinado para analisar boletins informativos e decisões judiciais. Sua função é extrair dados específicos do texto fornecido e estruturá-los rigorosamente no formato JSON solicitado.

# TAREFA
Analise o texto a seguir, que corresponde a um informativo de jurisprudência, e extraia as seguintes informações:

# ESPECIFICAÇÃO DOS DADOS (SCHEMA JSON)
- **Tipo_de_processo**: A classificação processual do expediente. (CAMPO CRÍTICO). Geralmente é uma sigla em maiúsculas (ex: "HC", "AI", "Ap", "CJUR") que aparece próximo ao número do processo na referência final. (Tipo: string)
- **numero_processo**: O número completo do processo. (CAMPO CRÍTICO). Procure por um padrão numérico longo com hifens e pontos, como "1027515-85.2024.4.01.0000". Geralmente está na referência final entre parênteses. (Tipo: string)
- **contexto**: Um resumo robusto do caso prático ou da situação que deu origem à discussão, com menção à legislação citada e à jurisprudência/precedentes mencionados (Tipo: string)
- **tese**: A tese jurídica principal, com a referência à legislação e à jurisprudência adotada pelo tribunal. É a "moral da história" jurídica ou também conhecida como "ratio decidendi" (Tipo: string)
- **relator**: O nome do relator ou relatora da decisão. Em caso de Ministro, sempre usar "Min." antes do nome (ex.: Min. Luiz Fux) ou , em caso de desembargadores, "Des." (ex.: Des. Carlos Augusto Pires Brandão). (Tipo: string)
- **data_julgamento**: A data do julgamento ou da publicação do informativo, no formato DD-MM-AAAA. (Tipo: string)
- **tema**: O título principal que resume o assunto tratado em 20 palavras. (Tipo: string)

# REGRAS DE FORMATAÇÃO E EXCEÇÕES
1. O resultado DEVE ser um único bloco de código JSON válido, sem comentários, explicações ou qualquer texto adicional.
2. Se uma informação não for encontrada no texto, o valor para a chave correspondente no JSON deve ser `null`.
3. Preste atenção especial para distinguir "Contexto" (a história do caso) de "Tese" (a decisão/regra jurídica).

# REGRAS DE FOCO ESPECIAL
- Os campos `Tipo_de_processo` e `numero_processo` são muito importantes. Eles quase sempre estão juntos na referência final do julgado, dentro dos parênteses. Analise essa referência com atenção máxima.
- Se você não encontrar um valor claro para eles, e apenas para eles, faça uma segunda verificação no texto antes de retornar `null`.

--- TEXTO PARA ANÁLISE ---

{TEXTO_PARA_ANALISE}

--- FIM DA ANÁLISE ---
"""

def processar_em_lote():
    """
    Encontra todos os arquivos .txt na pasta atual, divide cada um em julgados individuais,
    envia para a API da OpenAI para extração de dados e salva os resultados em um arquivo CSV.
    """
    # Encontra todos os arquivos .txt na pasta atual
    lista_de_arquivos_txt = glob.glob("*.txt")

    if not lista_de_arquivos_txt:
        print("Nenhum arquivo .txt foi encontrado na pasta para processar.")
        return

    print(f"Processamento em lote iniciado para {len(lista_de_arquivos_txt)} arquivo(s).\n")

    # Itera sobre cada arquivo .txt encontrado
    for arquivo_input_txt in lista_de_arquivos_txt:
        # Gera o nome do arquivo de saída dinamicamente
        nome_base, _ = os.path.splitext(arquivo_input_txt)
        nome_arquivo_saida_csv = f"{nome_base}.csv"

        print(f"--- Processando arquivo: {arquivo_input_txt} ---")

        if not os.path.exists(arquivo_input_txt):
            print(f"ERRO: Arquivo '{arquivo_input_txt}' não encontrado.")
            continue  # Pula para o próximo arquivo

        print(f"Lendo e analisando o arquivo: {arquivo_input_txt}...")
        try:
            with open(arquivo_input_txt, 'r', encoding='utf-8') as f:
                conteudo_bruto = f.read()
        except Exception as e:
            print(f"ERRO ao ler o arquivo '{arquivo_input_txt}': {e}")
            continue

        lista_de_julgados_txt = []
        ponteiro_inicio = 0

        # Divide o conteúdo do arquivo em julgados individuais usando a expressão regular
        for match in PADRAO_FINAL_JULGADO.finditer(conteudo_bruto):
            ponteiro_fim = match.end()
            texto_julgado = conteudo_bruto[ponteiro_inicio:ponteiro_fim]
            if texto_julgado.strip():
                lista_de_julgados_txt.append(texto_julgado.strip())
            ponteiro_inicio = ponteiro_fim

        if not lista_de_julgados_txt:
            print("Nenhum julgado individual foi encontrado neste arquivo. Pulando para o próximo.")
            continue

        print(f"Arquivo fatiado com sucesso em {len(lista_de_julgados_txt)} julgados. Iniciando chamadas à API...")

        resultados_finais = []

        # Itera sobre cada julgado extraído para enviar à API
        for i, julgado_txt in enumerate(lista_de_julgados_txt):
            print(f"   Processando Julgado #{i+1} de {len(lista_de_julgados_txt)}...")
            try:
                prompt_final = PROMPT_TEMPLATE.format(TEXTO_PARA_ANALISE=julgado_txt)

                response = openai.chat.completions.create(
                    model="gpt-5-mini",  # Modelo recomendado. Altere se preferir outro.
                    messages=[{"role": "user", "content": prompt_final}],
                    response_format={"type": "json_object"} # Força a saída em JSON
                )
                resposta_json_str = response.choices[0].message.content

                try:
                    # Limpa a string de resposta caso venha dentro de um bloco de código
                    if resposta_json_str.strip().startswith("```json"):
                        resposta_json_str = resposta_json_str.strip()[7:-3]
                    
                    dados_json = json.loads(resposta_json_str)

                    # --- INÍCIO DA CORREÇÃO ---
                    # Verifica se a resposta da API é uma lista de dicionários ou um único dicionário
                    if isinstance(dados_json, list):
                        # Se for uma lista, adiciona cada item da lista aos resultados
                        resultados_finais.extend(dados_json)
                        print(f"   -> Sucesso: {len(dados_json)} julgados extraídos do bloco #{i+1}.")
                    elif isinstance(dados_json, dict):
                        # Se for um dicionário, adiciona-o diretamente
                        resultados_finais.append(dados_json)
                        print(f"   -> Sucesso: Julgado #{i+1} extraído.")
                    else:
                        print(f"   -> AVISO: Formato de JSON inesperado no Julgado #{i+1}.")
                    # --- FIM DA CORREÇÃO ---

                except json.JSONDecodeError:
                    print(f"   -> ERRO DE JSON no Julgado #{i+1}. A IA não retornou um JSON válido.")
                    print(f"      Resposta recebida: {resposta_json_str[:200]}...") # Mostra o início da resposta inválida

                time.sleep(1)  # Pausa para não sobrecarregar a API
            except Exception as e:
                print(f"   -> ERRO GERAL ao processar o Julgado #{i+1}: {e}")

        if not resultados_finais:
            print("\nNenhum dado foi extraído com sucesso para este arquivo.")
            continue

        print("\n   Consolidando resultados...")
        df = pd.DataFrame(resultados_finais)

        df.to_csv(nome_arquivo_saida_csv, index=False, encoding='utf-8-sig')

        print(f"\n   PROCESSO CONCLUÍDO! {len(df)} registros foram salvos em '{nome_arquivo_saida_csv}'\n")

    print("--- Fim do processamento em lote. ---")


if __name__ == "__main__":
    processar_em_lote()