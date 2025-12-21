import csv
import ast
import os
import re
import glob # Módulo importado para encontrar arquivos

# --- CONFIGURAÇÃO ---
# Mantém a configuração original do seu script
COLUNAS_DESEJADAS = [
    "tipo_de_processo", "numero_processo", "relator", "data_julgamento",
    "tema", "subramo", "contexto", "tese", "bullet_points", "punchline",
    "texto_original"
]
MODO_VERBOSE = True
# --------------------

def normalizar_texto_original(texto: str) -> str:
    """
    Aplica uma série de limpezas para normalizar o texto, lidando com
    quebras de linha e palavras/símbolos separados.
    """
    if not isinstance(texto, str):
        return texto

    # Etapa 1: Normaliza todos os espaços em branco para um único espaço.
    texto_normalizado = ' '.join(texto.split())

    # Etapa 2: Corrige padrões específicos de espaçamento.
    texto_normalizado = re.sub(r'(\d)\s+([ºª])', r'\1\2', texto_normalizado)
    texto_normalizado = re.sub(r'\b(n)\s+([ºª])\s+(s)\b', r'\1\2\3', texto_normalizado, flags=re.IGNORECASE)
    texto_normalizado = re.sub(r'\s+([,\.:;!?])', r'\1', texto_normalizado)
    texto_normalizado = re.sub(r'\(\s+', '(', texto_normalizado)
    texto_normalizado = re.sub(r'\s+\)', ')', texto_normalizado)
    
    return texto_normalizado

def processar_csv(input_filename, output_filename, colunas_desejadas, verbose=False):
    """
    Processa um arquivo CSV para normalizar e reordenar colunas específicas.
    """
    try:
        print(f"\nIniciando processamento de '{input_filename}'...")
        
        linhas_processadas = []
        # Usar 'utf-8-sig' na leitura remove o "Byte Order Mark" (BOM) 
        # que pode corromper o nome da primeira coluna.
        with open(input_filename, mode='r', encoding='utf-8-sig', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            
            # Validação do cabeçalho original
            colunas_faltantes = [col for col in colunas_desejadas if col not in header]
            if colunas_faltantes:
                print(f"ERRO: O cabeçalho do CSV não contém as seguintes colunas: {colunas_faltantes}")
                print(f"Cabeçalho encontrado: {header}")
                return

            # Mapeamento de colunas por índice
            indices_desejados = [header.index(col) for col in colunas_desejadas]
            subramo_index = header.index('subramo') if 'subramo' in header else -1
            texto_original_index = header.index('texto_original') if 'texto_original' in header else -1

            # Adiciona o novo cabeçalho ao arquivo de saída
            linhas_processadas.append(colunas_desejadas)

            for i, row_original in enumerate(reader, 1):
                if not row_original or all(not cell for cell in row_original):
                    continue

                row_modificavel = list(row_original)

                # 1. Trata a coluna 'subramo'
                if subramo_index != -1 and subramo_index < len(row_modificavel) and row_modificavel[subramo_index]:
                    try:
                        lista = ast.literal_eval(row_modificavel[subramo_index])
                        tags = ', '.join([item[0] for item in lista if item and isinstance(item, list)])
                        row_modificavel[subramo_index] = tags
                    except (ValueError, SyntaxError): 
                        pass
                
                # 2. Trata a coluna 'texto_original'
                if texto_original_index != -1 and texto_original_index < len(row_modificavel) and row_modificavel[texto_original_index]:
                    texto_sujo = row_modificavel[texto_original_index]
                    texto_limpo = normalizar_texto_original(texto_sujo)
                    row_modificavel[texto_original_index] = texto_limpo
                    if verbose and texto_sujo != texto_limpo:
                        print(f"Linha {i} ('texto_original'): Normalizado.")

                nova_linha_filtrada = [row_modificavel[idx] for idx in indices_desejados]
                linhas_processadas.append(nova_linha_filtrada)

        with open(output_filename, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(linhas_processadas)

        print(f"\nSUCESSO! Arquivo final gerado: '{output_filename}'")

    except FileNotFoundError:
        print(f"\nERRO: O arquivo de entrada '{input_filename}' não foi encontrado.")
    except Exception as e:
        print(f"\nERRO INESPERADO: {e}")

def main():
    """
    Função principal que localiza automaticamente um arquivo CSV no diretório,
    define os nomes dos arquivos de entrada/saída e chama o processador.
    """
    # Procura por arquivos .csv no diretório atual
    arquivos_csv = glob.glob('*.csv')
    
    # Filtra para não selecionar um arquivo de saída de uma execução anterior
    arquivos_csv = [f for f in arquivos_csv if not f.endswith('_notion.csv')]

    if not arquivos_csv:
        print("ERRO: Nenhum arquivo .csv de entrada foi encontrado neste diretório.")
        print("Por favor, adicione um arquivo CSV na mesma pasta do script e tente novamente.")
        return

    # Usa o primeiro arquivo CSV encontrado como entrada
    input_file = arquivos_csv[0]
    
    if len(arquivos_csv) > 1:
        print(f"AVISO: Múltiplos arquivos CSV encontrados. Usando o primeiro: '{input_file}'")

    # Gera o nome do arquivo de saída a partir do nome do arquivo de entrada
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_notion.csv"
    
    # Chama a função de processamento
    processar_csv(input_file, output_file, COLUNAS_DESEJADAS, verbose=MODO_VERBOSE)

if __name__ == "__main__":
    main()