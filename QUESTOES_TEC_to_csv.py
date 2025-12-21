# -*- coding: utf-8 -*-
"""
Este script processa arquivos de texto (.txt) contendo questões de concurso,
extrai informações estruturadas de cada questão e as salva em um único
arquivo CSV.
"""
import os
import re
import csv

# --- CONFIGURAÇÃO ---
# Nome da pasta onde os arquivos .txt de entrada devem ser colocados.
# O script irá criar esta pasta se ela não existir.
INPUT_DIR = 'arquivos_txt'

# Nome do arquivo CSV que será gerado na mesma pasta do script.
OUTPUT_FILE = 'questoes_compiladas.csv'

# Cabeçalho do arquivo CSV, definindo as colunas.
CSV_HEADER = [
    'questao_TEC', 'ano', 'banca', 'tribunal', 'gabarito',
    'ramo', 'subramo', 'questao'
]
# --- FIM DA CONFIGURAÇÃO ---

def parse_question_block(block_text):
    """
    Analisa um bloco de texto contendo uma única questão e extrai os dados.
    Retorna um dicionário com os dados ou None se o bloco for inválido.
    """
    data = {}
    # Divide o bloco em linhas e remove linhas vazias para facilitar o processamento.
    lines = [line.strip() for line in block_text.strip().split('\n') if line.strip()]
    if len(lines) < 4:
        return None  # Bloco muito curto para ser uma questão válida.

    # 1. Extrai o 'questao_TEC' (número da questão).
    # Ele aparece no início do bloco, vindo da URL que usamos como separador.
    match_id = re.match(r'(\d+)', lines[0])
    if not match_id:
        return None  # Não é um bloco de questão válido se não começar com um número.
    data['questao_TEC'] = match_id.group(1)

    # 2. Extrai 'banca', 'tribunal' e 'ano'.
    # Exemplo de linha: "FGV - JE TJMG/TJ MG/2022"
    info_line = lines[1]
    ano_match = re.search(r'(\d{4})$', info_line)
    ano = ano_match.group(1) if ano_match else ''
    
    banca_tribunal_part = info_line[:ano_match.start()].strip('/') if ano_match else info_line
    parts = banca_tribunal_part.split(' - ', 1)
    
    data['banca'] = parts[0].strip()
    # Trata casos com múltiplas barras no nome do tribunal (ex: TJMG/TJ MG)
    data['tribunal'] = parts[1].strip().replace('/', ' ') if len(parts) > 1 else ''
    data['ano'] = ano

    # 3. Extrai 'ramo' e 'subramo'.
    # Exemplo de linha: "Direito Civil - Direito Civil (Conceito, Objeto...)"
    subject_line = lines[2]
    subject_parts = subject_line.split(' - ', 1)
    if len(subject_parts) == 2:
        data['ramo'] = subject_parts[0].strip().lower()
        data['subramo'] = subject_parts[1].strip()
    else:
        data['ramo'] = subject_line.strip().lower()
        data['subramo'] = ''

    # 4. Extrai 'questao' e 'gabarito'.
    # O texto da questão está entre a linha do subramo e a linha do gabarito.
    question_text_lines = []
    gabarito_found = False
    # Itera a partir da quarta linha (índice 3).
    for line in lines[3:]:
        # Verifica se a linha é a do gabarito.
        if line.lower().strip().startswith('gabarito:'):
            data['gabarito'] = line.split(':')[-1].strip()
            gabarito_found = True
            break  # Para a busca ao encontrar o gabarito.
        
        # Filtra elementos de navegação e rodapés indesejados que aparecem no meio do texto.
        if re.search(r'\d{2}/\d{2}/\d{4}.*Tec Concursos', line):
            continue
        if re.search(r'https://www.tecconcursos.com.br/questoes/cadernos', line):
            continue
        # Filtra marcadores de página como "9) "
        if re.match(r'^\d+\)\s*$', line.strip()):
            continue
            
        question_text_lines.append(line)

    if not gabarito_found:
        return None  # Se não encontrar gabarito, o bloco é considerado inválido.

    # Junta as linhas do texto da questão.
    data['questao'] = '\n'.join(question_text_lines).strip()

    return data


def main():
    """
    Função principal que orquestra a leitura dos arquivos .txt,
    o processamento dos dados e a escrita do arquivo .csv.
    """
    # Verifica se a pasta de entrada existe; se não, a cria e informa o usuário.
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Pasta '{INPUT_DIR}' criada.")
        print("Por favor, adicione seus arquivos .txt de questões nesta pasta e rode o script novamente.")
        return

    # Lista todos os arquivos .txt na pasta de entrada.
    try:
        txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    except OSError as e:
        print(f"Erro ao acessar a pasta '{INPUT_DIR}': {e}")
        return

    if not txt_files:
        print(f"Nenhum arquivo .txt encontrado na pasta '{INPUT_DIR}'.")
        print("Por favor, adicione seus arquivos e tente novamente.")
        return

    all_questions_data = []
    # O padrão separador agora usa uma expressão regular para ser mais preciso.
    # Ele garante que a URL da questão esteja no início de uma linha,
    # ignorando URLs parecidas que possam aparecer nos rodapés das páginas.
    separator_pattern = r'\nwww\.tecconcursos\.com\.br/questoes/'

    print("\nIniciando o processamento dos arquivos...")
    for filename in txt_files:
        filepath = os.path.join(INPUT_DIR, filename)
        print(f"  - Lendo '{filename}'...")

        try:
            # Tenta ler o arquivo com a codificação UTF-8, que é a mais comum.
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Se falhar, é um sinal de que o arquivo tem outra codificação.
            print(f"    -> Aviso: Falha ao ler '{filename}' como UTF-8. Tentando com 'latin-1'...")
            try:
                # Tenta novamente com a codificação 'latin-1', comum em sistemas Windows.
                with open(filepath, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                print(f"    -> Erro: Não foi possível ler o arquivo '{filename}' com nenhuma codificação: {e}")
                continue
        except Exception as e:
            print(f"    -> Erro ao ler o arquivo '{filename}': {e}")
            continue

        # Divide o conteúdo do arquivo em blocos de questões usando a expressão regular.
        question_blocks = re.split(separator_pattern, content)
        
        if len(question_blocks) < 2:
            print(f"    -> Aviso: O padrão separador não foi encontrado em '{filename}'. O arquivo pode estar em um formato inesperado.")
            continue
            
        # O primeiro item da lista [0] é o cabeçalho do arquivo, então o ignoramos com [1:].
        for block in question_blocks[1:]:
            parsed_data = parse_question_block(block)
            if parsed_data:
                all_questions_data.append(parsed_data)
            else:
                first_line = block.strip().split('\n')[0]
                print(f"    -> Aviso: Um bloco de questão em '{filename}' (iniciando com '{first_line[:50]}...') não pôde ser processado e foi ignorado.")

    if not all_questions_data:
        print("\nNenhuma questão válida foi extraída dos arquivos fornecidos.")
        return

    # Escreve os dados coletados no arquivo CSV.
    print(f"\nEscrevendo {len(all_questions_data)} questões no arquivo '{OUTPUT_FILE}'...")
    try:
        # 'utf-8-sig' garante a compatibilidade com Excel.
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
            writer.writeheader()
            writer.writerows(all_questions_data)
        print("\n--- Processo Concluído com Sucesso! ---")
        print(f"O arquivo '{OUTPUT_FILE}' foi gerado na mesma pasta deste script.")
    except IOError as e:
        print(f"\nERRO: Não foi possível escrever no arquivo '{OUTPUT_FILE}'.")
        print(f"Verifique se o arquivo não está aberto em outro programa ou se você tem permissão para escrever na pasta. Detalhe: {e}")


if __name__ == "__main__":
    # Esta parte garante que a função main() seja executada quando
    # o script é rodado diretamente (ex: com o botão "play" do VS Code).
    main()