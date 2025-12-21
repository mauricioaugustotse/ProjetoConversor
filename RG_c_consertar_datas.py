import csv
import os

# --- Configurações ---
# Coloque o nome do seu arquivo CSV aqui
NOME_ARQUIVO_ENTRADA = 'entrada.csv'

# O nome do arquivo que será gerado
NOME_ARQUIVO_SAIDA = 'saida_formatada.csv'

# (NOVA CONFIGURAÇÃO)
# Coloque os índices das colunas que você quer formatar.
# Ex: Para processar "C" (índice 2) e "O" (índice 14)
INDICES_COLUNAS_ALVO = [2, 14] 
# ---------------------

# Mapeamento de meses em português para o formato numérico MM
meses_map = {
    'janeiro': '01',
    'fevereiro': '02',
    'março': '03',
    'abril': '04',
    'maio': '05',
    'junho': '06',
    'julho': '07',
    'agosto': '08',
    'setembro': '09',
    'outubro': '10',
    'novembro': '11',
    'dezembro': '12'
}

def converter_data_pt_para_us(data_pt):
    """
    Converte uma data do formato 'DD de MesNome de AAAA' para 'MM/DD/AAAA'.
    Mantém strings vazias e formatações inesperadas.
    """
    # 1. Respeita as strings vazias (Constraint #2)
    if not data_pt.strip():
        return data_pt

    try:
        # 2. Divide a data: ['DD', 'MesNome', 'AAAA']
        partes = data_pt.split(' de ')
        
        # Se não for o formato esperado (ex: o cabeçalho "Trânsito em Julgado"),
        # retorna o valor original.
        if len(partes) != 3:
            return data_pt
        
        dia = partes[0]
        mes_nome = partes[1].lower() # .lower() para garantir a correspondência
        ano = partes[2]
        
        # 3. Busca o número do mês no mapeamento
        if mes_nome not in meses_map:
            # Se o mês não for encontrado, retorna o original
            return data_pt
            
        mes_num = meses_map[mes_nome]
        
        # 4. Garante que o dia tenha dois dígitos (ex: '5' -> '05')
        dia_formatado = dia.zfill(2)
        
        # 5. Retorna no formato americano MM/DD/AAAA
        return f"{mes_num}/{dia_formatado}/{ano}"

    except Exception as e:
        # Em caso de qualquer erro, retorna o valor original para segurança
        print(f"Erro ao processar a data '{data_pt}': {e}. Mantendo original.")
        return data_pt

# --- Lógica Principal ---

def processar_csv():
    # Verifica se o arquivo de entrada existe
    if not os.path.exists(NOME_ARQUIVO_ENTRADA):
        print(f"Erro: Arquivo '{NOME_ARQUIVO_ENTRADA}' não encontrado.")
        print("Por favor, verifique se o arquivo está na mesma pasta que o script.")
        return

    # Lista para armazenar todas as linhas processadas
    dados_processados = []

    print(f"Lendo o arquivo '{NOME_ARQUIVO_ENTRADA}'...")

    try:
        # 'newline=""' é a recomendação oficial da biblioteca 'csv'
        with open(NOME_ARQUIVO_ENTRADA, mode='r', encoding='utf-8', newline='') as f_in:
            reader = csv.reader(f_in)
            
            for linha in reader:
                # (LÓGICA ATUALIZADA)
                # Itera sobre cada índice que definimos na configuração
                for indice in INDICES_COLUNAS_ALVO:
                    # Proteção para linhas que não têm colunas suficientes
                    if len(linha) > indice:
                        # Pega o valor original da coluna alvo
                        data_original = linha[indice]
                        
                        # Converte o valor
                        data_convertida = converter_data_pt_para_us(data_original)
                        
                        # Atualiza o valor na linha
                        linha[indice] = data_convertida
                
                # Adiciona a linha (modificada ou não) à nossa lista
                dados_processados.append(linha)

        # Escreve o novo arquivo CSV com os dados processados
        with open(NOME_ARQUIVO_SAIDA, mode='w', encoding='utf-8', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerows(dados_processados)
            
        print("-" * 30)
        print("Processamento concluído com sucesso!")
        print(f"Processando colunas (índices): {INDICES_COLUNAS_ALVO}")
        print(f"O novo arquivo foi salvo como: '{NOME_ARQUIVO_SAIDA}'")

    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o processamento: {e}")

# --- Executa o script ---
if __name__ == "__main__":
    processar_csv()