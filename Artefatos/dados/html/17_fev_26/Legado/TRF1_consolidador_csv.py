# -*- coding: utf-8 -*-

"""
Script para processar múltiplos arquivos CSV, adicionar colunas e consolidá-los.

FUNCIONALIDADES:
1.  Busca todos os arquivos .csv no mesmo diretório onde o script está localizado.
2.  Para cada arquivo, adiciona duas novas colunas:
    - 'tribunal': Preenchida com o valor fixo "TRF1".
    - 'informativo': Preenchida com o número extraído do final do nome do arquivo
      (ex: para '11052-Bij_715.csv', o valor será '715').
3.  Consolida os dados de todos os arquivos processados em um único arquivo de saída
    chamado 'consolidado.csv'.
"""

import os
import pandas as pd
import re

def processar_e_consolidar_csvs():
    """
    Função principal que executa todo o processo de leitura, modificação
    e consolidação dos arquivos CSV.
    """
    # Pega o caminho do diretório onde o script está sendo executado.
    diretorio_atual = os.getcwd()
    print(f"Executando no diretório: {diretorio_atual}\n")

    # Lista para armazenar os DataFrames de cada arquivo processado.
    lista_de_dataframes = []

    # Lista todos os arquivos no diretório.
    arquivos_no_diretorio = os.listdir(diretorio_atual)

    # Filtra para pegar apenas os arquivos CSV que não sejam o arquivo de saída.
    arquivos_csv_para_processar = [
        arquivo for arquivo in arquivos_no_diretorio
        if arquivo.lower().endswith('.csv') and arquivo.lower() != 'consolidado.csv'
    ]

    if not arquivos_csv_para_processar:
        print("Nenhum arquivo .csv encontrado para processar neste diretório.")
        return

    print(f"Arquivos .csv encontrados para processar: {len(arquivos_csv_para_processar)}")
    for arquivo in arquivos_csv_para_processar:
        print(f" - {arquivo}")
    print("-" * 30)

    # Itera sobre cada arquivo CSV encontrado.
    for nome_arquivo in arquivos_csv_para_processar:
        try:
            # Tenta extrair o número do informativo do nome do arquivo.
            # A expressão regular `_(\d+)\.csv` busca por um underscore, seguido por um ou mais dígitos,
            # e que termina com '.csv'.
            match = re.search(r'_(\d+)\.csv$', nome_arquivo, re.IGNORECASE)

            if not match:
                print(f"AVISO: O arquivo '{nome_arquivo}' não segue o padrão esperado (ex: 'nome_NUMERO.csv') e será ignorado.")
                continue

            numero_informativo = match.group(1)
            caminho_completo = os.path.join(diretorio_atual, nome_arquivo)

            # Lê o arquivo CSV para um DataFrame do pandas.
            df = pd.read_csv(caminho_completo)

            # Adiciona as duas novas colunas.
            df['tribunal'] = 'TRF1'
            df['informativo'] = numero_informativo

            # Adiciona o DataFrame modificado à lista.
            lista_de_dataframes.append(df)
            print(f"Processado: '{nome_arquivo}' | Informativo: {numero_informativo}")

        except Exception as e:
            print(f"ERRO ao processar o arquivo '{nome_arquivo}': {e}")

    # Verifica se algum arquivo foi processado com sucesso.
    if not lista_de_dataframes:
        print("\nNenhum arquivo foi processado com sucesso. O arquivo consolidado não será gerado.")
        return

    # Concatena todos os DataFrames da lista em um único DataFrame.
    df_consolidado = pd.concat(lista_de_dataframes, ignore_index=True)

    # Define o nome do arquivo de saída.
    arquivo_saida = 'consolidado.csv'

    # Salva o DataFrame consolidado em um novo arquivo CSV.
    # `index=False` evita que o índice do DataFrame seja salvo como uma coluna.
    # `encoding='utf-8-sig'` garante compatibilidade com Excel e caracteres especiais.
    df_consolidado.to_csv(arquivo_saida, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"\nProcesso finalizado com sucesso!")
    print(f"O arquivo consolidado foi salvo como: '{arquivo_saida}'")
    print(f"Total de linhas no arquivo final: {len(df_consolidado)}")


if __name__ == '__main__':
    # Garante que a função principal só será executada quando o script for rodado diretamente.
    processar_e_consolidar_csvs()