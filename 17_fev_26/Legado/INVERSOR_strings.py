# -*- coding: utf-8 -*-

"""
Script para inverter a ordem das linhas de um arquivo CSV, preservando o cabeçalho.
Versão aprimorada com detecção automática de arquivos CSV.
"""

import csv
import os
import sys
import glob

def inverter_linhas_csv():
    """
    Função principal que orquestra a leitura, inversão e escrita dos dados do CSV.
    """
    print("--- Início do Script de Inversão de Linhas de CSV ---")
    print("Este script lê um arquivo CSV, inverte a ordem de suas linhas (mantendo o cabeçalho no topo) e salva em um novo arquivo.")
    print("-" * 60)

    try:
        # --- Passo 1: Detectar arquivos CSV automaticamente ---
        # Procura por arquivos .csv no mesmo diretório do script.
        # Isso evita problemas com o diretório de trabalho do terminal.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Procurando por arquivos .csv no diretório: {script_dir}\n")
        
        # Usa glob para encontrar todos os arquivos que terminam com .csv
        csv_files = glob.glob(os.path.join(script_dir, "*.csv"))
        
        # Filtra para não incluir arquivos que já são o resultado de uma inversão anterior.
        csv_files = [f for f in csv_files if not f.endswith('_invertido.csv')]

        input_filename = ""

        if not csv_files:
            # Caso nenhum arquivo CSV seja encontrado
            print("[ERRO] Nenhum arquivo .csv foi encontrado na pasta deste script.")
            print("Por favor, adicione um arquivo CSV e tente novamente.")
            print("-" * 60)
            sys.exit()
        elif len(csv_files) == 1:
            # Caso apenas um arquivo CSV seja encontrado, ele é selecionado automaticamente.
            input_filename = csv_files[0]
            print(f"[INFO] Apenas um arquivo CSV foi detectado e selecionado automaticamente: '{os.path.basename(input_filename)}'")
        else:
            # Caso múltiplos arquivos CSV sejam encontrados, pede ao usuário para escolher.
            print("[INFO] Múltiplos arquivos .csv foram detectados. Por favor, escolha um:")
            for i, filename in enumerate(csv_files):
                print(f"  {i + 1}: {os.path.basename(filename)}")
            
            while True:
                try:
                    choice = int(input("\nDigite o número do arquivo que deseja processar: "))
                    if 1 <= choice <= len(csv_files):
                        input_filename = csv_files[choice - 1]
                        break
                    else:
                        print("[ERRO] Escolha inválida. Por favor, digite um número da lista.")
                except ValueError:
                    print("[ERRO] Entrada inválida. Por favor, digite apenas o número.")

    except Exception as e:
        print(f"\n[ERRO INESPERADO] Ocorreu um problema ao procurar arquivos: {e}")
        sys.exit()


    print(f"\n[SUCESSO] Arquivo a ser processado: '{os.path.basename(input_filename)}'.")

    try:
        # --- Passo 2: Ler o conteúdo do arquivo CSV ---
        print("\nIniciando a leitura do arquivo CSV...")
        
        with open(input_filename, mode='r', newline='', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            print(f"Cabeçalho identificado: {header}")

            data_rows = list(reader)
            num_rows = len(data_rows)
            print(f"Leitura concluída. Foram encontradas {num_rows} linhas de dados.")

            if num_rows == 0:
                print("[AVISO] O arquivo contém apenas o cabeçalho. O arquivo de saída terá apenas o cabeçalho.")

    except Exception as e:
        print(f"\n[ERRO] Ocorreu um problema ao ler o arquivo: {e}")
        print("-" * 60)
        sys.exit()

    # --- Passo 3: Inverter a ordem das linhas de dados ---
    print("\nInvertendo a ordem das linhas de dados...")
    reversed_data_rows = data_rows[::-1]
    print("A ordem das linhas foi invertida com sucesso.")

    # --- Passo 4: Definir o nome do arquivo de saída ---
    base_name, extension = os.path.splitext(input_filename)
    output_filename = f"{base_name}_invertido{extension}"
    print(f"O resultado será salvo no arquivo: '{os.path.basename(output_filename)}'")

    try:
        # --- Passo 5: Escrever os dados no novo arquivo CSV ---
        print("\nIniciando a escrita do novo arquivo CSV...")
        
        with open(output_filename, mode='w', newline='', encoding='utf-8-sig') as outfile:
            writer = csv.writer(outfile)
            print("Escrevendo o cabeçalho...")
            writer.writerow(header)
            print(f"Escrevendo as {len(reversed_data_rows)} linhas de dados invertidas...")
            writer.writerows(reversed_data_rows)

    except Exception as e:
        print(f"\n[ERRO] Ocorreu um problema ao escrever o arquivo de saída: {e}")
        print("-" * 60)
        sys.exit()

    # --- Conclusão ---
    full_path = os.path.abspath(output_filename)
    print("\n[PROCESSO CONCLUÍDO COM SUCESSO!]")
    print(f"O arquivo invertido foi salvo em: {full_path}")
    print("-" * 60)

if __name__ == "__main__":
    inverter_linhas_csv()