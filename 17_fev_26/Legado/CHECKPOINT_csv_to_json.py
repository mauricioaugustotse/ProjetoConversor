# -*- coding: utf-8 -*-
"""
Ferramenta para criar um arquivo de checkpoint (.json) a partir de um CSV existente.

Objetivo:
Este script foi projetado para "resgatar" o progresso de um arquivo CSV gerado
por uma versão anterior do processador de julgados (que não usava checkpoints).
Ele lê um CSV, compara seu conteúdo com o arquivo .txt original e gera um
arquivo .checkpoint.json que permite ao script principal (v9+) retomar o
processamento do ponto onde parou.

v3: Muda drasticamente a lógica para ser mais precisa. Em vez de procurar
pelo primeiro "buraco", o script agora identifica o último julgado salvo no
CSV e localiza sua posição exata no arquivo .txt original. Isso garante que
o checkpoint seja criado no ponto correto, mesmo que haja falhas ou julgados
faltando no meio do CSV.

Como usar:
1. Salve este script no mesmo diretório onde estão seus arquivos .txt e .csv.
2. Execute o script no Visual Studio Code (apertando o botão "play").
3. O script listará os arquivos .csv disponíveis.
4. Digite o número correspondente ao arquivo .csv que você deseja converter.
5. O script criará o arquivo .checkpoint.json correspondente, pronto para uso.
"""

import os
import re
import json
import pandas as pd
import glob
import ast # Para converter strings de listas em listas reais

# IMPORTANTE: Esta seção deve ser IDÊNTICA à do script principal para garantir
# que a extração de julgados seja feita da mesma forma.
# -------------------- Regex de final de julgado ------------------------------
PADRAO_FINAL_JULGADO = re.compile(
    r"""
    \(
      \s*
      (?:Ac\.|Res\.|EDcl|Embargos?|AgR(?:-[A-Za-z]+)?|MS|REspE?|REspe|REsp|RO-?El|RMS|RCEd|AE|TutCautAnt|MC|AAg|Rec\.?)
      [\s\S]*?
      rel\.\s*(?:Min\.|Des\.|Juiz(?:a)?|in\.)[\s\S]*?
    \)
    \s*
    """,
    re.IGNORECASE | re.VERBOSE
)

def extrair_julgados(texto: str) -> list[str]:
    """Divide o texto em julgados individuais. (Cópia da função principal)"""
    julgados, ultimo_ponto_corte = [], 0
    for match in PADRAO_FINAL_JULGADO.finditer(texto):
        ponto_corte_atual = match.end()
        julgado_completo = texto[ultimo_ponto_corte:ponto_corte_atual].strip()
        if julgado_completo:
            julgados.append(julgado_completo)
        ultimo_ponto_corte = ponto_corte_atual
    return [j for j in julgados if j]
# -----------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Remove todos os caracteres de espaço em branco (espaços, tabs, newlines) para uma comparação robusta."""
    if not isinstance(text, str):
        return ""
    # re.sub() substitui um ou mais caracteres de espaço (\s+) por uma string vazia ('')
    return re.sub(r'\s+', '', text)


def criar_checkpoint_a_partir_de_csv():
    """Função principal que orquestra a conversão."""
    print("--- Ferramenta de Criação de Checkpoint a partir de CSV (v3) ---")
    
    # 1. Encontrar e listar arquivos CSV disponíveis
    arquivos_csv = glob.glob("*.csv")
    if not arquivos_csv:
        print("\nAVISO: Nenhum arquivo .csv encontrado neste diretório.")
        input("Pressione Enter para sair.")
        return

    print("\nArquivos .csv encontrados:")
    for i, nome_arquivo in enumerate(arquivos_csv):
        print(f"  [{i + 1}] {nome_arquivo}")

    # 2. Obter a escolha do usuário
    escolha = -1
    while escolha < 1 or escolha > len(arquivos_csv):
        try:
            escolha_str = input(f"\nDigite o número do arquivo que deseja converter (1 a {len(arquivos_csv)}): ")
            escolha = int(escolha_str)
        except (ValueError, IndexError):
            print("Entrada inválida. Por favor, digite um número da lista.")

    csv_selecionado = arquivos_csv[escolha - 1]
    base_nome = os.path.splitext(csv_selecionado)[0]
    arquivo_txt_correspondente = base_nome + ".txt"
    arquivo_checkpoint_saida = arquivo_txt_correspondente + ".checkpoint.json"

    print(f"\n-> Arquivo selecionado: '{csv_selecionado}'")

    # 3. Validar a existência dos arquivos necessários
    if not os.path.exists(arquivo_txt_correspondente):
        print(f"\nERRO: O arquivo de texto original '{arquivo_txt_correspondente}' não foi encontrado.")
        input("Pressione Enter para sair.")
        return
        
    if os.path.exists(arquivo_checkpoint_saida):
        print(f"\nAVISO: Um arquivo de checkpoint '{arquivo_checkpoint_saida}' já existe.")
        confirmacao = input("Deseja sobrescrevê-lo? (s/n): ").lower()
        if confirmacao != 's':
            print("Operação cancelada.")
            return

    try:
        # 4. Ler o arquivo .txt e extrair todos os julgados
        print(f"\nAnalisando o arquivo de origem '{arquivo_txt_correspondente}'...")
        with open(arquivo_txt_correspondente, 'r', encoding='utf-8') as f:
            conteudo_txt = f.read()
        julgados_originais = extrair_julgados(conteudo_txt)
        print(f"Total de {len(julgados_originais)} julgados encontrados no arquivo .txt.")

        # 5. Ler o arquivo .csv e mapear o progresso
        print(f"Lendo o progresso do arquivo '{csv_selecionado}'...")
        df = pd.read_csv(csv_selecionado)

        if 'texto_original' not in df.columns or df['texto_original'].dropna().empty:
            print("\nERRO: A coluna 'texto_original' está ausente ou vazia no CSV.")
            input("Pressione Enter para sair.")
            return
            
        # --- LÓGICA DE COMPARAÇÃO CORRIGIDA (v3) ---
        # Pega o texto do ÚLTIMO julgado que foi processado e salvo no CSV.
        ultimo_texto_processado_no_csv = df['texto_original'].dropna().iloc[-1]
        ultimo_texto_normalizado = _normalize_text(ultimo_texto_processado_no_csv)
        
        # Agora, busca a posição (índice) exata desse último julgado na lista original do TXT.
        ultimo_indice_encontrado = -1
        for i, julgado_original in enumerate(julgados_originais):
            if _normalize_text(julgado_original) == ultimo_texto_normalizado:
                ultimo_indice_encontrado = i
                break # Achou a correspondência, pode parar a busca.

        if ultimo_indice_encontrado == -1:
            print("\nERRO CRÍTICO: O último julgado encontrado no CSV não pôde ser localizado no arquivo .txt original.")
            print("Isso pode acontecer se os arquivos não corresponderem. Verifique os arquivos.")
            input("Pressione Enter para sair.")
            return

        total_processado = ultimo_indice_encontrado + 1
        print(f"Progresso real encontrado: {total_processado} de {len(julgados_originais)} julgados já foram processados.")

        # 6. Preparar os dados para o formato JSON
        registros_salvos = df.to_dict('records')
        for registro in registros_salvos:
            if 'subramo' in registro and isinstance(registro['subramo'], str):
                try:
                    registro['subramo'] = ast.literal_eval(registro['subramo'])
                except (ValueError, SyntaxError):
                    # Mantém como string se a conversão falhar
                    pass
        
        checkpoint_data = {
            "ultimo_julgado_processado_idx": ultimo_indice_encontrado,
            "registros_salvos": registros_salvos
        }

        # 7. Escrever o arquivo de checkpoint
        with open(arquivo_checkpoint_saida, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)

        print(f"\nSUCESSO! O arquivo de checkpoint '{arquivo_checkpoint_saida}' foi criado.")
        print(f"O script principal irá retomar o trabalho a partir do julgado #{total_processado + 1}.")

    except Exception as e:
        print(f"\nERRO: Ocorreu um erro inesperado: {e}")
    
    input("\nPressione Enter para finalizar.")


if __name__ == "__main__":
    criar_checkpoint_a_partir_de_csv()