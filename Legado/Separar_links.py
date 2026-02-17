import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os

def processar_arquivo():
    # Configuração da janela oculta (necessário para o filedialog)
    root = tk.Tk()
    root.withdraw()

    # 1. Abrir caixa de seleção para o arquivo de INPUT
    caminho_arquivo = filedialog.askopenfilename(
        title="Selecione o arquivo CSV original",
        filetypes=[("Arquivos CSV", "*.csv")]
    )

    # Se o usuário cancelar a seleção, encerra o script
    if not caminho_arquivo:
        return

    try:
        # Carrega o CSV
        # 'error_bad_lines' ou 'on_bad_lines' depende da versão do pandas, 
        # mas o padrão geralmente funciona bem para arquivos bem formatados.
        df = pd.read_csv(caminho_arquivo)
        
        coluna_alvo = "noticia_geral"

        # Verifica se a coluna existe
        if coluna_alvo not in df.columns:
            messagebox.showerror("Erro", f"A coluna '{coluna_alvo}' não foi encontrada no arquivo.")
            return

        # 2. Lógica de separação
        # Separa por vírgula e expande em novas colunas
        df_split = df[coluna_alvo].str.split(',', expand=True)

        # Remove espaços em branco (strip) de todas as células resultantes
        df_split = df_split.apply(lambda x: x.str.strip())

        # Remove links repetidos em qualquer coluna, preservando a posicao original
        def remover_repetidos(linha):
            vistos = set()
            resultado = []
            for valor in linha:
                if pd.isna(valor) or valor == "":
                    resultado.append(pd.NA)
                    continue
                if valor in vistos:
                    resultado.append(pd.NA)
                else:
                    vistos.add(valor)
                    resultado.append(valor)
            return pd.Series(resultado, index=linha.index)

        df_split = df_split.apply(remover_repetidos, axis=1)

        # Renomeia as colunas (noticia_geral_1, noticia_geral_2...)
        novas_colunas = [f"{coluna_alvo}_{i+1}" for i in range(df_split.shape[1])]
        df_split.columns = novas_colunas

        # Junta o dataframe original (removendo a coluna antiga) com as novas colunas
        df_final = pd.concat([df.drop(columns=[coluna_alvo]), df_split], axis=1)

        # 3. Salvar no mesmo local do arquivo de entrada com o sufixo solicitado
        diretorio_entrada = os.path.dirname(caminho_arquivo)
        nome_base = os.path.splitext(os.path.basename(caminho_arquivo))[0]
        caminho_salvar = os.path.join(diretorio_entrada, f"{nome_base} - links separados.csv")

        df_final.to_csv(caminho_salvar, index=False)
        messagebox.showinfo("Sucesso", f"Arquivo processado e salvo em:\n{caminho_salvar}")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro durante o processamento:\n{str(e)}")

if __name__ == "__main__":
    processar_arquivo()
