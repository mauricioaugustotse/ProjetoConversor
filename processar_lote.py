#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
from pathlib import Path

def extrair_data(texto):
    """Extrai data em formatos dd/mm/yyyy ou dd/mm/yy"""
    # Tenta encontrar datas
    patterns = [
        r'(\d{2})/(\d{2})/(\d{4})',  # dd/mm/yyyy
        r'(\d{2})/(\d{2})/(\d{2})',  # dd/mm/yy
        r'(\d{4})-(\d{2})-(\d{2})',  # yyyy-mm-dd
    ]

    for pattern in patterns:
        match = re.search(pattern, texto)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if pattern == r'(\d{2})/(\d{2})/(\d{4})':
                    day, month, year = groups
                    return f"{year}-{month}-{day}"
                elif pattern == r'(\d{2})/(\d{2})/(\d{2})':
                    day, month, year = groups
                    # Ajustar século
                    year_int = int(year)
                    full_year = 2000 + year_int if year_int < 50 else 1900 + year_int
                    return f"{full_year}-{month}-{day}"
                elif pattern == r'(\d{4})-(\d{2})-(\d{2})':
                    return f"{groups[0]}-{groups[1]}-{groups[2]}"
    return None

def extrair_numero_documento(texto):
    """Extrai números de CPF, CNPJ, processo, etc."""
    # CPF: 025.332.581-13
    cpf_match = re.search(r'\d{3}\.\d{3}\.\d{3}-\d{2}', texto)
    if cpf_match:
        return cpf_match.group(0)

    # Processo judicial: números de 7+ dígitos com pontos
    # Nota fiscal, boleto: números soltos de 10+ dígitos
    return None

def gerar_nome_novo(id_num, pasta, nome_atual, trecho):
    """Gera nome novo baseado em contexto e conteúdo"""

    # Remover extensão
    nome_base = os.path.splitext(nome_atual)[0]

    # Se o nome já é descritivo (não é genérico), manter
    genericos = ['Doc', 'File', 'Documento', 'Arquivo', 'Sem título', 'Untitled',
                 'Anexo', 'Email', 'E-mail', 'Despacho', 'Boleto']

    # Se nome atual é razoável, apenas melhorar levemente
    if not any(g.lower() in nome_base.lower() for g in genericos[:5]):
        # Tentar extrair data e adicionar
        data = extrair_data(trecho)
        if data:
            return f"{nome_base} ({data})"
        return nome_base

    # Se não, tentar extrair informações do trecho
    trecho_limpo = trecho[:200].strip()  # Primeiros 200 caracteres

    # Extrair data
    data = extrair_data(trecho)

    # Procurar por padrões comuns
    # Declaração, Comprovante, Relatório, Parecer, Nota Fiscal, etc.

    # Declaração
    if 'declaração' in trecho_limpo.lower() or 'declaramos' in trecho_limpo.lower():
        if 'irpf' in pasta.lower() or 'irpf' in trecho_limpo.lower():
            tipo = "Declaração IRPF"
        elif 'tse' in trecho_limpo.lower() or 'servidor' in trecho_limpo.lower():
            tipo = "Declaração - servidor"
        else:
            tipo = "Declaração"
        if data:
            return f"{tipo} ({data})"
        return tipo

    # Comprovante
    if 'comprovante' in trecho_limpo.lower() or 'pagamento' in trecho_limpo.lower():
        if 'boleto' in trecho_limpo.lower() or 'pgmto' in nome_base.lower():
            tipo = "Comprovante de pagamento"
        else:
            tipo = "Comprovante"
        if data:
            return f"{tipo} ({data})"
        return tipo

    # Relatório
    if 'relatório' in trecho_limpo.lower() or 'relatorio' in trecho_limpo.lower():
        tipo = "Relatório"
        if 'investimento' in trecho_limpo.lower():
            tipo = "Relatório - investimento"
        elif 'atividade' in trecho_limpo.lower():
            tipo = "Relatório - atividade"
        if data:
            return f"{tipo} ({data})"
        return tipo

    # Nota Fiscal
    if 'nota fiscal' in trecho_limpo.lower() or 'nfe' in trecho_limpo.lower():
        tipo = "Nota Fiscal"
        if data:
            return f"{tipo} ({data})"
        return tipo

    # Parecer
    if 'parecer' in trecho_limpo.lower():
        tipo = "Parecer"
        if data:
            return f"{tipo} ({data})"
        return tipo

    # Fatura
    if 'fatura' in trecho_limpo.lower():
        tipo = "Fatura"
        if data:
            return f"{tipo} ({data})"
        return tipo

    # Edital
    if 'edital' in trecho_limpo.lower():
        tipo = "Edital"
        if data:
            return f"{tipo} ({data})"
        return tipo

    # Recurso, Petição, Ação
    if 'recurso' in trecho_limpo.lower():
        return "Recurso"
    if 'petição' in trecho_limpo.lower() or 'peticao' in trecho_limpo.lower():
        return "Petição"

    # Se nada funcionar, retornar nome atual
    if data:
        return f"{nome_base} ({data})"
    return nome_base

# Processar o arquivo
input_file = r"C:\Users\mauri\ProjetoConversor\lotes_ren\lote_005.txt"
output_file = r"C:\Users\mauri\ProjetoConversor\lotes_ren_out\res_005.csv"

resultados = ["id|nomeNovo"]

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) == 4:
            id_num = parts[0].strip()
            pasta = parts[1].strip()
            nome_atual = parts[2].strip()
            trecho = parts[3].strip()

            nome_novo = gerar_nome_novo(id_num, pasta, nome_atual, trecho)
            resultados.append(f"{id_num}|{nome_novo}")

# Salvar resultado
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(resultados))

print(f"Processadas {len(resultados)-1} linhas")
print(f"Arquivo salvo em: {output_file}")
print("\nPrimeiras 10 linhas do resultado:")
for linha in resultados[:10]:
    print(linha)
