#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os

def extrair_data(texto):
    """Extrai data em formatos dd/mm/yyyy ou dd/mm/yy ou yyyy-mm-dd"""
    patterns = [
        (r'(\d{2})/(\d{2})/(\d{4})', 'dmy4'),   # dd/mm/yyyy
        (r'(\d{2})/(\d{2})/(\d{2})', 'dmy2'),   # dd/mm/yy
        (r'(\d{4})-(\d{2})-(\d{2})', 'ymd'),    # yyyy-mm-dd
        (r'(\d{2})/(\d{1,2})', 'md'),           # dd/mm sem ano
    ]

    for pattern, tipo in patterns:
        match = re.search(pattern, texto)
        if match:
            groups = match.groups()
            if tipo == 'dmy4':
                return f"{groups[2]}-{groups[1]}-{groups[0]}"
            elif tipo == 'dmy2':
                year = int(groups[2])
                full_year = 2000 + year if year < 50 else 1900 + year
                return f"{full_year}-{groups[1]}-{groups[0]}"
            elif tipo == 'ymd':
                return f"{groups[0]}-{groups[1]}-{groups[2]}"
            elif tipo == 'md':
                return f"{groups[0]}-{groups[1]}"
    return None

def extrair_numero_nf(texto):
    """Extrai número de nota fiscal"""
    # Procura por padrões como "NFe XXX" ou "NF-e número"
    match = re.search(r'(?:NF[e-]*\s*)?n[úu]mero\s+(\d+)', texto, re.IGNORECASE)
    if match:
        return f"nota fiscal {match.group(1)}"

    # Procura por documento padrão
    match = re.search(r'(?:documento|número do documento)[:\s]+(\d+)', texto, re.IGNORECASE)
    if match:
        return f"documento {match.group(1)}"
    return None

def limpar_nome(nome):
    """Remove extensão e caracteres especiais problemáticos"""
    nome = os.path.splitext(nome)[0]
    nome = nome.strip()
    return nome

def gerar_nome_inteligente(id_num, pasta, nome_atual, trecho):
    """Gera nome descritivo baseado em pasta + nome + conteúdo"""

    nome_base = limpar_nome(nome_atual)
    trecho_limpo = trecho[:300].strip()

    # Alguns nomes já são muito bons - manter com pequenas correções
    bons_nomes = [
        'Relatório', 'Declaração', 'Comprovante', 'Procuração', 'Contrato',
        'Parecer', 'Edital', 'Sentença', 'Acórdão', 'Resolução',
        'Termo', 'Ata', 'Nota Fiscal', 'Fatura', 'Boleto', 'Guia'
    ]

    # Se já é bom nome, extrair data e retornar
    if any(bom in nome_base for bom in bons_nomes):
        data = extrair_data(trecho_limpo)
        if data and data not in nome_base:
            return f"{nome_base} ({data})"
        return nome_base

    # LOGICA POR PASTA
    pasta_lower = pasta.lower()

    # ===== JURÍDICO E TRABALHO =====
    if 'jurídico' in pasta_lower or 'trabalho' in pasta_lower or 'conle' in pasta_lower:
        if 'procuração' in trecho_limpo.lower():
            return f"Procuração - {nome_base[:30]}"
        if 'parecer' in trecho_limpo.lower():
            data = extrair_data(trecho_limpo)
            return f"Parecer {data if data else ''}"
        if 'despacho' in trecho_limpo.lower():
            data = extrair_data(trecho_limpo)
            return f"Despacho {data if data else ''}"
        if 'sentença' in trecho_limpo.lower():
            return f"Sentença"
        if 'recurso' in trecho_limpo.lower():
            return f"Recurso"
        if 'edital' in trecho_limpo.lower():
            return f"Edital"
        if 'resolução' in trecho_limpo.lower():
            return f"Resolução"

    # ===== ESTUDOS, CONCURSOS =====
    if 'estudos' in pasta_lower or 'concursos' in pasta_lower or 'tese' in pasta_lower:
        if 'prova' in trecho_limpo.lower():
            # Extrair ano ou tipo de prova
            match = re.search(r'(\d{4})', trecho_limpo)
            ano = match.group(1) if match else None
            return f"Prova {ano if ano else ''}"
        if 'edital' in trecho_limpo.lower():
            match = re.search(r'edital\s+(?:n[º°]?\s*)?(\d+)', trecho_limpo, re.IGNORECASE)
            num = match.group(1) if match else ""
            return f"Edital {num}"
        if 'conteúdo' in trecho_limpo.lower() or 'programático' in trecho_limpo.lower():
            return "Conteúdo programático"
        if 'relatório' in trecho_limpo.lower():
            if 'atividade' in trecho_limpo.lower():
                return "Relatório - atividade didática"
            data = extrair_data(trecho_limpo)
            return f"Relatório {data if data else ''}"

    # ===== FINANCEIRO =====
    if 'financeiro' in pasta_lower:
        if 'irpf' in pasta_lower:
            match = re.search(r'(\d{4})', trecho_limpo)
            ano = match.group(1) if match else None
            return f"Declaração IRPF {ano if ano else ''}"
        if 'fatura' in trecho_limpo.lower():
            data = extrair_data(trecho_limpo)
            return f"Fatura {data if data else ''}"
        if 'boleto' in trecho_limpo.lower() or 'boleto' in nome_base.lower():
            data = extrair_data(trecho_limpo)
            return f"Boleto {data if data else ''}"
        if 'conta' in pasta_lower or 'contas' in pasta_lower:
            if 'relatório' in trecho_limpo.lower():
                match = re.search(r'(\w+)/(\d{4})', trecho_limpo)
                periodo = f"{match.group(1)}/{match.group(2)}" if match else None
                return f"Contas {periodo if periodo else ''}"
            data = extrair_data(trecho_limpo)
            return f"Contas {data if data else ''}"
        if 'pgmto' in nome_base.lower() or 'pagamento' in trecho_limpo.lower():
            data = extrair_data(trecho_limpo)
            return f"Pagamento {data if data else ''}"

    # ===== IMÓVEIS =====
    if 'imóvel' in pasta_lower or 'apto' in pasta_lower or 'imobiliário' in pasta_lower:
        if 'vistoria' in trecho_limpo.lower():
            data = extrair_data(trecho_limpo)
            return f"Vistoria {data if data else ''}"
        if 'aluguel' in trecho_limpo.lower() or 'pgmto' in nome_base.lower():
            data = extrair_data(trecho_limpo)
            return f"Aluguel {data if data else ''}"
        if 'iptu' in trecho_limpo.lower() or 'iptu' in pasta_lower:
            data = extrair_data(trecho_limpo)
            return f"IPTU {data if data else ''}"
        if 'condomínio' in trecho_limpo.lower() or 'condominio' in pasta_lower:
            data = extrair_data(trecho_limpo)
            return f"Condomínio {data if data else ''}"

    # ===== VEÍCULOS =====
    if 'veículo' in pasta_lower or 'cruze' in pasta_lower:
        if 'ipva' in trecho_limpo.lower() or 'ipva' in pasta_lower:
            match = re.search(r'(\d{4})', trecho_limpo)
            ano = match.group(1) if match else None
            return f"IPVA {ano if ano else ''}"
        if 'licenciamento' in trecho_limpo.lower():
            match = re.search(r'(\d{4})', trecho_limpo)
            ano = match.group(1) if match else None
            return f"Licenciamento {ano if ano else ''}"
        if 'parcela' in nome_base.lower():
            match = re.search(r'(\d+)[ªaº]', nome_base)
            num = match.group(1) if match else ""
            data = extrair_data(trecho_limpo)
            return f"Parcela {num} ({data if data else ''})"

    # ===== SAÚDE =====
    if 'saúde' in pasta_lower:
        if 'receita' in trecho_limpo.lower():
            data = extrair_data(trecho_limpo)
            return f"Receita {data if data else ''}"
        if 'laudo' in trecho_limpo.lower():
            data = extrair_data(trecho_limpo)
            return f"Laudo {data if data else ''}"

    # PADRÃO FINAL: Se tudo falhar, manter nome original com data
    data = extrair_data(trecho_limpo)
    if data and data not in nome_base:
        return f"{nome_base} ({data})"
    return nome_base

# Processar arquivo
input_file = r"C:\Users\mauri\ProjetoConversor\lotes_ren\lote_005.txt"
output_file = r"C:\Users\mauri\ProjetoConversor\lotes_ren_out\res_005.csv"

resultados = ["id|nomeNovo"]
count = 0

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) == 4:
            id_num = parts[0].strip()
            pasta = parts[1].strip()
            nome_atual = parts[2].strip()
            trecho = parts[3].strip()

            nome_novo = gerar_nome_inteligente(id_num, pasta, nome_atual, trecho)
            resultados.append(f"{id_num}|{nome_novo}")
            count += 1

# Salvar com encoding UTF-8 sem BOM
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(resultados))

print(f"OK - Processadas {count} linhas")
print(f"OK - Arquivo: {output_file}")
