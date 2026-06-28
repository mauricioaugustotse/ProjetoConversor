#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os

def extrair_data_completa(texto):
    """Extrai apenas datas completas (dd/mm/yyyy ou yyyy-mm-dd)"""
    patterns = [
        r'(\d{2})/(\d{2})/(\d{4})',   # dd/mm/yyyy
        r'(\d{4})-(\d{2})-(\d{2})',    # yyyy-mm-dd
    ]

    for pattern in patterns:
        match = re.search(pattern, texto)
        if match:
            groups = match.groups()
            if pattern == r'(\d{2})/(\d{2})/(\d{4})':
                return f"{groups[2]}-{groups[1]}-{groups[0]}"
            else:
                return f"{groups[0]}-{groups[1]}-{groups[2]}"
    return None

def extrair_ano(texto):
    """Extrai apenas o ano (4 dígitos)"""
    match = re.search(r'\b(19|20)\d{2}\b', texto)
    return match.group(0) if match else None

def limpar_nome(nome):
    """Remove extensão e espaços"""
    nome = os.path.splitext(nome)[0].strip()
    return nome

def gerar_nome_novo(id_num, pasta, nome_atual, trecho):
    """Gera nome descritivo inteligente"""

    nome_base = limpar_nome(nome_atual)
    trecho_lower = trecho[:400].lower()
    pasta_lower = pasta.lower()

    # Se nome já é bom e descritivo
    if len(nome_base) > 20 and not any(x in nome_base.lower() for x in ['doc', 'arquivo', 'untitled', 'sem título']):
        # Tentar adicionar data se houver
        data = extrair_data_completa(trecho)
        if data:
            return f"{nome_base} ({data})"
        return nome_base

    # ESTRATÉGIA POR CONTEXTO DA PASTA

    # === JURÍDICO ===
    if 'jurídico' in pasta_lower or 'tse' in pasta_lower or 'conle' in pasta_lower:
        # Procuração
        if 'procuração' in trecho_lower:
            match = re.search(r'(?:outorgad[ao]|outorgante):\s+([^\n,]+)', trecho_lower)
            pessoa = match.group(1).strip()[:30] if match else "judicial"
            return f"Procuração - {pessoa}"

        # Despacho/Parecer
        if 'despacho' in trecho_lower or 'parecer' in trecho_lower:
            # Extrair referência SEI se houver
            match = re.search(r'sei[:/\s]+(\d{4}\.\d{2}\.\d{6}|\d+)', trecho_lower)
            ref = match.group(1) if match else None
            if ref:
                return f"Despacho {ref}"
            ano = extrair_ano(trecho)
            return f"Despacho {ano if ano else ''}"

        # Sentença/Acórdão
        if 'sentença' in trecho_lower:
            match = re.search(r'(?:número|nº|n°)?\s*(\d{4,})', trecho_lower)
            num = match.group(1) if match else None
            return f"Sentença {num if num else ''}"

        if 'acórdão' in trecho_lower or 'acordao' in trecho_lower:
            return f"Acórdão"

        # Edital/Resolução
        if 'edital' in trecho_lower:
            match = re.search(r'edital\s+(?:n[º°]?\s*)?(\d+/\d+|\d+)', trecho_lower)
            num = match.group(1) if match else None
            return f"Edital {num if num else ''}"

        if 'resolução' in trecho_lower or 'resolucao' in trecho_lower:
            match = re.search(r'res\.?-?tse\s+(\d+\.\d+/\d+|\d+)', trecho_lower)
            num = match.group(1) if match else None
            return f"Resolução {num if num else ''}"

        # Ação/Processo
        if 'ação' in trecho_lower or 'acao' in trecho_lower:
            match = re.search(r'(?:aije|acp|adi)\s+(\d+)', trecho_lower)
            tipo_num = match.group(0) if match else "Ação"
            return tipo_num

    # === ESTUDOS/CONCURSOS/TESE ===
    if 'estudos' in pasta_lower or 'concurso' in pasta_lower or 'tese' in pasta_lower:
        # Prova
        if 'prova' in trecho_lower or 'questão' in trecho_lower or 'questao' in trecho_lower:
            match = re.search(r'(20\d{2}|trf|oab|bndes|tjdft|mpdft|pgdf)', trecho_lower)
            tipo = match.group(1).upper() if match else None
            ano = extrair_ano(trecho)
            resultado = f"Prova {tipo or ''} {ano if ano else ''}".strip()
            # Limpar espaços múltiplos
            return ' '.join(resultado.split())

        # Edital
        if 'edital' in trecho_lower:
            match = re.search(r'edital\s+(?:n[º°]?\s*)?(\d+/\d+|\d+)', trecho_lower)
            num = match.group(1) if match else None
            return f"Edital {num if num else ''}"

        # Relatório
        if 'relatório' in trecho_lower or 'relatorio' in trecho_lower:
            if 'atividade' in trecho_lower:
                return "Relatório - atividade didática"
            if 'aulas' in trecho_lower:
                match = re.search(r'(\d{2}\.\d{2}\.\d{4})', trecho)
                datas = f" ({match.group(1)})" if match else ""
                return f"Relatório - aulas{datas}"
            ano = extrair_ano(trecho)
            return f"Relatório {ano if ano else ''}".strip()

        # Seminário/Leitura
        if 'seminário' in trecho_lower or 'seminario' in trecho_lower:
            match = re.search(r'seminário?\s+(\d+)', trecho_lower)
            num = match.group(1) if match else None
            return f"Seminário {num if num else ''}"

    # === FINANCEIRO ===
    if 'financeiro' in pasta_lower:
        # IRPF
        if 'irpf' in pasta_lower:
            match = re.search(r'(\d{4})', trecho)
            ano = match.group(1) if match else None
            resultado = f"Declaração IRPF {ano if ano else ''}".strip()
            return ' '.join(resultado.split())

        # Fatura/Crédito
        if 'fatura' in trecho_lower:
            match = re.search(r'(?:vencimento|fatura).*?(\d{1,2})/(\d{1,2})/(\d{4})', trecho, re.IGNORECASE)
            if match:
                return f"Fatura ({match.group(3)}-{match.group(2)}-{match.group(1)})"
            data = extrair_data_completa(trecho)
            return f"Fatura {data if data else ''}"

        # Contas
        if 'contas' in pasta_lower or 'conta' in pasta_lower:
            match = re.search(r'(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\w+[/\s]*(\d{4})', trecho_lower)
            if match:
                mes_ano = f"{match.group(1).upper()}/{match.group(2)}"
                return f"Contas {mes_ano}"
            data = extrair_data_completa(trecho)
            return f"Contas {data if data else ''}"

        # Boleto/Pagamento
        if 'boleto' in trecho_lower or 'pgmto' in nome_base.lower():
            data = extrair_data_completa(trecho)
            if data:
                return f"Boleto ({data})"
            return "Boleto"

        # Relatório Investimento
        if 'investimento' in trecho_lower:
            return "Relatório - investimento"

    # === IMÓVEIS ===
    if 'imóvel' in pasta_lower or 'apto' in pasta_lower:
        # Vistoria
        if 'vistoria' in trecho_lower:
            data = extrair_data_completa(trecho)
            return f"Vistoria {data if data else ''}"

        # Aluguel
        if 'aluguel' in trecho_lower or 'locação' in trecho_lower:
            data = extrair_data_completa(trecho)
            return f"Aluguel {data if data else ''}"

        # IPTU
        if 'iptu' in trecho_lower:
            ano = extrair_ano(trecho)
            resultado = f"IPTU {ano if ano else ''}".strip()
            return ' '.join(resultado.split())

        # Condomínio
        if 'condomínio' in trecho_lower or 'condominio' in trecho_lower:
            ano = extrair_ano(trecho)
            data = extrair_data_completa(trecho)
            return f"Condomínio {data or ano or ''}".strip()

    # === VEÍCULOS ===
    if 'veículo' in pasta_lower or 'cruze' in pasta_lower:
        # IPVA
        if 'ipva' in trecho_lower:
            ano = extrair_ano(trecho)
            resultado = f"IPVA {ano if ano else ''}".strip()
            return ' '.join(resultado.split())

        # Licenciamento
        if 'licenciamento' in trecho_lower:
            ano = extrair_ano(trecho)
            resultado = f"Licenciamento {ano if ano else ''}".strip()
            return ' '.join(resultado.split())

        # Parcela/Pgmto
        if 'parcela' in trecho_lower:
            match = re.search(r'(\d+)[ªaº]', trecho)
            num = match.group(1) if match else None
            data = extrair_data_completa(trecho)
            if num:
                return f"Parcela {num} ({data})" if data else f"Parcela {num}"
            return "Parcela"

    # === SAÚDE ===
    if 'saúde' in pasta_lower:
        if 'receita' in trecho_lower:
            data = extrair_data_completa(trecho)
            return f"Receita ({data})" if data else "Receita"
        if 'laudo' in trecho_lower:
            data = extrair_data_completa(trecho)
            return f"Laudo ({data})" if data else "Laudo"

    # === PADRÃO FINAL ===
    data = extrair_data_completa(trecho)
    if data:
        return f"{nome_base} ({data})"
    return nome_base

# Processar
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

            nome_novo = gerar_nome_novo(id_num, pasta, nome_atual, trecho)
            resultados.append(f"{id_num}|{nome_novo}")
            count += 1

# Salvar
with open(output_file, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(resultados))

print(f"Processadas: {count} linhas")
print(f"Arquivo: {output_file}")
