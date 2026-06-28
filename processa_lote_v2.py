import re

# Leitura do arquivo
linhas = []
with open(r"C:\Users\mauri\ProjetoConversor\lotes_ren\lote_012.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4:
            linhas.append(parts)

def extrair_data(trecho):
    """Extrai data em formato YYYY-MM-DD"""
    datas = re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', trecho)
    if datas:
        data_raw = datas[0]
        if '-' in data_raw and len(data_raw) == 10:
            return data_raw
        elif '/' in data_raw:
            partes = data_raw.split('/')
            if len(partes) == 3:
                try:
                    dia, mes, ano = partes
                    return f"{ano}-{mes}-{dia}"
                except:
                    pass
    return None

def gerar_nome_novo(id_num, pasta, nome_atual, trecho):
    """Gera nome novo descritivo e coerente com pasta."""

    # Remover extensão
    if '.' in nome_atual:
        nome_base = '.'.join(nome_atual.split('.')[:-1])
    else:
        nome_base = nome_atual

    # Se já começa com data válida, manter
    if re.match(r'^\d{4}-\d{2}(-\d{2})?', nome_base):
        return nome_base

    pasta_upper = pasta.upper()
    trecho_upper = trecho.upper()

    # Extrair data
    data = extrair_data(trecho)

    # ===== FINANCEIRO - FATURAS =====
    if 'CONTAS' in pasta_upper and 'FATURAS' in pasta_upper:
        match = re.search(r'([A-Z]{3})_(\d{4})', nome_atual, re.IGNORECASE)
        if match:
            mes_br = match.group(1)
            ano = match.group(2)
            meses = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05', 'JUN': '06',
                     'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'}
            if mes_br in meses:
                return f"{ano}-{meses[mes_br]} - Fatura"
        if data:
            return f"{data} - Fatura"
        return "Fatura"

    # ===== SAÚDE - DENTISTA =====
    if 'DENTISTA' in pasta_upper:
        if 'CLAREAMENTO' in trecho_upper:
            return "Nota fiscal - clareamento"
        return "Documentação dentista"

    # ===== FINANCEIRO - IRPF =====
    if 'IRPF' in pasta_upper:
        if 'GENIAL' in trecho_upper or 'MERCADO' in trecho_upper:
            return "Informe IRPF 2025"
        elif 'ALUGUEIS' in nome_atual.upper():
            return "Rendimentos aluguel - IRPF 2023"
        return "IRPF"

    # ===== FINANCEIRO - SEGUROS =====
    if 'SEGUROS' in pasta_upper:
        if 'ICATU' in nome_atual.upper() or 'ICATU' in trecho_upper:
            if 'FORMULÁRIO' in nome_atual.upper() or 'ADESÃO' in trecho_upper:
                return "Seguro - Formulário adesão CAR"
            elif 'AUMENTO' in nome_atual.upper():
                return "Seguro - Aumento CAR"
            else:
                return "Seguro Icatu - informações financeiras"
        return "Documentação seguro"

    # ===== IMÓVEIS =====
    if 'APTO' in pasta_upper or 'IMÓVEL' in pasta_upper or 'MYKONOS' in pasta_upper:
        if 'ALUGUEL' in pasta_upper or 'ALUGUEL' in trecho_upper:
            if 'BOLETO' in nome_atual.upper():
                match = re.search(r'(\d{2})/(\d{2})/(\d{4})', trecho)
                if match:
                    return f"{match.group(3)}-{match.group(2)}-{match.group(1)} - Boleto aluguel"
            return "Aluguel"
        if 'IPTU' in pasta_upper:
            return "IPTU"
        return "Documentação imóvel"

    # ===== NOTA FISCAL ELETRÔNICA =====
    if 'NFe' in pasta_upper:
        if 'NOTEBOOK' in trecho_upper or 'SAMSUNG' in trecho_upper:
            return "2023-12-01 - NFe - Notebook Samsung Galaxy Book3 360"
        elif 'TABLET' in trecho_upper:
            return "2022-12-15 - NFe - Tablet Samsung Galaxy Tab S7 FE"
        elif 'BALANÇA' in trecho_upper:
            return "2022-12-20 - NFe - Balança culinária digital Black&Decker"
        elif 'ROUPAS' in pasta_upper or 'ARAMIS' in trecho_upper:
            return "2024-04-21 - NFe - Roupas e calçados"
        if data:
            return f"{data} - Nota fiscal"
        return "Nota fiscal"

    # ===== JURÍDICO E TSE =====
    if 'JURIDICO' in pasta_upper or 'TSE' in pasta_upper:
        if 'PROCESSO' in trecho_upper or 'TRIBUNAL' in trecho_upper:
            if data:
                return f"{data} - Processo jurídico"
            return "Processo jurídico"
        if 'HISTÓRICO' in nome_atual.upper():
            return "Histórico escolar pós-graduação"
        if 'NOMEAÇÃO' in nome_atual.upper():
            return "Ato de nomeação TSE"
        if 'DECLARAÇÃO' in nome_atual.upper():
            return "Declaração TSE"
        if 'CONTRATO' in nome_atual.upper():
            return "Contrato"
        if 'MANDADO' in nome_atual.upper():
            return "Mandado de segurança"
        return "Documentação jurídica"

    # ===== SAÚDE - RECEITAS =====
    if 'SAÚDE' in pasta_upper or 'FITNESS' in pasta_upper:
        if 'RECEITA' in nome_atual.upper():
            if 'ÓCULOS' in trecho_upper:
                return "2026-03-06 - Receita - óculos"
            return "Receita médica"
        if 'MEDICAMENTO' in trecho_upper or 'REMÉDIO' in nome_atual.upper():
            if 'MANIPULADO' in pasta_upper:
                return "Medicamentos manipulados (NOV2018)"
        if 'NFS-e' in nome_atual.upper():
            return "NFS-e - Ortopedia (2023)"
        return "Documentação saúde"

    # ===== CONTAS BANCÁRIAS =====
    if 'BRB' in pasta_upper:
        if 'EXTRATO' in trecho_upper or 'EXTRATO' in nome_atual.upper():
            return "Extrato bancário BRB"
        if 'CONTRACHEQUE' in nome_atual.upper():
            match = re.search(r'([A-Z]{3})[-_](\d{4})', nome_atual, re.IGNORECASE)
            if match:
                return f"Contracheque {match.group(1).upper()}/{match.group(2)}"
            return "Contracheque"
        if 'APLICAÇÃO' in trecho_upper:
            return "Aplicação financeira Itaú"
        return "Documentação BRB"

    # ===== CONTAS PESSOAIS =====
    if 'CONTAS' in pasta_upper:
        if 'CAROL' in pasta_upper or 'ANA CAROLINA' in trecho_upper:
            match = re.search(r'([A-Z]{3})[-_/](\d{4})', nome_atual, re.IGNORECASE)
            if match:
                return f"{match.group(2)}-{match.group(1).upper()} - Contas Ana Carolina"
        if 'MATHEUS' in pasta_upper:
            match = re.search(r'([A-Z]{3})[-_/](\d{4})', nome_atual, re.IGNORECASE)
            if match:
                return f"{match.group(2)}-{match.group(1).upper()} - Contas Matheus"
        match = re.search(r'([A-Z]{3})[-_/](\d{4})', nome_atual, re.IGNORECASE)
        if match:
            return f"{match.group(2)}-{match.group(1).upper()} - Contas"
        return "Contas"

    # ===== VEÍCULOS =====
    if 'CRUZE' in pasta_upper or 'VEÍCULO' in pasta_upper:
        if 'PARCELA' in pasta_upper:
            return "Parcela financiamento Cruze"
        if 'BO' in nome_atual.upper():
            return "2019 - Boletim ocorrência (acidente)"
        return "Documentação veículo"

    # ===== TECNOLOGIA =====
    if 'TECNOLOGIA' in pasta_upper:
        if 'SENHAS' in pasta_upper:
            if 'STF' in nome_atual.upper():
                return "Senhas - STF"
            if 'GMAIL' in nome_atual.upper():
                return "Senhas - Gmail e Google Drive"
            if 'SCRIBD' in nome_atual.upper():
                return "Senhas - Scribd e 4shared"
            return "Senhas"
        return "Documentação tecnologia"

    # ===== VIAGEM =====
    if 'VIAGEM' in pasta_upper:
        if 'ESTA' in nome_atual.upper():
            return "2024-01-30 - Autorização ESTA (pendente)"
        if data:
            return f"{data} - Comprovante viagem"
        return "Comprovante viagem"

    # ===== FAMÍLIA/DIVERSOS =====
    if 'FAMÍLIA' in pasta_upper or 'ESPÍRITAS' in pasta_upper:
        return "Documentação pessoal"

    # ===== FALLBACK =====
    if len(nome_base) > 5 and not nome_base[0].isdigit():
        return nome_base[:90]

    return nome_base

# Processar
saida = ["id|nomeNovo"]

for parts in linhas:
    id_num = parts[0]
    pasta = parts[1] if len(parts) > 1 else ""
    nome_atual = parts[2] if len(parts) > 2 else ""
    trecho = parts[3] if len(parts) > 3 else ""

    nome_novo = gerar_nome_novo(id_num, pasta, nome_atual, trecho)
    saida.append(f"{id_num}|{nome_novo}")

# Salvar
with open(r"C:\Users\mauri\ProjetoConversor\lotes_ren_out\res_012.csv", "w", encoding="utf-8") as f:
    f.write('\n'.join(saida))

print(f"Processadas {len(linhas)} linhas")
