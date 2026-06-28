import re

# Leitura do arquivo
linhas = []
with open(r"C:\Users\mauri\ProjetoConversor\lotes_ren\lote_012.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4:
            linhas.append(parts)

def gerar_nome_novo(id_num, pasta, nome_atual, trecho):
    """Gera um nome novo descritivo e coerente com a pasta."""
    
    # Remover extensão do nome_atual
    if '.' in nome_atual:
        nome_base = '.'.join(nome_atual.split('.')[:-1])
    else:
        nome_base = nome_atual
    
    # Se o nome já começa com data ou tem padrão bom, manter
    if nome_base and nome_base[0].isdigit() and '-' in nome_base:
        return nome_base
    
    descricao = ""
    data_str = ""
    
    # Procurar datas
    datas = re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', trecho)
    if datas:
        data_raw = datas[0]
        if '-' in data_raw and len(data_raw) == 10:
            data_str = data_raw
        elif '/' in data_raw:
            partes = data_raw.split('/')
            if len(partes) == 3:
                try:
                    dia, mes, ano = partes
                    data_str = f"{ano}-{mes}-{dia}"
                except:
                    pass
    
    # Procurar números de documento
    trecho_upper = trecho.upper()
    
    # Classificar por palavra-chave
    if 'FATURA' in trecho_upper or 'FATURAS' in pasta.upper():
        descricao = 'Fatura'
    elif 'CLAREAMENTO' in trecho_upper:
        descricao = 'Nota fiscal - clareamento'
    elif 'IRPF' in pasta.upper() or 'IMPOSTO DE RENDA' in trecho_upper:
        descricao = 'IRPF'
    elif 'SEGURO' in pasta.upper() or 'APÓLICE' in trecho_upper:
        descricao = 'Seguro'
    elif 'EXTRATO' in trecho_upper or ('CONTAS' in pasta.upper() and 'EXTRATO' in trecho_upper):
        descricao = 'Extrato'
    elif 'NOTA FISCAL' in trecho_upper or 'NFe' in nome_atual:
        descricao = 'Nota fiscal'
    elif 'BOLETO' in nome_atual.upper() or 'ALUGUEL' in trecho_upper:
        descricao = 'Boleto aluguel'
    elif 'IPTU' in pasta.upper():
        descricao = 'IPTU'
    elif 'NOTA FISCAL ELETRÔNICA' in trecho_upper or 'DANFE' in trecho_upper:
        descricao = 'NF-e'
    elif 'RECIBO' in trecho_upper:
        descricao = 'Recibo'
    elif any(x in pasta.upper() for x in ['ESTUDOS', 'TESE', 'CONCURSO', 'LEITURA']):
        descricao = nome_base
    elif 'MÉDICO' in trecho_upper or 'RECEITA' in nome_atual.upper():
        descricao = 'Receita médica'
    elif 'PROCESSO' in trecho_upper or 'TRIBUNAL' in trecho_upper:
        descricao = 'Processo jurídico'
    elif 'VIAGEM' in pasta.upper():
        descricao = 'Comprovante viagem'
    elif 'CONTRATO' in trecho_upper:
        descricao = 'Contrato'
    elif 'SALÁRIO' in trecho_upper or 'CONTRACHEQUE' in nome_atual.upper():
        descricao = 'Contracheque'
    else:
        descricao = nome_base
    
    # Montar nome final
    if descricao and descricao != nome_base:
        if data_str:
            nome_novo = f"{data_str} - {descricao}"
        else:
            nome_novo = descricao
    else:
        nome_novo = nome_base
    
    # Limitar a 90 caracteres
    nome_novo = nome_novo[:90]
    
    return nome_novo

# Processar linhas
saida = ["id|nomeNovo"]

for parts in linhas:
    id_num = parts[0]
    pasta = parts[1] if len(parts) > 1 else ""
    nome_atual = parts[2] if len(parts) > 2 else ""
    trecho = parts[3] if len(parts) > 3 else ""
    
    nome_novo = gerar_nome_novo(id_num, pasta, nome_atual, trecho)
    saida.append(f"{id_num}|{nome_novo}")

# Escrever saída
with open(r"C:\Users\mauri\ProjetoConversor\lotes_ren_out\res_012.csv", "w", encoding="utf-8") as f:
    f.write('\n'.join(saida))

print(f"Processadas {len(linhas)} linhas")
