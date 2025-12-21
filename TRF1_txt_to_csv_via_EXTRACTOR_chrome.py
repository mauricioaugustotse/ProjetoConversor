# -*- coding: utf-8 -*-
"""
Correção do extrator de acórdãos: emparelha o bloco-título com o bloco de conteúdo,
normaliza as chaves das seções, preenche corretamente as colunas do CSV e corrige
a formatação da coluna "ementa".
"""

import re
import csv
import os
import unicodedata
from pathlib import Path

def _strip_accents_lower(s: str) -> str:
    """Normaliza string para minúsculas e sem acentos."""
    if s is None:
        return ""
    return unicodedata.normalize("NFKD", s).encode("ASCII","ignore").decode("ASCII").strip().lower()

def split_acordaos(raw: str):
    """
    O arquivo traz: ===== (linha) / 'Acórdão NNN' / ===== (linha) / [seções...]
    Emparelha o título com o bloco imediatamente seguinte (conteúdo real).
    """
    parts = re.split(r"[=]{5,}", raw)
    acordaos = []
    i = 0
    while i < len(parts) - 1:
        title_block = parts[i]
        if re.search(r"^\s*Ac[oó]rd[ãa]o\s+\S+", title_block, flags=re.IGNORECASE | re.MULTILINE):
            content_block = parts[i+1] if i+1 < len(parts) else ""
            full = title_block.strip() + "\n" + content_block.strip()
            acordaos.append(full)
            i += 2
        else:
            i += 1
    return acordaos

def build_sections_dict(acordao_text: str):
    """
    Constrói dicionário {chave_normalizada: conteudo} para:
    [ Chave ]
    valor...
    """
    chunks = re.split(r"\n\s*\[([^\]]+)\]\s*\n", acordao_text, flags=re.DOTALL)
    sections = {}
    for i in range(1, len(chunks), 2):
        key = chunks[i].strip()
        value = chunks[i+1] if i+1 < len(chunks) else ""
        sections[_strip_accents_lower(key)] = value.strip()
    return sections

def parse_acordao(acordao_text):
    data = {
        'numero': '', 'classe': '', 'relator': '', 'orgao_julgador': '',
        'data_da_publicacao': '', 'ementa': '', 'decisao': '', 'resultado': ''
    }

    # número CNJ (linha do título já ajuda aqui)
    m_num = re.search(r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})', acordao_text)
    if m_num:
        data['numero'] = m_num.group(1)

    sections = build_sections_dict(acordao_text)

    # Classe (pega sigla em parênteses, se houver)
    classe_text = sections.get(_strip_accents_lower("Classe"), "") or sections.get(_strip_accents_lower("Classe/Assunto"), "")
    if classe_text:
        m_sigla = re.search(r'\(([^)]+)\)', classe_text)
        data['classe'] = (m_sigla.group(1).strip() if m_sigla else classe_text.strip())

    # Relator(a): remove titulações mais comuns e preserva o nome
    relator_text = sections.get(_strip_accents_lower("Relator(a)"), "") or sections.get(_strip_accents_lower("Relator"), "")
    if relator_text:
        relator_clean = re.sub(
            r'^(Desembargadora?\s+Federal|Desembargador\s+Federal|Ju[ií]za?\s+Federal|Ju[ií]z\s+Federal)\s+',
            '', relator_text.strip(), flags=re.IGNORECASE
        ).strip()
        data['relator'] = relator_clean

    # Órgão julgador (CORRIGIDO: converte ordinais por extenso)
    orgao_julgador_text = sections.get(_strip_accents_lower("Órgão julgador"), "") or sections.get(_strip_accents_lower("Orgao julgador"), "")
    if orgao_julgador_text:
        texto_formatado = orgao_julgador_text.strip()
        
        # Mapeamento de ordinais por extenso para sua forma numérica.
        # A ordem é importante: os mais longos (compostos) devem vir primeiro.
        ordinais_map = {
            r'DÉCIMA\s*-?\s*PRIMEIRA': '11ª',
            r'DÉCIMA\s*-?\s*SEGUNDA': '12ª',
            r'DÉCIMA\s*-?\s*TERCEIRA': '13ª',
            r'DÉCIMA\s*-?\s*QUARTA': '14ª',
            r'DÉCIMA\s*-?\s*QUINTA': '15ª',
            r'DÉCIMA\s*-?\s*SEXTA': '16ª',
            r'DÉCIMA\s*-?\s*SÉTIMA': '17ª',
            r'DÉCIMA\s*-?\s*OITAVA': '18ª',
            r'DÉCIMA\s*-?\s*NONA': '19ª',
            r'PRIMEIRA': '1ª',
            r'SEGUNDA': '2ª',
            r'TERCEIRA': '3ª',
            r'QUARTA': '4ª',
            r'QUINTA': '5ª',
            r'SEXTA': '6ª',
            r'SÉTIMA': '7ª',
            r'OITAVA': '8ª',
            r'NONA': '9ª',
            r'DÉCIMA': '10ª',
            r'VIGÉSIMA': '20ª'
        }
        
        # Substitui os ordinais por extenso
        for extenso, abreviado in ordinais_map.items():
            texto_formatado = re.sub(r'\b' + extenso + r'\b', abreviado, texto_formatado, flags=re.IGNORECASE | re.UNICODE)
            
        # Garante a capitalização correta de "Turma"
        texto_formatado = re.sub(r'\bTURMA\b', 'Turma', texto_formatado, flags=re.IGNORECASE)
        
        data['orgao_julgador'] = texto_formatado.strip()
    else:
        data['orgao_julgador'] = ""

    # Data da publicação (DD/MM/AAAA)
    data_pub_text = sections.get(_strip_accents_lower("Data da publicação"), "") or sections.get(_strip_accents_lower("Data"), "")
    if data_pub_text:
        m_data = re.search(r'(\d{2}/\d{2}/\d{4})', data_pub_text)
        if m_data:
            data['data_da_publicacao'] = m_data.group(1)

    # Ementa (CORRIGIDO: remove quebras de linha indevidas, preservando parágrafos)
    ementa_text = sections.get(_strip_accents_lower("Ementa"), "")
    if ementa_text:
        # 1. Normaliza quebras de parágrafo (2 ou mais newlines) para um marcador temporário
        text_with_placeholder = re.sub(r'(\s*\n\s*){2,}', '%%P%%', ementa_text.strip())
        
        # 2. Substitui as quebras de linha simples restantes (dentro de um parágrafo) por um espaço
        single_lines_joined = re.sub(r'\s*\n\s*', ' ', text_with_placeholder)
        
        # 3. Restaura as quebras de parágrafo, usando um único \n para separação
        final_text = re.sub(r'\s*%%P%%\s*', '\n', single_lines_joined)
        
        # 4. Limpa eventuais espaços múltiplos criados no processo
        final_text = re.sub(r' +', ' ', final_text)
        
        data['ementa'] = final_text.strip()

    # Decisão + Resultado
    decisao_text = sections.get(_strip_accents_lower("Decisão"), "") or sections.get(_strip_accents_lower("Decisao"), "")
    if decisao_text:
        line = ' '.join(decisao_text.strip().split())
        tipo = re.search(r'\bpor\s+(unanimidade|maioria)\b', line, flags=re.IGNORECASE)
        if tipo:
            data['decisao'] = tipo.group(1).lower()
        res = re.search(r'\bpor\s+(?:unanimidade|maioria)\s*,\s*(.*?)(?:\.\s*$|$)', line, flags=re.IGNORECASE)
        if res:
            rbruto = res.group(1).strip()
            data['resultado'] = re.sub(r',?\s*nos termos do voto.*$', '', rbruto, flags=re.IGNORECASE).strip()
        else:
            res2 = re.search(r'A\s+Turma,\s+por\s+(?:unanimidade|maioria)\s*,\s*(.*?)(?:\.\s*$|$)', line, flags=re.IGNORECASE)
            if res2:
                rbruto = res2.group(1).strip()
                data['resultado'] = re.sub(r',?\s*nos termos do voto.*$', '', rbruto, flags=re.IGNORECASE).strip()

    return data

def processar_lote(padrao_txt_dir='.', saida_csv=None):
    """Processa todos os arquivos .txt no diretório e gera um único CSV."""
    print('--- Iniciando extração ---')
    all_files = sorted([f for f in os.listdir(padrao_txt_dir) if f.endswith('.txt')])
    if not all_files:
        print(f'Nenhum arquivo .txt encontrado em: {os.path.abspath(padrao_txt_dir)}')
        return

    # Define o nome do arquivo de saída CSV
    if not saida_csv:
        saida_csv = os.path.splitext(all_files[0])[0] + '.csv'

    headers = ['numero','classe','relator','orgao_julgador','data_da_publicacao','ementa','decisao','resultado']
    rows = []
    print(f'Processando {len(all_files)} arquivo(s)...')
    for fname in all_files:
        fpath = os.path.join(padrao_txt_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            raw = f.read()
        for bloco in split_acordaos(raw):
            rows.append(parse_acordao(bloco))

    with open(saida_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    print(f'Sucesso! {len(rows)} registros salvos em: {os.path.abspath(saida_csv)}')

if __name__ == '__main__':
    # Para executar, coloque este script e os arquivos .txt no mesmo diretório
    # e rode o script. Ou, especifique o diretório como argumento.
    # Exemplo: processar_lote(padrao_txt_dir='C:/Users/SeuUsuario/Documentos/Acordaos')
    processar_lote()