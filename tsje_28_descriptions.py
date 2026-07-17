# -*- coding: utf-8 -*-
"""tsje_28: preenche as descriptions das 3 databases TSJE e de cada
property — ajuda a IA (Notion AI/MCP) a interpretar o esquema.

Property description exige Notion-Version recente; tenta no PATCH da
database e reporta o que a API aceitou.

Uso: python tsje_28_descriptions.py
"""
import io
import json
import sys
import time

import requests

sys.stdout.reconfigure(encoding='utf-8')

TOKEN = io.open(r'c:\Users\mauri\ProjetoConversor\Chave_Notion.txt',
                encoding='utf-8').read().strip()
IDS = json.load(io.open(r'D:\TSJE_TRABALHO\tsje_notion_ids.json',
                        encoding='utf-8'))


def req(method, path, body=None, version='2022-06-28'):
    h = {'Authorization': f'Bearer {TOKEN}', 'Notion-Version': version,
         'Content-Type': 'application/json'}
    for i in range(5):
        r = requests.request(method, f'https://api.notion.com/v1{path}',
                             headers=h, json=body, timeout=60)
        if r.status_code == 429:
            time.sleep(float(r.headers.get('Retry-After', 2)))
            continue
        return r
    return r


# a API nao grava description em properties de RELATION (200 sem efeito,
# verificado nas rotas /databases e /data_sources) — as arestas do grafo
# ficam documentadas AQUI, na description da database, que a IA le no fetch
DB_DESC = {
    'db_atas': (
        'Atas das sessões do Tribunal Superior de Justiça Eleitoral '
        '(1932-1937), primeira fase da Justiça Eleitoral. 1 página = 1 '
        'sessão; o CORPO da página é a transcrição fiel da ata publicada '
        'no Boletim Eleitoral (ortografia modernizada, nomes de época). '
        'Fonte: acervo AtoM/TSE + coleção Searq. Relation "Processos '
        'desta sessão" = deliberações julgadas nesta sessão (aresta '
        'ata→processos do grafo).'),
    'db_processos': (
        'Processos e deliberações julgados pelo TSJE (1932-1937), '
        'extraídos das transcrições fiéis das atas. 1 página = 1 '
        'deliberação/processo apreciado em sessão. Relations: "Ata de '
        'origem" = sessão em que foi julgado (vazia apenas nos processos '
        'fonte=acórdão); "Acórdão" = acórdão publicado deste processo, '
        'quando localizado.'),
    'db_acordaos': (
        'Acórdãos do TSJE publicados na seção de jurisprudência dos '
        'Boletins Eleitorais (1932-1937). 1 página = 1 acórdão; o CORPO '
        'é o teor restaurado do OCR (sem invenção; [ilegível] onde '
        'irrecuperável). Relation "Processo" = deliberação/processo '
        'correspondente na base de Processos (aresta acórdão→processo).'),
}

PROPS_DESC = {
    'db_atas': {
        'data_sessao': 'Data em que a sessão foi realizada (impressa na ata).',
        'tipo': ('Tipo da sessão: ordinária, extraordinária, preparatória, '
                 'de instalação ou outra.'),
        'num': 'Número ordinal da sessão no ano (a numeração reinicia a cada ano).',
        'boletim': 'Boletim Eleitoral (nº e data) em que a ata foi publicada.',
        'arquivo_pdf': 'Arquivo PDF local do boletim (acervo D:\\TSJE_ATAS).',
        'paginas': 'Página(s) do PDF onde a ata está impressa.',
        'presidencia': 'Linha de presidência impressa no cabeçalho da ata.',
        'confianca': 'Confiança da transcrição fiel (alta/média/baixa).',
        'ilegiveis': 'Quantidade de trechos [ilegível] na transcrição.',
        'origem': 'Como a ata entrou no corpus: índice automático ou recuperada em auditoria.',
        'resumo_ia': 'Resumo da sessão gerado por IA (gpt-5.6-luna) a partir da transcrição fiel.',
        'palavras_chave': 'Termos de busca gerados por IA.',
        'incluir_no_rag': 'Flag de opt-in para indexação em RAG.',
        'ata_id': 'Identificador estável da ata no pipeline local.',
        'ano': 'Ano da sessão.',
        'id_seq': 'Chave numérica de partição (AAAAMMDD) para paginação por segmentos.',
        'atom_pdf': 'PDF oficial no acervo AtoM/TSE, aberto na página do dado (#page=N). Uso: validação humana.',
        'atom_item': 'Página catalográfica do boletim no AtoM (metadados arquivísticos oficiais).',
        'Processos desta sessão': 'Deliberações/processos extraídos desta ata (relation).',
    },
    'db_processos': {
        'classe': 'Classe processual normalizada (CONSULTA, RECURSO ELEITORAL, PROCESSO, ATOS E EXPEDIENTES...).',
        'UF': 'Estado de origem (siglas da época; DF = Distrito Federal no Rio).',
        'relator': 'Juiz relator da deliberação.',
        'numero': 'Número do processo como impresso (pode conter separador de milhar).',
        'interessado': 'Parte interessada/consulente.',
        'parte_recorrente': 'Parte recorrente (quando recurso).',
        'parte_recorrida': 'Parte recorrida (quando recurso).',
        'decisao': 'O que o Tribunal decidiu (extraído da ata; herdado do acórdão nos processos fonte=acórdão).',
        'ementa': 'Síntese decisória extraída da ata.',
        'punchline': 'Resumo de uma frase da deliberação.',
        'votacao': 'Como votaram (unânime, vencidos etc.).',
        'quorum': 'Juízes presentes na sessão.',
        'data_da_decisao': 'Data da sessão em que foi julgado.',
        'fonte': 'Origem da extração: ata (deliberação de sessão) ou acórdão (processo criado a partir de acórdão órfão).',
        'num_sessao': 'Número da sessão em que foi julgado.',
        'proc_id': 'Identificador estável no pipeline (ata_id#seq ou deacordao-*).',
        'id_seq': 'Chave numérica de partição para paginação por segmentos.',
        'incluir_no_rag': 'Flag de opt-in para indexação em RAG.',
        'Ata de origem': 'Ata da sessão em que a deliberação ocorreu (vazio apenas nos processos fonte=acórdão).',
        'Acórdão': 'Acórdão publicado correspondente (quando localizado).',
        'atom_pdf': 'PDF oficial no AtoM aberto na página da ata de origem (#page=N). Uso: validação humana.',
        'atom_item': 'Página catalográfica do boletim no AtoM.',
    },
    'db_acordaos': {
        'classe': 'Classe processual do acórdão.',
        'UF': 'Estado de origem.',
        'relator': 'Relator do acórdão (como impresso).',
        'numero_processo': 'Número do processo no cabeçalho do acórdão.',
        'data_julgamento': 'Data do julgamento (impressa ou herdada da sessão do processo).',
        'data_publicacao': 'Data do Boletim Eleitoral em que foi publicado.',
        'boletim': 'Identificação do boletim.',
        'arquivo_pdf': 'Arquivo PDF local do boletim.',
        'pagina': 'Página do PDF onde o acórdão está.',
        'confianca_ocr': 'Qualidade do teor: alta/média/baixa (restauro LLM) ou bruta (texto OCR sem restauro).',
        'resumo_ia': 'Dispositivo/resumo do acórdão.',
        'palavras_chave': 'Termos de busca gerados por IA.',
        'acordao_id': 'Identificador estável no pipeline local.',
        'id_seq': 'Chave numérica de partição para paginação por segmentos.',
        'incluir_no_rag': 'Flag de opt-in para indexação em RAG.',
        'Processo': 'Deliberação/processo correspondente na base de Processos.',
        'atom_pdf': 'PDF oficial no AtoM aberto na página do acórdão (#page=N). Uso: validação humana.',
        'atom_item': 'Página catalográfica do boletim no AtoM.',
    },
}


# data sources (schema mora aqui no modelo 2025-09-03)
DS = {'db_atas': '39f72195-5c64-8161-adf1-000bf49c617d',
      'db_acordaos': '39f72195-5c64-81c8-bb6e-000b4ccb0478',
      'db_processos': '39f72195-5c64-8166-83d6-000b47b2ccdf'}


def main():
    for db_key, desc in DB_DESC.items():
        db_id = IDS[db_key]
        r = req('PATCH', f'/databases/{db_id}', {
            'description': [{'type': 'text', 'text': {'content': desc}}]})
        print(f'{db_key} description: {r.status_code}')

        # property descriptions: PATCH no DATA SOURCE, incluindo o TIPO
        # com a config ATUAL (senao a API rejeita / apagaria options)
        ds_id = DS[db_key]
        atual = req('GET', f'/data_sources/{ds_id}',
                    version='2025-09-03').json()
        props_atuais = atual.get('properties') or {}
        corpo = {}
        for nome, d in PROPS_DESC.get(db_key, {}).items():
            spec = props_atuais.get(nome)
            if not spec:
                print(f'  (prop inexistente: {nome})')
                continue
            tipo = spec.get('type')
            if tipo in ('formula', 'rollup', 'unique_id'):
                continue
            # LIMITACAO da API (verificada 17/07/2026): em relations o
            # PATCH retorna 200 mas a description NAO e gravada — as
            # relations ficam documentadas so na description da database
            corpo[nome] = {tipo: spec.get(tipo) or {}, 'description': d}
        if not corpo:
            continue
        r = req('PATCH', f'/data_sources/{ds_id}', {'properties': corpo},
                version='2025-09-03')
        print(f'  props ({len(corpo)}): {r.status_code}'
              + ('' if r.status_code == 200 else f' {r.text[:160]}'))
        if r.status_code == 200:
            g = req('GET', f'/data_sources/{ds_id}',
                    version='2025-09-03').json()
            ok = sum(1 for n in corpo
                     if (g['properties'].get(n) or {}).get('description'))
            print(f'  verificadas com description: {ok}/{len(corpo)}')


if __name__ == '__main__':
    main()
