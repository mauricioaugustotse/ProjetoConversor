# -*- coding: utf-8 -*-
"""tsje_12: cria no Notion a estrutura da RAG do TSJE (primeira fase da
Justica Eleitoral): pagina-mae + 3 databases relacionadas.

  📜 Atas — Sessoes do TSJE   (1 linha por ata; corpo = transcricao fiel)
  📑 Acordaos — TSJE          (1 linha por acordao publicado nos BEs)
  ⚖️ Processos e Julgados     (1 linha por deliberacao; relations p/ ambas)

A pagina-mae nasce sob a pagina "Sessões" (viva e compartilhada com a
integracao — a API nao cria nada no root do workspace, onde vivem as bases
stj/stf/tsje-v1; mover para o root SO ao final de toda a populacao, senao o
token perde o acesso herdado). IDs gravados em
D:\\TSJE_TRABALHO\\tsje_notion_ids.json (idempotente: se o JSON existe e os
IDs respondem, nao recria nada).

Uso: python tsje_12_notion_dbs.py
"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _lia_notion as ln

# pagina "Sessões" (viva, compartilhada com a integracao) — hospedeira
# provisoria; mover a pagina-mae para o root so ao final da populacao
PARENT_PAGE = '2f972195-5c64-800e-83ae-d9f796b35eef'
IDS_PATH = r'D:\TSJE_TRABALHO\tsje_notion_ids.json'

TITULO_PAGINA = 'TSJE — Primeira fase da Justiça Eleitoral (RAG)'

SEL = lambda *ops: {'select': {'options': [{'name': o} for o in ops]}}
RT = {'rich_text': {}}
NUM = {'number': {}}
DATE = {'date': {}}
CHK = {'checkbox': {}}

SCHEMA_ATAS = {
    'Name': {'title': {}},
    'data_sessao': DATE,
    'tipo': SEL('ordinária', 'extraordinária', 'preparatória',
                'instalação', 'outra'),
    'num': NUM,
    'ano': SEL('1932', '1933', '1934', '1935', '1936', '1937'),
    'boletim': RT,
    'arquivo_pdf': RT,
    'paginas': RT,
    'presidencia': RT,
    'confianca': SEL('alta', 'média', 'baixa'),
    'ilegiveis': NUM,
    'origem': SEL('índice', 'recuperada'),
    'ata_id': RT,
    'id_seq': NUM,
    'resumo_ia': RT,
    'palavras_chave': RT,
    'incluir_no_rag': CHK,
}

SCHEMA_ACORDAOS = {
    'Name': {'title': {}},
    'numero_processo': RT,
    'classe': {'select': {}},
    'UF': {'select': {}},
    'relator': {'select': {}},
    'data_julgamento': DATE,
    'data_publicacao': DATE,
    'boletim': RT,
    'arquivo_pdf': RT,
    'pagina': NUM,
    'confianca_ocr': SEL('alta', 'média', 'baixa'),
    'acordao_id': RT,
    'id_seq': NUM,
    'resumo_ia': RT,
    'palavras_chave': RT,
    'incluir_no_rag': CHK,
}


def schema_processos(db_atas, db_acordaos):
    return {
        'Name': {'title': {}},
        'classe': {'select': {}},
        'UF': {'select': {}},
        'relator': {'select': {}},
        'numero': RT,
        'data_da_decisao': DATE,
        'num_sessao': RT,
        'punchline': RT,
        'decisao': RT,
        'ementa': RT,
        'votacao': RT,
        'quorum': {'multi_select': {}},
        'parte_recorrente': RT,
        'parte_recorrida': RT,
        'interessado': RT,
        'fonte': SEL('ata', 'acórdão'),
        'proc_id': RT,
        'id_seq': NUM,
        'incluir_no_rag': CHK,
        'Ata de origem': {'relation': {'database_id': db_atas,
                                       'type': 'dual_property',
                                       'dual_property': {}}},
        'Acórdão': {'relation': {'database_id': db_acordaos,
                                 'type': 'dual_property',
                                 'dual_property': {}}},
    }


def vivo(obj_id, tipo='databases'):
    try:
        ln._req('GET', f'/{tipo}/{obj_id}')
        return True
    except Exception:
        return False


def renomear_synced(db_id, para):
    """Renomeia a property sincronizada 'Related to ...' criada pelo dual."""
    db = ln._req('GET', f'/databases/{db_id}')
    for nome in db['properties']:
        if nome.startswith('Related to'):
            ln._req('PATCH', f'/databases/{db_id}',
                    {'properties': {nome: {'name': para}}})
            print(f'  synced renomeada: "{nome}" -> "{para}"')
            return


def main():
    ids = {}
    if os.path.exists(IDS_PATH):
        with io.open(IDS_PATH, encoding='utf-8') as f:
            ids = json.load(f)

    if not (ids.get('pagina') and vivo(ids['pagina'], 'pages')):
        ids['pagina'] = ln.criar_pagina(PARENT_PAGE, TITULO_PAGINA,
                                        icon_emoji='🗳️')
        print(f'pagina-mae criada: {ids["pagina"]}')
    else:
        print(f'pagina-mae ok: {ids["pagina"]}')

    if not (ids.get('db_atas') and vivo(ids['db_atas'])):
        ids['db_atas'] = ln.criar_database(
            ids['pagina'], 'Atas — Sessões do TSJE (1932–1937)',
            SCHEMA_ATAS, icon_emoji='📜')
        print(f'db atas criada: {ids["db_atas"]}')
    else:
        print(f'db atas ok: {ids["db_atas"]}')

    if not (ids.get('db_acordaos') and vivo(ids['db_acordaos'])):
        ids['db_acordaos'] = ln.criar_database(
            ids['pagina'], 'Acórdãos — TSJE (1932–1937)',
            SCHEMA_ACORDAOS, icon_emoji='📑')
        print(f'db acordaos criada: {ids["db_acordaos"]}')
    else:
        print(f'db acordaos ok: {ids["db_acordaos"]}')

    if not (ids.get('db_processos') and vivo(ids['db_processos'])):
        ids['db_processos'] = ln.criar_database(
            ids['pagina'], 'Processos e Julgados — TSJE (1932–1937)',
            schema_processos(ids['db_atas'], ids['db_acordaos']),
            icon_emoji='⚖️')
        print(f'db processos criada: {ids["db_processos"]}')
        renomear_synced(ids['db_atas'], 'Processos desta sessão')
        renomear_synced(ids['db_acordaos'], 'Processo')
    else:
        print(f'db processos ok: {ids["db_processos"]}')

    os.makedirs(os.path.dirname(IDS_PATH), exist_ok=True)
    with io.open(IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(ids, f, indent=2)
    print(f'ids -> {IDS_PATH}')


if __name__ == '__main__':
    main()
