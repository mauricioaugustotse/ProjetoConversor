# -*- coding: utf-8 -*-
"""tsje_15: popula a database Notion "Processos e Julgados — TSJE" a partir
das deliberacoes extraidas pelo tsje_14 (processos_extraidos.jsonl), com
relation para a ata de origem (page_id via atas_inserted.jsonl).

Chave idempotente: proc_id = "<ata_id>#<indice>" (ordem das deliberacoes na
ata). id_seq = AAAAMMDD*100 + indice (particao anti-10k). Title:
"<classe> n. <numero> (<UF>)" ou "<classe> — <punchline curta>".

  python tsje_15_popular_processos.py                # dry-run
  python tsje_15_popular_processos.py --apply [--limite N]
"""
import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import sys
import threading

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _lia_notion as ln
import _stj_lib as L

WORK = r'D:\TSJE_TRABALHO\tsje_notion_work'
EXTRAIDOS = os.path.join(WORK, 'processos_extraidos.jsonl')
ATAS_INSERTED = os.path.join(WORK, 'atas_inserted.jsonl')
INSERTED = os.path.join(WORK, 'processos_inserted.jsonl')
IDS_PATH = r'D:\TSJE_TRABALHO\tsje_notion_ids.json'

_lock = threading.Lock()

CLASSES_CANON = ['CONSULTA', 'REPRESENTAÇÃO', 'RECURSO ELEITORAL',
                 'HABEAS CORPUS', 'PROCESSO', 'REGISTRO', 'RECLAMAÇÃO',
                 'APELAÇÃO CRIMINAL', 'APELAÇÃO', 'RECURSO CRIMINAL',
                 'RECURSO PARCIAL', 'RECURSO', 'AÇÃO PENAL',
                 'PEDIDO DE RECONSIDERAÇÃO', 'ATOS E EXPEDIENTES']


def canon_classe(c):
    c = (c or '').strip().upper().replace('-', ' ')
    c = re.sub(r'\s+', ' ', c)
    if not c:
        return 'ATOS E EXPEDIENTES'
    if c in CLASSES_CANON:
        return c
    for k in CLASSES_CANON:
        if c.startswith(k) or k.startswith(c):
            return k
    return c[:95]


def titulo_proc(d):
    classe = canon_classe(d.get('classe'))
    num = (d.get('numero_do_processo') or '').strip()
    uf = (d.get('UF') or '').strip().upper()
    if num:
        t = f'{classe.title()} n. {num}'
        if uf:
            t += f' ({uf})'
        return t
    punch = (d.get('punchline') or '').strip()
    return f'{classe.title()} — {punch[:120]}' if punch else classe.title()


def carregar_extraidos():
    """Ultima versao de cada ata (o jsonl pode ter re-extracoes)."""
    por_ata = {}
    if os.path.exists(EXTRAIDOS):
        for line in io.open(EXTRAIDOS, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                por_ata[r['ata_id']] = r
    return por_ata


def ata_pages():
    m = {}
    if os.path.exists(ATAS_INSERTED):
        for line in io.open(ATAS_INSERTED, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                m[r['chave']] = r['id']
    return m


def montar_linhas(por_ata, atapg):
    linhas = []
    for ata_id, r in sorted(por_ata.items()):
        d0 = r.get('data_sessao', '')
        data_ok = bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', d0))
        base_seq = int(d0.replace('-', '')) * 100 if data_ok else 0
        for i, d in enumerate(r.get('deliberacoes', []), 1):
            linhas.append({
                'proc_id': f'{ata_id}#{i:02d}',
                'ata_page': atapg.get(ata_id, ''),
                'ata_id': ata_id,
                'data': d0 if data_ok else '',
                'id_seq': base_seq + i if base_seq else None,
                'num_sessao': str(r.get('num_sessao', ''))[:100],
                'd': d,
            })
    return linhas


def props_do_processo(ln_reg):
    d = ln_reg['d']
    quorum = [q.strip() for q in (d.get('quorum') or '').split(',')
              if q.strip()][:30]
    props = ln.build_props(
        title=titulo_proc(d),
        select={'classe': canon_classe(d.get('classe')),
                'UF': (d.get('UF') or '').strip().upper()[:6],
                'relator': (d.get('relator') or '').strip()[:60],
                'fonte': 'ata'},
        rich={'numero': (d.get('numero_do_processo') or '').strip(),
              'num_sessao': ln_reg['num_sessao'],
              'punchline': d.get('punchline') or '',
              'decisao': d.get('decisao') or '',
              'ementa': d.get('ementa') or '',
              'votacao': d.get('votacao') or '',
              'parte_recorrente': d.get('parte_recorrente') or '',
              'parte_recorrida': d.get('parte_recorrida') or '',
              'interessado': d.get('interessado') or '',
              'proc_id': ln_reg['proc_id']},
        multi={'quorum': quorum},
        checkbox={'incluir_no_rag': True},
    )
    if ln_reg['data']:
        props['data_da_decisao'] = {'date': {'start': ln_reg['data']}}
    if ln_reg['id_seq']:
        props['id_seq'] = {'number': ln_reg['id_seq']}
    if ln_reg['ata_page']:
        props['Ata de origem'] = {'relation': [{'id': ln_reg['ata_page']}]}
    return props


def load_inserted():
    m = {}
    if os.path.exists(INSERTED):
        for line in io.open(INSERTED, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                m[r['chave']] = r['id']
    return m


def gravar_inserted(chave, pid):
    with _lock:
        with io.open(INSERTED, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'chave': chave, 'id': pid},
                               ensure_ascii=False) + '\n')


def reconciliar(db_id, manif):
    pages = L.query_all(db_id)
    novos = 0
    for pg in pages:
        pid = L.plain(pg['properties'].get('proc_id'))
        if pid and pid not in manif:
            manif[pid] = pg['id']
            gravar_inserted(pid, pg['id'])
            novos += 1
    if novos:
        print(f'reconciliação: +{novos} do estado real do database')
    return manif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--limite', type=int, default=0)
    args = ap.parse_args()

    with io.open(IDS_PATH, encoding='utf-8') as f:
        db_id = json.load(f)['db_processos']

    por_ata = carregar_extraidos()
    atapg = ata_pages()
    linhas = montar_linhas(por_ata, atapg)
    sem_ata = sum(1 for l in linhas if not l['ata_page'])
    manif = load_inserted()
    if args.apply:
        manif = reconciliar(db_id, manif)
    pend = [l for l in linhas if l['proc_id'] not in manif]
    if args.limite:
        pend = pend[:args.limite]
    print(f'deliberacoes: {len(linhas)} (de {len(por_ata)} atas; '
          f'{sem_ata} sem page da ata) | inseridas: {len(manif)} | '
          f'pendentes: {len(pend)}')

    if not args.apply:
        for l in pend[:8]:
            print(f"  - {titulo_proc(l['d'])[:90]} | {l['proc_id']}")
        print('[DRY-RUN] rode com --apply')
        return

    cont = {'ok': 0, 'err': 0}

    def do(l):
        try:
            pid = ln.inserir_linha(db_id, props_do_processo(l))
            gravar_inserted(l['proc_id'], pid)
            with _lock:
                cont['ok'] += 1
                if cont['ok'] % 200 == 0:
                    print(f"  ... {cont['ok']} inseridas")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f"  ERRO {l['proc_id']}: {str(e)[:180]}")

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(do, pend))
    print(f"[FIM] inseridas={cont['ok']} erros={cont['err']}")


if __name__ == '__main__':
    main()
