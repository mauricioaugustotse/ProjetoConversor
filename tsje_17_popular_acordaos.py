# -*- coding: utf-8 -*-
"""tsje_17: popula a database Notion "Acórdãos — TSJE" com o teor restaurado
(tsje_16b) e ASSOCIA cada acordao ao processo correspondente na database
"Processos e Julgados" (relation dos dois lados via dual_property).

Match acordao -> deliberacao (cascata):
  a) mesmo numero + data_julgamento == data_sessao        -> CONFIRMADO
  b) mesmo numero + unica deliberacao com esse numero     -> CONFIRMADO
  c) mesmo numero + data mais proxima (ate 60 dias)       -> PROVAVEL
  d) mesmo numero + nenhuma data compativel               -> AMBIGUO (fila)
  e) sem match de numero -> SEM_PROCESSO: cria processo minimo
     (fonte=acórdão) e relaciona
Acordaos sem numero legivel: SEM_NUMERO (entram sem relation).

Relatorio: D:\\TSJE_TRABALHO\\tsje_acordaos_associacao.csv
Idempotente: manifesto acordaos_inserted.jsonl por acordao_id.

  python tsje_17_popular_acordaos.py                # dry-run
  python tsje_17_popular_acordaos.py --apply [--limite N]
"""
import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import re
import sys
import threading
from datetime import date

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _lia_notion as ln
import _stj_lib as L

WORK = r'D:\TSJE_TRABALHO\tsje_notion_work'
TEOR = os.path.join(WORK, 'acordaos_teor.jsonl')
PROC_EXTR = os.path.join(WORK, 'processos_extraidos.jsonl')
PROC_INS = os.path.join(WORK, 'processos_inserted.jsonl')
INSERTED = os.path.join(WORK, 'acordaos_inserted.jsonl')
IDS_PATH = r'D:\TSJE_TRABALHO\tsje_notion_ids.json'
ASSOC_CSV = r'D:\TSJE_TRABALHO\tsje_acordaos_associacao.csv'

_lock = threading.Lock()

MESES_DIR = {'JANEIRO': 1, 'FEVEREIRO': 2, 'MARÇO': 3, 'MARCO': 3,
             'ABRIL': 4, 'MAIO': 5, 'JUNHO': 6, 'JULHO': 7, 'AGOSTO': 8,
             'SETEMBRO': 9, 'OUTUBRO': 10, 'NOVEMBRO': 11, 'DEZEMBRO': 12}


def num_norm(n):
    return re.sub(r'\D', '', n or '')


def data_do_pdf(rel):
    """'BEs 1933\\JULHO\\05\\99287\\x.pdf' -> 1933-07-05 (publicacao);
    'ATOM 1935\\1935-03-12 - BE n25 [atom x].pdf' -> 1935-03-12."""
    m = re.match(r'BEs (\d{4})[\\/]([A-ZÇ]+)[\\/](\d{2})(?=[\\/])', rel)
    if m and m.group(3) != '00':
        mes = MESES_DIR.get(m.group(2))
        if mes:
            return f'{m.group(1)}-{mes:02d}-{int(m.group(3)):02d}'
    m = re.search(r'[\\/](\d{4}-\d{2}-\d{2}) - BE', rel)
    if m:
        return m.group(1)
    return ''


def iso_dias(d):
    try:
        a, m, dd = int(d[:4]), int(d[5:7]), int(d[8:10])
        return date(a, m, dd).toordinal()
    except Exception:
        return None


def carregar_teor():
    regs = {}
    if os.path.exists(TEOR):
        for line in io.open(TEOR, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                regs[r['acordao_id']] = r
    return list(regs.values())


def indice_processos():
    """numero_norm -> [(proc_id, data_sessao, classe, uf)]."""
    idx = {}
    if os.path.exists(PROC_EXTR):
        for line in io.open(PROC_EXTR, encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            for i, d in enumerate(r.get('deliberacoes', []), 1):
                nn = num_norm(d.get('numero_do_processo'))
                if nn:
                    idx.setdefault(nn, []).append(
                        (f"{r['ata_id']}#{i:02d}", r.get('data_sessao', ''),
                         (d.get('classe') or '').upper(),
                         (d.get('UF') or '').upper()))
    return idx


def proc_pages():
    m = {}
    if os.path.exists(PROC_INS):
        for line in io.open(PROC_INS, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                m[r['chave']] = r['id']
    return m


def casar(ac, idx):
    """-> (veredito, proc_id|None, detalhe)."""
    nn = num_norm(ac.get('numero_processo'))
    if not nn:
        return 'SEM_NUMERO', None, ''
    cands = idx.get(nn, [])
    if not cands:
        return 'SEM_PROCESSO', None, f'n.{nn} nao consta das atas'
    dj = ac.get('data_julgamento', '')
    if dj:
        exatos = [c for c in cands if c[1] == dj]
        if exatos:
            return 'CONFIRMADO', exatos[0][0], 'data exata'
    if len(cands) == 1:
        return 'CONFIRMADO', cands[0][0], 'numero unico'
    oj = iso_dias(dj) if dj else None
    if oj:
        com_d = [(abs(iso_dias(c[1]) - oj), c) for c in cands
                 if iso_dias(c[1]) is not None]
        com_d.sort(key=lambda x: x[0])
        if com_d and com_d[0][0] <= 60:
            return 'PROVAVEL', com_d[0][1][0], f'delta {com_d[0][0]}d'
    return 'AMBIGUO', None, f'{len(cands)} candidatos sem data compativel'


def titulo_ac(ac):
    num = ac.get('numero_processo', '')
    cl = (ac.get('classe') or 'Processo').title()
    uf = ac.get('uf', '')
    t = f'Acórdão — {cl}'
    if num:
        t += f' n. {num}'
    if uf:
        t += f' ({uf})'
    return t


def blocos_ac(ac):
    blocos = []
    if ac.get('ementa'):
        blocos.append({'object': 'block', 'type': 'quote', 'quote': {
            'rich_text': ln._rt_chunks(ac['ementa'])}})
    for par in (ac.get('teor') or '').split('\n\n'):
        par = par.strip()
        if par:
            blocos.append({'object': 'block', 'type': 'paragraph',
                           'paragraph': {'rich_text': ln._rt_chunks(par)}})
    rodape = (f"Fonte: {ac['pdf']}, pág. {ac['pagina']} "
              f"(teor restaurado de OCR; confiança {ac.get('confianca')})")
    blocos.append({'object': 'block', 'type': 'paragraph', 'paragraph': {
        'rich_text': [{'type': 'text', 'text': {'content': rodape[:1900]},
                       'annotations': {'italic': True}}]}})
    return blocos


def props_ac(ac, proc_page):
    dp = data_do_pdf(ac['pdf'])
    props = ln.build_props(
        title=titulo_ac(ac),
        select={'classe': (ac.get('classe') or '')[:95],
                'UF': ac.get('uf', ''),
                'relator': ac.get('relator', '')[:60],
                'confianca_ocr': {'media': 'média'}.get(
                    ac.get('confianca'), ac.get('confianca', ''))},
        rich={'numero_processo': ac.get('numero_processo', ''),
              'boletim': '',
              'arquivo_pdf': ac['pdf'],
              'acordao_id': ac['acordao_id'],
              'resumo_ia': ac.get('dispositivo', '')},
        checkbox={'incluir_no_rag': True},
    )
    if ac.get('pagina'):
        props['pagina'] = {'number': int(ac['pagina'])}
    if ac.get('data_julgamento'):
        props['data_julgamento'] = {'date': {'start': ac['data_julgamento']}}
    if dp:
        props['data_publicacao'] = {'date': {'start': dp}}
        props['id_seq'] = {'number': int(dp.replace('-', '')) * 100
                           + int(ac['pagina'] or 0) % 100}
    if proc_page:
        props['Processo'] = {'relation': [{'id': proc_page}]}
    return props


def props_proc_minimo(ac, db_processos):
    """Processo criado a partir de acordao orfao (fonte=acórdão)."""
    num = ac.get('numero_processo', '')
    cl = (ac.get('classe') or 'PROCESSO').upper()
    uf = ac.get('uf', '')
    t = f'{cl.title()} n. {num}' + (f' ({uf})' if uf else '')
    props = ln.build_props(
        title=t,
        select={'classe': cl[:95], 'UF': uf,
                'relator': ac.get('relator', '')[:60], 'fonte': 'acórdão'},
        rich={'numero': num,
              'punchline': ac.get('dispositivo', ''),
              'ementa': ac.get('ementa', ''),
              'votacao': ac.get('votacao', ''),
              'proc_id': f"deacordao-{ac['acordao_id']}"},
        checkbox={'incluir_no_rag': True},
    )
    if ac.get('data_julgamento'):
        props['data_da_decisao'] = {'date': {'start': ac['data_julgamento']}}
        props['id_seq'] = {'number':
                           int(ac['data_julgamento'].replace('-', '')) * 100}
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
    for pg in L.query_all(db_id):
        aid = L.plain(pg['properties'].get('acordao_id'))
        if aid and aid not in manif:
            manif[aid] = pg['id']
            gravar_inserted(aid, pg['id'])
    return manif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--limite', type=int, default=0)
    args = ap.parse_args()

    with io.open(IDS_PATH, encoding='utf-8') as f:
        ids = json.load(f)
    db_ac, db_proc = ids['db_acordaos'], ids['db_processos']

    acs = carregar_teor()
    idx = indice_processos()
    ppages = proc_pages()
    manif = load_inserted()
    if args.apply:
        manif = reconciliar(db_ac, manif)

    from collections import Counter
    planos = []
    veredx = Counter()
    for ac in acs:
        ver, proc_id, det = casar(ac, idx)
        veredx[ver] += 1
        planos.append((ac, ver, proc_id, det))

    with io.open(ASSOC_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['acordao_id', 'numero', 'veredito', 'proc_id',
                    'detalhe', 'data_julgamento', 'pdf', 'pagina'])
        for ac, ver, proc_id, det in planos:
            w.writerow([ac['acordao_id'], ac.get('numero_processo', ''),
                        ver, proc_id or '', det,
                        ac.get('data_julgamento', ''), ac['pdf'],
                        ac['pagina']])
    pend = [p for p in planos if p[0]['acordao_id'] not in manif]
    if args.limite:
        pend = pend[:args.limite]
    print(f'acordaos com teor: {len(acs)} | inseridos: {len(manif)} | '
          f'pendentes: {len(pend)}')
    print('vereditos: ' + ', '.join(f'{k}={v}' for k, v
                                    in veredx.most_common()))
    print(f'associacao -> {ASSOC_CSV}')
    if not args.apply:
        print('[DRY-RUN] rode com --apply')
        return

    cont = Counter()

    def do(item):
        ac, ver, proc_id, det = item
        try:
            proc_page = ppages.get(proc_id) if proc_id else None
            if ver == 'SEM_PROCESSO' and num_norm(ac.get('numero_processo')):
                chave = f"deacordao-{ac['acordao_id']}"
                if chave in ppages:
                    proc_page = ppages[chave]
                else:
                    proc_page = ln.inserir_linha(
                        db_proc, props_proc_minimo(ac, db_proc))
                    with _lock:
                        ppages[chave] = proc_page
                    with io.open(PROC_INS, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({'chave': chave,
                                            'id': proc_page},
                                           ensure_ascii=False) + '\n')
            blocos = blocos_ac(ac)
            pid = ln.inserir_linha(db_ac, props_ac(ac, proc_page),
                                   children=blocos[:100])
            for i in range(100, len(blocos), 100):
                ln._req('PATCH', f'/blocks/{pid}/children',
                        {'children': blocos[i:i + 100]})
            gravar_inserted(ac['acordao_id'], pid)
            with _lock:
                cont['ok'] += 1
                if cont['ok'] % 50 == 0:
                    print(f"  ... {cont['ok']}")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f"  ERRO {ac['acordao_id']}: {str(e)[:160]}")

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(do, pend))
    print(f"[FIM] ok={cont['ok']} erros={cont['err']}")


if __name__ == '__main__':
    main()
