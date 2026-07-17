# -*- coding: utf-8 -*-
"""tsje_27: completa metadados e religa orfaos nas bases TSJE (auditoria
de 17/07/2026).

1. Acordaos sem relation com Processo (138) -> cria processo minimo
   (fonte=acordao) para TODOS e liga nos dois sentidos.
2. Processos sem "Ata de origem" indevidos (~56; os `deacordao-` sao
   legitimos) -> religa pelo ata_id embutido na chave do manifesto.
3. Acordaos sem data_julgamento (58) -> herda a data da sessao do processo
   relacionado; senao regex no teor ("Rio de Janeiro, D de MES de AAAA").
4. Acordaos sem relator/UF (23/33) -> regex no teor; relator restante via
   luna (mesmo prompt do tsje_26).
5. Processos sem decisao/data/ementa -> herda do acordao (dispositivo/
   data_julgamento/ementa) — majoritariamente os `deacordao-`.
6. 1 ata sem data_sessao (titulo truncado) -> corrige pelo .md local.

Uso: python tsje_27_completar_metadados.py [--apply] [--etapa N]
Manifesto: tsje_notion_work\\meta_done.jsonl
"""
import argparse
import importlib.util
import io
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
TRAB = r'D:\TSJE_TRABALHO'
WORK = os.path.join(TRAB, 'tsje_notion_work')
DONE = os.path.join(WORK, 'meta_done.jsonl')
IDS = json.load(io.open(r'D:\TSJE_TRABALHO\tsje_notion_ids.json',
                        encoding='utf-8'))
_lock = threading.Lock()


def _mod(nome):
    spec = importlib.util.spec_from_file_location(
        nome, os.path.join(BASE, nome + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ln = _mod('_lia_notion')
t17 = _mod('tsje_17_popular_acordaos')

MESES = {'janeiro': 1, 'fevereiro': 2, 'março': 3, 'marco': 3, 'abril': 4,
         'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8, 'setembro': 9,
         'outubro': 10, 'novembro': 11, 'dezembro': 12}


def query_filtrada(db_id, filtro):
    out, cursor = [], None
    while True:
        body = {'page_size': 100, 'filter': filtro}
        if cursor:
            body['start_cursor'] = cursor
        r = ln._req('POST', f'/databases/{db_id}/query', body)
        out += r.get('results', [])
        cursor = r.get('next_cursor')
        if not r.get('has_more'):
            break
    return out


def manif(chave=None, dado=None):
    if chave is None:
        s = set()
        if os.path.exists(DONE):
            with io.open(DONE, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        s.add(json.loads(line)['k'])
        return s
    with _lock:
        with io.open(DONE, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'k': chave, **(dado or {})},
                               ensure_ascii=False) + '\n')


def carregar_mapas():
    m = {'teor': {r['acordao_id']: r for r in t17.carregar_teor()},
         'ac_pages': {}, 'proc_pages': {}, 'ata_pages': {}, 'extr': {}}
    for nome, alvo in (('acordaos_inserted', 'ac_pages'),
                       ('processos_inserted', 'proc_pages'),
                       ('atas_inserted', 'ata_pages')):
        with io.open(os.path.join(WORK, nome + '.jsonl'),
                     encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    m[alvo][d['chave']] = d['id']
    with io.open(os.path.join(WORK, 'processos_extraidos.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for i, d in enumerate(r.get('deliberacoes', []), 1):
                d['_ata_id'] = r['ata_id']
                d['_data'] = r.get('data_sessao', '')
                m['extr'][f"{r['ata_id']}#{i:02d}"] = d
    return m


# ---------- etapa 1: acordaos orfaos -> processo minimo ----------

def etapa1(m, apply_, feitos):
    orfaos = query_filtrada(IDS['db_acordaos'],
                            {'property': 'Processo',
                             'relation': {'is_empty': True}})
    print(f'[1] acordaos sem processo: {len(orfaos)}')
    page2chave = {v: k for k, v in m['ac_pages'].items()}
    cont = Counter()

    def do(pg):
        pid = pg['id']
        aid = page2chave.get(pid, '')
        k = f'orfao|{aid or pid}'
        if k in feitos:
            return
        ac = m['teor'].get(aid)
        if not ac:
            manif(k, {'res': 'sem_teor'})
            return
        if not apply_:
            cont['plan'] += 1
            return
        chave_proc = f'deacordao-{aid}'
        proc_page = m['proc_pages'].get(chave_proc)
        if not proc_page:
            props = t17.props_proc_minimo(ac, None)
            novo = ln._req('POST', '/pages', {
                'parent': {'database_id': IDS['db_processos']},
                'properties': props})
            proc_page = novo['id']
            with _lock:
                with io.open(os.path.join(WORK, 'processos_inserted.jsonl'),
                             'a', encoding='utf-8') as f:
                    f.write(json.dumps({'chave': chave_proc,
                                        'id': proc_page}) + '\n')
                m['proc_pages'][chave_proc] = proc_page
        ln._req('PATCH', f'/pages/{pid}', {'properties': {
            'Processo': {'relation': [{'id': proc_page}]}}})
        manif(k, {'res': 'criado_ligado'})
        with _lock:
            cont['ok'] += 1

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(do, orfaos))
    print(f'  {dict(cont)}')


# ---------- etapa 2: processos sem ata de origem ----------

def etapa2(m, apply_, feitos):
    sem = query_filtrada(IDS['db_processos'],
                         {'property': 'Ata de origem',
                          'relation': {'is_empty': True}})
    page2chave = {v: k for k, v in m['proc_pages'].items()}
    fila = []
    legit = 0
    for pg in sem:
        ch = page2chave.get(pg['id'], '')
        if ch.startswith('deacordao-') or not ch:
            legit += 1
            continue
        fila.append((pg['id'], ch))
    print(f'[2] processos sem ata: {len(sem)} (legitimos deacordao: {legit}'
          f' | a tratar: {len(fila)})')
    cont = Counter()
    expurgar_atas = set()
    for pid, ch in fila:
        k = f'religa|{ch}'
        ata_id = ch.split('#')[0]
        ata_page = m['ata_pages'].get(ata_id)
        if not ata_page:
            # deliberacao-fantasma de ata DUPLICADA (o cache de extracao
            # re-contaminou o tsje_15) -> arquivar pagina + limpar fontes
            if apply_:
                ln._req('PATCH', f'/pages/{pid}', {'archived': True})
                manif(k, {'res': 'arquivado_fantasma'})
            expurgar_atas.add(ata_id)
            cont['arquivado' if apply_ else 'plan_arquivar'] += 1
            continue
        if k in feitos:
            continue
        if apply_:
            ln._req('PATCH', f'/pages/{pid}', {'properties': {
                'Ata de origem': {'relation': [{'id': ata_page}]}}})
            manif(k, {'res': 'religado'})
        cont['ok' if apply_ else 'plan_religar'] += 1
    print(f'  {dict(cont)}')
    if apply_ and expurgar_atas:
        # expurga do extraidos e do inserted para nao recontaminar
        for nome in ('processos_extraidos.jsonl', 'processos_inserted.jsonl'):
            p = os.path.join(WORK, nome)
            mantem = []
            with io.open(p, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    aid = (d.get('ata_id') or
                           d.get('chave', '').split('#')[0])
                    if aid not in expurgar_atas:
                        mantem.append(line)
            with io.open(p, 'w', encoding='utf-8') as f:
                f.writelines(mantem)
            print(f'  expurgo {nome}: {len(mantem)} linhas mantidas')


# ---------- etapa 3: acordaos sem data_julgamento ----------

RE_DATA_TEOR = re.compile(
    r'Rio de Janeiro,?\s+(?:em\s+)?(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})',
    re.I)


def etapa3(m, apply_, feitos):
    sem = query_filtrada(IDS['db_acordaos'],
                         {'property': 'data_julgamento',
                          'date': {'is_empty': True}})
    page2chave = {v: k for k, v in m['ac_pages'].items()}
    # data da sessao do processo relacionado (via associacao)
    assoc = {}
    import csv
    for r in csv.DictReader(io.open(
            os.path.join(TRAB, 'tsje_acordaos_associacao.csv'),
            encoding='utf-8-sig')):
        if r.get('proc_id'):
            assoc[r['acordao_id']] = r['proc_id']
    print(f'[3] acordaos sem data_julgamento: {len(sem)}')
    cont = Counter()
    for pg in sem:
        aid = page2chave.get(pg['id'], '')
        k = f'data|{aid}'
        if k in feitos or not aid:
            continue
        data = ''
        proc = m['extr'].get(assoc.get(aid, ''))
        if proc and proc.get('_data'):
            data, met = proc['_data'], 'sessao_do_processo'
        else:
            teor = (m['teor'].get(aid) or {}).get('teor', '')
            md = RE_DATA_TEOR.search(teor)
            if md and MESES.get(md.group(2).lower()):
                data = (f'{md.group(3)}-{MESES[md.group(2).lower()]:02d}-'
                        f'{int(md.group(1)):02d}')
                met = 'regex_teor'
        if not data:
            manif(k, {'res': 'sem_fonte'})
            cont['sem_fonte'] += 1
            continue
        if apply_:
            ln._req('PATCH', f"/pages/{pg['id']}", {'properties': {
                'data_julgamento': {'date': {'start': data}}}})
            manif(k, {'res': met, 'data': data})
        cont['ok' if apply_ else 'plan'] += 1
    print(f'  {dict(cont)}')


# ---------- etapa 4: acordaos sem relator / sem UF ----------

def etapa4(m, apply_, feitos):
    sys.path.insert(0, BASE)
    from conle_gerador import llm
    from tsje_14_extrair_processos import MINISTROS_ABREV
    t26 = _mod('tsje_26_normalizar_selects')

    for prop, tipo in (('relator', 'select'), ('UF', 'select')):
        sem = query_filtrada(IDS['db_acordaos'],
                             {'property': prop,
                              'select': {'is_empty': True}})
        page2chave = {v: k for k, v in m['ac_pages'].items()}
        print(f'[4] acordaos sem {prop}: {len(sem)}')
        cont = Counter()
        for pg in sem:
            aid = page2chave.get(pg['id'], '')
            k = f'{prop}|{aid}'
            if k in feitos or not aid:
                continue
            teor = (m['teor'].get(aid) or {}).get('teor', '')
            if not teor:
                manif(k, {'res': 'sem_teor'})
                continue
            valor = ''
            if prop == 'relator':
                mm = re.search(
                    r'[Rr]elator[,:]?\s+(?:o\s+)?(?:[Ss]r\.?\s+)?'
                    r'(?:[Mm]inistro\s+|[Dd]esembargador\s+|[Dd]r\.?\s+)?'
                    r'([A-ZÀ-Ú][\wÀ-ú\'.]+(?:\s+(?:de|da|do|dos)?\s*'
                    r'[A-ZÀ-Ú][\wÀ-ú\'.]+){0,3})', teor)
                if mm:
                    cand = t26.limpar_nome(mm.group(1))
                    if cand and not t26.eh_lixo(cand):
                        valor = cand
                if not valor and apply_:
                    valor = t26.luna_relator(teor) or ''
            else:
                mm = re.search(r'Estado d[oea]s?\s+([A-ZÀ-Ú][\wÀ-ú ]{2,25})',
                               teor)
                if mm:
                    valor = t26.luna_uf(teor[:3000], mm.group(1)) \
                        if apply_ else 'via_luna'
            if not valor:
                manif(k, {'res': 'nao_achado'})
                cont['nao_achado'] += 1
                continue
            if apply_:
                ln._req('PATCH', f"/pages/{pg['id']}", {'properties': {
                    prop: {'select': {'name': str(valor)[:95]}}}})
                manif(k, {'res': 'preenchido', 'valor': str(valor)})
            cont['ok' if apply_ else 'plan'] += 1
        print(f'  {prop}: {dict(cont)}')


# ---------- etapa 5: processos sem decisao/data/ementa ----------

def etapa5(m, apply_, feitos):
    page2chave = {v: k for k, v in m['proc_pages'].items()}
    for prop, tipo, fonte in (('decisao', 'rich_text', 'dispositivo'),
                              ('ementa', 'rich_text', 'ementa'),
                              ('data_da_decisao', 'date', 'data_julgamento')):
        sem = query_filtrada(IDS['db_processos'],
                             {'property': prop, tipo: {'is_empty': True}})
        cont = Counter()
        print(f'[5] processos sem {prop}: {len(sem)}')
        for pg in sem:
            ch = page2chave.get(pg['id'], '')
            k = f'p5|{prop}|{ch}'
            if k in feitos or not ch.startswith('deacordao-'):
                cont['nao_deacordao'] += 0 if not ch else 1
                continue
            ac = m['teor'].get(ch[len('deacordao-'):]) or {}
            valor = ac.get(fonte) or ''
            if not valor:
                manif(k, {'res': 'sem_fonte'})
                continue
            if apply_:
                if tipo == 'date':
                    props = {prop: {'date': {'start': valor}}}
                else:
                    props = {prop: {'rich_text': [
                        {'type': 'text',
                         'text': {'content': valor[:1900]}}]}}
                ln._req('PATCH', f"/pages/{pg['id']}",
                        {'properties': props})
                manif(k, {'res': 'herdado'})
            cont['ok' if apply_ else 'plan'] += 1
        print(f'  {prop}: {dict(cont)}')


# ---------- etapa 6: a ata sem data ----------

def etapa6(m, apply_, feitos):
    sem = query_filtrada(IDS['db_atas'],
                         {'property': 'data_sessao',
                          'date': {'is_empty': True}})
    print(f'[6] atas sem data_sessao: {len(sem)}')
    for pg in sem:
        tit = ''.join(t['plain_text'] for t in
                      (pg['properties'].get('Name') or {}).get('title', []))
        page2chave = {v: k for k, v in m['ata_pages'].items()}
        print(f"  page {pg['id']} | titulo: {tit!r} | "
              f"chave: {page2chave.get(pg['id'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--etapa', type=int, default=0)
    args = ap.parse_args()

    m = carregar_mapas()
    feitos = manif()
    etapas = {1: etapa1, 2: etapa2, 3: etapa3, 4: etapa4, 5: etapa5,
              6: etapa6}
    alvo = [args.etapa] if args.etapa else [1, 2, 3, 4, 5, 6]
    for n in alvo:
        etapas[n](m, args.apply, feitos)


if __name__ == '__main__':
    main()
