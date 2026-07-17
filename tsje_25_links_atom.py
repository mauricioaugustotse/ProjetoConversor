# -*- coding: utf-8 -*-
"""tsje_25: adiciona a TODAS as bases TSJE do Notion os links de validacao
no acervo oficial (AtoM/TSE):
- atom_pdf : link direto do PDF master com fragmento #page=N — o navegador
  abre o boletim NA PAGINA onde o dado esta;
- atom_item: pagina catalografica do item no AtoM (html).

Ponte arquivo local -> item do AtoM (nesta ordem):
1. nome com "[atom <hash8>]" -> hash8 casa o diretorio do url_pdf;
2. logs de download com duplicata_sha1 (slug -> arquivo local identico);
3. (ano, n do BE) do nome local; 4. data unica no ano.

Popula:
- Atas (coletar() do tsje_13: fm.arquivo + fm.paginas; pagina do PDF —
  quando ha "(imagem pNNN)" usa NNN);
- Acordaos (tsje_acordaos_catalogo.csv: pdf + pagina);
- Processos (processos_extraidos: ata_id -> arquivo/pagina da ata;
  orfaos "deacordao-<id>" -> pdf/pagina do acordao).

Idempotente: manifesto tsje_notion_work/links_done.jsonl.
Uso: python tsje_25_links_atom.py [--apply] [--limite N]
"""
import argparse
import csv
import importlib.util
import io
import json
import os
import re
import sys
import threading
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
TRAB = r'D:\TSJE_TRABALHO'
WORK = os.path.join(TRAB, 'tsje_notion_work')
CATALOGO = os.path.join(TRAB, 'tsje_atom_catalogo.csv')
DONE = os.path.join(WORK, 'links_done.jsonl')
ATOM_ITEM = 'https://atom.tse.jus.br/index.php/'

_lock = threading.Lock()


def _mod(nome):
    spec = importlib.util.spec_from_file_location(
        nome, os.path.join(BASE, nome + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t13 = _mod('tsje_13_popular_atas')
ln = _mod('_lia_notion')


# ---------- catalogo e pontes ----------

def catalogo():
    """slug -> dict(url_pdf, url_item, ano, num, data, hash8)."""
    itens = {}
    for r in csv.DictReader(io.open(CATALOGO, encoding='utf-8-sig')):
        url = r['url_pdf']
        m = re.search(r'/([0-9a-f]{8})[0-9a-f]*/[^/]+\.pdf$', url)
        mnum = re.search(r'\d+', r['num_be'] or '')
        itens[r['slug']] = {
            'url_pdf': url, 'url_item': ATOM_ITEM + r['slug'],
            'ano': (r['ano'] or '')[:4],
            'num': int(mnum.group()) if mnum else None,
            'data': r['data_producao'] or '',
            'hash8': m.group(1) if m else ''}
    return itens


def pontes(itens):
    """nome_do_arquivo_local (basename) -> slug."""
    por_hash = {v['hash8']: s for s, v in itens.items() if v['hash8']}
    por_ano_num = {(v['ano'], v['num']): s for s, v in itens.items()
                   if v['num'] is not None}
    por_data = defaultdict(list)
    for s, v in itens.items():
        if v['data']:
            por_data[v['data']].append(s)

    mapa = {}
    # 2) duplicatas sha1 dos logs
    import glob
    for lg in glob.glob(os.path.join(TRAB, 'log_tsje_atom_download_*.csv')):
        for r in csv.DictReader(io.open(lg, encoding='utf-8-sig')):
            if r['resultado'] == 'duplicata_sha1' and r['detalhe']:
                base = os.path.basename(r['detalhe'].replace('\\', '/'))
                if r['slug'] in itens:
                    mapa[base] = r['slug']
            elif r['resultado'] in ('ok', 'ja_baixado'):
                base = os.path.basename(r['destino'].replace('\\', '/'))
                if r['slug'] in itens:
                    mapa[base] = r['slug']

    def resolver(rel):
        """rel do cache/manifest ('BEs 1935\\...' ou 'ATOM 1936\\...') ->
        slug ou None."""
        base = os.path.basename(rel.replace('\\', '/'))
        if base in mapa:
            return mapa[base]
        m = re.search(r'\[atom ([0-9a-f]{8})\]', base)
        if m and m.group(1) in por_hash:
            mapa[base] = por_hash[m.group(1)]
            return mapa[base]
        mano = (re.match(r'(?:BEs|ATOM) (\d{4})', rel)
                or re.match(r'(\d{4})-\d{2}-\d{2}', base))
        ano = mano.group(1) if mano else ''
        mnum = (re.search(r'BE n0*(\d+)', base)
                or re.search(r'_a\d+_n0*(\d+)\.pdf$', base, re.I))
        if ano and mnum:
            s = por_ano_num.get((ano, int(mnum.group(1))))
            if s:
                mapa[base] = s
                return s
        mdata = re.match(r'(\d{4}-\d{2}-\d{2})', base)
        if mdata and len(por_data.get(mdata.group(1), [])) == 1:
            mapa[base] = por_data[mdata.group(1)][0]
            return mapa[base]
        # pasta 'BEs ANO\\MES\\DD\\id\\arquivo.pdf' -> data da pasta
        mp = re.match(r'BEs (\d{4})[\\/]([A-ZÇ]+)[\\/](\d{2})(?=[\\/])', rel)
        if mp:
            meses = {'JANEIRO': 1, 'FEVEREIRO': 2, 'MARÇO': 3, 'ABRIL': 4,
                     'MAIO': 5, 'JUNHO': 6, 'JULHO': 7, 'AGOSTO': 8,
                     'SETEMBRO': 9, 'OUTUBRO': 10, 'NOVEMBRO': 11,
                     'DEZEMBRO': 12}
            mm = meses.get(mp.group(2))
            if mm:
                d = f'{mp.group(1)}-{mm:02d}-{int(mp.group(3)):02d}'
                if len(por_data.get(d, [])) == 1:
                    return por_data[d][0]
        return None

    return resolver


def pagina_de(fm_paginas):
    """Extrai a pagina do PDF do campo `paginas` do front-matter."""
    if not fm_paginas:
        return None
    m = re.search(r'\(imagem p0*(\d+)\)', fm_paginas)
    if m:
        return int(m.group(1))
    m = re.match(r'\s*(\d+)', str(fm_paginas))
    if m and int(m.group(1)) <= 60:
        return int(m.group(1))
    return None


# ---------- montagem das filas ----------

def fila_atas(itens, resolver):
    pages = {}
    with io.open(os.path.join(WORK, 'atas_inserted.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pages[d['chave']] = d['id']
    out = []
    for r in t13.coletar():
        pid = pages.get(r['ata_id'])
        if not pid:
            continue
        fm = r['fm']
        arq = fm.get('arquivo', '')
        slug = resolver(arq) if arq else None
        out.append((f"ata:{r['ata_id']}", pid, slug,
                    pagina_de(fm.get('paginas'))))
    return out


def fila_acordaos(itens, resolver):
    pages = {}
    with io.open(os.path.join(WORK, 'acordaos_inserted.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pages[d['chave']] = d['id']
    cat = {}
    for r in csv.DictReader(io.open(
            os.path.join(TRAB, 'tsje_acordaos_catalogo.csv'),
            encoding='utf-8-sig')):
        cat[r['acordao_id']] = (r['pdf'], r['pagina'])
    out = []
    for aid, pid in pages.items():
        pdf, pag = cat.get(aid, ('', ''))
        slug = resolver(pdf) if pdf else None
        out.append((f'ac:{aid}', pid, slug,
                    int(pag) if str(pag).isdigit() else None))
    return out


def fila_processos(itens, resolver):
    pages = {}
    with io.open(os.path.join(WORK, 'processos_inserted.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pages[d['chave']] = d['id']
    # ata_id -> (arquivo, pagina)
    ata_fonte = {}
    for r in t13.coletar():
        fm = r['fm']
        ata_fonte[r['ata_id']] = (fm.get('arquivo', ''),
                                  pagina_de(fm.get('paginas')))
    cat = {}
    for r in csv.DictReader(io.open(
            os.path.join(TRAB, 'tsje_acordaos_catalogo.csv'),
            encoding='utf-8-sig')):
        cat[r['acordao_id']] = (r['pdf'], r['pagina'])
    out = []
    for chave, pid in pages.items():
        if chave.startswith('deacordao-'):
            pdf, pag = cat.get(chave[len('deacordao-'):], ('', ''))
            pagn = int(pag) if str(pag).isdigit() else None
        else:
            ata_id = chave.split('#')[0]
            pdf, pagn = ata_fonte.get(ata_id, ('', None))
        slug = resolver(pdf) if pdf else None
        out.append((f'proc:{chave}', pid, slug, pagn))
    return out


# ---------- schema + escrita ----------

def garantir_colunas(ids):
    for db in ('db_atas', 'db_acordaos', 'db_processos'):
        ln._req('PATCH', f"/databases/{ids[db]}",
                {'properties': {'atom_pdf': {'url': {}},
                                'atom_item': {'url': {}}}})
        print(f'colunas url garantidas em {db}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--limite', type=int, default=0)
    args = ap.parse_args()

    itens = catalogo()
    resolver = pontes(itens)
    ids = json.load(io.open(r'D:\TSJE_TRABALHO\tsje_notion_ids.json',
                            encoding='utf-8'))

    fila = fila_atas(itens, resolver) + fila_acordaos(itens, resolver) \
        + fila_processos(itens, resolver)

    feitos = set()
    if os.path.exists(DONE):
        with io.open(DONE, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    feitos.add(json.loads(line)['chave'])
    pend = [x for x in fila if x[0] not in feitos]
    sem_slug = sum(1 for x in pend if not x[2])
    print(f'registros: {len(fila)} | pendentes: {len(pend)} '
          f'| sem correspondencia AtoM: {sem_slug}')
    if args.limite:
        pend = pend[:args.limite]
    if not args.apply:
        for chave, pid, slug, pag in pend[:8]:
            v = itens.get(slug, {})
            frag = f"#page={pag}" if pag else ''
            print(f"  {chave[:44]:44s} -> "
                  f"{(v.get('url_pdf', 'SEM_LINK') + frag)[-60:]}")
        print('(dry-run; use --apply)')
        return

    garantir_colunas(ids)
    cont = {'ok': 0, 'vazio': 0, 'err': 0}

    def do(item):
        chave, pid, slug, pag = item
        try:
            if slug:
                v = itens[slug]
                frag = f'#page={pag}' if pag and pag > 1 else ''
                props = {'atom_pdf': {'url': v['url_pdf'] + frag},
                         'atom_item': {'url': v['url_item']}}
                ln._req('PATCH', f'/pages/{pid}', {'properties': props})
                res = 'ok'
            else:
                res = 'sem_slug'
            with _lock:
                with io.open(DONE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'chave': chave, 'res': res}) + '\n')
                cont['ok' if res == 'ok' else 'vazio'] += 1
                n = cont['ok'] + cont['vazio']
                if n % 250 == 0:
                    print(f'  ... {n}')
        except Exception as e:
            with _lock:
                cont['err'] += 1
                if cont['err'] <= 10:
                    print(f'  ERRO {chave}: {str(e)[:90]}')

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(do, pend))
    print(f"[FIM] {cont}")


if __name__ == '__main__':
    main()
