# -*- coding: utf-8 -*-
"""tsje_10: inventario da colecao "Primeira fase da Justica Eleitoral" no
AtoM do TSE (atom.tse.jus.br) e cruzamento com o acervo local D:\\TSJE_ATAS.

Fase A (crawl, retomavel): browse da colecao 2495 filtrado por nivel
"Record group" (levels=428 = os boletins; os demais 9 itens sao fundo/serie/
subseries) -> slugs; depois a pagina de cada item -> URL do PDF master,
data de producao e ano do breadcrumb. Rate-limit educado + retry.
Saida: D:\\TSJE_TRABALHO\\tsje_atom_catalogo.csv (1 linha por boletim).

Fase B (cruzamento): compara (ano, n do BE) do catalogo com os nomes do
banco local `AAAA-MM-DD - BE nNNN [id].pdf`; empate de numero sem data
compativel e resolvido pela data. Saida: tsje_atom_plano.csv com decisao
baixar / ja_temos / sem_pdf / fora_escopo.

Uso: python tsje_10_atom_inventario.py [--so-cruzar]
"""
import argparse
import csv
import html as htmllib
import io
import os
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding='utf-8')

BASE = 'https://atom.tse.jus.br'
BROWSE = (BASE + '/index.php/informationobject/browse'
          '?levels=428&collection=2495&topLod=0&sort=alphabetic&sortDir=asc')
BANCO = r'D:\TSJE_ATAS'
TRAB = r'D:\TSJE_TRABALHO'
CATALOGO = os.path.join(TRAB, 'tsje_atom_catalogo.csv')
PLANO = os.path.join(TRAB, 'tsje_atom_plano.csv')
PAUSA = 0.8          # s entre requests
UA = {'User-Agent': 'Mozilla/5.0 (pesquisa historica; contato via TSE)'}

CAMPOS = ['slug', 'titulo', 'ano', 'num_be', 'data_producao', 'url_pdf',
          'arquivo_pdf', 'obs']


def get(url, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 502, 503, 504):
                time.sleep(5 * (i + 1))
                continue
            raise RuntimeError(f'HTTP {r.status_code} em {url}')
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError(f'esgotou tentativas: {url}')


def listar_slugs():
    """Browse paginado -> [(slug, titulo)] na ordem alfabetica do site."""
    vistos, ordem = set(), []
    pag = 1
    while True:
        url = BROWSE if pag == 1 else BROWSE.replace(
            '/browse?', f'/browse?page={pag}&')
        h = get(url)
        blocos = re.findall(
            r'<a class="h5[^"]*" href="/index\.php/([^"]+)" title="([^"]*)"', h)
        novos = 0
        for slug, tit in blocos:
            if slug not in vistos:
                vistos.add(slug)
                ordem.append((slug, htmllib.unescape(tit).strip()))
                novos += 1
        m = re.search(r'de\s+([\d.]+)\s+resultados', h)
        total = int(m.group(1).replace('.', '')) if m else None
        print(f'  browse pag {pag}: +{novos} (acum {len(ordem)}'
              f'{" de " + str(total) if total else ""})')
        if not blocos or (total and len(ordem) >= total):
            break
        pag += 1
        time.sleep(PAUSA)
    return ordem


def parse_item(slug, titulo):
    """Pagina do item -> dict do catalogo."""
    h = get(f'{BASE}/index.php/{slug}')
    reg = {'slug': slug, 'titulo': titulo, 'ano': '', 'num_be': '',
           'data_producao': '', 'url_pdf': '', 'arquivo_pdf': '', 'obs': ''}
    pdfs = sorted(set(re.findall(r'/uploads/r/[^"\']+?\.pdf', h)))
    if pdfs:
        reg['url_pdf'] = BASE + pdfs[0]
        reg['arquivo_pdf'] = pdfs[0].rsplit('/', 1)[-1]
        if len(pdfs) > 1:
            reg['obs'] = f'{len(pdfs)} pdfs na pagina'
    else:
        reg['obs'] = 'sem pdf'
    m = re.search(r'(\d{2}/\d{2}/\d{4})\s*\(Produ', h)
    if m:
        d, mo, a = m.group(1).split('/')
        reg['data_producao'] = f'{a}-{mo}-{d}'
    m = re.search(r'Boletim Eleitoral (\d{4})', h)
    if m:
        reg['ano'] = m.group(1)
    elif reg['data_producao']:
        reg['ano'] = reg['data_producao'][:4]
    m = re.search(r'n[º°.]?\s*(\d+)\s*$', titulo.strip()) or \
        re.search(r'n[º°.]?\s*(\d+)', titulo)
    if m:
        reg['num_be'] = m.group(1)
    return reg


def crawl():
    feitos = {}
    if os.path.exists(CATALOGO):
        with io.open(CATALOGO, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                feitos[r['slug']] = r
        print(f'catalogo existente: {len(feitos)} itens (retomando)')

    print('listando slugs da colecao (levels=428)...')
    slugs = listar_slugs()
    print(f'{len(slugs)} boletins no AtoM')

    pend = [(s, t) for s, t in slugs if s not in feitos]
    print(f'{len(pend)} itens a detalhar')
    novos = 0
    with io.open(CATALOGO, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction='ignore')
        w.writeheader()
        for s, t in slugs:
            if s in feitos:
                w.writerow(feitos[s])
        f.flush()
        for i, (s, t) in enumerate(pend, 1):
            try:
                reg = parse_item(s, t)
            except Exception as e:
                reg = {'slug': s, 'titulo': t, 'ano': '', 'num_be': '',
                       'data_producao': '', 'url_pdf': '', 'arquivo_pdf': '',
                       'obs': f'ERRO: {e}'}
            w.writerow(reg)
            novos += 1
            if i % 25 == 0:
                f.flush()
                print(f'  itens {i}/{len(pend)}')
            time.sleep(PAUSA)
    print(f'catalogo -> {CATALOGO} ({len(feitos) + novos} linhas)')


# nomes do banco: "AAAA-MM-DD - BE n108 [id].pdf" OU "AAAA-MM-DD - BE [id].pdf"
RE_LOCAL = re.compile(r'^(\d{4})-(\d{2})-(\d{2}) - BE (?:n(\d+)\s*)?\[')


def acervo_local():
    """Dois indices do banco: por (ano, num_be) e por data AAAA-MM-DD."""
    por_num, por_data = {}, {}
    n = 0
    for ano in os.listdir(BANCO):
        d = os.path.join(BANCO, ano)
        if not (os.path.isdir(d) and ano.isdigit()):
            continue
        for nome in os.listdir(d):
            m = RE_LOCAL.match(nome)
            if not m:
                continue
            n += 1
            a, mo, dia, num = m.groups()
            rel = os.path.join(ano, nome)
            data = f'{a}-{mo}-{dia}'
            if num:
                por_num.setdefault((ano, str(int(num))), []).append(
                    (data, rel))
            por_data.setdefault(data, []).append(rel)
    return por_num, por_data, n


def cruzar():
    with io.open(CATALOGO, encoding='utf-8-sig') as f:
        cat = list(csv.DictReader(f))
    por_num, por_data, n_loc = acervo_local()
    print(f'catalogo AtoM: {len(cat)} | acervo local: {n_loc} PDFs '
          f'({sum(len(v) for v in por_num.values())} com num no nome)')

    linhas = []
    for r in cat:
        dec, motivo = '', ''
        if not r['url_pdf']:
            dec, motivo = 'sem_pdf', r['obs']
        elif not (r['ano'] and '1932' <= r['ano'] <= '1937'):
            dec, motivo = 'fora_escopo', f"ano={r['ano'] or '?'}"
        else:
            cand = por_num.get((r['ano'], r['num_be']), []) \
                if r['num_be'] else []
            if cand:
                dec, motivo = 'ja_temos', f'num: {cand[0][1]}'
            elif r['data_producao'] and por_data.get(r['data_producao']):
                dec = 'ja_temos'
                motivo = f"data: {por_data[r['data_producao']][0]}"
            else:
                dec, motivo = 'baixar', ('' if r['num_be']
                                         else 'sem num no titulo')
        linhas.append({**r, 'decisao': dec, 'motivo': motivo})

    with io.open(PLANO, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS + ['decisao', 'motivo'],
                           extrasaction='ignore')
        w.writeheader()
        w.writerows(linhas)

    from collections import Counter
    c = Counter(l['decisao'] for l in linhas)
    por_ano = Counter(l['ano'] for l in linhas if l['decisao'] == 'baixar')
    print(f'plano -> {PLANO}')
    for k in ('baixar', 'ja_temos', 'sem_pdf', 'fora_escopo'):
        print(f'  {k}: {c.get(k, 0)}')
    if por_ano:
        print('  baixar por ano: ' + ', '.join(
            f'{a}={n}' for a, n in sorted(por_ano.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--so-cruzar', action='store_true',
                    help='pula o crawl; refaz so o cruzamento')
    args = ap.parse_args()
    if not args.so_cruzar:
        crawl()
    cruzar()


if __name__ == '__main__':
    main()
