# -*- coding: utf-8 -*-
"""tsje_20: auditoria de cobertura do acervo para a revisao das "82 ausentes".

Produz tres mapas em D:\\TSJE_TRABALHO:
1. tsje_paginas_cegas.csv  — paginas com OCR inutil (<LIMIAR chars norm.)
   em QUALQUER pdf do cache (nao so os 54 sem OCR nenhum): sao invisiveis
   para o detector textual de cabecalhos.
2. tsje_sessoes_alvo.csv   — as sessoes sem ata formal, com a janela de
   datas estimada (data da sessao interpolada pelas vizinhas transcritas;
   janela de PUBLICACAO = [data_min, data_max + FOLGA_PUB dias], cruzando
   a fronteira do ano quando aberta — pedido do usuario).
3. tsje_bes_faltantes.csv  — contraprova pela numeracao sequencial dos BEs:
   numeros presentes no catalogo AtoM x presentes no acervo local, por ano.
   BE que NEM o AtoM tem = lacuna do acervo oficial; BE que o AtoM tem e o
   acervo local nao = falha de download (re-baixar).

Uso: python tsje_20_auditoria_cobertura.py
"""
import csv
import io
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date, timedelta

sys.stdout.reconfigure(encoding='utf-8')

CACHE = r'D:\TSJE_TRABALHO\tsje_cache2.tsv'
DIAG = r'D:\TSJE_TRANSCRICOES\_diagnostico_completude.csv'
RAIZ = r'D:\TSJE_TRANSCRICOES'
BANCO = r'D:\TSJE_ATAS'
ATOM_CAT = r'D:\TSJE_TRABALHO\tsje_atom_catalogo.csv'
OUT_CEGAS = r'D:\TSJE_TRABALHO\tsje_paginas_cegas.csv'
OUT_ALVO = r'D:\TSJE_TRABALHO\tsje_sessoes_alvo.csv'
OUT_BES = r'D:\TSJE_TRABALHO\tsje_bes_faltantes.csv'

LIMIAR = 200      # chars normalizados por pagina abaixo disso = cega
FOLGA_PUB = 90    # dias apos a sessao em que a ata pode ter sido publicada


def norm(t):
    t = unicodedata.normalize('NFD', t.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^0-9a-z]', '', t)


def unesc(t):
    return t.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')


def paginas_cegas():
    linhas = []
    tot_pdf = tot_pag = 0
    with io.open(CACHE, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) != 6 or not re.match(r'(?:BEs|ATOM) \d{4}', p[0]):
                continue
            tot_pdf += 1
            texto = unesc(p[5])
            pags = texto.split('\f') if texto else []
            npag_cache = int(p[4]) if p[4].isdigit() else len(pags)
            # se o texto nao tem \f suficientes (sem OCR), todas as paginas
            # do pdf sao cegas
            for i in range(npag_cache):
                t = pags[i] if i < len(pags) else ''
                n = len(norm(t))
                tot_pag += 1
                if n < LIMIAR:
                    linhas.append((p[0], i + 1, n))
    with io.open(OUT_CEGAS, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pdf', 'pagina', 'chars_norm'])
        w.writerows(linhas)
    por_pdf = defaultdict(int)
    for rel, pag, n in linhas:
        por_pdf[rel] += 1
    print(f'[cegas] {len(linhas)} paginas cegas em {len(por_pdf)} PDFs '
          f'(acervo: {tot_pdf} PDFs / {tot_pag} paginas) -> {OUT_CEGAS}')
    return linhas


def datas_transcritas():
    """ano -> {(tipo,num): data} das atas do TSJE ja transcritas."""
    datas = defaultdict(dict)
    for ano in os.listdir(RAIZ):
        d = os.path.join(RAIZ, ano)
        if not (os.path.isdir(d) and ano.isdigit()):
            continue
        for nome in os.listdir(d):
            if not nome.endswith('.md'):
                continue
            with io.open(os.path.join(d, nome), encoding='utf-8') as f:
                cab = f.read(700)
            if 'tribunal: superior' not in cab or 'problema:' in cab:
                continue
            mn = re.search(r'\nnum: (\d+)', cab)
            mt = re.search(r'\ntipo: (\w+)', cab)
            md = re.search(r'data_sessao: (\d{4}-\d{2}-\d{2})', cab)
            if mn and md:
                y = md.group(1)[:4]
                t = 'e' if (mt and mt.group(1).startswith('extra')) else 'o'
                datas[y][(t, int(mn.group(1)))] = md.group(1)
    return datas


def sessoes_alvo():
    datas = datas_transcritas()
    aus = [r for r in csv.DictReader(io.open(DIAG, encoding='utf-8-sig'))
           if r['situacao'] in ('BOLETIM_AUSENTE_NO_ACERVO', 'RECUPERAVEL')]
    linhas = []
    for r in aus:
        ano, num = r['ano'], int(r['num'])
        t = 'e' if r['tipo'] == 'extraordinaria' else 'o'
        viz = datas.get(ano, {})
        antes = max((d for (tt, n), d in viz.items()
                     if tt == t and n < num), default='')
        depois = min((d for (tt, n), d in viz.items()
                      if tt == t and n > num), default='')
        # data provavel da SESSAO — janela EXCLUSIVE: as datas das sessoes
        # vizinhas nao entram (senao a busca por data acha a ata da vizinha)
        def _mais(d, dias):
            y, m, dd = map(int, d.split('-'))
            return (date(y, m, dd) + timedelta(days=dias)).isoformat()
        dmin = _mais(antes, 1) if antes else f'{ano}-01-01'
        dmax = _mais(depois, -1) if depois else f'{ano}-12-31'
        if dmax < dmin:
            dmax = dmin
        # janela de PUBLICACAO da ata: da data minima da sessao ate
        # dmax + FOLGA (a seccao ACTAS saia semanas depois; cruza o ano)
        y, m, dd = map(int, dmax.split('-'))
        pub_max = (date(y, m, dd) + timedelta(days=FOLGA_PUB)).isoformat()
        linhas.append({'ano': ano, 'tipo': r['tipo'], 'num': num,
                       'sessao_min': dmin, 'sessao_max': dmax,
                       'pub_min': dmin, 'pub_max': pub_max,
                       'janela_aberta': 'sim' if not (antes and depois)
                       else 'nao'})
    with io.open(OUT_ALVO, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f'[alvo] {len(linhas)} sessoes-alvo com janela -> {OUT_ALVO}')
    return linhas


def bes_faltantes():
    """Numeracao sequencial dos BEs por ano: AtoM x acervo local."""
    atom = defaultdict(set)
    if os.path.exists(ATOM_CAT):
        for r in csv.DictReader(io.open(ATOM_CAT, encoding='utf-8-sig')):
            ano = (r.get('ano') or '')[:4]
            mnum = re.search(r'\d+', r.get('num_be') or '')
            if ano.isdigit() and mnum:
                atom[ano].add(int(mnum.group()))
    local = defaultdict(set)
    for ano in os.listdir(BANCO):
        d = os.path.join(BANCO, ano)
        if not (os.path.isdir(d) and ano.isdigit()):
            continue
        for nome in os.listdir(d):
            m = re.search(r'BE n(\d+)', nome)
            if m:
                local[ano].add(int(m.group(1)))
    linhas = []
    for ano in sorted(set(atom) | set(local)):
        universo = atom[ano] | local[ano]
        if not universo:
            continue
        seq = set(range(1, max(universo) + 1))
        nem_atom = sorted(seq - atom[ano] - local[ano])
        so_atom = sorted(atom[ano] - local[ano])
        so_local = sorted(local[ano] - atom[ano])
        linhas.append({'ano': ano, 'max_be': max(universo),
                       'no_atom': len(atom[ano]), 'no_local': len(local[ano]),
                       'nem_atom_nem_local': ' '.join(map(str, nem_atom)),
                       'so_no_atom_rebaixar': ' '.join(map(str, so_atom)),
                       'so_no_local': ' '.join(map(str, so_local))})
    with io.open(OUT_BES, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f'[BEs] contraprova por numeracao -> {OUT_BES}')
    for ln in linhas:
        na = ln['nem_atom_nem_local'].split()
        sa = ln['so_no_atom_rebaixar'].split()
        print(f"  {ln['ano']}: max BE n{ln['max_be']} | atom {ln['no_atom']} "
              f"| local {ln['no_local']} | nem_atom_nem_local: {len(na)} "
              f"| so_no_atom (re-baixar): {len(sa)}")


def main():
    paginas_cegas()
    sessoes_alvo()
    bes_faltantes()


if __name__ == '__main__':
    main()
