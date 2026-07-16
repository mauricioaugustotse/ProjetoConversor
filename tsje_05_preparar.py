# -*- coding: utf-8 -*-
"""tsje_05: prepara os insumos da transcricao fiel das atas do TSJE.

Para cada ata do indice (tribunal=superior):
- intervalo de paginas no PDF do banco (do cabecalho ate o proximo cabecalho
  de ata no mesmo PDF; cap +2 paginas);
- PNGs ~200 DPI grayscale em D:\\TSJE_TRANSCRICOES\\_paginas\\<id>\\pNNN.png
  (pagina compartilhada renderizada 1 vez);
- recorte do texto OCR (apoio) em _ocr\\<ata_id>.txt;
- cruzamento com o ensaio (D:\\Atas TSJE - Transc IA - 1932-1937.zip):
  marca no_ensaio e grava _ensaio_confronto.csv com os dois sentidos.

Manifesto: D:\\TSJE_TRANSCRICOES\\manifest.csv (estado retomavel da transcricao).
Reexecucao preserva status/confianca/transcricao de manifest existente.

Uso: python tsje_05_preparar.py [--render-ano 1932] [--sem-render]
"""
import argparse
import csv
import io
import os
import re
import time
import unicodedata
import zipfile
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
INDICE = os.path.join(BASE, 'tsje_indice_atas.csv')
CATALOGO = os.path.join(BASE, 'tsje_catalogo.tsv')
RAIZ = r'D:\TSJE_TRANSCRICOES'
BANCO = r'D:\TSJE_ATAS'
CACHE_TEXTO = r'D:\TSJE_TRABALHO\tsje_cache2.tsv'
ENSAIO_ZIP = r'D:\Atas TSJE - Transc IA - 1932-1937.zip'
MANIFEST = os.path.join(RAIZ, 'manifest.csv')
CONFRONTO = os.path.join(RAIZ, '_ensaio_confronto.csv')
DPI = 200

MESES_PT = {'janeiro': 1, 'fevereiro': 2, 'marco': 3, 'abril': 4, 'maio': 5,
            'junho': 6, 'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10,
            'novembro': 11, 'dezembro': 12}


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def normalizar(texto):
    t = unicodedata.normalize('NFD', texto.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^0-9a-z]', '', t)


def unesc(t):
    out, i, n = [], 0, len(t)
    while i < n:
        c = t[i]
        if c == '\\' and i + 1 < n:
            nx = t[i + 1]
            if nx == 'n':
                out.append('\n'); i += 2; continue
            if nx == 't':
                out.append('\t'); i += 2; continue
            if nx == '\\':
                out.append('\\'); i += 2; continue
        out.append(c)
        i += 1
    return ''.join(out)


def carregar_textos():
    """relpath (origem) -> texto com '\\f' separando paginas."""
    textos = {}
    with io.open(CACHE_TEXTO, encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 6:
                textos[parts[0]] = unesc(parts[5])
    return textos


def parse_titulo_ensaio(titulo):
    """'ATA DA 6ª SESSÃO EXTRAORDINÁRIA EM 22 DE JULHO DE 1932'
    -> (data_iso, tipo, num) com campos '' quando ausentes."""
    norm = normalizar(titulo)
    m = re.search(r'(\d{1,2})de([a-z]+)de(19\d{2})', norm)
    data = ''
    if m:
        mes = MESES_PT.get(m.group(2))
        if mes:
            data = f'{int(m.group(3)):04d}-{mes:02d}-{int(m.group(1)):02d}'
    mn = re.search(r'(\d{1,3})a?sessao', norm)
    num = mn.group(1).lstrip('0') if mn else ''
    if 'extraordinaria' in norm:
        tipo = 'extraordinaria'
    elif 'ordinaria' in norm:
        tipo = 'ordinaria'
    else:
        tipo = ''
    return data, tipo, num


def titulos_do_ensaio():
    """[(titulo, data, tipo, num, ano_docx)] extraidos dos 6 docx do zip."""
    from docx import Document
    pasta = os.path.join(RAIZ, '_ensaio')
    os.makedirs(pasta, exist_ok=True)
    with zipfile.ZipFile(ENSAIO_ZIP) as zf:
        for nome in zf.namelist():
            if nome.endswith('.docx'):
                alvo = os.path.join(pasta, os.path.basename(nome))
                if not os.path.exists(alvo):
                    with zf.open(nome) as src, open(alvo, 'wb') as dst:
                        dst.write(src.read())
    saida = []
    for ano in range(1932, 1938):
        path = os.path.join(pasta, f'{ano}.docx')
        if not os.path.exists(path):
            continue
        doc = Document(path)
        for p in doc.paragraphs:
            t = p.text.strip()
            if re.match(r'atas?\s+d', t, re.I) and len(t) < 130:
                data, tipo, num = parse_titulo_ensaio(t)
                saida.append((t, data, tipo, num, str(ano)))
    return saida


def dificuldade_do_ocr(trecho):
    """Heuristica: densidade de palavras comuns de portugues no OCR."""
    palavras = re.findall(r'[a-zà-ú]{2,}', trecho.lower())
    if len(palavras) < 120:
        return 'dificil'
    comuns = {'que', 'de', 'da', 'do', 'os', 'as', 'em', 'para', 'com', 'uma',
              'não', 'nao', 'senhor', 'pelo', 'pela', 'sobre', 'foi', 'ser'}
    taxa = sum(1 for w in palavras if w in comuns) / len(palavras)
    return 'facil' if taxa >= 0.14 else 'dificil'


def montar_titulo(a):
    num = f'{a["num_ordinal"]}ª ' if a['num_ordinal'] else ''
    tipo = {'ordinaria': 'SESSÃO ORDINÁRIA', 'extraordinaria':
            'SESSÃO EXTRAORDINÁRIA', '': 'SESSÃO'}[a['tipo']]
    if a['data_sessao'] != '?':
        aa, mm, dd = a['data_sessao'].split('-')
        mes = [k for k, v in MESES_PT.items() if v == int(mm)][0].upper()
        quando = f', EM {int(dd)} DE {mes} DE {aa}'
    else:
        quando = ''
    return f'ATA DA {num}{tipo}{quando}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-ano', help='renderiza PNGs so deste ano')
    ap.add_argument('--sem-render', action='store_true')
    args = ap.parse_args()

    os.makedirs(RAIZ, exist_ok=True)
    with io.open(INDICE, encoding='utf-8-sig') as f:
        todas = list(csv.DictReader(f))
    sup = [a for a in todas if a['tribunal'] == 'superior']

    # estado anterior (retomavel)
    anterior = {}
    if os.path.exists(MANIFEST):
        with io.open(MANIFEST, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                anterior[r['ata_id']] = r

    # paginas por PDF (todas as atas, qualquer tribunal, p/ limites de recorte)
    pags_por_pdf = defaultdict(list)
    for a in todas:
        pags_por_pdf[a['arquivo_destino']].append(int(a['pagina'] or 1))
    for k in pags_por_pdf:
        pags_por_pdf[k] = sorted(set(pags_por_pdf[k]))

    textos = carregar_textos()

    npags = {}
    with io.open(CATALOGO, encoding='utf-8') as f:
        cols_cat = f.readline().rstrip('\n').split('\t')
        i_rel, i_pg = cols_cat.index('relpath'), cols_cat.index('paginas')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == len(cols_cat):
                npags[parts[i_rel]] = int(parts[i_pg] or 1)

    # confronto com o ensaio
    ensaio = titulos_do_ensaio()
    chaves_banco = {(a['data_sessao'], a['tipo'], a['num_ordinal'].lstrip('0'))
                    for a in sup}
    chaves_banco_data = {a['data_sessao'] for a in sup}
    so_ensaio = []
    chaves_ensaio = set()
    for t, data, tipo, num, ano in ensaio:
        chaves_ensaio.add((data, tipo, num))
        if ((data, tipo, num) not in chaves_banco
                and data not in chaves_banco_data):
            so_ensaio.append([t, data, tipo, num, ano])

    import fitz
    linhas = []
    render = {}
    for a in sup:
        ano = (a['data_sessao'] if a['data_sessao'] != '?'
               else a['data_boletim'])[:4]
        pag = int(a['pagina'] or 1)
        arq = a['arquivo_destino']
        pdf = os.path.join(BANCO, a['data_boletim'][:4], arq)
        pags = pags_por_pdf[arq]
        seguintes = [p for p in pags if p > pag]
        pag_fim = min(seguintes[0] if seguintes else pag + 2, pag + 2,
                      npags.get(a['origem'], pag))
        mid = re.search(r'\[([0-9a-f]+)\]', arq)
        ident = mid.group(1) if mid else re.sub(r'\W', '', arq)[:12]
        tipo1 = {'ordinaria': 'o', 'extraordinaria': 'e', '': 'x'}[a['tipo']]
        num = a['num_ordinal'].lstrip('0') or '0'
        dt = a['data_sessao'] if a['data_sessao'] != '?' else 'semdata'
        ata_id = f'{dt}-{tipo1}{num}-{ident}-p{pag}'

        # recorte OCR
        ocr_rel = os.path.join('_ocr', f'{ata_id}.txt')
        texto = textos.get(a['origem'], '')
        paginas_txt = texto.split('\f')
        trecho = '\n\f\n'.join(paginas_txt[pag - 1:pag_fim])
        os.makedirs(os.path.join(RAIZ, '_ocr'), exist_ok=True)
        with io.open(os.path.join(RAIZ, ocr_rel), 'w', encoding='utf-8') as f:
            f.write(trecho)

        pngs = []
        for p in range(pag, pag_fim + 1):
            rel = os.path.join('_paginas', ident, f'p{p:03d}.png')
            pngs.append(rel)
            render.setdefault((pdf, p, ident), rel)

        chave_e = (a['data_sessao'], a['tipo'], num if num != '0' else '')
        no_ensaio = ('1' if chave_e in chaves_ensaio
                     or (a['data_sessao'], '', '') in chaves_ensaio else '0')

        ant = anterior.get(ata_id, {})
        linhas.append({
            'ata_id': ata_id, 'ano': ano, 'data_sessao': a['data_sessao'],
            'tipo': a['tipo'], 'num': a['num_ordinal'],
            'titulo': ant.get('titulo') or montar_titulo(a),
            'be_n': a['n_boletim_be'], 'be_data': a['data_boletim'],
            'arquivo': arq, 'origem': a['origem'],
            'pag_ini': pag, 'pag_fim': pag_fim,
            'pngs': ';'.join(pngs), 'ocr_txt': ocr_rel,
            'dificuldade': dificuldade_do_ocr(trecho),
            'no_ensaio': no_ensaio,
            'status': ant.get('status', 'pendente'),
            'confianca': ant.get('confianca', ''),
            'transcricao': ant.get('transcricao',
                                   os.path.join(ano, f'{ata_id}.md')),
            'parecer': ant.get('parecer', ''),
            'fidelidade': ant.get('fidelidade', ''),
        })

    # renderizacao (por ano, se pedido)
    feitos = 0
    if not args.sem_render:
        alvo_ids = {l['ata_id']: l for l in linhas
                    if not args.render_ano or l['ano'] == args.render_ano}
        precisa = {}
        for l in alvo_ids.values():
            for p in range(int(l['pag_ini']), int(l['pag_fim']) + 1):
                pdf = os.path.join(BANCO, l['be_data'][:4], l['arquivo'])
                mid = re.search(r'\[([0-9a-f]+)\]', l['arquivo'])
                ident = mid.group(1) if mid else re.sub(r'\W', '', l['arquivo'])[:12]
                precisa[(pdf, p, ident)] = os.path.join(
                    '_paginas', ident, f'p{p:03d}.png')
        por_pdf = defaultdict(list)
        for (pdf, p, ident), rel in precisa.items():
            por_pdf[pdf].append((p, ident, rel))
        zoom = DPI / 72.0
        for pdf, pgs in por_pdf.items():
            doc = None
            for p, ident, rel in sorted(pgs):
                alvo = os.path.join(RAIZ, rel)
                if os.path.exists(lp(alvo)):
                    continue
                if doc is None:
                    doc = fitz.open(lp(pdf))
                os.makedirs(lp(os.path.dirname(alvo)), exist_ok=True)
                pix = doc[p - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                            colorspace=fitz.csGRAY)
                pix.save(lp(alvo))
                feitos += 1
            if doc is not None:
                doc.close()

    cols = list(linhas[0].keys())
    with io.open(MANIFEST, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(linhas, key=lambda l: (l['ano'], l['data_sessao'],
                                                  l['be_data'])))

    with io.open(CONFRONTO, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sentido', 'titulo', 'data', 'tipo', 'num', 'ano'])
        for row in so_ensaio:
            w.writerow(['so_no_ensaio'] + row)
        for a in sup:
            n = a['num_ordinal'].lstrip('0')
            if ((a['data_sessao'], a['tipo'], n) not in chaves_ensaio
                    and a['data_sessao'] != '?'):
                w.writerow(['so_no_banco', montar_titulo(a), a['data_sessao'],
                            a['tipo'], n, a['data_sessao'][:4]])

    pend = sum(1 for l in linhas if l['status'] == 'pendente')
    dif = sum(1 for l in linhas if l['dificuldade'] == 'dificil')
    print(f'{len(linhas)} atas no manifesto ({pend} pendentes, {dif} dificeis)'
          f' -> {MANIFEST}')
    print(f'PNGs renderizados agora: {feitos}')
    print(f'Confronto ensaio: {len(so_ensaio)} atas só no ensaio -> {CONFRONTO}')


if __name__ == '__main__':
    main()
