# -*- coding: utf-8 -*-
"""tsje_22: busca dirigida das sessoes-alvo ("82 ausentes") — o revisor.

Para cada sessao de tsje_sessoes_alvo.csv:
1. Candidatos = PDFs cuja data de publicacao (do caminho/nome) cai na janela
   [pub_min, pub_max] — CRUZA a fronteira do ano (a secao ACTAS saia com
   defasagem; janelas abertas foram estendidas pelo tsje_20).
2. Busca textual RELAXADA no OCR dos candidatos (alem da estrita ja rodada):
   - cabecalho fuzzy: numero + ruido ate 3 chars + 'sess' na mesma linha;
   - por data: 'em D de MES de AAAA' proximo de 'sessao' (datas da janela
     da sessao);
3. Cruza com a visao do sol (tsje_visao_cache.jsonl): atas formais do
   superior detectadas em paginas cegas, por numero OU data na janela.
4. Veredito em tsje_revisao_82.csv:
   ENCONTRADA_OCR / ENCONTRADA_VISAO (pdf, pagina, evidencia)
   AUSENTE_CONFIRMADA (todos os candidatos examinados sem achado; lista
     BEs da janela que nem o AtoM preserva)
   INDETERMINADA (ha paginas cegas de candidatos ainda sem visao — gera
     fila de escalada tsje_escalada_visao.csv para o tsje_21 --fila)

Uso: python tsje_22_busca_dirigida.py
"""
import csv
import io
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date

sys.stdout.reconfigure(encoding='utf-8')

TRAB = r'D:\TSJE_TRABALHO'
CACHE = os.path.join(TRAB, 'tsje_cache2.tsv')
ALVOS = os.path.join(TRAB, 'tsje_sessoes_alvo.csv')
CEGAS = os.path.join(TRAB, 'tsje_paginas_cegas.csv')
VISAO = os.path.join(TRAB, 'tsje_visao_cache.jsonl')
SAIDA = os.path.join(TRAB, 'tsje_revisao_82.csv')
ESCALADA = os.path.join(TRAB, 'tsje_escalada_visao.csv')
RAIZ = r'D:\TSJE_TRANSCRICOES'

MESES_DIR = {'JANEIRO': 1, 'FEVEREIRO': 2, 'MARÇO': 3, 'ABRIL': 4,
             'MAIO': 5, 'JUNHO': 6, 'JULHO': 7, 'AGOSTO': 8,
             'SETEMBRO': 9, 'OUTUBRO': 10, 'NOVEMBRO': 11, 'DEZEMBRO': 12}
MESES_EXT = {1: 'janeiro', 2: 'fevereiro', 3: 'marco', 4: 'abril',
             5: 'maio', 6: 'junho', 7: 'julho', 8: 'agosto',
             9: 'setembro', 10: 'outubro', 11: 'novembro', 12: 'dezembro'}


def norm(t):
    t = unicodedata.normalize('NFD', t.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^0-9a-z]', '', t)


def unesc(t):
    return t.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')


def data_do_pdf(rel):
    m = re.match(r'BEs (\d{4})[\\/]([A-ZÇ]+)[\\/](\d{2})(?=[\\/])', rel)
    if m and m.group(3) != '00':
        mes = MESES_DIR.get(m.group(2))
        if mes:
            return f'{m.group(1)}-{mes:02d}-{int(m.group(3)):02d}'
    m = re.search(r'[\\/](\d{4}-\d{2}-\d{2}) - BE', rel)
    if m:
        return m.group(1)
    m = re.match(r'BEs (\d{4})[\\/]([A-ZÇ]+)', rel)
    if m:
        mes = MESES_DIR.get(m.group(2))
        if mes:
            return f'{m.group(1)}-{mes:02d}-15'
    return ''


def carregar_pdfs():
    """rel -> (data_pub, texto, [linhas_norm por pagina])"""
    pdfs = {}
    with io.open(CACHE, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) != 6 or not re.match(r'(?:BEs|ATOM) \d{4}', p[0]):
                continue
            d = data_do_pdf(p[0])
            if d:
                pdfs[p[0]] = (d, unesc(p[5]))
    return pdfs


def paginas_do_texto(texto):
    """lista de paginas; cada pagina = lista de linhas normalizadas."""
    pags = []
    for pag in texto.split('\f'):
        pags.append([norm(l) for l in pag.split('\n')])
    return pags


def busca_relaxada(pags, num, tipo, datas_ext):
    """Procura cabecalho fuzzy OU data por extenso perto de 'sess'.
    Retorna (pagina_1based, evidencia) ou None."""
    tp = 'extraordinaria' if tipo == 'extraordinaria' else 'ordinaria'
    # fuzzy: numero (fronteira) + ate 3 chars de ruido + 'sess' + ate 12
    # chars + inicio do tipo (ord/extr)
    rx_fuzzy = re.compile(r'(?<!\d)' + str(num) +
                          r'\D{0,3}sess\w{0,12}' + tp[:4])
    rx_datas = [re.compile(re.escape(d)) for d in datas_ext]
    for i, linhas in enumerate(pags):
        for j, l in enumerate(linhas):
            if rx_fuzzy.search(l):
                janela = ''.join(linhas[j:j + 14])
                if 'presid' in janela or 'presentes' in janela:
                    return i + 1, f'fuzzy: ...{l[:80]}'
            if 'sess' in l:
                jan = ''.join(linhas[max(0, j - 2):j + 6])
                for rx in rx_datas:
                    if rx.search(jan) and ('presid' in jan
                                           or 'presentes' in jan):
                        return i + 1, f'data: ...{l[:80]}'
    return None


def datas_extenso(dmin, dmax):
    """['12demarcode1935', ...] para cada dia da janela da sessao."""
    try:
        y1, m1, d1 = map(int, dmin.split('-'))
        y2, m2, d2 = map(int, dmax.split('-'))
    except ValueError:
        return []
    a, b = date(y1, m1, d1), date(y2, m2, d2)
    out = []
    d = a
    from datetime import timedelta
    while d <= b and len(out) < 45:
        out.append(f'{d.day}de{MESES_EXT[d.month]}de{d.year}')
        d += timedelta(days=1)
    return out


def transcritas_por_num():
    """(tipo,num,ano_da_data) -> arquivo .md — para o anti-homonimo."""
    idx = {}
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
            md = re.search(r'data_sessao: (\d{4})', cab)
            if mn and md:
                t = ('extraordinaria' if mt and mt.group(1).startswith('ext')
                     else 'ordinaria')
                idx[(t, int(mn.group(1)), md.group(1))] = f'{ano}\\{nome}'
    return idx


def main():
    alvos = list(csv.DictReader(io.open(ALVOS, encoding='utf-8-sig')))
    pdfs = carregar_pdfs()
    print(f'alvos: {len(alvos)} | pdfs datados: {len(pdfs)}')

    # paginas cegas por pdf e cobertura da visao
    cegas = defaultdict(set)
    for r in csv.DictReader(io.open(CEGAS, encoding='utf-8-sig')):
        cegas[r['pdf']].add(int(r['pagina']))
    visao_cobre = set()
    visao_atas = []   # (pdf, pagina, num, tipo, data, tribunal, formal, evid)
    if os.path.exists(VISAO):
        with io.open(VISAO, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                visao_cobre.add((d['pdf'], d['pagina']))
                for a in d.get('atas') or []:
                    visao_atas.append((d['pdf'], d['pagina'], a.get('num'),
                                       a.get('tipo'), a.get('data_impressa'),
                                       a.get('tribunal'),
                                       a.get('e_ata_formal'),
                                       (a.get('evidencia') or '')[:120]))
    print(f'visao: {len(visao_cobre)} paginas cobertas, '
          f'{len(visao_atas)} atas mencionadas')

    ja = transcritas_por_num()
    textos_pag = {}   # cache de paginas normalizadas por pdf

    resultados = []
    escalada = set()
    for al in alvos:
        ano, num = al['ano'], int(al['num'])
        tipo = al['tipo']
        cands = [(rel, dt) for rel, (dt, _) in pdfs.items()
                 if al['pub_min'] <= dt <= al['pub_max']]
        cands.sort(key=lambda x: x[1])
        datas_ext = datas_extenso(al['sessao_min'], al['sessao_max'])

        achado = None
        # 1) OCR relaxado
        for rel, dt in cands:
            if rel not in textos_pag:
                textos_pag[rel] = paginas_do_texto(pdfs[rel][1])
            hit = busca_relaxada(textos_pag[rel], num, tipo, datas_ext)
            if hit:
                achado = ('ENCONTRADA_OCR', rel, hit[0], hit[1])
                break
        # 2) visao do sol
        if not achado:
            tp_v = 'extraordinaria' if tipo == 'extraordinaria' else 'ordinaria'
            for pdf, pag, vnum, vtipo, vdata, vtrib, vformal, evid in visao_atas:
                if vtrib == 'regional' or not vformal:
                    continue
                num_ok = (vnum == num and (vtipo or tp_v) == tp_v)
                data_ok = (vdata and al['sessao_min'] <= vdata
                           <= al['sessao_max'])
                if num_ok or data_ok:
                    dt = data_do_pdf(pdf)
                    if dt and al['pub_min'] <= dt <= al['pub_max']:
                        achado = ('ENCONTRADA_VISAO', pdf, pag,
                                  f'sol: {evid}')
                        break
        # 3) cobertura: paginas cegas dos candidatos sem visao?
        pendentes = set()
        if not achado:
            for rel, dt in cands:
                for pg in cegas.get(rel, ()):
                    if (rel, pg) not in visao_cobre:
                        pendentes.add((rel, pg))
        # anti-homonimo
        obs = ''
        if achado:
            for (t, n, y), arq in ja.items():
                if t == tipo and n == num and y != ano:
                    obs = f'ATENCAO homonimo transcrito: {n}a {t} de {y} ({arq})'
                    break
        if achado:
            verd, rel, pag, evid = achado
        elif pendentes:
            verd, rel, pag, evid = 'INDETERMINADA', '', '', \
                f'{len(pendentes)} paginas cegas sem visao nos candidatos'
            escalada |= pendentes
        else:
            verd, rel, pag, evid = 'AUSENTE_CONFIRMADA', '', '', \
                f'{len(cands)} BEs da janela examinados (OCR+visao)'
        resultados.append({
            'ano': ano, 'tipo': tipo, 'num': num,
            'janela_pub': f"{al['pub_min']}..{al['pub_max']}",
            'candidatos': len(cands), 'veredito': verd,
            'pdf': rel, 'pagina': pag, 'evidencia': evid, 'obs': obs})

    with io.open(SAIDA, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        w.writeheader()
        w.writerows(resultados)
    with io.open(ESCALADA, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pdf', 'pagina'])
        for rel, pg in sorted(escalada):
            w.writerow([rel, pg])

    from collections import Counter
    c = Counter(r['veredito'] for r in resultados)
    print(f'vereditos: {dict(c)}')
    print(f'-> {SAIDA}')
    print(f'escalada de visao: {len(escalada)} paginas -> {ESCALADA}')
    for r in resultados:
        if r['veredito'].startswith('ENCONTRADA'):
            print(f"  {r['ano']} {r['num']}a {r['tipo'][:5]}: "
                  f"{r['veredito']} {r['pdf'][-50:]} p{r['pagina']} "
                  f"{r['obs']}")


if __name__ == '__main__':
    main()
