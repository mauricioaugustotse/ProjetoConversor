# -*- coding: utf-8 -*-
"""tsje_30: inventaria a serie do Boletim Eleitoral na Biblioteca Digital
do TSE (BDTSE/DSpace) via o espelho de busca do Consorcio BDJur (VuFind),
que responde a curl (a BDTSE bloqueia/oscila para acesso automatizado).

Pagina a busca por titulo "boletim eleitoral", extrai (bdtse-ID, titulo),
parseia ano/numero/mes-ano do titulo, filtra 1932-1937 e cruza com o
acervo local D:\\TSJE_ATAS e o catalogo AtoM -> plano de download.

Uso: python tsje_30_bdtse_inventario.py
Saidas: D:\\TSJE_TRABALHO\\tsje_bdtse_catalogo.csv, tsje_bdtse_plano.csv
"""
import csv
import html
import io
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

TRAB = r'D:\TSJE_TRABALHO'
BANCO = r'D:\TSJE_ATAS'
ATOM_CAT = os.path.join(TRAB, 'tsje_atom_catalogo.csv')
CATALOGO = os.path.join(TRAB, 'tsje_bdtse_catalogo.csv')
PLANO = os.path.join(TRAB, 'tsje_bdtse_plano.csv')

BASE = ('https://consorciobdjur.stj.jus.br/vufind/Search/Results'
        '?lookfor=%22boletim+eleitoral%22&type=Title&limit=100')
UA = 'Mozilla/5.0 (Windows NT 10.0; pesquisa historica TSJE)'

MESES = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
         'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}


def baixar(url):
    for tent in range(4):
        r = subprocess.run(['curl', '-s', '-m', '120', '-A', UA, url],
                           capture_output=True)
        if r.returncode == 0 and len(r.stdout) > 5000:
            return r.stdout.decode('utf-8', errors='replace')
        time.sleep(3 * (tent + 1))
    return ''


def parse_pagina(t):
    """[(bdtse_id, titulo_limpo)] dos anchors class="title"."""
    out = []
    for m in re.finditer(
            r'Record/([^"]*?bdtse-\d+)"\s+class="title">\s*(.*?)</a>',
            t, re.S):
        rid = m.group(1).split(':')[-1]
        tit = re.sub(r'<[^>]+>', '', m.group(2))
        tit = html.unescape(re.sub(r'\s+', ' ', tit)).strip()
        out.append((rid, tit))
    return out


RE_TIT = re.compile(
    r'ano?n?o?\s+(\d{1,2})\s*,?\s*n\.?\s*(\d{1,3})\s*\((?:(\w{3})[a-z.]*\s*)?'
    r'(\d{4})\)', re.I)


def parse_titulo(tit):
    """'Boletim eleitoral: ano 4, n. 115 (set. 1935)' ->
    (ano_serie, num, ano_civil, mes)."""
    m = RE_TIT.search(tit)
    if not m:
        return None
    mes = MESES.get((m.group(3) or '').lower()[:3])
    return int(m.group(1)), int(m.group(2)), int(m.group(4)), mes


def total_declarado(t):
    m = re.search(r'de\s+<strong>\s*([\d.,]+)\s*</strong>', t) \
        or re.search(r'de\s+([\d.,]+)\s+resultado', t)
    return int(re.sub(r'[.,]', '', m.group(1))) if m else None


def main():
    vistos = {}
    # uma busca POR ANO CIVIL (o sort global re-embaralha empates de titulo
    # entre paginas e perde registros); parada pelo total declarado
    for ano in range(1932, 1938):
        url_ano = ('https://consorciobdjur.stj.jus.br/vufind/Search/Results'
                   f'?lookfor=%22boletim+eleitoral%22+%28{ano}%29'
                   '&type=AllFields&limit=20')
        t = baixar(url_ano)
        tot = total_declarado(t) or 0
        paginas = (tot + 19) // 20 + 1
        print(f'{ano}: {tot} resultados declarados ({paginas} paginas)')
        novos_ano = 0
        for pg in range(1, min(paginas, 40) + 1):
            tp = t if pg == 1 else baixar(f'{url_ano}&page={pg}')
            for rid, tit in parse_pagina(tp):
                if rid not in vistos:
                    vistos[rid] = tit
                    novos_ano += 1
            time.sleep(0.8)
        print(f'  novos de {ano}: {novos_ano} (acumulado {len(vistos)})')

    linhas = []
    for rid, tit in sorted(vistos.items()):
        p = parse_titulo(tit)
        linhas.append({
            'bdtse_id': rid, 'titulo': tit,
            'ano_serie': p[0] if p else '', 'num': p[1] if p else '',
            'ano_civil': p[2] if p else '', 'mes': p[3] if p else '',
            'handle': f'https://bibliotecadigital.tse.jus.br/xmlui/handle/'
                      f'bdtse/{rid.split("-")[-1]}'})
    with io.open(CATALOGO, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    alvo = [l for l in linhas if l['ano_civil']
            and 1932 <= l['ano_civil'] <= 1937]
    print(f'catalogo: {len(linhas)} itens ({len(alvo)} de 1932-1937) '
          f'-> {CATALOGO}')

    # acervo local por (ano, num)
    local = defaultdict(set)
    for ano in os.listdir(BANCO):
        d = os.path.join(BANCO, ano)
        if not (os.path.isdir(d) and ano.isdigit()):
            continue
        for nome in os.listdir(d):
            m = re.search(r'BE n0*(\d+)', nome)
            if m:
                local[int(ano)].add(int(m.group(1)))
    atom = defaultdict(set)
    for r in csv.DictReader(io.open(ATOM_CAT, encoding='utf-8-sig')):
        mnum = re.search(r'\d+', r['num_be'] or '')
        if (r['ano'] or '')[:4].isdigit() and mnum:
            atom[int(r['ano'][:4])].add(int(mnum.group()))

    plano = []
    for l in alvo:
        ano, num = l['ano_civil'], l['num']
        tem_local = num in local.get(ano, set())
        tem_atom = num in atom.get(ano, set())
        decisao = 'ja_temos' if tem_local else 'baixar'
        plano.append({**l, 'tem_local': tem_local, 'tem_atom': tem_atom,
                      'decisao': decisao})
    with io.open(PLANO, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(plano[0].keys()))
        w.writeheader()
        w.writerows(plano)
    baixar_n = [p for p in plano if p['decisao'] == 'baixar']
    ineditos = [p for p in baixar_n if not p['tem_atom']]
    print(f'plano: {len(plano)} itens 1932-1937 | baixar: {len(baixar_n)} '
          f'| INEDITOS (nem no AtoM): {len(ineditos)} -> {PLANO}')
    por_ano = defaultdict(int)
    for p in ineditos:
        por_ano[p['ano_civil']] += 1
    for a in sorted(por_ano):
        print(f'  ineditos {a}: {por_ano[a]}')


if __name__ == '__main__':
    main()
