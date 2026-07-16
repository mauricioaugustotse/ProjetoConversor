# -*- coding: utf-8 -*-
r"""acv_05: reequilibrio estrutural pos-reordenacao de C:\Users\mauri\HD_Mau.

Pastas com >400 arquivos DIRETOS sao subdivididas (padrao org2_04/08):
 - se >=60% dos arquivos tem ano no nome: faixas de ano contiguas com teto
   ~300 ("2016", "2017-2019"); sem-ano permanecem na pasta-mae;
 - senao: fica plana (13/07/2026: usuario aboliu faixas alfabeticas —
   biblioteca ordena pelo nome; acv_09 desfez as existentes).
Matérias canonicas mirradas do 01 sao preservadas (uniformidade tipo\materia).
Log de reversao log_acv_reeq_<ts>.csv. Idempotente (pastas ja dentro do teto
viram no-op).
"""
import csv
import datetime
import os
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from acv_02_cruzar import lp
from acv_03_organizar import RE_FAIXA_ANO, RE_FAIXA_ALFA, ano_do, ascii_seg

ROOT = r'C:\Users\mauri\HD_Mau'
TETO = 400
ALVO = 300


def particionar_anos(itens_ano):
    """[(ano, path)] ordenado -> faixas contiguas com ~ALVO itens."""
    por_ano = Counter(a for a, _ in itens_ano)
    anos = sorted(por_ano)
    faixas = []
    ini = None
    acum = 0
    for a in anos:
        if ini is None:
            ini, acum = a, 0
        acum += por_ano[a]
        if acum >= ALVO:
            faixas.append((ini, a))
            ini = None
    if ini is not None:
        faixas.append((ini, anos[-1]))
    return faixas


def particionar_alfa(nomes):
    """nomes ordenados -> faixas por inicial (2 chars) com ~ALVO itens."""
    def chave(n):
        s = ascii_seg(n, 10).lower()
        s = re.sub(r'[^0-9a-z]', '', s) or '0'
        return s[:2].ljust(2, '0')
    ordenados = sorted(nomes, key=chave)
    faixas = []
    bloco = []
    for n in ordenados:
        bloco.append(n)
        if len(bloco) >= ALVO:
            faixas.append((chave(bloco[0]), chave(bloco[-1]), bloco))
            bloco = []
    if bloco:
        faixas.append((chave(bloco[0]), chave(bloco[-1]), bloco))
    return faixas


def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_reeq_{ts}.csv')
    movidos = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['de', 'para'])

        grandes = []
        for dp, dn, fn in os.walk(lp(ROOT)):
            if len(fn) > TETO:
                p = dp[4:] if dp.startswith('\\\\?\\') else dp
                grandes.append((p, list(fn)))
        print(f'Pastas a subdividir (> {TETO} diretos): {len(grandes)}')

        for pasta, arquivos in grandes:
            rel = os.path.relpath(pasta, ROOT)
            com_ano = []
            sem_ano = []
            for fn in arquivos:
                a = ano_do(os.path.splitext(fn)[0], '')
                (com_ano if a else sem_ano).append((a, fn))
            usa_ano = len(com_ano) >= 0.6 * len(arquivos)
            if not usa_ano:
                continue  # sem faixas alfabeticas: pasta grande fica plana
            planos = []  # (fn, subpasta)
            faixas = particionar_anos(sorted(com_ano))
            for a1, a2 in faixas:
                nome_faixa = str(a1) if a1 == a2 else f'{a1}-{a2}'
                for a, fn in com_ano:
                    if a1 <= a <= a2:
                        planos.append((fn, nome_faixa))
            # sem-ano ficam na pasta-mae
            n_sub = len({s for _, s in planos})
            if n_sub < 2:
                continue  # particao degenerada: no-op
            for fn, sub in planos:
                src = os.path.join(pasta, fn)
                dstdir = os.path.join(pasta, sub)
                dst = os.path.join(dstdir, fn)
                try:
                    os.makedirs(lp(dstdir), exist_ok=True)
                    if not os.path.exists(lp(dst)):
                        os.rename(lp(src), lp(dst))
                        log.writerow([src, dst])
                        movidos += 1
                except OSError as e:
                    log.writerow([src, f'ERRO: {e}'])
            modo = 'anos' if usa_ano else 'alfabetica'
            print(f'  {rel}: {len(arquivos)} -> {n_sub} faixas ({modo})'
                  + (f' + {len(sem_ano)} sem-ano na raiz' if usa_ano and sem_ano else ''))
    print(f'Movidos p/ faixas: {movidos:,} | log: {log_path}')


if __name__ == '__main__':
    main()
