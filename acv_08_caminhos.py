# -*- coding: utf-8 -*-
r"""acv_08: encurta nomes cujo caminho completo excede 259 chars (MAX_PATH).

Orcamento = 259 - len(dir) - 1 - len(ext). Se ha CNJ no fim do stem, ele e
preservado e a descricao e cortada; senao corta o fim. Log reversivel.
"""
import csv
import datetime
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from acv_02_cruzar import lp
from acv_03_organizar import RE_CNJ

ROOT = r'C:\Users\mauri\HD_Mau'
MAX = 259


def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_encurta_{ts}.csv')
    longos = []
    for dp, dn, fn in os.walk(lp(ROOT)):
        base = dp[4:] if dp.startswith('\\\\?\\') else dp
        for f in fn:
            p = os.path.join(base, f)
            if len(p) > MAX:
                longos.append(p)
    ok = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['de', 'para'])
        for p in longos:
            d, nome = os.path.split(p)
            stem, ext = os.path.splitext(nome)
            orcamento = MAX - len(d) - 1 - len(ext)
            m = RE_CNJ.search(stem)
            if m and m.end() >= len(stem) - 5:  # CNJ no final: preserva
                cnj = stem[m.start():]
                desc = stem[:m.start()].rstrip(' -')
                desc = desc[:max(10, orcamento - len(cnj) - 3)].rstrip(' -')
                novo_stem = f'{desc} - {cnj}'
            else:
                novo_stem = stem[:orcamento].rstrip(' -.')
            novo = os.path.join(d, novo_stem + ext)
            n = 1
            while os.path.exists(lp(novo)) and novo.lower() != p.lower():
                n += 1
                sufixo = f' ({n})'
                novo = os.path.join(d, novo_stem[:orcamento - len(sufixo)] + sufixo + ext)
            if novo.lower() == p.lower():
                continue
            try:
                os.rename(lp(p), lp(novo))
                log.writerow([p, novo])
                ok += 1
            except OSError as e:
                log.writerow([p, f'ERRO: {e}'])
    print(f'Caminhos >259 encontrados: {len(longos)} | encurtados: {ok} | log: {log_path}')


if __name__ == '__main__':
    main()
