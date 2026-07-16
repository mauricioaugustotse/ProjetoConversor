# -*- coding: utf-8 -*-
"""tsje_03: executa o tsje_plano.csv copiando os PDFs incluidos para D:\\TSJE_ATAS.

Sem argumento: dry-run (contagens + amostra do que sera copiado).
Com --exec:    copia (copy2 + verificacao sha1) e grava log_tsje_<ts>.csv.
Nada e apagado; os extraidos permanecem em D:\\TSJE_TRABALHO.

Uso: python tsje_03_executar.py [--exec]
"""
import csv
import hashlib
import io
import os
import shutil
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PLANO = os.path.join(BASE, 'tsje_plano.csv')
ORIGEM = r'D:\TSJE_TRABALHO\extraido'


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def sha1_de(path):
    h = hashlib.sha1()
    with open(lp(path), 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    exec_ = '--exec' in sys.argv
    with io.open(PLANO, encoding='utf-8-sig') as f:
        regs = list(csv.DictReader(f))
    incluir = [r for r in regs if r['acao'] == 'incluir']
    if not incluir:
        print('Nenhuma linha "incluir" no plano.')
        sys.exit(1)

    if not exec_:
        print(f'DRY-RUN: {len(incluir)} PDFs seriam copiados para D:\\TSJE_ATAS')
        for r in incluir[:10]:
            print(f'  {r["relpath"]}  ->  {r["destino"]}')
        print('  ...')
        print('Rode com --exec para copiar.')
        return

    ts = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_tsje_{ts}.csv')
    ok = falha = 0
    with io.open(log_path, 'w', encoding='utf-8-sig', newline='') as flog:
        wlog = csv.writer(flog)
        wlog.writerow(['origem', 'destino', 'sha1_esperado', 'status'])
        for i, r in enumerate(incluir, 1):
            src = os.path.join(ORIGEM, r['relpath'])
            dst = r['destino']
            status = 'ok'
            try:
                os.makedirs(lp(os.path.dirname(dst)), exist_ok=True)
                if not (os.path.exists(lp(dst))
                        and sha1_de(dst) == r['sha1']):
                    shutil.copy2(lp(src), lp(dst))
                    if sha1_de(dst) != r['sha1']:
                        status = 'sha1_divergente'
            except Exception as e:
                status = f'erro:{type(e).__name__}'
            if status == 'ok':
                ok += 1
            else:
                falha += 1
            wlog.writerow([r['relpath'], dst, r['sha1'], status])
            if i % 100 == 0:
                print(f'  {i}/{len(incluir)}')
    print(f'{ok} copiados/verificados, {falha} falhas -> log em {log_path}')


if __name__ == '__main__':
    main()
