# -*- coding: utf-8 -*-
"""img_05: executa o plano — MOVE os campeões para D:\\IMAGENS.

Mesmo volume (D:->D:) = os.rename instantâneo, sem reescrita de dados.
Fallback: copy2 + verificação sha1 + remoção da origem. Retomável (tabela
movidos). Log de reversão completo em log_img_move_<ts>.csv. Nunca sobrescreve:
destino ocupado é registrado como pendência.
"""
import csv
import datetime
import hashlib
import os
import shutil

from img_lib import conectar, lp

AQUI = os.path.dirname(os.path.abspath(__file__))
# origens a PRESERVAR (13/07/2026, OneDrive): copia em vez de mover
COPIAR_PREFIXOS = ('C:\\Users\\mauri\\OneDrive\\',)


def sha1_de(path):
    h = hashlib.sha1()
    with open(lp(path), 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    con = conectar()
    ja = {p for (p,) in con.execute('SELECT path FROM movidos')}
    plano = [(p, d) for p, d in
             con.execute('SELECT path, destino FROM plano WHERE destino IS NOT NULL')
             if p not in ja]
    print(f'A mover: {len(plano):,} (já movidos antes: {len(ja):,})')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(AQUI, f'log_img_move_{ts}.csv')
    ok = err = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['origem', 'destino', 'metodo', 'status'])
        for i, (src, dst) in enumerate(plano, 1):
            metodo, status = 'rename', 'ok'
            try:
                if not os.path.exists(lp(src)):
                    status = 'origem_sumiu'
                elif os.path.exists(lp(dst)):
                    status = 'destino_ocupado'
                elif src.startswith(COPIAR_PREFIXOS):
                    metodo = 'copia_preservada'
                    os.makedirs(lp(os.path.dirname(dst)), exist_ok=True)
                    shutil.copy2(lp(src), lp(dst))
                    sha_cat = con.execute('SELECT sha1 FROM arquivos WHERE path=?',
                                          (src,)).fetchone()
                    if sha_cat and sha_cat[0] and sha1_de(dst) != sha_cat[0]:
                        os.remove(lp(dst))
                        raise OSError('sha1 divergente pos-copia')
                else:
                    os.makedirs(lp(os.path.dirname(dst)), exist_ok=True)
                    try:
                        os.rename(lp(src), lp(dst))
                    except OSError:
                        metodo = 'copy+verify'
                        shutil.copy2(lp(src), lp(dst))
                        if sha1_de(src) != sha1_de(dst):
                            os.remove(lp(dst))
                            raise OSError('sha1 divergente pos-copia')
                        os.remove(lp(src))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([src, dst, metodo, status])
            if status == 'ok':
                ok += 1
                con.execute('INSERT OR REPLACE INTO movidos(path, destino, metodo, ts) '
                            'VALUES(?,?,?,?)', (src, dst, metodo, ts))
            else:
                err += 1
            if i % 1000 == 0:
                con.commit()
                flog.flush()
                print(f'  {i:,}/{len(plano):,} (erros/pendências {err})', flush=True)
    con.commit()
    con.close()
    print(f'Movidos: {ok:,} | pendências: {err:,} | log: {log_path}')


if __name__ == '__main__':
    main()
