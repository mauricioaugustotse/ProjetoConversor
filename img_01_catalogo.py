# -*- coding: utf-8 -*-
"""img_01: inventário + hashes de todas as imagens dos dumps do D:\\.

Arquitetura à prova de arquivo patológico (decodificador do PIL pode travar
indefinidamente — já aconteceu nas ~48k): um SUPERVISOR divide os pendentes em
fatias de 1.000 e roda cada fatia num subprocess com timeout. Fatia que estoura
é morta (taskkill /T) e reprocessada item a item com timeout individual; o item
que travar é gravado com erro='travado' e o pipeline segue.

Uso:
    python img_01_catalogo.py                      # supervisor (todos os dumps)
    python img_01_catalogo.py "D:\\Takeout ..."     # supervisor (raízes escolhidas)
    python img_01_catalogo.py --fatia lote.json    # interno: processa uma fatia
    python img_01_catalogo.py --fatia lote.json --serial   # interno: sem pool
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

from img_lib import conectar, dumps_origem, iter_imagens, processar_arquivo

COLS = ('path', 'dump', 'album', 'size', 'mtime', 'ext', 'width', 'height',
        'sha1', 'pixmd5', 'dhash', 'dhash_pop', 'exif_dto', 'exif_dtd',
        'exif_dt', 'exif_make', 'exif_model', 'gps', 'n_tags', 'erro')
SQL = f"INSERT OR REPLACE INTO arquivos({','.join(COLS)}) VALUES({','.join('?' * len(COLS))})"

FATIA = 1000
TIMEOUT_ITEM = 90


def timeout_fatia(itens):
    """900s base + folga p/ vídeos grandes (piso de 15 MB/s de leitura)."""
    total = sum(i[2] for i in itens)
    return 900 + int(total / (15 * 1024 * 1024))


def gravar(con, rows):
    con.executemany(SQL, [tuple(r[c] for c in COLS) for r in rows])
    con.commit()


def modo_fatia(lote_json, serial):
    with open(lote_json, encoding='utf-8') as f:
        itens = [tuple(x) for x in json.load(f)]
    con = conectar()
    feitos = {p for (p,) in con.execute(
        f"SELECT path FROM arquivos WHERE path IN ({','.join('?' * len(itens))})",
        [i[0] for i in itens])} if itens else set()
    pend = [i for i in itens if i[0] not in feitos]
    if not pend:
        con.close()
        return
    lote = []
    if serial:
        for item in pend:
            lote.append(processar_arquivo(item))
    else:
        with ProcessPoolExecutor(max_workers=6, max_tasks_per_child=250) as pool:
            for row in pool.map(processar_arquivo, pend, chunksize=8):
                lote.append(row)
                if len(lote) >= 200:
                    gravar(con, lote)
                    lote.clear()
    if lote:
        gravar(con, lote)
    con.close()


def rodar_filho(args, timeout):
    """Roda subprocess com kill de ÁRVORE no timeout (workers do pool inclusos)."""
    proc = subprocess.Popen([sys.executable, '-X', 'utf8', os.path.abspath(__file__)] + args,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)],
                       capture_output=True)
        proc.wait()
        return 'timeout'


def marcar_travado(con, item):
    path, dump, size, mtime = item
    row = dict.fromkeys(COLS)
    row.update(path=path, dump=dump, album=os.path.basename(os.path.dirname(path)),
               size=size, mtime=mtime, ext=os.path.splitext(path)[1].lower(),
               gps=0, n_tags=0, erro='travado')
    gravar(con, [row])


def modo_supervisor(raizes):
    print('Raizes:')
    for r in raizes:
        print('  ', r)
    con = conectar()
    feitos = {p: (s, m) for p, s, m in
              con.execute('SELECT path, size, mtime FROM arquivos')}
    print(f'Ja catalogados: {len(feitos):,}')
    pendentes = []
    for path, dump, size, mtime in iter_imagens(raizes):
        prev = feitos.get(path)
        if prev and prev[0] == size and abs(prev[1] - mtime) < 2:
            continue
        pendentes.append((path, dump, size, mtime))
    print(f'A processar: {len(pendentes):,}', flush=True)
    if not pendentes:
        con.close()
        return

    tmpdir = tempfile.mkdtemp(prefix='img01_')
    t0 = time.time()
    travados = 0
    for f0 in range(0, len(pendentes), FATIA):
        fatia = pendentes[f0:f0 + FATIA]
        lote_json = os.path.join(tmpdir, f'lote_{f0}.json')
        with open(lote_json, 'w', encoding='utf-8') as f:
            json.dump(fatia, f)
        rc = rodar_filho(['--fatia', lote_json], timeout_fatia(fatia))
        if rc != 0:
            print(f'  fatia {f0}: {"TIMEOUT" if rc == "timeout" else f"exit {rc}"} '
                  f'-> item a item', flush=True)
            for item in fatia:
                ja = con.execute('SELECT 1 FROM arquivos WHERE path=?',
                                 (item[0],)).fetchone()
                if ja:
                    continue
                with open(lote_json, 'w', encoding='utf-8') as f:
                    json.dump([item], f)
                rci = rodar_filho(['--fatia', lote_json, '--serial'], TIMEOUT_ITEM)
                if rci != 0:
                    marcar_travado(con, item)
                    travados += 1
                    print(f'    TRAVADO: {item[0]}', flush=True)
        os.remove(lote_json)
        n = min(f0 + FATIA, len(pendentes))
        taxa = n / (time.time() - t0)
        falta = (len(pendentes) - n) / taxa / 3600
        print(f'  {n:,}/{len(pendentes):,}  ({taxa:.0f}/s, ~{falta:.1f}h restantes, '
              f'travados {travados})', flush=True)
    total = con.execute('SELECT COUNT(*) FROM arquivos').fetchone()[0]
    erros = con.execute('SELECT COUNT(*) FROM arquivos WHERE erro IS NOT NULL').fetchone()[0]
    con.close()
    print(f'Concluido em {(time.time()-t0)/3600:.1f}h. Catalogo: {total:,} '
          f'({erros:,} com erro, {travados} travados)')


def main():
    args = sys.argv[1:]
    if args and args[0] == '--fatia':
        modo_fatia(args[1], '--serial' in args)
        return
    modo_supervisor(args or dumps_origem())


if __name__ == '__main__':
    main()
