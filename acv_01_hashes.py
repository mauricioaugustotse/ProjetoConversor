# -*- coding: utf-8 -*-
"""acv_01: SHA1 binário de D:\\ACERVO e C:\\Users\\mauri\\HD_Mau p/ confronto.

Estado em acv_confronto.sqlite (tabela hashes: lado 'C'|'D'). Retomável por
(path,size,mtime). ThreadPool (IO-bound). \\?\\ p/ caminhos longos e
LiteralPath-safe (colchetes no nome não são glob aqui — Python puro).
"""
import hashlib
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, 'acv_confronto.sqlite')
LADOS = {'C': r'C:\Users\mauri\HD_Mau', 'D': r'D:\ACERVO'}
IGNORAR = {'00 - indice.md', '00 - catalogo.csv', 'desktop.ini', 'thumbs.db'}


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def sha1_de(path):
    h = hashlib.sha1()
    with open(lp(path), 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    con = sqlite3.connect(DB)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute("""CREATE TABLE IF NOT EXISTS hashes(
        path TEXT PRIMARY KEY, lado TEXT, size INTEGER, mtime REAL,
        sha1 TEXT, erro TEXT)""")
    con.execute('CREATE INDEX IF NOT EXISTS ix_sha1 ON hashes(sha1)')
    feitos = {p: (s, m) for p, s, m in
              con.execute('SELECT path, size, mtime FROM hashes')}

    pend = []
    for lado, raiz in LADOS.items():
        for root, dirs, files in os.walk(lp(raiz)):
            for fn in files:
                if fn.lower() in IGNORAR:
                    continue
                full = os.path.join(root, fn)
                p = full[4:] if full.startswith('\\\\?\\') else full
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                prev = feitos.get(p)
                if prev and prev[0] == st.st_size and abs(prev[1] - st.st_mtime) < 2:
                    continue
                pend.append((p, lado, st.st_size, st.st_mtime))
    print(f'A hashear: {len(pend):,} (já no cache: {len(feitos):,})', flush=True)

    def worker(item):
        p, lado, size, mtime = item
        try:
            return (p, lado, size, mtime, sha1_de(p), None)
        except OSError as e:
            return (p, lado, size, mtime, None, str(e))

    t0 = time.time()
    lote = []
    n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for row in ex.map(worker, pend):
            lote.append(row)
            n += 1
            if len(lote) >= 500:
                con.executemany('INSERT OR REPLACE INTO hashes VALUES(?,?,?,?,?,?)', lote)
                con.commit()
                lote.clear()
            if n % 5000 == 0:
                taxa = n / (time.time() - t0)
                print(f'  {n:,}/{len(pend):,} ({taxa:.0f}/s)', flush=True)
    if lote:
        con.executemany('INSERT OR REPLACE INTO hashes VALUES(?,?,?,?,?,?)', lote)
        con.commit()
    tot = {l: c for l, c in con.execute('SELECT lado, COUNT(*) FROM hashes GROUP BY lado')}
    err = con.execute('SELECT COUNT(*) FROM hashes WHERE erro IS NOT NULL').fetchone()[0]
    print(f"Concluído em {(time.time()-t0)/60:.0f} min. C: {tot.get('C',0):,} | "
          f"D: {tot.get('D',0):,} | erros {err}")
    con.close()


if __name__ == '__main__':
    main()
