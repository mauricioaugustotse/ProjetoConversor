# -*- coding: utf-8 -*-
"""tsje_11b: integra os BEs baixados do AtoM (D:\\TSJE_ATAS\\<ano>\\*[atom *].pdf)
ao cache de texto OCR tsje_cache2.tsv, no mesmo formato do tsje_01
(rel \\t size \\t mtime \\t sha1 \\t paginas \\t texto com \\f por pagina).

A chave rel usa o prefixo "ATOM <ano>\\<arquivo>" — nao colide com os
"BEs <ano>\\..." dos zips; quem consome o cache resolve o caminho fisico:
  BEs ...  -> D:\\TSJE_TRABALHO\\extraido\\<rel>
  ATOM ... -> D:\\TSJE_ATAS\\<resto>

Idempotente: rel ja presente no cache e pulado.

Uso: python tsje_11b_cache_atom.py
"""
import hashlib
import io
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import fitz

sys.stdout.reconfigure(encoding='utf-8')

BANCO = r'D:\TSJE_ATAS'
CACHE = r'D:\TSJE_TRABALHO\tsje_cache2.tsv'

_lock = threading.Lock()


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def esc(t):
    return t.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n')


def sha1_arquivo(path):
    h = hashlib.sha1()
    with open(lp(path), 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()


def texto_pdf(path):
    doc = fitz.open(lp(path))
    partes = []
    for i, pg in enumerate(doc):
        if i:
            partes.append('\f')
        partes.append(pg.get_text())
    n = doc.page_count
    doc.close()
    return '\n'.join(partes), n


def main():
    ja = set()
    if os.path.exists(CACHE):
        with io.open(CACHE, encoding='utf-8', errors='replace') as f:
            for line in f:
                ja.add(line.split('\t', 1)[0])

    alvos = []
    for ano in sorted(os.listdir(BANCO)):
        d = os.path.join(BANCO, ano)
        if not (os.path.isdir(d) and ano.isdigit()):
            continue
        for nome in sorted(os.listdir(d)):
            # BEs baixados de repositorios externos (AtoM ou BDTSE); ambos
            # entram no cache com prefixo ATOM (o caminho fisico e o mesmo
            # BANCO\<ano>\<nome>, resolvido por caminho_fisico nos consumidores)
            if ('[atom ' in nome or '[bdtse ' in nome) \
                    and nome.lower().endswith('.pdf'):
                rel = f'ATOM {ano}\\{nome}'
                if rel not in ja:
                    alvos.append((rel, os.path.join(d, nome)))
    print(f'novos a cachear: {len(alvos)} (cache tinha {len(ja)})')

    cont = {'ok': 0, 'err': 0}

    def do(item):
        rel, path = item
        try:
            st = os.stat(lp(path))
            texto, npag = texto_pdf(path)
            sha1 = sha1_arquivo(path)
            with _lock:
                with io.open(CACHE, 'a', encoding='utf-8') as f:
                    f.write('\t'.join([rel, str(st.st_size),
                                       str(int(st.st_mtime)), sha1,
                                       str(npag), esc(texto)]) + '\n')
                cont['ok'] += 1
                if cont['ok'] % 50 == 0:
                    print(f"  ... {cont['ok']}")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f'  ERRO {rel}: {str(e)[:120]}')

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(do, alvos))
    print(f"[FIM] ok={cont['ok']} erros={cont['err']} -> {CACHE}")


if __name__ == '__main__':
    main()
