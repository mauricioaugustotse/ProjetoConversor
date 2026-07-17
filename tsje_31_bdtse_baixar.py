# -*- coding: utf-8 -*-
"""tsje_31: baixa da Biblioteca Digital do TSE (DSpace) os boletins da
fila dirigida (tsje_bdtse_fila.csv: o inedito 115/1935 + resgates de
legibilidade), via Playwright/Chromium (o host bloqueia clientes nao-
navegador) em cadencia respeitosa (30s+ entre requisicoes).

Destino: D:\\TSJE_ATAS\\<ano>\\AAAA-MM-DD - BE nNNN [bdtse NNNN].pdf
(a data vem da pagina do item; fallback AAAA-MM-01 do titulo).

Uso: python tsje_31_bdtse_baixar.py
Log: D:\\TSJE_TRABALHO\\log_tsje_bdtse_<ts>.csv
"""
import csv
import hashlib
import io
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

TRAB = r'D:\TSJE_TRABALHO'
BANCO = r'D:\TSJE_ATAS'
FILA = os.path.join(TRAB, 'tsje_bdtse_fila.csv')
PAUSA = 30
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36')


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def sha1s_do_banco():
    """sha1 dos PDFs locais (para dedup byte a byte)."""
    cache = os.path.join(TRAB, 'tsje_cache2.tsv')
    s = set()
    with io.open(cache, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 6:
                s.add(p[3])
    return s


def main():
    fila = list(csv.DictReader(io.open(FILA, encoding='utf-8-sig')))
    print(f'fila dirigida: {len(fila)} itens')
    shas = sha1s_do_banco()
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_p = os.path.join(TRAB, f'log_tsje_bdtse_{ts}.csv')
    log = io.open(log_p, 'w', encoding='utf-8-sig', newline='')
    w = csv.writer(log)
    w.writerow(['ano', 'num', 'motivo', 'handle', 'resultado', 'destino'])

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(user_agent=UA, locale='pt-BR',
                            accept_downloads=True)
        pg = ctx.new_page()
        for item in fila:
            ano, num = item['ano'], item['num']
            handle = item['handle'].replace('/xmlui/', '/')
            res, destino = '', ''
            try:
                r = pg.goto(handle, timeout=120000,
                            wait_until='domcontentloaded')
                html = pg.content()
                # data do item (dc.date: AAAA-MM-DD) se exposta
                md = re.search(r'(\d{4}-\d{2}-\d{2})', html)
                data = md.group(1) if md and md.group(1)[:4] == ano \
                    else f'{ano}-01-01'
                mb = re.search(r'href="([^"]*bitstream[^"]*\.pdf[^"]*)"',
                               html, re.I)
                if not mb:
                    res = f'sem_bitstream (HTTP {r.status if r else "?"})'
                else:
                    url_pdf = mb.group(1)
                    if url_pdf.startswith('/'):
                        url_pdf = ('https://bibliotecadigital.tse.jus.br'
                                   + url_pdf)
                    bid = handle.rstrip('/').split('/')[-1]
                    nome = f'{data} - BE n{num} [bdtse {bid}].pdf'
                    alvo = os.path.join(BANCO, ano, nome)
                    resp = pg.request.get(url_pdf, timeout=180000)
                    corpo = resp.body()
                    if resp.status == 200 and corpo[:4] == b'%PDF':
                        h = hashlib.sha1(corpo).hexdigest()
                        if h in shas:
                            res = 'duplicata_sha1'
                        else:
                            os.makedirs(os.path.dirname(alvo),
                                        exist_ok=True)
                            with open(lp(alvo), 'wb') as f:
                                f.write(corpo)
                            res, destino = 'ok', alvo
                    else:
                        res = f'download_falhou ({resp.status})'
            except Exception as e:
                res = f'erro: {str(e)[:80]}'
            print(f'  {ano} BE n{num}: {res}')
            w.writerow([ano, num, item['motivo'], handle, res, destino])
            log.flush()
            time.sleep(PAUSA)
        b.close()
    log.close()
    print(f'log -> {log_p}')


if __name__ == '__main__':
    main()
