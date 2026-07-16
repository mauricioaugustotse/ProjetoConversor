# -*- coding: utf-8 -*-
"""tsje_11: baixa do AtoM os boletins marcados `baixar` no tsje_atom_plano.csv
para D:\\TSJE_ATAS\\<ano>\\ no padrao de nome do banco.

Nome: `AAAA-MM-DD - BE nNNN [atom <hash8>].pdf` — data de producao do AtoM;
<hash8> = 8 primeiros hex do diretorio do upload (identificador estavel da
origem). Validacao: fitz abre + n de paginas > 0. Dedup: sha1 contra o banco
(se o PDF do AtoM for byte a byte igual a um ja no acervo, nao duplica).

Sem --exec: so mostra o que faria. Log: log_tsje_atom_download_<ts>.csv em
D:\\TSJE_TRABALHO. Retomavel: alvo ja existente e valido e pulado.

Uso: python tsje_11_atom_baixar.py [--exec] [--limite N]
"""
import argparse
import csv
import hashlib
import io
import os
import re
import sys
import time

import fitz
import requests

sys.stdout.reconfigure(encoding='utf-8')

BANCO = r'D:\TSJE_ATAS'
TRAB = r'D:\TSJE_TRABALHO'
PLANO = os.path.join(TRAB, 'tsje_atom_plano.csv')
UA = {'User-Agent': 'Mozilla/5.0 (pesquisa historica; contato via TSE)'}
PAUSA = 1.0


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def sha1_arquivo(path):
    h = hashlib.sha1()
    with open(lp(path), 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()


def sha1_banco():
    """sha1 -> caminho relativo, de todos os PDFs ja no banco."""
    print('calculando sha1 do banco local...')
    vistos = {}
    for ano in os.listdir(BANCO):
        d = os.path.join(BANCO, ano)
        if not (os.path.isdir(d) and ano.isdigit()):
            continue
        for nome in os.listdir(d):
            if nome.lower().endswith('.pdf'):
                p = os.path.join(d, nome)
                vistos[sha1_arquivo(p)] = os.path.join(ano, nome)
    print(f'  {len(vistos)} PDFs no banco')
    return vistos


def nome_destino(reg):
    data = reg['data_producao'] or f"{reg['ano']}-01-01"
    m = re.search(r'/uploads/r/[^/]+/./././([0-9a-f]{8})', reg['url_pdf'])
    h8 = m.group(1) if m else 'x' * 8
    num = reg['num_be'] or '0'
    return f'{data} - BE n{num} [atom {h8}].pdf'


def baixar(url, destino, tries=5):
    tmp = destino + '.part'
    for i in range(tries):
        try:
            with requests.get(url, headers=UA, timeout=180, stream=True) as r:
                if r.status_code != 200:
                    raise RuntimeError(f'HTTP {r.status_code}')
                with open(lp(tmp), 'wb') as f:
                    for bloco in r.iter_content(1 << 20):
                        f.write(bloco)
            os.replace(lp(tmp), lp(destino))
            return
        except Exception:
            if os.path.exists(lp(tmp)):
                os.remove(lp(tmp))
            if i == tries - 1:
                raise
            time.sleep(8 * (i + 1))


def pdf_valido(path):
    try:
        doc = fitz.open(lp(path))
        n = doc.page_count
        doc.close()
        return n > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exec', dest='exec_', action='store_true')
    ap.add_argument('--limite', type=int, default=0)
    args = ap.parse_args()

    with io.open(PLANO, encoding='utf-8-sig') as f:
        plano = [r for r in csv.DictReader(f) if r['decisao'] == 'baixar']
    if args.limite:
        plano = plano[:args.limite]
    print(f'{len(plano)} boletins a baixar')
    if not args.exec_:
        for r in plano[:20]:
            print(f"  {r['ano']}\\{nome_destino(r)}  <- {r['arquivo_pdf']}")
        if len(plano) > 20:
            print(f'  ... e mais {len(plano) - 20}')
        print('(sem --exec: nada foi baixado)')
        return

    vistos = sha1_banco()
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(TRAB, f'log_tsje_atom_download_{ts}.csv')
    ok = dup = err = pul = 0
    with io.open(log_path, 'w', encoding='utf-8-sig', newline='') as lg:
        w = csv.writer(lg)
        w.writerow(['slug', 'destino', 'resultado', 'detalhe'])
        for i, r in enumerate(plano, 1):
            ano = r['ano']
            destino = os.path.join(BANCO, ano, nome_destino(r))
            rel = os.path.join(ano, nome_destino(r))
            if os.path.exists(lp(destino)) and pdf_valido(destino):
                pul += 1
                w.writerow([r['slug'], rel, 'ja_baixado', ''])
                continue
            os.makedirs(os.path.join(BANCO, ano), exist_ok=True)
            try:
                baixar(r['url_pdf'], destino)
                if not pdf_valido(destino):
                    os.remove(lp(destino))
                    raise RuntimeError('pdf invalido apos download')
                h = sha1_arquivo(destino)
                if h in vistos:
                    os.remove(lp(destino))
                    dup += 1
                    w.writerow([r['slug'], rel, 'duplicata_sha1', vistos[h]])
                else:
                    vistos[h] = rel
                    ok += 1
                    w.writerow([r['slug'], rel, 'ok', ''])
            except Exception as e:
                err += 1
                w.writerow([r['slug'], rel, 'ERRO', str(e)])
            if i % 10 == 0:
                lg.flush()
                print(f'  {i}/{len(plano)} (ok={ok} dup={dup} '
                      f'pulados={pul} erros={err})')
            time.sleep(PAUSA)
    print(f'FIM: ok={ok} duplicatas={dup} ja_baixados={pul} erros={err}')
    print(f'log -> {log_path}')


if __name__ == '__main__':
    main()
