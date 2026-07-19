# -*- coding: utf-8 -*-
"""img_18: ANO-00 -> 0000 SEM DATA + reorganização por evento/classe/ano.

--anos00-plano : varre D:\\MÍDIA\\20??\\20??-00, resolve o álbum de origem no
                 catálogo e grava img_semdata_anos00.csv (arquivo -> destino)
--anos00-exec  : executa o CSV (os.rename, movidos.destino, log), remove
                 as pastas ANO-00 esvaziadas
--reorg-plano  : lista pastas de 1º nível de 0000 SEM DATA e sugere grupo
                 (EVENTOS / PESSOAS E ALBUNS / TRABALHO E ESTUDOS / VIDEOS /
                 FOTOS AVULSAS / MANTER) em img_semdata_reorg.csv — revisável
--reorg-exec   : move cada PASTA para dentro do grupo do CSV (rename de topo)
                 e atualiza movidos.destino por prefixo
--vazias       : remove pastas vazias sob D:\\MÍDIA (bottom-up)
"""
import csv
import datetime
import glob
import os
import re
import sys
import unicodedata

from img_lib import DESTINO, conectar, lp

AQUI = os.path.dirname(os.path.abspath(__file__))
SEM_DATA = os.path.join(DESTINO, '0000 SEM DATA')
CSV_ANOS = os.path.join(AQUI, 'img_semdata_anos00.csv')
CSV_REORG = os.path.join(AQUI, 'img_semdata_reorg.csv')
IGNORAR = {'desktop.ini', 'thumbs.db'}

# álbum de origem (lower, sem acento) -> subpasta de EVENTOS ou PESSOAS E ALBUNS
ALBUM_EVENTO = {
    'buffet': 'EVENTOS\\Casamento do Maurício',
    'balada': 'EVENTOS\\Casamento do Maurício',
    'jardim': 'EVENTOS\\Casamento do Maurício',
    'dia da noiva': 'EVENTOS\\Casamento do Maurício',
    'dia do noivo': 'EVENTOS\\Casamento do Maurício',
    'aniversario do toninho dib': 'EVENTOS\\Aniversário do Toninho Dib',
    'ano novo': 'EVENTOS\\Ano Novo',
    'festa junina': 'EVENTOS\\Festa Junina',
    'trote': 'EVENTOS\\Trote',
    'itajuba agost-set': 'EVENTOS\\Itajubá',
    'a galera': 'PESSOAS E ALBUNS\\A galera',
    'fotos do orkut': 'PESSOAS E ALBUNS\\Fotos do Orkut',
    'galaxy s8+': 'PESSOAS E ALBUNS\\Galaxy S8+',
    'celular mario': 'PESSOAS E ALBUNS\\Celular Mário',
}

GRUPOS_RE = [
    ('MANTER', r'^(telas|web|documentos|eventos|pessoas e albuns|'
               r'trabalho e estudos|videos|fotos avulsas)$'),
    ('EVENTOS', r'casamento|cola[cç][aã]o|despedida|bodas|baile|formatura|'
                r'solenidade|espcex|aniversari|festa|ano novo|trote|'
                r'pico da bandeira|cirurgia|reforma|rj_bh'),
    ('TRABALHO E ESTUDOS', r'^rp \d|curriculo|notion|termo de refer|precatorio|'
                           r'processo|cartorio|siade|^aft$|certidao|zelo|'
                           r'materiais de revis|toefl|prova usp|^13\.'),
    ('VIDEOS', r'^videos?($|_)|sample videos|sample music'),
    ('PESSOAS E ALBUNS', r'galera|familia|fotos minhas|orkut|galaxy|mario|'
                         r'marcos|gen duarte|^eu \+|fotos extras|fotos 3x4|'
                         r'fotos do|colacao'),
    ('FOTOS AVULSAS', r'diversas|outras|anteriores|banco onedrive|exports?$|'
                      r'extracao|^s2$|icone|carros|cobalt|^jun \d|^\d'),
]


def sem_acento(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()


def anos00_plano():
    con = conectar()
    origem_de = {d: p for p, d in con.execute('SELECT path, destino FROM movidos')}
    album_de = {p: a for p, a in con.execute('SELECT path, album FROM arquivos')}
    con.close()
    linhas = []
    for pasta in sorted(glob.glob(os.path.join(DESTINO, '20??', '20??-00'))):
        ano = os.path.basename(pasta)[:4]
        for root, _, files in os.walk(lp(pasta)):
            for fn in files:
                if fn.lower() in IGNORAR:
                    continue
                atual = os.path.join(root, fn)
                atual = atual[4:] if atual.startswith('\\\\?\\') else atual
                orig = origem_de.get(atual)
                album = album_de.get(orig, '') if orig else ''
                sub = ALBUM_EVENTO.get(sem_acento(album or ''),
                                       f'FOTOS AVULSAS\\{ano}')
                novo = os.path.join(SEM_DATA, sub, fn)
                linhas.append((atual, novo, album or '',
                               'catalogo' if orig else 'fora_do_catalogo'))
    with open(CSV_ANOS, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['origem_atual', 'destino_novo', 'album_origem', 'fonte'])
        w.writerows(linhas)
    stats = {}
    for _, novo, *_ in linhas:
        k = os.path.relpath(os.path.dirname(novo), SEM_DATA)
        stats[k] = stats.get(k, 0) + 1
    print(f'ANO-00: {len(linhas):,} arquivos')
    for k, n in sorted(stats.items(), key=lambda x: -x[1]):
        print(f'  {k:45} {n:,}')
    print(CSV_ANOS)


def anos00_exec():
    con = conectar()
    origem_de = {d: p for p, d in con.execute('SELECT path, destino FROM movidos')}
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(AQUI, f'log_img_semdata_{ts}.csv')
    ok = err = 0
    with open(CSV_ANOS, encoding='utf-8-sig') as f, \
            open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        rd = csv.reader(f)
        next(rd)
        log = csv.writer(flog)
        log.writerow(['origem_atual', 'destino_novo', 'status'])
        for atual, novo, album, fonte in rd:
            status = 'ok'
            try:
                if not os.path.exists(lp(atual)):
                    status = 'origem_sumiu'
                elif os.path.exists(lp(novo)):
                    status = 'destino_ocupado'
                else:
                    os.makedirs(lp(os.path.dirname(novo)), exist_ok=True)
                    os.rename(lp(atual), lp(novo))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([atual, novo, status])
            if status == 'ok':
                ok += 1
                orig = origem_de.get(atual)
                if orig:
                    con.execute('UPDATE movidos SET destino=? WHERE path=?',
                                (novo, orig))
            else:
                err += 1
    con.commit()
    con.close()
    removidas = 0
    for pasta in glob.glob(os.path.join(DESTINO, '20??', '20??-00')):
        try:
            os.rmdir(lp(pasta))
            removidas += 1
        except OSError:
            pass
    print(f'Movidos: {ok:,} | pendências: {err:,} | pastas ANO-00 removidas: '
          f'{removidas} | log: {log_path}')


def reorg_plano():
    linhas = []
    for entry in sorted(os.scandir(lp(SEM_DATA)), key=lambda e: e.name.lower()):
        if not entry.is_dir():
            continue
        nome = entry.name
        chave = sem_acento(nome)
        grupo = ''
        for g, rx in GRUPOS_RE:
            if re.search(rx, chave):
                grupo = g
                break
        n = sum(len(fs) for _, _, fs in os.walk(entry.path))
        linhas.append((nome, grupo or '???', n))
    with open(CSV_REORG, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['pasta', 'grupo', 'n_arquivos'])
        w.writerows(linhas)
    for nome, grupo, n in linhas:
        print(f'  {grupo or "???":22} {nome:45} {n:,}')
    print(CSV_REORG)


def reorg_exec():
    con = conectar()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(AQUI, f'log_img_semdata_reorg_{ts}.csv')
    ok = err = 0
    with open(CSV_REORG, encoding='utf-8-sig') as f, \
            open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        rd = csv.reader(f)
        next(rd)
        log = csv.writer(flog)
        log.writerow(['pasta_de', 'pasta_para', 'status'])
        for nome, grupo, _n in rd:
            if grupo in ('MANTER', '???', ''):
                continue
            de = os.path.join(SEM_DATA, nome)
            para = os.path.join(SEM_DATA, grupo, nome)
            status = 'ok'
            try:
                if not os.path.isdir(lp(de)):
                    status = 'origem_sumiu'
                elif os.path.exists(lp(para)):
                    status = 'destino_ocupado'
                else:
                    os.makedirs(lp(os.path.dirname(para)), exist_ok=True)
                    os.rename(lp(de), lp(para))
                    con.execute(
                        "UPDATE movidos SET destino = ? || substr(destino, ?) "
                        "WHERE destino LIKE ? || '\\%'",
                        (para, len(de) + 1, de))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([de, para, status])
            ok += status == 'ok'
            err += status != 'ok'
    con.commit()
    con.close()
    print(f'Pastas movidas: {ok} | pendências: {err} | log: {log_path}')


def vazias():
    removidas = 0
    for root, dirs, files in os.walk(lp(DESTINO), topdown=False):
        if root.rstrip('\\') == lp(DESTINO).rstrip('\\'):
            continue
        try:
            os.rmdir(root)  # só remove se realmente vazia
            removidas += 1
        except OSError:
            pass
    print(f'Pastas vazias removidas: {removidas}')


def main():
    modos = {'--anos00-plano': anos00_plano, '--anos00-exec': anos00_exec,
             '--reorg-plano': reorg_plano, '--reorg-exec': reorg_exec,
             '--vazias': vazias}
    for m, fn in modos.items():
        if m in sys.argv:
            fn()
            return
    print('uso: img_18_semdata.py ' + ' | '.join(modos))


if __name__ == '__main__':
    main()
