# -*- coding: utf-8 -*-
"""img_02: datação em cascata — EXIF > takeout (photoTakenTime) > nome > caminho.

Grava na tabela datas: data ('YYYY' | 'YYYY-MM' | 'YYYY-MM-DD' |
'YYYY-MM-DD HH:MM:SS'), fonte (exif|takeout|nome|caminho) e gran
(ano|mes|dia|datetime). Arquivo sem nenhuma pista fica fora da tabela (vai
para 0000 SEM DATA). mtime NÃO é usado: as cópias de 2024 recarimbaram tudo.
A fonte takeout vem da tabela takeout_meta (sidecars JSON do Google Fotos,
img_12); epoch UTC convertido para hora local (consistente com nomes Samsung).
"""
import datetime
import os
import re

from img_lib import EXT_VID, conectar

ANO_MIN, ANO_MAX = 1990, 2026

MESES_PT = {
    'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
    'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12,
}
# pastas cujo ano é data de backup/cópia, não das fotos
BLACKLIST_PASTA = re.compile(
    r'backup|takeout|anterior|consertado|banco|extra[çc][ãa]o|onedrive|'
    r'duplicate|sem.?data|recuperad|restaurad', re.IGNORECASE)

RE_NOME = [
    # IMG-20180908-WA0046 (WhatsApp: data de recebimento, sem hora)
    (re.compile(r'(?:IMG|VID)-(\d{4})(\d{2})(\d{2})-WA\d+', re.I), 'dia'),
    # 20181006_183043 / IMG_20181006_183043 / PXL_20181006_183043
    (re.compile(r'(?<!\d)(\d{4})(\d{2})(\d{2})[_\- ](\d{2})(\d{2})(\d{2})(?!\d)'), 'datetime'),
    # 2016-06-01 21.29.42 / 2016-06-01_21-29-42
    (re.compile(r'(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})[ _T](\d{2})[.\-h:](\d{2})[.\-m:](\d{2})'), 'datetime'),
    # 2016-06-01 (sem hora)
    (re.compile(r'(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})(?!\d)'), 'dia'),
    # 20181006 solto (WP_20181006.jpg, Screenshot_20181006-1830.png)
    (re.compile(r'(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)'), 'dia'),
]
RE_EPOCH = re.compile(r'(?<!\d)(\d{10}|\d{13})(?!\d)')
RE_PHOTOS_FROM = re.compile(r'(?:Photos from|Fotos de) (\d{4})', re.I)
RE_ANO_COMP = re.compile(r'^(19[89]\d|20[0-2]\d)$')
RE_MES_COMP = re.compile(r'^(0[1-9]|1[0-2])$')
RE_ANO_TXT = re.compile(r'(?<!\d)(19[89]\d|20[0-2]\d)(?!\d)')


def valida(a, m=1, d=1, hh=0, mi=0, ss=0):
    if not (ANO_MIN <= a <= ANO_MAX):
        return None
    try:
        return datetime.datetime(a, m, d, hh, mi, ss)
    except ValueError:
        return None


def data_exif(row):
    for campo in ('exif_dto', 'exif_dtd', 'exif_dt'):
        s = row[campo]
        if not s:
            continue
        m = re.match(r'(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})', s)
        if not m:
            continue
        a, mo, d, hh, mi, ss = map(int, m.groups())
        dt = valida(a, mo, d, hh, mi, ss)
        if dt:
            return dt.strftime('%Y-%m-%d %H:%M:%S'), 'datetime'
    return None


def data_nome(nome):
    for rx, gran in RE_NOME:
        m = rx.search(nome)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        if gran == 'datetime':
            dt = valida(*g)
            if dt:
                return dt.strftime('%Y-%m-%d %H:%M:%S'), 'datetime'
            dt = valida(*g[:3])  # hora inválida -> degrada p/ dia
            if dt:
                return dt.strftime('%Y-%m-%d'), 'dia'
        else:
            dt = valida(*g)
            if dt:
                return dt.strftime('%Y-%m-%d'), 'dia'
    # Unix epoch (1387572622492.jpg do Android; 10 díg. = segundos)
    for m in RE_EPOCH.finditer(nome):
        v = int(m.group(1))
        if len(m.group(1)) == 13:
            v //= 1000
        if not (789_000_000 <= v <= 1_800_000_000):  # 1995..2027
            continue
        dt = datetime.datetime.fromtimestamp(v)
        if ANO_MIN <= dt.year <= ANO_MAX:
            return dt.strftime('%Y-%m-%d %H:%M:%S'), 'datetime'
    return None


def data_caminho(path):
    partes = path.split('\\')
    comps = partes[2:-1]  # abaixo do dump de topo, sem o nome do arquivo
    # 1) componente-ano exato (Camera Roll\2023\08, Imagens da Câmera\2024)
    for i, c in enumerate(comps):
        if RE_ANO_COMP.match(c):
            ano = int(c)
            if i + 1 < len(comps) and RE_MES_COMP.match(comps[i + 1]):
                return f'{ano}-{comps[i+1]}', 'mes'
            return str(ano), 'ano'
    # 2) "Photos from 2022" (Google Takeout)
    m = RE_PHOTOS_FROM.search(path)
    if m and ANO_MIN <= int(m.group(1)) <= ANO_MAX:
        return m.group(1), 'ano'
    # 3) álbum com ano no nome ("Cancun 2019", "Passeio em Natal - Nov 2021")
    for c in reversed(comps):
        if BLACKLIST_PASTA.search(c):
            continue
        m = RE_ANO_TXT.search(c)
        if not m:
            continue
        ano = int(m.group(1))
        low = c.lower()
        for abrev, num in MESES_PT.items():
            if re.search(r'\b' + abrev, low):
                return f'{ano}-{num:02d}', 'mes'
        return str(ano), 'ano'
    return None


def main():
    con = conectar()
    rows = con.execute("""
        SELECT path, exif_dto, exif_dtd, exif_dt FROM arquivos WHERE erro IS NULL
    """).fetchall()
    print(f'Datando {len(rows):,} imagens...')
    takeout = {}
    if con.execute("SELECT name FROM sqlite_master WHERE name='takeout_meta'").fetchone():
        for p, epoch in con.execute('SELECT path, epoch FROM takeout_meta'):
            try:
                dt = datetime.datetime.fromtimestamp(epoch)
            except (OSError, OverflowError, ValueError):
                continue
            if ANO_MIN <= dt.year <= ANO_MAX:
                takeout[p] = dt.strftime('%Y-%m-%d %H:%M:%S')
    con.execute('DELETE FROM datas')

    stats = {'exif': 0, 'takeout': 0, 'nome': 0, 'caminho': 0, 'nenhuma': 0}
    lote = []
    for path, dto, dtd, dt in rows:
        r_exif = data_exif({'exif_dto': dto, 'exif_dtd': dtd, 'exif_dt': dt})
        r_nome = data_nome(path.split('\\')[-1])
        r_tk = (takeout[path], 'datetime') if path in takeout else None
        # vídeo: mvhd é UTC; nome com hora completa é hora LOCAL da gravação
        # (ordena melhor junto às fotos) — mas nome só-dia (WhatsApp = data de
        # recebimento) perde do container (gravação real)
        video = os.path.splitext(path)[1].lower() in EXT_VID
        if video and r_nome and r_nome[1] == 'datetime':
            r, fonte = r_nome, 'nome'
        elif r_exif:
            r, fonte = r_exif, 'exif'
        elif r_tk:
            r, fonte = r_tk, 'takeout'
        elif r_nome:
            r, fonte = r_nome, 'nome'
        else:
            r = data_caminho(path)
            fonte = 'caminho'
        if r:
            lote.append((path, r[0], fonte, r[1]))
            stats[fonte] += 1
        else:
            stats['nenhuma'] += 1
    con.executemany('INSERT INTO datas(path, data, fonte, gran) VALUES(?,?,?,?)', lote)
    con.commit()
    con.close()
    total = len(rows) or 1
    print('Fontes de data:')
    for k, v in stats.items():
        print(f'  {k:8}: {v:>8,} ({100*v/total:.1f}%)')


if __name__ == '__main__':
    main()
