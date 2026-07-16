# -*- coding: utf-8 -*-
"""img_00: extrai imagens de dentro dos .zip do D:\\ para D:\\_ZIP_EXTRAIDO.

Os ZIPs originais ficam intactos (extração é cópia; a dedup do img_03 descarta
o que já existir solto nos dumps). Idempotente: membro já extraído com o mesmo
tamanho é pulado. Relatório em img_zip_relatorio.csv.
"""
import csv
import os
import re
import zipfile

from img_lib import EXT_MEDIA, EXCLUIR_TOPO, RAIZ, ZIP_EXTRAIDO, conectar, lp

SAIDA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img_zip_relatorio.csv')
INVALIDOS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def nome_membro(info):
    """Decodifica o nome interno; zips antigos do Windows vêm em cp850/cp437."""
    nome = info.filename
    if info.flag_bits & 0x800:  # bit UTF-8 do zip
        return nome
    try:
        return nome.encode('cp437').decode('cp850')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return nome


def sanitizar(seg):
    seg = INVALIDOS.sub('_', seg).strip(' .')
    return seg or '_'


def main():
    # idempotência via CATÁLOGO, não via disco: membro já catalogado não é
    # re-extraído mesmo que o arquivo tenha sido MOVIDO para D:\MÍDIA depois
    # (bug da rodada de 13/07/2026: 4.586 re-extrações de campeões já movidos)
    con = conectar()
    ja_catalogado = {p.lower(): s for p, s in con.execute(
        'SELECT path, size FROM arquivos WHERE path LIKE ?',
        (ZIP_EXTRAIDO + '\\%',))}
    con.close()

    zips = []
    for root, dirs, files in os.walk(RAIZ):
        partes = root.rstrip('\\').split('\\')
        if len(partes) >= 2 and partes[1] in EXCLUIR_TOPO:
            dirs[:] = []
            continue
        if root.startswith(ZIP_EXTRAIDO):
            dirs[:] = []
            continue
        for fn in files:
            if fn.lower().endswith('.zip'):
                zips.append(os.path.join(root, fn))
    print(f'ZIPs encontrados: {len(zips)}')

    linhas = []
    tot_extraidos = tot_pulados = 0
    for i, zp in enumerate(sorted(zips), 1):
        base = sanitizar(os.path.splitext(os.path.basename(zp))[0])
        destino_zip = os.path.join(ZIP_EXTRAIDO, f'{base}_{i:03d}')
        try:
            with zipfile.ZipFile(lp(zp)) as zf:
                membros = [m for m in zf.infolist()
                           if not m.is_dir()
                           and os.path.splitext(m.filename)[1].lower() in EXT_MEDIA]
                extraidos = 0
                for m in membros:
                    interno = nome_membro(m)
                    rel = '\\'.join(sanitizar(s) for s in re.split(r'[\\/]+', interno) if s)
                    alvo = os.path.join(destino_zip, rel)
                    try:
                        if ja_catalogado.get(alvo.lower()) == m.file_size or \
                                (os.path.exists(lp(alvo))
                                 and os.path.getsize(lp(alvo)) == m.file_size):
                            tot_pulados += 1
                            continue
                        os.makedirs(lp(os.path.dirname(alvo)), exist_ok=True)
                        with zf.open(m) as src, open(lp(alvo), 'wb') as dst:
                            while True:
                                chunk = src.read(1 << 20)
                                if not chunk:
                                    break
                                dst.write(chunk)
                        extraidos += 1
                    except (OSError, zipfile.BadZipFile, RuntimeError) as e:
                        linhas.append([zp, interno, 'erro_membro', str(e)])
                tot_extraidos += extraidos
                linhas.append([zp, '', 'ok', f'{len(membros)} imagens, {extraidos} extraidas'])
        except (zipfile.BadZipFile, RuntimeError, OSError) as e:
            linhas.append([zp, '', 'zip_ilegivel', str(e)])
        if i % 10 == 0:
            print(f'  {i}/{len(zips)} zips...')

    with open(SAIDA_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['zip', 'membro', 'status', 'detalhe'])
        w.writerows(linhas)
    print(f'Extraidas: {tot_extraidos} | ja existiam: {tot_pulados} | relatorio: {SAIDA_CSV}')


if __name__ == '__main__':
    main()
