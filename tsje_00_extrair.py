# -*- coding: utf-8 -*-
"""tsje_00: extrai os zips "BEs AAAA-*.zip" de Downloads para D:\\TSJE_TRABALHO\\extraido.

Os zips ficam intactos. Reexecucao e retomavel: entrada ja extraida com o mesmo
tamanho e pulada. Nomes de entrada sem flag UTF-8 sao redecodificados de cp437
(pegadinha classica de zip com acentos: MARÇO etc.).

Uso: python tsje_00_extrair.py
"""
import glob
import os
import sys
import zipfile

ZIPS = glob.glob(r"C:\Users\mauri\Downloads\BEs *.zip")
DEST = r"D:\TSJE_TRABALHO\extraido"


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def nome_real(info):
    """Nome da entrada com acentos corretos (cp437 -> utf-8 quando sem flag)."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode('cp437').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return info.filename


def main():
    if not ZIPS:
        print('Nenhum zip "BEs *.zip" em Downloads.')
        sys.exit(1)
    os.makedirs(DEST, exist_ok=True)
    tot_novos = tot_pulados = 0
    for zpath in sorted(ZIPS):
        novos = pulados = 0
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = nome_real(info).replace('/', '\\')
                destino = os.path.join(DEST, rel)
                if (os.path.exists(lp(destino))
                        and os.path.getsize(lp(destino)) == info.file_size):
                    pulados += 1
                    continue
                os.makedirs(lp(os.path.dirname(destino)), exist_ok=True)
                with zf.open(info) as src, open(lp(destino), 'wb') as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                novos += 1
        tot_novos += novos
        tot_pulados += pulados
        print(f'{os.path.basename(zpath)}: {novos} extraidos, {pulados} ja existiam')
    print(f'TOTAL: {tot_novos} extraidos, {tot_pulados} pulados -> {DEST}')


if __name__ == '__main__':
    main()
