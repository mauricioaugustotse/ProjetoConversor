# -*- coding: utf-8 -*-
"""tsje_05b: gera arquivos de lote para a transcricao (blocos autocontidos
por ata, com caminhos absolutos de PNGs/OCR/destino), a partir do manifest.

Uso: python tsje_05b_lotes.py --ano 1932 [--tamanho 12]
Saida: D:\\TSJE_TRANSCRICOES\\_lotes\\lote_<ano>_<nn>.txt (so atas pendentes)
"""
import argparse
import csv
import io
import os

RAIZ = r'D:\TSJE_TRANSCRICOES'
MANIFEST = os.path.join(RAIZ, 'manifest.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ano', required=True)
    ap.add_argument('--tamanho', type=int, default=12)
    args = ap.parse_args()

    with io.open(MANIFEST, encoding='utf-8-sig') as f:
        regs = [r for r in csv.DictReader(f)
                if r['ano'] == args.ano and r['status'] == 'pendente'
                and not os.path.exists(os.path.join(RAIZ, r['transcricao']))]

    pasta = os.path.join(RAIZ, '_lotes')
    os.makedirs(pasta, exist_ok=True)
    lotes = [regs[i:i + args.tamanho] for i in range(0, len(regs), args.tamanho)]
    for n, lote in enumerate(lotes, 1):
        path = os.path.join(pasta, f'lote_{args.ano}_{n:02d}.txt')
        with io.open(path, 'w', encoding='utf-8') as f:
            for r in lote:
                pngs = [os.path.join(RAIZ, p) for p in r['pngs'].split(';')]
                f.write('=== ATA ===\n')
                f.write(f'ata_id: {r["ata_id"]}\n')
                f.write(f'titulo_estimado: {r["titulo"]}\n')
                f.write(f'data_sessao: {r["data_sessao"]} | tipo: {r["tipo"]}'
                        f' | num: {r["num"] or "?"}\n')
                f.write(f'boletim: n. {r["be_n"] or "?"}, de {r["be_data"]}'
                        f' | arquivo: {r["arquivo"]}\n')
                f.write(f'pagina_do_cabecalho: {r["pag_ini"]}'
                        f' (recorte ate p. {r["pag_fim"]})\n')
                f.write('imagens:\n')
                for p in pngs:
                    f.write(f'  {p}\n')
                f.write(f'ocr_apoio: {os.path.join(RAIZ, r["ocr_txt"])}\n')
                f.write(f'gravar_em: {os.path.join(RAIZ, r["transcricao"])}\n\n')
        print(f'{path}: {len(lote)} atas')
    if not lotes:
        print(f'Nenhuma ata pendente de {args.ano}.')


if __name__ == '__main__':
    main()
