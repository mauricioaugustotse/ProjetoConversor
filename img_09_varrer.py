# -*- coding: utf-8 -*-
"""img_09: varre dos dumps a mídia duplicada/excluída — APAGA DEFINITIVO.

Só apaga o que o catálogo conhece e nas seguintes condições:
 - duplicata (perdedor de grupo): o campeão do grupo consta em `movidos` E o
   arquivo de destino EXISTE em D:\\MÍDIA agora; e o tamanho do perdedor no
   disco confere com o catalogado (se mudou desde o catálogo, NÃO apaga);
 - exclusões deliberadas do plano (corrompido/truncado/ilegível/ícone).
Nada fora do catálogo é tocado. Pastas vazias só são removidas em
D:\\_ZIP_EXTRAIDO (criada pelo pipeline). Log: log_img_varrida_<ts>.csv.
Ao final, relata a mídia que restou nos dumps e o motivo.
"""
import collections
import csv
import datetime
import os

from img_lib import EXT_MEDIA, EXCLUIR_TOPO, RAIZ, ZIP_EXTRAIDO, conectar, lp

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    con = conectar()
    movidos = {p: d for p, d in con.execute('SELECT path, destino FROM movidos')}
    campeao_grupo = {}
    perdedores = []
    for p, g, c in con.execute('SELECT path, grupo, campeao FROM grupos'):
        if c:
            campeao_grupo[g] = p
        else:
            perdedores.append((p, g))
    exclusoes = list(con.execute(
        'SELECT path, exclusao FROM plano WHERE exclusao IS NOT NULL'))
    tam_cat = {p: s for p, s in con.execute('SELECT path, size FROM arquivos')}
    con.close()
    print(f'Candidatos: {len(perdedores):,} duplicatas + {len(exclusoes):,} exclusões')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(AQUI, f'log_img_varrida_{ts}.csv')
    apagados = pulados = ja_sumidos = 0
    bytes_liberados = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['path', 'motivo', 'campeao_em_midia', 'status'])

        def apagar(path, motivo, campeao_dest):
            nonlocal apagados, pulados, ja_sumidos, bytes_liberados
            try:
                st = os.stat(lp(path))
            except OSError:
                ja_sumidos += 1
                return
            if tam_cat.get(path) is not None and st.st_size != tam_cat[path]:
                log.writerow([path, motivo, campeao_dest, 'PULADO_tamanho_mudou'])
                pulados += 1
                return
            try:
                os.remove(lp(path))
                bytes_liberados += st.st_size
                apagados += 1
                log.writerow([path, motivo, campeao_dest, 'apagado'])
            except OSError as e:
                log.writerow([path, motivo, campeao_dest, f'erro: {e}'])
                pulados += 1

        for path, grupo in perdedores:
            camp = campeao_grupo.get(grupo)
            dest = movidos.get(camp)
            if not dest or not os.path.exists(lp(dest)):
                log.writerow([path, 'duplicata', dest or '', 'PULADO_campeao_ausente'])
                pulados += 1
                continue
            apagar(path, 'duplicata', dest)
        for path, motivo in exclusoes:
            apagar(path, motivo, '')

    # limpar árvore vazia do _ZIP_EXTRAIDO (criada pelo pipeline)
    removidas = 0
    if os.path.isdir(ZIP_EXTRAIDO):
        for root, dirs, files in os.walk(lp(ZIP_EXTRAIDO), topdown=False):
            if not os.listdir(root):
                os.rmdir(root)
                removidas += 1
    print(f'Apagados: {apagados:,} ({bytes_liberados/1e9:.1f} GB) | '
          f'pulados: {pulados:,} | já não existiam: {ja_sumidos:,} | '
          f'pastas vazias removidas no _ZIP_EXTRAIDO: {removidas}')
    print(f'Log: {log_path}')

    # o que restou de mídia nos dumps (e amostra do porquê)
    restam = collections.Counter()
    amostra = collections.defaultdict(list)
    for entry in os.scandir(RAIZ):
        if not entry.is_dir() or entry.name in EXCLUIR_TOPO \
                or entry.name.startswith(('_ZIP_EXTRAIDO', '_ZIP_DOCUMENTOS')):
            continue
        for root, dirs, files in os.walk(entry.path):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in EXT_MEDIA:
                    restam[entry.name] += 1
                    if len(amostra[entry.name]) < 3:
                        amostra[entry.name].append(os.path.join(root, fn))
    print(f'\nMídia restante nos dumps: {sum(restam.values()):,}')
    for k, v in restam.most_common():
        print(f'  {v:>6,}  {k}')
        for a in amostra[k]:
            print(f'          ex.: {a}')


if __name__ == '__main__':
    main()
