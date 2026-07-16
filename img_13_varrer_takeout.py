# -*- coding: utf-8 -*-
r"""img_13: varre D:\_TAKEOUT_EXTRAIDO e os 3 zips de takeout — APAGA DEFINITIVO.

Regra por arquivo restante no extraido:
 - sidecar .json: descartavel (dados ja em takeout_meta);
 - midia: apaga SO se o grupo dela tem membro em `movidos` cujo destino EXISTE
   em D:\MÍDIA agora, e o tamanho no disco confere com o catalogado;
 - corrompidos catalogados (erro IS NOT NULL): lixo binario, apaga;
 - qualquer outra coisa fica e e reportada.
Os zips da raiz so sao apagados se o extraido zerar. Log: log_img_varrida_tk_*.
"""
import csv
import datetime
import glob
import os

from img_lib import EXT_MEDIA, conectar, lp

AQUI = os.path.dirname(os.path.abspath(__file__))
TK = r'D:\_TAKEOUT_EXTRAIDO'


def main():
    con = conectar()
    movidos = {p: d for p, d in con.execute('SELECT path, destino FROM movidos')}
    grupo_de = {}
    membros = {}
    for p, g in con.execute('SELECT path, grupo FROM grupos'):
        grupo_de[p] = g
        membros.setdefault(g, []).append(p)
    tam_cat = {p: s for p, s in con.execute(
        'SELECT path, size FROM arquivos WHERE path LIKE ?', (TK + '\\%',))}
    erros_cat = {p for (p,) in con.execute(
        'SELECT path FROM arquivos WHERE erro IS NOT NULL AND path LIKE ?',
        (TK + '\\%',))}
    con.close()

    # destino coberto por grupo (cache)
    cobertos = {}

    def grupo_coberto(g):
        if g not in cobertos:
            dest = next((movidos[m] for m in membros.get(g, ())
                         if m in movidos and os.path.exists(lp(movidos[m]))), None)
            cobertos[g] = dest
        return cobertos[g]

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(AQUI, f'log_img_varrida_tk_{ts}.csv')
    apagados = jsons = restantes = 0
    bytes_lib = 0
    sobras = []
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['path', 'motivo', 'campeao_em_midia', 'status'])
        for root, dirs, files in os.walk(lp(TK)):
            base = root[4:] if root.startswith('\\\\?\\') else root
            for fn in files:
                path = os.path.join(base, fn)
                ext = os.path.splitext(fn)[1].lower()
                if ext == '.json':
                    try:
                        os.remove(lp(path))
                        jsons += 1
                    except OSError as e:
                        log.writerow([path, 'sidecar', '', f'erro: {e}'])
                    continue
                motivo = dest = None
                if path in erros_cat:
                    motivo = 'corrompido'
                elif ext in EXT_MEDIA and path in grupo_de:
                    dest = grupo_coberto(grupo_de[path])
                    if dest:
                        motivo = 'duplicata_do_acervo'
                if not motivo:
                    restantes += 1
                    sobras.append(path)
                    log.writerow([path, 'sem_cobertura', '', 'MANTIDO'])
                    continue
                try:
                    st = os.stat(lp(path))
                    if tam_cat.get(path) is not None and st.st_size != tam_cat[path]:
                        restantes += 1
                        sobras.append(path)
                        log.writerow([path, motivo, dest or '', 'PULADO_tamanho_mudou'])
                        continue
                    os.remove(lp(path))
                    apagados += 1
                    bytes_lib += st.st_size
                    log.writerow([path, motivo, dest or '', 'apagado'])
                except OSError as e:
                    restantes += 1
                    log.writerow([path, motivo, dest or '', f'erro: {e}'])

    removidas = 0
    for root, dirs, files in os.walk(lp(TK), topdown=False):
        try:
            os.rmdir(root)
            removidas += 1
        except OSError:
            pass
    print(f'Extraido: apagados {apagados:,} midias ({bytes_lib/1e9:.1f} GB) + '
          f'{jsons:,} jsons | mantidos {restantes:,} | pastas removidas {removidas}')
    for s in sobras[:10]:
        print('   sobra:', s)

    if restantes == 0 and not os.path.isdir(lp(TK)):
        zips = sorted(glob.glob(r'D:\takeout-*.zip'))
        gb = sum(os.path.getsize(z) for z in zips) / 1e9
        for z in zips:
            os.remove(z)
        print(f'ZIPs apagados: {len(zips)} ({gb:.1f} GB)')
    else:
        print('ZIPs preservados (ha sobras no extraido — revisar log).')
    print(f'Log: {log_path}')


if __name__ == '__main__':
    main()
