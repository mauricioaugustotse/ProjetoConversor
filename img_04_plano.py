# -*- coding: utf-8 -*-
"""img_04: monta o plano de destino em D:\\IMAGENS e roda as checagens de sanidade.

Só campeões entram no plano (perdedores ficam nos dumps). A data do grupo é a
melhor entre TODOS os membros (um clone do 4DDig com epoch no nome pode datar o
campeão sem EXIF). Nome final: 'YYYY-MM-DD HH.MM.SS [hash8].ext' conforme a
granularidade; sem-data mantém nome original + [hash8] em 0000 SEM DATA\\<contexto>.
Exclusões (não movem, só relatório): corrompidos/truncados, ícones <50px,
thumbnails de cache <=320px.
"""
import collections
import os
import re

from img_lib import DESTINO, PASTAS_GENERICAS, conectar, lp

FONTE_PRIO = {'exif': 4, 'takeout': 3, 'nome': 2, 'caminho': 1}
GRAN_PRIO = {'datetime': 4, 'dia': 3, 'mes': 2, 'ano': 1}
INVALIDOS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
RE_THUMB = re.compile(r'\.thumbnails|\\thumbs\\|\\cache\\', re.IGNORECASE)


def sanitizar(s, maxlen=80):
    s = INVALIDOS.sub(' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip(' .')
    return (s[:maxlen].strip(' .')) or 'sem nome'


def nome_final(data, gran, hash8, ext, nome_orig):
    if gran == 'datetime':
        base = data.replace(':', '.')          # 2019-07-14 16.32.05
    elif gran in ('dia', 'mes', 'ano'):
        base = data                            # 2019-07-14 | 2019-07 | 2019
    else:
        base = sanitizar(os.path.splitext(nome_orig)[0])
    return f'{base} [{hash8}]{ext}'


def pasta_final(data, gran, album, dump):
    if gran is None:
        ctx = album if album and album.lower() not in PASTAS_GENERICAS else dump
        return os.path.join(DESTINO, '0000 SEM DATA', sanitizar(ctx, 60))
    ano = data[:4]
    mes = data[5:7] if gran != 'ano' else '00'
    return os.path.join(DESTINO, ano, f'{ano}-{mes}')


def main():
    con = conectar()
    datas = {p: (d, f, g) for p, d, f, g in
             con.execute('SELECT path, data, fonte, gran FROM datas')}
    membros_por_grupo = collections.defaultdict(list)
    campeao_de = {}
    for path, grupo, campeao in con.execute('SELECT path, grupo, campeao FROM grupos'):
        membros_por_grupo[grupo].append(path)
        if campeao:
            campeao_de[grupo] = path
    info = {r[0]: r for r in con.execute("""
        SELECT path, ext, sha1, width, height, size, album, dump FROM arquivos
        WHERE erro IS NULL""")}

    def melhor_data(grupo):
        cand = []
        for p in membros_por_grupo[grupo]:
            if p in datas:
                d, f, g = datas[p]
                cand.append((FONTE_PRIO[f], GRAN_PRIO[g], p == campeao_de[grupo], d, f, g))
        if not cand:
            return None
        cand.sort(reverse=True)
        return cand[0][3], cand[0][4], cand[0][5]

    con.execute('DELETE FROM plano')
    lote = []
    stats = collections.Counter()
    bytes_mover = 0
    destinos = {}
    colisoes = []

    # incremental: grupo que já tem membro movido para o acervo não move
    # ninguém (o takeout seria duplicata); campeão precisa existir no disco
    movidos = {p for (p,) in con.execute('SELECT path FROM movidos')}

    for grupo, campeao in campeao_de.items():
        path = campeao
        _, ext, sha1, w, h, size, album, dump = info[path]
        ext = '.jpg' if ext == '.jpeg' else ext
        # exclusões
        if any(m in movidos for m in membros_por_grupo[grupo]):
            lote.append((path, None, 'ja_no_acervo'))
            stats['excl_ja_no_acervo'] += 1
            continue
        if not os.path.exists(lp(path)):
            lote.append((path, None, 'origem_inexistente'))
            stats['excl_origem_inexistente'] += 1
            continue
        if w and h and (w < 50 or h < 50):
            lote.append((path, None, 'icone_minusculo'))
            stats['excl_icone'] += 1
            continue
        if RE_THUMB.search(path) and w and h and max(w, h) <= 320:
            lote.append((path, None, 'thumbnail_cache'))
            stats['excl_thumb'] += 1
            continue
        md = melhor_data(grupo)
        data, fonte, gran = md if md else (None, None, None)
        pasta = pasta_final(data, gran, album, dump)
        nome = nome_final(data, gran, sha1[:8], ext, os.path.basename(path))
        destino = os.path.join(pasta, nome)
        if destino.lower() in destinos:
            nome = nome_final(data, gran, sha1[:16], ext, os.path.basename(path))
            destino = os.path.join(pasta, nome)
            if destino.lower() in destinos:
                colisoes.append(path)
                continue
        destinos[destino.lower()] = path
        lote.append((path, destino, None))
        stats['mover'] += 1
        stats['sem_data' if gran is None else f'gran_{gran}'] += 1
        bytes_mover += size or 0

    # corrompidos/truncados (fora dos grupos) entram no plano só como exclusão
    for path, erro in con.execute('SELECT path, erro FROM arquivos WHERE erro IS NOT NULL'):
        lote.append((path, None, erro))
        stats[f'excl_{erro}'] += 1

    con.executemany('INSERT INTO plano(path, destino, exclusao) VALUES(?,?,?)', lote)
    con.commit()

    # ---- sanidade ----
    anos = collections.Counter()
    for dest, path in destinos.items():
        rel = dest[len(DESTINO):].lstrip('\\').split('\\')[0]
        anos[rel] += 1
    anos_ruins = [a for a in anos if a not in ('0000 sem data',)
                  and not (a.isdigit() and 1990 <= int(a) <= 2026)]
    print(f"Plano: {stats['mover']:,} a mover ({bytes_mover/1e9:.1f} GB) | "
          f"exclusões: ícones {stats['excl_icone']:,}, thumbs {stats['excl_thumb']:,}, "
          f"corrompidos {stats['excl_corrompido']+stats['excl_truncado']+stats['excl_ilegivel']:,}")
    print(f"Granularidade: datetime {stats['gran_datetime']:,} | dia {stats['gran_dia']:,} | "
          f"mes {stats['gran_mes']:,} | ano {stats['gran_ano']:,} | SEM DATA {stats['sem_data']:,}")
    print('Distribuição por ano:')
    for a in sorted(anos):
        print(f'  {a}: {anos[a]:,}')
    ok = True
    if anos_ruins:
        print(f'FALHA SANIDADE: anos fora da faixa: {anos_ruins}')
        ok = False
    if colisoes:
        print(f'FALHA SANIDADE: {len(colisoes)} colisões irresolvíveis: {colisoes[:5]}')
        ok = False
    if stats['mover'] == 0:
        print('FALHA SANIDADE: plano vazio')
        ok = False
    print('SANIDADE: OK' if ok else 'SANIDADE: FALHOU — não executar img_05')
    con.close()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
