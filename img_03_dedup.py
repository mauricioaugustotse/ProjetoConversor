# -*- coding: utf-8 -*-
"""img_03: agrupa duplicatas e elege o campeão de cada grupo.

Camadas de união:
  1. sha1 igual        (arquivos idênticos byte a byte)
  2. pixmd5 igual      (pixels idênticos; EXIF/container diferentes)
  3. dhash igual       (quase-duplicata visual) — só com salvaguardas:
     popcount 6..58 (descarta imagens uniformes) e razão de aspecto ±1%.

Campeão = mais metadados: tem DateTimeOriginal > nº tags EXIF > GPS >
resolução > tamanho > origem mais "original" (álbuns nomeados vencem dumps
de recuperação). dHash igual não-unido e vizinhos a 1-2 bits vão para
img_revisao_quaseduplicatas.csv (revisão manual, nada é movido/apagado).
"""
import collections
import csv
import os

from img_lib import conectar

SAIDA_REVISAO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'img_revisao_quaseduplicatas.csv')
DUMP_PRIO = {
    'HD EXTERNO Mário': 3,
    'Camera Roll': 2, 'Takeout em 17_12_2023': 2,
    'Banco Onedrive_fotos_consertado_16_9_2024': 2,
    '4DDig': 0, '_ZIP_EXTRAIDO': 0, 'Duplicate': 0,
}


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
            return True
        return False


def main():
    con = conectar()
    rows = con.execute("""
        SELECT path, sha1, pixmd5, dhash, dhash_pop, width, height,
               exif_dto, n_tags, gps, size, dump
        FROM arquivos WHERE erro IS NULL
    """).fetchall()
    n = len(rows)
    print(f'Imagens válidas: {n:,}')
    uf = UF(n)

    # camadas 1 e 2: chaves exatas
    unioes = {'sha1': 0, 'pixmd5': 0, 'dhash': 0}
    for col, camada in ((1, 'sha1'), (2, 'pixmd5')):
        chaves = collections.defaultdict(list)
        for i, r in enumerate(rows):
            if r[col]:
                chaves[r[col]].append(i)
        for idxs in chaves.values():
            for j in idxs[1:]:
                if uf.union(idxs[0], j):
                    unioes[camada] += 1

    # camada 3: dhash igual + entropia ok + aspecto ±1%
    por_dhash = collections.defaultdict(list)
    for i, r in enumerate(rows):
        dh, pop, w, h = r[3], r[4], r[5], r[6]
        if dh and pop is not None and 6 <= pop <= 58 and w and h:
            por_dhash[dh].append(i)
    revisao = []
    for dh, idxs in por_dhash.items():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: rows[i][5] / rows[i][6])
        cluster = [idxs[0]]
        for i in idxs[1:]:
            r_ant = rows[cluster[-1]][5] / rows[cluster[-1]][6]
            r_cur = rows[i][5] / rows[i][6]
            if abs(r_cur - r_ant) / r_ant <= 0.01:
                cluster.append(i)
            else:
                # aspecto divergente: não une, vai p/ revisão
                revisao.append((rows[cluster[-1]][0], rows[i][0], 'dhash_aspecto_difere'))
                cluster = [i]
        for j in cluster[1:] if len(cluster) > 1 else []:
            if uf.union(cluster[0], j):
                unioes['dhash'] += 1

    # dhash igual mas baixa entropia: só relatório (grupos até 20 membros)
    baixa = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r[3] and r[4] is not None and not (6 <= r[4] <= 58):
            baixa[r[3]].append(i)
    n_baixa_sup = 0
    for dh, idxs in baixa.items():
        if 2 <= len(idxs) <= 20:
            for j in idxs[1:]:
                if uf.find(idxs[0]) != uf.find(j):
                    revisao.append((rows[idxs[0]][0], rows[j][0], 'dhash_baixa_entropia'))
        elif len(idxs) > 20:
            n_baixa_sup += len(idxs)

    print(f"Uniões: sha1 {unioes['sha1']:,} | pixmd5 {unioes['pixmd5']:,} | "
          f"dhash {unioes['dhash']:,}")

    # grupos e campeões
    grupos = collections.defaultdict(list)
    for i in range(n):
        grupos[uf.find(i)].append(i)

    def score(i):
        r = rows[i]
        return (1 if r[7] else 0, r[8] or 0, r[9] or 0,
                (r[5] or 0) * (r[6] or 0), r[10] or 0,
                DUMP_PRIO.get(r[11], 1))

    con.execute('DELETE FROM grupos')
    lote = []
    multi = 0
    for gid, membros in grupos.items():
        camada = 'unico'
        if len(membros) > 1:
            multi += 1
            sha1s = {rows[i][1] for i in membros}
            pix = {rows[i][2] for i in membros}
            camada = 'sha1' if len(sha1s) == 1 else ('pixmd5' if len(pix) == 1 else 'dhash')
        campeao = max(membros, key=lambda i: (score(i), rows[i][0]))
        for i in membros:
            lote.append((rows[i][0], gid, 1 if i == campeao else 0, camada))
    con.executemany('INSERT INTO grupos(path, grupo, campeao, camada) VALUES(?,?,?,?)', lote)
    con.commit()

    unicos = len(grupos)
    print(f'Grupos: {unicos:,} únicos ({multi:,} com duplicatas; '
          f'{n - unicos:,} arquivos redundantes)')

    with open(SAIDA_REVISAO, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['path_a', 'path_b', 'motivo'])
        w.writerows(revisao)
    print(f'Revisão manual: {len(revisao):,} pares em {SAIDA_REVISAO}'
          + (f' (+{n_baixa_sup:,} imagens uniformes em grupos >20, omitidas)' if n_baixa_sup else ''))
    con.close()


if __name__ == '__main__':
    main()
