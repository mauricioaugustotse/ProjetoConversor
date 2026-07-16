# -*- coding: utf-8 -*-
"""img_06: gera o produto final auditável dentro de D:\\IMAGENS.

- _CATALOGO.csv: uma linha por imagem movida (nome final, data, proveniência,
  duplicatas absorvidas).
- _RELATORIO.md: estatísticas por ano/mês, por dump, dedup por camada,
  exclusões e pendências.
- log_img_duplicatas.csv (no ProjetoConversor): cada duplicata que FICOU nos
  dumps -> campeão que a representa em D:\\IMAGENS.
"""
import collections
import csv
import datetime
import os

from img_lib import DESTINO, conectar

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    con = conectar()
    movidos = {p: d for p, d in con.execute('SELECT path, destino FROM movidos')}
    datas = {p: (d, f, g) for p, d, f, g in
             con.execute('SELECT path, data, fonte, gran FROM datas')}
    arq = {r[0]: r for r in con.execute(
        'SELECT path, dump, album, size, width, height, sha1, erro FROM arquivos')}
    grupo_de = {}
    campeao_de = {}
    membros = collections.defaultdict(list)
    camada_de = {}
    for p, g, c, cam in con.execute('SELECT path, grupo, campeao, camada FROM grupos'):
        grupo_de[p] = g
        membros[g].append(p)
        camada_de[g] = cam
        if c:
            campeao_de[g] = p
    exclusoes = list(con.execute(
        'SELECT path, exclusao FROM plano WHERE exclusao IS NOT NULL'))

    # ---- _CATALOGO.csv ----
    cat_path = os.path.join(DESTINO, '_CATALOGO.csv')
    with open(cat_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['nome_final', 'pasta', 'data', 'fonte_data', 'granularidade',
                    'origem', 'dump', 'album', 'sha1', 'largura', 'altura',
                    'tamanho_bytes', 'duplicatas_absorvidas'])
        for src, dst in sorted(movidos.items(), key=lambda kv: kv[1]):
            d, fo, g = datas.get(src, ('', '', 'sem data'))
            a = arq.get(src)
            ndup = len(membros[grupo_de[src]]) - 1 if src in grupo_de else 0
            rel = os.path.dirname(dst)[len(DESTINO):].lstrip('\\')
            w.writerow([os.path.basename(dst), rel, d, fo, g, src,
                        a[1] if a else '', a[2] if a else '',
                        a[6] if a else '', a[4] if a else '', a[5] if a else '',
                        a[3] if a else '', ndup])

    # ---- log_img_duplicatas.csv ----
    dup_path = os.path.join(AQUI, 'log_img_duplicatas.csv')
    n_dup = 0
    with open(dup_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['duplicata_no_dump', 'camada', 'campeao_origem', 'campeao_em_imagens'])
        for g, ms in membros.items():
            if len(ms) < 2:
                continue
            camp = campeao_de[g]
            for m in ms:
                if m != camp:
                    w.writerow([m, camada_de[g], camp, movidos.get(camp, '(nao movido)')])
                    n_dup += 1

    # ---- _RELATORIO.md ----
    por_mes = collections.Counter()
    por_dump = collections.Counter()
    por_fonte = collections.Counter()
    bytes_tot = 0
    for src, dst in movidos.items():
        rel = os.path.dirname(dst)[len(DESTINO):].lstrip('\\')
        por_mes[rel.split('\\')[-1] if '\\' in rel else rel] += 1
        a = arq.get(src)
        if a:
            por_dump[a[1]] += 1
            bytes_tot += a[3] or 0
        _, fo, g = datas.get(src, (None, None, None))
        por_fonte[fo or 'sem data'] += 1
    excl_count = collections.Counter(e for _, e in exclusoes)
    camadas = collections.Counter()
    for g, ms in membros.items():
        if len(ms) > 1:
            camadas[camada_de[g]] += len(ms) - 1

    rel_path = os.path.join(DESTINO, '_RELATORIO.md')
    with open(rel_path, 'w', encoding='utf-8') as f:
        f.write('# D:\\IMAGENS — relatório de consolidação\n\n')
        f.write(f'Gerado em {datetime.date.today().isoformat()} pelo pipeline '
                f'img_* do ProjetoConversor.\n\n')
        f.write(f'- **Imagens únicas movidas: {len(movidos):,}** ({bytes_tot/1e9:.1f} GB)\n')
        f.write(f'- Duplicatas que permaneceram nos dumps: {n_dup:,} '
                f'(mapa em log_img_duplicatas.csv)\n')
        f.write(f'- Excluídas do produto (permanecem nos dumps): '
                f'{sum(excl_count.values()):,}\n\n')
        f.write('## Fonte da data\n\n')
        for k, v in por_fonte.most_common():
            f.write(f'- {k}: {v:,}\n')
        f.write('\n## Duplicatas eliminadas por camada\n\n')
        for k, v in camadas.most_common():
            f.write(f'- {k}: {v:,}\n')
        f.write('\n## Proveniência (dump de origem dos campeões)\n\n')
        for k, v in por_dump.most_common():
            f.write(f'- {k}: {v:,}\n')
        f.write('\n## Exclusões (não movidas, listadas no plano)\n\n')
        for k, v in excl_count.most_common():
            f.write(f'- {k}: {v:,}\n')
        f.write('\n## Imagens por pasta mensal\n\n')
        for k in sorted(por_mes):
            f.write(f'- {k}: {por_mes[k]:,}\n')
        f.write('\n## Notas\n\n')
        f.write('- Vídeos (7.584 MP4 e outros) NÃO foram tratados — possível fase 2.\n')
        f.write('- Quase-duplicatas com assinatura visual igual mas aspecto/entropia '
                'divergentes estão em img_revisao_quaseduplicatas.csv (revisão manual).\n')
        f.write('- Datas: EXIF > nome do arquivo > pasta; mtime nunca foi usado '
                '(cópias de 2024 recarimbaram os arquivos).\n')
    print(f'Catálogo: {cat_path}\nRelatório: {rel_path}\nDuplicatas: {dup_path}')
    con.close()


if __name__ == '__main__':
    main()
