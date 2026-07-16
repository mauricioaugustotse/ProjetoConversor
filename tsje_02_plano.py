# -*- coding: utf-8 -*-
"""tsje_02: dedup + decisao + nomes finais a partir de tsje_catalogo.tsv.

Dedup em 2 camadas (sha1 binario; sha1 do texto normalizado p/ digitalizacoes
iguais com bytes diferentes). Inclui no banco quem tem >=1 ata do TSJE (atas
so de Tribunais Regionais nao contam). Sem OCR ou sem data completa -> REVISAR
(nunca excluir em silencio).

Saidas: tsje_plano.csv       (todas as linhas, acao+motivo+destino)
        tsje_indice_atas.csv (uma linha por ata dos PDFs incluidos)

Uso: python tsje_02_plano.py
"""
import csv
import io
import os
import re
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
CATALOGO = os.path.join(BASE, 'tsje_catalogo.tsv')
PLANO = os.path.join(BASE, 'tsje_plano.csv')
INDICE = os.path.join(BASE, 'tsje_indice_atas.csv')
DESTINO = r'D:\TSJE_ATAS'
MIN_NORM = 300

# Datas conferidas manualmente no cabecalho impresso (OCR errou so o ano:
# "11 DE JULHO DE 1931" com ANO III/atas de jun-1934; "17 DE SETEMBRO DE 1980"
# com ANNO V) — pasta do zip sem o dia.
DATAS_MANUAIS = {
    r'BEs 1934\JULHO\1934_boletim_eleitoral_a3_n59.pdf': '1934-07-11',
    r'BEs 1936\SETEMBRO\1936_boletim_eleitoral_a5_n110.pdf': '1936-09-17',
}


def sanitizar(s, maxlen=120):
    s = re.sub(r'[\\/:*?"<>|\r\n]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()[:maxlen]


def ler_catalogo():
    with io.open(CATALOGO, encoding='utf-8') as f:
        cols = f.readline().rstrip('\n').split('\t')
        regs = []
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == len(cols):
                regs.append(dict(zip(cols, parts)))
    return regs


def data_final(r):
    """(data_iso_completa|'', fonte). Pasta curada > nome > cabecalho impresso.

    Excecao: se o ANO da pasta difere do ano impresso no BE, vale o impresso
    (ha volumes mal-arquivados no acervo, ex.: BE a6 n53 de 1937 em 1936\\MAIO).
    """
    dp, dn, dt = r['data_pasta'], r['data_nome'], r['data_texto']
    if r['relpath'] in DATAS_MANUAIS:
        return DATAS_MANUAIS[r['relpath']], 'manual'
    if len(dp) == 10:
        # so confia no ano impresso se dia+mes batem com a pasta (padrao real
        # dos volumes mal-arquivados); divergencia de dia/mes = OCR suspeito
        if len(dt) == 10 and dt[:4] != dp[:4] and dt[5:] == dp[5:]:
            return dt, 'texto_ano_divergente'
        return dp, 'pasta'
    if len(dn) == 10:
        return dn, 'nome'
    if len(dt) == 10:
        return dt, 'texto'
    return '', ''


def escolher_campeao(grupo):
    return max(grupo, key=lambda r: (len(data_final(r)[0]) == 10,
                                     int(r['len_norm'] or 0),
                                     int(r['size'] or 0),
                                     -len(r['relpath'])))


def main():
    regs = ler_catalogo()

    # camada 1: sha1 binario
    por_sha1 = defaultdict(list)
    for r in regs:
        por_sha1[r['sha1']].append(r)
    for grupo in por_sha1.values():
        if len(grupo) < 2:
            continue
        campeao = escolher_campeao(grupo)
        for r in grupo:
            if r is not campeao:
                r['acao'] = 'duplicata_bin'
                r['duplicata_de'] = campeao['relpath']

    # camada 2: sha1 do texto normalizado (entre sobreviventes com texto)
    por_norm = defaultdict(list)
    for r in regs:
        if 'acao' not in r and r['sha1_norm'] and int(r['len_norm'] or 0) >= MIN_NORM:
            por_norm[r['sha1_norm']].append(r)
    for grupo in por_norm.values():
        if len(grupo) < 2:
            continue
        campeao = escolher_campeao(grupo)
        for r in grupo:
            if r is not campeao:
                r['acao'] = 'duplicata_texto'
                r['duplicata_de'] = campeao['relpath']

    # decisao + nome final
    nomes_usados = set()
    for r in regs:
        r.setdefault('duplicata_de', '')
        data, fonte = data_final(r)
        r['data_final'], r['fonte_data'] = data, fonte
        r['n_boletim'] = r['n_texto'] or r['n_nome']
        r['destino'] = ''
        if 'acao' in r:
            r['motivo'] = 'identico_binario' if r['acao'] == 'duplicata_bin' \
                else 'mesmo_texto_ocr'
            continue
        flags = r['flags']
        n_sup = int(r['n_atas_sup'] or 0)
        if 'erro_pdf' in flags or 'sem_texto' in flags:
            r['acao'], r['motivo'] = 'revisar', 'sem_texto_ocr'
        elif n_sup == 0:
            r['acao'] = 'excluir'
            r['motivo'] = ('so_ata_regional' if int(r['n_atas_reg'] or 0) > 0
                           else 'sem_ata')
        elif not data:
            r['acao'], r['motivo'] = 'revisar', 'sem_data_completa'
        else:
            r['acao'], r['motivo'] = 'incluir', ''
            n = f' n{int(r["n_boletim"]):03d}' if r['n_boletim'].isdigit() else ''
            ident = r['id_pasta'] or r['sha1'][:8]
            nome = sanitizar(f'{data} - BE{n} [{ident}].pdf')
            seq = 2
            while nome.lower() in nomes_usados:
                nome = sanitizar(f'{data} - BE{n} [{ident}] ({seq}).pdf')
                seq += 1
            nomes_usados.add(nome.lower())
            r['destino'] = os.path.join(DESTINO, data[:4], nome)

    ordem = {'incluir': 0, 'revisar': 1, 'duplicata_bin': 2,
             'duplicata_texto': 3, 'excluir': 4}
    regs.sort(key=lambda r: (ordem[r['acao']], r['ano_zip'],
                             r['data_final'] or '9999', r['relpath']))

    cols = ['acao', 'motivo', 'relpath', 'ano_zip', 'data_final', 'fonte_data',
            'n_boletim', 'destino', 'duplicata_de', 'n_atas_sup', 'n_atas_reg',
            'paginas', 'len_norm', 'flags', 'sha1']
    with io.open(PLANO, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in regs:
            w.writerow([r.get(c, '') for c in cols])

    with io.open(INDICE, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['data_sessao', 'tribunal', 'tipo', 'num_ordinal', 'pagina',
                    'n_boletim_be', 'data_boletim', 'arquivo_destino', 'origem'])
        idx = []
        for r in regs:
            if r['acao'] != 'incluir' or not r['atas']:
                continue
            for ata in r['atas'].split(';'):
                partes = ata.split(':')
                if len(partes) == 5:
                    trib, tipo, num, dt, pag = partes
                    idx.append([dt or '?', trib, tipo, num, pag,
                                r['n_boletim'], r['data_final'],
                                os.path.basename(r['destino']), r['relpath']])
        idx.sort()
        w.writerows(idx)

    tot = defaultdict(int)
    por_ano = defaultdict(int)
    atas_sup = 0
    for r in regs:
        tot[r['acao']] += 1
        if r['acao'] == 'incluir':
            por_ano[r['ano_zip']] += 1
            atas_sup += int(r['n_atas_sup'] or 0)
    print(f'{len(regs)} PDFs no catalogo -> {PLANO}')
    for acao in ordem:
        print(f'  {acao}: {tot[acao]}')
    print('incluidos por ano:', dict(sorted(por_ano.items())))
    print(f'atas do TSJE indexadas: {atas_sup} -> {INDICE}')


if __name__ == '__main__':
    main()
