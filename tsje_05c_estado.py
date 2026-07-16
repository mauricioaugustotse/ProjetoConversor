# -*- coding: utf-8 -*-
"""tsje_05c: sincroniza o manifest com os .md de transcricao existentes.

- md existe + status era 'pendente'  -> status 'transcrita'
- copia confianca/ilegiveis do front-matter; corrige titulo/data se o
  transcritor os corrigiu pela imagem (campos corrigido_*).
- md com 'problema:' -> status 'problema' (nao vira transcrita).

Uso: python tsje_05c_estado.py
"""
import csv
import io
import json
import os
import re

RAIZ = r'D:\TSJE_TRANSCRICOES'
MANIFEST = os.path.join(RAIZ, 'manifest.csv')


def front_matter(path):
    fm = {}
    try:
        with io.open(path, encoding='utf-8') as f:
            texto = f.read()
    except OSError:
        return fm, ''
    m = re.match(r'---\n(.*?)\n---\n?(.*)', texto, re.S)
    if not m:
        return fm, texto
    for linha in m.group(1).split('\n'):
        if ':' in linha:
            k, v = linha.split(':', 1)
            fm[k.strip()] = v.strip()
    return fm, m.group(2)


def main():
    with io.open(MANIFEST, encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        regs = list(rd)

    mudou = trans = prob = 0
    for r in regs:
        md = os.path.join(RAIZ, r['transcricao'])
        if not os.path.exists(md):
            continue
        fm, corpo = front_matter(md)
        novo = dict(r)
        presid = (fm.get('presidencia', '') + ' ' + fm.get('titulo', '')).lower()
        eh_regional = (fm.get('tribunal') == 'regional'
                       or 'ataulpho' in presid or 'regional' in presid)
        prob = fm.get('problema', '')
        # anexos truncados / divergencias do impresso NAO invalidam a ata (o
        # corpo foi transcrito) — seguem a classificacao normal; so duplicata
        # e cabecalho ausente ficam fora do docx.
        bloqueia = prob and not any(
            k in prob for k in ('continua_fora', 'divergencia'))
        if 'duplicata' in prob:
            novo['status'] = 'duplicada'
        elif bloqueia:
            novo['status'] = 'problema'
        elif eh_regional:
            novo['status'] = 'regional'
        elif r['status'] in ('pendente', 'problema'):
            novo['status'] = 'transcrita'
        novo['confianca'] = fm.get('confianca', r['confianca'])
        if fm.get('titulo'):
            novo['titulo'] = fm['titulo']
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', fm.get('data_sessao', '')):
            novo['data_sessao'] = fm['data_sessao']
            # volume mal-arquivado: a data lida na imagem manda no ano do docx
            ano_real = fm['data_sessao'][:4]
            if 1932 <= int(ano_real) <= 1937 and ano_real != novo['ano']:
                novo['ano'] = ano_real
        if fm.get('num', '').isdigit():
            novo['num'] = fm['num']
        if fm.get('tipo') in ('ordinaria', 'extraordinaria'):
            novo['tipo'] = fm['tipo']
        parecer = os.path.join(RAIZ, r['ano'], f'{r["ata_id"]}.parecer.json')
        if fm.get('revisao_aplicada'):
            novo['status'] = 'final'
        elif novo['status'] == 'transcrita' and os.path.exists(parecer):
            novo['status'] = 'revisada'
            novo['parecer'] = os.path.join(r['ano'],
                                           f'{r["ata_id"]}.parecer.json')
            try:
                with io.open(parecer, encoding='utf-8') as f:
                    novo['fidelidade'] = str(
                        json.load(f).get('fidelidade_0_10', ''))
            except Exception:
                pass
        if novo != r:
            r.update(novo)
            mudou += 1

    # reconciliacao: o OCR do indice as vezes leu a data da caixa "RECURSOS"
    # (sessoes futuras) e nao do cabecalho, gerando ata_ids diferentes para a
    # MESMA ata fisica. Agrupa os md prontos por (data_sessao, arquivo BE,
    # tribunal); grupos com >1 mantem 1 campeao e rebaixa os outros.
    def corpo_len(r):
        p = os.path.join(RAIZ, r['transcricao'])
        if not os.path.exists(p):
            return 0
        _, corpo = front_matter(p)
        return len(corpo)

    grupos = {}
    for r in regs:
        if r['status'] not in ('transcrita', 'revisada', 'final', 'regional'):
            continue
        if r['data_sessao'] in ('', 'semdata') or '?' in r['data_sessao']:
            continue
        grupos.setdefault(
            (r['data_sessao'], r['arquivo'], r['status'] == 'regional'),
            []).append(r)
    recon = 0
    ordem = {'final': 3, 'revisada': 2, 'transcrita': 1, 'regional': 1}
    for chave, grupo in grupos.items():
        if len(grupo) < 2:
            continue
        # salvaguarda: se o grupo tem tipos distintos AMBOS preenchidos
        # (ordinaria x extraordinaria), sao sessoes diferentes -> nao funde
        tipos = {r['tipo'] for r in grupo if r['tipo']}
        if len(tipos) > 1:
            continue
        campeao = max(grupo, key=lambda r: (ordem.get(r['status'], 0),
                                            r['num'].isdigit(), corpo_len(r)))
        for r in grupo:
            if r is not campeao:
                r['status'] = 'duplicada'
                r['parecer'] = f'reconciliada->{campeao["ata_id"]}'
                recon += 1
    if recon:
        print(f'reconciliadas (duplicatas cruzadas de OCR): {recon}')

    with io.open(MANIFEST, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(regs)

    por = {}
    for r in regs:
        por[r['status']] = por.get(r['status'], 0) + 1
    print(f'{mudou} linhas atualizadas | status: {por}')


if __name__ == '__main__':
    main()
