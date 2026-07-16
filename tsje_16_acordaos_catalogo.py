# -*- coding: utf-8 -*-
"""tsje_16: cataloga os ACORDAOS publicados nos Boletins Eleitorais a partir
do OCR (cache tsje_cache2.tsv, com \\f de pagina).

Estrutura tipica da secao "JURISPRUDENCIA DO TRIBUNAL SUPERIOR" (1932-37):
    N . 22
    NATUREZA DO PROCESSO — <descricao>       -> classe/UF
    Juiz RELATOR — O Sr. ministro <nome>
    Acórdão
    <ementa>
    VISTOS, relatados e discutidos ...
    Acordam os juizes do Tribunal Superior ...
    Tribunal Superior ..., em <data>. — <assinaturas> (Decisão unanime.)
Ha tambem cabecalhos avulsos "ACCORDÃO NO RECURSO PARCIAL N. 56" (1936+).

Unidade comeca em "N. <num>" (com NATUREZA DO PROCESSO por perto) ou em
cabecalho ACCORDÃO; termina na proxima unidade ou na troca de secao.
Secoes de Tribunal REGIONAL sao ignoradas.

Saidas (D:\\TSJE_TRABALHO):
  tsje_acordaos_catalogo.csv  — metadados por regex + pdf/pagina
  tsje_acordaos_texto.jsonl   — texto OCR bruto de cada unidade

Uso: python tsje_16_acordaos_catalogo.py
"""
import csv
import io
import json
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding='utf-8')

CACHE = r'D:\TSJE_TRABALHO\tsje_cache2.tsv'
TRAB = r'D:\TSJE_TRABALHO'
CSV_OUT = os.path.join(TRAB, 'tsje_acordaos_catalogo.csv')
JSONL_OUT = os.path.join(TRAB, 'tsje_acordaos_texto.jsonl')

MESES = {'janeiro': 1, 'fevereiro': 2, 'marco': 3, 'março': 3, 'abril': 4,
         'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8, 'setembro': 9,
         'outubro': 10, 'novembro': 11, 'dezembro': 12}

UFS = {
    'acre': 'AC', 'alagoas': 'AL', 'amazonas': 'AM', 'bahia': 'BA',
    'ceara': 'CE', 'distritofederal': 'DF', 'espiritosanto': 'ES',
    'goyaz': 'GO', 'goias': 'GO', 'maranhao': 'MA', 'mattogrosso': 'MT',
    'matogrosso': 'MT', 'minasgeraes': 'MG', 'minasgerais': 'MG',
    'para': 'PA', 'parahyba': 'PB', 'paraiba': 'PB', 'parana': 'PR',
    'pernambuco': 'PE', 'piauhy': 'PI', 'piaui': 'PI',
    'riodejaneiro': 'RJ', 'riograndedonorte': 'RN',
    'riograndedosul': 'RS', 'santacatharina': 'SC', 'santacatarina': 'SC',
    'saopaulo': 'SP', 'sergipe': 'SE', 'territoriodoacre': 'AC',
}


def norm(t):
    t = unicodedata.normalize('NFD', t.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^0-9a-z]', '', t)


def unesc(t):
    return t.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')


def caixa_alta(l):
    up = sum(1 for c in l if c.isupper())
    al = sum(1 for c in l if c.isalpha())
    return al >= 4 and up / al >= 0.7


RE_N = re.compile(r'^n\.?\s*(\d{1,4})$')
# "Processo n. 273" / "Processo JI. 368" (OCR) — norm: processon273
RE_PROC_N = re.compile(r'^processo[a-z]{0,3}(\d{1,4})$')
RE_CAB_AC = re.compile(r'^ac?c?ordaon(?:o|a)?(.*?)n?(\d{1,4})?$')


def data_iso(trecho_norm, ano_hint):
    """'em10deagostode1932' -> 1932-08-10 (busca no texto normalizado)."""
    m = re.search(r'em(\d{1,2})de([a-z]+)de(\d{4})', trecho_norm)
    if not m:
        return ''
    d, mes, a = int(m.group(1)), m.group(2), int(m.group(3))
    mn = MESES.get(mes)
    if mn and 1 <= d <= 31 and 1930 <= a <= 1938:
        return f'{a:04d}-{mn:02d}-{d:02d}'
    return ''


def extrair_meta(linhas, ano_be):
    """Metadados por regex de uma unidade (lista de linhas)."""
    texto = '\n'.join(linhas)
    tn = norm(texto)
    meta = {'natureza': '', 'relator': '', 'data_julgamento': '',
            'unanime': '', 'uf': ''}
    m = re.search(r'NATUREZA\s+DO\s+PROCESSO\s*[—\-–]?\s*(.{0,300}?)'
                  r'(?:\n\s*\n|Juiz|RELATOR)', texto, re.S | re.I)
    if m:
        meta['natureza'] = re.sub(r'\s+', ' ', m.group(1)).strip()[:280]
        # formato 1933+: "Natureza do processo — <UF> — <classe> — ..."
        pedacos = re.split(r'\s*[—–]\s*|\s+-\s+', meta['natureza'])
        if pedacos:
            uf = UFS.get(norm(pedacos[0]).replace('eoutros', ''))
            if uf:
                meta['uf'] = uf
    m = re.search(r'RELATOR\s*[—\-–]?\s*(?:O\s+Sr\.?\s*)?'
                  r'(?:ministro|desembargador|dr\.?|doutor)?\s*'
                  r'([A-ZÀ-Ü][^\n—]{3,60})', texto, re.I)
    if m:
        meta['relator'] = re.sub(r'\s+', ' ', m.group(1)).strip(' .-')[:60]
    meta['data_julgamento'] = data_iso(tn, ano_be)
    if 'decisaounanime' in tn or 'unanime' in tn[-400:]:
        meta['unanime'] = 'sim'
    for chave, sigla in UFS.items():
        if 'estadod' + chave[:1] in tn or 'estadode' + chave in tn \
                or 'estadodo' + chave in tn or 'estadoda' + chave in tn:
            meta['uf'] = sigla
            break
    if not meta['uf'] and 'distritofederal' in tn:
        meta['uf'] = 'DF'
    return meta


def classe_da_natureza(natureza, cab):
    t = norm(natureza + ' ' + cab)
    for pat, cl in [('consulta', 'CONSULTA'), ('representac', 'REPRESENTAÇÃO'),
                    ('reclamac', 'RECLAMAÇÃO'), ('habeascorpus', 'HABEAS CORPUS'),
                    ('recursoparcial', 'RECURSO PARCIAL'),
                    ('recursoeleitoral', 'RECURSO ELEITORAL'),
                    ('apelacaocriminal', 'APELAÇÃO CRIMINAL'),
                    ('recurso', 'RECURSO'), ('registro', 'REGISTRO'),
                    ('communicacao', 'COMUNICAÇÃO'),
                    ('comunicacao', 'COMUNICAÇÃO')]:
        if pat in t:
            return cl
    return 'PROCESSO'


def processar_pdf(rel, texto):
    """Retorna lista de unidades de acordao do PDF."""
    m = re.search(r'(?:BEs|ATOM) (\d{4})', rel)
    ano_be = m.group(1) if m else ''
    linhas = texto.split('\n')
    nl = [norm(l) for l in linhas]

    unidades = []
    pag = 1
    em_juris = False
    atual = None       # dict da unidade em construcao

    def fechar(i_fim):
        nonlocal atual
        if atual and i_fim - atual['i_ini'] >= 4:
            atual['linhas'] = linhas[atual['i_ini']:i_fim]
            unidades.append(atual)
        atual = None

    for i, l in enumerate(linhas):
        if l == '\f':
            pag += 1
            continue
        n = nl[i]
        if not n:
            continue
        # troca de secao (linha caixa alta)
        if caixa_alta(l.strip()):
            if n.startswith('jurisprudencia'):
                if 'regiona' in n:
                    fechar(i)
                    em_juris = False
                else:
                    em_juris = True
                continue
            # cabecalho ACCORDÃO avulso inicia unidade mesmo fora da secao
            mc = re.match(r'^ac?c?ordaon', n)
            if mc and 'sessao' not in n:
                fechar(i)
                mnum = re.search(r'n?(\d{1,4})$', n)
                atual = {'i_ini': i, 'pag': pag, 'num': mnum.group(1)
                         if mnum else '', 'cab': l.strip()[:120]}
                em_juris = True
                continue
            if ('tribunalregional' in n or n.startswith('atada')
                    or n.startswith('edita') or n.startswith('secretaria')
                    or n.startswith('tribunaesregiona')
                    or n.startswith('tribunaisregiona')):
                fechar(i)
                em_juris = False
                continue
        if not em_juris:
            continue
        # inicio "N . 22" / "Processo n. 273" com NATUREZA/RELATOR perto
        mn_ = RE_N.match(n) or RE_PROC_N.match(n)
        if mn_:
            janela = ''.join(nl[i + 1:i + 10])
            if 'naturezadoprocesso' in janela or 'relator' in janela:
                fechar(i)
                atual = {'i_ini': i, 'pag': pag, 'num': mn_.group(1),
                         'cab': ''}
                continue
        # "Natureza do processo — ..." sem linha-numero antes tambem abre
        if n.startswith('naturezadoprocesso') and (
                atual is None or i - atual['i_ini'] > 12):
            fechar(i)
            atual = {'i_ini': i, 'pag': pag, 'num': '', 'cab': ''}
    fechar(len(linhas))

    regs = []
    partes = re.split(r'[\\/]', rel)
    ident = re.sub(r'\W', '', '_'.join(partes[-3:]))[:24]
    for seq, u in enumerate(unidades, 1):
        meta = extrair_meta(u['linhas'], ano_be)
        classe = classe_da_natureza(meta['natureza'], u.get('cab', ''))
        regs.append({
            # id estavel: pdf + ordem da unidade dentro do pdf (re-execucoes
            # com acervo maior nao mudam ids de PDFs ja processados)
            'acordao_id': f'ac-{ident}-{seq:02d}',
            'pdf': rel, 'ano_be': ano_be, 'pagina': u['pag'],
            'numero_processo': u['num'], 'classe': classe,
            'uf': meta['uf'], 'relator': meta['relator'],
            'data_julgamento': meta['data_julgamento'],
            'unanime': meta['unanime'],
            'natureza': meta['natureza'],
            'n_linhas': len(u['linhas']),
            'texto': '\n'.join(u['linhas']),
        })
    return regs


def main():
    todos = []
    n_pdf = 0
    with io.open(CACHE, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) != 6:
                continue
            n_pdf += 1
            regs = processar_pdf(p[0], unesc(p[5]))
            todos.extend(regs)

    campos = ['acordao_id', 'pdf', 'ano_be', 'pagina', 'numero_processo',
              'classe', 'uf', 'relator', 'data_julgamento', 'unanime',
              'natureza', 'n_linhas']
    with io.open(CSV_OUT, 'w', encoding='utf-8-sig', newline='') as f, \
            io.open(JSONL_OUT, 'w', encoding='utf-8') as fj:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
        w.writeheader()
        for r in todos:
            w.writerow(r)
            fj.write(json.dumps(r, ensure_ascii=False) + '\n')

    from collections import Counter
    por_ano = Counter(r['ano_be'] for r in todos)
    por_classe = Counter(r['classe'] for r in todos)
    com_num = sum(1 for r in todos if r['numero_processo'])
    com_rel = sum(1 for r in todos if r['relator'])
    com_data = sum(1 for r in todos if r['data_julgamento'])
    print(f'PDFs varridos: {n_pdf} | acordaos catalogados: {len(todos)}')
    print(f'  com numero: {com_num} | com relator: {com_rel} | '
          f'com data: {com_data}')
    print('  por ano: ' + ', '.join(f'{a}={n}' for a, n
                                    in sorted(por_ano.items())))
    print('  por classe: ' + ', '.join(f'{c}={n}' for c, n
                                       in por_classe.most_common(8)))
    print(f'catalogo -> {CSV_OUT}')
    print(f'textos   -> {JSONL_OUT}')


if __name__ == '__main__':
    main()
