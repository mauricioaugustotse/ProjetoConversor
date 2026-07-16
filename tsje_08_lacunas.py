# -*- coding: utf-8 -*-
"""tsje_08: diagnostica os buracos na sequencia de sessoes do TSJE e prepara
insumos (PNGs + lote) para as lacunas cujo boletim EXISTE no acervo.

Para cada ano/tipo, acha os numeros de sessao ausentes entre os transcritos.
Depois cruza com o texto OCR de TODOS os PDFs (cache tsje_cache2.tsv): se um
cabecalho "Na SESSAO ..." do numero faltante aparece em algum PDF, essa ata
esta no acervo e pode ser recuperada — renderiza a pagina e gera lote_lacunas.

Buracos cujo cabecalho NAO aparece em nenhum PDF = boletim provavelmente
ausente do acervo (nao veio do Drive) — vao para o relatorio como tal.

Uso: python tsje_08_lacunas.py
Saidas: D:\\TSJE_TRANSCRICOES\\_lote_lacunas.txt
        D:\\TSJE_TRANSCRICOES\\_diagnostico_completude.csv
"""
import io
import os
import re
import unicodedata
from collections import defaultdict

BANCO = r'D:\TSJE_ATAS'
RAIZ = r'D:\TSJE_TRANSCRICOES'
CACHE = r'D:\TSJE_TRABALHO\tsje_cache2.tsv'
EXTRAIDO = r'D:\TSJE_TRABALHO\extraido'
LOTE = os.path.join(RAIZ, '_lote_lacunas.txt')
DIAG = os.path.join(RAIZ, '_diagnostico_completude.csv')
DPI = 200

MESES3 = {1: 'jan', 2: 'fev', 3: 'mar', 4: 'abr', 5: 'mai', 6: 'jun',
          7: 'jul', 8: 'ago', 9: 'set', 10: 'out', 11: 'nov', 12: 'dez'}


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def caminho_fisico(rel):
    """Resolve a chave do cache: 'BEs ...' -> extraido; 'ATOM ...' -> banco."""
    if rel.startswith('ATOM '):
        return os.path.join(BANCO, rel[5:])
    return os.path.join(EXTRAIDO, rel)


def norm(t):
    t = unicodedata.normalize('NFD', t.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^0-9a-z]', '', t)


def unesc(t):
    return t.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')


def transcritos():
    """(ano -> {('o'|'e', num)}) das atas do TSJE ja transcritas."""
    ords = defaultdict(set)
    for ano in os.listdir(RAIZ):
        d = os.path.join(RAIZ, ano)
        if not (os.path.isdir(d) and ano.isdigit()):
            continue
        for nome in os.listdir(d):
            if not nome.endswith('.md'):
                continue
            with io.open(os.path.join(d, nome), encoding='utf-8') as f:
                cab = f.read(700)
            if not re.search(r'tribunal:\s*superior', cab):
                continue
            if re.search(r'problema:\s*(duplicata|cabecalho)', cab):
                continue
            mn = re.search(r'\nnum:\s*(\d+)', cab)
            if not mn:
                continue
            mt = re.search(r'\ntipo:\s*(\w+)', cab)
            md = re.search(r'data_sessao:\s*(\d{4})', cab)
            y = md.group(1) if md else ano
            t = 'e' if (mt and mt.group(1) == 'extraordinaria') else 'o'
            ords[y].add((t, int(mn.group(1))))
    return ords


def indexar_ocr():
    """Constroi indice: para cada PDF, o texto normalizado (para busca de
    cabecalhos) e a lista de (pagina, texto_da_pagina)."""
    pdfs = {}
    with io.open(CACHE, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 6:
                pdfs[p[0]] = unesc(p[5])
    return pdfs


def _re_cab(num, tipo):
    """Regex do cabecalho de ata: numero (com fronteira, sem substring de
    numero maior) + 'sessao ord/extra'. Ex.: rejeita '5' dentro de '15'."""
    tp = 'extraordinaria' if tipo == 'e' else 'ordinaria'
    return re.compile(r'(?<!\d)' + str(num) + r'a?sessao' + tp)


def achar_cabecalho(texto, num, tipo):
    """True SO se ha um cabecalho de ata REAL: a linha do numero+sessao e,
    perto dela, a marca de ata ('presidencia' ou 'presentesosjuizes').
    Isso rejeita resumos da secao JULGAMENTOS (que citam o numero mas nao
    tem o preambulo formal da ata)."""
    rx = _re_cab(num, tipo)
    linhas = texto.split('\n')
    nl = [norm(l) for l in linhas]
    for i, l in enumerate(nl):
        if rx.search(l):
            janela = ''.join(nl[i:i + 12])
            if 'presidencia' in janela or 'presentesosjuizes' in janela \
                    or 'presentesos' in janela:
                return True
    return False


def pagina_do_cabecalho(texto, num, tipo):
    """Pagina (1-based) do cabecalho de ata REAL (com preambulo proximo)."""
    rx = _re_cab(num, tipo)
    linhas = texto.split('\n')
    nl = [norm(l) for l in linhas]
    pag = 1
    for i, linha in enumerate(linhas):
        if linha == '\f':
            pag += 1
            continue
        if rx.search(nl[i]):
            janela = ''.join(nl[i:i + 12])
            if 'presidencia' in janela or 'presentesos' in janela:
                return pag
    return None


def main():
    feitos = transcritos()
    pdfs = indexar_ocr()
    # pre-normaliza (uma vez) o texto de cada PDF por ano
    por_ano_pdf = defaultdict(list)
    for rel, texto in pdfs.items():
        m = re.match(r'(?:BEs|ATOM) (\d{4})', rel)
        if m:
            por_ano_pdf[m.group(1)].append((rel, texto, norm(texto)))

    achaveis = []   # (ano, tipo, num, rel_pdf, pagina)
    ausentes = []   # (ano, tipo, num) sem boletim no acervo
    for ano in sorted(feitos):
        for tp in ('o', 'e'):
            nums = sorted(n for t, n in feitos[ano] if t == tp)
            if not nums:
                continue
            falt = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
            for num in falt:
                achou = None
                for rel, texto, tn in por_ano_pdf.get(ano, []):
                    # pre-filtro frouxo barato; o match REAL exige pagina
                    # do cabecalho formal (detector estrito por linha) —
                    # sem parar no 1o PDF com mera resenha
                    if not achar_cabecalho(tn, num, tp):
                        continue
                    pag = pagina_do_cabecalho(texto, num, tp)
                    if pag:
                        achou = (rel, pag)
                        break
                if achou:
                    achaveis.append((ano, tp, num, achou[0], achou[1]))
                else:
                    ausentes.append((ano, tp, num))

    # diagnostico
    with io.open(DIAG, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('ano,tipo,num,situacao,pdf,pagina\n')
        for ano, tp, num, rel, pag in achaveis:
            t = 'extraordinaria' if tp == 'e' else 'ordinaria'
            f.write(f'{ano},{t},{num},RECUPERAVEL,{rel},{pag}\n')
        for ano, tp, num in ausentes:
            t = 'extraordinaria' if tp == 'e' else 'ordinaria'
            f.write(f'{ano},{t},{num},BOLETIM_AUSENTE_NO_ACERVO,,\n')

    # renderiza PNGs e monta lote das recuperaveis
    import fitz
    os.makedirs(os.path.join(RAIZ, '_paginas_lac'), exist_ok=True)
    zoom = DPI / 72.0
    blocos = []
    for ano, tp, num, rel, pag in achaveis:
        if not pag:
            continue
        pdf = caminho_fisico(rel)
        # ident UNICO por caminho (evita colisao de basename '00000020.pdf'
        # que existe em varios boletins): usa as 2 ultimas pastas + arquivo
        partes = re.split(r'[\\/]', rel)
        ident = re.sub(r'\W', '', '_'.join(partes[-3:]))[:24]
        pngs = []
        try:
            doc = fitz.open(lp(pdf))
            for p in range(pag, min(pag + 2, doc.page_count) + 1):
                rela = os.path.join('_paginas_lac', f'{ident}_p{p:03d}.png')
                alvo = os.path.join(RAIZ, rela)
                if not os.path.exists(lp(alvo)):
                    pix = doc[p - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                                colorspace=fitz.csGRAY)
                    pix.save(lp(alvo))
                pngs.append(os.path.join(RAIZ, rela))
            doc.close()
        except Exception as e:
            print(f'  erro render {rel} p{pag}: {e}')
            continue
        t = 'extraordinaria' if tp == 'e' else 'ordinaria'
        aid = f'lacuna-{ano}-{tp}{num}-{ident}'
        blocos.append((aid, ano, num, t, rel, pag, pngs))

    with io.open(LOTE, 'w', encoding='utf-8') as f:
        for aid, ano, num, t, rel, pag, pngs in blocos:
            f.write('=== ATA ===\n')
            f.write(f'ata_id: {aid}\n')
            f.write(f'titulo_estimado: {num}a SESSAO {t.upper()} de {ano}\n')
            f.write(f'ano: {ano} | num: {num} | tipo: {t}\n')
            f.write(f'origem_pdf: {rel} | pagina_do_cabecalho: {pag}\n')
            f.write('imagens:\n')
            for p in pngs:
                f.write(f'  {p}\n')
            f.write(f'gravar_em: {os.path.join(RAIZ, ano, aid + ".md")}\n\n')

    print(f'RECUPERAVEIS (boletim no acervo): {len(achaveis)} -> {LOTE}')
    print(f'BOLETIM AUSENTE NO ACERVO: {len(ausentes)}')
    print(f'diagnostico completo -> {DIAG}')
    porano = defaultdict(lambda: [0, 0])
    for ano, tp, num, rel, pag in achaveis:
        porano[ano][0] += 1
    for ano, tp, num in ausentes:
        porano[ano][1] += 1
    for ano in sorted(porano):
        print(f'  {ano}: {porano[ano][0]} recuperaveis, '
              f'{porano[ano][1]} ausentes no acervo')


if __name__ == '__main__':
    main()
