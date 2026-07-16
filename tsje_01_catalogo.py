# -*- coding: utf-8 -*-
"""tsje_01: cataloga os PDFs de Boletins Eleitorais extraidos em D:\\TSJE_TRABALHO.

Por arquivo: sha1 binario, texto OCR (fitz), sha1 do texto normalizado, datas
(pasta / nome / cabecalho impresso), numero do BE e as ATAS detectadas, cada uma
atribuida ao tribunal da secao corrente (Superior x Regional).

A deteccao de atas e por LINHA em caixa alta (>=70% maiusculas): cabecalhos de
secao ("TRIBUNAL SUPERIOR...", "TRIBUNAL REGIONAL...") mudam o estado corrente;
linhas "ATA DA Na SESSAO..." emitem uma ata. Mencoes em corpo de texto ("aprovada
a ata da sessao anterior", "consulta do Tribunal Regional...") sao caixa mista e
nao disparam. A normalizacao (sem acentos/espacos) resolve o OCR espacado
("A T A D A 6a SESSÃO" -> "atada6asessao").

Cache retomavel (sha1+texto) chaveado por relpath|size|mtime em tsje_cache.tsv.

Uso: python tsje_01_catalogo.py [--ano 1932] [--recalc]
Saida: tsje_catalogo.tsv (uma linha por PDF varrido)
"""
import argparse
import hashlib
import io
import os
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

ROOT = r"D:\TSJE_TRABALHO\extraido"
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(r"D:\TSJE_TRABALHO", 'tsje_cache2.tsv')
SAIDA = os.path.join(BASE, 'tsje_catalogo.tsv')
MIN_NORM = 200  # abaixo disso considera "sem camada de OCR util"

MESES3 = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
          'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

RE_DATA_NORM = re.compile(r'(\d{1,2})de([a-z]{3,12})de(19\d{2})')
RE_CABEC = re.compile(r'riodejaneiro(\d{1,2})de([a-z]{3,12})de(19\d{2})(?:no?(\d{1,3}))?')
RE_ATA_NUM = re.compile(r'(\d{1,3})a?sessao')
# "ATA(S) (DA 6a) SESSAO..." com sessao logo apos (rejeita "ATAULPHO...")
RE_ATA_CABEC = re.compile(r'^atas?(?:da|de|das)?\d{0,3}a?.{0,6}sess')
RE_BE_NOME = re.compile(r'be\s*(\d{1,2})-(\d{1,2})-(\d{4})', re.I)
RE_LOTE_NOME = re.compile(r'(\d{4})_boletim_eleitoral_a(\d+)_n(\d+)', re.I)

_lock = threading.Lock()


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def normalizar(texto):
    t = unicodedata.normalize('NFD', texto.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^0-9a-z]', '', t)


def data_iso(dia, mes_txt, ano):
    mes = MESES3.get(normalizar(mes_txt)[:3])
    if not mes:
        return ''
    try:
        d, a = int(dia), int(ano)
    except ValueError:
        return ''
    if not (1 <= d <= 31 and 1932 <= a <= 1937):
        return ''
    return f'{a:04d}-{mes:02d}-{d:02d}'


def sha1_de(path):
    h = hashlib.sha1()
    with open(lp(path), 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extrair_texto(path):
    """(texto, paginas) via PyMuPDF; (None, 0) em falha."""
    import fitz
    try:
        with fitz.open(lp(path)) as doc:
            # paginas separadas por linha '\f' (rastreio de pagina nas atas)
            return '\n\f\n'.join(p.get_text('text') for p in doc), doc.page_count
    except Exception:
        return None, 0


def esc(t):
    return t.replace('\\', '\\\\').replace('\t', '\\t').replace('\r', '').replace('\n', '\\n')


def unesc(t):
    out, i, n = [], 0, len(t)
    while i < n:
        c = t[i]
        if c == '\\' and i + 1 < n:
            nx = t[i + 1]
            if nx == 'n':
                out.append('\n'); i += 2; continue
            if nx == 't':
                out.append('\t'); i += 2; continue
            if nx == '\\':
                out.append('\\'); i += 2; continue
        out.append(c)
        i += 1
    return ''.join(out)


def carregar_cache():
    cache = {}
    if os.path.exists(CACHE):
        with io.open(CACHE, encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 6:
                    rel, size, mtime, sha1, paginas, texto = parts
                    cache[(rel, size, mtime)] = (sha1, paginas, texto)
    return cache


def detectar_atas(texto):
    """Atas (tribunal, tipo, num, data_iso, pagina) pelo estado de secao corrente."""
    atas, vistos = [], set()
    estado = ''
    pagina = 1
    linhas = texto.split('\n')
    for i, linha in enumerate(linhas):
        if linha == '\f':
            pagina += 1
            continue
        letras = [c for c in linha if c.isalpha()]
        if not letras or sum(c.isupper() for c in letras) / len(letras) < 0.7:
            continue
        nl = normalizar(linha)
        if nl.startswith('tribunalsuperior'):
            estado = 'superior'
            continue
        if nl.startswith('tribunalregional'):
            estado = 'regional'
            continue
        # cabecalho de ata: linha que comeca com "ATA..." OU linha com
        # "SESSAO (EXTRA)ORDINARIA/EM DD DE MES" (OCR que perdeu o "ATA DA")
        eh_ata = nl.startswith('ata') and nl != 'atas'
        eh_sessao = ('sessao' in nl and len(nl) < 90
                     and ('ordin' in nl or RE_DATA_NORM.search(normalizar(
                         ' '.join(linhas[i:i + 3]))[:160])))
        if not (eh_ata or eh_sessao):
            continue
        h = normalizar(' '.join(linhas[i:i + 4]))[:200]
        if 'sessao' not in h:
            continue
        if eh_ata and not eh_sessao and not RE_ATA_CABEC.search(h):
            continue
        m = RE_ATA_NUM.search(h[:h.find('sessao') + 6])
        num = m.group(1) if m else ''
        pos_sessao = h.find('sessao')
        janela = h[pos_sessao:pos_sessao + 40]
        if 'extraord' in janela:
            tipo = 'extraordinaria'
        elif 'rdin' in janela:
            tipo = 'ordinaria'
        else:
            tipo = ''
        md = RE_DATA_NORM.search(h)
        data = data_iso(md.group(1), md.group(2), md.group(3)) if md else ''
        chave = (estado, tipo, num, data)
        if chave in vistos:
            continue
        vistos.add(chave)
        atas.append((estado or 'desconhecido', tipo, num, data, str(pagina)))
    return atas


def parse_caminho(rel):
    """(ano_zip, data_pasta, id_pasta) a partir de BEs AAAA\\MES[\\DD[\\ID]]\\arq.pdf"""
    partes = rel.split('\\')
    ano = mes = dia = idp = ''
    m = re.match(r'bes(\d{4})', normalizar(partes[0]))
    if m:
        ano = m.group(1)
    if len(partes) > 2:
        mes = MESES3.get(normalizar(partes[1])[:3], '')
    if len(partes) > 3 and re.fullmatch(r'\d{1,2}', partes[2]):
        dia = partes[2]
    for p in partes[3:-1]:
        if re.fullmatch(r'\d{4,6}', p):
            idp = p
    data = ''
    if ano and mes:
        data = f'{int(ano):04d}-{mes:02d}-{int(dia):02d}' if dia else f'{int(ano):04d}-{mes:02d}'
    return ano, data, idp


def parse_nome(nome):
    """(data_nome, n_nome, dia_nome) do nome do arquivo."""
    m = RE_BE_NOME.search(nome)
    if m:
        d, mm, a = m.groups()
        try:
            return f'{int(a):04d}-{int(mm):02d}-{int(d):02d}', '', ''
        except ValueError:
            pass
    n_nome = dia = ''
    m = RE_LOTE_NOME.search(nome)
    if m:
        n_nome = m.group(3)
    m = re.match(r'(\d{1,2})[_ -]\d{4}_boletim', nome, re.I)
    if m and 1 <= int(m.group(1)) <= 31:
        dia = m.group(1)
    return '', n_nome, dia


def processar(rel, cache, recalc):
    path = os.path.join(ROOT, rel)
    st = os.stat(lp(path))
    chave = (rel, str(st.st_size), str(int(st.st_mtime)))
    hit = None if recalc else cache.get(chave)
    if hit:
        sha1, paginas, texto = hit[0], hit[1], unesc(hit[2])
    else:
        sha1 = sha1_de(path)
        texto, pg = extrair_texto(path)
        paginas = str(pg)
        with _lock, io.open(CACHE, 'a', encoding='utf-8') as f:
            f.write('\t'.join([rel, chave[1], chave[2], sha1, paginas,
                               esc(texto or '')]) + '\n')
    flags = []
    if texto is None or texto == '':
        texto = ''
        flags.append('erro_pdf' if paginas == '0' else 'sem_texto')
    norm = normalizar(texto)
    if texto and len(norm) < MIN_NORM:
        flags.append('sem_texto')
    sha1_norm = hashlib.sha1(norm.encode()).hexdigest() if norm else ''

    ano_zip, data_pasta, id_pasta = parse_caminho(rel)
    data_nome, n_nome, dia_nome = parse_nome(os.path.basename(rel))
    if not data_nome and dia_nome and len(data_pasta) == 7:
        data_nome = f'{data_pasta}-{int(dia_nome):02d}'
    data_texto = n_texto = ''
    mc = RE_CABEC.search(norm)
    if mc:
        data_texto = data_iso(mc.group(1), mc.group(2), mc.group(3))
        n_texto = mc.group(4) or ''
    atas = detectar_atas(texto)
    n_sup = sum(1 for a in atas if a[0] == 'superior')
    n_reg = sum(1 for a in atas if a[0] == 'regional')
    n_inc = len(atas) - n_sup - n_reg
    if (data_pasta and data_texto and len(data_pasta) == 10
            and data_pasta != data_texto):
        flags.append('data_divergente')
    if not (data_pasta or data_texto or data_nome):
        flags.append('sem_data')
    return [rel, str(st.st_size), sha1, paginas, str(len(norm)), sha1_norm,
            ano_zip, data_pasta, data_nome, n_nome, data_texto, n_texto,
            id_pasta, str(n_sup), str(n_reg), str(n_inc),
            ';'.join(':'.join(a) for a in atas), ','.join(sorted(set(flags)))]


COLUNAS = ['relpath', 'size', 'sha1', 'paginas', 'len_norm', 'sha1_norm',
           'ano_zip', 'data_pasta', 'data_nome', 'n_nome', 'data_texto',
           'n_texto', 'id_pasta', 'n_atas_sup', 'n_atas_reg', 'n_atas_incertas',
           'atas', 'flags']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ano', help='limita a BEs <ano>')
    ap.add_argument('--recalc', action='store_true', help='ignora cache')
    args = ap.parse_args()

    rels = []
    for dirpath, _, filenames in os.walk(lp(ROOT)):
        for name in filenames:
            if not name.lower().endswith('.pdf'):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), lp(ROOT))
            if args.ano and not rel.startswith(f'BEs {args.ano}'):
                continue
            rels.append(rel)
    rels.sort()
    if not rels:
        print('Nenhum PDF encontrado (rode tsje_00_extrair.py antes).')
        sys.exit(1)

    cache = carregar_cache()
    t0 = time.time()
    feitos = [0]
    linhas = {}

    def worker(rel):
        try:
            linhas[rel] = processar(rel, cache, args.recalc)
        except Exception as e:
            linhas[rel] = [rel] + [''] * 16 + [f'erro:{type(e).__name__}']
        with _lock:
            feitos[0] += 1
            if feitos[0] % 50 == 0:
                print(f'  {feitos[0]}/{len(rels)} ({time.time()-t0:.0f}s)')

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(worker, rels))

    with io.open(SAIDA, 'w', encoding='utf-8') as f:
        f.write('\t'.join(COLUNAS) + '\n')
        for rel in rels:
            f.write('\t'.join(linhas[rel]) + '\n')

    com_sup = sum(1 for r in linhas.values() if r[13] not in ('', '0'))
    sem_texto = sum(1 for r in linhas.values() if 'sem_texto' in r[17])
    print(f'{len(rels)} PDFs catalogados em {time.time()-t0:.0f}s -> {SAIDA}')
    print(f'  com ata do TSJE: {com_sup} | sem texto OCR: {sem_texto}')


if __name__ == '__main__':
    main()
