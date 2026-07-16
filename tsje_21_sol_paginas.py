# -*- coding: utf-8 -*-
"""tsje_21: auxiliar gpt-5.6-sol le por IMAGEM as paginas cegas do acervo
(sem OCR util) e cataloga o que ha nelas: cabecalhos de ata (com sessao,
tipo, data e tribunal), secao ACTAS, acordaos.

Entrada: D:\\TSJE_TRABALHO\\tsje_paginas_cegas.csv (do tsje_20).
Render: PNG 200 DPI grayscale em D:\\TSJE_TRANSCRICOES\\_paginas_rev\\.
Cache idempotente por (pdf, pagina): tsje_visao_cache.jsonl.
Saida: tsje_visao_paginas.csv.

Uso: python tsje_21_sol_paginas.py [--limite N] [--so-pdf SUBSTR]
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fitz
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

# decisao de custo 16/07/2026: luna varre (~1/5 do custo do sol); terra
# reprocessa so as paginas que o luna marcou ilegiveis/duvidosas
# (--duvidosos-terra); sol descontinuado nesta tarefa
MODELO = 'gpt-5.6-luna'
TRAB = r'D:\TSJE_TRABALHO'
EXTRAIDO = r'D:\TSJE_TRABALHO\extraido'
BANCO = r'D:\TSJE_ATAS'
RENDER = r'D:\TSJE_TRANSCRICOES\_paginas_rev'
CEGAS = os.path.join(TRAB, 'tsje_paginas_cegas.csv')
CACHE = os.path.join(TRAB, 'tsje_visao_cache.jsonl')
SAIDA = os.path.join(TRAB, 'tsje_visao_paginas.csv')
DPI = 200

SYSTEM = """Você é um paleógrafo especialista em jornais oficiais brasileiros \
da década de 1930 (Boletim Eleitoral do Tribunal Superior de Justiça \
Eleitoral, 1932-1937). Vai receber a IMAGEM de UMA página. Analise-a e \
responda SOMENTE com JSON válido, sem markdown, no formato:
{"legivel": true|false,
 "cabecalho_pagina": "o que está impresso no topo (cidade, data, nº do BE) ou null",
 "atas": [{"num": <int|null>, "tipo": "ordinaria|extraordinaria|null",
           "data_impressa": "AAAA-MM-DD|null",
           "tribunal": "superior|regional|null",
           "e_ata_formal": true|false,
           "evidencia": "trecho curto do cabeçalho/preâmbulo impresso"}],
 "tem_secao_actas": true|false,
 "acordaos": [{"processo": "nº impresso|null", "classe": "...|null",
               "uf": "..|null"}],
 "conteudo_predominante": "atas|acordaos|resenha_julgamentos|alistamento|expediente|outro"}

REGRAS CRÍTICAS (armadilhas conhecidas deste acervo):
1. ATA FORMAL exige o cabeçalho "ATA DA Nª SESSÃO ..." seguido de preâmbulo \
("Aos ... dias" ou "Às ... horas, presentes os juízes...") e/ou linha de \
PRESIDÊNCIA. Uma RESENHA ("O Tribunal, em sua Nª sessão, resolveu: 1º...") \
NÃO é ata formal — registre em "atas" com e_ata_formal=false.
2. TRIBUNAL: só marque "superior" se a seção/página for do Tribunal \
SUPERIOR (presidência do ministro Hermenegildo de Barros em regra). Seções \
"TRIBUNAL REGIONAL DE ..." são "regional".
3. A página pode conter VÁRIAS atas — liste todas.
4. Use o ano do CABEÇALHO impresso da página, não deduza.
5. NÃO invente: se não conseguir ler um campo, use null. Se a imagem for \
ilegível/em branco, {"legivel": false, ...} com listas vazias."""

_lock = threading.Lock()


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def caminho_fisico(rel):
    if rel.startswith('ATOM '):
        return os.path.join(BANCO, rel[5:])
    return os.path.join(EXTRAIDO, rel)


BLACKLIST = os.path.join(TRAB, 'tsje_visao_render_blacklist.txt')
EM_CURSO = os.path.join(TRAB, 'tsje_visao_render_emcurso.txt')


def png_de(rel, pag):
    ident = re.sub(r'\W', '', '_'.join(re.split(r'[\\/]', rel)[-3:]))[:40]
    return os.path.join(RENDER, f'{ident}_p{pag:03d}.png')


def render_lote(fila):
    """Renderiza sequencialmente com blacklist automatica: o PDF corrente
    fica anotado em EM_CURSO antes do open; se o MuPDF abortar o processo,
    a retomada ve o arquivo e poe o PDF na blacklist (paginas dele saem da
    fila). Retorna a fila filtrada aos itens com PNG pronto."""
    black = set()
    if os.path.exists(BLACKLIST):
        black = {l.strip() for l in io.open(BLACKLIST, encoding='utf-8')
                 if l.strip()}
    if os.path.exists(EM_CURSO):
        rel = io.open(EM_CURSO, encoding='utf-8').read().strip()
        if rel and rel not in black:
            with io.open(BLACKLIST, 'a', encoding='utf-8') as f:
                f.write(rel + '\n')
            black.add(rel)
            print(f'  [blacklist] {rel} derrubou o render anterior')
        os.remove(EM_CURSO)

    zoom = DPI / 72.0
    por_pdf = {}
    for rel, pag in fila:
        por_pdf.setdefault(rel, []).append(pag)
    prontos = []
    for rel, pags in por_pdf.items():
        if rel in black:
            continue
        falta = [p for p in pags if not os.path.exists(lp(png_de(rel, p)))]
        if not falta:
            prontos += [(rel, p) for p in pags]
            continue
        with io.open(EM_CURSO, 'w', encoding='utf-8') as f:
            f.write(rel)
        try:
            doc = fitz.open(lp(caminho_fisico(rel)))
            for p in falta:
                if p <= doc.page_count:
                    pix = doc[p - 1].get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom),
                        colorspace=fitz.csGRAY)
                    pix.save(lp(png_de(rel, p)))
            doc.close()
            prontos += [(rel, p) for p in pags
                        if os.path.exists(lp(png_de(rel, p)))]
        except Exception as e:
            print(f'  erro render {rel}: {str(e)[:80]}')
            with io.open(BLACKLIST, 'a', encoding='utf-8') as f:
                f.write(rel + '\n')
        os.remove(EM_CURSO)
    return prontos


def chamar_sol(cli, png, modelo=None):
    with open(lp(png), 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    conteudo = [
        {'type': 'input_text',
         'text': 'Analise esta página de Boletim Eleitoral (1932-1937).'},
        {'type': 'input_image', 'image_url': f'data:image/png;base64,{b64}'}]
    ultimo = None
    for i in range(3):
        try:
            resp = cli.responses.create(
                model=modelo or MODELO,
                input=[{'role': 'system', 'content': SYSTEM},
                       {'role': 'user', 'content': conteudo}],
                max_output_tokens=2500)
            texto = (resp.output_text or '').strip()
            texto = re.sub(r'^```(?:json)?|```$', '', texto, flags=re.M).strip()
            return json.loads(texto)
        except Exception as e:
            ultimo = e
            if i < 2:
                time.sleep(3.0 * (i + 1))
    raise RuntimeError(f'sol falhou: {ultimo}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limite', type=int, default=0)
    ap.add_argument('--so-pdf', default='')
    ap.add_argument('--modelo', default=MODELO)
    ap.add_argument('--duvidosos-terra', action='store_true',
                    help='reprocessa com gpt-5.6-terra as paginas que o '
                         'luna marcou legivel=false (2o nivel do funil)')
    args = ap.parse_args()

    os.makedirs(RENDER, exist_ok=True)
    chave = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)

    registros = {}   # (pdf, pagina) -> ultimo registro do cache
    if os.path.exists(CACHE):
        with io.open(CACHE, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    registros[(d['pdf'], d['pagina'])] = d
    feitos = set(registros)

    if args.duvidosos_terra:
        # 2o nivel do funil: reprocessa com terra o que o luna nao leu
        args.modelo = 'gpt-5.6-terra'
        fila = [k for k, d in registros.items()
                if d.get('legivel') is False
                and d.get('_modelo') != 'gpt-5.6-terra']
    else:
        fila = []
        for r in csv.DictReader(io.open(CEGAS, encoding='utf-8-sig')):
            k = (r['pdf'], int(r['pagina']))
            if k in feitos:
                continue
            if args.so_pdf and args.so_pdf not in r['pdf']:
                continue
            fila.append(k)
    if args.limite:
        fila = fila[:args.limite]
    print(f'modelo: {args.modelo} | fila: {len(fila)} '
          f'(cache: {len(feitos)})')
    fila = render_lote(fila)
    print(f'com PNG pronto: {len(fila)}')

    cont = {'ok': 0, 'err': 0}

    def do(item):
        rel, pag = item
        try:
            png = png_de(rel, pag)
            res = chamar_sol(cli, png, args.modelo)
            res.update({'pdf': rel, 'pagina': pag, 'png': png,
                        '_modelo': args.modelo})
            with _lock:
                with io.open(CACHE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
                cont['ok'] += 1
                if cont['ok'] % 25 == 0:
                    print(f"  ... {cont['ok']}")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f'  ERRO {rel} p{pag}: {str(e)[:100]}')

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(do, fila))
    print(f"[FIM] ok={cont['ok']} erros={cont['err']}")

    # consolida CSV
    linhas = []
    with io.open(CACHE, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            for a in d.get('atas') or []:
                linhas.append({
                    'pdf': d['pdf'], 'pagina': d['pagina'],
                    'num': a.get('num'), 'tipo': a.get('tipo'),
                    'data_impressa': a.get('data_impressa'),
                    'tribunal': a.get('tribunal'),
                    'e_ata_formal': a.get('e_ata_formal'),
                    'evidencia': (a.get('evidencia') or '')[:150],
                    'conteudo_pag': d.get('conteudo_predominante'),
                    'n_acordaos_pag': len(d.get('acordaos') or [])})
            if not (d.get('atas')):
                linhas.append({
                    'pdf': d['pdf'], 'pagina': d['pagina'], 'num': '',
                    'tipo': '', 'data_impressa': '', 'tribunal': '',
                    'e_ata_formal': '', 'evidencia': '',
                    'conteudo_pag': d.get('conteudo_predominante'),
                    'n_acordaos_pag': len(d.get('acordaos') or [])})
    with io.open(SAIDA, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    com_ata = sum(1 for l in linhas if l['e_ata_formal'] is True)
    print(f'catalogo -> {SAIDA} ({len(linhas)} registros; '
          f'{com_ata} atas formais detectadas)')


if __name__ == '__main__':
    main()
