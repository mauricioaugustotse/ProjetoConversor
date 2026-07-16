# -*- coding: utf-8 -*-
"""tsje_22b: verificacao por IMAGEM (gpt-5.6-sol) dos achados da busca
dirigida (tsje_revisao_82.csv, vereditos ENCONTRADA_*).

Para cada achado: renderiza a pagina apontada (+1 seguinte) e pergunta ao
sol se ha ATA FORMAL da sessao N tipo T cuja data cai na janela да sessao —
com as armadilhas do acervo (resenha nao vale; conferir ANO impresso;
tribunal superior, nao TRE).

Saida: tsje_verificacao_achados.csv (aprovado/reprovado + o que ha na
pagina). Cache: tsje_verif_cache.jsonl. Os aprovados seguem para
transcricao; reprovados voltam a AUSENTE (com evidencia).

Uso: python tsje_22b_verificar_achados.py [--limite N]
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

MODELO = 'gpt-5.6-terra'  # troca de custo 16/07: verificacao com terra
TRAB = r'D:\TSJE_TRABALHO'
EXTRAIDO = r'D:\TSJE_TRABALHO\extraido'
BANCO = r'D:\TSJE_ATAS'
RENDER = r'D:\TSJE_TRANSCRICOES\_paginas_rev'
REVISAO = os.path.join(TRAB, 'tsje_revisao_82.csv')
ALVOS = os.path.join(TRAB, 'tsje_sessoes_alvo.csv')
CACHE = os.path.join(TRAB, 'tsje_verif_cache.jsonl')
SAIDA = os.path.join(TRAB, 'tsje_verificacao_achados.csv')
DPI = 200

SYSTEM = """Você é um paleógrafo verificador de achados em Boletins \
Eleitorais brasileiros de 1932-1937 (Tribunal Superior de Justiça \
Eleitoral). Receberá a pergunta sobre UMA sessão específica e 1-2 imagens \
de páginas. Responda SOMENTE com JSON válido:
{"veredito": "APROVADO"|"REPROVADO"|"INCERTO",
 "o_que_ha": "descrição curta do que a página traz sobre a sessão",
 "titulo_impresso": "cabeçalho da ata como impresso, ou null",
 "data_impressa": "AAAA-MM-DD ou null",
 "ano_do_boletim": "ano impresso no cabeçalho da PÁGINA, ou null",
 "pagina_do_cabecalho": 1|2|null (qual das imagens tem o cabeçalho da ata)}

APROVADO exige TODAS as condições:
1. A página contém ATA FORMAL da sessão perguntada: cabeçalho "ATA DA Nª \
SESSÃO ..." + preâmbulo ("Aos ... dias"/"Às ... horas, presentes os \
juízes...") e/ou linha de PRESIDÊNCIA. RESENHA de julgamentos ("O \
Tribunal, em sua Nª sessão, resolveu: 1º...") NÃO vale → REPROVADO com \
o_que_ha="resenha".
2. O número e tipo (ordinária/extraordinária) conferem com a pergunta.
3. A DATA impressa da sessão cai dentro da janela informada (ATENÇÃO: se o \
ano impresso for outro, é ata homônima de outro ano → REPROVADO).
4. Tribunal SUPERIOR (não Tribunal Regional).
Não invente: se a imagem não permitir decidir, use INCERTO."""

_lock = threading.Lock()


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def caminho_fisico(rel):
    if rel.startswith('ATOM '):
        return os.path.join(BANCO, rel[5:])
    return os.path.join(EXTRAIDO, rel)


def png_de(rel, pag):
    ident = re.sub(r'\W', '', '_'.join(re.split(r'[\\/]', rel)[-3:]))[:40]
    return os.path.join(RENDER, f'{ident}_p{pag:03d}.png')


def render(rel, pag):
    alvo = png_de(rel, pag)
    if os.path.exists(lp(alvo)):
        return alvo
    try:
        doc = fitz.open(lp(caminho_fisico(rel)))
        if pag > doc.page_count:
            doc.close()
            return None
        zoom = DPI / 72.0
        pix = doc[pag - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                      colorspace=fitz.csGRAY)
        pix.save(lp(alvo))
        doc.close()
        return alvo
    except Exception:
        return None


def chamar(cli, pergunta, pngs):
    conteudo = [{'type': 'input_text', 'text': pergunta}]
    for png in pngs:
        with open(lp(png), 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        conteudo.append({'type': 'input_image',
                         'image_url': f'data:image/png;base64,{b64}'})
    ultimo = None
    for i in range(3):
        try:
            resp = cli.responses.create(
                model=MODELO,
                input=[{'role': 'system', 'content': SYSTEM},
                       {'role': 'user', 'content': conteudo}],
                max_output_tokens=1500)
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
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    chave = io.open(os.path.join(base, 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)

    janelas = {}
    for a in csv.DictReader(io.open(ALVOS, encoding='utf-8-sig')):
        janelas[(a['ano'], a['tipo'], a['num'])] = (a['sessao_min'],
                                                    a['sessao_max'])

    feitos = {}
    if os.path.exists(CACHE):
        with io.open(CACHE, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    feitos[d['chave']] = d

    fila = []
    for r in csv.DictReader(io.open(REVISAO, encoding='utf-8-sig')):
        if not r['veredito'].startswith('ENCONTRADA'):
            continue
        chave_r = f"{r['ano']}-{r['tipo']}-{r['num']}"
        if chave_r in feitos:
            continue
        fila.append(r)
    if args.limite:
        fila = fila[:args.limite]
    print(f'achados a verificar: {len(fila)} (cache: {len(feitos)})')

    cont = {'ap': 0, 'rep': 0, 'inc': 0, 'err': 0}

    def do(r):
        chave_r = f"{r['ano']}-{r['tipo']}-{r['num']}"
        try:
            pag = int(r['pagina'])
            pngs = [p for p in (render(r['pdf'], pag),
                                render(r['pdf'], pag + 1)) if p]
            if not pngs:
                raise ValueError('render falhou')
            smin, smax = janelas.get((r['ano'], r['tipo'], r['num']),
                                     (f"{r['ano']}-01-01",
                                      f"{r['ano']}-12-31"))
            perg = (f"A busca textual indicou que a página contém a ATA "
                    f"FORMAL da {r['num']}ª SESSÃO "
                    f"{r['tipo'].upper()} do Tribunal Superior de Justiça "
                    f"Eleitoral, realizada entre {smin} e {smax}. "
                    f"Verifique nas imagens (página apontada e seguinte). "
                    f"Evidência textual da busca: {r['evidencia'][:150]}")
            res = chamar(cli, perg, pngs)
            res.update({'chave': chave_r, 'pdf': r['pdf'],
                        'pagina': pag, '_modelo': MODELO})
            with _lock:
                with io.open(CACHE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
                v = res.get('veredito', 'INCERTO')
                cont['ap' if v == 'APROVADO' else
                     'rep' if v == 'REPROVADO' else 'inc'] += 1
                n = cont['ap'] + cont['rep'] + cont['inc']
                if n % 10 == 0:
                    print(f"  ... {n} (ap={cont['ap']} rep={cont['rep']} "
                          f"inc={cont['inc']})")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f'  ERRO {chave_r}: {str(e)[:100]}')

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(do, fila))
    print(f"[FIM] aprovados={cont['ap']} reprovados={cont['rep']} "
          f"incertos={cont['inc']} erros={cont['err']}")

    # consolida
    linhas = []
    with io.open(CACHE, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                linhas.append({
                    'chave': d['chave'], 'pdf': d['pdf'],
                    'pagina': d['pagina'],
                    'veredito': d.get('veredito'),
                    'o_que_ha': (d.get('o_que_ha') or '')[:150],
                    'titulo_impresso': (d.get('titulo_impresso') or '')[:100],
                    'data_impressa': d.get('data_impressa') or '',
                    'ano_do_boletim': d.get('ano_do_boletim') or ''})
    with io.open(SAIDA, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f'-> {SAIDA}')


if __name__ == '__main__':
    main()
