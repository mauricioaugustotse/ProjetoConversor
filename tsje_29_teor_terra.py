# -*- coding: utf-8 -*-
"""tsje_29: gpt-5.6-terra relê por IMAGEM os acordaos de teor fraco
(confianca_ocr baixa/bruta, ~46) e restaura o teor fiel.

Fluxo por acordao: render das paginas (pagina..pagina+1, 200 DPI) ->
terra transcreve a unidade indicada (convencoes de fidelidade; JSON) ->
gate de plausibilidade -> substitui o corpo da pagina no Notion (delete
children + append) -> atualiza props (confianca_ocr e campos vazios) ->
regrava acordaos_teor.jsonl. Depois: tsje_18 --base acordaos re-enriquece.

Uso: python tsje_29_teor_terra.py [--limite N]
Manifesto: tsje_notion_work\\teor_terra_done.jsonl
"""
import argparse
import base64
import importlib.util
import io
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import fitz
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO = 'gpt-5.6-terra'
TRAB = r'D:\TSJE_TRABALHO'
WORK = os.path.join(TRAB, 'tsje_notion_work')
EXTRAIDO = os.path.join(TRAB, 'extraido')
BANCO = r'D:\TSJE_ATAS'
RENDER = r'D:\TSJE_TRANSCRICOES\_paginas_rev'
TEOR = os.path.join(WORK, 'acordaos_teor.jsonl')
DONE = os.path.join(WORK, 'teor_terra_done.jsonl')
_lock = threading.Lock()


def _mod(nome):
    spec = importlib.util.spec_from_file_location(
        nome, os.path.join(BASE, nome + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ln = _mod('_lia_notion')

SYSTEM = """Você é um paleógrafo restaurando ACÓRDÃOS do Tribunal Superior \
de Justiça Eleitoral (Brasil, 1932-1937) impressos em Boletins Eleitorais. \
Receberá 1-2 imagens de páginas e a indicação de qual acórdão transcrever \
(início do texto conhecido). Transcreva FIELMENTE a unidade INTEIRA (do \
cabeçalho "N./Processo n. ..." até a última assinatura), parando na \
unidade seguinte. Convenções: ortografia modernizada; nomes próprios e \
topônimos na grafia da época; [ilegível] para o irrecuperável; palavra[?] \
para leitura incerta; NÃO invente nem resuma. Responda SOMENTE JSON:
{"teor": "...", "ementa": "1-2 frases (do impresso, se houver)",
 "dispositivo": "conclusão decisória",
 "numero_processo": "...", "classe": "...", "uf": "XX",
 "relator": "...", "data_julgamento": "AAAA-MM-DD|null",
 "confianca": "alta|media|baixa"}"""


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def fisico(rel):
    return (os.path.join(BANCO, rel[5:]) if rel.startswith('ATOM ')
            else os.path.join(EXTRAIDO, rel))


def png_de(rel, pag):
    ident = re.sub(r'\W', '', '_'.join(re.split(r'[\\/]', rel)[-3:]))[:40]
    return os.path.join(RENDER, f'{ident}_p{pag:03d}.png')


def render(rel, pag):
    alvo = png_de(rel, pag)
    if os.path.exists(lp(alvo)):
        return alvo
    try:
        doc = fitz.open(lp(fisico(rel)))
        if pag > doc.page_count:
            doc.close()
            return None
        zoom = 200 / 72.0
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
                max_output_tokens=16000)
            texto = (resp.output_text or '').strip()
            texto = re.sub(r'^```(?:json)?|```$', '', texto, flags=re.M).strip()
            return json.loads(texto)
        except Exception as e:
            ultimo = e
            if i < 2:
                time.sleep(3.0 * (i + 1))
    raise RuntimeError(f'terra falhou: {ultimo}')


def substituir_corpo(page_id, teor):
    # apaga blocos atuais
    cursor = None
    ids = []
    while True:
        q = f'/blocks/{page_id}/children?page_size=100'
        if cursor:
            q += f'&start_cursor={cursor}'
        r = ln._req('GET', q)
        ids += [b['id'] for b in r.get('results', [])]
        cursor = r.get('next_cursor')
        if not r.get('has_more'):
            break
    for bid in ids:
        ln._req('DELETE', f'/blocks/{bid}')
    # append novo
    paras = [p.strip() for p in teor.split('\n\n') if p.strip()] or [teor]
    blocos = []
    for p in paras:
        while p:
            blocos.append(p[:1900])
            p = p[1900:]
    for i in range(0, len(blocos), 90):
        children = [{'object': 'block', 'type': 'paragraph', 'paragraph': {
            'rich_text': [{'type': 'text', 'text': {'content': b}}]}}
            for b in blocos[i:i + 90]]
        ln._req('PATCH', f'/blocks/{page_id}/children',
                {'children': children})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limite', type=int, default=0)
    args = ap.parse_args()

    chave = io.open(os.path.join(BASE, 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)

    regs = {}
    with io.open(TEOR, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                regs[r['acordao_id']] = r
    pages = {}
    with io.open(os.path.join(WORK, 'acordaos_inserted.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pages[d['chave']] = d['id']

    feitos = set()
    if os.path.exists(DONE):
        with io.open(DONE, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get('res') == 'restaurado':
                        feitos.add(d['acordao_id'])

    fila = [r for r in regs.values()
            if r.get('confianca') in ('baixa', 'bruta')
            and r['acordao_id'] not in feitos
            and r['acordao_id'] in pages]
    if args.limite:
        fila = fila[:args.limite]
    print(f'teores fracos na fila: {len(fila)} (feitos: {len(feitos)})')
    cont = Counter()

    def do(r):
        aid = r['acordao_id']
        try:
            pag = int(r.get('pagina') or 1)
            pngs = [p for p in (render(r['pdf'], pag),
                                render(r['pdf'], pag + 1)) if p]
            if not pngs:
                raise ValueError('render falhou')
            bruto = r.get('teor') or ''
            perg = (f"Transcreva o acórdão que começa assim: "
                    f"\"{bruto[:220]}\" (processo n. "
                    f"{r.get('numero_processo') or '?'}). A unidade está "
                    f"na 1ª imagem, página {pag} do boletim.")
            res = chamar(cli, perg, pngs)
            teor = (res.get('teor') or '').strip()
            if len(teor) < 300:
                raise ValueError(f'teor curto ({len(teor)})')
            # gate por comprimento so vale quando o bruto delimita bem a
            # unidade; nos 'bruta' o texto de referencia e uma unidade
            # mal-delimitada (pode englobar dezenas de itens) e nao serve
            # de regua — o terra transcreve so o acordao pedido da imagem
            if r.get('confianca') != 'bruta' and len(bruto) > 800 \
                    and not (0.4 * len(bruto) <= len(teor)
                             <= 1.6 * len(bruto)):
                raise ValueError(
                    f'gate: bruto={len(bruto)} novo={len(teor)}')
            pid = pages[aid]
            substituir_corpo(pid, teor)
            props = {'confianca_ocr': {'select': {'name':
                     {'media': 'média'}.get(res.get('confianca'),
                                            res.get('confianca', 'média'))}}}
            for prop, chave_r in (('relator', 'relator'), ('UF', 'uf')):
                v = (res.get(chave_r) or '').strip()
                if v and not r.get(chave_r):
                    props[prop] = {'select': {'name': v[:95]}}
            if res.get('data_julgamento') and not r.get('data_julgamento'):
                if re.fullmatch(r'\d{4}-\d{2}-\d{2}',
                                res['data_julgamento']):
                    props['data_julgamento'] = {
                        'date': {'start': res['data_julgamento']}}
            ln._req('PATCH', f'/pages/{pid}', {'properties': props})
            novo = dict(r)
            novo.update({'teor': teor,
                         'ementa': res.get('ementa') or r.get('ementa', ''),
                         'dispositivo': res.get('dispositivo')
                         or r.get('dispositivo', ''),
                         'confianca': res.get('confianca', 'media'),
                         'restaurado_por': MODELO})
            with _lock:
                with io.open(TEOR, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(novo, ensure_ascii=False) + '\n')
                with io.open(DONE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'acordao_id': aid,
                                        'res': 'restaurado'}) + '\n')
                cont['ok'] += 1
                if cont['ok'] % 10 == 0:
                    print(f"  ... {cont['ok']}")
        except Exception as e:
            with _lock:
                with io.open(DONE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'acordao_id': aid,
                        'res': f'falhou: {str(e)[:80]}'}) + '\n')
                cont['err'] += 1
                print(f'  ERRO {aid}: {str(e)[:90]}')

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(do, fila))
    print(f'[FIM] {dict(cont)}')


if __name__ == '__main__':
    main()
