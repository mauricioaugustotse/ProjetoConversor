# -*- coding: utf-8 -*-
"""tsje_32: extrator DIRIGIDO dos acordaos substantivos do BE 54/1937
(inedito da BDTSE) por VISAO (gpt-5.6-terra). O OCR desse PDF fragmenta o
texto e mistura as colunas do jornal, inviabilizando a extracao por regex;
o terra le a IMAGEM das paginas de jurisprudencia (10-13; p16 e de TRE, nao
entra) e devolve cada acordao do TRIBUNAL SUPERIOR com numero, classe, UF,
relator, partes, data, ementa e teor fiel. Gera entradas no formato de
tsje_acordaos_texto.jsonl + o teor limpo direto em acordaos_teor.jsonl
(pula o 16b, ja que o terra ja restaura), para o tsje_17 popular/associar.

Uso: python tsje_32_be54_acordaos.py [--apply]
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time

import fitz
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO = 'gpt-5.6-terra'
PDF = r'D:\TSJE_ATAS\1937\1937-05-15 - BE n54 [bdtse 2467].pdf'
REL = 'ATOM 1937\\1937-05-15 - BE n54 [bdtse 2467].pdf'
RENDER = r'D:\TSJE_TRANSCRICOES\_paginas_rev'
TEXTO = r'D:\TSJE_TRABALHO\tsje_acordaos_texto.jsonl'
TEOR = r'D:\TSJE_TRABALHO\tsje_notion_work\acordaos_teor.jsonl'
ANO = '1937'
PAGINAS_JURIS = [10, 11, 12, 13]
DATA_PUB = '1937-05-15'

SYSTEM = """Você é um paleógrafo extraindo ACÓRDÃOS do Tribunal Superior de \
Justiça Eleitoral (Brasil, 1937) da seção JURISPRUDÊNCIA de um Boletim \
Eleitoral. A imagem é de jornal em colunas; leia com cuidado a ordem das \
colunas. Extraia SOMENTE os acórdãos julgados pelo próprio TRIBUNAL \
SUPERIOR ("Accordam os Juízes do Tribunal Superior de Justiça \
Eleitoral..."). NÃO inclua acórdãos de Tribunais REGIONAIS reproduzidos \
como jurisprudência comparada, nem as listas de "revisão de provas" ou de \
qualificação de eleitores. Para cada acórdão do Superior, responda no \
array JSON (sem markdown):
{"acordaos": [{"numero_processo": "n. impresso", "classe": "APELAÇÃO \
CRIMINAL|RECURSO ELEITORAL|CONSULTA|PROCESSO|...", "uf": "XX (estado de \
origem)", "relator": "nome", "partes": "recorrente x recorrido, se houver",
"data_julgamento": "AAAA-MM-DD|null", "unanime": true|false,
"ementa": "1-2 frases da tese/decisão", "teor": "transcrição fiel do \
acórdão, do 'Vistos' ao fecho; [ilegível] onde não der; sem inventar"}]}
Se a página não tiver acórdão do Superior, {"acordaos": []}."""


def lp(p):
    return p if p.startswith('\\\\?\\') else '\\\\?\\' + p


def render(pag):
    ident = re.sub(r'\W', '', 'be54_1937')
    alvo = os.path.join(RENDER, f'{ident}_p{pag:03d}.png')
    if not os.path.exists(lp(alvo)):
        d = fitz.open(lp(PDF))
        zoom = 220 / 72.0
        pix = d[pag - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                    colorspace=fitz.csGRAY)
        pix.save(lp(alvo))
        d.close()
    return alvo


def chamar(cli, pngs):
    conteudo = [{'type': 'input_text',
                 'text': f'Estas são {len(pngs)} páginas consecutivas da '
                         'seção Jurisprudência (um acórdão pode começar numa '
                         'página e terminar na seguinte — junte as partes). '
                         'Extraia os acórdãos COMPLETOS do Tribunal Superior.'}]
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
                max_output_tokens=12000)
            t = re.sub(r'^```(?:json)?|```$', '', (resp.output_text or '').strip(),
                       flags=re.M).strip()
            return json.loads(t)
        except Exception as e:
            ultimo = e
            if i < 2:
                time.sleep(3.0 * (i + 1))
    raise RuntimeError(f'terra falhou: {ultimo}')


def extrair():
    chave = io.open(os.path.join(BASE, 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)
    uni = []
    seq = 0
    vistos = set()
    pngs = [render(p) for p in PAGINAS_JURIS]
    res = chamar(cli, pngs)
    if True:
        for a in res.get('acordaos', []):
            num = (a.get('numero_processo') or '').strip()
            chave_dedup = (num, (a.get('ementa') or '')[:40])
            if chave_dedup in vistos:
                continue
            vistos.add(chave_dedup)
            seq += 1
            uni.append({
                'acordao_id': f'ac-BDTSE1937_BEn54-{seq:02d}',
                'pdf': REL, 'ano_be': ANO,
                'pagina': str(a.get('pagina') or PAGINAS_JURIS[0]),
                'numero_processo': re.sub(r'[^\d.]', '', num),
                'classe': (a.get('classe') or 'PROCESSO').upper()[:40],
                'uf': (a.get('uf') or '')[:6].upper().strip(),
                'relator': (a.get('relator') or '')[:60],
                'data_julgamento': a.get('data_julgamento') or '',
                'unanime': 'sim' if a.get('unanime') else '',
                'natureza': '', 'n_linhas': '',
                'ementa': (a.get('ementa') or '').strip(),
                'partes': (a.get('partes') or '').strip(),
                'texto': (a.get('teor') or '').strip()[:8000]})
    return uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    uni = extrair()
    print(f'acordaos substantivos do TSJE no BE 54/1937: {len(uni)}')
    for u in uni:
        print(f"  {u['acordao_id']} | n.{u['numero_processo']} | "
              f"{u['classe']} | pag {u['pagina']} | data {u['data_julgamento']}"
              f" | {len(u['texto'])} chars")
        print(f"     inicio: {u['texto'][:110]}")
    if not args.apply:
        print('\n(dry-run; use --apply para gravar)')
        return
    ja = set()
    if os.path.exists(TEXTO):
        with io.open(TEXTO, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ja.add(json.loads(line)['acordao_id'])
    novos = [u for u in uni if u['acordao_id'] not in ja]
    # 1) texto bruto (rastreabilidade)
    with io.open(TEXTO, 'a', encoding='utf-8') as f:
        for u in novos:
            reg = {k: u[k] for k in
                   ('acordao_id', 'pdf', 'ano_be', 'pagina',
                    'numero_processo', 'classe', 'uf', 'relator',
                    'data_julgamento', 'unanime', 'natureza', 'n_linhas')}
            reg['texto'] = u['texto']
            f.write(json.dumps(reg, ensure_ascii=False) + '\n')
    # 2) teor JA LIMPO pelo terra -> acordaos_teor.jsonl (pula 16b)
    with io.open(TEOR, 'a', encoding='utf-8') as f:
        for u in novos:
            f.write(json.dumps({
                'acordao_id': u['acordao_id'], 'pdf': u['pdf'],
                'ano_be': u['ano_be'], 'pagina': u['pagina'],
                'teor': u['texto'], 'ementa': u['ementa'],
                'dispositivo': u['ementa'],
                'numero_processo': u['numero_processo'],
                'classe': u['classe'], 'uf': u['uf'],
                'relator': u['relator'],
                'data_julgamento': u['data_julgamento'],
                'votacao': 'unânime' if u['unanime'] else '',
                'confianca': 'alta', 'extraido_por': 'terra-visao-be54'
            }, ensure_ascii=False) + '\n')
    print(f'\ngravados: {len(novos)} acordaos (texto + teor limpo)')
    print('proximo: python tsje_17_popular_acordaos.py --apply '
          '&& python tsje_18_enriquecer.py --base acordaos')


if __name__ == '__main__':
    main()
