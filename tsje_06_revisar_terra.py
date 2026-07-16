# -*- coding: utf-8 -*-
"""tsje_06: revisor/companheiro gpt-5.6-terra para as transcricoes das atas.

Seleciona atas transcritas que sejam: dificeis (OCR fraco) OU com confianca
media/baixa OU ~20%% de amostra deterministica das demais. Envia as IMAGENS
das paginas + a transcricao ao gpt-5.6-terra (Responses API, visao) e grava
um parecer JSON por ata: fidelidade 0-10, divergencias materiais, lacunas e
leituras propostas para trechos [ilegivel]. O manifest ganha status=revisada.

A aplicacao das correcoes e feita pelo Claude (le os pareceres, reconfere na
imagem e edita o .md) — este script NAO altera transcricoes.

Uso: python tsje_06_revisar_terra.py [--ano 1932] [--limite 50] [--dry]
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = r'D:\TSJE_TRANSCRICOES'
MANIFEST = os.path.join(RAIZ, 'manifest.csv')
CHAVE = os.path.join(BASE, 'Chave_secreta_OpenAI.txt')
MODELO = 'gpt-5.6-terra'
AMOSTRA_MOD = 5  # 1 em 5 (~20%) das atas "faceis com confianca alta"

SYSTEM = (
    'Você é um revisor especializado em transcrições paleográficas de '
    'Boletins Eleitorais brasileiros dos anos 1930 (Tribunal Superior de '
    'Justiça Eleitoral). Sua tarefa: comparar a TRANSCRIÇÃO fornecida com as '
    'IMAGENS das páginas originais e apontar APENAS divergências MATERIAIS: '
    'palavras trocadas que alteram o sentido, texto omitido, nomes próprios '
    'errados, números/datas errados, trechos inventados. ATENÇÃO CRÍTICA: '
    'uma MESMA PÁGINA do Boletim costuma trazer DUAS OU MAIS atas de '
    'sessões diferentes (em colunas distintas). Revise APENAS a ata cujo '
    'cabeçalho corresponde ao título informado — localize esse cabeçalho na '
    'imagem e siga só o texto dele até "Levanta-se a sessão". NÃO compare a '
    'transcrição com o conteúdo das outras atas da mesma página: isso já '
    'gerou pareceres falsos. NÃO aponte: '
    'ortografia modernizada (é intencional: "installação"→"instalação", '
    '"E\'"→"É"), pontuação equivalente, hifenização desfeita, nem a '
    'ausência do cabeçalho/título da ata e da linha "PRESIDÊNCIA DO SR..." '
    '(ficam nos metadados, fora do corpo, por convenção). Se conseguir '
    'ler trechos marcados [ilegível] ou [?], proponha a leitura. Responda '
    'SOMENTE com um objeto JSON: {"fidelidade_0_10": <número>, '
    '"divergencias": [{"onde": "<trecho de contexto>", "na_imagem": "<o que '
    'está impresso>", "na_transcricao": "<o que foi transcrito>"}], '
    '"lacunas": ["<texto impresso que falta na transcrição>"], '
    '"leituras_de_ilegiveis": [{"onde": "<contexto>", "leitura": "<proposta>"}], '
    '"observacoes": "<curto>"}'
)


def front_matter(path):
    with io.open(path, encoding='utf-8') as f:
        texto = f.read()
    m = re.match(r'---\n(.*?)\n---\n?(.*)', texto, re.S)
    fm = {}
    if not m:
        return fm, texto
    for linha in m.group(1).split('\n'):
        if ':' in linha:
            k, v = linha.split(':', 1)
            fm[k.strip()] = v.strip()
    return fm, m.group(2).strip()


def cliente():
    from openai import OpenAI
    with io.open(CHAVE, encoding='utf-8') as f:
        key = f.read().strip()
    return OpenAI(api_key=key, timeout=180)


def selecionar(regs, ano):
    alvo = []
    for r in regs:
        if r['status'] != 'transcrita':
            continue
        if ano and r['ano'] != ano:
            continue
        dificil = r['dificuldade'] == 'dificil'
        fraca = r['confianca'] in ('media', 'baixa')
        amostra = int(hashlib.sha1(r['ata_id'].encode()).hexdigest(),
                      16) % AMOSTRA_MOD == 0
        if dificil or fraca or amostra:
            r['_motivo'] = ('dificil' if dificil else
                            'confianca' if fraca else 'amostra')
            alvo.append(r)
    return alvo


def revisar(cli, r):
    fm, corpo = front_matter(os.path.join(RAIZ, r['transcricao']))
    conteudo = [{'type': 'input_text', 'text':
                 f'TRANSCRIÇÃO A REVISAR (ata: {fm.get("titulo", r["titulo"])}; '
                 f'as imagens a seguir são as páginas do Boletim onde a ata '
                 f'está — a ata começa na página da 1ª imagem):\n\n{corpo}'}]
    for rel in r['pngs'].split(';'):
        with open(os.path.join(RAIZ, rel), 'rb') as f:
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
                max_output_tokens=4000)
            texto = (resp.output_text or '').strip()
            texto = re.sub(r'^```(?:json)?|```$', '', texto,
                           flags=re.M).strip()
            uso = getattr(resp, 'usage', None)
            tokens = ((getattr(uso, 'input_tokens', 0) or 0),
                      (getattr(uso, 'output_tokens', 0) or 0)) if uso else (0, 0)
            return json.loads(texto), tokens
        except Exception as e:
            ultimo = e
            if i < 2:
                time.sleep(3.0 * (i + 1))
    raise RuntimeError(f'terra falhou: {ultimo}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ano')
    ap.add_argument('--limite', type=int, default=200)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    with io.open(MANIFEST, encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        regs = list(rd)

    alvo = selecionar(regs, args.ano)[:args.limite]
    print(f'{len(alvo)} atas selecionadas para revisão do terra '
          f'({sum(1 for a in alvo if a["_motivo"] == "dificil")} difíceis, '
          f'{sum(1 for a in alvo if a["_motivo"] == "confianca")} confiança, '
          f'{sum(1 for a in alvo if a["_motivo"] == "amostra")} amostra)')
    if args.dry:
        for a in alvo[:20]:
            print(f'  {a["ata_id"]} [{a["_motivo"]}]')
        return

    cli = cliente()
    tin = tout = 0
    ok = err = 0
    for n, r in enumerate(alvo, 1):
        parecer_rel = os.path.join(r['ano'], f'{r["ata_id"]}.parecer.json')
        try:
            ja = os.path.join(RAIZ, parecer_rel)
            if os.path.exists(ja):
                with io.open(ja, encoding='utf-8') as f:
                    parecer = json.load(f)
                ti = to = 0
            else:
                parecer, (ti, to) = revisar(cli, r)
            parecer['_motivo_selecao'] = r['_motivo']
            parecer['_modelo'] = MODELO
            with io.open(os.path.join(RAIZ, parecer_rel), 'w',
                         encoding='utf-8') as f:
                json.dump(parecer, f, ensure_ascii=False, indent=1)
            r.pop('_motivo', None)
            r['status'] = 'revisada'
            r['parecer'] = parecer_rel
            r['fidelidade'] = str(parecer.get('fidelidade_0_10', ''))
            tin += ti
            tout += to
            ok += 1
            ndiv = len(parecer.get('divergencias', []))
            print(f'  [{n}/{len(alvo)}] {r["ata_id"]}: fidelidade '
                  f'{r["fidelidade"]}, {ndiv} divergências')
        except Exception as e:
            err += 1
            print(f'  [{n}/{len(alvo)}] {r["ata_id"]}: ERRO {e}')
        if n % 10 == 0 or n == len(alvo):
            with io.open(MANIFEST, 'w', encoding='utf-8-sig',
                         newline='') as f:
                w = csv.DictWriter(f, fieldnames=cols,
                                   extrasaction='ignore')
                w.writeheader()
                w.writerows(regs)
    print(f'revisadas: {ok}, erros: {err} | tokens in/out: {tin}/{tout}')


if __name__ == '__main__':
    main()
