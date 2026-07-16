# -*- coding: utf-8 -*-
"""tsje_16b: limpa o teor OCR de cada acordao catalogado (tsje_16) e refina
metadados, via gpt-5.6-luna com gate anti-alucinacao (proibido acrescentar
conteudo; [ilegivel] onde irrecuperavel; validacao de comprimento).

Entrada: D:\\TSJE_TRABALHO\\tsje_acordaos_texto.jsonl
Saida:   D:\\TSJE_TRABALHO\\tsje_notion_work\\acordaos_teor.jsonl
Idempotente por acordao_id + hash do texto bruto.

  python tsje_16b_acordaos_teor.py [--limite N] [--model gpt-5.6-luna]
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import threading
import concurrent.futures as cf

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conle_gerador import llm
from tsje_14_extrair_processos import MINISTROS_ABREV

ENTRADA = r'D:\TSJE_TRABALHO\tsje_acordaos_texto.jsonl'
WORK = r'D:\TSJE_TRABALHO\tsje_notion_work'
SAIDA = os.path.join(WORK, 'acordaos_teor.jsonl')
MODELO_PADRAO = 'gpt-5.6-luna'

_lock = threading.Lock()

SYSTEM = ('Você restaura textos de acórdãos do Tribunal Superior de Justiça '
          'Eleitoral (1932-1937) a partir de OCR ruidoso de jornais da '
          'década de 1930. REGRA ABSOLUTA: não inventar, não acrescentar e '
          'não resumir conteúdo — apenas corrigir erros evidentes de OCR '
          '(letras trocadas, espaços, hifenização de quebra de linha) e '
          'modernizar a ortografia comum (ex. "installação"→"instalação"), '
          'mantendo nomes próprios na grafia da época. Trechos '
          'irrecuperáveis viram [ilegível]. Responda SEMPRE e SOMENTE em '
          'JSON válido.')

USER_TMPL = """TEXTO OCR DE UM ACÓRDÃO/JULGADO publicado no Boletim \
Eleitoral (pode conter a estrutura: "Processo n. X" ou "N. X" / "Natureza \
do processo — ..." / "Relator" / ementa / VISTOS / Acordam... / data e \
assinaturas):
---
{texto}
---
Retorne JSON com:
- "teor": o texto INTEGRAL restaurado, parágrafos separados por linha em \
branco, sem as palavras cortadas do OCR; NADA de conteúdo novo.
- "numero_processo": número do processo como impresso (só dígitos e ponto \
de milhar); vazio se ilegível.
- "classe": "CONSULTA", "REPRESENTAÇÃO", "RECLAMAÇÃO", "RECURSO", \
"RECURSO PARCIAL", "HABEAS CORPUS", "REGISTRO", "COMUNICAÇÃO", \
"PROCESSO" etc.
- "uf": sigla da UF a que o feito se refere (ex. "ES", "PE"); vazio se \
não houver.
- "relator": nome ABREVIADO do relator, escolhido desta lista quando \
possível: [{ministros}]. Vazio se não identificado.
- "data_julgamento": data da decisão no formato AAAA-MM-DD; vazio se \
não constar.
- "ementa": a ementa/súmula destacada no início, restaurada; vazio se \
não houver.
- "dispositivo": UMA frase com o que foi decidido ("Acordam ... em ...").
- "votacao": "Unânime" se constar decisão unânime; senão os vencidos; \
vazio se não constar.
- "confianca": "alta", "media" ou "baixa" — qualidade da restauração \
(baixa = OCR muito degradado, muitos [ilegível]).
"""


def hash_txt(t):
    return hashlib.sha1(t.encode('utf-8')).hexdigest()[:16]


def carregar_feitos():
    feitos = {}
    if os.path.exists(SAIDA):
        for line in io.open(SAIDA, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                feitos[r['acordao_id']] = r.get('hash', '')
    return feitos


def validar(bruto, teor):
    """Gate anti-alucinacao: comprimento do teor dentro de [-45%, +15%] do
    trecho efetivamente ENVIADO (o prompt trunca em 24000 chars; unidades
    maiores nao podem ser cobradas pelo excedente)."""
    lb, lt = min(len(bruto), 24000), len(teor)
    if lt < lb * 0.55 or lt > lb * 1.15:
        raise ValueError(f'teor suspeito: enviado={lb} limpo={lt}')


def processar(reg, model):
    bruto = reg['texto']
    user = USER_TMPL.format(texto=bruto[:24000], ministros=MINISTROS_ABREV)
    data = llm.chat(SYSTEM, user, json_mode=True, model=model,
                    max_output_tokens=16000)
    if isinstance(data, str):
        data = json.loads(data)
    teor = (data.get('teor') or '').strip()
    validar(bruto, teor)
    d = data.get('data_julgamento') or ''
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
        d = reg.get('data_julgamento', '')
    return {
        'acordao_id': reg['acordao_id'],
        'hash': hash_txt(bruto),
        'pdf': reg['pdf'], 'ano_be': reg['ano_be'],
        'pagina': reg['pagina'],
        'teor': teor,
        'ementa': (data.get('ementa') or '').strip(),
        'dispositivo': (data.get('dispositivo') or '').strip(),
        'numero_processo': (data.get('numero_processo')
                            or reg.get('numero_processo') or '').strip(),
        'classe': (data.get('classe') or reg.get('classe') or '').strip(),
        'uf': (data.get('uf') or reg.get('uf') or '').strip().upper()[:4],
        'relator': (data.get('relator') or reg.get('relator') or '').strip(),
        'data_julgamento': d,
        'votacao': (data.get('votacao') or '').strip(),
        'confianca': (data.get('confianca') or 'media').strip().lower(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limite', type=int, default=0)
    ap.add_argument('--model', default=MODELO_PADRAO)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    regs = []
    with io.open(ENTRADA, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                regs.append(json.loads(line))
    feitos = carregar_feitos()
    pend = [r for r in regs
            if feitos.get(r['acordao_id']) != hash_txt(r['texto'])]
    if args.limite:
        pend = pend[:args.limite]
    print(f'acordaos: {len(regs)} | limpos: {len(feitos)} | '
          f'pendentes: {len(pend)} | modelo: {args.model}')

    cont = {'ok': 0, 'err': 0}

    def do(reg):
        try:
            r = processar(reg, args.model)
            with _lock:
                with io.open(SAIDA, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                cont['ok'] += 1
                if cont['ok'] % 50 == 0:
                    print(f"  ... {cont['ok']}")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f"  ERRO {reg['acordao_id']}: {str(e)[:160]}")

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(do, pend))
    print(f"[FIM] ok={cont['ok']} erros={cont['err']} -> {SAIDA}")


if __name__ == '__main__':
    main()
