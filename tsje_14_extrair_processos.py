# -*- coding: utf-8 -*-
"""tsje_14: decompoe cada transcricao fiel em deliberacoes/processos via LLM
(gpt-5.6-luna, Responses API json_mode), no molde do prompt validado de
conversores/TSJE_docx_to_csv_viaAPI.py — agora com as transcricoes FIEIS
(1932-1937) como fonte, em vez das parafrases do ensaio.

Saida: D:\\TSJE_TRABALHO\\tsje_notion_work\\processos_extraidos.jsonl —
1 linha por ata {ata_id, hash, data, num_sessao, deliberacoes:[...]}.
Idempotente: ata ja extraida com o mesmo hash do corpo e pulada.

  python tsje_14_extrair_processos.py [--limite N] [--model gpt-5.6-luna]
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
from tsje_13_popular_atas import coletar

WORK = r'D:\TSJE_TRABALHO\tsje_notion_work'
SAIDA = os.path.join(WORK, 'processos_extraidos.jsonl')
MODELO_PADRAO = 'gpt-5.6-luna'

_lock = threading.Lock()

MINISTROS_ABREV = (
    'Hermenegildo de Barros, Eduardo Espinola, Carvalho Mourão, '
    'José Linhares, Renato Tavares, Affonso Penna Júnior, '
    'Prudente de Moraes Filho, Affonso Celso, Plínio Casado, '
    'Collares Moreira, João Cabral, Miranda Valverde, Soriano de Souza, '
    'Laudo de Camargo, Levi Carneiro, Lima Rocha, '
    'Cândido de Oliveira Filho, Sampaio Dória, Armando Prado, '
    'Carneiro de Salles, Cezário Alvim, Mac Dowell da Costa, '
    'Ovídio Romeiro, José Lisboa')

SYSTEM = ('Você é um historiador do direito eleitoral brasileiro, '
          'especialista nas atas do Tribunal Superior de Justiça Eleitoral '
          '(1932-1937). Responda SEMPRE e SOMENTE em JSON válido.')

USER_TMPL = """Analise a TRANSCRIÇÃO FIEL de uma ata de sessão do Tribunal \
Superior de Justiça Eleitoral e DECOMPONHA seu conteúdo em deliberações.

TRANSCRIÇÃO DA ATA ({rotulo}):
---
{texto}
---

INSTRUÇÕES:
1. Identifique CADA deliberação, processo, consulta, recurso, habeas corpus \
ou ato administrativo individual — cada assunto distinto vira um registro.
2. Retorne um objeto JSON com a chave "deliberacoes", contendo a LISTA de \
objetos, um por deliberação, na ordem em que aparecem na ata.

Campos de CADA objeto:
- "classe": (ESSENCIAL) "CONSULTA", "REPRESENTAÇÃO", "RECURSO ELEITORAL", \
"HABEAS CORPUS", "PROCESSO", "REGISTRO", "RECLAMAÇÃO", "APELAÇÃO", etc. \
para feitos formais; "ATOS E EXPEDIENTES" para os demais atos e deliberações.
- "punchline": (ESSENCIAL) resumo de UMA frase, clara e direta.
- "decisao": (ESSENCIAL) resumo completo e informativo da deliberação, \
compreensível por si só.
- "numero_do_processo": o número como impresso (ex. "1.015", "595"), \
somente dígitos e ponto de milhar; vazio se não houver.
- "UF": sigla da unidade federativa a que se refere o feito, se mencionada \
(ex. "SP", "RN", "DF"); vazio se não houver.
- "relator": nome ABREVIADO do juiz que relata/propõe, escolhido desta \
lista: [{ministros}]. Vazio se não identificado.
- "parte_recorrente": quem inicia/apresenta (recorrente, representante, \
consulente); só o nome, sem parênteses ou traços. Vazio se não houver.
- "parte_recorrida": contra quem; vazio se não aplicável.
- "interessado": outras partes (terceiros, litisconsortes), separadas por \
vírgula; NÃO incluir juízes do Tribunal. Vazio se não houver.
- "ementa": (ESSENCIAL) o trecho EXATO E LITERAL da transcrição que \
corresponde a esta deliberação (não parafrasear).
- "quorum": (ESSENCIAL e IGUAL em todos os objetos desta ata) os juízes \
presentes na sessão, nomes abreviados separados por vírgula, conforme a \
lista acima; inclua o presidente.
- "votacao": "Unânime" quando unânime ou sem divergência registrada; se \
houver divergência, os nomes abreviados dos vencidos separados por vírgula; \
vazio quando não decisório.
"""


def hash_corpo(t):
    return hashlib.sha1(t.encode('utf-8')).hexdigest()[:16]


def rotulo_da(fm):
    num, tipo = fm.get('num', ''), fm.get('tipo', '')
    d = fm.get('data_sessao', '')
    if num and tipo:
        return f'{num}ª sessão {tipo} de {d}'
    return f"{fm.get('titulo', 'sessão')} — {d}"


def carregar_feitos():
    feitos = {}
    if os.path.exists(SAIDA):
        for line in io.open(SAIDA, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                feitos[r['ata_id']] = r.get('hash', '')
    return feitos


def extrair(reg, model):
    fm, corpo = reg['fm'], reg['corpo']
    user = USER_TMPL.format(rotulo=rotulo_da(fm), texto=corpo,
                            ministros=MINISTROS_ABREV)
    data = llm.chat(SYSTEM, user, json_mode=True, model=model,
                    max_output_tokens=16000)
    if isinstance(data, str):
        data = json.loads(data)
    dels = data.get('deliberacoes', [])
    if not isinstance(dels, list):
        raise ValueError('resposta sem lista deliberacoes')
    return {
        'ata_id': reg['ata_id'],
        'hash': hash_corpo(corpo),
        'data_sessao': fm.get('data_sessao', ''),
        'num_sessao': (f"{fm.get('num', '')}"
                       f"{'E' if fm.get('tipo') == 'extraordinaria' else ''}"
                       if fm.get('num') else fm.get('titulo', '')[:40]),
        'deliberacoes': dels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limite', type=int, default=0)
    ap.add_argument('--model', default=MODELO_PADRAO)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    regs = coletar()
    feitos = carregar_feitos()
    pend = [r for r in regs
            if feitos.get(r['ata_id']) != hash_corpo(r['corpo'])]
    if args.limite:
        pend = pend[:args.limite]
    print(f'atas: {len(regs)} | extraidas: {len(feitos)} | '
          f'pendentes: {len(pend)} | modelo: {args.model}')

    cont = {'ok': 0, 'err': 0, 'dels': 0}

    def do(reg):
        try:
            r = extrair(reg, args.model)
            with _lock:
                with io.open(SAIDA, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                cont['ok'] += 1
                cont['dels'] += len(r['deliberacoes'])
                if cont['ok'] % 25 == 0:
                    print(f"  ... {cont['ok']} atas "
                          f"({cont['dels']} deliberacoes)")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f"  ERRO {reg['ata_id']}: {str(e)[:180]}")

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(do, pend))
    print(f"[FIM] atas={cont['ok']} deliberacoes={cont['dels']} "
          f"erros={cont['err']}")
    print(f'saida -> {SAIDA}')


if __name__ == '__main__':
    main()
