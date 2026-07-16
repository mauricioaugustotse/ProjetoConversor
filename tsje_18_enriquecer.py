# -*- coding: utf-8 -*-
"""tsje_18: enriquecimento das bases Notion do TSJE (padrao _stj_enriquecer):
gera "resumo_ia" e "palavras_chave" via gpt-5.6-luna e grava via PATCH.

--base atas     : fonte = corpo dos .md locais (coletar() do tsje_13);
                  page_id via atas_inserted.jsonl.
--base acordaos : fonte = teor restaurado (acordaos_teor.jsonl);
                  page_id via acordaos_inserted.jsonl; o resumo_ia gravado
                  no tsje_17 (dispositivo) e mantido — so completa keywords
                  e resumo quando vazios.

Cache idempotente por chave+hash em enriquecer_cache.jsonl.

  python tsje_18_enriquecer.py --base atas [--limite N]
"""
import argparse
import hashlib
import io
import json
import os
import sys
import threading
import concurrent.futures as cf

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _lia_notion as ln
from conle_gerador import llm

WORK = r'D:\TSJE_TRABALHO\tsje_notion_work'
CACHE = os.path.join(WORK, 'enriquecer_cache.jsonl')
MODELO = 'gpt-5.6-luna'

_lock = threading.Lock()

SYSTEM = ('Você é um historiador do direito eleitoral brasileiro. Responda '
          'SEMPRE e SOMENTE em JSON válido.')

USER_TMPL = """Texto de {oque} do Tribunal Superior de Justiça Eleitoral \
({rotulo}):
---
{texto}
---
Retorne JSON com:
- "resumo": 1-2 frases (ate ~55 palavras) com o essencial — temas tratados \
e o que foi decidido.
- "palavras_chave": lista de 6 a 12 termos de busca (temas juridicos, \
institutos, nomes de partes/orgaos relevantes; sem nomes dos juizes).
"""


def hash_txt(t):
    return hashlib.sha1(t.encode('utf-8')).hexdigest()[:16]


def load_jsonl_map(path, kc='chave', vc='id'):
    m = {}
    if os.path.exists(path):
        for line in io.open(path, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                m[r[kc]] = r[vc]
    return m


def load_cache():
    m = {}
    if os.path.exists(CACHE):
        for line in io.open(CACHE, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                m[r['chave']] = r
    return m


def gravar_cache(reg):
    with _lock:
        with io.open(CACHE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(reg, ensure_ascii=False) + '\n')


def fontes_atas():
    from tsje_13_popular_atas import coletar, nome_da_ata
    pages = load_jsonl_map(os.path.join(WORK, 'atas_inserted.jsonl'))
    itens = []
    for reg in coletar():
        pid = pages.get(reg['ata_id'])
        if pid:
            itens.append({'chave': f"ata:{reg['ata_id']}",
                          'page': pid,
                          'rotulo': nome_da_ata(reg['fm']),
                          'texto': reg['corpo'][:20000],
                          'oque': 'uma ata de sessão'})
    return itens


def fontes_acordaos():
    pages = load_jsonl_map(os.path.join(WORK, 'acordaos_inserted.jsonl'))
    itens = []
    path = os.path.join(WORK, 'acordaos_teor.jsonl')
    vistos = set()
    if os.path.exists(path):
        for line in io.open(path, encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            if r['acordao_id'] in vistos:
                continue
            vistos.add(r['acordao_id'])
            pid = pages.get(r['acordao_id'])
            if pid:
                rot = (f"acórdão no processo n. "
                       f"{r.get('numero_processo') or '?'}")
                itens.append({'chave': f"ac:{r['acordao_id']}",
                              'page': pid, 'rotulo': rot,
                              'texto': (r.get('teor') or '')[:20000],
                              'oque': 'um acórdão',
                              'resumo_existente': r.get('dispositivo', '')})
    return itens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', choices=['atas', 'acordaos'], required=True)
    ap.add_argument('--limite', type=int, default=0)
    ap.add_argument('--model', default=MODELO)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    itens = fontes_atas() if args.base == 'atas' else fontes_acordaos()
    cache = load_cache()
    pend = []
    for it in itens:
        c = cache.get(it['chave'])
        if not (c and c.get('hash') == hash_txt(it['texto'])
                and c.get('gravado')):
            pend.append(it)
    if args.limite:
        pend = pend[:args.limite]
    print(f'{args.base}: {len(itens)} itens | pendentes: {len(pend)}')

    cont = {'ok': 0, 'err': 0}

    def do(it):
        try:
            c = cache.get(it['chave'])
            h = hash_txt(it['texto'])
            if c and c.get('hash') == h:
                resumo, kw = c['resumo'], c['palavras']
            else:
                data = llm.chat(SYSTEM,
                                USER_TMPL.format(**it),
                                json_mode=True, model=args.model,
                                max_output_tokens=1200)
                if isinstance(data, str):
                    data = json.loads(data)
                resumo = (data.get('resumo') or '').strip()
                kw = data.get('palavras_chave') or []
                if isinstance(kw, str):
                    kw = [k.strip() for k in kw.split(',')]
            props = {'palavras_chave':
                     {'rich_text': ln._rt_chunks(', '.join(kw))}}
            if not it.get('resumo_existente'):
                props['resumo_ia'] = {'rich_text': ln._rt_chunks(resumo)}
            ln._req('PATCH', f"/pages/{it['page']}", {'properties': props})
            gravar_cache({'chave': it['chave'], 'hash': h, 'resumo': resumo,
                          'palavras': kw, 'gravado': True})
            with _lock:
                cont['ok'] += 1
                if cont['ok'] % 50 == 0:
                    print(f"  ... {cont['ok']}")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f"  ERRO {it['chave']}: {str(e)[:150]}")

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(do, pend))
    print(f"[FIM] ok={cont['ok']} erros={cont['err']}")


if __name__ == '__main__':
    main()
