# -*- coding: utf-8 -*-
"""tsje_13: popula a database Notion "Atas — Sessões do TSJE" com as
transcricoes fieis de D:\\TSJE_TRANSCRICOES.

Fonte da verdade = front-matter de cada .md (ha atas arquivadas em pasta de
outro ano, com `corrigido_de`; a data_sessao do front-matter vence). Entram:
- atas do manifest com status transcrita/revisada/final (mesma regra do
  tsje_07), e
- lacuna-*.md com corpo, tribunal superior e sem `problema:` (recuperadas).

Corpo da pagina = titulo impresso (heading) + paragrafos fieis; blocos alem
de 100 sao anexados via PATCH /blocks. id_seq = AAAAMMDD da sessao (particao
estavel). Idempotente: manifesto atas_inserted.jsonl com chave ata_id +
reconciliacao contra o estado real do database.

  python tsje_13_popular_atas.py                # dry-run
  python tsje_13_popular_atas.py --apply [--limite N]
"""
import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import re
import sys
import threading

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _lia_notion as ln
import _stj_lib as L

RAIZ = r'D:\TSJE_TRANSCRICOES'
MANIFEST_CSV = os.path.join(RAIZ, 'manifest.csv')
IDS_PATH = r'D:\TSJE_TRABALHO\tsje_notion_ids.json'
WORK = r'D:\TSJE_TRABALHO\tsje_notion_work'
INSERTED = os.path.join(WORK, 'atas_inserted.jsonl')

_lock = threading.Lock()


def front_matter(path):
    with io.open(path, encoding='utf-8') as f:
        texto = f.read()
    m = re.match(r'---\n(.*?)\n---\n?(.*)', texto, re.S)
    fm = {}
    if not m:
        return fm, texto.strip()
    for linha in m.group(1).split('\n'):
        if ':' in linha:
            k, v = linha.split(':', 1)
            fm[k.strip()] = v.strip()
    return fm, m.group(2).strip()


def coletar():
    """Lista de registros a inserir, cada um {ata_id, fm, corpo, origem}."""
    regs = []
    vistos = set()
    with io.open(MANIFEST_CSV, encoding='utf-8-sig') as f:
        man = list(csv.DictReader(f))
    for r in man:
        if r['status'] not in ('transcrita', 'revisada', 'final'):
            continue
        path = os.path.join(RAIZ, r['transcricao'])
        if not os.path.exists(path):
            continue
        fm, corpo = front_matter(path)
        # 'problema' so invalida se for duplicata/cabecalho (mesma regra do
        # tsje_08); notas como 'continua_fora_do_recorte' nao excluem a ata
        if not corpo or re.match(r'(duplicata|cabecalho)',
                                 fm.get('problema', '')):
            continue
        aid = fm.get('ata_id') or r['ata_id']
        if aid in vistos:
            continue
        vistos.add(aid)
        regs.append({'ata_id': aid, 'fm': fm, 'corpo': corpo,
                     'origem': 'índice'})
    for ano in sorted(os.listdir(RAIZ)):
        d = os.path.join(RAIZ, ano)
        if not (os.path.isdir(d) and ano.isdigit()):
            continue
        for nome in sorted(os.listdir(d)):
            if not (nome.startswith(('lacuna-', 'rev-'))
                    and nome.endswith('.md')):
                continue
            fm, corpo = front_matter(os.path.join(d, nome))
            if not corpo or fm.get('problema') \
                    or fm.get('tribunal') != 'superior':
                continue
            aid = fm.get('ata_id') or nome[:-3]
            if aid in vistos:
                continue
            vistos.add(aid)
            regs.append({'ata_id': aid, 'fm': fm, 'corpo': corpo,
                         'origem': 'recuperada'})
    return regs


TIPO_NOTION = {'ordinaria': 'ordinária', 'extraordinaria': 'extraordinária'}


def nome_da_ata(fm):
    num, tipo = fm.get('num', ''), fm.get('tipo', '')
    d = fm.get('data_sessao', '')
    dd = f'{d[8:10]}/{d[5:7]}/{d[:4]}' if len(d) == 10 else d
    t = fm.get('titulo', '')
    if num and tipo in TIPO_NOTION:
        return f'{num}ª Sessão {TIPO_NOTION[tipo].capitalize()} — {dd}'
    tn = norm_titulo(t)
    return f'{tn} — {dd}' if dd else (tn or 'Sessão sem título')


def norm_titulo(t):
    """"SESSÃO PREPARATÓRIA, EM 17..." -> "Sessão Preparatória"."""
    t = re.split(r',?\s+em\s+', t, flags=re.I)[0].strip()
    t = re.sub(r'^ata da\s+', '', t, flags=re.I)
    if t.isupper():
        t = t.title()
        t = re.sub(r'\b(De|Da|Do|Das|Dos|E)\b',
                   lambda m: m.group(1).lower(), t)
    return t


def blocos_do_corpo(fm, corpo):
    blocos = [{'object': 'block', 'type': 'heading_3', 'heading_3': {
        'rich_text': ln._rt_chunks(fm.get('titulo', ''))}}]
    if fm.get('presidencia'):
        blocos.append({'object': 'block', 'type': 'paragraph', 'paragraph': {
            'rich_text': [{'type': 'text',
                           'text': {'content': fm['presidencia'][:1900]},
                           'annotations': {'italic': True}}]}})
    for par in corpo.split('\n\n'):
        par = par.strip()
        if par:
            blocos.append({'object': 'block', 'type': 'paragraph',
                           'paragraph': {'rich_text': ln._rt_chunks(par)}})
    rodape = (f'Fonte: Boletim Eleitoral {fm.get("boletim", "")} — '
              f'{fm.get("arquivo", "")}, pág. {fm.get("paginas", "")}')
    blocos.append({'object': 'block', 'type': 'paragraph', 'paragraph': {
        'rich_text': [{'type': 'text', 'text': {'content': rodape[:1900]},
                       'annotations': {'italic': True}}]}})
    if fm.get('problema'):
        blocos.append({'object': 'block', 'type': 'paragraph', 'paragraph': {
            'rich_text': [{'type': 'text',
                           'text': {'content':
                                    f'Nota: {fm["problema"]}'[:1900]},
                           'annotations': {'italic': True}}]}})
    return blocos


def props_da_ata(reg):
    fm = reg['fm']
    d = fm.get('data_sessao', '')
    data_ok = bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', d))
    ano = d[:4] if data_ok else ''
    num = fm.get('num', '')
    props = ln.build_props(
        title=nome_da_ata(fm),
        select={'tipo': TIPO_NOTION.get(fm.get('tipo', ''),
                                        '' if fm.get('tipo') else ''),
                'ano': ano,
                'confianca': fm.get('confianca', ''),
                'origem': reg['origem']},
        rich={'boletim': fm.get('boletim', ''),
              'arquivo_pdf': fm.get('arquivo', ''),
              'paginas': fm.get('paginas', ''),
              'presidencia': fm.get('presidencia', ''),
              'ata_id': reg['ata_id']},
        number={},
        checkbox={'incluir_no_rag': True},
    )
    if data_ok:
        props['data_sessao'] = {'date': {'start': d}}
        props['id_seq'] = {'number': int(d.replace('-', ''))}
    if num.isdigit():
        props['num'] = {'number': int(num)}
    il = fm.get('ilegiveis', '')
    if il.isdigit():
        props['ilegiveis'] = {'number': int(il)}
    if not fm.get('tipo') and not props.get('tipo', {}).get('select'):
        t = fm.get('titulo', '').lower()
        if 'preparat' in t:
            props['tipo'] = {'select': {'name': 'preparatória'}}
        elif 'instala' in t:
            props['tipo'] = {'select': {'name': 'instalação'}}
        else:
            props['tipo'] = {'select': {'name': 'outra'}}
    return props


def load_inserted():
    m = {}
    if os.path.exists(INSERTED):
        for line in io.open(INSERTED, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                m[r['chave']] = r['id']
    return m


def gravar_inserted(chave, pid):
    with _lock:
        with io.open(INSERTED, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'chave': chave, 'id': pid},
                               ensure_ascii=False) + '\n')


def reconciliar(db_id, manif):
    pages = L.query_all(db_id)
    novos = 0
    for pg in pages:
        aid = L.plain(pg['properties'].get('ata_id'))
        if aid and aid not in manif:
            manif[aid] = pg['id']
            gravar_inserted(aid, pg['id'])
            novos += 1
    if novos:
        print(f'reconciliação: +{novos} do estado real do database')
    return manif


def inserir(db_id, reg):
    props = props_da_ata(reg)
    blocos = blocos_do_corpo(reg['fm'], reg['corpo'])
    pid = ln.inserir_linha(db_id, props, children=blocos[:100])
    for i in range(100, len(blocos), 100):
        ln._req('PATCH', f'/blocks/{pid}/children',
                {'children': blocos[i:i + 100]})
    return pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--limite', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    with io.open(IDS_PATH, encoding='utf-8') as f:
        db_id = json.load(f)['db_atas']

    regs = coletar()
    manif = load_inserted()
    if args.apply:
        manif = reconciliar(db_id, manif)
    pend = [r for r in regs if r['ata_id'] not in manif]
    if args.limite:
        pend = pend[:args.limite]
    print(f'atas coletadas: {len(regs)} | inseridas: {len(manif)} | '
          f'pendentes: {len(pend)}')

    if not args.apply:
        for r in pend[:8]:
            print(f"  - {nome_da_ata(r['fm'])} | {r['origem']} | "
                  f"corpo {len(r['corpo'])} chars")
        print('[DRY-RUN] rode com --apply')
        return

    cont = {'ok': 0, 'err': 0}

    def do(reg):
        try:
            pid = inserir(db_id, reg)
            gravar_inserted(reg['ata_id'], pid)
            with _lock:
                cont['ok'] += 1
                if cont['ok'] % 50 == 0:
                    print(f'  ... {cont["ok"]} inseridas')
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f'  ERRO {reg["ata_id"]}: {str(e)[:180]}')

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(do, pend))
    print(f'[FIM] inseridas={cont["ok"]} erros={cont["err"]}')


if __name__ == '__main__':
    main()
