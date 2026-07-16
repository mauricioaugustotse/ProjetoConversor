# -*- coding: utf-8 -*-
"""tsje_23: depuracao da fila de acordaos pendentes com o auxiliar
gpt-5.6-sol.

Tres filas (de tsje_acordaos_associacao.csv + acordaos_teor.jsonl):
1. AMBIGUO (~96): sol compara o TEOR do acordao com os processos
   candidatos (mesmo numero em atas distintas) e escolhe — ou declara
   insoluvel. Aplica a relation "Processo" na pagina do acordao.
2. SEM_NUMERO (~98): sol le a IMAGEM da pagina do acordao e recupera o
   numero/classe/UF impressos; atualiza a property e tenta re-casar
   (cascata do tsje_17); aplica relation quando casa.
3. GATE (13): acordaos sem teor por gate anti-alucinacao; re-extracao com
   truncamento maior (48k) via motor do tsje_16b; se passar, o tsje_17
   os popula na proxima execucao.

Cache idempotente: tsje_acordaos_rev_cache.jsonl. Log final:
tsje_acordaos_revisao.csv.

Uso: python tsje_23_acordaos_fila.py [--fila ambiguo|semnumero|gate|todas]
                                     [--limite N] [--apply]
"""
import argparse
import base64
import csv
import importlib.util
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

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO = 'gpt-5.6-sol'
TRAB = r'D:\TSJE_TRABALHO'
WORK = os.path.join(TRAB, 'tsje_notion_work')
EXTRAIDO = os.path.join(TRAB, 'extraido')
BANCO = r'D:\TSJE_ATAS'
RENDER = r'D:\TSJE_TRANSCRICOES\_paginas_rev'
ASSOC = os.path.join(TRAB, 'tsje_acordaos_associacao.csv')
ACORD_INS = os.path.join(WORK, 'acordaos_inserted.jsonl')
CACHE = os.path.join(TRAB, 'tsje_acordaos_rev_cache.jsonl')
SAIDA = os.path.join(TRAB, 'tsje_acordaos_revisao.csv')
DPI = 200

_lock = threading.Lock()


def _mod(nome):
    spec = importlib.util.spec_from_file_location(
        nome, os.path.join(BASE, nome + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t17 = _mod('tsje_17_popular_acordaos')
ln = t17.ln  # helper notion (_lia_notion)


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
        zoom = DPI / 72.0
        pix = doc[pag - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                      colorspace=fitz.csGRAY)
        pix.save(lp(alvo))
        doc.close()
        return alvo
    except Exception:
        return None


def detalhes_candidatos(idx_extr, nn):
    """Para o prompt do sol: dados ricos de cada candidato."""
    out = []
    for proc_id, data, classe, uf in idx_extr.get(nn, []):
        out.append({'proc_id': proc_id, 'data_sessao': data,
                    'classe': classe, 'uf': uf})
    return out


def enriquecer_candidatos(cands):
    """Adiciona ementa/partes do processos_extraidos.jsonl."""
    quer = {c['proc_id'] for c in cands}
    extra = {}
    with io.open(t17.PROC_EXTR, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for i, d in enumerate(r.get('deliberacoes', []), 1):
                pid = f"{r['ata_id']}#{i:02d}"
                if pid in quer:
                    extra[pid] = {
                        'ementa': (d.get('ementa') or '')[:400],
                        'partes': ', '.join(filter(None, [
                            d.get('parte_recorrente') or '',
                            d.get('parte_recorrida') or '',
                            d.get('interessado') or '']))[:200],
                        'relator': d.get('relator') or ''}
    for c in cands:
        c.update(extra.get(c['proc_id'], {}))
    return cands


SYSTEM_DESEMPATE = """Você é um jurista-arquivista do Tribunal Superior de \
Justiça Eleitoral (Brasil, 1932-1937). Um ACÓRDÃO publicado no Boletim \
Eleitoral precisa ser associado à deliberação correspondente registrada em \
ata de sessão. Você receberá o teor do acórdão e a lista de deliberações \
candidatas (mesmo número de processo em sessões diferentes). Escolha a \
correspondente comparando: partes, UF, classe, relator, matéria e datas \
(o julgamento em sessão precede a publicação; o acórdão costuma ser \
assinado na sessão ou pouco depois). Responda SOMENTE JSON:
{"escolha": "<proc_id>"|null, "confianca": "alta|media|baixa",
 "justificativa": "1 frase"}
Se nenhum candidato corresponder com segurança, "escolha": null. NÃO chute."""

SYSTEM_NUMERO = """Você é um paleógrafo lendo um acórdão do Tribunal \
Superior de Justiça Eleitoral impresso em Boletim Eleitoral (1932-1937). \
Na imagem, localize o acórdão indicado e leia o cabeçalho da unidade \
(padrões: "N. 66 — Classe 3ª", "Processo n. 273", "Recurso eleitoral \
n. 1.093 — Classe 3ª — Estado de ..."). Responda SOMENTE JSON:
{"numero_processo": "<como impresso>"|null, "classe": "..."|null,
 "uf": ".."|null, "evidencia": "trecho do cabeçalho lido"}
NÃO invente: null quando não conseguir ler."""


def chamar(cli, system, conteudo, max_tokens=1200):
    ultimo = None
    for i in range(3):
        try:
            resp = cli.responses.create(
                model=MODELO,
                input=[{'role': 'system', 'content': system},
                       {'role': 'user', 'content': conteudo}],
                max_output_tokens=max_tokens)
            texto = (resp.output_text or '').strip()
            texto = re.sub(r'^```(?:json)?|```$', '', texto, flags=re.M).strip()
            return json.loads(texto)
        except Exception as e:
            ultimo = e
            if i < 2:
                time.sleep(3.0 * (i + 1))
    raise RuntimeError(f'sol falhou: {ultimo}')


def acordao_pages():
    m = {}
    if os.path.exists(ACORD_INS):
        with io.open(ACORD_INS, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    m[r['chave']] = r['id']
    return m


def aplicar_relation(ac_page, proc_page):
    ln._req('PATCH', f'/pages/{ac_page}',
            {'properties': {'Processo': {'relation': [{'id': proc_page}]}}})


def aplicar_numero(ac_page, numero, classe, uf):
    props = {'numero_processo': {'rich_text': [
        {'type': 'text', 'text': {'content': numero[:1900]}}]}}
    if classe:
        props['classe'] = {'select': {'name': classe[:95]}}
    if uf:
        props['UF'] = {'select': {'name': uf[:5]}}
    ln._req('PATCH', f'/pages/{ac_page}', {'properties': props})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fila', default='todas',
                    choices=['ambiguo', 'semnumero', 'gate', 'todas'])
    ap.add_argument('--limite', type=int, default=0)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    chave = io.open(os.path.join(BASE, 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)

    teor = {r['acordao_id']: r for r in t17.carregar_teor()}
    idx = t17.indice_processos()
    pages_proc = t17.proc_pages()
    pages_ac = acordao_pages()

    feitos = {}
    if os.path.exists(CACHE):
        with io.open(CACHE, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    feitos[d['acordao_id']] = d

    assoc = list(csv.DictReader(io.open(ASSOC, encoding='utf-8-sig')))
    fila_amb = [r for r in assoc if r['veredito'] == 'AMBIGUO'
                and r['acordao_id'] not in feitos]
    fila_num = [r for r in assoc if r['veredito'] == 'SEM_NUMERO'
                and r['acordao_id'] not in feitos]
    if args.limite:
        fila_amb, fila_num = fila_amb[:args.limite], fila_num[:args.limite]

    cont = {'resolvido': 0, 'insoluvel': 0, 'num_ok': 0, 'num_nao': 0,
            'err': 0, 'aplicado': 0}

    def grava(d):
        with _lock:
            with io.open(CACHE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # --- fila 1: ambiguos (texto)
    def do_amb(r):
        aid = r['acordao_id']
        try:
            ac = teor.get(aid) or {}
            nn = t17.num_norm(r['numero'])
            cands = enriquecer_candidatos(detalhes_candidatos(idx, nn))
            if not cands:
                grava({'acordao_id': aid, 'fila': 'ambiguo',
                       'resultado': 'sem_candidatos'})
                return
            perg = [{'type': 'input_text', 'text':
                     f"ACÓRDÃO (processo n. {r['numero']}, publicado em "
                     f"{r['pdf'][-40:]} p.{r['pagina']}, data de julgamento "
                     f"impressa: {r['data_julgamento'] or '?'}):\n"
                     f"{(ac.get('teor') or '')[:3500]}\n\n"
                     f"CANDIDATOS:\n"
                     + json.dumps(cands, ensure_ascii=False, indent=1)}]
            res = chamar(cli, SYSTEM_DESEMPATE, perg)
            escolha = res.get('escolha')
            ok = (escolha and escolha in {c['proc_id'] for c in cands}
                  and res.get('confianca') in ('alta', 'media'))
            d = {'acordao_id': aid, 'fila': 'ambiguo',
                 'resultado': 'resolvido' if ok else 'insoluvel',
                 'proc_id': escolha if ok else '',
                 'confianca': res.get('confianca'),
                 'justificativa': (res.get('justificativa') or '')[:200]}
            if ok and args.apply:
                ac_page = pages_ac.get(aid)
                proc_page = pages_proc.get(escolha)
                if ac_page and proc_page:
                    aplicar_relation(ac_page, proc_page)
                    d['aplicado'] = True
                    with _lock:
                        cont['aplicado'] += 1
            grava(d)
            with _lock:
                cont['resolvido' if ok else 'insoluvel'] += 1
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f'  ERRO amb {aid}: {str(e)[:90]}')

    # --- fila 2: sem numero (visao)
    def do_num(r):
        aid = r['acordao_id']
        try:
            png = render(r['pdf'], int(r['pagina']))
            if not png:
                grava({'acordao_id': aid, 'fila': 'semnumero',
                       'resultado': 'render_falhou'})
                return
            ac = teor.get(aid) or {}
            trecho = (ac.get('teor') or '')[:300]
            with open(lp(png), 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            perg = [
                {'type': 'input_text', 'text':
                 f'O acórdão-alvo começa assim: "{trecho}". Localize-o na '
                 f'página e leia o cabeçalho da unidade.'},
                {'type': 'input_image',
                 'image_url': f'data:image/png;base64,{b64}'}]
            res = chamar(cli, SYSTEM_NUMERO, perg)
            numero = res.get('numero_processo')
            d = {'acordao_id': aid, 'fila': 'semnumero',
                 'resultado': 'numero_lido' if numero else 'ilegivel',
                 'numero': numero or '', 'classe': res.get('classe') or '',
                 'uf': res.get('uf') or '',
                 'evidencia': (res.get('evidencia') or '')[:150]}
            if numero and args.apply:
                ac_page = pages_ac.get(aid)
                if ac_page:
                    aplicar_numero(ac_page, numero, d['classe'], d['uf'])
                    # tenta re-casar
                    ac2 = dict(ac)
                    ac2['numero_processo'] = numero
                    ac2['data_julgamento'] = r['data_julgamento']
                    verd, proc_id, det = t17.casar(ac2, idx)
                    d['recasamento'] = f'{verd}:{det}'
                    if verd in ('CONFIRMADO', 'PROVAVEL'):
                        proc_page = pages_proc.get(proc_id)
                        if proc_page:
                            aplicar_relation(ac_page, proc_page)
                            d['aplicado'] = True
                            with _lock:
                                cont['aplicado'] += 1
            grava(d)
            with _lock:
                cont['num_ok' if numero else 'num_nao'] += 1
        except Exception as e:
            with _lock:
                cont['err'] += 1
                print(f'  ERRO num {aid}: {str(e)[:90]}')

    if args.fila in ('ambiguo', 'todas'):
        print(f'fila AMBIGUO: {len(fila_amb)}')
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(do_amb, fila_amb))
    if args.fila in ('semnumero', 'todas'):
        print(f'fila SEM_NUMERO: {len(fila_num)}')
        with ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(do_num, fila_num))
    if args.fila in ('gate', 'todas'):
        # sem-teor (gate anti-alucinacao barrou a limpeza LLM 2x):
        # grava o texto BRUTO do OCR como teor (confianca 'bruta') —
        # honesto e pesquisavel, sem risco de invencao
        brutos = {}
        with io.open(os.path.join(TRAB, 'tsje_acordaos_texto.jsonl'),
                     encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    brutos[r['acordao_id']] = r
        sem_teor = [aid for aid in brutos
                    if aid not in teor and aid not in feitos]
        print(f'fila GATE (sem teor): {len(sem_teor)}')
        for aid in sem_teor:
            reg = dict(brutos[aid])
            bruto = re.sub(r'[ \t]+', ' ', reg.pop('texto', '') or '')
            if not bruto.strip():
                grava({'acordao_id': aid, 'fila': 'gate',
                       'resultado': 'sem_texto_bruto'})
                continue
            if args.apply:
                reg['teor'] = bruto
                reg['confianca'] = 'bruta'
                with io.open(t17.TEOR, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(reg, ensure_ascii=False) + '\n')
            grava({'acordao_id': aid, 'fila': 'gate',
                   'resultado': 'teor_bruto_gravado' if args.apply
                   else 'teor_bruto_pendente'})
            cont['resolvido'] += 1

    print(f"[FIM] {cont}")

    # consolida CSV
    linhas = []
    with io.open(CACHE, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                linhas.append(json.loads(line))
    if linhas:
        campos = sorted({k for l in linhas for k in l})
        with io.open(SAIDA, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(linhas)
        print(f'-> {SAIDA} ({len(linhas)} registros)')


if __name__ == '__main__':
    main()
