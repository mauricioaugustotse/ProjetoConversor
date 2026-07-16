# -*- coding: utf-8 -*-
r"""acv_03: integra os OMISSOS de D:\ACERVO ao acervo C:\Users\mauri\HD_Mau,
renomeando e organizando conforme a convencao do 00 - INDICE.md.

Decisao de destino em camadas:
 1. Regras deterministicas: CNJ no nome/trecho -> 01 - Juridico (tipo por
    keyword; MATERIA pelo digito J do CNJ - autoritativa sobre a IA);
    contracheque/fatura/imposto/curriculo/etc -> 04; planilhas -> 03;
    executaveis/instaladores -> 05\Informatica.
 2. IA (gpt-5.4-nano, lotes de 12): destino na whitelist + novo nome na
    convencao + confianca. Gate < 0.6 -> 05 - Diversos.
 3. Enderecamento fino: subpastas de faixa de ano/alfabeticas existentes.

Movimento D:->C: = copy2 + verificacao SHA1 (hash de origem ja esta no
acv_confronto.sqlite) + remocao da origem. Log log_acv_integra_<ts>.csv.

Uso: python acv_03_organizar.py --plano   (so gera acv_plano_integra.csv)
     python acv_03_organizar.py --exec    (aplica o plano gravado)
"""
import csv
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from acv_02_cruzar import extrair_completo, lp

ROOT = r'C:\Users\mauri\HD_Mau'
DB = os.path.join(BASE, 'acv_confronto.sqlite')
CAB_D = os.path.join(BASE, 'acv_cabecalhos_d.tsv')
RESP = os.path.join(BASE, 'acv_ia_integra.jsonl')
PLANO = os.path.join(BASE, 'acv_plano_integra.csv')
MODELO = 'gpt-5.6-luna'  # luna para tudo (13/07/2026); ~1.2k primeiras respostas vieram do sol (cache aproveitado)
MAXPATH = 255
GATE = 0.6

RE_CNJ = re.compile(r'(\d{7})-?(\d{2})\.?(\d{4})\.?(\d)\.?(\d{2})\.?(\d{4})')
MATERIA_J = {'4': 'Civel', '8': 'Civel', '5': 'Trabalhista', '6': 'Eleitoral',
             '2': 'Administrativo', '1': 'Outros', '3': 'Outros'}
TIPOS_JUR = [
    (re.compile(r'\bacord[aã]o|\bement[aá]', re.I), 'Acordaos e decisoes'),
    (re.compile(r'\bcontesta|manifesta[cç][aã]o|impugna', re.I), 'Contestacoes e manifestacoes'),
    (re.compile(r'\bdespacho|\bvoto\b|\bsenten[cç]a', re.I), 'Despachos e votos'),
    (re.compile(r'\bminuta|\bmodelo', re.I), 'Modelos e minutas'),
    (re.compile(r'\bparecer|nota t[eé]cnica', re.I), 'Pareceres e notas tecnicas'),
    (re.compile(r'peti[cç][aã]o|inicial\b', re.I), 'Peticoes'),
    (re.compile(r'\brecurso|\bREspe?\b|\bAgR\b|\bapela[cç]|embargos|agravo', re.I), 'Recursos'),
]
REGRAS_04 = [
    (re.compile(r'contra.?cheque|holerite|recibo|comprovante', re.I),
     r'04 - Pessoal e financeiro\Comprovantes e recibos'),
    (re.compile(r'\bfatura|\bboleto', re.I), r'04 - Pessoal e financeiro\Faturas e boletos'),
    (re.compile(r'\bIRPF\b|\bDIRPF\b|imposto de renda|\bDARF\b', re.I),
     r'04 - Pessoal e financeiro\Impostos'),
    (re.compile(r'curr[ií]cul|\bCV\b|lattes', re.I),
     r'04 - Pessoal e financeiro\Documentos pessoais\Curriculos'),
    (re.compile(r'certid[aã]o|diploma|certificado', re.I),
     r'04 - Pessoal e financeiro\Documentos pessoais\Certificados, diplomas e certidoes'),
    (re.compile(r'atestado|exame|laudo m[eé]dico|resultado.*exame', re.I),
     r'04 - Pessoal e financeiro\Documentos pessoais\Saude e atestados'),
    (re.compile(r'ap[oó]lice|seguro\b', re.I),
     r'04 - Pessoal e financeiro\Documentos pessoais\Seguros e apolices'),
]
EXT_REGRA = {
    '.exe': r'05 - Diversos\Informatica',
    '.dll': r'05 - Diversos\Informatica',
    '.jar': r'05 - Diversos\Informatica',
    '.fll': r'05 - Diversos\Informatica',
    '.aui': r'05 - Diversos\Informatica',
    '.wma': r'05 - Diversos\Outros',
    '.mp3': r'05 - Diversos\Outros',
}
RE_FAIXA_ANO = re.compile(r'^(\d{4})(?:-(\d{4}))?$')
RE_FAIXA_ALFA = re.compile(r'^(0?[0-9A-Za-z]{1,2})-([0-9A-Za-z]{1,2})$')


def ascii_seg(s, maxlen=120):
    t = unicodedata.normalize('NFD', s)
    t = ''.join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r'[\\/:*?"<>|\x00-\x1f]', ' ', t)
    t = re.sub(r'\s{2,}', ' ', t).strip(' .')
    return t[:maxlen].strip(' .') or 'sem nome'


def chave_openai():
    for linha in io.open(os.path.join(BASE, '.env'), encoding='utf-8'):
        if linha.startswith('OPENAI_API_KEY='):
            return linha.split('=', 1)[1].strip().strip('"')
    raise RuntimeError('sem chave')


def whitelist():
    """Destinos validos: cat\\tipo e cat\\tipo\\materia existentes (sem faixas)."""
    dests = []
    for cat in sorted(os.listdir(ROOT)):
        pc = os.path.join(ROOT, cat)
        if not os.path.isdir(pc) or not re.match(r'^0[1-5] - ', cat):
            continue
        for tipo in sorted(os.listdir(pc)):
            pt = os.path.join(pc, tipo)
            if not os.path.isdir(pt):
                continue
            dests.append(f'{cat}\\{tipo}')
            for sub in sorted(os.listdir(pt)):
                ps = os.path.join(pt, sub)
                if os.path.isdir(ps) and not RE_FAIXA_ANO.match(sub) \
                        and not RE_FAIXA_ALFA.match(sub):
                    dests.append(f'{cat}\\{tipo}\\{sub}')
    return dests


RE_CTX_PROC = re.compile(
    r'processo|autos|relator|senten[cç]a|ac[oó]rd[aã]o|\bvara\b|ju[ií]z|'
    r'tribunal|recurso|peti[cç]|intima[cç]|cita[cç]', re.I)


def regra_deterministica(nome, trecho, ext):
    # CNJ no NOME decide sozinho; CNJ só no trecho exige contexto processual
    # (boleto GRU de concurso tem numeração que casa o padrão CNJ)
    m = RE_CNJ.search(nome)
    if not m:
        m = RE_CNJ.search(trecho[:400])
        if m and not RE_CTX_PROC.search(trecho):
            m = None
    if m:
        materia = MATERIA_J.get(m.group(4), 'Outros')
        tipo = 'Outros processuais'
        for rx, t in TIPOS_JUR:
            if rx.search(nome):
                tipo = t
                break
        return f'01 - Juridico\\{tipo}\\{materia}', 'cnj'
    for rx, dest in REGRAS_04:
        if rx.search(nome):
            return dest, 'regra04'
    if ext in EXT_REGRA:
        return EXT_REGRA[ext], 'ext'
    return None, None


def endereco_fino(dest_rel, nome, ano):
    """Desce para subpasta de faixa (ano ou alfabetica) se existirem no destino."""
    p = os.path.join(ROOT, dest_rel)
    if not os.path.isdir(p):
        return dest_rel
    subs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    faixas_ano = []
    for s in subs:
        m = RE_FAIXA_ANO.match(s)
        if m:
            a1 = int(m.group(1))
            a2 = int(m.group(2) or m.group(1))
            faixas_ano.append((a1, a2, s))
    if faixas_ano and ano:
        for a1, a2, s in faixas_ano:
            if a1 <= ano <= a2:
                return f'{dest_rel}\\{s}'
        mais_prox = min(faixas_ano, key=lambda f: min(abs(ano - f[0]), abs(ano - f[1])))
        return f'{dest_rel}\\{mais_prox[2]}'
    alfas = [(s, RE_FAIXA_ALFA.match(s)) for s in subs]
    alfas = [(s, m) for s, m in alfas if m]
    if alfas:
        ini = ascii_seg(nome, 10).lower().lstrip()[:2]
        for s, m in alfas:
            if m.group(1).lower() <= ini <= m.group(2).lower() + 'zz':
                return f'{dest_rel}\\{s}'
    return dest_rel


def ano_do(nome, trecho):
    m = re.search(r'\b(19[89]\d|20[0-2]\d)\b', nome)
    if m:
        return int(m.group(1))
    m = re.search(r'\b(19[89]\d|20[0-2]\d)\b', trecho[:400])
    return int(m.group(1)) if m else None


SYS = """Voce organiza o acervo pessoal de um servidor do TSE (ex-advogado dos Correios).
O acervo tambem recebe documentos da familia (pai Mario, mae Ana Lucia) e da associacao AEED.
Categorias: 01 - Juridico (pecas processuais; tipo+materia), 02 - Estudos e concursos,
03 - Administrativo (docs de orgaos/empresas/associacoes), 04 - Pessoal e financeiro,
05 - Diversos (nao identificaveis).
Destinos validos (use EXATAMENTE um deles):
{DESTINOS}
Convencao de nomes (ASCII sem acento, segmentos " - ", minusculas exceto siglas/nomes proprios/inicio):
- 01 Juridico: "CLASSE numero - UF - assunto - MM-AAAA - CNJ ..."
- 02 Estudos: "Tema ou titulo - autor - ano"
- 03 Administrativo: "Tipo doc - assunto - orgao ou pessoa - DD-MM-AAAA"
- 04 Pessoal/financeiro: "Tipo - pessoa - detalhe - data"
Para cada item (pasta de origem, nome, trecho do conteudo):
- "destino": um caminho da lista acima;
- "novo_stem": nome novo na convencao SE o atual nao indicar o conteudo (senao "");
  PRESERVE numeros de documento, CNJ e datas do nome atual; nada de inventar; max ~110 chars;
- "confianca": 0-1.
Responda APENAS JSON: {{"itens": [{{"id": <n>, "destino": "<caminho>", "novo_stem": "<nome|vazio>", "confianca": <0-1>}}]}}"""


def rodar_ia(itens, textos, dests):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    system = SYS.replace('{DESTINOS}', '\n'.join(dests))
    ja = {}
    if os.path.exists(RESP):
        for line in io.open(RESP, encoding='utf-8'):
            try:
                d = json.loads(line)
                ja[d['path']] = d
            except (json.JSONDecodeError, KeyError):
                pass
    pend = [p for p in itens if p not in ja]
    print(f'IA: {len(pend):,} pendentes (cache {len(ja):,})', flush=True)
    lotes = [pend[i:i + 12] for i in range(0, len(pend), 12)]

    def um_lote(lote):
        linhas = [f'id={i} | origem="{"\\".join(p.split(chr(92))[2:-1])}" | '
                  f'nome="{os.path.basename(p)}" | trecho="{textos.get(p, "")[:500]}"'
                  for i, p in enumerate(lote)]
        for tent in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODELO,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': '\n\n'.join(linhas)}],
                    response_format={'type': 'json_object'})
                data = json.loads(r.choices[0].message.content)
                out = []
                for it in data.get('itens', []):
                    i = int(it.get('id', -1))
                    if 0 <= i < len(lote):
                        out.append({'path': lote[i],
                                    'destino': str(it.get('destino', '')),
                                    'novo_stem': str(it.get('novo_stem', '')),
                                    'confianca': float(it.get('confianca', 0))})
                return out
            except Exception:
                time.sleep(2 * (tent + 1))
        return [{'path': p, 'destino': '', 'novo_stem': '', 'confianca': 0}
                for p in lote]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex, \
            io.open(RESP, 'a', encoding='utf-8') as f:
        for i, res in enumerate(ex.map(um_lote, lotes), 1):
            for d in res:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
                ja[d['path']] = d
            if i % 25 == 0:
                print(f'  lote {i}/{len(lotes)} ({time.time()-t0:.0f}s)', flush=True)
    return ja


def montar_plano():
    con = sqlite3.connect(DB)
    omissos = [p for (p,) in con.execute(
        "SELECT path FROM confronto WHERE classe LIKE 'omisso%' ORDER BY path")]
    sha_d = {p: s for p, s in con.execute(
        "SELECT path, sha1 FROM hashes WHERE lado='D'")}
    con.close()
    lixo = [p for p in omissos
            if os.path.basename(p).startswith('~$')
            or os.path.splitext(p)[1].lower() == '.tmp']
    omissos = [p for p in omissos if p not in set(lixo)]
    print(f'Omissos: {len(omissos):,} (descartados {len(lixo)} temporários ~$/tmp)')
    dests = whitelist()
    print(f'Whitelist de destinos: {len(dests)}')

    # trechos p/ IA (cache proprio; extracao ja validada no acv_02)
    cab = {}
    if os.path.exists(CAB_D):
        for line in io.open(CAB_D, encoding='utf-8'):
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2:
                cab[parts[0]] = parts[1]
    falta = [p for p in omissos if p not in cab
             and os.path.splitext(p)[1].lower() in ('.pdf', '.doc', '.docx', '.rtf', '.txt')]
    if falta:
        print(f'Extraindo trecho de {len(falta):,}...', flush=True)
        def trecho(p):
            t = extrair_completo(p) or ''
            return re.sub(r'\s+', ' ', t).strip()[:1200]
        with ThreadPoolExecutor(max_workers=8) as ex, \
                io.open(CAB_D, 'a', encoding='utf-8') as f:
            for p, t in zip(falta, ex.map(trecho, falta)):
                cab[p] = t
                f.write(f'{p}\t{t}\n')

    # camada 1: regras
    decisao = {}
    for p in omissos:
        nome = os.path.basename(p)
        dest, fonte = regra_deterministica(nome, cab.get(p, ''),
                                           os.path.splitext(p)[1].lower())
        if dest:
            decisao[p] = {'destino': dest, 'fonte': fonte, 'novo_stem': '',
                          'confianca': 1.0}
    print(f'Regras deterministicas: {len(decisao):,}')

    # camada 2: IA nos demais
    para_ia = [p for p in omissos if p not in decisao]
    respostas = rodar_ia(para_ia, cab, dests)
    dests_set = set(dests)
    for p in para_ia:
        d = respostas.get(p, {'destino': '', 'novo_stem': '', 'confianca': 0})
        destino = d['destino'] if d['destino'] in dests_set else ''
        if not destino or d['confianca'] < GATE:
            destino = r'05 - Diversos\Outros'
            d['novo_stem'] = ''
            fonte = 'gate_baixa_conf'
        else:
            fonte = 'ia'
        decisao[p] = {'destino': destino, 'fonte': fonte,
                      'novo_stem': d['novo_stem'], 'confianca': d['confianca']}

    # nome final + endereco fino + colisoes
    usados = set()
    linhas = []
    for p in omissos:
        d = decisao[p]
        ext = os.path.splitext(p)[1].lower()
        stem_orig = os.path.splitext(os.path.basename(p))[0]
        stem = d['novo_stem'].strip() or stem_orig
        # validacao do nome IA: preservar CNJ/numeros longos do original
        for m in RE_CNJ.finditer(stem_orig):
            if m.group(0) not in stem:
                stem = stem_orig
                break
        stem = ascii_seg(stem, 110)
        dest_rel = endereco_fino(d['destino'], stem,
                                 ano_do(stem_orig, cab.get(p, '')))
        pasta = os.path.join(ROOT, dest_rel)
        maxstem = max(20, MAXPATH - len(pasta) - len(ext) - 6)
        stem = stem[:maxstem].rstrip(' .-')
        destino = os.path.join(pasta, stem + ext)
        n = 1
        while destino.lower() in usados or os.path.exists(lp(destino)):
            n += 1
            destino = os.path.join(pasta, f'{stem} ({n}){ext}')
        usados.add(destino.lower())
        linhas.append((p, destino, d['fonte'], f"{d['confianca']:.2f}", sha_d.get(p, '')))

    with open(PLANO, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['origem', 'destino', 'fonte', 'confianca', 'sha1'])
        w.writerows(linhas)
    stats = Counter(x[2] for x in linhas)
    tops = Counter('\\'.join(x[1][len(ROOT) + 1:].split('\\')[:2]) for x in linhas)
    print(f'\nPlano: {len(linhas):,} -> {PLANO}')
    print(f'Fontes: {dict(stats)}')
    print('Destinos (top 15):')
    for k, v in tops.most_common(15):
        print(f'  {v:>5,}  {k}')


def executar():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_integra_{ts}.csv')
    ok = err = 0
    with open(PLANO, encoding='utf-8-sig') as f, \
            open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['origem', 'destino', 'status'])
        linhas = list(csv.DictReader(f))
        for i, r in enumerate(linhas, 1):
            src, dst, sha1 = r['origem'], r['destino'], r['sha1']
            status = 'ok'
            try:
                if not os.path.exists(lp(src)):
                    status = 'origem_sumiu'
                elif os.path.exists(lp(dst)):
                    status = 'destino_ocupado'
                else:
                    os.makedirs(lp(os.path.dirname(dst)), exist_ok=True)
                    shutil.copy2(lp(src), lp(dst))
                    h = hashlib.sha1()
                    with open(lp(dst), 'rb') as fd:
                        for chunk in iter(lambda: fd.read(1 << 20), b''):
                            h.update(chunk)
                    if sha1 and h.hexdigest() != sha1:
                        os.remove(lp(dst))
                        status = 'sha1_divergente'
                    else:
                        os.remove(lp(src))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([src, dst, status])
            if status == 'ok':
                ok += 1
            else:
                err += 1
            if i % 500 == 0:
                print(f'  {i:,}/{len(linhas):,} (problemas {err})', flush=True)
    print(f'Integrados: {ok:,} | problemas: {err:,} | log: {log_path}')


if __name__ == '__main__':
    if '--exec' in sys.argv:
        executar()
    else:
        montar_plano()
