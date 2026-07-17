# -*- coding: utf-8 -*-
"""tsje_26: normaliza os selects contaminados por lixo de OCR nas 3 bases
TSJE do Notion (achados da auditoria de 17/07/2026).

Alvos: relator (acordaos+processos), UF (acordaos+processos), classe
(acordaos+processos), quorum/votacao (processos, multi_select),
confianca (atas: media->média).

Etapas: dump paginado -> tabela canonica de ministros (semente
MINISTROS_ABREV + agrupamento por grafia de epoca) -> plano CSV ->
mini-lotes luna (relator irrecuperavel via teor; UF composta via
ementa+decisao) -> apply (PATCH paginas) -> limpeza das options orfas
do schema.

Uso: python tsje_26_normalizar_selects.py [--apply] [--limite N]
Artefatos: D:\\TSJE_TRABALHO\\tsje_norm_dump.jsonl, tsje_norm_plano.csv,
           tsje_notion_work\\norm_done.jsonl
"""
import argparse
import csv
import importlib.util
import io
import json
import os
import re
import sys
import threading
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
TRAB = r'D:\TSJE_TRABALHO'
WORK = os.path.join(TRAB, 'tsje_notion_work')
DUMP = os.path.join(TRAB, 'tsje_norm_dump.jsonl')
PLANO = os.path.join(TRAB, 'tsje_norm_plano.csv')
DONE = os.path.join(WORK, 'norm_done.jsonl')
IDS = json.load(io.open(r'D:\TSJE_TRABALHO\tsje_notion_ids.json',
                        encoding='utf-8'))

_lock = threading.Lock()


def _mod(nome):
    spec = importlib.util.spec_from_file_location(
        nome, os.path.join(BASE, nome + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ln = _mod('_lia_notion')
t17 = _mod('tsje_17_popular_acordaos')

sys.path.insert(0, BASE)
from conle_gerador import llm  # noqa: E402
from tsje_14_extrair_processos import MINISTROS_ABREV  # noqa: E402

UFS_1930 = {'DF', 'RJ', 'SP', 'MG', 'ES', 'BA', 'SE', 'AL', 'PE', 'PB',
            'RN', 'CE', 'PI', 'MA', 'PA', 'AM', 'AC', 'MT', 'GO', 'PR',
            'SC', 'RS'}

# nomes vistos no corpus que nao estao na semente
CANONICOS_EXTRA = ['Monteiro de Salles', 'Públio de Mello',
                   'Pedro de Moura Ferro', 'Leopoldo Augusto de Lima',
                   'José Duarte', 'Vicente Piragibe', 'Castro Nunes']

PALAVRAS_LIXO = {'vistos', 'vota', 'votado', 'autos', 'parecer', 'accordao',
                 'acordao', 'designado', 'desi', 'apresentada', 'relatados',
                 'discutidos', 'divisao', 'pedida', 'jtisi', 'desembargador',
                 'ministro'}

PROPS_POR_BASE = {
    'acordaos': ['relator', 'UF', 'classe'],
    'processos': ['relator', 'UF', 'classe', 'quorum', 'votacao'],
    'atas': ['confianca'],
}


def norm(t):
    t = unicodedata.normalize('NFD', (t or '').lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9 ]', ' ', t).strip()


def chave_grafia(t):
    """Chave que unifica grafias de epoca: ll->l, ss->s, y->i, ph->f..."""
    n = norm(t).replace(' ', '')
    for a, b in (('ll', 'l'), ('ss', 's'), ('nn', 'n'), ('tt', 't'),
                 ('pp', 'p'), ('ff', 'f'), ('mm', 'm'), ('y', 'i'),
                 ('ph', 'f'), ('th', 't'), ('h', '')):
        n = n.replace(a, b)
    return n


# alternancia ORDENADA: padroes espacados ('S r l .') antes dos curtos,
# senao 'sr?s?' consome so o 'S' e deixa 'r .' para tras
RE_PREFIXO = re.compile(
    r'^(?:os?\s+|a\s+)?(?:exmo\.?\s+)?'
    r'(?:s\s*r\s*l?\s*\.?\s+|senhores?\s+|senhor\s+|srs?\.?\s+)?'
    r'(?:d\s*r\s*\.?\s+|dr\.?\s+|ministro\s+|min\.?\s+|'
    r'desembargador\s+|des\.?\s+|professor\s+|prof\.?\s+)*', re.I)


def limpar_nome(v):
    """Remove prefixos honorificos (em loop, ate estabilizar), sufixo
    'relator', sufixo pontuado ('. Habeas', '. Processo') e pega o
    primeiro nome de lista 'A; B'."""
    v = (v or '').strip()
    if ';' in v:
        v = v.split(';')[0].strip()
    v = re.sub(r'[\{\}\(\)\[\]"\']', ' ', v)
    for _ in range(4):
        antes = v
        v = RE_PREFIXO.sub('', v).strip(' .,-–')
        v = re.sub(r'\s+relator[a]?\s*$', '', v, flags=re.I)
        # sufixo de secao apos o nome: 'Plínio Casado. Habeas' (APOS o
        # prefixo, e so palavras de secao — nao comer iniciais 'A. de V.')
        v = re.sub(r'\.\s+(Habeas|Processos?|Recursos?|Consultas?|'
                   r'Registros?|Representa\w*|Embargos?)\b.*$', '', v)
        v = re.sub(r'\s+', ' ', v).strip()
        if v == antes:
            break
    return v


def eh_lixo(v):
    n = norm(v)
    toks = n.split()
    if not toks or len(v) > 45 or len(toks) > 5:
        return True
    if any(t in PALAVRAS_LIXO for t in toks):
        return True
    if len(n.replace(' ', '')) < 4:
        return True
    return False


# ---------------- dump ----------------

def dump_base(db_key, props):
    db_id = IDS[f'db_{db_key}']
    out = []
    cursor = None
    while True:
        body = {'page_size': 100}
        if cursor:
            body['start_cursor'] = cursor
        r = ln._req('POST', f'/databases/{db_id}/query', body)
        for pg in r.get('results', []):
            reg = {'base': db_key, 'id': pg['id']}
            for p in props:
                pr = pg['properties'].get(p) or {}
                if pr.get('type') == 'select':
                    reg[p] = (pr.get('select') or {}).get('name')
                elif pr.get('type') == 'multi_select':
                    reg[p] = [o['name'] for o in pr.get('multi_select') or []]
            out.append(reg)
        cursor = r.get('next_cursor')
        if not r.get('has_more'):
            break
    return out


def fazer_dump():
    with io.open(DUMP, 'w', encoding='utf-8') as f:
        for db_key, props in PROPS_POR_BASE.items():
            regs = dump_base(db_key, props)
            print(f'  {db_key}: {len(regs)} paginas')
            for r in regs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


# ---------------- canonizacao ----------------

def construir_canonicos(valores_freq):
    """Grupos por chave de grafia; canonico = forma mais frequente do
    grupo, com semente tendo prioridade."""
    semente = [x.strip() for x in MINISTROS_ABREV.split(',')] \
        + CANONICOS_EXTRA
    grupos = defaultdict(Counter)
    for v, n in valores_freq.items():
        lim = limpar_nome(v)
        if not lim or eh_lixo(lim):
            continue
        grupos[chave_grafia(lim)][lim] += n
    chaves_semente = {chave_grafia(s) for s in semente}
    for s in semente:
        grupos[chave_grafia(s)][s] += 3  # peso da semente
    canonico = {}
    for ch, c in grupos.items():
        # so vira canonico se toca a semente ou aparece 2+ vezes no corpus
        # (evita que lixo unico de OCR se auto-canonize, ex. 'Jtisi')
        if ch in chaves_semente or sum(c.values()) >= 2:
            canonico[ch] = c.most_common(1)[0][0]
    # funde canonicos-subsequencia: 'Leopoldo Lima' aponta para
    # 'Leopoldo Augusto de Lima' quando ha alvo unico com nome mais completo
    for ch, nome in list(canonico.items()):
        toks = set(norm(nome).split())
        alvos = [n for c2, n in canonico.items()
                 if c2 != ch and toks < set(norm(n).split())]
        if len(alvos) == 1:
            canonico[ch] = alvos[0]
    return canonico


def resolver_nome(v, canonico):
    """-> (novo_valor|None, metodo)."""
    lim = limpar_nome(v)
    if lim and not eh_lixo(lim):
        alvo = canonico.get(chave_grafia(lim))
        if alvo:
            return alvo, ('igual' if alvo == v else 'mesclado')
        # subsequencia: 'Leopoldo Lima' ⊂ 'Leopoldo Augusto de Lima';
        # token unico que e prefixo inequivoco: 'Laudo' -> 'Laudo de Camargo'
        toks = set(norm(lim).split())
        cands = [c for c in set(canonico.values())
                 if toks <= set(norm(c).split())]
        if len(cands) == 1:
            return cands[0], 'subsequencia'
    return None, 'lixo'


def resolver_uf(v):
    """-> (novo|None, metodo). None+luna = precisa de contexto."""
    s = re.sub(r'[^A-Z ]', '', (v or '').upper()).strip()
    if s in UFS_1930:
        return s, ('igual' if s == v else 'limpo')
    toks = [t for t in s.split() if t in UFS_1930]
    if len(toks) == 1:
        return toks[0], 'limpo'
    return None, 'luna'


def mapa_classe_num(teores):
    """'6' -> 'CONSULTA' etc., por co-ocorrencia no teor dos acordaos."""
    co = defaultdict(Counter)
    for ac in teores:
        m = re.search(r'[Cc]lasse\s+(\d{1,2})', ac.get('teor') or '')
        cl = (ac.get('classe') or '').upper()
        # exclui o rotulo generico 'PROCESSO', que contamina a contagem
        if m and cl and cl != 'PROCESSO' \
                and not re.match(r'^(classe\s*)?\d', cl.lower()):
            co[m.group(1)][cl] += 1
    return {n: c.most_common(1)[0][0] for n, c in co.items()
            if c and c.most_common(1)[0][1] >= 2}


def resolver_classe(v, teor):
    """Classe numerica ('6', 'Classe 6.ª') -> rotulo textual decidido pelo
    CABECALHO impresso da propria unidade (inicio do teor)."""
    m = re.match(r'^(?:classe\s*)?(\d{1,2})\s*[ª.\s]*$', (v or '').lower())
    if not m:
        up = v.upper().strip()
        return up, ('igual' if v == up else 'caixa')
    ini = (teor or '')[:120].lower()
    for padrao, rotulo in (('consulta', 'CONSULTA'),
                           ('recurso eleitoral', 'RECURSO ELEITORAL'),
                           ('recurso parcial', 'RECURSO PARCIAL'),
                           ('recurso', 'RECURSO'),
                           ('representa', 'REPRESENTAÇÃO'),
                           ('registro', 'REGISTRO'),
                           ('habeas', 'HABEAS CORPUS')):
        if padrao in ini:
            return rotulo, 'classe_num'
    return 'PROCESSO', 'classe_num'


# ---------------- luna ----------------

def luna_relator(teor, candidato=''):
    extra = (f' O campo atual diz "{candidato}" — confirme, corrija ou '
             f'negue pelo texto.' if candidato else '')
    data = llm.chat(
        'Você é um arquivista do Tribunal Superior de Justiça Eleitoral '
        '(1932-1937). Identifique o RELATOR no texto do acórdão. Ministros '
        f'da época: {MINISTROS_ABREV}.{extra} Responda SOMENTE JSON: '
        '{"relator": "<nome>"|null}. NÃO invente.',
        (teor or '')[:6000], json_mode=True, model='gpt-5.6-luna',
        max_output_tokens=200)
    if isinstance(data, str):
        data = json.loads(data)
    return data.get('relator')


def luna_uf(contexto, atual):
    data = llm.chat(
        'Você é um arquivista da Justiça Eleitoral brasileira (1932-1937). '
        'O campo UF deste registro está corrompido pelo OCR. Pelas '
        'informações, diga a UF correta (sigla de estado da época). '
        'Responda SOMENTE JSON: {"uf": "XX"|null}. Se o caso envolver '
        'genuinamente dois estados sem principal claro, null. NÃO invente.',
        f'UF corrompida: "{atual}"\n\n{contexto[:4000]}',
        json_mode=True, model='gpt-5.6-luna', max_output_tokens=100)
    if isinstance(data, str):
        data = json.loads(data)
    uf = (data.get('uf') or '').upper().strip()
    return uf if uf in UFS_1930 else None


# ---------------- plano ----------------

def montar_plano():
    regs = [json.loads(l) for l in io.open(DUMP, encoding='utf-8')
            if l.strip()]
    teores = {r['acordao_id']: r for r in t17.carregar_teor()}
    ac_por_page = {}
    with io.open(os.path.join(WORK, 'acordaos_inserted.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                ac_por_page[d['id']] = d['chave']
    # dump de processos p/ contexto de UF (ementa/decisao nao estao no dump;
    # usa extraidos)
    proc_ctx = {}
    with io.open(os.path.join(WORK, 'processos_inserted.jsonl'),
                 encoding='utf-8') as f:
        chave_por_page = {json.loads(l)['id']: json.loads(l)['chave']
                          for l in f if l.strip()}
    extr = {}
    with io.open(os.path.join(WORK, 'processos_extraidos.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for i, d in enumerate(r.get('deliberacoes', []), 1):
                extr[f"{r['ata_id']}#{i:02d}"] = d

    freq_nomes = Counter()
    for r in regs:
        for p in ('relator',):
            if r.get(p):
                freq_nomes[r[p]] += 1
        for p in ('quorum', 'votacao'):
            for v in r.get(p) or []:
                freq_nomes[v] += 1
    canonico = construir_canonicos(freq_nomes)
    print(f'canonicos: {len(canonico)}')

    plano = []
    n_luna_rel = n_luna_uf = 0
    for r in regs:
        base, pid = r['base'], r['id']
        # --- relator (select)
        if 'relator' in r and r.get('relator'):
            v = r['relator']
            novo, met = resolver_nome(v, canonico)
            if met == 'lixo' and not eh_lixo(limpar_nome(v)) \
                    and base != 'acordaos':
                # nome PLAUSIVEL raro sem canonico e sem teor para conferir
                # (relator de instancia, procurador): manter (no maximo a
                # limpeza de prefixos)
                lim = limpar_nome(v)
                novo, met = (lim, 'so_limpeza') if lim != v else (v, 'igual')
            if met == 'lixo':
                plausivel = not eh_lixo(limpar_nome(v))
                teor = ''
                if base == 'acordaos':
                    aid = ac_por_page.get(pid, '')
                    teor = (teores.get(aid) or {}).get('teor', '')
                if plausivel and teor:
                    # nome raro plausivel: NAO substituir as cegas pelo
                    # regex — o luna confirma pelo teor (mantem se null)
                    met = 'fila_luna_confirma'
                    n_luna_rel += 1
                elif not plausivel and teor:
                    m = re.search(
                        r'[Rr]elator[,:]?\s+(?:o\s+)?(?:[Ss]r\.?\s+)?'
                        r'(?:[Mm]inistro\s+|[Dd]esembargador\s+|'
                        r'[Dd]r\.?\s+)?'
                        r'([A-ZÀ-Ú][\wÀ-ú\'.]+(?:\s+(?:de|da|do|dos)?\s*'
                        r'[A-ZÀ-Ú][\wÀ-ú\'.]+){0,3})', teor)
                    if m:
                        cand, met2 = resolver_nome(m.group(1), canonico)
                        if cand:
                            novo, met = cand, 'regex_teor'
                    if not novo:
                        met = 'fila_luna_relator'
                        n_luna_rel += 1
                elif not novo:
                    met = 'esvaziar'
            if novo != v or met != 'igual':
                plano.append({'base': base, 'page_id': pid,
                              'prop': 'relator', 'de': v,
                              'para': novo or '', 'metodo': met})
        # --- UF
        if 'UF' in r and r.get('UF'):
            v = r['UF']
            novo, met = resolver_uf(v)
            if met == 'luna':
                n_luna_uf += 1
                plano.append({'base': base, 'page_id': pid, 'prop': 'UF',
                              'de': v, 'para': '', 'metodo': 'fila_luna_uf'})
            elif novo != v:
                plano.append({'base': base, 'page_id': pid, 'prop': 'UF',
                              'de': v, 'para': novo, 'metodo': met})
        # --- classe
        if 'classe' in r and r.get('classe'):
            v = r['classe']
            ctx = ''
            if base == 'acordaos':
                aid = ac_por_page.get(pid, '')
                ctx = (teores.get(aid) or {}).get('teor', '')
            else:
                d = extr.get(chave_por_page.get(pid, '')) or {}
                ctx = f"{d.get('classe', '')} {d.get('ementa', '')}"
            novo, met = resolver_classe(v, ctx)
            if novo and novo != v:
                plano.append({'base': base, 'page_id': pid, 'prop': 'classe',
                              'de': v, 'para': novo, 'metodo': met})
        # --- confianca (atas)
        if 'confianca' in r and r.get('confianca') == 'media':
            plano.append({'base': base, 'page_id': pid, 'prop': 'confianca',
                          'de': 'media', 'para': 'média', 'metodo': 'acento'})
        # --- quorum / votacao (multi)
        for p in ('quorum', 'votacao'):
            vals = r.get(p) or []
            if not vals:
                continue
            novos, mudou = [], False
            for v in vals:
                nv, met = resolver_nome(v, canonico)
                if nv is None:
                    mudou = True     # lixo sai da lista
                    continue
                if nv != v:
                    mudou = True
                if nv not in novos:
                    novos.append(nv)
            if mudou:
                plano.append({'base': base, 'page_id': pid, 'prop': p,
                              'de': '; '.join(vals),
                              'para': '; '.join(novos),
                              'metodo': 'multi'})

    with io.open(PLANO, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['base', 'page_id', 'prop', 'de',
                                          'para', 'metodo'])
        w.writeheader()
        w.writerows(plano)
    met = Counter(x['metodo'] for x in plano)
    print(f'plano: {len(plano)} mudancas -> {PLANO}')
    print(f'  por metodo: {dict(met)}')
    print(f'  filas luna: relator={n_luna_rel} uf={n_luna_uf}')
    return plano


# ---------------- apply ----------------

def aplicar(plano, limite=0):
    teores = {r['acordao_id']: r for r in t17.carregar_teor()}
    ac_por_page = {}
    with io.open(os.path.join(WORK, 'acordaos_inserted.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                ac_por_page[d['id']] = d['chave']
    extr = {}
    with io.open(os.path.join(WORK, 'processos_extraidos.jsonl'),
                 encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for i, d in enumerate(r.get('deliberacoes', []), 1):
                extr[f"{r['ata_id']}#{i:02d}"] = d
    with io.open(os.path.join(WORK, 'processos_inserted.jsonl'),
                 encoding='utf-8') as f:
        chave_por_page = {}
        for line in f:
            if line.strip():
                d = json.loads(line)
                chave_por_page[d['id']] = d['chave']

    feitos = set()
    if os.path.exists(DONE):
        with io.open(DONE, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    feitos.add(json.loads(line)['k'])
    pend = [x for x in plano
            if f"{x['page_id']}|{x['prop']}" not in feitos]
    if limite:
        pend = pend[:limite]
    print(f'aplicar: {len(pend)} (feitos: {len(feitos)})')
    cont = Counter()

    def do(x):
        k = f"{x['page_id']}|{x['prop']}"
        try:
            met, para = x['metodo'], x['para']
            if met in ('fila_luna_relator', 'fila_luna_confirma'):
                aid = ac_por_page.get(x['page_id'], '')
                teor = (teores.get(aid) or {}).get('teor', '')
                cand = (limpar_nome(x['de'])
                        if met == 'fila_luna_confirma' else '')
                para = luna_relator(teor, cand) if teor else None
                if not para:
                    # confirma: sem resposta segura, MANTEM o nome atual
                    para = cand if met == 'fila_luna_confirma' else ''
            elif met == 'fila_luna_uf':
                chave = chave_por_page.get(x['page_id'], '')
                d = extr.get(chave) or {}
                ctx = (f"classe: {d.get('classe')}\n"
                       f"ementa: {d.get('ementa')}\n"
                       f"decisao: {d.get('decisao')}")
                if x['base'] == 'acordaos':
                    aid = ac_por_page.get(x['page_id'], '')
                    ctx = (teores.get(aid) or {}).get('teor', '')[:4000]
                para = luna_uf(ctx, x['de']) or ''
            if x['prop'] in ('quorum', 'votacao'):
                vals = [v for v in para.split('; ') if v]
                props = {x['prop']: {'multi_select':
                                     [{'name': v} for v in vals]}}
            elif para:
                props = {x['prop']: {'select': {'name': para[:95]}}}
            else:
                props = {x['prop']: {'select': None}}
            ln._req('PATCH', f"/pages/{x['page_id']}",
                    {'properties': props})
            with _lock:
                with io.open(DONE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'k': k, 'para': para}) + '\n')
                cont['ok'] += 1
                if cont['ok'] % 200 == 0:
                    print(f"  ... {cont['ok']}")
        except Exception as e:
            with _lock:
                cont['err'] += 1
                if cont['err'] <= 10:
                    print(f"  ERRO {k}: {str(e)[:90]}")

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(do, pend))
    print(f'[FIM apply] {dict(cont)}')


def limpar_schema():
    """Remove do schema as options que nao estao mais em uso."""
    for db_key, props in PROPS_POR_BASE.items():
        regs = dump_base(db_key, props)
        db_id = IDS[f'db_{db_key}']
        atual = ln._req('GET', f'/databases/{db_id}')
        for p in props:
            usados = set()
            for r in regs:
                v = r.get(p)
                if isinstance(v, list):
                    usados |= set(v)
                elif v:
                    usados.add(v)
            spec = (atual.get('properties') or {}).get(p) or {}
            tipo = spec.get('type')
            if tipo not in ('select', 'multi_select'):
                continue
            opts = [o for o in spec[tipo]['options']
                    if o['name'] in usados]
            if len(opts) < len(spec[tipo]['options']):
                ln._req('PATCH', f'/databases/{db_id}', {'properties': {
                    p: {tipo: {'options':
                               [{'name': o['name'], 'color': o['color']}
                                for o in opts]}}}})
                print(f'  {db_key}.{p}: {len(spec[tipo]["options"])} -> '
                      f'{len(opts)} options')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--limite', type=int, default=0)
    ap.add_argument('--so-schema', action='store_true')
    args = ap.parse_args()

    if args.so_schema:
        limpar_schema()
        return
    if not os.path.exists(DUMP) or not args.apply:
        print('dump das 3 bases...')
        fazer_dump()
    plano = montar_plano()
    if not args.apply:
        print('(dry-run; use --apply)')
        return
    aplicar(plano, args.limite)
    print('limpeza do schema...')
    limpar_schema()


if __name__ == '__main__':
    main()
