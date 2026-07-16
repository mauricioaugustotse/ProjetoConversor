# -*- coding: utf-8 -*-
r"""acv_06: reclassifica os moradores de "Outros"/"Diversos" do HD_Mau — SEM OpenAI.

Camadas de decisao (plano aprovado 13/07/2026):
 0. lixo digital remanescente -> Lixeira (padroes ampliados; politica aprovada);
 1. resposta LUNA ja paga (cache jsonl) com destino util (nao-Outros) e conf>=0.5;
 2. regras deterministicas novas (CNJ, guias, trabalho TSE, musicas, idiomas,
    binarios de sistema);
 3. lotes exportados p/ classificacao do PROPRIO CLAUDE na sessao
    (acv_claude_lote_NN.csv -> acv_claude_resp_NN.csv);
 4. residuo -> pastas por TIPO de arquivo em 05 - Diversos.

Uso: python acv_06_moradores.py --plano       (camadas 0/1/2 + exporta lotes p/ Claude)
     python acv_06_moradores.py --incorporar  (le acv_claude_resp_*.csv + camada 4)
     python acv_06_moradores.py --exec        (aplica acv_plano_moradores.csv)
"""
import csv
import datetime
import glob
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from acv_02_cruzar import lp
from acv_03_organizar import (MATERIA_J, RE_CNJ, RE_CTX_PROC, TIPOS_JUR,
                              ano_do, ascii_seg, endereco_fino, whitelist)
from acv_04_reordenar_hdmau import valida_stem

ROOT = r'C:\Users\mauri\HD_Mau'
PLANO = os.path.join(BASE, 'acv_plano_moradores.csv')
LOTES_DIR = os.path.join(BASE, 'acv_claude_lotes')
TAM_LOTE = 350
GATE_LUNA = 0.5

NOVAS = [
    r'03 - Administrativo\TSE - gabinete e sessoes',
    r'03 - Administrativo\Cartorio de protesto',
    r'03 - Administrativo\Manuais ECT',
    r'03 - Administrativo\Relatorios operacionais',
    r'04 - Pessoal e financeiro\Financas domesticas',
    r'02 - Estudos e concursos\Instrumentais\Idiomas',
    r'05 - Diversos\Audios e musicas',
    r'05 - Diversos\Sistema e programas',
    r'05 - Diversos\Planilhas',
    r'05 - Diversos\Apresentacoes',
    r'05 - Diversos\Textos avulsos',
    r'05 - Diversos\Documentos nao identificados',
    r'05 - Diversos\Web',
]
# sub-tipos dentro de 01\Outros processuais\<materia> (criados sob demanda)
SUBTIPOS_OP = [
    (re.compile(r'calculo|liquidacao|planilha de? ?calc', re.I), 'Calculos e liquidacoes'),
    (re.compile(r'certid[aã]o|intima[cç]', re.I), 'Certidoes e intimacoes'),
    (re.compile(r'ata de audiencia|\bata audiencia', re.I), 'Atas de audiencia'),
    (re.compile(r'execu[cç][aã]o|embargos (a|de)? ?execu|beneficio (de )?ordem|'
                r'penhora|edital de pra[cç]a', re.I), 'Execucao'),
]

RE_LIXO = re.compile(
    r'^(Cookies? [\w.\-\[]|Aviso de (compatibilidade|incompatibilidade|atualizacao)|'
    r'licenca de usuario HP SureStore|amzn1[ .\-]drm|.*arquivo temporario)', re.I)
RE_TSE_TRAB = re.compile(
    r'pauta de julgamento|pauta jurisdicional|jurisdicional[ \-]+sessao|'
    r'lista de processos|pauta.{0,12}sessao|lista triplice|'
    r'\bGMAJ\b|\bJurisdicional \d|sessao (ordinaria|extraordinaria|virtual|plenaria)|'
    r'estudo - sessao|retirada de autos', re.I)
RE_GUIA = re.compile(r'\bguia de recolhimento|\bGRCTJ\b|\bGRU\b|\bDARF\b|\bGPS\b '
                     r'|guia previdencia', re.I)
RE_CARTORIO = re.compile(
    r'protesto|tabeli[aã]o|tabelionato|IEPTB|\bCDA\b|\bDAE\b|registro.{0,4}log|'
    r'log.{0,4}(cartorio|envio|registro)|registro de? ?protocolo|edital de pra[cç]a',
    re.I)
RE_MANUAL_ECT = re.compile(r'\bman(lic|pat|pes|cod|afi|jur)\b|manual de licitacoes',
                           re.I)
RE_REL_OPER = re.compile(r'relatori?o.{0,4}(diari|semanal|mensal|atividades|analise|'
                         r'captur|sadits|visita)|rel\.? ?vigep|vigep|\bSADITS\b',
                         re.I)
RE_FIN_DOM = re.compile(r'planilha.{0,10}gastos|balanco mensal|contas mensais|'
                        r'resumo.{0,6}contas|demonstrativo.{0,6}contas|'
                        r'tabela.{0,6}gastos|contas.{0,3}(ana|matheus|mario)|'
                        r'planilha.{0,6}mario', re.I)
RE_NOTA_PUB = re.compile(r'^nota (publica|tecnica)|^solicitacao de? ?servico', re.I)
RE_RETORNO_BANCO = re.compile(r'^retorno (brades|bb|banco|prot)', re.I)
RE_CONTRATO = re.compile(r'prorroga[cç][aã]o.{0,4}contrat|^emv.{0,4}loca[cç]|'
                         r'contrato de loca[cç]', re.I)
RE_DOC_PESSOAL = re.compile(r'avalia[cç][aã]o de? ?desempenho|encontro de? ?turma|'
                            r'foto.{0,4}formatura', re.I)
RE_FICHA_FIN = re.compile(r'ficha financeira', re.I)
RE_INFO_CSV = re.compile(r'^info ?\(\d+\)$', re.I)
RE_AULA_SOLTA = re.compile(r'^(_?aula|capitulo) ?\d', re.I)
RE_QUALIFICACAO = re.compile(r'qualifica[cç][aã]o.{0,4}anexo', re.I)
RE_PREPOSTO = re.compile(r'orientac[aã]o.{0,4}preposto', re.I)
RE_CLASSE_INICIO = re.compile(
    r'^(AgR|A?REspe?|AI|RO|RMS|EDcl|ED|Rcl|MS|HC)\b[ \-]', re.I)
RE_IDIOMA = re.compile(r'\bunita \d|\bitaliano\b|\bingles\b|\bespanhol\b|\bfrances\b|'
                       r'\btoefl\b|\bielts\b|grammar|vocabulary', re.I)
RE_MUSICA_NOME = re.compile(r'^\d{1,2}[ _.\-]+\w|faixa \d|\btrilha\b|track \d', re.I)
EXT_AUDIO = {'.wma', '.mp3', '.wav', '.m4a', '.ogg'}
EXT_SISTEMA = {'.dll', '.exe', '.aui', '.fll', '.ini', '.inf', '.sys', '.cab',
               '.jar', '.bat', '.reg', '.lnk', '.url', '.hlp', '.chm'}
TIPO_POR_EXT = [
    ({'.xls', '.xlsx', '.csv', '.ods'}, r'05 - Diversos\Planilhas'),
    ({'.ppt', '.pptx', '.pps', '.ppsx'}, r'05 - Diversos\Apresentacoes'),
    ({'.txt', '.log'}, r'05 - Diversos\Textos avulsos'),
    ({'.html', '.htm', '.css', '.mht', '.mhtml', '.xml'}, r'05 - Diversos\Web'),
    (EXT_AUDIO, r'05 - Diversos\Audios e musicas'),
    (EXT_SISTEMA, r'05 - Diversos\Sistema e programas'),
]
FALLBACK_TIPO = r'05 - Diversos\Documentos nao identificados'


def eh_morador(rel_dir):
    partes = rel_dir.split('\\')
    return any(p in ('Outros', 'Outros processuais') for p in partes) \
        or partes[0] == '05 - Diversos'


def coletar():
    out = []
    for dp, dn, fn in os.walk(lp(ROOT)):
        base = dp[4:] if dp.startswith('\\\\?\\') else dp
        rel = os.path.relpath(base, ROOT)
        if rel == '.' or not eh_morador(rel):
            continue
        for f in fn:
            if not f.startswith('00 - '):
                out.append(os.path.join(base, f))
    return out


def mapa_logs():
    """path_antigo(lower) -> path_atual, encadeando todos os logs de movimento."""
    mapa = {}
    padroes = ('log_acv_integra_*.csv', 'log_acv_reord_*.csv',
               'log_acv_reeq_*.csv', 'log_acv_encurta_*.csv')
    for pat in padroes:
        for lg in sorted(glob.glob(os.path.join(BASE, pat))):
            for r in csv.DictReader(open(lg, encoding='utf-8-sig')):
                de = r.get('origem') or r.get('de')
                para = r.get('destino') or r.get('para')
                st = r.get('status', 'ok')
                if de and para and not para.startswith('ERRO') \
                        and (st == 'ok' or 'status' not in r):
                    mapa[de.lower()] = para

    def resolve(antigo):
        p = antigo
        for _ in range(8):
            nxt = mapa.get(p.lower())
            if not nxt:
                return p
            p = nxt
    return mapa, resolve


def indexar_ativos(moradores):
    """Por path ATUAL: cabecalho de texto + resposta luna (dos caches pagos)."""
    mapa, resolve = mapa_logs()
    atual_de = {}
    for antigo in mapa:
        atual_de.setdefault(resolve(antigo).lower(), antigo)

    alvo = {m.lower() for m in moradores}
    cab = {}
    for tsv in ('org2_cabecalhos.tsv', 'acv_cabecalhos_c_extra.tsv',
                'acv_cabecalhos_d.tsv'):
        fp = os.path.join(BASE, tsv)
        if not os.path.exists(fp):
            continue
        for line in io.open(fp, encoding='utf-8', errors='replace'):
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            antigo = parts[0]
            texto = parts[-1]
            atual = resolve(antigo).lower()
            if atual in alvo and len(texto) > len(cab.get(atual, '')):
                cab[atual] = texto
    luna = {}
    for jl in ('acv_ia_reord.jsonl', 'acv_ia_integra.jsonl'):
        fp = os.path.join(BASE, jl)
        if not os.path.exists(fp):
            continue
        for line in io.open(fp, encoding='utf-8'):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            atual = resolve(d.get('path', '')).lower()
            if atual in alvo:
                luna[atual] = d
    return cab, luna


# tipos com regex ESTENDIDO (local; nao mexe no TIPOS_JUR global)
TIPOS_JUR_EXT = TIPOS_JUR + [
    (re.compile(r'\bdecis[aã]o\b|\bsenten[cç]a\b|monocratic', re.I), 'Acordaos e decisoes'),
    (re.compile(r'\bdefesa\b|\bacordo\b|termo de reconhecimento|carta de anuencia|'
                r'contrarraz|contranotifica', re.I), 'Contestacoes e manifestacoes'),
    (re.compile(r'\bmemoria(l|is)\b', re.I), 'Peticoes'),
]
RE_TRIB_TRAB = re.compile(r'vara do trabalho|juiz(a)? do trabalho|reclamante|'
                          r'reclamada|\bECT\b.*(trabalh|reclamat)', re.I)


def regra_mae_confirma(rel_dir, nome, cabecalho):
    """Morador de 01\\<tipo>\\...\\Outros cujo nome/cabecalho confirma o TIPO da
    propria linhagem -> volta p/ a pasta tipo\\materia (sai do balde)."""
    partes = rel_dir.split('\\')
    if partes[0] != '01 - Juridico' or len(partes) < 3 or 'Outros' not in partes[2:]:
        return None
    tipo = partes[1]
    materia = partes[2] if partes[2] != 'Outros' else None
    alvo = f'{nome} || {cabecalho[:200]}'
    rx = [r for r, t in TIPOS_JUR_EXT if t == tipo]
    if not any(r.search(alvo) for r in rx):
        return None
    if not materia:
        materia = 'Trabalhista' if RE_TRIB_TRAB.search(cabecalho[:300]) else None
        if not materia:
            m = RE_CNJ.search(f'{nome} {cabecalho[:300]}')
            materia = MATERIA_J.get(m.group(4)) if m else None
        if not materia or materia == 'Outros':
            return None
    return f'01 - Juridico\\{tipo}\\{materia}'


def materia_linhagem(rel_dir):
    MATERIAS = {'Administrativo', 'Ambiental', 'Civel', 'Constitucional',
                'Consumidor', 'Eleitoral', 'Empresarial', 'Familia e Sucessoes',
                'Penal', 'Previdenciario', 'Trabalhista', 'Tributario'}
    return next((p for p in rel_dir.split('\\') if p in MATERIAS), None)


def regra_camada2(rel_dir, nome, cabecalho, ext):
    alvo = f'{nome} || {cabecalho[:300]}'
    if RE_TSE_TRAB.search(alvo):
        return r'03 - Administrativo\TSE - gabinete e sessoes'
    if RE_CARTORIO.search(nome):
        return r'03 - Administrativo\Cartorio de protesto'
    if RE_MANUAL_ECT.search(nome):
        return r'03 - Administrativo\Manuais ECT'
    if RE_REL_OPER.search(nome):
        return r'03 - Administrativo\Relatorios operacionais'
    if RE_FIN_DOM.search(nome):
        return r'04 - Pessoal e financeiro\Financas domesticas'
    if RE_QUALIFICACAO.search(nome):
        return r'01 - Juridico\Outros processuais\Penal'
    if RE_PREPOSTO.search(nome):
        return r'01 - Juridico\Modelos e minutas\Trabalhista'
    if RE_RETORNO_BANCO.search(nome):
        return r'03 - Administrativo\Cartorio de protesto'
    if RE_NOTA_PUB.search(nome):
        return r'03 - Administrativo\Oficios e memorandos'
    if RE_CONTRATO.search(nome):
        return r'03 - Administrativo\Contratos e termos'
    if RE_FICHA_FIN.search(nome):
        return r'04 - Pessoal e financeiro\Documentos pessoais\Fichas cadastrais e financeiras'
    if RE_DOC_PESSOAL.search(nome):
        return r'04 - Pessoal e financeiro\Documentos pessoais'
    stem_l = os.path.splitext(nome)[0]
    if RE_INFO_CSV.match(stem_l):
        return r'05 - Diversos\Sistema e programas'
    if RE_AULA_SOLTA.match(nome):
        return r'02 - Estudos e concursos\Cursos e aulas'
    if RE_GUIA.search(nome):
        return r'04 - Pessoal e financeiro\Faturas e boletos'
    if RE_IDIOMA.search(nome):
        return r'02 - Estudos e concursos\Instrumentais\Idiomas'
    if ext in EXT_AUDIO:
        return r'05 - Diversos\Audios e musicas'
    if ext in EXT_SISTEMA:
        return r'05 - Diversos\Sistema e programas'
    # sub-tipos processuais: qualquer morador do 01 com materia na linhagem
    if rel_dir.startswith('01 - Juridico'):
        mat = materia_linhagem(rel_dir)
        if mat:
            for rx, sub in SUBTIPOS_OP:
                if rx.search(nome):
                    return rf'01 - Juridico\Outros processuais\{mat}\{sub}'
    # prefixo de classe recursal no inicio do nome (morador do 01)
    if rel_dir.startswith('01 - ') and RE_CLASSE_INICIO.match(nome):
        mat = materia_linhagem(rel_dir) or 'Eleitoral'
        return rf'01 - Juridico\Recursos\{mat}'
    m = RE_CNJ.search(nome)
    if not m:
        m = RE_CNJ.search(cabecalho[:400])
        if m and not RE_CTX_PROC.search(cabecalho):
            m = None
    if m:
        materia = MATERIA_J.get(m.group(4), 'Outros')
        if materia != 'Outros':  # p/ morador de Outros, so vale se sai do balde
            tipo = next((t for rx, t in TIPOS_JUR if rx.search(nome)), None)
            if tipo:
                return f'01 - Juridico\\{tipo}\\{materia}'
    return None


def dests_validos():
    dests = set(whitelist()) | set(NOVAS)
    MATERIAS = ['Administrativo', 'Civel', 'Eleitoral', 'Penal', 'Previdenciario',
                'Trabalhista', 'Tributario']
    for mat in MATERIAS:
        for _, sub in SUBTIPOS_OP:
            dests.add(rf'01 - Juridico\Outros processuais\{mat}\{sub}')
    return dests


def montar_plano():
    moradores = coletar()
    print(f'Moradores: {len(moradores):,}')
    cab, luna = indexar_ativos(moradores)
    print(f'Ativos mapeados: cabecalho {len(cab):,} | luna {len(luna):,}')
    dests = dests_validos()

    decisao = {}   # path -> (destino_rel|LIXEIRA, fonte, novo_stem)
    stats = Counter()
    residuo = []
    for p in moradores:
        rel_dir = os.path.dirname(os.path.relpath(p, ROOT))
        nome = os.path.basename(p)
        stem, ext = os.path.splitext(nome)
        ext = ext.lower()
        texto = cab.get(p.lower(), '')
        # camada 0: lixo
        if RE_LIXO.match(nome):
            decisao[p] = ('LIXEIRA', 'lixo', '')
            stats['0_lixeira'] += 1
            continue
        # camada 1: luna pago
        d = luna.get(p.lower())
        if d:
            dst = d.get('destino', 'manter')
            util = (dst not in ('', 'manter') and dst in dests
                    and 'Outros' not in dst and d.get('confianca', 0) >= GATE_LUNA
                    and not (rel_dir == dst or rel_dir.startswith(dst + '\\')))
            if util:
                ns = valida_stem(d.get('novo_stem', '').strip(), stem)
                decisao[p] = (dst, 'luna_cache', ns)
                stats['1_luna'] += 1
                continue
        # camada 2: regras (inclui "mae confirma tipo" p/ baldes do 01)
        dst = regra_mae_confirma(rel_dir, nome, texto)
        if dst and rel_dir != dst:
            decisao[p] = (dst, 'regra_mae', '')
            stats['2_mae'] += 1
            continue
        dst = regra_camada2(rel_dir, nome, texto, ext)
        if dst and not (rel_dir == dst or rel_dir.startswith(dst + '\\')):
            decisao[p] = (dst, 'regra', '')
            stats['2_regra'] += 1
            continue
        residuo.append(p)
    print(f"Camadas 0-2: {dict(stats)} | residuo p/ Claude: {len(residuo):,}")

    # exporta lotes p/ camada 3 (Claude na sessao)
    os.makedirs(LOTES_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(LOTES_DIR, 'acv_claude_lote_*.csv')):
        os.remove(f)
    for i in range(0, len(residuo), TAM_LOTE):
        n = i // TAM_LOTE + 1
        with open(os.path.join(LOTES_DIR, f'acv_claude_lote_{n:02d}.csv'),
                  'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['path', 'pasta', 'nome', 'cabecalho'])
            for p in residuo[i:i + TAM_LOTE]:
                rel_dir = os.path.dirname(os.path.relpath(p, ROOT))
                trecho = re.sub(r'\s+', ' ', cab.get(p.lower(), ''))[:150]
                w.writerow([p, rel_dir, os.path.basename(p), trecho])
    print(f'Lotes exportados: {(len(residuo)+TAM_LOTE-1)//TAM_LOTE} em {LOTES_DIR}')

    _gravar_plano(decisao)
    return decisao


def incorporar():
    """Le acv_claude_resp_*.csv, valida e completa com a camada 4 (tipos)."""
    decisao = {}
    for r in csv.DictReader(open(PLANO, encoding='utf-8-sig')):
        decisao[r['origem_atual']] = (r['destino_rel'], r['fonte'], r['novo_stem'])
    dests = dests_validos()
    stats = Counter()
    # indice id->path por lote (respostas podem vir por 'id' em vez de 'path')
    paths_lote = {}
    for fp in sorted(glob.glob(os.path.join(LOTES_DIR, 'acv_claude_lote_*.csv'))):
        n = re.search(r'lote_(\d+)', fp).group(1)
        paths_lote[n] = [r['path'] for r in
                         csv.DictReader(open(fp, encoding='utf-8-sig'))]
    respondidos = set()
    for fp in sorted(glob.glob(os.path.join(LOTES_DIR, 'acv_claude_resp_*.csv'))):
        n = re.search(r'resp_(\d+)', fp).group(1)
        for r in csv.DictReader(open(fp, encoding='utf-8-sig')):
            if r.get('id', '').strip():
                try:
                    p = paths_lote[n][int(r['id'])]
                except (KeyError, IndexError, ValueError):
                    stats['3_id_invalido'] += 1
                    continue
            else:
                p = r.get('path', '')
            dst = r.get('destino', '').strip()
            respondidos.add(p)
            if dst == 'MANTER':
                stats['3_manter'] += 1
                continue
            if dst == 'LIXEIRA':
                decisao[p] = ('LIXEIRA', 'claude', '')
                stats['3_claude_lixeira'] += 1
                continue
            if dst not in dests:
                stats['3_resp_invalida'] += 1
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            ns = valida_stem(r.get('novo_stem', '').strip(), stem)
            decisao[p] = (dst, 'claude', ns)
            stats['3_claude'] += 1
    # residuo sem resposta -> camada 4 (tipo de arquivo)
    for fp in sorted(glob.glob(os.path.join(LOTES_DIR, 'acv_claude_lote_*.csv'))):
        for r in csv.DictReader(open(fp, encoding='utf-8-sig')):
            p = r['path']
            if p in respondidos or p in decisao:
                continue
            ext = os.path.splitext(p)[1].lower()
            dst = next((d for exts, d in TIPO_POR_EXT if ext in exts), FALLBACK_TIPO)
            decisao[p] = (dst, 'tipo_arquivo', '')
            stats['4_tipo'] += 1
    print(f'Incorporacao: {dict(stats)}')
    _gravar_plano(decisao)


def _gravar_plano(decisao):
    with open(PLANO, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['origem_atual', 'destino_rel', 'fonte', 'novo_stem'])
        for p, (dst, fonte, ns) in sorted(decisao.items()):
            w.writerow([p, dst, fonte, ns])
    tops = Counter(d[0] if d[0] == 'LIXEIRA' else '\\'.join(d[0].split('\\')[:2])
                   for d in decisao.values())
    print(f'Plano: {len(decisao):,} decisões -> {PLANO}')
    for k, v in tops.most_common(18):
        print(f'  {v:>6,}  {k}')


def executar():
    from send2trash import send2trash
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_moradores_{ts}.csv')
    usados = set()
    ok = err = lixo = 0
    rows = list(csv.DictReader(open(PLANO, encoding='utf-8-sig')))
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['de', 'para', 'fonte', 'status'])
        for i, r in enumerate(rows, 1):
            p, dst, fonte, ns = (r['origem_atual'], r['destino_rel'],
                                 r['fonte'], r['novo_stem'])
            status = 'ok'
            try:
                if not os.path.exists(lp(p)):
                    status = 'origem_sumiu'
                elif dst == 'LIXEIRA':
                    send2trash(p)
                    lixo += 1
                else:
                    stem, ext = os.path.splitext(os.path.basename(p))
                    novo_stem = ns or ascii_seg(stem, 110)
                    alvo_rel = endereco_fino(dst, novo_stem, ano_do(stem, ''))
                    pasta = os.path.join(ROOT, alvo_rel)
                    maxstem = max(20, 255 - len(pasta) - len(ext) - 6)
                    novo_stem = novo_stem[:maxstem].rstrip(' .-')
                    destino = os.path.join(pasta, novo_stem + ext)
                    n = 1
                    while destino.lower() in usados or os.path.exists(lp(destino)):
                        n += 1
                        destino = os.path.join(pasta, f'{novo_stem} ({n}){ext}')
                    usados.add(destino.lower())
                    os.makedirs(lp(pasta), exist_ok=True)
                    os.rename(lp(p), lp(destino))
                    log.writerow([p, destino, fonte, 'ok'])
            except OSError as e:
                status = f'erro: {e}'
            if status != 'ok':
                log.writerow([p, dst, fonte, status])
                err += 1
            else:
                ok += 1
            if i % 1000 == 0:
                print(f'  {i:,}/{len(rows):,} (problemas {err})', flush=True)
    # pastas esvaziadas
    vazias = 0
    for dp, dn, fn in os.walk(lp(ROOT), topdown=False):
        if not os.listdir(dp):
            try:
                os.rmdir(dp)
                vazias += 1
            except OSError:
                pass
    print(f'Aplicados: {ok:,} (Lixeira {lixo:,}) | problemas: {err:,} | '
          f'vazias removidas: {vazias} | log: {log_path}')


if __name__ == '__main__':
    if '--exec' in sys.argv:
        executar()
    elif '--incorporar' in sys.argv:
        incorporar()
    else:
        montar_plano()
