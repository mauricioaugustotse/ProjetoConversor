# -*- coding: utf-8 -*-
r"""acv_09: biblioteca digital em C:\Users\mauri\HD_Mau — objeto > assunto > data.

Diretriz do usuario (13/07/2026): fim das faixas alfabeticas; boletos todos
numa pasta, recibos noutra, livros separados de apostilas etc. Hierarquia:
objeto organizado > assunto > data. Anexos processuais ficam no 01 (decisao);
no 04, tema/credor vira subpasta so quando ha massa (>= MIN_TEMA).

Escopos e regras:
 - faixas alfabeticas (RE_FAIXA_ALFA) em toda a arvore: dissolvidas na mae;
 - 02\Doutrina e artigos\<disc>: paginas de PDF (pypdf) / tamanho separam
   Livros (>=100 pags ou >=15 MB) | Apostilas (kw) | Artigos e textos;
 - 02\Concursos e provas: Editais de concurso | Provas e simulados |
   Apostilas\Concursos; resto fica plano;
 - 03\TSE - gabinete e sessoes: Pautas e sessoes | Estudos e roteiros |
   Planejamento e relatorios; resto raiz;
 - 04: objetos por keyword (Boletos, Faturas, Recibos, Comprovantes de
   pagamento, Extratos bancarios, Notas fiscais, Contracheques e fichas,
   Guias e impostos, Certidoes, Declaracoes, Curriculos, Saude e exames,
   Viagens, Planilhas de gastos); tema (Claro, Unimed, ECT...) quando massa;
 - residuos do 04 sem keyword: lotes p/ classificacao do Claude na sessao
   (acv9_claude_lote_NN.csv -> acv9_claude_resp_NN.csv, keyword MANTER).
Depois do --exec: acv_05 (agora so faixas de ANO) refatia o que passar de 400.

Uso: --plano | --incorporar | --exec
"""
import csv
import datetime
import os
import re
import sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from acv_02_cruzar import lp
from acv_03_organizar import RE_FAIXA_ALFA, RE_FAIXA_ANO, ano_do

ROOT = r'C:\Users\mauri\HD_Mau'
PLANO = os.path.join(BASE, 'acv_plano_biblio.csv')
DIR_LOTES = os.path.join(BASE, 'acv_claude_lotes')
TAM_LOTE = 400
MIN_TEMA = 60

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

C02 = '02 - Estudos e concursos'
C03 = '03 - Administrativo'
C04 = '04 - Pessoal e financeiro'

# ---- objetos do 04 (ordem = precedencia) ----
OBJ04 = [
    ('Contracheques e fichas financeiras',
     r'contracheque|holerite|ficha financeira|demonstrativo de pagamento|folha de pagamento'),
    ('Guias e impostos',
     r'\bgru\b|\bgrctj\b|\bdarf\b|\bgps\b|\birpf\b|imposto de renda|\bdirpf\b|'
     r'\biptu\b|\bipva\b|\bitcd\b|carne[ -]?leao|receita federal|restituicao|\bdae\b'),
    ('Notas fiscais', r'\bnfe?\b|nota fiscal|\bnfse?\b|\bdanfe\b|cupom fiscal'),
    ('Boletos', r'\bboleto'),
    ('Faturas', r'\bfatura'),
    ('Extratos bancarios', r'\bextrato'),
    ('Recibos', r'\brecibo'),
    ('Comprovantes de pagamento',
     r'\bcomprovante|\bpix\b|\bted\b|transferencia bancaria|pagamento efetuado'),
    ('Certidoes', r'\bcertidao'),
    ('Declaracoes', r'\bdeclaracao'),
    ('Curriculos', r'\bcurriculo|\blattes\b'),
    ('Saude e exames',
     r'\bexames?\b|\blaudo\b|atestado|hemograma|receituario|\bvacina|'
     r'ressonancia|tomografia|ultrassom|cirurgia|colonoscopia|endoscopia'),
    ('Viagens',
     r'\bvoucher|\bpassagem|\breserva\b|e-?ticket|boarding|check-?in|'
     r'itinerar|\bhotel\b|hospedagem|localizador'),
]
OBJ04_RX = [(n, re.compile(rx, re.I)) for n, rx in OBJ04]
EXT_PLANILHA = {'.xls', '.xlsx', '.xlsm', '.csv', '.ods'}

TEMAS = [
    ('Claro', r'\bclaro\b'), ('Oi', r'\boi\b'), ('Vivo', r'\bvivo\b'),
    ('TIM', r'\btim\b'), ('NET', r'\bnet\b'), ('CEMIG', r'\bcemig\b'),
    ('COPASA', r'\bcopasa\b'), ('Unimed', r'\bunimed\b'),
    ('Condominio', r'\bcondominio|\bsindic'),
    ('ECT Correios', r'\bect\b|\bcorreios\b'), ('Conectcar', r'\bconectcar\b'),
    ('Caixa', r'\bcaixa\b|\bcef\b|\bfgts\b'),
    ('Banco do Brasil', r'banco do brasil|\bbb\b|ourocard'),
    ('Cartao de credito', r'\bcartao\b|\bvisa\b|\bmastercard|\bnanquim|\binfinite\b'),
]
TEMAS_RX = [(n, re.compile(rx, re.I)) for n, rx in TEMAS]

KW_APOSTILA = re.compile(r'\bapostila|caderno de (questoes|exercicios)', re.I)
KW_LIVRO = re.compile(r'\blivro\b|\be-?book\b', re.I)
KW_PROVA = re.compile(r'\bprovas?\b|\bsimulado|\bgabarito|\bquestoes?\b|'
                      r'discursiva|recurso administrativo', re.I)
KW_EDITAL = re.compile(r'^edital\b|\bedital (de|n)\b|\beditais\b', re.I)
KW_PAUTA = re.compile(r'\bpauta|\bespelho|\bsessao\b|\bsessoes\b|lista de julgamento', re.I)
KW_ESTUDO = re.compile(r'\bestudos?\b|\broteiro|\banotac|resumo do processo', re.I)
KW_PLANEJ = re.compile(r'plano de trabalho|planejamento|\brelatorio|\bbalanc|\bgestao\b', re.I)

# pastas do 04 que NAO sao dissolvidas (nucleo mantido; so keywords fortes saem)
NUCLEO_04 = {'Documentos pessoais', 'Declaracoes'}
# objetos que podem ser extraidos do nucleo Documentos pessoais
PUXAR_DO_NUCLEO = {'Curriculos', 'Saude e exames', 'Viagens', 'Certidoes',
                   'Contracheques e fichas financeiras'}


def paginas_pdf(path):
    if not PdfReader:
        return None
    try:
        with open(lp(path), 'rb') as f:
            return len(PdfReader(f, strict=False).pages)
    except Exception:
        return None


def classificar_04(nome, ext, pasta_atual):
    for obj, rx in OBJ04_RX:
        if rx.search(nome):
            return obj
    if ext in EXT_PLANILHA and 'Financas domesticas' in pasta_atual:
        return 'Planilhas de gastos'
    return None


def tema_de(nome):
    for t, rx in TEMAS_RX:
        if rx.search(nome):
            return t
    return None


def montar_plano():
    decisoes = []   # (path, destino_dir, camada)
    residuos = []   # (path,) p/ lotes Claude (so 04)
    stats = Counter()

    # ---- 1. faixas alfabeticas em toda a arvore -> conteudo sobe p/ mae ----
    faixas = []
    for dp, dn, fn in os.walk(lp(ROOT)):
        base = dp[4:] if dp.startswith('\\\\?\\') else dp
        nome = os.path.basename(base)
        if RE_FAIXA_ALFA.match(nome) and not RE_FAIXA_ANO.match(nome):
            faixas.append(base)
    print(f'Pastas-faixa alfabeticas a dissolver: {len(faixas)}')
    origem_efetiva = {}  # path -> pasta logica (mae da faixa)
    alvos = []           # (path, pasta_logica_relativa)
    for fx in faixas:
        mae = os.path.dirname(fx)
        for f in os.listdir(lp(fx)):
            p = os.path.join(fx, f)
            if os.path.isfile(lp(p)):
                alvos.append((p, os.path.relpath(mae, ROOT)))

    # ---- 2. demais escopos (arquivos fora de faixa) ----
    escopos = [
        os.path.join(ROOT, C02, 'Doutrina e artigos'),
        os.path.join(ROOT, C02, 'Concursos e provas'),
        os.path.join(ROOT, C03, 'TSE - gabinete e sessoes'),
        os.path.join(ROOT, C04),
    ]
    for esc in escopos:
        for dp, dn, fn in os.walk(lp(esc)):
            base = dp[4:] if dp.startswith('\\\\?\\') else dp
            if RE_FAIXA_ALFA.match(os.path.basename(base)) \
                    and not RE_FAIXA_ANO.match(os.path.basename(base)):
                continue  # ja coletada na etapa 1
            for f in fn:
                if f.startswith('00 - '):
                    continue
                alvos.append((os.path.join(base, f), os.path.relpath(base, ROOT)))

    # ---- 3. regra por escopo ----
    contagem_tema = defaultdict(Counter)  # obj -> tema -> n (1o passe p/ MIN_TEMA)
    pre = []  # (path, obj_ou_None, contexto)
    for path, rel in alvos:
        nome = os.path.basename(path)
        stem, ext = os.path.splitext(nome)
        ext = ext.lower()
        partes = rel.split(os.sep)
        topo = partes[0]

        if topo == C04:
            sub = partes[1] if len(partes) > 1 else ''
            obj = classificar_04(stem, ext, rel)
            if sub == 'Declaracoes' and obj == 'Declaracoes':
                pre.append((path, 'MANTER', None))
            elif obj and (sub not in NUCLEO_04 or obj in PUXAR_DO_NUCLEO):
                t = tema_de(stem)
                if t:
                    contagem_tema[obj][t] += 1
                pre.append((path, ('OBJ04', obj), t))
            elif sub in NUCLEO_04:
                pre.append((path, 'MANTER', None))
            else:
                pre.append((path, None, rel))  # residuo 04
        elif topo == C02 and len(partes) > 1 and partes[1] == 'Doutrina e artigos':
            disc = partes[2] if len(partes) > 2 else 'Geral'
            if KW_APOSTILA.search(stem):
                dest = f'{C02}\\Apostilas\\{disc}'
            else:
                pgs = paginas_pdf(path) if ext == '.pdf' else None
                tam = os.path.getsize(lp(path)) if os.path.exists(lp(path)) else 0
                if KW_LIVRO.search(stem) or (pgs and pgs >= 100) \
                        or (pgs is None and tam >= 15 * 1024 * 1024):
                    dest = f'{C02}\\Livros\\{disc}'
                else:
                    dest = f'{C02}\\Artigos e textos\\{disc}'
            pre.append((path, dest, 'DIRETO'))
        elif topo == C02:  # Concursos e provas
            if KW_EDITAL.search(stem):
                dest = f'{C02}\\Editais de concurso'
            elif KW_APOSTILA.search(stem):
                dest = f'{C02}\\Apostilas\\Concursos'
            elif KW_PROVA.search(stem):
                dest = f'{C02}\\Provas e simulados'
            else:
                dest = f'{C02}\\Concursos e provas'  # plano na raiz
            pre.append((path, dest, 'DIRETO'))
        elif topo == C03 and 'TSE - gabinete' in rel:
            if KW_PAUTA.search(stem):
                dest = f'{C03}\\TSE - gabinete e sessoes\\Pautas e sessoes'
            elif KW_ESTUDO.search(stem):
                dest = f'{C03}\\TSE - gabinete e sessoes\\Estudos e roteiros'
            elif KW_PLANEJ.search(stem):
                dest = f'{C03}\\TSE - gabinete e sessoes\\Planejamento e relatorios'
            else:
                dest = f'{C03}\\TSE - gabinete e sessoes'
            pre.append((path, dest, 'DIRETO'))
        elif topo == '05 - Diversos':
            pre.append((path, '05 - Diversos\\Documentos nao identificados', 'DIRETO'))
        else:
            # faixa dissolvida fora dos escopos (01, Cartorio): sobe p/ a mae
            pre.append((path, rel, 'DIRETO'))

    # ---- 4. resolve destinos finais ----
    for path, marc, extra in pre:
        atual = os.path.relpath(os.path.dirname(path), ROOT)
        if marc is None:
            residuos.append(path)
            continue
        if marc == 'MANTER':
            stats['manter_nucleo'] += 1
            continue
        if isinstance(marc, tuple):  # objeto do 04 (tema se houver massa)
            obj = marc[1]
            t = extra
            if t and contagem_tema[obj][t] >= MIN_TEMA:
                dest = f'{C04}\\{obj}\\{t}'
            else:
                dest = f'{C04}\\{obj}'
        else:
            dest = marc
        if dest == atual:
            stats['ja_no_lugar'] += 1
            continue
        decisoes.append((path, dest, 'regra'))
        stats[f'-> {dest.split(chr(92))[0]}|{dest}'] += 1

    # ---- 5. grava plano + lotes de residuo ----
    with open(PLANO, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['path', 'destino', 'camada'])
        w.writerows(decisoes)
    os.makedirs(DIR_LOTES, exist_ok=True)
    for velho in os.listdir(DIR_LOTES):
        if velho.startswith('acv9_claude_lote_'):
            os.remove(os.path.join(DIR_LOTES, velho))
    for i in range(0, len(residuos), TAM_LOTE):
        n = i // TAM_LOTE + 1
        with open(os.path.join(DIR_LOTES, f'acv9_claude_lote_{n:02d}.csv'),
                  'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['id', 'path', 'pasta', 'nome'])
            for j, p in enumerate(residuos[i:i + TAM_LOTE]):
                w.writerow([j, p, os.path.relpath(os.path.dirname(p), ROOT),
                            os.path.basename(p)])
    print(f'Plano: {len(decisoes):,} movimentos -> {PLANO}')
    print(f"Mantidos no nucleo: {stats['manter_nucleo']:,} | "
          f"ja no lugar: {stats['ja_no_lugar']:,} | "
          f"residuos p/ lotes Claude: {len(residuos):,} "
          f"({(len(residuos) + TAM_LOTE - 1) // TAM_LOTE} lotes em {DIR_LOTES})")
    print('Destinos (top 40):')
    ordenado = sorted(((v, k) for k, v in stats.items() if k.startswith('->')),
                      reverse=True)
    for v, k in ordenado[:40]:
        print(f'  {v:5,}  {k.split("|", 1)[1]}')


def dests_validos():
    ok = set()
    for n, _ in OBJ04:
        ok.add(f'{C04}\\{n}')
    ok.add(f'{C04}\\Planilhas de gastos')
    ok.add(f'{C03}\\Cartorio de protesto\\Operacao financeira')
    ok.add(f'{C04}\\Documentos pessoais\\Senhas e acessos')
    ok.add(f'{C04}\\Documentos pessoais\\Imoveis')
    for dp, dn, _ in os.walk(lp(ROOT)):
        base = dp[4:] if dp.startswith('\\\\?\\') else dp
        rel = os.path.relpath(base, ROOT)
        if rel != '.':
            ok.add(rel)
    return ok


def incorporar():
    validos = dests_validos()
    linhas = []
    stats = Counter()
    lotes = {}
    for f in sorted(os.listdir(DIR_LOTES)):
        m = re.match(r'acv9_claude_lote_(\d+)\.csv$', f)
        if m:
            with open(os.path.join(DIR_LOTES, f), encoding='utf-8-sig') as fh:
                lotes[int(m.group(1))] = {r['id']: r['path']
                                          for r in csv.DictReader(fh)}
    for f in sorted(os.listdir(DIR_LOTES)):
        m = re.match(r'acv9_claude_resp_(\d+)\.csv$', f)
        if not m:
            continue
        mapa = lotes.get(int(m.group(1)), {})
        with open(os.path.join(DIR_LOTES, f), encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                path = mapa.get(r['id'])
                dest = (r.get('destino') or '').strip().strip('\\')
                if not path:
                    stats['id_desconhecido'] += 1
                    continue
                if dest.upper() == 'MANTER':
                    stats['manter'] += 1
                    continue
                if dest not in validos:
                    stats['destino_invalido'] += 1
                    continue
                linhas.append((path, dest, 'claude'))
                stats['claude'] += 1
    with open(PLANO, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(linhas)
    print(f'Incorporacao: {dict(stats)}')


def executar():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_biblio_{ts}.csv')
    ok = err = 0
    vistos = set()
    dirs_origem = set()
    with open(PLANO, encoding='utf-8-sig') as f, \
            open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['de', 'para', 'status'])
        linhas = [r for r in csv.DictReader(f)]
        for i, r in enumerate(linhas, 1):
            src = r['path']
            if src.lower() in vistos:
                continue
            vistos.add(src.lower())
            destdir = os.path.join(ROOT, r['destino'])
            nome = os.path.basename(src)
            dst = os.path.join(destdir, nome)
            status = 'ok'
            try:
                if not os.path.exists(lp(src)):
                    status = 'origem_sumiu'
                else:
                    os.makedirs(lp(destdir), exist_ok=True)
                    if os.path.exists(lp(dst)) and dst.lower() != src.lower():
                        stem, ext = os.path.splitext(nome)
                        n = 2
                        while os.path.exists(lp(dst)):
                            dst = os.path.join(destdir, f'{stem} ({n}){ext}')
                            n += 1
                    if dst.lower() != src.lower():
                        os.rename(lp(src), lp(dst))
                        dirs_origem.add(os.path.dirname(src))
                    else:
                        status = 'ja_no_lugar'
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([src, dst, status])
            ok, err = (ok + 1, err) if status == 'ok' else (ok, err + 1)
            if i % 1000 == 0:
                print(f'  {i:,}/{len(linhas):,} (problemas {err})', flush=True)
    # remove pastas esvaziadas (sobe enquanto vazio)
    removidas = 0
    for d in sorted(dirs_origem, key=len, reverse=True):
        while d and len(d) > len(ROOT):
            try:
                os.rmdir(lp(d))
                removidas += 1
                d = os.path.dirname(d)
            except OSError:
                break
    print(f'Aplicados: {ok:,} | problemas: {err:,} | '
          f'pastas vazias removidas: {removidas} | log: {log_path}')


if __name__ == '__main__':
    if '--incorporar' in sys.argv:
        incorporar()
    elif '--exec' in sys.argv:
        executar()
    else:
        montar_plano()
