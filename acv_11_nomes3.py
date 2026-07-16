# -*- coding: utf-8 -*-
r"""acv_11: 3a rodada de nomes — PREFIXO CANONICO por pasta de objeto (13/07/2026).

Diretriz do usuario: "em cada pasta aberta, deve haver um padrao no nome dos
documentos ali juntados". Exemplos de ruido: "DOC 3 - contracheques - Luiz
Flavio Fernandes", "Lucas de Souza - contracheques", "2° Via Fatura - ...".

Por pasta de objeto (04), em cadeia:
 (a) strip de prefixo de juntada "DOC N -"/"Anexo N -" (todo 02/03/04/05);
 (b) normaliza "2ª/segunda via" -> "2a via"; se a palavra-objeto aparece FORA
     do inicio, remove e prefixa "Objeto - "; no inicio, canoniza p/ singular;
 (c) [pastas 'forte'] nome neutro (sem outra palavra de tipo documental)
     ganha o prefixo default da pasta.
Depois: caixa (whitelist do acv_10), preposicoes, sentence case, espacos.

Uso: --plano | --exec
"""
import csv
import datetime
import os
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from acv_02_cruzar import lp
from acv_10_nomes2 import caixa_gritados

ROOT = r'C:\Users\mauri\HD_Mau'
PLANO = os.path.join(BASE, 'acv_plano_nomes3.csv')
C04 = '04 - Pessoal e financeiro'

RE_DOC_JUNTADA = re.compile(r'^(?:DOC|Doc)\.?\s*\d+\s*[-. ]+\s*')
RE_ANEXO = re.compile(r'^Anexos?\s*\d+\s*[A-Z]?\s*[- ]+\s*(?=[a-zA-Z]{2,})', re.I)
RE_VIA = re.compile(r'\b(?:(2|3)\s*[ªºao°]\s*|segunda\s+)vias?\b', re.I)

# (pasta, prefixo default, regex da palavra-objeto, modo)
# modo: forte = a+b+c | medio = a+b | leve = a
OBJ = [
    (f'{C04}\\Boletos', 'Boleto', r'\bboletos?\b', 'forte'),
    (f'{C04}\\Faturas', 'Fatura', r'\bfaturas?\b', 'forte'),
    (f'{C04}\\Recibos', 'Recibo', r'\brecibos?\b', 'forte'),
    (f'{C04}\\Comprovantes de pagamento', 'Comprovante de pagamento',
     r'\bcomprovantes?(?:\s+de)?\s+(?:pagamentos?|pgtos?\.?)\b|\bcomprovantes?\b', 'medio'),
    (f'{C04}\\Extratos bancarios', 'Extrato', r'\bextratos?\b', 'medio'),
    (f'{C04}\\Notas fiscais', 'Nota fiscal', r'\bnotas?\s+fisca(?:l|is)\b', 'medio'),
    (f'{C04}\\Contracheques e fichas financeiras', 'Contracheque',
     r'\bcontra-?cheques?\b|\bholerites?\b', 'medio'),
    (f'{C04}\\Certidoes', 'Certidao', r'\bcertid(?:ao|oes)\b', 'medio'),
    (f'{C04}\\Declaracoes', 'Declaracao', r'\bdeclarac(?:ao|oes)\b', 'medio'),
    (f'{C04}\\Curriculos', 'Curriculo', r'\bcurriculum(?:\s+vitae)?\b|\bcurriculos?\b', 'medio'),
    (f'{C04}\\Planilhas de gastos', 'Planilha', r'\bplanilhas?\b', 'forte'),
    (f'{C04}\\Guias e impostos', None, None, 'leve'),
    (f'{C04}\\Saude e exames', None, None, 'leve'),
    (f'{C04}\\Viagens', None, None, 'leve'),
]
OBJ = [(p, d, re.compile(rx, re.I) if rx else None, m) for p, d, rx, m in OBJ]

# nome que ja contem outro tipo documental nao ganha prefixo em (c)
RE_TIPO_DOC = re.compile(
    r'\bcontratos?\b|\bextratos?\b|\btermos?\b|\bprocuracao\b|\bavisos?\b|'
    r'\bguias?\b|\bfichas?\b|\bcomprovantes?\b|\bnotas?\b|\brecibos?\b|'
    r'\bdeclarac|\bcertid|\binformes?\b|\bdemonstrativos?\b|\brelatorios?\b|'
    r'\bcontra-?cheques?\b|\bholerites?\b|\bfaturas?\b|\bboletos?\b|\bdados\b|'
    r'\bplanilhas?\b|\bcartas?\b|\boficios?\b|\bpeticao\b|\blaudos?\b|'
    r'\bexames?\b|\batestados?\b|\breceit|\bcurriculos?\b|\bedita(?:l|is)\b|'
    r'\balvara\b|\bapolices?\b|\bpropostas?\b|\borcamentos?\b|\bcontroles?\b|'
    r'\bbalanc|\bdae\b|\bdarf\b|\biptu\b|\bipva\b|\brpv\b|\bnfс?e\b|\bgps\b|'
    r'\bcheques?\b|\bcarne\b|\bcupom\b|\bpedidos?\b|\bconfirmacao\b', re.I)

PREPOS = r'(?<=[a-z0-9] )(DE|DA|DO|DOS|DAS|E|EM|NO|NA|COM|PARA|POR|SEM)(?= [A-Z0-9])'


def regras_pasta(rel_dir):
    for pasta, default, rx, modo in OBJ:
        if rel_dir == pasta or rel_dir.startswith(pasta + '\\'):
            return default, rx, modo
    return None, None, None


def novo_stem(rel_dir, stem):
    topo = rel_dir.split('\\')[0]
    # (a) prefixo de juntada — fora do 01 (la a ordem dos DOC e processual)
    if topo != '01 - Juridico':
        stem = RE_DOC_JUNTADA.sub('', stem)
        stem = RE_ANEXO.sub('', stem)
    default, rx, modo = regras_pasta(rel_dir)
    if modo in ('forte', 'medio') and rx:
        stem = re.sub(r'^\d{1,3} (?=[A-Za-z])', '', stem)  # mes/nº de serie
        stem = RE_VIA.sub(lambda m: f'{m.group(1) or 2}a via', stem)
        # serie "<Emissor> fatura <data>" -> "Fatura - <Emissor> - <data>"
        if default and not RE_TIPO_DOC.match(stem):
            stem = re.sub(
                r'^([A-Za-z][\w ]{1,24}?)\s+(?i:' + rx.pattern + r')\s*[- ]*(?=\d)',
                lambda m: f'{default} - {m.group(1)} - ', stem, count=1)
            stem = re.sub(r'\s*-\s*(-\s*)+', ' - ', stem).strip(' -.')
        # "2a via da Fatura X" -> processa "Fatura X" e anexa "2a via" ao fim
        via = None
        mv = re.match(r'^([23])a via (?:d[aeo]s? )?', stem)
        if mv:
            via = f'{mv.group(1)}a via'
            stem = stem[mv.end():]
        m = rx.search(stem)
        antes = stem[:m.start()].rstrip() if m else ''
        if m and m.start() > 0 and antes.endswith('-') \
                and not RE_TIPO_DOC.match(stem):
            # reordena so segmento proprio ("X - boleto ..."), nunca frase
            resto = stem[:m.start()] + stem[m.end():]
            resto = re.sub(r'\s{2,}', ' ', resto)
            resto = re.sub(r'\s*-\s*(-\s*)+', ' - ', resto).strip(' -.,')
            stem = f'{default} - {resto}' if resto else default
        elif m and m.start() == 0:
            resto = stem[m.end():]
            pleno = ' ' in m.group(0).strip()
            if not resto:
                stem = default
            elif pleno or re.match(r'^\s*[-\d]', resto):
                if not re.match(r'^\s*-', resto) and not pleno:
                    resto = ' - ' + resto.lstrip(' -')
                stem = default + resto
            # senao: variacao legitima ("Comprovante de transferencia...") fica
        elif not m and modo == 'forte' and default \
                and not RE_TIPO_DOC.search(stem):
            stem = f'{default} - {stem}'
        if via:
            m_data = re.search(r' - (?:\d{2}-)?\d{2}-\d{4}$', stem)
            if m_data:
                stem = f'{stem[:m_data.start()]} - {via}{stem[m_data.start():]}'
            else:
                stem = f'{stem} - {via}'
    # caixa e acabamento (mesma politica do acv_10)
    stem = caixa_gritados(stem)
    stem = re.sub(PREPOS, lambda m: m.group(0).lower(), stem)
    if stem and stem[0].islower():
        stem = stem[0].upper() + stem[1:]
    stem = re.sub(r'\s{2,}', ' ', stem)
    stem = re.sub(r'\s*-\s*(-\s*)+', ' - ', stem).strip(' -.')
    return stem


def montar_plano():
    linhas = []
    usados = set()
    stats = Counter()
    for dp, dn, fn in os.walk(lp(ROOT)):
        base = dp[4:] if dp.startswith('\\\\?\\') else dp
        rel = os.path.relpath(base, ROOT)
        if rel.startswith('05 - Diversos\\Sistema e programas'):
            continue
        for f in fn:
            if f.startswith('00 - ') or f.lower() in ('desktop.ini', 'thumbs.db'):
                continue
            stem, ext = os.path.splitext(f)
            s2 = novo_stem(rel, stem)
            if s2 == stem:
                continue
            alvo = os.path.join(base, s2 + ext.lower())
            n = 1
            while alvo.lower() in usados or \
                    (alvo.lower() != os.path.join(base, f).lower()
                     and os.path.exists(lp(alvo))):
                n += 1
                alvo = os.path.join(base, f'{s2} ({n}){ext.lower()}')
            usados.add(alvo.lower())
            linhas.append((os.path.join(base, f), alvo))
            stats[rel.split('\\')[0]] += 1
    with open(PLANO, 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['de', 'para'])
        w.writerows(linhas)
    print(f'Plano nomes3: {sum(stats.values()):,} renames {dict(stats)} -> {PLANO}')
    for de, para in linhas[:18]:
        print(f'  {os.path.basename(de)}\n    -> {os.path.basename(para)}')


def executar():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_nomes3_{ts}.csv')
    ok = err = 0
    with open(PLANO, encoding='utf-8-sig') as f, \
            open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['de', 'para', 'status'])
        linhas = list(csv.DictReader(f))
        for i, r in enumerate(linhas, 1):
            de, para = r['de'], r['para']
            status = 'ok'
            try:
                if not os.path.exists(lp(de)):
                    status = 'origem_sumiu'
                elif os.path.exists(lp(para)) and de.lower() != para.lower():
                    status = 'destino_ocupado'
                else:
                    os.rename(lp(de), lp(para))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([de, para, status])
            ok, err = (ok + 1, err) if status == 'ok' else (ok, err + 1)
            if i % 1000 == 0:
                print(f'  {i:,}/{len(linhas):,} (problemas {err})', flush=True)
    print(f'Renomeados: {ok:,} | problemas: {err:,} | log: {log_path}')


if __name__ == '__main__':
    if '--exec' in sys.argv:
        executar()
    else:
        montar_plano()
