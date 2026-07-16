# -*- coding: utf-8 -*-
r"""acv_07: padronizacao GLOBAL de nomes do HD_Mau — deterministica, sem IA.

Pipeline por stem (ordem importa), reusando org2_09:
 1. nao-ASCII -> ascii_seg; underscores -> espaco; espacos/pontuacao duplicada;
 2. prefixo "(N) " removido; sufixo " (N)"/" [N]" orfao removido (se nome livre);
 3. prefixo numerico ruido (^NN[_.- ]) removido se o resto e descritivo e nao e data;
 4. GRITADOS -> normalizar_caixa (siglas/romanos preservados); normalizar_datas;
 5. valida_stem (acv_04): rejeita rename que perde CNJ ou desloca "CLASSE numero";
 6. colisoes -> " (2)"; teto de caminho 259.

Uso: python acv_07_nomes.py --plano   (gera acv_plano_nomes.csv)
     python acv_07_nomes.py --exec
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
from acv_03_organizar import ascii_seg
from acv_04_reordenar_hdmau import valida_stem
from org2_09_nomes_determinismo import (MESES, normalizar_caixa,
                                        normalizar_datas, normalizar_espacos)

# regra de mes por extenso do org2_09 usa "mai[a-z]*" e engoliria sobrenomes
# ("Maia - 2010" -> "05-2010", "Julgamento 2015" -> "07-2015"); versao estrita:
MES_FULL = (r'(?:jan(?:eiro)?|fev(?:ereiro)?|mar(?:co)?|abr(?:il)?|maio?|'
            r'jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|'
            r'nov(?:embro)?|dez(?:embro)?)')
RE_MES_ANO = re.compile(
    r'\b(' + MES_FULL + r')\b[ .\-]*(?:de )?(\d{4}|\d{2})\b(?![\-/.]\d)', re.I)
# "Abr 14 19" (extratos, mes dia ano) -> 14-04-2019
RE_MES_DIA_AA = re.compile(
    r'\b(' + MES_FULL + r')\b[ .](\d{1,2}) (\d{2})\b(?![\-/.\d])', re.I)
# intocaveis: citacao oficial "30 de marco de 2022" e intervalo "janeiro-marco"
RE_CITACAO = re.compile(
    r'\b\d{1,2} de ' + MES_FULL + r' de (?:19|20)\d{2}\b', re.I)
RE_INTERVALO = re.compile(
    r'\b' + MES_FULL + r'\s*(?:-|a|e)\s*' + MES_FULL + r'\b', re.I)
RE_MES_TRAP = re.compile(
    r'\b(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[a-z]*[ .\-]*(?:de )?'
    r'(\d{4}|\d{2})\b(?![\-/.]\d)', re.I)


def normalizar_datas_seg(s):
    # protege citacoes oficiais e intervalos de meses
    protegidos = {m.group(0) for rx in (RE_CITACAO, RE_INTERVALO)
                  for m in rx.finditer(s)}

    def fdia(m):
        mo = MESES[m.group(1)[:3].lower()]
        d, aa = int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and aa <= 26:
            return f'{d:02d}-{mo}-20{aa:02d}'
        return m.group(0)

    def fmes(m):
        mo = MESES[m.group(1)[:3].lower()]
        a = m.group(2)
        if len(a) == 2:
            if int(a) > 26:
                return m.group(0)
            a = '20' + a
        return f'{mo}-{a}'

    mascaras = {t: f'\x00{i}\x00' for i, t in enumerate(protegidos)}
    for t, k in mascaras.items():
        s = s.replace(t, k)
    s = RE_MES_DIA_AA.sub(fdia, s)
    s = RE_MES_ANO.sub(fmes, s)
    # neutraliza a regra frouxa do org2_09 ("mai[a-z]*" engole "Maia")
    n = len(mascaras)
    for m in RE_MES_TRAP.finditer(s):
        t = m.group(0)
        if t not in mascaras:
            mascaras[t] = f'\x00{n}\x00'
            n += 1
    for t, k in mascaras.items():
        s = s.replace(t, k)
    s = normalizar_datas(s)
    for t, k in mascaras.items():
        s = s.replace(k, t)
    return s

ROOT = r'C:\Users\mauri\HD_Mau'
PLANO = os.path.join(BASE, 'acv_plano_nomes.csv')
MAXPATH = 259

RE_PREF_N = re.compile(r'^\(\d+\)\s*')
RE_SUF_N = re.compile(r'[ _]\((\d+)\)$|[ _]\[(\d+)\]$')
RE_PREF_NUM = re.compile(r'^\d{1,2}\s*[_.\-]+\s*')
RE_DATA_INICIO = re.compile(r'^\d{1,2}[-._ ]\d{1,2}[-._ ]\d{2,4}|^\d{4}[-._ ]\d{1,2}')


def novo_stem(stem, dirpath):
    s = stem
    # 1. ascii + underscores + espacos
    s = s.replace('_', ' ')
    s = ascii_seg(s, 200)
    s = normalizar_espacos(s)
    # 2. prefixo (N)
    s = RE_PREF_N.sub('', s)
    # 3. prefixo numerico ruido (nao remover se for data ou se o resto ficar fraco)
    m = RE_PREF_NUM.match(s)
    if m and not RE_DATA_INICIO.match(s):
        resto = s[m.end():]
        if len(re.sub(r'[^A-Za-z]', '', resto)) >= 8:
            s = resto
    # 4. caixa gritada + datas
    juridico = '01 - Juridico' in dirpath
    if len(s) > 12 and s.isupper():
        s = normalizar_caixa(s, juridico)
    s = normalizar_datas_seg(s)
    s = normalizar_espacos(s).strip(' .-')
    return s or stem


def montar_plano():
    linhas = []
    usados = set()
    stats = Counter()
    for dp, dn, fn in os.walk(lp(ROOT)):
        base = dp[4:] if dp.startswith('\\\\?\\') else dp
        for f in fn:
            if f.startswith('00 - '):
                continue
            stem, ext = os.path.splitext(f)
            ext_l = ext.lower()
            s = novo_stem(stem, base)
            # sufixo (N)/[N] orfao: so se o nome limpo estiver livre
            m = RE_SUF_N.search(s)
            if m:
                limpo = RE_SUF_N.sub('', s).rstrip()
                cand = os.path.join(base, limpo + ext_l)
                if not os.path.exists(lp(cand)) and cand.lower() not in usados:
                    s = limpo
                    stats['sufixo_removido'] += 1
            # validacao juridica (CNJ / prefixo de classe) — so como portao,
            # sem usar o retorno (que trunca a 110 chars)
            if s != stem and not valida_stem(s, stem):
                stats['rejeitado_valida_stem'] += 1
                continue
            if s == stem and ext_l == ext:
                continue
            maxstem = max(20, MAXPATH - len(base) - len(ext_l) - 6)
            s = s[:maxstem].rstrip(' .-')
            destino = os.path.join(base, s + ext_l)
            n = 1
            while destino.lower() in usados or \
                    (destino.lower() != os.path.join(base, f).lower()
                     and os.path.exists(lp(destino))):
                n += 1
                destino = os.path.join(base, f'{s} ({n}){ext_l}')
            if destino == os.path.join(base, f):
                continue
            usados.add(destino.lower())
            linhas.append((os.path.join(base, f), destino))
            stats['renomear'] += 1
    with open(PLANO, 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['de', 'para'])
        w.writerows(linhas)
    print(f'Plano de nomes: {stats["renomear"]:,} renames -> {PLANO}')
    print(f'Stats: {dict(stats)}')
    for de, para in linhas[:12]:
        print(f'  {os.path.basename(de)}\n    -> {os.path.basename(para)}')


def executar():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_nomes_{ts}.csv')
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
                    # NTFS case-insensitive: rename so de caixa e permitido
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
