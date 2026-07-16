# -*- coding: utf-8 -*-
r"""acv_10: 2a rodada de nomes — canonizacao INTRA-PASTA (13/07/2026).

Diretriz do usuario: documentos da mesma natureza na mesma pasta devem seguir
o MESMO padrao nominal (prefixo canonico, caixa, numeracao), para ordenar por
nome. Ex.: "Listas - sessao X" == "Listas para publicacao - sessao X".

--audio : envia 05\Audios e musicas (inteira) + audio avulso a Lixeira.
--plano : gera acv_plano_nomes2.csv (rename e/ou realocacao leve).
--exec  : aplica com log reversivel.

Camadas por arquivo: regra da pasta (mapa REGRAS, pode ter destino novo) ->
siglas iniciais -> caixa (runs GRITADOS -> Capitalize; sigla curta fica) ->
sentence case -> espacos -> ext minuscula -> colisao " (2)".
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
from org2_09_nomes_determinismo import PALAVRAS_COMUNS

# so des-gritar palavra CONHECIDA (siglas desconhecidas ficam como estao;
# licao 13/07: runs por " - " destruiam CEF/FIES/TJMG/SINTRACOOP)
WL_GRITADAS = set(PALAVRAS_COMUNS) | {
    'JANEIRO', 'FEVEREIRO', 'MARCO', 'ABRIL', 'MAIO', 'JUNHO', 'JULHO',
    'AGOSTO', 'SETEMBRO', 'OUTUBRO', 'NOVEMBRO', 'DEZEMBRO',
    'FLUXO', 'CAIXA', 'SALDO', 'BALANCO', 'MENSAL', 'VERSAO', 'FINAL',
    'GASTOS', 'PLANILHA', 'LISTA', 'LISTAS', 'TITULO', 'TITULOS',
    'PROTESTO', 'TABELIONATO', 'RECEBIMENTO', 'PAGAMENTO', 'PAGAMENTOS',
    'REGISTRO', 'PESSOAIS', 'DESPESAS', 'ORCAMENTO', 'HOLERITE', 'CLASSE',
    'DOCUMENTO', 'DOCUMENTOS', 'CADASTRAL', 'FINANCEIRA', 'FINANCEIRO',
    'CONTRACHEQUE', 'EXTRATO', 'FATURA', 'BOLETO', 'GUIA', 'IMPOSTO',
    'PAUTA', 'SESSAO', 'JULGAMENTO', 'JURISDICIONAL', 'CONFIRMACAO',
    'TRANSFERENCIA', 'DEPOSITO', 'AGENDAMENTO', 'RETORNO', 'REPASSE',
}

ROOT = r'C:\Users\mauri\HD_Mau'
PLANO = os.path.join(BASE, 'acv_plano_nomes2.csv')
EXT_AUDIO = {'.mp3', '.wma', '.wav', '.m4a', '.ogg', '.flac', '.aac',
             '.mid', '.midi', '.amr', '.aif', '.aiff', '.ra', '.m3u', '.pls'}

C01 = '01 - Juridico'
C03 = '03 - Administrativo'
C04 = '04 - Pessoal e financeiro'
PG = f'{C04}\\Planilhas de gastos'
FD = f'{C04}\\Financas domesticas'
SEN = f'{C04}\\Documentos pessoais\\Senhas e acessos'


def R(p, flags=re.I):
    return re.compile(p, flags)


# (pasta-prefixo, [(regex, repl, destino_ou_None), ...]) — 1a que casar aplica TODAS em cadeia
REGRAS = [
    (PG, [
        (R(r'^\d{1,2} (?=[A-Za-z])'), '', None),
        (R(r'^FLUXO DE CAIXA[ -]*'), 'Planilha - fluxo de caixa - ', None),
        (R(r'^SALDO DO DIA[ -]*'), 'Planilha - saldo do dia - ', None),
        (R(r'^Planilha CAIXA[ -]*'), 'Planilha - caixa - ', None),
        (R(r'^Apto 43[ -]*'), 'Planilha - Apto 43 - ', None),
        (R(r'^(\d{2}-\d{2}-\d{4})$'), r'Planilha de gastos - \1', None),
    ]),
    (f'{C04}\\Contracheques e fichas financeiras', [
        (R(r'^\d{1,2} (Contracheque|Holerite)\b[ -]*'), 'Contracheque - ', None),
        (R(r'^Holerite\b'), 'Contracheque', None),
        (R(r'^Ficha cadastral CLT (Empresa Brasileira(?: de)? Correios(?: e)? Telegrafos|Correios|ECT)[ -]*'),
         'Ficha cadastral ECT - ', None),
        (R(r'^Ficha cadastral Correios[ -]*'), 'Ficha cadastral ECT - ', None),
        (R(r'^(.{3,60}?) - financeir[ao]$'), r'Ficha financeira - \1', None),
        (R(r'^Ficha Financeira\b', 0), 'Ficha financeira', None),
        (R(r'^Ficha Cadastral\b', 0), 'Ficha cadastral', None),
    ]),
    (f'{C03}\\TSE - gabinete e sessoes\\Pautas e sessoes', [
        (R(r'^Jurisdicional\s*-\s*'), 'Pauta jurisdicional - ', None),
        (R(r'^Jurisdicional (?=\d)'), 'Pauta jurisdicional ', None),
        (R(r'^(\d{1,2}) pauta jurisdicional$'), r'Pauta jurisdicional \1', None),
        (R(r'^Listas? para publicacao - sessao'), 'Lista de publicacao - sessao', None),
        (R(r'^Listas? - sessao'), 'Lista de publicacao - sessao', None),
        (R(r'^Pauta REspe\b'), 'Pauta de julgamento - REspe', None),
    ]),
    (f'{C01}\\Peticoes\\Trabalhista\\Peticoes diversas', [
        (R(r'^Pet\b\s*-?\s*', 0), 'Peticao - ', None),
    ]),
    (f'{C04}\\Comprovantes de pagamento', [
        (R(r'^(\d{1,2})a (mensalidade|parcela) - (?:comprovante de )?pagamento$'),
         r'Comprovante de pagamento - \2 \1a', None),
        (R(r'^Comprovante Marajinho\b'), 'Comprovante de pagamento - Marajinho', None),
        (R(r'^Comprovante - '), 'Comprovante de pagamento - ', None),
        (R(r'^Comprovante de (TED|DOC)\b'), r'Comprovante de pagamento - \1', None),
        (R(r'^Confirmacao pedido\b'), 'Comprovante de pagamento - pedido', None),
    ]),
    (f'{C03}\\Listas e planilhas', [
        (R(r'^\d{2} Planilha[ -]*(?=\d)'), 'Planilha - ', PG),
        (R(r'^(\d{2}) Balanco Mensal - ([A-Za-z]+)$'), r'Balanco mensal - \2', FD),
        (R(r'^Apto 43 - (?=\d)'), 'Planilha - Apto 43 - ', PG),
    ]),
    (f'{C03}\\Cartorio de protesto', [
        (R(r'^Registro (?:de )?log[ -]*(?:upload[ -]*)?(?:confirmacao[ -]*)?(?:cartorio[ -]*)?'),
         'Registro de log - ', None),
    ]),
    (f'{C03}\\Portarias e resolucoes', [
        (R(r'^Portaria n[oº]\s*'), 'Portaria ', None),
    ]),
]

RE_SENHA = R(r'^(Credencia\w*|Logins?\b|Senhas?\b|Dados de acesso|Chave -|Acesso -|Iupp\b)')
SIGLAS_INI = {'dae', 'darf', 'iptu', 'ipva', 'grctj', 'gru', 'gps', 'cda',
              'rpv', 'nfe', 'gfip', 'sefip', 'dimob', 'irpf', 'dirf', 'cnj',
              'oab', 'inss', 'fgts', 'tfj', 'grrf', 'dpvat', 'cnis', 'pccs'}
RE_TOKEN_UP = re.compile(r'\b[A-Z]{3,}\b')


def caixa_gritados(stem):
    """Des-grita apenas palavras CONHECIDAS; siglas desconhecidas ficam."""
    return RE_TOKEN_UP.sub(
        lambda m: m.group(0).capitalize() if m.group(0) in WL_GRITADAS
        else m.group(0), stem)


def novo_nome(rel_dir, stem, ext):
    dest = None
    for pasta, regras in REGRAS:
        if rel_dir == pasta or rel_dir.startswith(pasta + '\\'):
            for rx, repl, d in regras:
                s2 = rx.sub(repl, stem, count=1)
                if s2 != stem and d:
                    dest = d
                stem = s2
            break
    # senhas soltas no 04 -> Senhas e acessos
    if rel_dir.startswith(C04) and not rel_dir.startswith(SEN) \
            and RE_SENHA.match(stem):
        dest = SEN
    # sigla inicial
    m = re.match(r'^([a-z]{2,6})\b', stem)
    if m and m.group(1) in SIGLAS_INI:
        stem = m.group(1).upper() + stem[m.end():]
    stem = caixa_gritados(stem)
    # AO/AOS fora: "Sentenca AO 2009..." e sigla de Acao Ordinaria
    stem = re.sub(r'(?<=[a-z0-9] )(DE|DA|DO|DOS|DAS|E|EM|NO|NA|COM|PARA|POR|SEM)(?= [A-Z0-9])',
                  lambda m: m.group(0).lower(), stem)
    if stem and stem[0].islower():
        stem = stem[0].upper() + stem[1:]
    stem = re.sub(r'\s{2,}', ' ', stem)
    stem = re.sub(r'\s*-\s*-\s*', ' - ', stem).strip(' -.')
    return stem, ext.lower(), dest


def montar_plano():
    linhas = []
    usados = set()
    stats = Counter()
    for dp, dn, fn in os.walk(lp(ROOT)):
        base = dp[4:] if dp.startswith('\\\\?\\') else dp
        rel = os.path.relpath(base, ROOT)
        if rel.startswith('05 - Diversos\\Sistema e programas'):
            continue  # artefatos tecnicos: nome original e referencia
        for f in fn:
            if f.startswith('00 - ') or f.lower() in ('desktop.ini', 'thumbs.db'):
                continue
            stem, ext = os.path.splitext(f)
            s2, e2, dest = novo_nome(rel, stem, ext)
            destdir = os.path.join(ROOT, dest) if dest else base
            if s2 == stem and e2 == ext and destdir == base:
                continue
            alvo = os.path.join(destdir, s2 + e2)
            n = 1
            while alvo.lower() in usados or \
                    (alvo.lower() != os.path.join(base, f).lower()
                     and os.path.exists(lp(alvo))):
                n += 1
                alvo = os.path.join(destdir, f'{s2} ({n}){e2}')
            usados.add(alvo.lower())
            linhas.append((os.path.join(base, f), alvo))
            stats['realocar' if dest else 'renomear'] += 1
    with open(PLANO, 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['de', 'para'])
        w.writerows(linhas)
    print(f'Plano nomes2: {dict(stats)} -> {PLANO}')
    for de, para in linhas[:15]:
        print(f'  {os.path.basename(de)}\n    -> {os.path.basename(para)}')


def excluir_audio():
    from send2trash import send2trash
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_audio_{ts}.csv')
    alvos = []
    pasta = os.path.join(ROOT, '05 - Diversos', 'Audios e musicas')
    if os.path.isdir(pasta):
        alvos.append(pasta)
    for dp, dn, fn in os.walk(lp(ROOT)):
        base = dp[4:] if dp.startswith('\\\\?\\') else dp
        if base.startswith(pasta):
            continue
        for f in fn:
            if os.path.splitext(f)[1].lower() in EXT_AUDIO:
                alvos.append(os.path.join(base, f))
    n = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['excluido_para_lixeira'])
        for a in alvos:
            try:
                send2trash(a)
                log.writerow([a])
                n += 1
            except OSError as e:
                log.writerow([f'ERRO {a}: {e}'])
    print(f'Enviados a Lixeira: {n} (pasta inteira conta 1) | log: {log_path}')


def executar():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_nomes2_{ts}.csv')
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
                    os.makedirs(lp(os.path.dirname(para)), exist_ok=True)
                    os.rename(lp(de), lp(para))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([de, para, status])
            ok, err = (ok + 1, err) if status == 'ok' else (ok, err + 1)
            if i % 1000 == 0:
                print(f'  {i:,}/{len(linhas):,} (problemas {err})', flush=True)
    print(f'Renomeados: {ok:,} | problemas: {err:,} | log: {log_path}')


if __name__ == '__main__':
    if '--audio' in sys.argv:
        excluir_audio()
    elif '--exec' in sys.argv:
        executar()
    else:
        montar_plano()
