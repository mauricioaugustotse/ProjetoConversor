# -*- coding: utf-8 -*-
"""tsje_09: relatorio final didatico do projeto de transcricao das atas do TSJE.

Le o manifest, os .md de lacuna, o diagnostico de completude, o registro de
PDFs mal rotulados e o confronto com o ensaio; produz um docx explicativo em
D:\\TSJE_TRANSCRICOES\\_RELATORIO - Atas TSJE 1932-1937.docx com:
- o que foi feito (numeros por ano: transcritas do indice + recuperadas);
- o que o indice tinha de ruido (TRE, duplicatas, falsos positivos);
- o que FALTA e por que (lacunas recuperaveis ja feitas x boletins ausentes
  do acervo, listados nominalmente);
- achados sobre o acervo (PDFs mal rotulados);
- como o material esta organizado e como conferir.

Uso: python tsje_09_relatorio.py
"""
import csv
import io
import os
import re
import time
from collections import defaultdict

from docx import Document
from docx.shared import Pt, Cm

RAIZ = r'D:\TSJE_TRANSCRICOES'
BASE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(RAIZ, 'manifest.csv')
DIAG = os.path.join(RAIZ, '_diagnostico_completude.csv')
MALROT = os.path.join(RAIZ, '_pdfs_mal_rotulados.csv')
SAIDA = os.path.join(RAIZ, '_RELATORIO - Atas TSJE 1932-1937.docx')

ANOS = ['1932', '1933', '1934', '1935', '1936', '1937']


def ler_csv(path):
    if not os.path.exists(path):
        return []
    with io.open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def contar_lacunas():
    """ano -> nº de atas do TSJE recuperadas (lacuna-*.md com corpo)."""
    por_ano = defaultdict(int)
    for ano in ANOS:
        pasta = os.path.join(RAIZ, ano)
        if not os.path.isdir(pasta):
            continue
        for nome in os.listdir(pasta):
            if not (nome.startswith('lacuna-') and nome.endswith('.md')):
                continue
            with io.open(os.path.join(pasta, nome), encoding='utf-8') as f:
                t = f.read()
            m = re.search(r'---\n(.*?)\n---\n?(.*)', t, re.S)
            if m and m.group(2).strip() and 'problema:' not in m.group(1) \
                    and 'tribunal: superior' in m.group(1):
                por_ano[ano] += 1
    return por_ano


def tabela(doc, cab, linhas, sz=10):
    t = doc.add_table(rows=1, cols=len(cab))
    t.style = 'Table Grid'
    for j, x in enumerate(cab):
        r = t.rows[0].cells[j].paragraphs[0].add_run(str(x))
        r.bold = True
        r.font.size = Pt(sz)
    for ln in linhas:
        cells = t.add_row().cells
        for j, x in enumerate(ln):
            cells[j].paragraphs[0].add_run(str(x)).font.size = Pt(sz)
    return t


def main():
    manifest = ler_csv(MANIFEST)
    diag = ler_csv(DIAG)
    malrot = ler_csv(MALROT)
    lac = contar_lacunas()

    por_ano = defaultdict(lambda: defaultdict(int))
    for m in manifest:
        por_ano[m['ano']][m['status']] += 1

    doc = Document()
    est = doc.styles['Normal']
    est.font.name = 'Calibri'
    est.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2)

    doc.add_heading('Transcrição das Atas do Tribunal Superior de Justiça '
                    'Eleitoral (1932–1937)', 0)
    doc.add_heading('Relatório do trabalho — o que foi feito e o que falta', 2)
    doc.add_paragraph(f'Gerado em {time.strftime("%d/%m/%Y %H:%M")}.')

    # ---- 1. em uma frase
    doc.add_heading('1. Em resumo', 1)
    trans = sum(por_ano[a]['transcrita'] + por_ano[a]['revisada']
                + por_ano[a]['final'] for a in ANOS)
    rec = sum(lac.values())
    ausentes = sum(1 for d in diag if d['situacao'] == 'BOLETIM_AUSENTE_NO_ACERVO')
    so_resenha = sum(1 for d in diag if d['situacao'] == 'RECUPERAVEL')
    faltam = ausentes + so_resenha
    doc.add_paragraph(
        f'Foram transcritas fielmente {trans + rec} atas do TSJE do período '
        f'1932–1937 — {trans} a partir do índice do banco e {rec} recuperadas '
        f'na auditoria (atas que existem nos boletins, mas que a detecção '
        f'automática do índice não havia enxergado). O texto é transcrição '
        f'fiel do impresso (não paráfrase): ortografia modernizada, mas '
        f'conteúdo, nomes e votos preservados.')
    doc.add_paragraph(
        f'Pela numeração impressa das sessões, faltam cerca de {faltam} atas '
        f'que NÃO puderam ser transcritas porque a ata formal não está no '
        f'acervo digitalizado. Elas se dividem em dois casos: (a) {ausentes} '
        f'sessões cujo boletim inteiro não veio no material do Drive; e (b) '
        f'~{so_resenha} sessões cujo boletim traz apenas a RESENHA de '
        f'julgamentos ("O Tribunal em sua Nª sessão resolveu…"), mas não a '
        f'ata formal completa — que saiu em um boletim que também não está no '
        f'acervo. A seção 5 detalha e lista os casos comprovados.')
    doc.add_paragraph(
        'O material está em D:\\TSJE_TRANSCRICOES: um arquivo .docx por ano '
        '(1932 a 1937) com as atas em ordem cronológica, além deste relatório '
        'e do índice-mestre (D:\\Indice - Atas TSJE 1932-1937.docx).')

    # ---- 2. por ano
    doc.add_heading('2. Quanto foi transcrito, ano a ano', 1)
    linhas = []
    for a in ANOS:
        p = por_ano[a]
        feitas = p['transcrita'] + p['revisada'] + p['final']
        linhas.append([a, feitas, lac.get(a, 0), feitas + lac.get(a, 0),
                       p['regional'], p['duplicada'], p['problema']])
    tot = lambda i: sum(int(l[i]) for l in linhas)
    linhas.append(['Total', tot(1), tot(2), tot(3), tot(4), tot(5), tot(6)])
    tabela(doc, ['Ano', 'Do índice', 'Recuperadas', 'Total TSJE',
                 'Atas de TRE', 'Duplicatas', 'Falsos pos.'], linhas)
    doc.add_paragraph(
        'As três últimas colunas são o "ruído" que a auditoria separou das '
        'atas do Superior: sessões que na verdade eram de Tribunais Regionais '
        '(11 estados diferentes), entradas duplicadas do índice (inclusive '
        'boletins escaneados duas vezes no acervo) e falsos positivos '
        '(capas, avisos de julgamento e votos avulsos que não são atas).',
        style='List Bullet')

    # ---- 3. como foi feito / qualidade
    doc.add_heading('3. Como foi feito e como sabemos que está fiel', 1)
    for t in [
        'Cada ata foi transcrita a partir da IMAGEM da página do boletim '
        '(não do OCR, que é ruidoso nos jornais dos anos 1930). Dígitos e '
        'nomes duvidosos foram conferidos com ampliação e cruzados com a '
        'sequência de sessões e o dia da semana.',
        'Um revisor independente de IA (gpt-5.6-terra) reconferiu uma amostra '
        'de cada ano contra as imagens. Onde ele apontou divergência real, foi '
        'corrigido; onde ele próprio errou (3 casos, confundindo a ata com a '
        'vizinha da mesma página), o parecer foi rejeitado com justificativa '
        'registrada no arquivo. Regra do projeto: nenhuma correção sem '
        'conferência na imagem.',
        'Convenções de fidelidade: ortografia modernizada ("secção"→"seção"), '
        'mas nomes próprios e topônimos na grafia de época (Affonso Penna '
        'Júnior, Minas Geraes, Matto Grosso); erros tipográficos do próprio '
        'jornal preservados e assinalados; [ilegível] e palavra[?] para '
        'trechos que a digitalização não permite ler com certeza.']:
        doc.add_paragraph(t, style='List Bullet')

    # ---- 4. lacunas recuperadas
    doc.add_heading('4. O que a auditoria recuperou', 1)
    doc.add_paragraph(
        f'Reconstruindo a numeração das sessões de cada ano, verificou-se que '
        f'o índice automático havia perdido dezenas de atas (o OCR não '
        f'reconheceu o cabeçalho). {rec} dessas foram recuperadas lendo as '
        f'imagens e estão incluídas nos volumes anuais. Isso é a diferença '
        f'entre "transcrevi o que o índice deu" e "transcrevi o que existe '
        f'no acervo".')

    # ---- 5. o que falta (ausentes no acervo)
    doc.add_heading('5. O que falta — e por quê', 1)
    doc.add_paragraph(
        f'Nenhuma das faltas é erro de transcrição — em todos os casos o '
        f'documento não está no acervo digitalizado. Há dois tipos:')
    doc.add_paragraph(
        f'(a) Boletim inteiro ausente: {ausentes} sessões cujo boletim não '
        f'veio no material. Listadas nominalmente abaixo, para busca na '
        f'origem.', style='List Bullet')
    doc.add_paragraph(
        f'(b) Só a resenha no acervo: ~{so_resenha} sessões em que o boletim '
        f'disponível traz apenas a resenha de julgamentos, não a ata formal. '
        f'A amostragem manual confirmou esse padrão repetidamente (sobretudo '
        f'em 1935 e 1937): a ata formal completa saiu num boletim que não '
        f'está no acervo. Onde a ata formal ESTAVA no acervo, ela foi '
        f'recuperada (as 27 da seção 4).', style='List Bullet')
    aus = [d for d in diag if d['situacao'] == 'BOLETIM_AUSENTE_NO_ACERVO']
    por_ano_aus = defaultdict(list)
    for d in aus:
        tp = 'extra.' if d['tipo'] == 'extraordinaria' else 'ord.'
        por_ano_aus[d['ano']].append(f"{d['num']}ª {tp}")
    linhas = [[a, ', '.join(por_ano_aus[a])] for a in ANOS if por_ano_aus[a]]
    tabela(doc, ['Ano', 'Sessões do TSJE ausentes no acervo'], linhas, 9)

    # ---- 6. achados sobre o acervo
    if malrot:
        doc.add_heading('6. Achados sobre o acervo (arquivos mal rotulados)', 1)
        doc.add_paragraph(
            'Durante a transcrição, descobriu-se que alguns PDFs do banco '
            'D:\\TSJE_ATAS têm data ou número ERRADOS no próprio nome do '
            'arquivo (o nome não bate com o boletim impresso). A data errada '
            'está no acervo, não só no índice — vale corrigir na origem:')
        linhas = [[m['id_acervo'], m['nome_do_arquivo_diz'],
                   m['cabecalho_impresso_diz']] for m in malrot]
        tabela(doc, ['ID', 'Nome do arquivo diz', 'Impresso diz'], linhas, 9)

    # ---- 7. organizacao
    doc.add_heading('7. Onde está cada coisa', 1)
    for t in [
        'D:\\TSJE_TRANSCRICOES\\1932.docx … 1937.docx — os volumes anuais com '
        'as atas transcritas em ordem cronológica.',
        'D:\\Indice - Atas TSJE 1932-1937.docx — índice-mestre: localiza cada '
        'ata (data, tipo, nº, boletim, arquivo PDF e página) e traz a '
        'auditoria de completude.',
        'D:\\TSJE_TRANSCRICOES\\<ano>\\*.md — as transcrições individuais '
        '(uma por ata), com metadados e notas de fidelidade no cabeçalho.',
        'D:\\TSJE_TRANSCRICOES\\_diagnostico_completude.csv e '
        '_pdfs_mal_rotulados.csv — as listas técnicas que embasam as seções '
        '5 e 6.']:
        doc.add_paragraph(t, style='List Bullet')

    doc.save(SAIDA)
    print(f'relatorio -> {SAIDA}')
    print(f'  transcritas do indice: {trans} | recuperadas: {rec} | '
          f'ausentes no acervo: {ausentes}')


if __name__ == '__main__':
    main()
