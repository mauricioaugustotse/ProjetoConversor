# -*- coding: utf-8 -*-
r"""acv_04: rodada de reordenacao de C:\Users\mauri\HD_Mau com gpt-5.6-sol.

Passa o acervo INTEIRO (01..05, fora "Julgados e sumulas" e 00-*) pelo sol:
para cada item (pasta atual, nome, trecho do conteudo) o modelo devolve
"manter" ou um destino melhor da whitelist + nome melhor na convencao.

Guard-rails (licoes org2, fases 4/5):
 - so move com confianca >= 0.7 e destino na whitelist;
 - em 01 - Juridico, a MATERIA do digito J do CNJ no nome e autoritativa
   sobre a IA (J: 4/8 Civel, 5 Trabalhista, 6 Eleitoral, 2 Administrativo);
 - rename rejeitado se perder CNJ/numeros do nome atual ou se deslocar o
   prefixo "CLASSE numero" do inicio;
 - "Julgados e sumulas" (colecao deliberada) fora do escopo;
 - enderecamento fino de faixas (ano/alfabeticas) deterministico.

Uso: python acv_04_reordenar_hdmau.py --plano
     python acv_04_reordenar_hdmau.py --exec   (aplica acv_plano_reord.csv)
Depois do --exec: rodar org2_06_indice.py (indice/catalogo) e conferir
caminhos >259 (org2_07).
"""
import csv
import datetime
import io
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from acv_02_cruzar import lp
from acv_03_organizar import (RE_CNJ, MATERIA_J, TIPOS_JUR, ano_do, ascii_seg,
                              chave_openai, endereco_fino, whitelist)
from org2_13_auditoria_conteudo import carregar_cab
from org2_15_nao_processuais import extrair_isolado

ROOT = r'C:\Users\mauri\HD_Mau'
RESP = os.path.join(BASE, 'acv_ia_reord.jsonl')
PLANO = os.path.join(BASE, 'acv_plano_reord.csv')
CAB_EXTRA = os.path.join(BASE, 'acv_cabecalhos_c_extra.tsv')

# --- Taxonomia NOVA do 02 (aprovada 13/07/2026): eixo por NATUREZA ---
# Os destinos abaixo entram na whitelist mesmo antes de existirem no disco
# (criados na execucao). Disciplina vira subnivel de Doutrina/Cursos.
DISCIPLINAS = ['Direito Administrativo', 'Direito Ambiental', 'Direito Civil',
               'Direito Constitucional', 'Direito Eleitoral', 'Direito Empresarial',
               'Direito Internacional', 'Direito Penal', 'Direito Previdenciario',
               'Direito Processual Civil', 'Direito Processual Penal',
               'Direito Tributario', 'Direito do Trabalho']
EIXOS_02 = (
    [r'02 - Estudos e concursos\Concursos e provas',
     r'02 - Estudos e concursos\Legislacao e codigos',
     r'02 - Estudos e concursos\Mestrado e pesquisa',
     r'02 - Estudos e concursos\Instrumentais\Portugues',
     r'02 - Estudos e concursos\Instrumentais\Raciocinio Logico e Matematica',
     r'02 - Estudos e concursos\Instrumentais\Informatica']
    + [rf'02 - Estudos e concursos\Doutrina e artigos\{d}' for d in DISCIPLINAS]
    + [rf'02 - Estudos e concursos\Cursos e aulas\{d}' for d in DISCIPLINAS]
    + [rf'02 - Estudos e concursos\Informativos e jurisprudencia\{d}' for d in DISCIPLINAS]
)
# conversoes 1:1 da estrutura antiga do 02 (sem IA): (regex do rel-dir, template novo)
MAPA_02 = [
    (re.compile(r'^02 - Estudos e concursos\\Direito Eleitoral\\Julgados e sumulas(\\.*)?$'),
     r'02 - Estudos e concursos\Informativos e jurisprudencia\Julgados e sumulas TSE{resto}'),
    (re.compile(r'^02 - Estudos e concursos\\(Direito [^\\]+)\\Julgados e sumulas$'),
     r'02 - Estudos e concursos\Informativos e jurisprudencia\{disc}'),
    (re.compile(r'^02 - Estudos e concursos\\(Direito [^\\]+)\\Doutrina e artigos$'),
     r'02 - Estudos e concursos\Doutrina e artigos\{disc}'),
    (re.compile(r'^02 - Estudos e concursos\\(Direito [^\\]+)\\Cursos e certificados$'),
     r'02 - Estudos e concursos\Cursos e aulas\{disc}'),
    (re.compile(r'^02 - Estudos e concursos\\(Portugues|Raciocinio Logico e Matematica|Informatica)$'),
     r'02 - Estudos e concursos\Instrumentais\{disc}'),
]
# lixo digital do 05 -> Lixeira (aprovado 13/07/2026); padroes ESTRITOS
RE_LIXO_05 = re.compile(
    r'^(Cookies? [\w.\-]|Aviso de compatibilidade|\d{6,} \S+\.txt$|'
    r'.*arquivo temporario)', re.I)
MODELO = 'gpt-5.6-luna'  # decisão de custo 13/07/2026: luna em tudo (~US$20) c/ gate 0.7
GATE = 0.7
MAXPATH = 255
RE_CLASSE_PREFIXO = re.compile(
    r'^((?:REspe?|AgR|AI|MS|HC|AC|RE|Resp|EDcl|Rcl|ADI|ADC|ADPF|CC|Pet|Inq|'
    r'RMS|RHC|TutAntAntec|Cta|PA|ED)\s+[\d.\-]+)', re.I)

SYS = """Voce REVISA a organizacao do acervo pessoal de um servidor do TSE (ex-advogado
dos Correios; o acervo tambem tem documentos da familia - pai Mario, mae Ana Lucia - e da
associacao AEED). O acervo JA FOI organizado; sua tarefa e apontar apenas o que esta em
lugar claramente ERRADO ou com nome que nao indica o conteudo. Na duvida, "manter".
Categorias: 01 - Juridico (pecas processuais; tipo+materia), 02 - Estudos e concursos,
03 - Administrativo, 04 - Pessoal e financeiro, 05 - Diversos.
ATENCAO: o 02 foi REESTRUTURADO por natureza do material (Concursos e provas / Cursos e
aulas / Doutrina e artigos / Legislacao e codigos / Informativos e jurisprudencia /
Mestrado e pesquisa / Instrumentais) - arquivos nas pastas antigas "Direito X" e
"Atualidades e Outros" DEVEM receber um destino novo do 02 (para eles nao use "manter").
Destinos validos (use EXATAMENTE um, ou "manter"):
{DESTINOS}
Convencao de nomes (ASCII, segmentos " - ", minusculas exceto siglas/nomes/inicio):
- 01: "CLASSE numero - UF - assunto - MM-AAAA - CNJ ..."   - 02: "Tema ou titulo - autor - ano"
- 03: "Tipo doc - assunto - orgao ou pessoa - DD-MM-AAAA"  - 04: "Tipo - pessoa - detalhe - data"
Para cada item (pasta atual, nome, trecho):
- "destino": "manter" OU caminho da lista (so quando a pasta atual esta claramente errada);
- "novo_stem": "" OU nome novo na convencao (so quando o atual nao indica o conteudo;
  PRESERVE CNJ, numeros e datas; NAO expanda siglas de classe; max ~110 chars);
- "confianca": 0-1.
Responda APENAS JSON: {{"itens": [{{"id": <n>, "destino": "<manter|caminho>", "novo_stem": "<vazio|nome>", "confianca": <0-1>}}]}}"""


def coletar():
    """Todos os arquivos de 01..05. 'Julgados e sumulas' ENTRA (a realocacao
    1:1 aprovada em 13/07/2026 e feita pelo MAPA_02, nunca pela IA)."""
    alvos = []
    for cat in sorted(os.listdir(ROOT)):
        pc = os.path.join(ROOT, cat)
        if not os.path.isdir(pc) or not re.match(r'^0[1-5] - ', cat):
            continue
        for dp, dn, fn in os.walk(pc):
            for f in fn:
                if f.startswith('00 - ') or f.lower() in ('desktop.ini', 'thumbs.db'):
                    continue
                alvos.append(os.path.join(dp, f))
    return alvos


def mapa_02(rel_dir):
    """Conversao deterministica 1:1 da estrutura antiga do 02. None se nao casa."""
    for rx, template in MAPA_02:
        m = rx.match(rel_dir)
        if not m:
            continue
        if '{resto}' in template:
            return template.replace('{resto}', m.group(1) or '')
        return template.replace('{disc}', m.group(1))
    return None


def whitelist_nova():
    """Whitelist alvo: pastas atuais MENOS a arvore antiga do 02, MAIS os eixos."""
    antigas_02 = re.compile(
        r'^02 - Estudos e concursos\\(Direito |Atualidades|Portugues|'
        r'Raciocinio|Informatica)')
    dests = [d for d in whitelist() if not antigas_02.match(d)]
    return dests + EIXOS_02


def textos_para(alvos):
    cab = {p: t for p, (k, t) in carregar_cab().items()}
    if os.path.exists(CAB_EXTRA):
        for line in io.open(CAB_EXTRA, encoding='utf-8'):
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2:
                cab[parts[0]] = parts[1]
    falta = [p for p in alvos if len(cab.get(p, '')) < 80
             and os.path.splitext(p)[1].lower() in ('.pdf', '.doc', '.docx', '.rtf', '.txt')]
    if falta:
        print(f'Extraindo {len(falta):,} cabecalhos novos (isolado c/ timeout)...',
              flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex, \
                io.open(CAB_EXTRA, 'a', encoding='utf-8') as f:
            for p, t in zip(falta, ex.map(extrair_isolado, falta)):
                t = re.sub(r'\s+', ' ', t or '').strip()[:1200]
                cab[p] = t
                f.write(f'{p}\t{t}\n')
    return cab


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
    print(f'IA {MODELO}: {len(pend):,} pendentes (cache {len(ja):,})', flush=True)
    lotes = [pend[i:i + 12] for i in range(0, len(pend), 12)]

    def um_lote(lote):
        linhas = []
        for i, p in enumerate(lote):
            rel = os.path.dirname(os.path.relpath(p, ROOT))
            linhas.append(f'id={i} | pasta="{rel}" | nome="{os.path.basename(p)}" | '
                          f'trecho="{textos.get(p, "")[:500]}"')
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
                                    'destino': str(it.get('destino', 'manter')),
                                    'novo_stem': str(it.get('novo_stem', '')),
                                    'confianca': float(it.get('confianca', 0))})
                return out
            except Exception:
                time.sleep(2 * (tent + 1))
        return [{'path': p, 'destino': 'manter', 'novo_stem': '', 'confianca': 0}
                for p in lote]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex, \
            io.open(RESP, 'a', encoding='utf-8') as f:
        for i, res in enumerate(ex.map(um_lote, lotes), 1):
            for d in res:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
                ja[d['path']] = d
            if i % 50 == 0:
                feito = min(i * 12, len(pend))
                taxa = feito / (time.time() - t0)
                print(f'  lote {i}/{len(lotes)} (~{taxa*60:.0f} itens/min, '
                      f'~{(len(pend)-feito)/taxa/60:.0f} min restantes)', flush=True)
    return ja


def valida_stem(stem_novo, stem_orig):
    """Rejeita rename que perde CNJ/numeros ou desloca o prefixo de classe."""
    if not stem_novo:
        return ''
    for m in RE_CNJ.finditer(stem_orig):
        if m.group(0) not in stem_novo:
            return ''
    m = RE_CLASSE_PREFIXO.match(stem_orig)
    if m and not stem_novo.lower().startswith(m.group(1).lower()):
        return ''
    return ascii_seg(stem_novo, 110)


def montar_plano():
    alvos = coletar()
    print(f'Arquivos no escopo: {len(alvos):,}')
    dests = whitelist_nova()
    print(f'Whitelist alvo (c/ eixos novos do 02): {len(dests)}')

    # separacao: conversao 1:1 do 02 | lixo do 05 | resto vai a IA
    alvos_mapa, alvos_lixo, alvos_ia = [], [], []
    for p in alvos:
        rel_dir = os.path.dirname(os.path.relpath(p, ROOT))
        novo = mapa_02(rel_dir)
        if novo:
            alvos_mapa.append((p, novo))
        elif rel_dir.startswith('05 - Diversos') and RE_LIXO_05.match(os.path.basename(p)):
            alvos_lixo.append(p)
        else:
            alvos_ia.append(p)
    print(f'Mapa 1:1 do 02: {len(alvos_mapa):,} | lixo 05->Lixeira: '
          f'{len(alvos_lixo):,} | para IA: {len(alvos_ia):,}')

    textos = textos_para(alvos_ia)
    respostas = rodar_ia(alvos_ia, textos, dests)
    dests_set = set(dests)
    re_02_antigo = re.compile(
        r'^02 - Estudos e concursos\\(Direito [^\\]+|Atualidades e Outros|'
        r'Portugues|Raciocinio Logico e Matematica|Informatica)')

    usados = set()
    linhas = []
    stats = Counter()
    for p, novo_rel in alvos_mapa:
        destino = os.path.join(ROOT, novo_rel, os.path.basename(p))
        if destino.lower() not in usados:
            usados.add(destino.lower())
            linhas.append((p, destino, 'mapa02', '1.00'))
            stats['mapa02'] += 1
    for p in alvos_lixo:
        linhas.append((p, '', 'lixeira', '1.00'))
        stats['lixeira'] += 1

    for p in alvos_ia:
        d = respostas.get(p)
        if not d:
            continue
        rel_dir = os.path.dirname(os.path.relpath(p, ROOT))
        do_02_antigo = bool(re_02_antigo.match(rel_dir))
        if do_02_antigo and (d['destino'] == 'manter' or d['destino'] not in dests_set
                             or d['confianca'] < GATE):
            # pasta de origem vai deixar de existir: fallback preserva a disciplina
            m = re.match(r'^02 - Estudos e concursos\\(Direito [^\\]+)', rel_dir)
            d = dict(d)
            d['destino'] = (rf'02 - Estudos e concursos\Doutrina e artigos\{m.group(1)}'
                            if m else r'05 - Diversos\Outros')
            d['confianca'] = max(d['confianca'], GATE)
            stats['fallback_02'] += 1
        # --- pos-filtros anti-achatamento (diagnostico 13/07/2026) ---
        if not do_02_antigo and d['destino'] not in ('', 'manter'):
            dst = d['destino']
            # 1) arquivo ja esta DENTRO do destino apontado (subpasta fina/faixa):
            #    IA confirmou a regiao; achatar destruiria a organizacao do org2_08
            if rel_dir == dst or rel_dir.startswith(dst + '\\'):
                d = dict(d)
                d['destino'] = 'manter'
                stats['ja_dentro'] += 1
            # 2) mesmo cat\tipo\materia (muda so faixa): reequilibrio cuida depois
            elif rel_dir.split('\\')[:3] == dst.split('\\')[:3]:
                d = dict(d)
                d['destino'] = 'manter'
                stats['so_faixa_ignorado'] += 1
            # 3) em 01, mudar de TIPO exige concordancia do NOME (licao org2 fase 4:
            #    acordao dentro de Recursos e conteudo legitimo)
            elif rel_dir.startswith('01 - ') and dst.startswith('01 - '):
                tipo_src = rel_dir.split('\\')[1] if '\\' in rel_dir else ''
                tipo_dst = dst.split('\\')[1] if '\\' in dst else ''
                if tipo_src != tipo_dst:
                    nome_arq = os.path.basename(p)
                    rx = next((r for r, t in TIPOS_JUR if t == tipo_dst), None)
                    if rx is None or not rx.search(nome_arq):
                        d = dict(d)
                        d['destino'] = 'manter'
                        stats['tipo01_nome_nao_concorda'] += 1
        destino_rel = d['destino']
        stem_orig = os.path.splitext(os.path.basename(p))[0]
        ext = os.path.splitext(p)[1].lower()
        novo_stem = valida_stem(d['novo_stem'].strip(), stem_orig)
        move = (destino_rel != 'manter' and destino_rel in dests_set
                and d['confianca'] >= GATE)
        if move and destino_rel.startswith('01 - Juridico'):
            m = RE_CNJ.search(stem_orig)
            if m:
                mat_cnj = MATERIA_J.get(m.group(4), 'Outros')
                partes = destino_rel.split('\\')
                if len(partes) == 3 and partes[2] != mat_cnj:
                    destino_rel = f'{partes[0]}\\{partes[1]}\\{mat_cnj}'
                    stats['materia_corrigida_cnj'] += 1
        renomeia = bool(novo_stem) and d['confianca'] >= GATE
        if not move and not renomeia:
            stats['manter'] += 1
            continue
        pasta_atual_rel = os.path.dirname(os.path.relpath(p, ROOT))
        alvo_rel = destino_rel if move else pasta_atual_rel
        if move:
            alvo_rel = endereco_fino(alvo_rel, novo_stem or stem_orig,
                                     ano_do(stem_orig, textos.get(p, '')))
            if alvo_rel == pasta_atual_rel and not renomeia:
                stats['manter'] += 1
                continue
        stem = novo_stem or ascii_seg(stem_orig, 110)
        pasta = os.path.join(ROOT, alvo_rel)
        maxstem = max(20, MAXPATH - len(pasta) - len(ext) - 6)
        stem = stem[:maxstem].rstrip(' .-')
        destino = os.path.join(pasta, stem + ext)
        if destino == p:
            stats['manter'] += 1
            continue
        n = 1
        while destino.lower() in usados or \
                (destino.lower() != p.lower() and os.path.exists(lp(destino))):
            n += 1
            destino = os.path.join(pasta, f'{stem} ({n}){ext}')
        usados.add(destino.lower())
        stats['move' if move else 'so_rename'] += 1
        linhas.append((p, destino, 'move' if move else 'rename',
                       f"{d['confianca']:.2f}"))

    with open(PLANO, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['origem', 'destino', 'acao', 'confianca'])
        w.writerows(linhas)
    print(f'\nPlano: {len(linhas):,} acoes -> {PLANO}')
    print(f'Stats: {dict(stats)}')


def executar():
    from send2trash import send2trash
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_acv_reord_{ts}.csv')
    ok = err = lixo = 0
    with open(PLANO, encoding='utf-8-sig') as f, \
            open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['origem', 'destino', 'acao', 'status'])
        linhas = list(csv.DictReader(f))
        for i, r in enumerate(linhas, 1):
            src, dst, acao = r['origem'], r['destino'], r['acao']
            status = 'ok'
            try:
                if not os.path.exists(lp(src)):
                    status = 'origem_sumiu'
                elif acao == 'lixeira':
                    send2trash(src)
                    lixo += 1
                elif os.path.exists(lp(dst)):
                    status = 'destino_ocupado'
                else:
                    os.makedirs(lp(os.path.dirname(dst)), exist_ok=True)
                    os.rename(lp(src), lp(dst))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([src, dst, acao, status])
            ok, err = (ok + 1, err) if status == 'ok' else (ok, err + 1)
            if i % 1000 == 0:
                print(f'  {i:,}/{len(linhas):,} (problemas {err})', flush=True)
    # pastas esvaziadas (arvore antiga do 02, faixas orfas) - bottom-up
    vazias = 0
    for dp, dn, fn in os.walk(lp(ROOT), topdown=False):
        if not os.listdir(dp):
            try:
                os.rmdir(dp)
                vazias += 1
            except OSError:
                pass
    print(f'Aplicados: {ok:,} (Lixeira: {lixo:,}) | problemas: {err:,} | '
          f'pastas vazias removidas: {vazias} | log: {log_path}')
    print('Agora rode: python org2_06_indice.py  (regenera indice/catalogo)')


if __name__ == '__main__':
    if '--exec' in sys.argv:
        executar()
    else:
        montar_plano()
