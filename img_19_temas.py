# -*- coding: utf-8 -*-
"""img_19: organização final de EVENTOS / FOTOS AVULSAS / Sem ano.

--meta-plano / --meta-exec : camada de METADADOS (sem IA):
    Casamento do Vinícius -> subpastas pelo álbum de origem (Dia da noiva,
    Dia do noivo, Jardim, Buffet, Balada);
    Sem ano: álbuns semânticos -> subpastas (Cancun, Fotos da Ana,
    Família - Diversas, Acervo do Mário); Celular 2017 -> FOTOS AVULSAS\\2017;
    Photos from 2024 -> FOTOS AVULSAS\\2024.
--rotular [--limite N] : gpt-5.6-luna rotula por TEMA os restantes de
    Sem ano + FOTOS AVULSAS (tabela luna_tema; vídeo = frame ffmpeg).
--plano / --exec : movimentação por tema:
    Sem ano -> subpastas temáticas; AVULSAS\\2013 casamento -> EVENTOS\\
    Casamento do Vinícius\\Avulsas da festa; demais anos: tema com >=15 no
    ano vira subpasta do ano.
"""
import base64
import csv
import datetime
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from openai import OpenAI

from img_lib import DB, EXT_VID, lp

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO = 'gpt-5.6-luna'
SD = r'D:\MÍDIA\0000 SEM DATA'
SEM_ANO = os.path.join(SD, 'Sem ano')
AVULSAS = os.path.join(SD, 'FOTOS AVULSAS')
CASAMENTO = os.path.join(SD, 'EVENTOS', 'Casamento do Vinícius')
FFBIN = (r'C:\Users\mauri\AppData\Local\Microsoft\WinGet\Packages'
         r'\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe'
         r'\ffmpeg-8.1.2-full_build\bin')
LOTE = 8
THREADS = 4
_lock = threading.Lock()

ALBUM_CASAMENTO = {'dia da noiva': 'Dia da noiva', 'dia do noivo': 'Dia do noivo',
                   'jardim': 'Jardim', 'buffet': 'Buffet', 'balada': 'Balada'}
ALBUM_SEMANO = {
    'Álbum 2 - Cancun': ('Sem ano', 'Cancun'),
    '02_Fotos da Ana': ('Sem ano', 'Fotos da Ana'),
    'Família - Diversas': ('Sem ano', 'Família - Diversas'),
    'BACKUP MÁRIO 2020': ('Sem ano', 'Acervo do Mário'),
    'Celular 2017': ('FOTOS AVULSAS', '2017'),
    'Photos from 2024': ('FOTOS AVULSAS', '2024'),
}

TEMAS = {
    'pessoas_retratos': 'Pessoas e retratos',
    'festas_eventos': 'Festas e eventos',
    'casamento_festa': 'Festas e eventos',   # exceto 2013 (vai p/ o evento)
    'viagens_paisagens': 'Viagens e paisagens',
    'militar_cerimonias': 'Militar e cerimônias',
    'casa_obras': 'Casa e obras',
    'carros_veiculos': 'Carros e veículos',
    'animais': 'Animais',
    'foto_antiga_scan': 'Fotos antigas (scans)',
    'outros': 'Outros',
}
MIN_TEMA_ANO = 15

SYSTEM = """Você é o curador de um acervo familiar brasileiro. Receberá N imagens \
numeradas (algumas são frames de vídeo). Classifique CADA UMA em exatamente um TEMA:

- pessoas_retratos: retratos, selfies, pessoas posando, grupos de pessoas no dia a dia
- festas_eventos: aniversários, confraternizações, churrascos, formaturas, festas em geral
- casamento_festa: festa de CASAMENTO (noiva de vestido, buquê, bolo de casamento, igreja, dança dos noivos)
- viagens_paisagens: praias, cidades turísticas, monumentos, paisagens, hotéis
- militar_cerimonias: militares fardados, quartéis, cerimônias e solenidades militares ou oficiais
- casa_obras: imóveis, reformas, obras, jardins, móveis, ambientes da casa
- carros_veiculos: carros, motos e outros veículos como assunto principal
- animais: animais de estimação ou outros animais como assunto principal
- foto_antiga_scan: digitalização de foto ANALÓGICA antiga (papel, bordas, granulação de filme, preto e branco ou cores desbotadas)
- outros: nada acima se aplica com clareza

Responda APENAS JSON: [{"n": 1, "tema": "...", "conf": 0.0-1.0}, ...]"""


def conectar_mt():
    con = sqlite3.connect(DB, timeout=60, check_same_thread=False)
    con.execute('PRAGMA journal_mode=WAL')
    return con


def alvo_rotular(con):
    """(path, destino, ext) de Sem ano + FOTOS AVULSAS ainda sem tema."""
    return con.execute("""
        SELECT m.path, m.destino, a.ext FROM movidos m
        JOIN arquivos a ON a.path = m.path
        WHERE (m.destino LIKE ? OR m.destino LIKE ?)
          AND m.path NOT IN (SELECT path FROM luna_tema)
        ORDER BY m.path""", (SEM_ANO + '\\%', AVULSAS + '\\%')).fetchall()


def thumb_b64(destino, ext):
    if ext in EXT_VID:
        tmp = os.path.join(tempfile.gettempdir(), f'img19_{os.getpid()}_'
                           f'{threading.get_ident()}.jpg')
        try:
            subprocess.run([os.path.join(FFBIN, 'ffmpeg.exe'), '-v', 'error',
                            '-ss', '3', '-i', lp(destino), '-frames:v', '1',
                            '-vf', 'scale=512:-2', '-q:v', '5', '-y', tmp],
                           capture_output=True, timeout=120)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                with open(tmp, 'rb') as f:
                    return base64.b64encode(f.read()).decode()
        except subprocess.TimeoutExpired:
            pass
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return None
    try:
        with open(lp(destino), 'rb') as f:
            with Image.open(f) as im:
                im.thumbnail((512, 512))
                buf = io.BytesIO()
                im.convert('RGB').save(buf, 'JPEG', quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def chamar(cli, lote_b64):
    conteudo = []
    for i, b64 in enumerate(lote_b64, 1):
        conteudo.append({'type': 'input_text', 'text': f'Imagem {i}:'})
        conteudo.append({'type': 'input_image', 'detail': 'low',
                         'image_url': f'data:image/jpeg;base64,{b64}'})
    ultimo = None
    for tent in range(3):
        try:
            resp = cli.responses.create(
                model=MODELO,
                input=[{'role': 'system', 'content': SYSTEM},
                       {'role': 'user', 'content': conteudo}],
                max_output_tokens=8000)
            texto = (resp.output_text or '').strip()
            m = re.search(r'\[.*\]', texto, re.S)
            dados = json.loads(m.group(0) if m else texto)
            out = {}
            for d in dados:
                t = str(d.get('tema', '')).strip().lower()
                if t in TEMAS:
                    out[int(d['n'])] = (t, float(d.get('conf', 0)))
            return out
        except Exception as e:
            ultimo = e
            time.sleep(8 * (tent + 1))
    raise RuntimeError(f'luna falhou: {ultimo}')


def modo_rotular():
    limite = 0
    if '--limite' in sys.argv:
        limite = int(sys.argv[sys.argv.index('--limite') + 1])
    chave = io.open(os.path.join(BASE, 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)
    con = conectar_mt()
    con.execute("""CREATE TABLE IF NOT EXISTS luna_tema(
        path TEXT PRIMARY KEY, tema TEXT, conf REAL, ts TEXT)""")
    con.commit()
    pend = alvo_rotular(con)
    if limite:
        pend = pend[:limite]
    print(f'a rotular: {len(pend):,} | chamadas: ~{(len(pend)+LOTE-1)//LOTE:,}',
          flush=True)
    if not pend:
        con.close()
        return
    lotes = [pend[i:i + LOTE] for i in range(0, len(pend), LOTE)]
    feitos = [0]
    t0 = time.time()

    def processar(lote):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        itens, rows = [], []
        for path, destino, ext in lote:
            b64 = thumb_b64(destino, ext)
            if b64 is None:
                rows.append((path, 'erro', 0.0, ts))
            else:
                itens.append((path, b64))
        if itens:
            try:
                resp = chamar(cli, [b for _, b in itens])
            except RuntimeError:
                resp = {}
            for i, (path, _) in enumerate(itens, 1):
                if i in resp:
                    rows.append((path, resp[i][0], resp[i][1], ts))
                else:
                    rows.append((path, 'erro', 0.0, ts))
        with _lock:
            con.executemany('INSERT OR REPLACE INTO luna_tema VALUES(?,?,?,?)',
                            rows)
            con.commit()
            feitos[0] += 1
            if feitos[0] % 25 == 0 or feitos[0] == len(lotes):
                taxa = feitos[0] / (time.time() - t0)
                print(f'  lote {feitos[0]:,}/{len(lotes):,} '
                      f'({taxa*60:.0f}/min, ~{(len(lotes)-feitos[0])/max(taxa,.01)/60:.0f} '
                      f'min restantes)', flush=True)

    with ThreadPoolExecutor(max_workers=THREADS) as tp:
        list(tp.map(processar, lotes))
    det = con.execute('SELECT tema, COUNT(*) FROM luna_tema GROUP BY 1 '
                      'ORDER BY 2 DESC').fetchall()
    print('temas:', dict(det))
    con.close()


def montar_meta(con):
    plano = []
    album_de = dict(con.execute(
        'SELECT m.path, a.album FROM movidos m JOIN arquivos a ON a.path=m.path'))
    for path, destino in con.execute(
            'SELECT path, destino FROM movidos WHERE destino LIKE ?',
            (CASAMENTO + '\\%',)).fetchall():
        if os.path.dirname(destino) != CASAMENTO:
            continue  # já em subpasta
        sub = ALBUM_CASAMENTO.get((album_de.get(path) or '').lower())
        if sub:
            plano.append((path, destino,
                          os.path.join(CASAMENTO, sub, os.path.basename(destino)),
                          f'album:{sub}'))
    for path, destino in con.execute(
            'SELECT path, destino FROM movidos WHERE destino LIKE ?',
            (SEM_ANO + '\\%',)).fetchall():
        if os.path.dirname(destino) != SEM_ANO:
            continue
        regra = ALBUM_SEMANO.get(album_de.get(path) or '')
        if regra:
            raiz, sub = regra
            base_dir = SEM_ANO if raiz == 'Sem ano' else AVULSAS
            plano.append((path, destino,
                          os.path.join(base_dir, sub, os.path.basename(destino)),
                          f'album:{sub}'))
    return plano


def montar_temas(con):
    plano = []
    rows = con.execute("""
        SELECT m.path, m.destino, t.tema FROM movidos m
        JOIN luna_tema t ON t.path = m.path
        WHERE t.tema NOT IN ('erro', 'outros')""").fetchall()
    # contagem por (pasta-ano, tema) p/ gate de massa nas AVULSAS
    conta = {}
    for _, destino, tema in rows:
        pai = os.path.dirname(destino)
        if pai.startswith(AVULSAS + os.sep):
            conta[(pai, TEMAS[tema])] = conta.get((pai, TEMAS[tema]), 0) + 1
    for path, destino, tema in rows:
        pai = os.path.dirname(destino)
        nome = os.path.basename(destino)
        legivel = TEMAS[tema]
        if pai == SEM_ANO:
            plano.append((path, destino,
                          os.path.join(SEM_ANO, legivel, nome), f'tema:{tema}'))
        elif pai.startswith(AVULSAS + os.sep) and os.path.dirname(pai) == AVULSAS:
            ano = os.path.basename(pai)
            if tema == 'casamento_festa' and ano == '2013':
                plano.append((path, destino,
                              os.path.join(CASAMENTO, 'Avulsas da festa', nome),
                              'tema:casamento_2013'))
            elif conta.get((pai, legivel), 0) >= MIN_TEMA_ANO:
                plano.append((path, destino,
                              os.path.join(pai, legivel, nome), f'tema:{tema}'))
    return plano


def executar(plano, rotulo):
    con = conectar_mt()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(BASE, f'log_img_temas_{rotulo}_{ts}.csv')
    ok = err = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['de', 'para', 'regra', 'status'])
        for path, de, para, regra in plano:
            status = 'ok'
            try:
                if not os.path.exists(lp(de)):
                    status = 'origem_sumiu'
                elif os.path.exists(lp(para)):
                    status = 'destino_ocupado'
                else:
                    os.makedirs(lp(os.path.dirname(para)), exist_ok=True)
                    os.rename(lp(de), lp(para))
                    con.execute('UPDATE movidos SET destino=? WHERE path=?',
                                (para, path))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([de, para, regra, status])
            ok += status == 'ok'
            err += status != 'ok'
    con.commit()
    con.close()
    print(f'movidos: {ok:,} | pendências: {err:,} | log: {log_path}')


def resumo(plano):
    stats = {}
    for _, _, para, _ in plano:
        k = os.path.relpath(os.path.dirname(para), SD)
        stats[k] = stats.get(k, 0) + 1
    for k, n in sorted(stats.items(), key=lambda x: -x[1]):
        print(f'  {k:55} {n:,}')


def main():
    con = conectar_mt()
    con.execute("""CREATE TABLE IF NOT EXISTS luna_tema(
        path TEXT PRIMARY KEY, tema TEXT, conf REAL, ts TEXT)""")
    con.commit()
    if '--meta-plano' in sys.argv:
        plano = montar_meta(con)
        print(f'camada metadados: {len(plano):,} a mover')
        resumo(plano)
    elif '--meta-exec' in sys.argv:
        plano = montar_meta(con)
        con.close()
        executar(plano, 'meta')
        return
    elif '--rotular' in sys.argv:
        con.close()
        modo_rotular()
        return
    elif '--plano' in sys.argv:
        plano = montar_temas(con)
        print(f'camada temas: {len(plano):,} a mover')
        resumo(plano)
    elif '--exec' in sys.argv:
        plano = montar_temas(con)
        con.close()
        executar(plano, 'temas')
        return
    else:
        print('uso: img_19_temas.py --meta-plano|--meta-exec|--rotular'
              '|--plano|--exec')
    con.close()


if __name__ == '__main__':
    main()
