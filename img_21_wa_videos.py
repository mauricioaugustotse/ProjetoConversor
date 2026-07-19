# -*- coding: utf-8 -*-
"""img_21: vídeos de WhatsApp — separa PESSOAL de LIXO ENCAMINHADO.

Universo: vídeos cujo path ORIGINAL tem VID-YYYYMMDD-WA / pasta WhatsApp.
--frames : extrai 2 frames (20%%/60%%) p/ D:\\MÍDIA\\_frames_cache (sha1[:16]),
           compatível com a galeria do img_17. Retomável.
--luna   : gpt-5.6-luna vê os 2 frames de cada vídeo (4 vídeos/chamada):
           pessoal (fica) x corrente/humor/política/futebol/propaganda/
           música-de-terceiros/tela (baixa). Gate: conf<0.55 -> fica.
           Tabela luna_wavid. Retomável.
--plano / --exec : baixa -> 9990 BAIXA RELEVANCIA\\VIDEOS WHATSAPP\\<categoria>.
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from img_lib import DB, EXT_VID, lp

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO = 'gpt-5.6-luna'
CACHE = r'D:\MÍDIA\_frames_cache'
RAIZ = r'D:\MÍDIA\9990 BAIXA RELEVANCIA\VIDEOS WHATSAPP'
FFBIN = (r'C:\Users\mauri\AppData\Local\Microsoft\WinGet\Packages'
         r'\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe'
         r'\ffmpeg-8.1.2-full_build\bin')
RE_WA = re.compile(r'VID-\d{8}-WA\d+|WhatsApp Video|WhatsApp Vídeo', re.I)
VIDS_POR_CHAMADA = 4
THREADS = 4
GATE = 0.55
_lock = threading.Lock()

CAT_BAIXA = {'corrente_religioso', 'humor_meme', 'politica_noticia',
             'futebol_esporte', 'propaganda_promo', 'musica_terceiros',
             'tela_gravacao'}
CATS = CAT_BAIXA | {'pessoal', 'outros'}

SYSTEM = """Você é curador de um acervo FAMILIAR de vídeos. Cada VÍDEO é \
representado por 2 frames consecutivos (frame A e frame B). Decida se é um \
vídeo PESSOAL (gravado pela própria família: pessoas conhecidas em casa, \
festas próprias, filhos, viagens da família, filmado de forma amadora) ou \
CONTEÚDO ENCAMINHADO típico de WhatsApp:

- pessoal: vídeo caseiro da família/amigos (FICA no acervo)
- corrente_religioso: mensagem de bom dia, oração, corrente, texto animado
- humor_meme: vídeo de piada/pegadinha/meme encaminhado, conteúdo viral
- politica_noticia: política, jornal, discurso, manifestação, notícia
- futebol_esporte: jogo/gol/lance transmitido pela TV ou retransmitido
- propaganda_promo: publicidade, promoção, divulgação de produto/serviço
- musica_terceiros: clipe musical profissional, show gravado da TV/YouTube
- tela_gravacao: gravação de tela de celular/computador
- outros: impossível decidir com clareza (FICA)

Na dúvida entre pessoal e encaminhado, escolha pessoal.
Responda APENAS JSON: [{"v": 1, "categoria": "...", "conf": 0.0-1.0}, ...]"""


def conectar_mt():
    con = sqlite3.connect(DB, timeout=60, check_same_thread=False)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute("""CREATE TABLE IF NOT EXISTS luna_wavid(
        path TEXT PRIMARY KEY, veredicto TEXT, categoria TEXT,
        conf REAL, ts TEXT)""")
    con.commit()
    return con


def universo(con):
    """Todos os vídeos do acervo CRONOLÓGICO (qualquer origem) + vídeos com
    nome WA nas coleções da 0000 SEM DATA. Álbuns curados e 9990 ficam fora."""
    vid = ','.join(f"'{e}'" for e in sorted(EXT_VID))
    sd = 'D:\\MÍDIA\\0000 SEM DATA\\'
    baixa = 'D:\\MÍDIA\\9990'
    colecoes = (sd + 'DIVERSOS\\', sd + 'FOTOS POR ANOS\\')
    out = []
    for r in con.execute(f"""
            SELECT m.path, m.destino, a.sha1 FROM movidos m
            JOIN arquivos a ON a.path = m.path
            WHERE a.ext IN ({vid})""").fetchall():
        destino = r[1]
        if destino.startswith(baixa):
            continue
        if not destino.startswith(sd):
            out.append(r)                      # cronológico inteiro
        elif destino.startswith(colecoes) and RE_WA.search(r[0]):
            out.append(r)                      # WA nas coleções
    return out


def frames_de(sha1):
    return [os.path.join(CACHE, sha1[:16] + '_1.jpg'),
            os.path.join(CACHE, sha1[:16] + '_2.jpg')]


def extrair_um(item):
    path, destino, sha1 = item
    alvos = frames_de(sha1)
    if all(os.path.exists(a) for a in alvos):
        return 'ok'
    try:
        r = subprocess.run([os.path.join(FFBIN, 'ffprobe.exe'), '-v', 'error',
                            '-show_entries', 'format=duration', '-of', 'csv=p=0',
                            lp(destino)], capture_output=True, text=True, timeout=60)
        dur = float(r.stdout.strip() or 0)
    except Exception:
        dur = 0
    if dur <= 0:
        dur = 2.0
    ok = 0
    for alvo, frac in zip(alvos, (0.2, 0.6)):
        try:
            subprocess.run([os.path.join(FFBIN, 'ffmpeg.exe'), '-v', 'error',
                            '-ss', f'{dur * frac:.2f}', '-i', lp(destino),
                            '-frames:v', '1', '-vf', 'scale=512:-2', '-q:v', '5',
                            '-y', lp(alvo)], capture_output=True, timeout=120)
            if os.path.exists(alvo) and os.path.getsize(alvo) > 0:
                ok += 1
        except subprocess.TimeoutExpired:
            pass
    return 'ok' if ok else 'erro'


def modo_frames():
    con = conectar_mt()
    os.makedirs(CACHE, exist_ok=True)
    pend = [u for u in universo(con)
            if os.path.exists(lp(u[1])) and not all(os.path.exists(a)
                                                    for a in frames_de(u[2]))]
    print(f'frames a extrair: {len(pend):,} vídeos WA', flush=True)
    t0 = time.time()
    feitos = erros = 0
    with ThreadPoolExecutor(max_workers=THREADS) as tp:
        for st in tp.map(extrair_um, pend):
            feitos += 1
            erros += st == 'erro'
            if feitos % 250 == 0:
                taxa = feitos / (time.time() - t0)
                print(f'  {feitos:,}/{len(pend):,} ({taxa:.1f}/s, '
                      f'~{(len(pend)-feitos)/taxa/60:.0f} min, erros {erros})',
                      flush=True)
    print(f'frames prontos: {feitos - erros:,} | sem frame: {erros}')
    con.close()


def modo_luna():
    from openai import OpenAI
    chave = io.open(os.path.join(BASE, 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)
    con = conectar_mt()
    feitos_ja = {p for (p,) in con.execute('SELECT path FROM luna_wavid')}
    pend = [u for u in universo(con) if u[0] not in feitos_ja
            and any(os.path.exists(a) for a in frames_de(u[2]))]
    n_ch = (len(pend) + VIDS_POR_CHAMADA - 1) // VIDS_POR_CHAMADA
    print(f'luna WA: {len(pend):,} vídeos (~{n_ch:,} chamadas)', flush=True)
    if not pend:
        con.close()
        return

    def b64_de(fr):
        with open(lp(fr), 'rb') as f:
            return base64.b64encode(f.read()).decode()

    def chamar(vids):
        conteudo = []
        for i, (path, destino, sha1) in enumerate(vids, 1):
            for rotulo, fr in zip('AB', frames_de(sha1)):
                if os.path.exists(fr):
                    conteudo.append({'type': 'input_text',
                                     'text': f'Vídeo {i} - frame {rotulo}:'})
                    conteudo.append({'type': 'input_image', 'detail': 'low',
                                     'image_url':
                                     f'data:image/jpeg;base64,{b64_de(fr)}'})
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
                    c = str(d.get('categoria', '')).strip().lower()
                    if c in CATS:
                        out[int(d['v'])] = (c, float(d.get('conf', 0)))
                return out
            except Exception as e:
                ultimo = e
                time.sleep(8 * (tent + 1))
        raise RuntimeError(ultimo)

    lotes = [pend[i:i + VIDS_POR_CHAMADA]
             for i in range(0, len(pend), VIDS_POR_CHAMADA)]
    prog = [0]
    t0 = time.time()

    def processar(lote):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        rows = []
        try:
            resp = chamar(lote)
        except RuntimeError:
            resp = {}
        for i, (path, destino, sha1) in enumerate(lote, 1):
            if i in resp:
                cat, conf = resp[i]
                ver = 'baixa' if cat in CAT_BAIXA and conf >= GATE else 'manter'
                rows.append((path, ver, cat, conf, ts))
            else:
                rows.append((path, 'erro', 'sem_resposta', 0.0, ts))
        with _lock:
            con.executemany('INSERT OR REPLACE INTO luna_wavid VALUES(?,?,?,?,?)',
                            rows)
            con.commit()
            prog[0] += 1
            if prog[0] % 25 == 0 or prog[0] == len(lotes):
                taxa = prog[0] / (time.time() - t0)
                print(f'  lote {prog[0]:,}/{len(lotes):,} ({taxa*60:.0f}/min, '
                      f'~{(len(lotes)-prog[0])/max(taxa,.01)/60:.0f} min)',
                      flush=True)

    with ThreadPoolExecutor(max_workers=THREADS) as tp:
        list(tp.map(processar, lotes))
    print(dict(con.execute("""SELECT veredicto || '/' || categoria, COUNT(*)
        FROM luna_wavid GROUP BY 1 ORDER BY 2 DESC""")))
    con.close()


def montar(con):
    plano = []
    for path, ver, cat in con.execute(
            "SELECT path, veredicto, categoria FROM luna_wavid "
            "WHERE veredicto='baixa'").fetchall():
        r = con.execute('SELECT destino FROM movidos WHERE path=?',
                        (path,)).fetchone()
        if not r or r[0].startswith(RAIZ + os.sep):
            continue
        plano.append((path, r[0],
                      os.path.join(RAIZ, cat, os.path.basename(r[0]))))
    return plano


def modo_plano():
    con = conectar_mt()
    plano = montar(con)
    stats = {}
    for _, _, para in plano:
        k = os.path.basename(os.path.dirname(para))
        stats[k] = stats.get(k, 0) + 1
    print(f'a mover: {len(plano):,} | {stats}')
    con.close()


def modo_exec():
    con = conectar_mt()
    plano = montar(con)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG = os.path.join(BASE, f'log_img_wavid_{ts}.csv')
    ok = err = 0
    with open(LOG, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['de', 'para', 'status'])
        for path, de, para in plano:
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
            w.writerow([de, para, status])
            ok += status == 'ok'
            err += status != 'ok'
    con.commit()
    con.close()
    print(f'movidos: {ok:,} | pendências: {err} | log: {LOG}')


def main():
    if '--frames' in sys.argv:
        modo_frames()
    elif '--luna' in sys.argv:
        modo_luna()
    elif '--plano' in sys.argv:
        modo_plano()
    elif '--exec' in sys.argv:
        modo_exec()
    else:
        print('uso: img_21_wa_videos.py --frames | --luna | --plano | --exec')


if __name__ == '__main__':
    main()
