# -*- coding: utf-8 -*-
"""img_20: caça fotos ESCANEADAS (digitalizações de foto analógica) no acervo
cronológico e leva p/ D:\\MÍDIA\\0000 SEM DATA\\DIVERSOS\\Fotos antigas (scans).

--clip  : (.venv-clip) zero-shot binário scan x digital em todo o cronológico;
          grava tabela scan_scores(path, p_scan). Retomável.
--luna  : gpt-5.6-luna confirma os suspeitos (p_scan >= LIMIAR); tabela
          luna_scan(path, veredicto, conf). Retomável.
--plano / --exec : move veredicto='scan' p/ a casa única; atualiza movidos.
"""
import base64
import csv
import datetime
import io
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from img_lib import DB, EXT_VID, lp

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO = 'gpt-5.6-luna'
CASA = r'D:\MÍDIA\0000 SEM DATA\DIVERSOS\Fotos antigas (scans)'
SD = r'D:\MÍDIA\0000 SEM DATA'
LIMIAR = 0.5
LOTE, THREADS = 8, 4
_lock = threading.Lock()

PROMPTS = {
    'scan': ['a scanned old analog photograph with faded colors and film grain',
             'a photo of a printed photograph with white paper border',
             'an old black and white family photo, scanned print',
             'a digitized film photo print with dust and scratches',
             'an old photo album page with a glued analog photograph'],
    'digital': ['a digital photo taken with a smartphone',
                'a modern digital photograph, sharp and clean',
                'a digital selfie or group photo',
                'a photo taken with a digital camera with natural lighting'],
    'outro': ['a screenshot of a phone or computer screen',
              'a document page or receipt photographed',
              'a meme or image downloaded from the internet'],
}
CLASSES = list(PROMPTS)

SYSTEM = """Você é curador de um acervo fotográfico. Para CADA imagem numerada, \
responda se ela é uma DIGITALIZAÇÃO DE FOTO ANALÓGICA ANTIGA (foto de papel \
escaneada ou fotografada: bordas de papel, granulação de filme, cores desbotadas, \
P&B antigo, página de álbum físico) ou não (foto digital nativa, print, documento).
Responda APENAS JSON: [{"n": 1, "scan": true/false, "conf": 0.0-1.0}, ...]"""


def conectar_mt():
    con = sqlite3.connect(DB, timeout=60, check_same_thread=False)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute("""CREATE TABLE IF NOT EXISTS scan_scores(
        path TEXT PRIMARY KEY, p_scan REAL, ts TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS luna_scan(
        path TEXT PRIMARY KEY, veredicto TEXT, conf REAL, ts TEXT)""")
    con.commit()
    return con


def cronologico(con):
    vid = ','.join(f"'{e}'" for e in sorted(EXT_VID))
    return con.execute(f"""
        SELECT m.path, m.destino FROM movidos m
        JOIN arquivos a ON a.path = m.path
        WHERE m.destino NOT LIKE ? AND a.ext NOT IN ({vid})
        ORDER BY m.path""", (SD + '\\%',)).fetchall()


def modo_clip():
    import torch
    import open_clip

    con = conectar_mt()
    feitos = {p for (p,) in con.execute('SELECT path FROM scan_scores')}
    pend = [(p, d) for p, d in cronologico(con) if p not in feitos]
    print(f'CLIP scan-hunt: {len(pend):,} imagens', flush=True)
    if not pend:
        con.close()
        return
    torch.set_num_threads(os.cpu_count())
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    with torch.no_grad():
        feats = []
        for c in CLASSES:
            e = model.encode_text(tokenizer(PROMPTS[c]))
            e = e / e.norm(dim=-1, keepdim=True)
            feats.append(e.mean(0) / e.mean(0).norm())
        txt = torch.stack(feats)

    def carregar(item):
        path, destino = item
        try:
            with open(lp(destino), 'rb') as f:
                with Image.open(f) as im:
                    im.thumbnail((256, 256))
                    return path, preprocess(im.convert('RGB'))
        except Exception:
            return path, None

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    BATCH = 64
    n = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as tp:
        for i in range(0, len(pend), BATCH):
            chunk = list(tp.map(carregar, pend[i:i + BATCH]))
            ok = [(p, t) for p, t in chunk if t is not None]
            lote = [(p, 0.0, ts) for p, t in chunk if t is None]
            if ok:
                with torch.no_grad():
                    emb = model.encode_image(torch.stack([t for _, t in ok]))
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    probs = (100.0 * emb @ txt.T).softmax(dim=-1)
                for (p, _), pr in zip(ok, probs):
                    lote.append((p, float(pr[0]), ts))
            con.executemany('INSERT OR REPLACE INTO scan_scores VALUES(?,?,?)', lote)
            con.commit()
            n += len(chunk)
            if n % 2048 == 0 or n == len(pend):
                taxa = n / (time.time() - t0)
                print(f'  {n:,}/{len(pend):,} ({taxa:.0f}/s, '
                      f'~{(len(pend)-n)/taxa/60:.0f} min restantes)', flush=True)
    tot = con.execute('SELECT COUNT(*) FROM scan_scores WHERE p_scan >= ?',
                      (LIMIAR,)).fetchone()[0]
    print(f'suspeitos (p_scan >= {LIMIAR}): {tot:,}')
    con.close()


def modo_luna():
    from openai import OpenAI
    chave = io.open(os.path.join(BASE, 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)
    con = conectar_mt()
    pend = con.execute("""
        SELECT s.path, m.destino FROM scan_scores s
        JOIN movidos m ON m.path = s.path
        WHERE s.p_scan >= ? AND s.path NOT IN (SELECT path FROM luna_scan)
        ORDER BY s.path""", (LIMIAR,)).fetchall()
    print(f'luna scan-hunt: {len(pend):,} suspeitos '
          f'(~{(len(pend)+LOTE-1)//LOTE:,} chamadas)', flush=True)
    if not pend:
        con.close()
        return

    def thumb(destino):
        try:
            with open(lp(destino), 'rb') as f:
                with Image.open(f) as im:
                    im.thumbnail((512, 512))
                    buf = io.BytesIO()
                    im.convert('RGB').save(buf, 'JPEG', quality=70)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    def chamar(lote_b64):
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
                return {int(d['n']): (bool(d.get('scan')), float(d.get('conf', 0)))
                        for d in dados}
            except Exception as e:
                ultimo = e
                time.sleep(8 * (tent + 1))
        raise RuntimeError(ultimo)

    lotes = [pend[i:i + LOTE] for i in range(0, len(pend), LOTE)]
    feitos = [0]
    t0 = time.time()

    def processar(lote):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        itens, rows = [], []
        for path, destino in lote:
            b64 = thumb(destino)
            if b64 is None:
                rows.append((path, 'erro', 0.0, ts))
            else:
                itens.append((path, b64))
        if itens:
            try:
                resp = chamar([b for _, b in itens])
            except RuntimeError:
                resp = {}
            for i, (path, _) in enumerate(itens, 1):
                if i in resp:
                    eh, conf = resp[i]
                    rows.append((path, 'scan' if eh and conf >= 0.5 else 'nao',
                                 conf, ts))
                else:
                    rows.append((path, 'erro', 0.0, ts))
        with _lock:
            con.executemany('INSERT OR REPLACE INTO luna_scan VALUES(?,?,?,?)', rows)
            con.commit()
            feitos[0] += 1
            if feitos[0] % 25 == 0 or feitos[0] == len(lotes):
                taxa = feitos[0] / (time.time() - t0)
                print(f'  lote {feitos[0]}/{len(lotes)} ({taxa*60:.0f}/min)',
                      flush=True)

    with ThreadPoolExecutor(max_workers=THREADS) as tp:
        list(tp.map(processar, lotes))
    print(dict(con.execute('SELECT veredicto, COUNT(*) FROM luna_scan GROUP BY 1')))
    con.close()


def montar(con):
    plano = []
    for path, destino in con.execute("""
            SELECT l.path, m.destino FROM luna_scan l
            JOIN movidos m ON m.path = l.path
            WHERE l.veredicto = 'scan'""").fetchall():
        if destino.startswith(CASA + os.sep):
            continue
        plano.append((path, destino, os.path.join(CASA, os.path.basename(destino))))
    return plano


def modo_plano():
    con = conectar_mt()
    plano = montar(con)
    print(f'a mover p/ Fotos antigas (scans): {len(plano):,}')
    for _, de, _ in plano[:20]:
        print('  ', de)
    con.close()


def modo_exec():
    con = conectar_mt()
    plano = montar(con)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG = os.path.join(BASE, f'log_img_scanhunt_{ts}.csv')
    ok = err = 0
    os.makedirs(lp(CASA), exist_ok=True)
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
    if '--clip' in sys.argv:
        modo_clip()
    elif '--luna' in sys.argv:
        modo_luna()
    elif '--plano' in sys.argv:
        modo_plano()
    elif '--exec' in sys.argv:
        modo_exec()
    else:
        print('uso: img_20_scan_hunt.py --clip | --luna | --plano | --exec')


if __name__ == '__main__':
    main()
