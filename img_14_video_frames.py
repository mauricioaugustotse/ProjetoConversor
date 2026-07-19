# -*- coding: utf-8 -*-
"""img_14: classifica os VÍDEOS de D:\\MÍDIA em foto (pessoal) | tela | tv.

--frames : ffmpeg extrai 2 frames por vídeo (20%% e 60%% da duração, 512px)
           para o cache D:\\MÍDIA\\_frames_cache\\<sha1_16>_{1,2}.jpg.
           Roda em qualquer python (sem torch); ThreadPool; retomável.
--clip   : CLIP ViT-B/32 zero-shot nos frames; classe do vídeo = frame de
           maior confiança; grava em classes (metodo='clip_frame').
           Rodar com o python de .venv-clip. Retomável (pula quem tem classe).

Heurística prévia (nome de screen recording) -> tela conf 1.0, sem frame.
Vídeo sem frame extraível -> classes (foto, 0.0, 'frame_erro') = fica no lugar.
"""
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from img_lib import EXT_VID, conectar, lp

FFBIN = (r'C:\Users\mauri\AppData\Local\Microsoft\WinGet\Packages'
         r'\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe'
         r'\ffmpeg-8.1.2-full_build\bin')
CACHE = r'D:\MÍDIA\_frames_cache'
RE_SCREENREC = re.compile(r'screen[_ -]?record|screenrec|gravar[_ -]?tela|'
                          r'record[_ -]?screen|scrcpy', re.I)

PROMPTS = {
    'foto': ['a home video frame of people or family', 'a video frame of a party with people',
             'a personal video of a place or landscape', 'a video frame of children playing',
             'a selfie video frame of a person talking to the camera'],
    'tela': ['a screen recording of a smartphone app', 'a screen recording of a computer desktop',
             'a screenshot of a text message conversation', 'a video frame of a software interface'],
    'tv': ['a frame of a television broadcast with a TV channel logo on screen',
           'a TV news program with a news anchor at a desk',
           'a soccer match broadcast on television with score overlay',
           'a TV show or telenovela scene recorded from television',
           'a music show or interview program on television'],
}
CLASSES = list(PROMPTS)


def em_lista(vals):
    return ','.join(f"'{v}'" for v in sorted(vals))


def pendentes(con):
    return con.execute(f"""
        SELECT m.path, m.destino, a.sha1 FROM movidos m
        JOIN arquivos a ON a.path = m.path
        WHERE a.ext IN ({em_lista(EXT_VID)})
          AND m.path NOT IN (SELECT path FROM classes)
        ORDER BY m.path""").fetchall()


def frames_de(sha1):
    base = os.path.join(CACHE, sha1[:16])
    return [base + '_1.jpg', base + '_2.jpg']


def ff(args, timeout):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def extrair_um(item):
    path, destino, sha1 = item
    alvos = frames_de(sha1)
    if all(os.path.exists(a) for a in alvos):
        return path, 'ok'
    try:
        r = ff([os.path.join(FFBIN, 'ffprobe.exe'), '-v', 'error',
                '-show_entries', 'format=duration', '-of', 'csv=p=0',
                lp(destino)], 60)
        dur = float(r.stdout.strip() or 0)
    except (subprocess.TimeoutExpired, ValueError):
        dur = 0
    if dur <= 0:
        dur = 2.0  # tenta frames no início mesmo assim
    oks = 0
    for alvo, frac in zip(alvos, (0.2, 0.6)):
        try:
            ff([os.path.join(FFBIN, 'ffmpeg.exe'), '-v', 'error',
                '-ss', f'{dur * frac:.2f}', '-i', lp(destino),
                '-frames:v', '1', '-vf', 'scale=512:-2', '-q:v', '5',
                '-y', lp(alvo)], 120)
            if os.path.exists(alvo) and os.path.getsize(alvo) > 0:
                oks += 1
        except subprocess.TimeoutExpired:
            pass
    return path, 'ok' if oks else 'erro'


def modo_frames():
    con = conectar()
    os.makedirs(CACHE, exist_ok=True)
    pend = []
    heur = []
    for path, destino, sha1 in pendentes(con):
        if RE_SCREENREC.search(os.path.basename(path)):
            heur.append((path, 'tela', 1.0, 'heuristica'))
        elif not os.path.exists(lp(destino)):
            continue  # usuário mexeu; fica para depois
        else:
            pend.append((path, destino, sha1))
    if heur:
        con.executemany('INSERT OR REPLACE INTO classes VALUES(?,?,?,?)', heur)
        con.commit()
        print(f'Heurística (screen recording no nome): {len(heur):,} -> tela')
    print(f'Frames a extrair: {len(pend):,} vídeos', flush=True)
    erros = []
    feitos = 0
    import time
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as tp:
        for path, st in tp.map(extrair_um, pend):
            feitos += 1
            if st == 'erro':
                erros.append((path, 'foto', 0.0, 'frame_erro'))
            if feitos % 250 == 0:
                taxa = feitos / (time.time() - t0)
                print(f'  {feitos:,}/{len(pend):,} ({taxa:.1f}/s, '
                      f'~{(len(pend)-feitos)/taxa/60:.0f} min restantes, '
                      f'erros {len(erros)})', flush=True)
    if erros:
        con.executemany('INSERT OR REPLACE INTO classes VALUES(?,?,?,?)', erros)
        con.commit()
    print(f'Frames prontos: {feitos - len(erros):,} vídeos | sem frame: {len(erros):,}')
    con.close()


def modo_clip():
    import torch
    import open_clip
    from PIL import Image

    con = conectar()
    pend = [(p, s) for p, d, s in pendentes(con)
            if any(os.path.exists(f) for f in frames_de(s))]
    print(f'CLIP frames: {len(pend):,} vídeos', flush=True)
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

    import time
    t0 = time.time()
    BATCH = 32
    for i in range(0, len(pend), BATCH):
        lote = []
        for path, sha1 in pend[i:i + BATCH]:
            tens, melhores = [], []
            for fr in frames_de(sha1):
                if not os.path.exists(fr):
                    continue
                try:
                    with Image.open(fr) as im:
                        tens.append(preprocess(im.convert('RGB')))
                except Exception:
                    pass
            if not tens:
                lote.append((path, 'foto', 0.0, 'frame_erro'))
                continue
            with torch.no_grad():
                emb = model.encode_image(torch.stack(tens))
                emb = emb / emb.norm(dim=-1, keepdim=True)
                probs = (100.0 * emb @ txt.T).softmax(dim=-1)
            for pr in probs:
                k = int(pr.argmax())
                melhores.append((float(pr[k]), CLASSES[k]))
            conf, classe = max(melhores)
            lote.append((path, classe, conf, 'clip_frame'))
        con.executemany('INSERT OR REPLACE INTO classes VALUES(?,?,?,?)', lote)
        con.commit()
        n = min(i + BATCH, len(pend))
        if n % 512 == 0 or n == len(pend):
            taxa = n / (time.time() - t0)
            print(f'  {n:,}/{len(pend):,} ({taxa:.0f}/s)', flush=True)
    res = con.execute("SELECT classe, COUNT(*) FROM classes WHERE metodo IN "
                      "('clip_frame','frame_erro') GROUP BY 1").fetchall()
    print('Vídeos por classe:', dict(res))
    con.close()


def main():
    if '--frames' in sys.argv:
        modo_frames()
    elif '--clip' in sys.argv:
        modo_clip()
    else:
        print('uso: img_14_video_frames.py --frames | --clip')


if __name__ == '__main__':
    main()
