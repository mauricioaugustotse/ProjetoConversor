# -*- coding: utf-8 -*-
"""img_07: classifica as imagens de D:\\IMAGENS em foto | tela | documento | web.

Camada 1 (--heuristica): metadados do catálogo — nome de screenshot/captura,
resolução exata de tela sem câmera, nomes de download/meme. Confiança 1.0.
Camada 2 (--clip [N]): CLIP ViT-B/32 local (zero-shot, nada sai da máquina)
nas imagens restantes; N limita a rodada (piloto). Retomável: pula quem já
está na tabela classes. Rodar com o python de .venv-clip.

A tabela classes guarda o path ORIGINAL (chave de movidos); o arquivo é lido
do destino atual em D:\\IMAGENS.
"""
import os
import re
import sys

from img_lib import EXT_VID, conectar, lp

FILTRO_VID = "AND a.ext NOT IN (%s)" % ','.join(f"'{e}'" for e in sorted(EXT_VID))

TELAS_RES = {(1080, 1920), (1920, 1080), (720, 1280), (1280, 720), (1440, 2560),
             (2560, 1440), (768, 1366), (1366, 768), (1080, 2340), (2340, 1080),
             (1080, 2400), (2400, 1080), (1080, 2220), (2220, 1080), (1440, 3200),
             (3200, 1440), (750, 1334), (1334, 750), (828, 1792), (1792, 828),
             (1170, 2532), (2532, 1170), (600, 1024), (1024, 600), (800, 1280),
             (1280, 800), (480, 800), (800, 480), (1536, 2048), (2048, 1536)}
RE_SS = re.compile(r'screenshot|captura\s*de\s*tela|capturadetela|print\s*screen|screen\s*shot', re.I)
RE_WEB = re.compile(r'^(images?\s*\(\d+\)|download|unnamed|received_|fb_img|giphy|meme|sticker|wallpaper)', re.I)

# prompts por classe (média dos embeddings de texto de cada grupo)
PROMPTS = {
    'foto': ['a photo of people', 'a photo of a landscape or place', 'a family photo',
             'a photo of animals', 'a photo of food on a table', 'a selfie',
             'a photo of a building or a room'],
    'tela': ['a screenshot of a smartphone app', 'a screenshot of a computer screen',
             'a screenshot of a text message conversation', 'a screenshot of a website',
             'a screenshot of a video call'],
    'documento': ['a photo of a paper document with text', 'a scanned document page',
                  'a photo of an identity card or certificate',
                  'a photocopy of a printed page', 'a photo of a receipt or invoice'],
    'web': ['a meme with caption text', 'a cartoon or comic strip image',
            'a promotional flyer or advertisement graphic',
            'a greeting card image with decorative text and flowers'],
}
CLASSES = list(PROMPTS)


def garantir_tabela(con):
    con.execute("""CREATE TABLE IF NOT EXISTS classes(
        path TEXT PRIMARY KEY, classe TEXT, conf REAL, metodo TEXT)""")


def heuristica():
    con = conectar()
    garantir_tabela(con)
    rows = con.execute("""
        SELECT a.path, a.width, a.height, a.exif_make, a.exif_model
        FROM movidos m JOIN arquivos a ON a.path = m.path
        WHERE a.path NOT IN (SELECT path FROM classes) """ + FILTRO_VID).fetchall()
    lote = []
    for path, w, h, make, model in rows:
        nome = path.split('\\')[-1]
        cam = bool(make or model)
        if RE_SS.search(nome) or (not cam and w and h and (w, h) in TELAS_RES):
            lote.append((path, 'tela', 1.0, 'heuristica'))
        elif RE_WEB.search(nome):
            lote.append((path, 'web', 1.0, 'heuristica'))
    con.executemany('INSERT OR REPLACE INTO classes VALUES(?,?,?,?)', lote)
    con.commit()
    n = {c: sum(1 for x in lote if x[1] == c) for c in ('tela', 'web')}
    print(f"Heurística: {len(lote):,} classificadas (tela {n['tela']:,}, web {n['web']:,})")
    con.close()


def clip_rodar(limite=None):
    import torch
    import open_clip
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    con = conectar()
    garantir_tabela(con)
    pend = con.execute("""
        SELECT m.path, m.destino FROM movidos m
        JOIN arquivos a ON a.path = m.path
        WHERE m.path NOT IN (SELECT path FROM classes) """ + FILTRO_VID + """
        ORDER BY m.path""").fetchall()
    if limite:
        import random
        random.seed(11)
        pend = random.sample(pend, min(limite, len(pend)))
    print(f'CLIP: {len(pend):,} imagens a classificar', flush=True)
    if not pend:
        return

    torch.set_num_threads(os.cpu_count())
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    with torch.no_grad():
        feats = []
        for c in CLASSES:
            t = tokenizer(PROMPTS[c])
            e = model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
            feats.append(e.mean(0) / e.mean(0).norm())
        txt = torch.stack(feats)  # [4, 512]

    def carregar(item):
        path, destino = item
        try:
            with open(lp(destino), 'rb') as f:
                with Image.open(f) as im:
                    im.thumbnail((256, 256))
                    return path, preprocess(im.convert('RGB'))
        except Exception:
            return path, None

    BATCH = 64
    feitos = 0
    import time
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as tp:
        for i in range(0, len(pend), BATCH):
            chunk = list(tp.map(carregar, pend[i:i + BATCH]))
            ok = [(p, t) for p, t in chunk if t is not None]
            lote = [(p, 'foto', 0.0, 'clip_erro') for p, t in chunk if t is None]
            if ok:
                with torch.no_grad():
                    imgs = torch.stack([t for _, t in ok])
                    emb = model.encode_image(imgs)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    probs = (100.0 * emb @ txt.T).softmax(dim=-1)
                for (p, _), pr in zip(ok, probs):
                    k = int(pr.argmax())
                    lote.append((p, CLASSES[k], float(pr[k]), 'clip'))
            con.executemany('INSERT OR REPLACE INTO classes VALUES(?,?,?,?)', lote)
            con.commit()
            feitos += len(chunk)
            if feitos % 1024 == 0 or feitos == len(pend):
                taxa = feitos / (time.time() - t0)
                print(f'  {feitos:,}/{len(pend):,} ({taxa:.0f}/s, '
                      f'~{(len(pend)-feitos)/taxa/60:.0f} min restantes)', flush=True)
    con.close()


def main():
    args = sys.argv[1:]
    if args and args[0] == '--heuristica':
        heuristica()
    elif args and args[0] == '--clip':
        clip_rodar(int(args[1]) if len(args) > 1 else None)
    else:
        print('uso: img_07_classificar.py --heuristica | --clip [N]')


if __name__ == '__main__':
    main()
