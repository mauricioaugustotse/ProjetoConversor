# -*- coding: utf-8 -*-
"""img_15: validador visual — gpt-5.6-luna arbitra os candidatos a baixa
relevância (tudo que o CLIP/heurística marcou como não-foto: tela, documento,
web, tv), em lotes de 8 thumbnails por chamada.

Categorias devolvidas:
  baixa  -> tela | recibo_boleto | cartao_convite | meme_web | tv_gravacao |
            documento_trivial
  manter -> documento_importante | foto_pessoal   (gate anti-perda)
Conf < 0.5 -> manter (conservador). Resultado na tabela
luna_valida(path PK, veredicto, categoria, conf, ts); retomável.

Uso: python img_15_luna.py [--limite N] [--refazer-erros]
"""
import base64
import datetime
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from openai import OpenAI

import sqlite3

from img_lib import DB, EXT_VID, lp


def conectar_mt():
    """Conexão utilizável pelas worker threads (gravações sob _lock)."""
    con = sqlite3.connect(DB, timeout=60, check_same_thread=False)
    con.execute('PRAGMA journal_mode=WAL')
    return con

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO = 'gpt-5.6-luna'
CACHE_FRAMES = r'D:\MÍDIA\_frames_cache'
LOTE = 8
THREADS = 4
CAT_BAIXA = {'tela', 'recibo_boleto', 'cartao_convite', 'meme_web',
             'tv_gravacao', 'documento_trivial'}
CAT_MANTER = {'documento_importante', 'foto_pessoal'}
CATS = CAT_BAIXA | CAT_MANTER
_lock = threading.Lock()

SYSTEM = """Você é o curador de um acervo FAMILIAR de fotos e vídeos. Sua tarefa \
é separar mídia de BAIXA RELEVÂNCIA (sem valor afetivo ou documental duradouro) \
da mídia que deve permanecer no acervo. Receberá N imagens numeradas (algumas \
são frames de vídeo). Classifique CADA UMA em exatamente uma categoria:

BAIXA RELEVÂNCIA (sai do acervo principal):
- tela: print/captura de tela de celular ou computador, conversas de WhatsApp
- recibo_boleto: recibo, boleto, comprovante de pagamento/pix, cupom fiscal, fatura
- cartao_convite: cartão virtual (bom dia, feliz aniversário, datas festivas), correntes ilustradas
- meme_web: meme, imagem baixada da internet, propaganda, papel de parede, sticker
- tv_gravacao: foto ou vídeo DE tela de televisão / programa de TV / transmissão
- documento_trivial: página de texto ou documento fotografado SEM valor pessoal (manual, panfleto, cardápio, etiqueta)

MANTER (fica no acervo):
- documento_importante: RG, CPF, CNH, passaporte, certidão, diploma, contrato, exame médico, foto 3x4, documento assinado, cartão bancário
- foto_pessoal: fotografia ou vídeo genuíno de pessoas, família, lugares, eventos, animais, objetos pessoais

Na dúvida entre baixa relevância e manter, escolha MANTER.
Responda APENAS JSON: [{"n": 1, "categoria": "...", "conf": 0.0-1.0}, ...] com um item por imagem."""


def thumb_b64(item):
    """JPEG 512px base64 do arquivo (foto) ou do frame de cache (vídeo)."""
    destino, sha1, ext = item
    if ext in EXT_VID:
        for suf in ('_1.jpg', '_2.jpg'):
            fr = os.path.join(CACHE_FRAMES, sha1[:16] + suf)
            if os.path.exists(fr):
                with open(fr, 'rb') as f:
                    return base64.b64encode(f.read()).decode()
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
                cat = str(d.get('categoria', '')).strip().lower()
                if cat in CATS:
                    out[int(d['n'])] = (cat, float(d.get('conf', 0)))
            return out
        except Exception as e:
            ultimo = e
            time.sleep(8 * (tent + 1))
    raise RuntimeError(f'luna falhou: {ultimo}')


def fila(con, refazer_erros, foto_conf_max=None):
    extra = "" if not refazer_erros else \
        " OR c.path IN (SELECT path FROM luna_valida WHERE veredicto='erro')"
    if refazer_erros:
        con.execute("DELETE FROM luna_valida WHERE veredicto='erro'")
        con.commit()
    if foto_conf_max is not None:
        # zona cega: o que o CLIP chamou de 'foto' com pouca confiança
        cond = "c.classe = 'foto' AND c.metodo = 'clip' AND c.conf < %f" % foto_conf_max
    else:
        cond = "c.classe != 'foto' AND c.metodo NOT IN ('clip_erro', 'frame_erro')"
    return con.execute("""
        SELECT c.path, m.destino, a.sha1, a.ext
        FROM classes c
        JOIN movidos m ON m.path = c.path
        JOIN arquivos a ON a.path = c.path
        WHERE """ + cond + """
          AND (c.path NOT IN (SELECT path FROM luna_valida)""" + extra + """)
        ORDER BY c.path""").fetchall()


def main():
    limite = 0
    if '--limite' in sys.argv:
        limite = int(sys.argv[sys.argv.index('--limite') + 1])
    foto_conf_max = None
    if '--zona-cega' in sys.argv:
        foto_conf_max = float(sys.argv[sys.argv.index('--zona-cega') + 1])
    chave = io.open(os.path.join(BASE, 'Chave_secreta_OpenAI.txt'),
                    encoding='utf-8').read().strip()
    cli = OpenAI(api_key=chave)

    con = conectar_mt()
    con.execute("""CREATE TABLE IF NOT EXISTS luna_valida(
        path TEXT PRIMARY KEY, veredicto TEXT, categoria TEXT,
        conf REAL, ts TEXT)""")
    con.commit()
    pend = fila(con, '--refazer-erros' in sys.argv, foto_conf_max)
    if limite:
        pend = pend[:limite]
    n_chamadas = (len(pend) + LOTE - 1) // LOTE
    print(f'Candidatos na fila: {len(pend):,} | chamadas ao {MODELO}: '
          f'~{n_chamadas:,} (lotes de {LOTE}, detail low)', flush=True)
    if not pend:
        con.close()
        return

    lotes = [pend[i:i + LOTE] for i in range(0, len(pend), LOTE)]
    stats = {'baixa': 0, 'manter': 0, 'erro': 0}
    feitos = [0]
    t0 = time.time()

    def processar(lote):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        itens, rows = [], []
        for path, destino, sha1, ext in lote:
            b64 = thumb_b64((destino, sha1, ext))
            if b64 is None:
                rows.append((path, 'erro', 'sem_imagem', 0.0, ts))
            else:
                itens.append((path, b64))
        if itens:
            try:
                resp = chamar(cli, [b for _, b in itens])
            except RuntimeError:
                resp = {}
            for i, (path, _) in enumerate(itens, 1):
                if i in resp:
                    cat, conf = resp[i]
                    ver = 'baixa' if cat in CAT_BAIXA and conf >= 0.5 else 'manter'
                    rows.append((path, ver, cat, conf, ts))
                else:
                    rows.append((path, 'erro', 'sem_resposta', 0.0, ts))
        with _lock:
            con.executemany('INSERT OR REPLACE INTO luna_valida VALUES(?,?,?,?,?)',
                            rows)
            con.commit()
            for r in rows:
                stats[r[1]] += 1
            feitos[0] += 1
            if feitos[0] % 25 == 0 or feitos[0] == len(lotes):
                taxa = feitos[0] / (time.time() - t0)
                print(f'  lote {feitos[0]:,}/{len(lotes):,} ({taxa*60:.0f} lotes/min, '
                      f'~{(len(lotes)-feitos[0])/max(taxa,0.01)/60:.0f} min restantes) '
                      f'| {stats}', flush=True)

    with ThreadPoolExecutor(max_workers=THREADS) as tp:
        list(tp.map(processar, lotes))

    det = con.execute("""SELECT veredicto, categoria, COUNT(*) FROM luna_valida
                         GROUP BY 1,2 ORDER BY 3 DESC""").fetchall()
    print('\nResultado acumulado (luna_valida):')
    for v, c, n in det:
        print(f'  {v:7} {c:22} {n:,}')
    con.close()


if __name__ == '__main__':
    main()
