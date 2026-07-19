# -*- coding: utf-8 -*-
"""img_17: galeria HTML de auditoria da pasta 9990 BAIXA RELEVANCIA.

Gera em D:\\MÍDIA\\9990 BAIXA RELEVANCIA\\_AUDITORIA:
  index.html + <categoria>_pNN.html (500 miniaturas/página, base64 embutido).
Cada miniatura: clique abre o arquivo real; checkbox marca para DESFAZER.
Botão "Copiar lista dos marcados" -> colar em desfazer.txt -> rodar
  python img_16_mover_baixa.py --desfazer desfazer.txt
"""
import base64
import html
import io
import os

from PIL import Image

from img_lib import DESTINO, EXT_VID, conectar, lp

RAIZ_BAIXA = os.path.join(DESTINO, '9990 BAIXA RELEVANCIA')
AUDIT = os.path.join(RAIZ_BAIXA, '_AUDITORIA')
CACHE_FRAMES = r'D:\MÍDIA\_frames_cache'
POR_PAGINA = 500
THUMB = 208

CSS = """body{font-family:Segoe UI,Arial,sans-serif;background:#181a1f;color:#ddd;margin:16px}
h1{font-size:1.2em}a{color:#7ab8ff}
.grid{display:flex;flex-wrap:wrap;gap:8px}
.item{width:%dpx;background:#23262d;border-radius:6px;padding:6px;font-size:11px;word-break:break-all}
.item img{width:100%%;border-radius:4px;display:block;cursor:pointer}
.item.marcado{outline:3px solid #e05f5f}
.top{position:sticky;top:0;background:#181a1f;padding:8px 0;z-index:9}
button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer;margin-right:8px}
nav a{margin-right:10px}
.vid{color:#ffc46b}""" % THUMB

JS = """
function marcar(cb){cb.closest('.item').classList.toggle('marcado',cb.checked);cont()}
function cont(){document.getElementById('n').textContent=document.querySelectorAll('input:checked').length}
function copiar(){
 const l=[...document.querySelectorAll('input:checked')].map(c=>c.dataset.p).join('\\n');
 if(!l){alert('Nada marcado.');return}
 navigator.clipboard.writeText(l).then(()=>alert('Lista copiada! Cole em desfazer.txt e rode:\\npython img_16_mover_baixa.py --desfazer desfazer.txt'));
}
"""


def thumb_b64(destino, ext, sha1):
    src = destino
    if ext in EXT_VID:
        for suf in ('_1.jpg', '_2.jpg'):
            fr = os.path.join(CACHE_FRAMES, sha1[:16] + suf)
            if os.path.exists(fr):
                src = fr
                break
        else:
            return None
    try:
        with open(lp(src), 'rb') as f:
            with Image.open(f) as im:
                im.thumbnail((THUMB, THUMB))
                buf = io.BytesIO()
                im.convert('RGB').save(buf, 'JPEG', quality=60)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def pagina(cat, num, total_pags, itens):
    linhas = [f'<!doctype html><meta charset="utf-8"><title>{cat} p{num}</title>',
              f'<style>{CSS}</style><script>{JS}</script>',
              '<div class="top">',
              f'<h1>{cat} — página {num}/{total_pags}</h1>',
              '<button onclick="copiar()">Copiar lista dos marcados</button>',
              '<span id="n">0</span> marcados para desfazer | ',
              '<a href="index.html">índice</a></div><div class="grid">']
    for destino, conf, b64, eh_vid in itens:
        nome = html.escape(os.path.basename(destino))
        url = 'file:///' + destino.replace('\\', '/')
        p = html.escape(destino)
        vid = ' <span class="vid">[vídeo]</span>' if eh_vid else ''
        img = (f'<img src="data:image/jpeg;base64,{b64}" '
               f'onclick="window.open(\'{url}\')">') if b64 else '<em>sem prévia</em>'
        linhas.append(f'<div class="item">{img}'
                      f'<label><input type="checkbox" data-p="{p}" '
                      f'onchange="marcar(this)"> desfazer</label>'
                      f'<div>{nome}{vid} <small>conf {conf:.2f}</small></div></div>')
    linhas.append('</div><nav>')
    for k in range(1, total_pags + 1):
        linhas.append(f'<a href="{cat}_p{k:02}.html">{k}</a>')
    linhas.append('</nav>')
    return '\n'.join(linhas)


def main():
    con = conectar()
    rows = con.execute("""
        SELECT v.categoria, m.destino, v.conf, a.ext, a.sha1
        FROM luna_valida v
        JOIN movidos m ON m.path = v.path
        JOIN arquivos a ON a.path = v.path
        WHERE v.veredicto='baixa' AND m.destino LIKE ?
        UNION ALL
        SELECT 'video_' || w.categoria, m.destino, w.conf, a.ext, a.sha1
        FROM luna_wavid w
        JOIN movidos m ON m.path = w.path
        JOIN arquivos a ON a.path = w.path
        WHERE w.veredicto='baixa' AND m.destino LIKE ?
        ORDER BY 1, 2""", (RAIZ_BAIXA + '%', RAIZ_BAIXA + '%')).fetchall()
    con.close()
    os.makedirs(AUDIT, exist_ok=True)
    por_cat = {}
    for cat, destino, conf, ext, sha1 in rows:
        por_cat.setdefault(cat, []).append((destino, conf, ext, sha1))
    print(f'Galeria: {len(rows):,} itens em {len(por_cat)} categorias', flush=True)

    indice = ['<!doctype html><meta charset="utf-8"><title>Auditoria baixa relevância</title>',
              f'<style>{CSS}</style><h1>Auditoria — 9990 BAIXA RELEVANCIA</h1>',
              '<p>Marque o que deve VOLTAR ao acervo, copie a lista e rode '
              '<code>python img_16_mover_baixa.py --desfazer desfazer.txt</code></p><ul>']
    for cat, itens in sorted(por_cat.items()):
        pags = (len(itens) + POR_PAGINA - 1) // POR_PAGINA
        for k in range(pags):
            fatia = itens[k * POR_PAGINA:(k + 1) * POR_PAGINA]
            corpo = [(d, c, thumb_b64(d, e, s), e in EXT_VID) for d, c, e, s in fatia]
            out = os.path.join(AUDIT, f'{cat}_p{k+1:02}.html')
            with open(out, 'w', encoding='utf-8') as f:
                f.write(pagina(cat, k + 1, pags, corpo))
            print(f'  {os.path.basename(out)} ({len(fatia)} itens)', flush=True)
        indice.append(f'<li><a href="{cat}_p01.html">{cat}</a> — {len(itens):,} itens</li>')
    indice.append('</ul>')
    with open(os.path.join(AUDIT, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(indice))
    print(f'Índice: {os.path.join(AUDIT, "index.html")}')


if __name__ == '__main__':
    main()
