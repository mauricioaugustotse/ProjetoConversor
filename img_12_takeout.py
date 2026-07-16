# -*- coding: utf-8 -*-
r"""img_12: incorpora os takeouts do Google Fotos (13/07/2026) ao pipeline img_*.

--extrair : extrai midia + sidecars .json dos zips D:\takeout-*.zip para
            D:\_TAKEOUT_EXTRAIDO (zips originais intactos). Membros sem
            extensao ganham ext por magic bytes. Idempotente via catalogo+disco.
--sidecars: casa cada sidecar .json (photoTakenTime) com seu arquivo de midia
            e grava na tabela takeout_meta (fonte de data p/ img_02).

Depois: img_01 "D:\_TAKEOUT_EXTRAIDO" -> img_02 -> img_03 -> img_04 -> img_05.
"""
import csv
import glob
import json
import os
import re
import sys
import zipfile

from img_lib import EXT_MEDIA, conectar, lp

AQUI = os.path.dirname(os.path.abspath(__file__))
DESTINO_TK = r'D:\_TAKEOUT_EXTRAIDO'
SAIDA_CSV = os.path.join(AQUI, 'img_takeout_relatorio.csv')
INVALIDOS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
EXTS_RX = '|'.join(e[1:] for e in EXT_MEDIA)
# "IMG.jpg.supplemental-metadata.json" (e truncagens) -> IMG.jpg
RE_BASE_MIDIA = re.compile(r'^(.+\.(?:' + EXTS_RX + r'))(\..*)?$', re.I)
# "IMG.jpg(1).json" -> IMG(1).jpg
RE_PAREN_POS = re.compile(r'^(.+)\.(' + EXTS_RX + r')\((\d+)\)$', re.I)
ALBUM_JSON = {'metadados.json', 'metadata.json', 'metadados(1).json',
              'print-subscriptions.json', 'shared_album_comments.json',
              'user-generated-memory-titles.json'}


def sanitizar(seg):
    seg = INVALIDOS.sub('_', seg).strip(' .')
    return seg or '_'


def sniff_ext(cabeca):
    if cabeca[:3] == b'\xff\xd8\xff':
        return '.jpg'
    if cabeca[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'
    if cabeca[:4] in (b'GIF8',):
        return '.gif'
    if cabeca[4:8] == b'ftyp':
        return '.mp4'
    if cabeca[:4] == b'RIFF' and cabeca[8:12] == b'WEBP':
        return '.webp'
    return None


def extrair():
    con = conectar()
    ja_catalogado = {p.lower(): s for p, s in con.execute(
        'SELECT path, size FROM arquivos WHERE path LIKE ?',
        (DESTINO_TK + '\\%',))}
    con.close()

    zips = sorted(glob.glob(r'D:\takeout-*.zip'))
    print(f'ZIPs de takeout: {len(zips)}')
    linhas = []
    tot_ex = tot_pul = tot_json = 0
    for zi, zp in enumerate(zips, 1):
        with zipfile.ZipFile(lp(zp)) as zf:
            membros = [m for m in zf.infolist() if not m.is_dir()]
            for m in membros:
                nome = m.filename  # takeout novo e sempre UTF-8
                ext = os.path.splitext(nome)[1].lower()
                sem_ext = '.' not in os.path.basename(nome)
                if ext not in EXT_MEDIA and ext != '.json' and not sem_ext:
                    linhas.append([zp, nome, 'ignorado_ext', ext])
                    continue
                partes = [sanitizar(s) for s in re.split(r'[\\/]+', nome) if s]
                if partes and partes[0].lower() == 'takeout':
                    partes = partes[1:]
                alvo = os.path.join(DESTINO_TK, *partes)
                try:
                    if ja_catalogado.get(alvo.lower()) == m.file_size or \
                            (os.path.exists(lp(alvo))
                             and os.path.getsize(lp(alvo)) == m.file_size):
                        tot_pul += 1
                        continue
                    os.makedirs(lp(os.path.dirname(alvo)), exist_ok=True)
                    with zf.open(m) as src:
                        cabeca = src.read(16)
                        if sem_ext:
                            ext_s = sniff_ext(cabeca)
                            if not ext_s:
                                linhas.append([zp, nome, 'sem_ext_desconhecido',
                                               cabeca[:8].hex()])
                                continue
                            alvo += ext_s
                            if os.path.exists(lp(alvo)) and \
                                    os.path.getsize(lp(alvo)) == m.file_size:
                                tot_pul += 1
                                continue
                        with open(lp(alvo), 'wb') as dst:
                            dst.write(cabeca)
                            while True:
                                chunk = src.read(1 << 20)
                                if not chunk:
                                    break
                                dst.write(chunk)
                    if alvo.lower().endswith('.json'):
                        tot_json += 1
                    else:
                        tot_ex += 1
                except (OSError, zipfile.BadZipFile, RuntimeError) as e:
                    linhas.append([zp, nome, 'erro_membro', str(e)])
        print(f'  {zi}/{len(zips)}: {os.path.basename(zp)} ok '
              f'(midia {tot_ex:,}, json {tot_json:,}, pulados {tot_pul:,})',
              flush=True)
    with open(SAIDA_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['zip', 'membro', 'status', 'detalhe'])
        w.writerows(linhas)
    print(f'Extraidos: midia {tot_ex:,} + json {tot_json:,} | '
          f'pulados {tot_pul:,} | avisos {len(linhas):,} -> {SAIDA_CSV}')


def alvo_do_sidecar(dirpath, jname, arquivos_lower):
    """Deduz o arquivo de midia dono do sidecar jname dentro de dirpath."""
    base = jname[:-5]  # tira .json
    cands = []
    m = RE_PAREN_POS.match(base)
    if m:  # IMG.jpg(1) -> IMG(1).jpg
        cands.append(f'{m.group(1)}({m.group(3)}).{m.group(2)}')
    m = RE_BASE_MIDIA.match(base)
    if m:  # IMG.jpg.supplemental-metadata -> IMG.jpg
        cands.append(m.group(1))
    cands.append(base)  # sem-extensao: ganhou ext no --extrair
    for c in list(cands):
        if '.' not in os.path.basename(c):
            cands += [c + e for e in ('.jpg', '.mp4', '.png', '.gif', '.webp')]
    for c in cands:
        hit = arquivos_lower.get(c.lower())
        if hit:
            return hit
    # nome truncado: casa por prefixo inequivoco
    pref = os.path.splitext(cands[0])[0][:40].lower()
    if len(pref) >= 20:
        hits = [v for k, v in arquivos_lower.items()
                if os.path.splitext(k)[0].lower().startswith(pref)]
        if len(hits) == 1:
            return hits[0]
    return None


def sidecars():
    con = conectar()
    con.execute("""CREATE TABLE IF NOT EXISTS takeout_meta(
        path TEXT PRIMARY KEY, epoch INTEGER, title TEXT)""")
    stats = {'json': 0, 'album': 0, 'sem_taken': 0, 'casado': 0, 'orfao': 0}
    orfaos = []
    lote = []
    for root, dirs, files in os.walk(lp(DESTINO_TK)):
        base = root[4:] if root.startswith('\\\\?\\') else root
        midias = {f: os.path.join(base, f) for f in files
                  if os.path.splitext(f)[1].lower() in EXT_MEDIA}
        midias_lower = {k.lower(): v for k, v in midias.items()}
        for f in files:
            if not f.lower().endswith('.json'):
                continue
            stats['json'] += 1
            if f.lower() in ALBUM_JSON:
                stats['album'] += 1
                continue
            try:
                with open(os.path.join(root, f), encoding='utf-8') as fh:
                    meta = json.load(fh)
            except (OSError, json.JSONDecodeError):
                stats['sem_taken'] += 1
                continue
            epoch = meta.get('photoTakenTime', {}).get('timestamp')
            title = meta.get('title', '')
            if not epoch:
                stats['album' if 'albumData' in meta or 'title' not in meta
                      else 'sem_taken'] += 1
                continue
            alvo = alvo_do_sidecar(base, f, midias_lower)
            if not alvo and title:
                alvo = midias_lower.get(title.lower())
            if alvo:
                lote.append((alvo, int(epoch), title))
                stats['casado'] += 1
            else:
                stats['orfao'] += 1
                orfaos.append(os.path.join(base, f))
    con.executemany('INSERT OR REPLACE INTO takeout_meta(path, epoch, title) '
                    'VALUES(?,?,?)', lote)
    con.commit()
    n_meta = con.execute('SELECT COUNT(*) FROM takeout_meta').fetchone()[0]
    con.close()
    print(f"Sidecars: {stats['json']:,} jsons | casados {stats['casado']:,} | "
          f"album/aux {stats['album']:,} | sem photoTakenTime {stats['sem_taken']:,} | "
          f"orfaos {stats['orfao']:,}")
    print(f'takeout_meta: {n_meta:,} midias com data do Google')
    if orfaos:
        with open(os.path.join(AQUI, 'img_takeout_orfaos.txt'), 'w',
                  encoding='utf-8') as f:
            f.write('\n'.join(orfaos))
        print(f'  orfaos listados em img_takeout_orfaos.txt (primeiros: '
              f'{[os.path.basename(o) for o in orfaos[:5]]})')


if __name__ == '__main__':
    if '--extrair' in sys.argv:
        extrair()
    elif '--sidecars' in sys.argv:
        sidecars()
    else:
        print(__doc__)
