# -*- coding: utf-8 -*-
"""Base comum do extrator de imagens D:\\ -> D:\\IMAGENS (família img_*).

Catálogo em SQLite (img_catalogo.sqlite). Caminhos gravados SEM prefixo \\?\
(legíveis); usar lp() em toda operação de arquivo.
"""
import hashlib
import os
import sqlite3

from PIL import Image

Image.MAX_IMAGE_PIXELS = 250_000_000  # panorâmicas legítimas passam de 89 MP

RAIZ = 'D:\\'
DESTINO = 'D:\\MÍDIA'
ZIP_EXTRAIDO = r'D:\_ZIP_EXTRAIDO'
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img_catalogo.sqlite')

EXT_IMG = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.webp', '.heic'}
EXT_VID = {'.mp4', '.mov', '.avi', '.3gp', '.wmv', '.mpg', '.mpeg', '.m4v',
           '.mts', '.m2ts', '.webm', '.mkv', '.flv', '.vob'}
EXT_MEDIA = EXT_IMG | EXT_VID
EXCLUIR_TOPO = {
    '$RECYCLE.BIN', 'RECYCLER', 'System Volume Information',
    'IMAGENS', 'MÍDIA', 'HP SureStore Application',
}

# nomes de pasta que não dizem nada sobre o conteúdo (não servem de "álbum")
PASTAS_GENERICAS = {
    'fotos', 'foto', 'imagens', 'imagem', 'img', 'image', 'images', 'pictures',
    'photos', 'camera', 'dcim', 'pics', 'galeria', 'downloads', 'download',
    '4ddig', 'jpg', 'jpeg', 'png', 'photo', 'camera roll', 'onedrive_sem_data',
}


def lp(path):
    """Prefixo \\\\?\\ para caminhos longos no Windows."""
    if path.startswith('\\\\?\\'):
        return path
    return '\\\\?\\' + os.path.abspath(path)


def conectar():
    con = sqlite3.connect(DB, timeout=60)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute("""
        CREATE TABLE IF NOT EXISTS arquivos(
            path TEXT PRIMARY KEY,
            dump TEXT, album TEXT,
            size INTEGER, mtime REAL, ext TEXT,
            width INTEGER, height INTEGER,
            sha1 TEXT, pixmd5 TEXT, dhash TEXT, dhash_pop INTEGER,
            exif_dto TEXT, exif_dtd TEXT, exif_dt TEXT,
            exif_make TEXT, exif_model TEXT,
            gps INTEGER, n_tags INTEGER,
            erro TEXT)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS datas(
            path TEXT PRIMARY KEY,
            data TEXT, fonte TEXT, gran TEXT)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS grupos(
            path TEXT PRIMARY KEY,
            grupo INTEGER, campeao INTEGER, camada TEXT)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS plano(
            path TEXT PRIMARY KEY,
            destino TEXT, exclusao TEXT)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS movidos(
            path TEXT PRIMARY KEY,
            destino TEXT, metodo TEXT, ts TEXT)
    """)
    con.execute('CREATE INDEX IF NOT EXISTS ix_sha1 ON arquivos(sha1)')
    con.execute('CREATE INDEX IF NOT EXISTS ix_pixmd5 ON arquivos(pixmd5)')
    con.execute('CREATE INDEX IF NOT EXISTS ix_dhash ON arquivos(dhash)')
    con.execute('CREATE INDEX IF NOT EXISTS ix_grupo ON grupos(grupo)')
    return con


def dumps_origem():
    """Pastas de topo do D:\\ a varrer (dumps + extração de ZIPs se existir)."""
    tops = []
    for entry in os.scandir(RAIZ):
        if entry.is_dir() and entry.name not in EXCLUIR_TOPO \
                and not entry.name.startswith('_ZIP_EXTRAIDO'):
            tops.append(entry.path)
    if os.path.isdir(ZIP_EXTRAIDO):
        tops.append(ZIP_EXTRAIDO)
    return sorted(tops)


def iter_imagens(raizes):
    """Gera (path, dump, size, mtime) de imagens E vídeos sob as raízes."""
    for raiz in raizes:
        dump = os.path.basename(raiz.rstrip('\\'))
        for root, dirs, files in os.walk(lp(raiz)):
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in EXT_MEDIA:
                    continue
                full = os.path.join(root, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                # gravar sem o prefixo \\?\
                yield full[4:] if full.startswith('\\\\?\\') else full, dump, st.st_size, st.st_mtime


def _mp4_creation_time(path):
    """Data de criação do container MP4/MOV/3GP/M4V (box moov>mvhd).

    Epoch QuickTime = 1904-01-01 UTC. Retorna 'YYYY:MM:DD HH:MM:SS' (formato
    EXIF, p/ reusar a cascata do img_02) ou None se zerado/implausível.
    """
    import datetime
    import struct
    try:
        tam = os.path.getsize(lp(path))
        with open(lp(path), 'rb') as f:
            def achar(tipo, limite):
                while f.tell() < limite - 8:
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        return None
                    size, typ = struct.unpack('>I4s', hdr)
                    corpo = f.tell()
                    if size == 1:
                        size = struct.unpack('>Q', f.read(8))[0]
                        corpo = f.tell()
                        size -= 16
                    elif size == 0:
                        size = limite - corpo + 8
                        size -= 8
                    else:
                        size -= 8
                    if typ == tipo:
                        return corpo, size
                    f.seek(corpo + size)
                return None

            moov = achar(b'moov', tam)
            if not moov:
                return None
            mvhd = achar(b'mvhd', moov[0] + moov[1])
            if not mvhd:
                return None
            versao = f.read(1)[0]
            f.read(3)  # flags
            bruto = struct.unpack('>Q' if versao == 1 else '>I',
                                  f.read(8 if versao == 1 else 4))[0]
        if bruto == 0:
            return None
        dt = datetime.datetime(1904, 1, 1) + datetime.timedelta(seconds=bruto)
        if not (1990 <= dt.year <= 2026):
            return None
        return dt.strftime('%Y:%m:%d %H:%M:%S')
    except (OSError, struct.error):
        return None


def _dhash64(im):
    """dHash 64-bit: grayscale 9x8, diferenças horizontais."""
    g = im.convert('L').resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return bits


def processar_arquivo(args):
    """Worker: lê um arquivo e devolve a linha completa p/ tabela arquivos."""
    path, dump, size, mtime = args
    album = os.path.basename(os.path.dirname(path))
    ext = os.path.splitext(path)[1].lower()
    row = {
        'path': path, 'dump': dump, 'album': album,
        'size': size, 'mtime': mtime, 'ext': ext,
        'width': None, 'height': None,
        'sha1': None, 'pixmd5': None, 'dhash': None, 'dhash_pop': None,
        'exif_dto': None, 'exif_dtd': None, 'exif_dt': None,
        'exif_make': None, 'exif_model': None,
        'gps': 0, 'n_tags': 0, 'erro': None,
    }
    # sha1 do arquivo inteiro
    try:
        h = hashlib.sha1()
        with open(lp(path), 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        row['sha1'] = h.hexdigest()
    except OSError:
        row['erro'] = 'ilegivel'
        return row
    # vídeo: sem decodificação de pixels; data vem do container (mvhd)
    if ext in EXT_VID:
        if ext in ('.mp4', '.mov', '.3gp', '.m4v'):
            ct = _mp4_creation_time(path)
            if ct:
                row['exif_dto'] = ct
                row['n_tags'] = 1
        return row
    # decodificação + EXIF + hashes visuais
    try:
        with open(lp(path), 'rb') as f:
            with Image.open(f) as im:
                row['width'], row['height'] = im.width, im.height
                try:
                    ex = im.getexif()
                    if ex:
                        ifd = ex.get_ifd(0x8769)
                        gps = ex.get_ifd(0x8825)
                        row['exif_dto'] = str(ifd.get(36867)) if ifd.get(36867) else None
                        row['exif_dtd'] = str(ifd.get(36868)) if ifd.get(36868) else None
                        row['exif_dt'] = str(ex.get(306)) if ex.get(306) else None
                        row['exif_make'] = str(ex.get(271)) if ex.get(271) else None
                        row['exif_model'] = str(ex.get(272)) if ex.get(272) else None
                        row['gps'] = 1 if gps and len(gps) > 0 else 0
                        row['n_tags'] = len(ex) + len(ifd) + (len(gps) if gps else 0)
                except Exception:
                    pass
                im.load()  # força decodificação completa (detecta truncado)
                rgb = im.convert('RGB')
                row['pixmd5'] = hashlib.md5(rgb.tobytes()).hexdigest()
                dh = _dhash64(rgb)
                row['dhash'] = f'{dh:016x}'
                row['dhash_pop'] = bin(dh).count('1')
    except Exception as e:
        msg = str(e).lower()
        row['erro'] = 'truncado' if 'truncat' in msg else 'corrompido'
    return row
