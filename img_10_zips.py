# -*- coding: utf-8 -*-
"""img_10: inspeciona e remove os .zip do D:\\ cuja mídia já está coberta.

Por ZIP:
 1. Membros NÃO-mídia: descartáveis (json/html/css/js de takeout, metadados)
    são ignorados; QUALQUER outro (pdf/doc/planilha/desconhecido) é extraído
    para D:\\_ZIP_DOCUMENTOS\\<zip>\\ (na dúvida, preserva).
 2. Cobertura da mídia: cada membro-mídia precisa constar no catálogo
    (path extraído em D:\\_ZIP_EXTRAIDO com mesmo tamanho). Se TODOS cobertos,
    o ZIP é apagado; senão, fica e é relatado.
Log: log_img_zips_<ts>.csv.
"""
import csv
import datetime
import os
import re
import zipfile

from img_lib import EXT_MEDIA, EXCLUIR_TOPO, RAIZ, ZIP_EXTRAIDO, conectar, lp

AQUI = os.path.dirname(os.path.abspath(__file__))
ZIP_DOCS = r'D:\_ZIP_DOCUMENTOS'
# metadados de takeout/web sem valor documental
EXT_DESCARTAVEL = {'.json', '.html', '.htm', '.css', '.js', '.mhtml', '.tbn',
                   '.ini', '.db', '.info', ''}
INVALIDOS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def nome_membro(info):
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode('cp437').decode('cp850')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def sanitizar(seg):
    seg = INVALIDOS.sub('_', seg).strip(' .')
    return seg or '_'


def main():
    con = conectar()
    # (path, size) de tudo que foi catalogado vindo dos ZIPs extraídos
    catalogo = {}
    for p, s in con.execute("SELECT path, size FROM arquivos WHERE path LIKE ?",
                            (ZIP_EXTRAIDO + '\\%',)):
        catalogo[p.lower()] = s
    con.close()
    print(f'Membros de mídia catalogados de ZIPs: {len(catalogo):,}')

    zips = []
    for root, dirs, files in os.walk(RAIZ):
        partes = root.rstrip('\\').split('\\')
        if len(partes) >= 2 and partes[1] in EXCLUIR_TOPO:
            dirs[:] = []
            continue
        if root.startswith((ZIP_EXTRAIDO, ZIP_DOCS)):
            dirs[:] = []
            continue
        zips.extend(os.path.join(root, fn) for fn in files
                    if fn.lower().endswith('.zip'))
    zips.sort()
    print(f'ZIPs: {len(zips)}')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(AQUI, f'log_img_zips_{ts}.csv')
    apagados = mantidos = docs_extraidos = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['zip', 'acao', 'detalhe'])
        for i, zp in enumerate(sorted(zips), 1):
            base = sanitizar(os.path.splitext(os.path.basename(zp))[0])
            pasta_zip_ext = f'{base}_{i:03d}'  # mesma numeração do img_00
            try:
                with zipfile.ZipFile(lp(zp)) as zf:
                    nao_cobertos = 0
                    docs = 0
                    for m in zf.infolist():
                        if m.is_dir():
                            continue
                        interno = nome_membro(m)
                        ext = os.path.splitext(interno)[1].lower()
                        rel = '\\'.join(sanitizar(s) for s in re.split(r'[\\/]+', interno) if s)
                        if ext in EXT_MEDIA:
                            esperado = os.path.join(ZIP_EXTRAIDO, pasta_zip_ext, rel).lower()
                            if catalogo.get(esperado) != m.file_size:
                                nao_cobertos += 1
                        elif ext not in EXT_DESCARTAVEL:
                            alvo = os.path.join(ZIP_DOCS, pasta_zip_ext, rel)
                            if not (os.path.exists(lp(alvo))
                                    and os.path.getsize(lp(alvo)) == m.file_size):
                                os.makedirs(lp(os.path.dirname(alvo)), exist_ok=True)
                                with zf.open(m) as src, open(lp(alvo), 'wb') as dst:
                                    while True:
                                        chunk = src.read(1 << 20)
                                        if not chunk:
                                            break
                                        dst.write(chunk)
                            docs += 1
                    docs_extraidos += docs
                if nao_cobertos == 0:
                    os.remove(lp(zp))
                    apagados += 1
                    log.writerow([zp, 'apagado', f'{docs} docs extraidos'])
                else:
                    mantidos += 1
                    log.writerow([zp, 'MANTIDO', f'{nao_cobertos} midias nao cobertas'])
            except (zipfile.BadZipFile, RuntimeError, OSError) as e:
                mantidos += 1
                log.writerow([zp, 'MANTIDO', f'erro: {e}'])
    print(f'ZIPs apagados: {apagados} | mantidos: {mantidos} | '
          f'documentos extraídos p/ {ZIP_DOCS}: {docs_extraidos}')
    print(f'Log: {log_path}')


if __name__ == '__main__':
    main()
