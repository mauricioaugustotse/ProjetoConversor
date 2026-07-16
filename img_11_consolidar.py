# -*- coding: utf-8 -*-
"""img_11: consolida tudo que restou fora de D:\\MÍDIA em D:\\ACERVO por tema
e apaga as pastas vazias remanescentes dos dumps.

- Move por os.rename (mesmo volume, instantâneo), árvore inteira por origem;
  fallback item a item se a pasta estiver presa (Explorer aberto).
- Lixo de sistema (Thumbs.db, desktop.ini, ehthumbs.db, .picasa.ini) é
  APAGADO, não movido.
- NÃO toca: D:\\MÍDIA, takeout*.zip na raiz (downloads do usuário),
  $RECYCLE.BIN / RECYCLER / System Volume Information.
Log: log_img_consolida_<ts>.csv.
"""
import csv
import datetime
import os
import shutil

from img_lib import lp

ACERVO = r'D:\ACERVO'
LIXO_SISTEMA = {'thumbs.db', 'desktop.ini', 'ehthumbs.db', '.picasa.ini'}
PROTEGIDOS = {'$RECYCLE.BIN', 'RECYCLER', 'System Volume Information',
              'MÍDIA', 'ACERVO'}

MAPA = {
    r'D:\HD_Mau': 'HD_Mau',
    r'D:\Documentos': 'Documentos',
    r'D:\Z_GABINETES': 'Gabinetes TSE',
    r'D:\HD EXTERNO Mário': 'HD Mário',
    r'D:\AEED - Documentos': 'AEED',
    r'D:\_ZIP_DOCUMENTOS': 'Resgatados de ZIPs',
    r'D:\Document': r'Diversos\Manuais (Document)',
    r'D:\Camera Roll': r'Diversos\Camera Roll',
    r'D:\4DDig': r'Diversos\4DDig',
    r'D:\Banco Onedrive_fotos_consertado_16_9_2024': r'Diversos\Banco Onedrive_fotos',
    r'D:\Banco Onedrive_vídeos_16_9_2024': r'Diversos\Banco Onedrive_vídeos',
    r'D:\HP SureStore Application': r'Diversos\HP SureStore',
    r'D:\Onedrive_sem_data': r'Diversos\Onedrive_sem_data',
    r'D:\Vídeos_sem_data': r'Diversos\Vídeos_sem_data',
    r'D:\CASAMENTO': r'Diversos\CASAMENTO',
    r'D:\Duplicate': r'Diversos\Duplicate',
    r'D:\Extração JPEG (HD mário)': r'Diversos\Extração JPEG',
    r'D:\Takeout em 17_12_2023': r'Diversos\Takeout 17_12_2023',
    r'D:\Takout em 14_9_2024': r'Diversos\Takout 14_9_2024',
}
ARQUIVOS_RAIZ = {r'D:\HPSureStore.exe': r'Diversos\HP SureStore\HPSureStore.exe'}


def vazio(p):
    for _, _, files in os.walk(lp(p)):
        if files:
            return False
    return True


def mover_arvore(origem, destino, log):
    os.makedirs(lp(os.path.dirname(destino)), exist_ok=True)
    try:
        os.rename(lp(origem), lp(destino))
        log.writerow([origem, destino, 'rename_arvore', 'ok'])
        return True
    except OSError:
        pass
    # fallback: item a item (pasta presa por handle)
    erros = 0
    for root, dirs, files in os.walk(lp(origem)):
        rel = root[len(lp(origem)):].lstrip('\\')
        for fn in files:
            src = os.path.join(root, fn)
            dst = os.path.join(lp(destino), rel, fn)
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
            except OSError as e:
                log.writerow([src, dst, 'item', f'erro: {e}'])
                erros += 1
    log.writerow([origem, destino, 'fallback_itens', f'{erros} erros'])
    return erros == 0


def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'log_img_consolida_{ts}.csv')
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['origem', 'destino', 'metodo', 'status'])

        movidas = puladas_vazias = 0
        for origem, tema in MAPA.items():
            if not os.path.isdir(lp(origem)):
                continue
            if vazio(origem):
                shutil.rmtree(lp(origem))
                log.writerow([origem, '', 'pasta_vazia', 'apagada'])
                puladas_vazias += 1
                continue
            if mover_arvore(origem, os.path.join(ACERVO, tema), log):
                movidas += 1
        for origem, rel in ARQUIVOS_RAIZ.items():
            if os.path.isfile(lp(origem)):
                dst = os.path.join(ACERVO, rel)
                os.makedirs(lp(os.path.dirname(dst)), exist_ok=True)
                os.rename(lp(origem), lp(dst))
                log.writerow([origem, dst, 'arquivo_raiz', 'ok'])

        # lixo de sistema dentro do ACERVO
        lixo = 0
        for root, dirs, files in os.walk(lp(ACERVO)):
            for fn in files:
                if fn.lower() in LIXO_SISTEMA:
                    try:
                        os.remove(os.path.join(root, fn))
                        lixo += 1
                    except OSError:
                        pass

        # pastas vazias: dentro do ACERVO e na raiz do D:\
        # (handle aberto no Explorer causa PermissionError — pular e relatar)
        vazias = presas = 0
        for base in [ACERVO] + [p for p in MAPA if os.path.isdir(lp(p))]:
            for root, dirs, files in os.walk(lp(base), topdown=False):
                if not os.listdir(root):
                    try:
                        os.rmdir(root)
                        vazias += 1
                    except OSError:
                        presas += 1
                        log.writerow([root, '', 'pasta_vazia', 'PRESA (handle aberto)'])
        for entry in os.scandir('D:\\'):
            if entry.is_dir() and entry.name not in PROTEGIDOS:
                if vazio(entry.path):
                    try:
                        shutil.rmtree(lp(entry.path))
                        log.writerow([entry.path, '', 'pasta_vazia_raiz', 'apagada'])
                        vazias += 1
                    except OSError:
                        presas += 1
                        log.writerow([entry.path, '', 'pasta_vazia_raiz', 'PRESA (handle aberto)'])

    print(f'Árvores movidas p/ ACERVO: {movidas} | vazias apagadas direto: {puladas_vazias}')
    print(f'Lixo de sistema removido: {lixo} | pastas vazias removidas: {vazias} | '
          f'presas (handle): {presas}')
    print(f'Log: {log_path}')
    print('\nRaiz do D:\\ agora:')
    for e in sorted(os.scandir('D:\\'), key=lambda x: x.name.lower()):
        print(f'  {"[D]" if e.is_dir() else "[A]"} {e.name}')


if __name__ == '__main__':
    main()
