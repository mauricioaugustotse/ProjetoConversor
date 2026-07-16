# -*- coding: utf-8 -*-
"""img_08: move telas/documentos/web para subpastas de D:\\IMAGENS\\0000 SEM DATA.

Gate calibrado no piloto de 12/07/2026 (erros do CLIP concentram-se em
conf<0.6 e em web/tela COM câmera no EXIF):
  - heurística            -> move sempre
  - clip documento >=0.85 -> move (inclui documento fotografado com câmera)
  - clip tela      >=0.85 -> move
  - clip web       >=0.85 -> move APENAS sem câmera no EXIF
Abaixo do gate (não-foto) -> img_revisao_classes.csv, arquivo fica onde está.
Move por os.rename, atualiza movidos.destino (índices do img_06 refletem) e
grava log de reversão log_img_classes_<ts>.csv.
"""
import csv
import datetime
import os

from img_lib import DESTINO, conectar, lp

AQUI = os.path.dirname(os.path.abspath(__file__))
SUBPASTA = {'tela': 'TELAS', 'documento': 'DOCUMENTOS', 'web': 'WEB'}
GATE = 0.85


def main():
    con = conectar()
    rows = con.execute("""
        SELECT c.path, c.classe, c.conf, c.metodo, m.destino,
               a.exif_make, a.exif_model
        FROM classes c
        JOIN movidos m ON m.path = c.path
        JOIN arquivos a ON a.path = c.path
        WHERE c.classe != 'foto' AND c.metodo != 'clip_erro'""").fetchall()

    mover, revisao = [], []
    for path, classe, conf, metodo, destino, make, model in rows:
        if destino.startswith(os.path.join(DESTINO, '0000 SEM DATA') + os.sep) \
                and f'\\{SUBPASTA[classe]}\\' in destino:
            continue  # já está no lugar (re-execução)
        cam = bool(make or model)
        passa = metodo == 'heuristica' or (
            conf >= GATE and (classe != 'web' or not cam))
        if passa:
            mover.append((path, classe, conf, metodo, destino))
        else:
            revisao.append((path, classe, round(conf, 3), metodo, destino,
                            'com camera' if cam else 'sem camera'))

    print(f'Não-foto classificadas: {len(rows):,} | mover: {len(mover):,} | '
          f'revisão: {len(revisao):,}')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(AQUI, f'log_img_classes_{ts}.csv')
    ok = err = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['origem_atual', 'destino_novo', 'classe', 'conf', 'metodo', 'status'])
        for path, classe, conf, metodo, destino in mover:
            pasta = os.path.join(DESTINO, '0000 SEM DATA', SUBPASTA[classe])
            novo = os.path.join(pasta, os.path.basename(destino))
            status = 'ok'
            try:
                if not os.path.exists(lp(destino)):
                    status = 'origem_sumiu'
                elif os.path.exists(lp(novo)):
                    status = 'destino_ocupado'
                else:
                    os.makedirs(lp(pasta), exist_ok=True)
                    os.rename(lp(destino), lp(novo))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([destino, novo, classe, f'{conf:.3f}', metodo, status])
            if status == 'ok':
                ok += 1
                con.execute('UPDATE movidos SET destino=? WHERE path=?', (novo, path))
            else:
                err += 1
        con.commit()

    rev_path = os.path.join(AQUI, 'img_revisao_classes.csv')
    with open(rev_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['path_original', 'classe_sugerida', 'conf', 'metodo',
                    'localizacao_atual', 'camera'])
        w.writerows(revisao)

    stats = {}
    for _, classe, *_ in mover:
        stats[classe] = stats.get(classe, 0) + 1
    print(f'Movidos: {ok:,} | pendências: {err:,} | por classe: {stats}')
    print(f'Log: {log_path}\nRevisão: {rev_path}')
    con.close()


if __name__ == '__main__':
    main()
