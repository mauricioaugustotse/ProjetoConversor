# -*- coding: utf-8 -*-
"""img_16: move os validados pelo luna (veredicto='baixa') para
D:\\MÍDIA\\9990 BAIXA RELEVANCIA\\<CATEGORIA>.

--plano            : só grava img_baixa_plano.csv (amostragem antes do exec)
--exec             : move (os.rename via lp), atualiza movidos.destino,
                     grava tabela baixa_undo + log log_img_baixa_<ts>.csv
--desfazer lista.txt : cada linha = caminho ATUAL do arquivo em 9990;
                     devolve ao destino anterior (tabela baixa_undo),
                     marca luna_valida.veredicto='manter_usuario'
"""
import csv
import datetime
import os
import sys

from img_lib import DESTINO, conectar, lp

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ_BAIXA = os.path.join(DESTINO, '9990 BAIXA RELEVANCIA')
SUBPASTA = {
    'tela': 'TELAS',
    'recibo_boleto': 'RECIBOS E BOLETOS',
    'cartao_convite': 'CARTOES E CONVITES',
    'meme_web': 'MEMES E WEB',
    'tv_gravacao': 'TV E GRAVACOES',
    'documento_trivial': 'DOCUMENTOS TRIVIAIS',
}


def garantir_undo(con):
    con.execute("""CREATE TABLE IF NOT EXISTS baixa_undo(
        path TEXT PRIMARY KEY, destino_anterior TEXT,
        destino_novo TEXT, ts TEXT)""")


def montar(con):
    rows = con.execute("""
        SELECT v.path, v.categoria, v.conf, m.destino
        FROM luna_valida v JOIN movidos m ON m.path = v.path
        WHERE v.veredicto = 'baixa'""").fetchall()
    plano = []
    for path, cat, conf, destino in rows:
        if destino.startswith(RAIZ_BAIXA + os.sep):
            continue  # já movido (re-execução)
        novo = os.path.join(RAIZ_BAIXA, SUBPASTA[cat], os.path.basename(destino))
        plano.append((path, destino, novo, cat, conf))
    return plano


def modo_plano():
    con = conectar()
    garantir_undo(con)
    plano = montar(con)
    out = os.path.join(AQUI, 'img_baixa_plano.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['path_original', 'origem_atual', 'destino_novo', 'categoria', 'conf'])
        w.writerows(plano)
    stats = {}
    for *_, cat, _c in plano:
        stats[cat] = stats.get(cat, 0) + 1
    print(f'Plano: {len(plano):,} a mover | por categoria: {stats}\n{out}')
    con.close()


def modo_exec():
    con = conectar()
    garantir_undo(con)
    plano = montar(con)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(AQUI, f'log_img_baixa_{ts}.csv')
    ok = err = 0
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as flog:
        log = csv.writer(flog)
        log.writerow(['origem_atual', 'destino_novo', 'categoria', 'conf', 'status'])
        for i, (path, destino, novo, cat, conf) in enumerate(plano, 1):
            status = 'ok'
            try:
                if not os.path.exists(lp(destino)):
                    status = 'origem_sumiu'
                elif os.path.exists(lp(novo)):
                    status = 'destino_ocupado'
                else:
                    os.makedirs(lp(os.path.dirname(novo)), exist_ok=True)
                    os.rename(lp(destino), lp(novo))
            except OSError as e:
                status = f'erro: {e}'
            log.writerow([destino, novo, cat, f'{conf:.2f}', status])
            if status == 'ok':
                ok += 1
                con.execute('UPDATE movidos SET destino=? WHERE path=?', (novo, path))
                con.execute('INSERT OR REPLACE INTO baixa_undo VALUES(?,?,?,?)',
                            (path, destino, novo, ts))
            else:
                err += 1
            if i % 1000 == 0:
                con.commit()
                print(f'  {i:,}/{len(plano):,} (pendências {err})', flush=True)
    con.commit()
    con.close()
    print(f'Movidos: {ok:,} | pendências: {err:,} | log: {log_path}')


def modo_desfazer(lista_txt):
    with open(lista_txt, encoding='utf-8') as f:
        alvos = [l.strip().strip('"') for l in f if l.strip()]
    con = conectar()
    garantir_undo(con)
    ok = err = 0
    for atual in alvos:
        row = con.execute('SELECT path, destino_anterior FROM baixa_undo '
                          'WHERE destino_novo=?', (atual,)).fetchone()
        if not row:
            print(f'  SEM REGISTRO: {atual}')
            err += 1
            continue
        path, anterior = row
        try:
            if not os.path.exists(lp(atual)):
                raise OSError('arquivo nao esta mais la')
            if os.path.exists(lp(anterior)):
                raise OSError('destino anterior ocupado')
            os.makedirs(lp(os.path.dirname(anterior)), exist_ok=True)
            os.rename(lp(atual), lp(anterior))
            con.execute('UPDATE movidos SET destino=? WHERE path=?', (anterior, path))
            con.execute("UPDATE luna_valida SET veredicto='manter_usuario' "
                        'WHERE path=?', (path,))
            con.execute('DELETE FROM baixa_undo WHERE path=?', (path,))
            ok += 1
        except OSError as e:
            print(f'  ERRO {e}: {atual}')
            err += 1
    con.commit()
    con.close()
    print(f'Desfeitos: {ok} | falhas: {err}')


def main():
    if '--plano' in sys.argv:
        modo_plano()
    elif '--exec' in sys.argv:
        modo_exec()
    elif '--desfazer' in sys.argv:
        modo_desfazer(sys.argv[sys.argv.index('--desfazer') + 1])
    else:
        print('uso: img_16_mover_baixa.py --plano | --exec | --desfazer lista.txt')


if __name__ == '__main__':
    main()
