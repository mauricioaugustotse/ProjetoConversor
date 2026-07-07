# -*- coding: utf-8 -*-
"""Org2 etapa 2: dedup binario (MD5) em HD_Mau.
Agrupa por tamanho, hasheia so os grupos, manda copias excedentes para a Lixeira.
Gera plano_org2_dedup_bin.csv e log_org2_dedup_bin.csv (reversao: o que saiu e quem ficou)."""
import csv, hashlib, os, re, sys
from collections import defaultdict
from send2trash import send2trash

ROOT = r"C:\Users\mauri\HD_Mau"
BASE = os.path.dirname(os.path.abspath(__file__))
PLANO = os.path.join(BASE, "plano_org2_dedup_bin.csv")
LOG = os.path.join(BASE, "log_org2_dedup_bin.csv")
EXEC = "--exec" in sys.argv

RE_SUFIXO = re.compile(r"[ _]\(\d+\)$|[ _]\[\d+\]$")
RE_CNJ = re.compile(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")

def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def score(path):
    """Maior = melhor candidato a sobreviver."""
    base = os.path.splitext(os.path.basename(path))[0]
    s = 0
    if not RE_SUFIXO.search(base):
        s += 1000
    if RE_CNJ.search(base):
        s += 500
    s += min(len(base), 200)
    return s

def main():
    por_tamanho = defaultdict(list)
    for dirpath, _, filenames in os.walk(ROOT):
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz > 0 and not name.lower().endswith((".ini", ".md")):
                por_tamanho[sz].append(p)

    candidatos = [g for g in por_tamanho.values() if len(g) > 1]
    print(f"Grupos de mesmo tamanho: {len(candidatos)} ({sum(len(g) for g in candidatos)} arquivos)")

    por_hash = defaultdict(list)
    erros = 0
    for grupo in candidatos:
        for p in grupo:
            try:
                por_hash[(md5(p), os.path.getsize(p))].append(p)
            except OSError:
                erros += 1

    dups = {k: v for k, v in por_hash.items() if len(v) > 1}
    plano = []  # (acao, caminho, hash, sobrevivente)
    for (h, sz), grupo in sorted(dups.items()):
        grupo = sorted(grupo, key=score, reverse=True)
        keeper = grupo[0]
        for perdedor in grupo[1:]:
            plano.append((perdedor, h, sz, keeper))

    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["remover", "md5", "tamanho", "sobrevivente"])
        w.writerows(plano)
    print(f"Duplicados binarios excedentes: {len(plano)} em {len(dups)} grupos (erros de leitura: {erros})")
    print(f"Plano: {PLANO}")

    if not EXEC:
        print("Rode com --exec para enviar a Lixeira.")
        return

    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["removido_para_lixeira", "md5", "tamanho", "sobrevivente"])
        ok = 0
        for perdedor, h, sz, keeper in plano:
            try:
                send2trash(perdedor)
                w.writerow([perdedor, h, sz, keeper])
                ok += 1
            except Exception as e:
                print(f"ERRO ao remover {perdedor}: {e}")
    print(f"Enviados a Lixeira: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
