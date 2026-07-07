# -*- coding: utf-8 -*-
r"""OD etapa 1: snapshot completo de OneDrive\Documentos (caminho, tamanho, mtime)."""
import csv, os, time

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "od_snapshot_antes.csv")

def main():
    rows = 0
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caminho", "tamanho", "mtime"])
        for dirpath, dirnames, filenames in os.walk(ROOT):
            for name in filenames:
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                w.writerow([p, st.st_size, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))])
                rows += 1
    print(f"Snapshot: {OUT} ({rows} arquivos)")

if __name__ == "__main__":
    main()
