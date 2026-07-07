# -*- coding: utf-8 -*-
r"""Org2 etapa 7: encurta nomes cujo caminho completo passa de 259 chars (limite do
Word/apps antigos). Trunca o miolo do nome preservando o inicio e o sufixo com CNJ,
se houver. Log de reversao: log_org2_encurtar.csv"""
import csv, os, re, sys

ROOT = r"C:\Users\mauri\HD_Mau"
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "log_org2_encurtar.csv")
MAX = 259
RX_CNJ_FIM = re.compile(r"( - CNJ \d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})$")

def encurtar(dirpath, nome, ocupados):
    stem, ext = os.path.splitext(nome)
    m = RX_CNJ_FIM.search(stem)
    sufixo = m.group(1) if m else ""
    miolo = stem[: len(stem) - len(sufixo)]
    orcamento = MAX - len(dirpath) - 1 - len(ext) - len(sufixo)
    if orcamento < 20:  # pasta profunda demais; corta o que der
        orcamento = 20
    novo_miolo = miolo[:orcamento].rstrip(" -,;")
    candidato = f"{novo_miolo}{sufixo}{ext}"
    i = 2
    while candidato.lower() in ocupados or os.path.exists(os.path.join(dirpath, candidato)):
        candidato = f"{novo_miolo[:orcamento-4].rstrip(' -,;')} ~{i}{sufixo}{ext}"
        i += 1
    return candidato

def main():
    execu = "--exec" in sys.argv
    plano = []
    for dirpath, _, filenames in os.walk(ROOT):
        ocupados = set(f.lower() for f in filenames)
        for nome in filenames:
            if len(os.path.join(dirpath, nome)) <= MAX:
                continue
            novo = encurtar(dirpath, nome, ocupados)
            ocupados.add(novo.lower())
            plano.append((os.path.join(dirpath, nome), os.path.join(dirpath, novo)))
    print(f"Caminhos longos a encurtar: {len(plano)}")
    if not execu:
        for de, para in plano[:5]:
            print("  DE :", os.path.basename(de))
            print("  P/ :", os.path.basename(para))
        print("Rode com --exec para renomear.")
        return
    ok = 0
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for de, para in plano:
            try:
                os.rename("\\\\?\\" + de, "\\\\?\\" + para)
                w.writerow([de, para])
                ok += 1
            except OSError as e:
                print(f"ERRO: {de}: {e}")
    print(f"Renomeados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
