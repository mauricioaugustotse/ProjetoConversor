# -*- coding: utf-8 -*-
r"""OD etapa 6: encurta nomes cujo caminho passa de 240 chars (margem OneDrive/Word).
Trunca o miolo preservando inicio e a data final se houver. Log: log_od_encurtar.csv"""
import csv, os, re, sys

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "log_od_encurtar.csv")
MAX = 240
PRE = "\\\\?\\"
PROT = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
        "Modelos Personalizados do Office", "My Kindle Content")
RX_FIM = re.compile(r"( - (?:\d{2}-\d{2}-\d{4}|\d{2}-\d{4}|\d{4}(?:-\d{2}-\d{2})?))$")

def main():
    execu = "--exec" in sys.argv
    plano = []
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        if rel != "." and any(seg in PROT for seg in rel.split(os.sep)):
            dn[:] = []
            continue
        ocupados = {f.lower() for f in fn}
        for nome in fn:
            if len(os.path.join(dp, nome)) <= MAX:
                continue
            stem, ext = os.path.splitext(nome)
            m = RX_FIM.search(stem)
            suf = m.group(1) if m else ""
            miolo = stem[: len(stem) - len(suf)]
            orc = MAX - len(dp) - 1 - len(ext) - len(suf)
            novo = miolo[: max(8, orc)].rstrip(" -.,;") + suf + ext
            i = 2
            while novo.lower() in ocupados:
                novo = f"{miolo[: max(20, orc - 4)].rstrip(' -.,;')} ({i}){suf}{ext}"
                i += 1
            ocupados.add(novo.lower())
            plano.append((os.path.join(dp, nome), os.path.join(dp, novo)))
    print("A encurtar:", len(plano))
    if not execu:
        for de, para in plano[:5]:
            print("  DE:", os.path.basename(de)[:100])
            print("  P/:", os.path.basename(para)[:100])
        return
    ok = 0
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for de, para in plano:
            try:
                os.rename(PRE + de, PRE + para)
                w.writerow([de, para])
                ok += 1
            except OSError as e:
                print("ERRO:", de, e)
    print(f"Encurtados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
