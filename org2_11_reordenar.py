# -*- coding: utf-8 -*-
r"""Org2 etapa 11: re-endereca arquivos renomeados pela IA (org2_10) cuja subpasta de
faixa deixou de corresponder ao nome novo.

- Faixa ALFABETICA (rotulos "0-C", "Ci-Co", "D-J"...): se a chave do nome novo cair em
  outra faixa irma existente, move para la.
- "Sem ano": se o nome novo tem ano, move para a faixa de ano irma que o cobre
  (cria pasta "AAAA" se nao houver).
So processa os arquivos listados em log_org2_nomes_ia.csv (destinos).
Uso: python org2_11_reordenar.py [--exec]   (log: log_org2_reordenar.csv)
"""
import csv, io, os, re, sys
from collections import Counter

ROOT = r"C:\Users\mauri\HD_Mau"
BASE = os.path.dirname(os.path.abspath(__file__))
LOG_IA = os.path.join(BASE, "log_org2_nomes_ia.csv")
PLANO = os.path.join(BASE, "plano_org2_reordenar.csv")
LOG = os.path.join(BASE, "log_org2_reordenar.csv")

from org2_04_quebra_pastas import ano_do_nome

RX_FAIXA_ALFA = re.compile(r"^([0-9A-Za-z]{1,2})-([0-9A-Za-z]{1,2})$")
RX_FAIXA_ANO = re.compile(r"^(\d{4})(?:-(\d{4}))?$")

# faixas construidas pela letra APOS o prefixo do tipo (regras da etapa 4)
PREFIXOS_FAIXA = {
    "Contestacoes": re.compile(r"^contesta\w*", re.I),
    "Peticoes diversas": re.compile(r"^pet(icao)?\b", re.I),
}

def chave_alfa(nome, largura, pai_faixa):
    s = nome
    rx = PREFIXOS_FAIXA.get(pai_faixa)
    if rx:
        s = rx.sub("", s, count=1)
    s = re.sub(r"[^0-9A-Za-z]", "", s.upper())
    return (s[:largura] or "0").ljust(largura, "0")

def faixa_alfa_cobre(rotulo, nome, pai_faixa):
    m = RX_FAIXA_ALFA.match(rotulo)
    if not m:
        return None
    ini, fim = m.group(1).upper(), m.group(2).upper()
    larg = max(len(ini), len(fim))
    k = chave_alfa(nome, larg, pai_faixa)
    return ini.ljust(larg, "0") <= k <= fim.ljust(larg, "Z")

def main():
    execu = "--exec" in sys.argv
    if not os.path.exists(LOG_IA):
        print("Sem log da etapa 10; nada a fazer.")
        return
    renomeados = [r[1] for r in list(csv.reader(io.open(LOG_IA, encoding="utf-8")))[1:]]
    plano = []
    stats = Counter()
    for p in renomeados:
        if not os.path.exists(p):
            stats["sumiu"] += 1
            continue
        dirpath, nome = os.path.split(p)
        rotulo = os.path.basename(dirpath)
        pai = os.path.dirname(dirpath)
        irmas = [d for d in os.listdir(pai) if os.path.isdir(os.path.join(pai, d))]

        if RX_FAIXA_ALFA.match(rotulo):
            pai_faixa = os.path.basename(pai)
            if faixa_alfa_cobre(rotulo, nome, pai_faixa):
                stats["alfa ok"] += 1
                continue
            destino = next((d for d in irmas if d != rotulo and faixa_alfa_cobre(d, nome, pai_faixa)), None)
            if destino:
                plano.append((p, os.path.join(pai, destino, nome)))
                stats["alfa movido"] += 1
            else:
                stats["alfa sem faixa"] += 1
        elif rotulo == "Sem ano":
            ano = ano_do_nome(nome)  # sem fallback mtime: so ano explicito no nome novo
            if not ano:
                stats["sem ano mesmo"] += 1
                continue
            destino = None
            for d in irmas:
                m = RX_FAIXA_ANO.match(d)
                if m and int(m.group(1)) <= ano <= int(m.group(2) or m.group(1)):
                    destino = d
                    break
            destino = destino or str(ano)
            plano.append((p, os.path.join(pai, destino, nome)))
            stats["ano movido"] += 1
        else:
            stats["pasta normal"] += 1

    print("Stats:", dict(stats))
    print(f"Plano: {len(plano)} re-enderecamentos")
    for de, para in plano[:8]:
        print("  ", os.path.relpath(de, ROOT)[:60], "->", os.path.dirname(os.path.relpath(para, ROOT))[-45:])
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    if not execu:
        print("Dry-run. Rode com --exec para mover.")
        return
    ok = 0
    vazias = set()
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for de, para in plano:
            try:
                os.makedirs(os.path.dirname(para), exist_ok=True)
                dest = para
                i = 2
                while os.path.exists("\\\\?\\" + dest):
                    stem, ext = os.path.splitext(para)
                    dest = f"{stem} ({i}){ext}"
                    i += 1
                os.rename("\\\\?\\" + de, "\\\\?\\" + dest)
                w.writerow([de, dest])
                vazias.add(os.path.dirname(de))
                ok += 1
            except OSError as e:
                print(f"ERRO: {de}: {e}")
    for d in vazias:
        try:
            os.rmdir(d)
            print("pasta vazia removida:", os.path.relpath(d, ROOT))
        except OSError:
            pass
    print(f"Movidos: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
