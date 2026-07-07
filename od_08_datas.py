# -*- coding: utf-8 -*-
r"""OD etapa 8: padroniza datas nos nomes e conserta artefatos "manter" do od_04.

1) Data duplicada equivalente (DD-MM-AAAA + AAAA-MM-DD do MESMO dia) -> mantem so a BR.
2) Data ISO solta (AAAA-MM-DD) -> converte para DD-MM-AAAA (padrao da convencao).
3) Stems que viraram "manter[ - ...]" (resposta da IA vazou como nome) -> restaura o stem
   original do log_od_arquivos.csv e aplica as normalizacoes de data.
Log: log_od_datas.csv. Pasta 12 e protegidas fora.

Uso: python od_08_datas.py [--exec]
"""
import csv, io, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\mauri\OneDrive\Documentos"
LOG_ARQ = os.path.join(BASE, "log_od_arquivos.csv")
LOG = os.path.join(BASE, "log_od_datas.csv")
PRE = "\\\\?\\"
PROT = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
        "Modelos Personalizados do Office", "My Kindle Content")
RX_BR = re.compile(r"\b(\d{2})-(\d{2})-(20\d{2})\b")
RX_ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")

def corrigir_datas(stem):
    brs = {(d, m, a) for d, m, a in RX_BR.findall(stem) if int(m) <= 12 and int(d) <= 31}
    def troca_iso(m):
        a, mo, d = m.group(1), m.group(2), m.group(3)
        if int(mo) > 12 or int(d) > 31:
            return m.group(0)
        if (d, mo, a) in brs:
            return ""  # redundante: BR equivalente ja esta no nome
        return f"{d}-{mo}-{a}"
    s = RX_ISO.sub(troca_iso, stem)
    # datas com espacos "17 11 2025" -> "17-11-2025"
    def troca_esp(m):
        d, mo, a = m.group(1), m.group(2), m.group(3)
        if int(mo) <= 12 and int(d) <= 31:
            return f"{d}-{mo}-{a}"
        return m.group(0)
    s = re.sub(r"\b(\d{2}) (\d{2}) (20\d{2})\b", troca_esp, s)
    s = re.sub(r"\s*-\s*(?=-|$)", "", s)      # separadores orfaos
    s = re.sub(r"\s{2,}", " ", s).strip(" -.,;")
    return s or stem

def main():
    execu = "--exec" in sys.argv
    # mapa de restauracao p/ artefatos "manter": destino atual -> stem original
    original = {}
    if os.path.exists(LOG_ARQ):
        for de, para in list(csv.reader(io.open(LOG_ARQ, encoding="utf-8")))[1:]:
            original[para] = os.path.splitext(os.path.basename(de))[0]

    plano = []
    artefatos = 0
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        if rel != "." and any(seg in PROT for seg in rel.split(os.sep)):
            dn[:] = []
            continue
        for f in fn:
            if f.startswith("00 - "):
                continue
            p = os.path.join(dp, f)
            stem, ext = os.path.splitext(f)
            novo = stem
            if re.fullmatch(r"manter([ \-].*)?", stem, re.I):
                base_orig = original.get(p)
                if base_orig and not re.fullmatch(r"manter([ \-].*)?", base_orig, re.I):
                    novo = base_orig
                    artefatos += 1
                else:
                    resto = re.sub(r"^manter[ \-]*", "", stem, flags=re.I).strip()
                    novo = resto or stem
                    artefatos += 1
            novo = corrigir_datas(novo)
            if novo != stem:
                plano.append((p, os.path.join(dp, novo + ext)))
    assert not any("12 - Consultoria Legislativa" in a for a, b in plano), "GUARDA: pasta 12!"
    print(f"Renomeacoes: {len(plano)} (artefatos 'manter' restaurados: {artefatos})")
    for de, para in plano[:8]:
        print("  DE:", os.path.basename(de)[:100])
        print("  P/:", os.path.basename(para)[:100])
    if not execu:
        print("Dry-run. Rode com --exec.")
        return
    ok = 0
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for de, para in plano:
            try:
                dest = para
                i = 2
                while os.path.exists(PRE + dest):
                    s, e = os.path.splitext(para)
                    dest = f"{s} ({i}){e}"
                    i += 1
                os.rename(PRE + de, PRE + dest)
                w.writerow([de, dest])
                ok += 1
            except OSError as e:
                print("ERRO:", de, e)
    print(f"Renomeados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
