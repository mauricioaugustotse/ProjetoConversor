# -*- coding: utf-8 -*-
r"""OD etapa 3: extrai cabecalhos (~1.200 chars) dos textuais de OneDrive\Documentos.
Blindado: subprocess isolado com timeout (extrair_isolado); PDFs so 2 paginas.
Cache retomavel od_cabecalhos.tsv. txt/md/csv/html lidos direto."""
import io, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_15_nao_processuais import extrair_isolado

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
CAB = os.path.join(BASE, "od_cabecalhos.tsv")
PROIBIDAS = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
             "Modelos Personalizados do Office", "My Kindle Content")
EXTS_SUB = (".pdf", ".doc", ".docx", ".rtf")
EXTS_TXT = (".txt", ".md", ".csv", ".html", ".htm")

def protegida(path):
    rel = os.path.relpath(path, ROOT)
    return any(rel.startswith(p) or (os.sep + p + os.sep) in rel for p in PROIBIDAS)

def ler_direto(p):
    try:
        raw = io.open(p, encoding="utf-8", errors="ignore").read(4000)
        raw = re.sub(r"<[^>]+>", " ", raw) if p.lower().endswith((".html", ".htm")) else raw
        return re.sub(r"\s+", " ", raw).strip()[:1200]
    except OSError:
        return ""

def main():
    alvos = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if not protegida(os.path.join(dp, d))]
        for f in fn:
            if f.startswith("00 - ") or f.lower() == "desktop.ini":
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in EXTS_SUB + EXTS_TXT:
                alvos.append(os.path.join(dp, f))
    print("Textuais no escopo:", len(alvos))

    cache = {}
    if os.path.exists(CAB):
        for linha in io.open(CAB, encoding="utf-8"):
            p = linha.rstrip("\n").split("\t")
            if len(p) == 3:
                cache[p[0]] = p[1]
    pend = []
    for p in alvos:
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = f"{st.st_size}|{int(st.st_mtime)}"
        if cache.get(p) != key:
            pend.append((p, key))
    print(f"Em cache: {len(alvos)-len(pend)}; extraindo: {len(pend)}")

    t0 = time.time()
    def w(item):
        p, key = item
        if p.lower().endswith(EXTS_TXT):
            return p, key, ler_direto(p)
        return p, key, extrair_isolado(p)
    with ThreadPoolExecutor(max_workers=6) as ex, io.open(CAB, "a", encoding="utf-8") as f:
        for i, (p, key, t) in enumerate(ex.map(w, pend), 1):
            f.write(f"{p}\t{key}\t{t}\n")
            if i % 500 == 0:
                print(f"  {i}/{len(pend)} ({time.time()-t0:.0f}s)")
                f.flush()
    print("Concluido.")

if __name__ == "__main__":
    main()
