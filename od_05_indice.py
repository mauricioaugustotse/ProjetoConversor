# -*- coding: utf-8 -*-
r"""OD etapa 5: regrava 00 - INDICE.md e gera 00 - CATÁLOGO.csv em OneDrive\Documentos,
e roda a verificacao final da rodada. Idempotente."""
import csv, datetime, io, os, re, unicodedata

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
BASE = os.path.dirname(os.path.abspath(__file__))
INDICE = os.path.join(ROOT, "00 - INDICE.md")
CATALOGO = os.path.join(ROOT, "00 - CATÁLOGO.csv")
PROTEGIDAS = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
              "Modelos Personalizados do Office", "My Kindle Content")
RX_DATA = re.compile(r"\b(\d{2}-\d{2}-\d{4}|\d{2}-\d{4}|\d{4}-\d{2}-\d{2})\b")

def main():
    hoje = datetime.date.today().strftime("%Y-%m-%d")
    linhas = []
    arvore = {}
    total = 0
    for dp, dn, fn in os.walk(ROOT):
        dn.sort()
        rel = os.path.relpath(dp, ROOT)
        comps = [] if rel == "." else rel.split(os.sep)
        for f in sorted(fn):
            if rel == "." and (f.startswith("00 - ") or f.lower() == "desktop.ini"):
                continue
            total += 1
            for i in range(1, min(len(comps), 2) + 1):
                k = tuple(comps[:i])
                arvore[k] = arvore.get(k, 0) + 1
            m = RX_DATA.search(f)
            linhas.append([os.path.join(rel, f) if comps else f,
                           comps[0] if comps else "", comps[1] if len(comps) > 1 else "",
                           f, m.group(1) if m else "",
                           os.path.splitext(f)[1].lower().lstrip(".")])
    with open(CATALOGO, "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo, delimiter=";")
        w.writerow(["caminho", "categoria", "subpasta", "nome", "data", "ext"])
        w.writerows(linhas)
    print(f"Catálogo: {CATALOGO} ({total} arquivos)")

    out = io.StringIO()
    out.write("# Índice - Documentos\n\n")
    out.write(f"> Acervo organizado. Atualizado em {hoje}. **{total:,} arquivos**.\n".replace(",", "."))
    out.write("> Nomes em português (com acentos), convenção `Tipo - descrição - pessoa/órgão - data`.\n\n")
    out.write("## Como encontrar\n")
    out.write("- **Navegando**: Categoria → Tema (→ Subtema; máx. 4 níveis).\n")
    out.write("- **Planilha**: `00 - CATÁLOGO.csv` — 1 linha por arquivo; filtre no Excel.\n\n")
    out.write("## Árvore\n\n")
    for comps in sorted(arvore):
        n = arvore[comps]
        indent = "  " * (len(comps) - 1)
        marc = "**" if len(comps) == 1 else ""
        out.write(f"{indent}- {marc}{comps[-1]}{marc} ({n})\n")
    out.write("\n## Histórico\n")
    out.write("- **2026-06-27**: 11 categorias, 5.095 renomeados (Opus), dedup do backup, árvore achatada.\n")
    out.write(f"- **{hoje}**: rodada de rigor — árvore a máx. 4 níveis, pastas uniformizadas, ")
    out.write("triagem+renomeação gpt-5.4 pela convenção com acentos, Luiz Celso Vieira em 09-Família. ")
    out.write("Logs de reversão: ProjetoConversor\\log_od_*.csv.\n")
    io.open(INDICE, "w", encoding="utf-8").write(out.getvalue())
    print(f"Índice: {INDICE}")

    # ---- verificacao ----
    prof_max = 0
    caps = 0
    longos = 0
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        if rel != "." and any(seg in PROTEGIDAS for seg in rel.split(os.sep)):
            dn[:] = []
            continue
        nivel = 0 if rel == "." else len(rel.split(os.sep))
        prof_max = max(prof_max, nivel)
        nome = os.path.basename(dp)
        letras = [c for c in nome if c.isalpha()]
        if rel != "." and nivel >= 2 and letras and all(c.isupper() for c in letras) and len(nome) > 6:
            caps += 1
        for f in fn:
            if len(os.path.join(dp, f)) > 240:
                longos += 1
    print(f"VERIFICAÇÃO: total={total} | profundidade máx (escopo)={prof_max} | "
          f"pastas CAPS restantes={caps} | caminhos >240={longos}")

if __name__ == "__main__":
    main()
