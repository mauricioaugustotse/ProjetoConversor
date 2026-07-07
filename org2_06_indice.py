# -*- coding: utf-8 -*-
r"""Org2 etapa 6: regrava 00 - INDICE.md (arvore com contagens) e gera 00 - CATALOGO.csv
(uma linha por arquivo: caminho, categoria, tipo, materia, nome, classe, CNJ, ano, ext).
Idempotente: rode apos qualquer reorganizacao."""
import csv, datetime, io, os, re
from collections import Counter

ROOT = r"C:\Users\mauri\HD_Mau"
INDICE = os.path.join(ROOT, "00 - INDICE.md")
CATALOGO = os.path.join(ROOT, "00 - CATALOGO.csv")

RX_CNJ = re.compile(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")
RX_ANO = [re.compile(r"\.((?:19|20)\d{2})\.\d\.\d{2}\."),
          re.compile(r"[Ee]leicoes (\d{4})"),
          re.compile(r"\b\d{1,2}-\d{1,2}-((?:19|20)\d{2})\b"),
          re.compile(r"\b\d{1,2}-((?:19|20)\d{2})\b"),
          re.compile(r"\b((?:19|20)\d{2})\b")]
SIGLAS = ["AgR-REspe", "AgR-AI", "AgR-RO", "E-ED-RR", "AgRg", "AIRR", "E-RR", "EDcl",
          "REspe", "RESPE", "ROEl", "RCED", "RMS", "AgR", "AC", "AI", "AR", "CC",
          "CTA", "Cta", "ED", "HC", "MS", "PA", "PC", "PP", "Pet", "RO", "RP", "RR", "Rcl"]
RX_CLASSE = re.compile(r"(?:^|[\s.\-])(" + "|".join(re.escape(s) for s in SIGLAS) + r")(?=[\s\d.\-])")

def ano_de(nome):
    for rx in RX_ANO:
        m = rx.search(nome)
        if m:
            a = int(m.group(1))
            if 1990 <= a <= 2026:
                return a
    return ""

def main():
    linhas_cat = []
    arvore = Counter()   # tupla de componentes -> n arquivos (recursivo)
    total = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames.sort()
        rel = os.path.relpath(dirpath, ROOT)
        comps = [] if rel == "." else rel.split(os.sep)
        for n in sorted(filenames):
            if comps == [] and n.startswith(("00 - ", "desktop.ini")):
                continue
            total += 1
            for i in range(1, len(comps) + 1):
                arvore[tuple(comps[:i])] += 1
            m = RX_CLASSE.search(n)
            linhas_cat.append([
                os.path.join(rel, n) if comps else n,
                comps[0] if len(comps) > 0 else "",
                comps[1] if len(comps) > 1 else "",
                comps[2] if len(comps) > 2 else "",
                n,
                m.group(1) if m else "",
                (RX_CNJ.search(n) or [""])[0] if RX_CNJ.search(n) else "",
                ano_de(n),
                os.path.splitext(n)[1].lower().lstrip("."),
            ])

    with open(CATALOGO, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["caminho", "categoria", "tipo", "materia", "nome", "classe", "cnj", "ano", "ext"])
        w.writerows(linhas_cat)
    print(f"Catalogo: {CATALOGO} ({total} arquivos)")

    hoje = datetime.date.today().strftime("%Y-%m-%d")
    out = io.StringIO()
    out.write("# Indice - HD_Mau\n\n")
    out.write(f"> Visao global do acervo. Atualizado em {hoje}. **{total:,} arquivos**.\n".replace(",", "."))
    out.write("> Nomes padronizados sem acentos (ASCII), pesquisaveis pelo Explorer.\n\n")
    out.write("## Como encontrar\n")
    out.write("- **Navegando**: Categoria -> Tipo -> Materia -> (Classe processual -> Ano, nos acervos grandes).\n")
    out.write("- **Buscando**: numeros de processo CNJ estao no NOME de milhares de arquivos; use a busca do Explorer.\n")
    out.write("- **Planilha**: `00 - CATALOGO.csv` (nesta pasta) tem 1 linha por arquivo com categoria, classe, CNJ e ano - filtre no Excel.\n\n")
    out.write("## Convencao de nomes (fase 3, 07/2026)\n")
    out.write("- ASCII sem acentos; segmentos separados por \" - \"; siglas canonicas (REspe, AgR, CNJ, TSE...); descricao em minusculas; nomes proprios capitalizados; datas DD-MM-AAAA ou MM-AAAA.\n")
    out.write("- Juridico: `CLASSE numero - UF - assunto - desfecho - MM-AAAA - CNJ ...`\n")
    out.write("- Administrativo: `Tipo doc - assunto - orgao/pessoa - data`; Pessoal: `Tipo - pessoa - detalhe - data`; Estudos: `Tema/titulo - autor - ano`.\n")
    out.write("- Nomes cripticos/genericos foram reescritos a partir do CONTEUDO (IA); logs de reversao em ProjetoConversor\\log_org2_nomes_*.csv.\n\n")
    out.write("## Arvore\n\n")
    for comps in sorted(arvore):
        if len(comps) > 3:
            continue
        n = arvore[comps]
        nome = comps[-1]
        indent = "  " * (len(comps) - 1)
        out.write(f"{indent}- **{nome}** ({n})\n" if len(comps) == 1 else f"{indent}- {nome} ({n})\n")
    out.write("\n## Historico\n")
    out.write("- **Fase 1 (2026-06-21/22)**: 6 categorias, subpastas por tipo/materia, ~29 mil nomes padronizados, ")
    out.write("texto extraido de ~36 mil docs, OCR nos escaneados, 2.498 reclassificados por conteudo (IA), CNJ no nome de 3.170.\n")
    out.write(f"- **Fase 2 ({hoje})**: dedup binario + dedup por CONTEUDO (texto integral), pastas gigantes divididas ")
    out.write("(classe processual -> ano; tipo de documento; faixa de anos), nova rodada de IA nos genericos, catalogo CSV.\n")
    out.write("- Scripts, planos e logs de reversao: `C:\\Users\\mauri\\ProjetoConversor\\org2_*` e anteriores.\n")

    io.open(INDICE, "w", encoding="utf-8").write(out.getvalue())
    print(f"Indice: {INDICE}")

if __name__ == "__main__":
    main()
