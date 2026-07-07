# -*- coding: utf-8 -*-
r"""Org2 etapa 8: refino das 14 pastas que ficaram com >400 arquivos apos a quebra.

- REspe\2016 e AgR\2016-2017 (anos unicos gigantes): subpastas irmas "ANO - Tema"
  (tema por palavra-chave no nome; residual dos AgR por subclasse AgR-REspe/AgR-AI).
- Faixas multi-ano >400: re-particiona por ano com alvo 250.
- Faixas alfabeticas "0-C" >400: re-particiona com chave de 2 caracteres, alvo 250.

As novas pastas substituem a antiga (mesmo nivel); a antiga e removida se esvaziar.
Uso: python org2_08_refino.py [--exec]   (log: log_org2_refino.csv)
"""
import csv, datetime, os, re, sys
from collections import Counter, defaultdict

ROOT = r"C:\Users\mauri\HD_Mau"
BASE = os.path.dirname(os.path.abspath(__file__))
PLANO = os.path.join(BASE, "plano_org2_refino.csv")
LOG = os.path.join(BASE, "log_org2_refino.csv")
ALVO = 250

from org2_04_quebra_pastas import ano_do_nome, faixas  # mesmos criterios da etapa 4

TEMAS = [
    ("Prestacao de contas", ["prestacao de contas", "contas", "doacao", "fundo partidario",
                             "gasto", "arrecadacao", "recursos de campanha"]),
    ("Registro de candidatura", ["registro", "rrc", "rcc", "drap", "elegibilidade", "inelegib",
                                 "quitacao", "filiacao", "desincompatibiliza"]),
    ("Abuso e conduta vedada", ["abuso", "conduta vedada", "aije", "aime", "captacao ilicita",
                                "compra de votos", "corrupcao", "41-a"]),
    ("Propaganda", ["propaganda", "direito de resposta", "horario eleitoral", "impulsionamento",
                    "outdoor", "banner", "showmicio"]),
]

def tema_de(nome):
    lw = nome.lower()
    for t, kws in TEMAS:
        if any(k in lw for k in kws):
            return t
    return None

def refino_tema_ano(dirpath, arquivos, ano, residual_por_subclasse):
    """-> nome_arquivo -> rotulo de pasta irma 'ANO - ...'."""
    out = {}
    for n in arquivos:
        t = tema_de(n)
        if t:
            out[n] = f"{ano} - {t}"
        elif not residual_por_subclasse:
            out[n] = f"{ano} - Diversos"
        else:
            lw = n.lower()
            if "agr-respe" in lw or "agr no respe" in lw:
                out[n] = f"{ano} - AgR-REspe diversos"
            elif "agr-ai" in lw or "agr no ai" in lw:
                out[n] = f"{ano} - AgR-AI diversos"
            else:
                out[n] = f"{ano} - Outros diversos"
    return out

def faixas_max(cont, maximo=380, fmt=str):
    """Agrupa chaves consecutivas SEM ultrapassar `maximo` por faixa (fecha antes de estourar)."""
    mapa = {}
    grupo, total = [], 0
    for k in sorted(cont):
        if grupo and total + cont[k] > maximo:
            rot = fmt(grupo[0]) if len(grupo) == 1 else f"{fmt(grupo[0])}-{fmt(grupo[-1])}"
            for g in grupo:
                mapa[g] = rot
            grupo, total = [], 0
        grupo.append(k)
        total += cont[k]
    if grupo:
        rot = fmt(grupo[0]) if len(grupo) == 1 else f"{fmt(grupo[0])}-{fmt(grupo[-1])}"
        for g in grupo:
            mapa[g] = rot
    return mapa

def refino_por_ano(dirpath, arquivos):
    anos = {}
    for n in arquivos:
        anos[n] = ano_do_nome(n, os.path.getmtime(os.path.join(dirpath, n)))
    cont = Counter(a for a in anos.values() if a)
    mapa = faixas_max(cont) if cont else {}
    return {n: (mapa[a] if a else "Sem ano") for n, a in anos.items()}

def chave2(nome):
    s = re.sub(r"[^0-9A-Za-z]", "", nome.upper())
    return (s[:2] or "0").ljust(2, "0")

def refino_alfabetico2(dirpath, arquivos):
    cont = Counter(chave2(n) for n in arquivos)
    fmt = lambda k: k[0] + k[1:].lower()
    mapa = faixas(cont, alvo=ALVO, fmt=fmt)
    return {n: mapa[chave2(n)] for n in arquivos}

ALVOS_REFINO = [
    (r"01 - Juridico\Recursos\Eleitoral\REspe - Recurso especial\2016",
     lambda d, a: refino_tema_ano(d, a, "2016", residual_por_subclasse=False)),
    (r"01 - Juridico\Recursos\Eleitoral\AgR - Agravos regimentais\2016",
     lambda d, a: refino_tema_ano(d, a, "2016", residual_por_subclasse=True)),
    (r"01 - Juridico\Recursos\Eleitoral\AgR - Agravos regimentais\2017",
     lambda d, a: refino_tema_ano(d, a, "2017", residual_por_subclasse=True)),
    (r"03 - Administrativo\Outros\2012-2013", refino_por_ano),
    (r"03 - Administrativo\Outros\1997-2011", refino_por_ano),
    (r"03 - Administrativo\Outros\2014-2016", refino_por_ano),
    (r"03 - Administrativo\Listas e planilhas\2015-2017", refino_por_ano),
    (r"04 - Pessoal e financeiro\Comprovantes e recibos\2015-2016", refino_por_ano),
    (r"01 - Juridico\Recursos\Eleitoral\AI - Agravo de instrumento\2017-2018", refino_por_ano),
    (r"01 - Juridico\Recursos\Eleitoral\REspe - Recurso especial\2004-2014", refino_por_ano),
    (r"01 - Juridico\Recursos\Eleitoral\ED - Embargos de declaracao\1999-2016", refino_por_ano),
    (r"01 - Juridico\Recursos\Eleitoral\AgR - Agravos regimentais\1999-2015", refino_por_ano),
    (r"05 - Diversos (a revisar)\Outros\0-C", refino_alfabetico2),
    (r"02 - Estudos e concursos\Atualidades e Outros\0-C", refino_alfabetico2),
]

def main():
    execu = "--exec" in sys.argv
    plano = []
    for rel, regra in ALVOS_REFINO:
        d = os.path.join(ROOT, rel)
        if not os.path.isdir(d):
            print(f"AVISO: nao existe {rel}")
            continue
        arquivos = [n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n))]
        rotulos = regra(d, arquivos)
        pai = os.path.dirname(d)
        atual = os.path.basename(d)
        cont = Counter(rotulos.values())
        # se a re-particao devolver rotulo unico igual ao atual, nada a fazer
        if len(cont) == 1 and atual in cont:
            print(f"=== {rel}: sem melhoria possivel, mantido")
            continue
        print(f"=== {rel} ({len(arquivos)} arqs)")
        for r, n in sorted(cont.items()):
            print(f"    {r:45s} {n}")
        for n, rot in rotulos.items():
            plano.append((os.path.join(d, n), os.path.join(pai, rot, n)))

    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origem", "destino"])
        w.writerows(plano)
    print(f"\nPlano: {len(plano)} movimentacoes -> {PLANO}")
    if not execu:
        print("Dry-run. Rode com --exec para mover.")
        return

    ok, erros = 0, 0
    vazias = set()
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for origem, destino in plano:
            try:
                base_d, nome_d = os.path.split(destino)
                os.makedirs(base_d, exist_ok=True)
                dest = destino
                i = 2
                while os.path.exists("\\\\?\\" + dest):
                    stem, ext = os.path.splitext(nome_d)
                    dest = os.path.join(base_d, f"{stem} ({i}){ext}")
                    i += 1
                os.rename("\\\\?\\" + origem, "\\\\?\\" + dest)
                w.writerow([origem, dest])
                vazias.add(os.path.dirname(origem))
                ok += 1
            except OSError as e:
                print(f"ERRO: {origem}: {e}")
                erros += 1
    for d in vazias:
        try:
            os.rmdir(d)  # so remove se vazia
            print(f"pasta vazia removida: {os.path.relpath(d, ROOT)}")
        except OSError:
            pass
    print(f"Movidos: {ok} (erros: {erros}). Log: {LOG}")

if __name__ == "__main__":
    main()
