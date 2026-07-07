# -*- coding: utf-8 -*-
"""Org2 etapa 4: quebra das pastas-folha gigantes de HD_Mau em subpastas navegaveis.

Regras por pasta (calibradas na analise de 05/07/2026):
- Juridico: classe processual -> faixa de anos (alvo ~300/pasta; ano unico pode exceder).
- Contestacoes/Peticoes trabalhistas: tipo de documento -> faixa alfabetica.
- Despachos e votos: tipo (despacho/voto/decisao) -> faixa de anos.
- Administrativo e Comprovantes: faixa de anos (ano do nome; fallback mtime).
- Documentos pessoais: tipo por palavra-chave.
- Estudos\Direito Eleitoral: Julgados (por classe) / Doutrina / Cursos / Outros.

Uso: python org2_04_quebra_pastas.py         -> dry-run (contagens por subpasta)
     python org2_04_quebra_pastas.py --exec  -> move + log_org2_quebra.csv (reversao)
"""
import csv, datetime, os, re, sys
from collections import Counter, defaultdict

ROOT = r"C:\Users\mauri\HD_Mau"
BASE = os.path.dirname(os.path.abspath(__file__))
PLANO = os.path.join(BASE, "plano_org2_quebra.csv")
LOG = os.path.join(BASE, "log_org2_quebra.csv")
ALVO = 300  # tamanho-alvo de pasta-folha

# --- classes processuais (ordem: mais especifica primeiro) ---
SIGLAS = ["AgR-REspe", "AgR-AI", "AgR-RO", "E-ED-RR", "AgRg", "AIRR", "E-RR", "EDcl",
          "REspe", "RESPE", "ROEl", "RCED", "RMS", "AgR", "AC", "AI", "AR", "CC",
          "CTA", "Cta", "ED", "HC", "MS", "PA", "PC", "PP", "Pet", "RO", "RP", "RR", "Rcl"]
RX_CLASSE = re.compile(r"(?:^|[\s.\-])(" + "|".join(re.escape(s) for s in SIGLAS) + r")(?=[\s\d.\-])")

GRUPO_ELEITORAL = {
    "REspe": "REspe - Recurso especial", "RESPE": "REspe - Recurso especial",
    "AgR-REspe": "AgR - Agravos regimentais", "AgR-AI": "AgR - Agravos regimentais",
    "AgR-RO": "AgR - Agravos regimentais", "AgR": "AgR - Agravos regimentais",
    "AgRg": "AgR - Agravos regimentais",
    "AI": "AI - Agravo de instrumento",
    "ED": "ED - Embargos de declaracao", "EDcl": "ED - Embargos de declaracao",
    "RO": "RO - Recurso ordinario", "ROEl": "RO - Recurso ordinario",
}
GRUPO_TRABALHISTA = {
    "RO": "RO - Recurso ordinario", "RR": "RR - Recurso de revista",
    "E-RR": "RR - Recurso de revista", "E-ED-RR": "RR - Recurso de revista",
    "ED": "ED - Embargos de declaracao", "EDcl": "ED - Embargos de declaracao",
    "AI": "AI e AIRR - Agravos", "AIRR": "AI e AIRR - Agravos",
}
GRUPO_ACORDAOS = {
    "AI": "AI e agravos", "AgR": "AI e agravos", "AgR-AI": "AI e agravos",
    "AgR-REspe": "REspe", "REspe": "REspe",
    "MS": "MS e RMS", "RMS": "MS e RMS",
    "PP": "PP e PC - Partidos e contas", "PC": "PP e PC - Partidos e contas",
}

RX_ANO = [re.compile(r"\.((?:19|20)\d{2})\.\d\.\d{2}\."),        # CNJ
          re.compile(r"[Ee]leicoes (\d{4})"),
          re.compile(r"\b\d{1,2}-\d{1,2}-((?:19|20)\d{2})\b"),   # dd-mm-yyyy
          re.compile(r"\b\d{1,2}-((?:19|20)\d{2})\b"),           # mm-yyyy
          re.compile(r"\b((?:19|20)\d{2})\b")]                   # ano isolado
RX_ANO2 = re.compile(r"\b\d{1,2}-\d{1,2}-(\d{2})\b")             # dd-mm-yy

def classe(nome):
    m = RX_CLASSE.search(nome)
    return m.group(1) if m else None

def ano_do_nome(nome, mtime=None):
    for rx in RX_ANO:
        m = rx.search(nome)
        if m:
            a = int(m.group(1))
            if 1990 <= a <= 2026:
                return a
    m = RX_ANO2.search(nome)
    if m:
        a = int(m.group(1))
        return 2000 + a if a <= 26 else 1900 + a
    if mtime:
        a = datetime.datetime.fromtimestamp(mtime).year
        if 1990 <= a <= 2026:
            return a
    return None

def faixas(chaves_contagem, alvo=ALVO, fmt=str):
    """Agrupa chaves ordenaveis em faixas de ~alvo itens. Retorna chave -> rotulo."""
    mapa = {}
    grupo, total = [], 0
    ordenadas = sorted(chaves_contagem)
    for k in ordenadas:
        grupo.append(k)
        total += chaves_contagem[k]
        if total >= alvo:
            rot = fmt(grupo[0]) if len(grupo) == 1 else f"{fmt(grupo[0])}-{fmt(grupo[-1])}"
            for g in grupo:
                mapa[g] = rot
            grupo, total = [], 0
    if grupo:
        rot = fmt(grupo[0]) if len(grupo) == 1 else f"{fmt(grupo[0])}-{fmt(grupo[-1])}"
        for g in grupo:
            mapa[g] = rot
    return mapa

def com_faixa_de_ano(itens, nivel1):
    """itens: [(nome, mtime)] -> destino [nivel1, faixa_ano]. Calibra faixas por nivel1."""
    anos = {}
    for nome, mt in itens:
        anos[nome] = ano_do_nome(nome, mt)
    cont = Counter(a for a in anos.values() if a)
    mapa = faixas(cont) if cont else {}
    out = {}
    for nome, mt in itens:
        a = anos[nome]
        out[nome] = [nivel1, mapa[a] if a else "Sem ano"]
    return out

def com_faixa_alfabetica(itens, nivel1, prefixo_rx=None):
    """Subdivide por letra inicial (apos remover prefixo do tipo) em faixas ~ALVO."""
    def letra(nome):
        s = nome
        if prefixo_rx:
            s = prefixo_rx.sub("", s, count=1)
        s = re.sub(r"^[\s\-_.,]+", "", s)
        for c in s.upper():
            if c.isalnum():
                return "0" if c.isdigit() else c
        return "0"
    cont = Counter(letra(n) for n, _ in itens)
    if sum(cont.values()) <= ALVO * 1.4:
        return {n: [nivel1] for n, _ in itens}
    mapa = faixas(cont)
    return {n: [nivel1, mapa[letra(n)]] for n, _ in itens}

# ------------------------------------------------------------------ regras

def regra_recursos_eleitoral(itens):
    por_grupo = defaultdict(list)
    for nome, mt in itens:
        c = classe(nome)
        por_grupo[GRUPO_ELEITORAL.get(c, "Outras classes")].append((nome, mt))
    out = {}
    for g, sub in por_grupo.items():
        out.update(com_faixa_de_ano(sub, g))
    return out

def regra_recursos_trabalhista(itens):
    out = {}
    for nome, mt in itens:
        c = classe(nome)
        out[nome] = [GRUPO_TRABALHISTA.get(c, "Outras classes e diversos")]
    return out

def regra_contestacoes_trab(itens):
    tipos = defaultdict(list)
    for nome, mt in itens:
        lw = nome.lower()
        if lw.startswith("contestacao") or lw.startswith("contesta"):
            tipos["Contestacoes"].append((nome, mt))
        elif "preposicao" in lw or lw.startswith("carta"):
            tipos["Cartas de preposicao"].append((nome, mt))
        elif lw.startswith(("manifestacao", "impugnacao", "razoes", "contrarrazoes", "contraminuta")):
            tipos["Manifestacoes e impugnacoes"].append((nome, mt))
        else:
            tipos["Outros"].append((nome, mt))
    out = {}
    out.update(com_faixa_alfabetica(tipos.pop("Contestacoes", []), "Contestacoes",
                                    re.compile(r"^contesta\w*", re.I)))
    for t, sub in tipos.items():
        out.update({n: [t] for n, _ in sub})
    return out

def regra_peticoes_trab(itens):
    tipos = defaultdict(list)
    for nome, mt in itens:
        lw = nome.lower()
        if lw.startswith(("pet", "peticao")):
            tipos["Peticoes diversas"].append((nome, mt))
        elif lw.startswith(("ms", "mandado de seguranca")):
            tipos["MS - Mandado de seguranca"].append((nome, mt))
        elif lw.startswith(("acao", "interpelacao", "reclamacao")):
            tipos["Acoes e medidas"].append((nome, mt))
        else:
            tipos["Outros"].append((nome, mt))
    out = {}
    out.update(com_faixa_alfabetica(tipos.pop("Peticoes diversas", []), "Peticoes diversas",
                                    re.compile(r"^pet(icao)?\b[\s\-]*", re.I)))
    for t, sub in tipos.items():
        out.update({n: [t] for n, _ in sub})
    return out

def regra_despachos_eleitoral(itens):
    tipos = defaultdict(list)
    for nome, mt in itens:
        lw = nome.lower()
        if "despacho" in lw:
            tipos["Despachos"].append((nome, mt))
        elif "voto" in lw or "relatorio e voto" in lw:
            tipos["Votos e relatorios"].append((nome, mt))
        elif "decisao" in lw or "decisao monocratica" in lw:
            tipos["Decisoes"].append((nome, mt))
        else:
            tipos["Outros"].append((nome, mt))
    out = {}
    for t, sub in tipos.items():
        if len(sub) > ALVO * 1.4:
            out.update(com_faixa_de_ano(sub, t))
        else:
            out.update({n: [t] for n, _ in sub})
    return out

def regra_acordaos_eleitoral(itens):
    out = {}
    for nome, mt in itens:
        c = classe(nome)
        out[nome] = [GRUPO_ACORDAOS.get(c, "Outras classes")]
    return out

def regra_por_ano(itens):
    return com_faixa_de_ano(itens, None)

def regra_docs_pessoais(itens):
    out = {}
    for nome, mt in itens:
        lw = nome.lower()
        if "curricul" in lw:
            t = "Curriculos"
        elif "ficha cadastral" in lw or "ficha financeira" in lw:
            t = "Fichas cadastrais e financeiras"
        elif any(k in lw for k in ("certificado", "diploma", "certidao")):
            t = "Certificados, diplomas e certidoes"
        elif any(k in lw for k in ("apolice", "seguro")):
            t = "Seguros e apolices"
        elif any(k in lw for k in ("atestado", "exame", "receita", "vacina", "medic")):
            t = "Saude e atestados"
        elif any(k in lw for k in ("rg", "cpf", "identidade", "passaporte", "titulo de eleitor", "carteira", "cnh")):
            t = "Identificacao"
        else:
            t = "Outros"
        out[nome] = [t]
    return out

def regra_estudos_eleitoral(itens):
    julgados = []
    out = {}
    for nome, mt in itens:
        lw = nome.lower()
        if any(k in lw for k in ("certificado", "curso", "aula", "slides")):
            out[nome] = ["Cursos e certificados"]
        elif classe(nome) or "sumula" in lw:
            julgados.append((nome, mt))
        elif nome.lower().endswith((".pdf", ".epub")) and not classe(nome):
            out[nome] = ["Doutrina e artigos"]
        else:
            out[nome] = ["Outros"]
    por_grupo = defaultdict(list)
    for nome, mt in julgados:
        c = classe(nome)
        g = GRUPO_ELEITORAL.get(c, "Outras classes")
        por_grupo[g].append((nome, mt))
    for g, sub in por_grupo.items():
        if len(sub) > ALVO * 1.4:
            out.update(com_faixa_de_ano(sub, os.path.join("Julgados e sumulas", g)))
        else:
            out.update({n: [os.path.join("Julgados e sumulas", g)] for n, _ in sub})
    return out

def regra_alfabetica(itens):
    return com_faixa_alfabetica(itens, None)

def regra_estudos_generica(itens):
    """Estudos nao-eleitorais: Julgados / Doutrina / Cursos / Outros, faixa alfa se grande."""
    tipos = defaultdict(list)
    for nome, mt in itens:
        lw = nome.lower()
        if any(k in lw for k in ("certificado", "curso", "aula", "slides")):
            tipos["Cursos e certificados"].append((nome, mt))
        elif classe(nome) or "sumula" in lw or re.search(r"\b(adi|adpf|adc|re|resp|ag|hc|ms)\s*\d", lw):
            tipos["Julgados e sumulas"].append((nome, mt))
        elif lw.endswith(".pdf"):
            tipos["Doutrina e artigos"].append((nome, mt))
        else:
            tipos["Outros"].append((nome, mt))
    out = {}
    for t, sub in tipos.items():
        out.update(com_faixa_alfabetica(sub, t))
    return out

REGRAS = {
    r"01 - Juridico\Recursos\Eleitoral": regra_recursos_eleitoral,
    r"01 - Juridico\Recursos\Trabalhista": regra_recursos_trabalhista,
    r"01 - Juridico\Contestacoes e manifestacoes\Trabalhista": regra_contestacoes_trab,
    r"01 - Juridico\Peticoes\Trabalhista": regra_peticoes_trab,
    r"01 - Juridico\Despachos e votos\Eleitoral": regra_despachos_eleitoral,
    r"01 - Juridico\Acordaos e decisoes\Eleitoral": regra_acordaos_eleitoral,
    r"03 - Administrativo\Atas e sessoes": regra_por_ano,
    r"03 - Administrativo\Oficios e memorandos": regra_por_ano,
    r"03 - Administrativo\Listas e planilhas": regra_por_ano,
    r"03 - Administrativo\Comunicacoes internas": regra_por_ano,
    r"04 - Pessoal e financeiro\Comprovantes e recibos": regra_por_ano,
    r"04 - Pessoal e financeiro\Documentos pessoais": regra_docs_pessoais,
    r"02 - Estudos e concursos\Direito Eleitoral": regra_estudos_eleitoral,
    r"02 - Estudos e concursos\Direito Constitucional": regra_estudos_generica,
    r"02 - Estudos e concursos\Direito Administrativo": regra_estudos_generica,
    r"02 - Estudos e concursos\Atualidades e Outros": regra_alfabetica,
    r"03 - Administrativo\Outros": regra_por_ano,
    r"04 - Pessoal e financeiro\Outros": regra_por_ano,
    r"05 - Diversos (a revisar)\Outros": regra_alfabetica,
    r"01 - Juridico\Outros processuais\Outros": regra_alfabetica,
    # fase 5: pastas que engordaram com as auditorias
    r"01 - Juridico\Outros processuais\Eleitoral": regra_recursos_eleitoral,
    r"01 - Juridico\Despachos e votos\Eleitoral\Outros": regra_por_ano,
    r"03 - Administrativo\Contratos e termos": regra_por_ano,
    r"02 - Estudos e concursos\Direito Tributario": regra_estudos_generica,
}

def main():
    execu = "--exec" in sys.argv
    plano = []  # (origem, destino)
    for rel, regra in REGRAS.items():
        d = os.path.join(ROOT, rel)
        if not os.path.isdir(d):
            print(f"AVISO: nao existe {d}")
            continue
        itens = []
        for n in os.listdir(d):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                itens.append((n, os.path.getmtime(p)))
        destinos = regra(itens)
        cont = Counter()
        for nome, _ in itens:
            comps = [c for c in destinos.get(nome, []) if c]
            if not comps:
                continue
            sub = os.path.join(*comps)
            cont[sub] += 1
            plano.append((os.path.join(d, nome), os.path.join(d, sub, nome)))
        print(f"=== {rel} ({len(itens)} arqs -> {len(cont)} subpastas)")
        for sub, n in sorted(cont.items()):
            print(f"    {sub:60s} {n}")

    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origem", "destino"])
        w.writerows(plano)
    print(f"\nPlano: {len(plano)} movimentacoes -> {PLANO}")
    if not execu:
        print("Dry-run. Rode com --exec para mover.")
        return

    ok, erros = 0, 0
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for origem, destino in plano:
            try:
                dest = destino
                base_d, nome_d = os.path.split(destino)
                os.makedirs(base_d, exist_ok=True)
                i = 2
                while os.path.exists(dest):
                    stem, ext = os.path.splitext(nome_d)
                    dest = os.path.join(base_d, f"{stem} ({i}){ext}")
                    i += 1
                os.rename("\\\\?\\" + origem, "\\\\?\\" + dest)
                w.writerow([origem, dest])
                ok += 1
            except OSError as e:
                print(f"ERRO: {origem}: {e}")
                erros += 1
    print(f"Movidos: {ok} (erros: {erros}). Log: {LOG}")

if __name__ == "__main__":
    main()
