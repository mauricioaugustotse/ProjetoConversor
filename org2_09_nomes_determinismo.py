# -*- coding: utf-8 -*-
r"""Org2 etapa 9: normalizacao deterministica dos nomes (sem IA) em toda a base.

Convencao (decidida em 06/07/2026): siglas canonicas + descricao minuscula; primeiro
token alfabetico com inicial maiuscula; capitalizacao mista existente e intocavel;
datas inequivocas -> DD-MM-AAAA / MM-AAAA; espacamento e pontuacao limpos.

Uso: python org2_09_nomes_determinismo.py [--exec]   (log: log_org2_nomes_det.csv)
"""
import csv, os, re, sys
from collections import Counter

ROOT = r"C:\Users\mauri\HD_Mau"
BASE = os.path.dirname(os.path.abspath(__file__))
PLANO = os.path.join(BASE, "plano_org2_nomes_det.csv")
LOG = os.path.join(BASE, "log_org2_nomes_det.csv")

# siglas processuais/juridicas: aplicadas so nas categorias 01, 02 e 05
SIGLAS_JUR = ["REspe", "AgR-REspe", "AgR-AI", "AgR-RO", "AgR", "AgRg", "EDcl", "ED",
              "ROEl", "RO", "RR", "AIRR", "E-RR", "AI", "AC", "MS", "RMS", "RCED",
              "HC", "RHC", "Rcl", "CC", "CTA", "PA", "PC", "PP", "Pet", "RP", "AR",
              "AIJE", "AIME", "RRC", "RCC", "DRAP", "RVE", "RCAND", "RE", "REsp",
              "ADI", "ADC", "ADPF", "AP", "Inq", "MC", "SS", "TutCautAnt"]
# siglas institucionais/gerais: qualquer categoria
SIGLAS_GERAIS = ["CNJ", "TSE", "STF", "STJ", "TST", "TRE", "TRT", "TRF", "TRF1", "TJMG",
                 "TJSP", "TJRS", "TJRJ", "OAB", "ECT", "CLT", "FGTS", "INSS", "IPTU",
                 "IPVA", "IRPF", "ITBI", "ICMS", "ISS", "PIS", "COFINS", "SERASA",
                 "SPC", "SCPC", "DETRAN", "PJe", "DJe", "MPE", "MPF", "MPT", "AGU",
                 "PGE", "PGR", "DOU", "CPF", "RG", "CNH", "CEP", "CNPJ", "TED",
                 "BB", "CEF", "HSBC", "UOL", "MSN", "OCR", "GMAJ", "DEJUR",
                 "DG", "SINTECT", "PSOL", "PSDB", "MDB", "UF"]
UFS = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
       "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
MESES = {"jan": "01", "fev": "02", "mar": "03", "abr": "04", "mai": "05", "jun": "06",
         "jul": "07", "ago": "08", "set": "09", "out": "10", "nov": "11", "dez": "12"}

MAPA_GERAL = {s.lower(): s for s in SIGLAS_GERAIS}
MAPA_JUR = {s.lower(): s for s in SIGLAS_JUR}
MAPA_JUR.update(MAPA_GERAL)  # juridico usa ambos
SET_UFS = set(UFS)

RX_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*|[^A-Za-z0-9]+")

def normalizar_espacos(stem):
    s = stem.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*-\s*", " - ", s)      # " - - " duplicado
    s = re.sub(r"\s+([,;])", r"\1", s)
    s = s.strip(" -.,;")
    return s

def normalizar_datas(stem):
    s = stem
    # DDMMAAAA colado -> DD-MM-AAAA (validado)
    def f8(m):
        d, mo, a = m.group(1), m.group(2), m.group(3)
        if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12 and 1990 <= int(a) <= 2026:
            return f"{d}-{mo}-{a}"
        return m.group(0)
    s = re.sub(r"\b(\d{2})(\d{2})((?:19|20)\d{2})\b", f8, s)
    # AAAA-MM-DD -> DD-MM-AAAA
    def fiso(m):
        a, mo, d = m.group(1), m.group(2), m.group(3)
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{d}-{mo}-{a}"
        return m.group(0)
    s = re.sub(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", fiso, s)
    # D-M-AA -> DD-MM-20AA (AA<=26)
    def f2(m):
        d, mo, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and a <= 26:
            return f"{d:02d}-{mo:02d}-20{a:02d}"
        return m.group(0)
    s = re.sub(r"\b(\d{1,2})-(\d{1,2})-(\d{2})\b(?!\.\d)", f2, s)
    # D-M-AAAA com um digito -> zero-pad
    def fpad(m):
        d, mo, a = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{d:02d}-{mo:02d}-{a}"
        return m.group(0)
    s = re.sub(r"\b(\d{1,2})-(\d{1,2})-((?:19|20)\d{2})\b", fpad, s)
    # mes por extenso + ano: "mar2015", "mar 2015", "out de 2009", "mar-15"
    def fmes(m):
        mo = MESES[m.group(1)[:3].lower()]
        a = m.group(2)
        if len(a) == 2:
            if int(a) > 26:
                return m.group(0)
            a = "20" + a
        return f"{mo}-{a}"
    s = re.sub(r"\b(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[a-z]*[ .\-]*(?:de )?(\d{4}|\d{2})\b(?![\-/.]\d)",
               fmes, s, flags=re.I)
    return s

RX_ROMANO = re.compile(r"^[IVXLCDM]+$")
PREPOSICOES = {"de", "da", "do", "dos", "das", "e", "em", "para", "com", "a", "o"}
# palavras comuns que aparecem GRITADAS mas nao sao siglas (fase 4)
PALAVRAS_COMUNS = {"SENTENCA", "ACORDAO", "DECISAO", "DESPACHO", "CONTESTACAO", "PETICAO",
                   "RECURSO", "MANIFESTACAO", "EMBARGOS", "AGRAVO", "PARECER", "OFICIO",
                   "MEMORANDO", "EDITAL", "CONTRATO", "PROCURACAO", "DECLARACAO", "RELATORIO",
                   "EXECUCAO", "MANDADO", "SEGURANCA", "TRABALHISTA", "ELEITORAL", "PUBLICA",
                   "FAZENDA", "PROCESSO", "RECLAMACAO", "APELACAO", "IMPUGNACAO", "NOTIFICACAO",
                   "CERTIDAO", "COMPROVANTE", "RECIBO", "MODELO", "MINUTA", "LAUDO", "VOTO",
                   "INICIAL", "DEFESA", "HABEAS", "CORPUS", "CUMPRIMENTO", "ORDINARIA",
                   "CAUTELAR", "CIVEL", "PENAL", "FEDERAL"}

def normalizar_caixa(stem, juridico):
    mapa = MAPA_JUR if juridico else MAPA_GERAL
    toks = RX_TOKEN.findall(stem)
    # marca sequencias de >=2 tokens alfabeticos ALL CAPS separados so por espaco
    # (nome de pessoa/frase gritada -> Title Case); caps isolado = sigla, mantem
    caps_set = {i for i, t in enumerate(toks)
                if t[0].isalnum() and t.isalpha() and t.isupper() and len(t) >= 2
                and t.lower() not in mapa and not RX_ROMANO.match(t) and t not in SET_UFS}
    em_sequencia = set()
    run = []
    for i, t in enumerate(toks):
        if i in caps_set:
            run.append(i)
        elif run and not t[0].isalnum() and t.strip() == "" and "-" not in t:
            continue  # espaco entre membros do run
        else:
            if len(run) >= 2:
                em_sequencia.update(run)
            run = []
    if len(run) >= 2:
        em_sequencia.update(run)
    out = []
    primeiro_alfa = True
    for i, tok in enumerate(toks):
        if not tok[0].isalnum():
            out.append(tok)
            continue
        low = tok.lower()
        if low in mapa:                              # sigla canonica
            novo = mapa[low]
        elif tok.upper() in SET_UFS and len(tok) == 2 and tok.isalpha() and tok.isupper():
            novo = tok                               # UF ja maiuscula: mantem
        elif i in em_sequencia:                      # trecho gritado -> Title Case
            novo = low if low in PREPOSICOES else tok.capitalize()
        elif tok.isupper() and tok in PALAVRAS_COMUNS:
            novo = tok.capitalize()                  # palavra comum gritada nao e sigla
        else:
            novo = tok                               # sigla desconhecida/misto/minusculo: intocavel
        if primeiro_alfa and novo[0].isalpha():
            if novo[0].islower():
                novo = novo[0].upper() + novo[1:]
            primeiro_alfa = False
        elif primeiro_alfa and novo[0].isdigit():
            primeiro_alfa = False
        out.append(novo)
    return "".join(out)

def novo_nome(nome, juridico):
    stem, ext = os.path.splitext(nome)
    s = normalizar_espacos(stem)
    s = normalizar_datas(s)
    s = normalizar_caixa(s, juridico)
    s = normalizar_espacos(s)
    if not s:
        s = stem
    return s + ext.lower()

def main():
    execu = "--exec" in sys.argv
    plano = []
    tipos = Counter()
    for dirpath, _, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT)
        cat = rel.split(os.sep)[0] if rel != "." else ""
        juridico = cat.startswith(("01", "02", "05"))
        for nome in filenames:
            if nome.startswith("00 - ") or nome.lower() == "desktop.ini":
                continue
            novo = novo_nome(nome, juridico)
            if novo != nome:
                plano.append((os.path.join(dirpath, nome), os.path.join(dirpath, novo)))
                stem_a, stem_n = os.path.splitext(nome)[0], os.path.splitext(novo)[0]
                if normalizar_espacos(stem_a) != stem_a:
                    tipos["espacamento"] += 1
                if normalizar_datas(stem_a) != stem_a:
                    tipos["data"] += 1
                if stem_n != normalizar_datas(normalizar_espacos(stem_a)):
                    tipos["caixa/sigla"] += 1
    print(f"Renomeacoes propostas: {len(plano)} de 35766")
    print("Por tipo (nao exclusivo):", dict(tipos))
    print("\nAmostra:")
    import random
    random.seed(2)
    for de, para in random.sample(plano, min(20, len(plano))):
        print("  DE :", os.path.basename(de)[:110])
        print("  P/ :", os.path.basename(para)[:110])
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    print(f"\nPlano: {PLANO}")
    if not execu:
        print("Dry-run. Rode com --exec para renomear.")
        return
    ok, erros = 0, 0
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for de, para in plano:
            try:
                dest = para
                i = 2
                while os.path.exists("\\\\?\\" + dest):
                    stem, ext = os.path.splitext(para)
                    dest = f"{stem} ({i}){ext}"
                    i += 1
                os.rename("\\\\?\\" + de, "\\\\?\\" + dest)
                w.writerow([de, dest])
                ok += 1
            except OSError as e:
                print(f"ERRO: {de}: {e}")
                erros += 1
    print(f"Renomeados: {ok} (erros: {erros}). Log: {LOG}")

if __name__ == "__main__":
    main()
