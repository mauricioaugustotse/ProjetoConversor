# -*- coding: utf-8 -*-
r"""Org2 etapa 14: corrige os nomes defeituosos residuais e os poe na convencao.

Coleta: (a) palavras comuns GRITADAS (SENTENCA, EXECUCAO...) que a regra de caixa tratou
como sigla; (b) nomes com caracteres nao-ASCII/corrompidos ("10� decisao"); (c) palavras
quebradas ("execu cao"). Correcao deterministica + reescrita IA (gpt-5.4) lendo o cabecalho
(cache org2_cabecalhos.tsv da etapa 13). UM rename por arquivo, log unico.

Validacao da IA: todo numero de processo (run de >=6 digitos) do nome original deve constar
do novo (senao re-anexa); ASCII; <=259; conf >= 0.6.

Uso: python org2_14_nomes_defeituosos.py [--exec] [--so-plano]
     (log: log_org2_nomes_defeituosos.csv; cache IA: org2_ia_nomes2.jsonl)
"""
import csv, io, json, os, re, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_13_auditoria_conteudo import extrair_cabecalho, carregar_cab

ROOT = r"C:\Users\mauri\HD_Mau"
RESP = os.path.join(BASE, "org2_ia_nomes2.jsonl")
PLANO = os.path.join(BASE, "plano_org2_nomes_defeituosos.csv")
LOG = os.path.join(BASE, "log_org2_nomes_defeituosos.csv")
MODELO = "gpt-5.4"
MAXPATH = 259

PALAVRAS = {"SENTENCA", "ACORDAO", "DECISAO", "DESPACHO", "CONTESTACAO", "PETICAO", "RECURSO",
            "MANIFESTACAO", "EMBARGOS", "AGRAVO", "PARECER", "OFICIO", "MEMORANDO", "EDITAL",
            "CONTRATO", "PROCURACAO", "DECLARACAO", "RELATORIO", "EXECUCAO", "MANDADO",
            "SEGURANCA", "TRABALHISTA", "ELEITORAL", "PUBLICA", "FAZENDA", "PROCESSO",
            "RECLAMACAO", "APELACAO", "IMPUGNACAO", "NOTIFICACAO", "CERTIDAO", "COMPROVANTE",
            "RECIBO", "MODELO", "MINUTA", "LAUDO", "VOTO", "INICIAL", "DEFESA", "HABEAS",
            "CORPUS", "CUMPRIMENTO", "ORDINARIA", "CAUTELAR", "CIVEL", "PENAL", "FEDERAL"}
VOCAB = {"execucao", "trabalhista", "eleitoral", "contestacao", "manifestacao", "impugnacao",
         "reclamacao", "apelacao", "declaracao", "procuracao", "notificacao", "sentenca",
         "acordao", "decisao", "despacho", "peticao", "recurso", "embargos", "agravo",
         "parecer", "relatorio", "seguranca", "fazenda", "publica", "processo", "tributario",
         "administrativo", "constitucional", "previdenciario", "aposentadoria", "indenizacao",
         "responsabilidade", "periculosidade", "insalubridade", "honorarios", "liquidacao",
         "cumprimento", "intimacao", "citacao", "audiencia", "ordinaria", "cautelar"}
TIPOS_PREFIXO = ["Sentenca", "Acordao", "Decisao", "Despacho", "Voto", "Parecer", "Contestacao",
                 "Peticao", "Manifestacao", "Oficio", "Memorando", "Contrato", "Procuracao",
                 "Declaracao", "Laudo", "Edital", "Recibo", "Certidao", "Comprovante"]

def defeituoso(stem):
    if any(ord(c) > 127 for c in stem):
        return True
    toks = re.findall(r"[A-Za-z]+", stem)
    if any(t.upper() in PALAVRAS and t.isupper() and len(t) > 2 for t in toks):
        return True
    return any((a + b).lower() in VOCAB and (a.islower() or b.islower())
               for a, b in zip(toks, toks[1:]))

def corrigir_det(stem):
    s = stem
    # ordinais e chars corrompidos
    s = s.replace("º", "o").replace("ª", "a").replace("°", "o")
    s = re.sub(r"(\d)[�\x80-\x9f]", r"\1a", s)          # "10?" apos digito -> ordinal
    # char invalido entre letras: junta se formar palavra do vocabulario, senao vira espaco
    def junta(m):
        cand = (m.group(1) + m.group(2)).lower()
        return m.group(1) + m.group(2) if cand in VOCAB else m.group(1) + " " + m.group(2)
    s = re.sub(r"([A-Za-z]+)[�\x80-\x9f]([A-Za-z]+)", junta, s)
    # remove nao-ASCII restante (acentos legitimos -> sem acento)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.encode("ascii", "ignore").decode()
    # palavras quebradas: junta bigramas do vocabulario
    toks = s.split(" ")
    out = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and (toks[i] + toks[i + 1]).lower() in VOCAB and toks[i].isalpha() and toks[i + 1].isalpha():
            out.append(toks[i] + toks[i + 1])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    s = " ".join(out)
    # palavras gritadas -> Capitalizada
    s = re.sub(r"\b[A-Z]{3,}\b", lambda m: m.group(0).capitalize() if m.group(0) in PALAVRAS else m.group(0), s)
    # prefixo de tipo ganha separador
    for t in TIPOS_PREFIXO:
        if re.match(rf"^{t}\s+(?!-)", s, re.I):
            s = re.sub(rf"^{t}\s+", f"{t} - ", s, count=1, flags=re.I)
            break
    s = re.sub(r"\s+", " ", s).strip(" -.,;")
    return s

def runs_digitos(s, minimo=6):
    return {r for r in re.findall(r"\d+", re.sub(r"[.\-\s]", "", s)) if len(r) >= minimo}

SIGLAS_CLASSE = ["AgR-REspe", "AgR-AI", "AgR-RO", "AgR-DREspe", "ED-AgR-REspe", "AgRg", "AIRR",
                 "E-RR", "EDcl", "REspe", "ROEl", "RCED", "RMS", "AgR", "AC", "AI", "AR", "CC",
                 "CTA", "ED", "HC", "RHC", "MS", "PA", "PC", "PP", "Pet", "RO", "RP", "RR",
                 "Rcl", "RE", "LT", "Inst", "RCand"]
RX_PREFIXO_CLASSE = re.compile(
    r"^(" + "|".join(re.escape(s) for s in sorted(SIGLAS_CLASSE, key=len, reverse=True)) +
    r")[ ]?(\d[\d.\-]*\d|\d)", re.I)

def preserva_prefixo_classe(stem_orig, cand):
    """Se o original comeca com 'CLASSE numero', o novo deve comecar igual (sem expandir sigla)."""
    m = RX_PREFIXO_CLASSE.match(stem_orig)
    if not m:
        return True
    esperado = f"{m.group(1)} {m.group(2)}"
    return cand.lower().startswith(esperado.lower())

SYSTEM = """Voce corrige nomes de arquivos do acervo juridico de um servidor do TSE (ex-advogado dos Correios).
Cada item traz pasta, nome atual (ja pre-corrigido) e o cabecalho do documento.
Reescreva o nome na convencao: "Tipo - descricao do objeto - partes relevantes - numero do processo - ano".
Regras: ASCII sem acento; segmentos " - "; minusculas exceto siglas/nomes proprios/inicio de segmento-tipo;
PRESERVE os numeros de processo e datas do nome atual; maximo ~130 caracteres; nada de inventar alem do cabecalho.
Se o nome atual ja estiver adequado, devolva-o igual com confianca alta.
Responda APENAS JSON: {"itens": [{"id": <n>, "novo_stem": "<nome sem extensao>", "confianca": <0-1>}]}"""

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def ia_reescrever(itens, cabecalhos):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    lotes = [itens[i:i + 10] for i in range(0, len(itens), 10)]
    def um_lote(lote):
        linhas = [f'id={i} | pasta="{os.path.dirname(os.path.relpath(p, ROOT))}" | nome="{stem_det}" | cabecalho="{cabecalhos.get(p, "")[:700]}"'
                  for i, (p, stem_det) in enumerate(lote)]
        for tent in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODELO,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": "\n\n".join(linhas)}],
                    response_format={"type": "json_object"})
                data = json.loads(r.choices[0].message.content)
                out = []
                for it in data.get("itens", []):
                    i = int(it.get("id", -1))
                    if 0 <= i < len(lote):
                        out.append({"path": lote[i][0], "novo_stem": str(it.get("novo_stem", "")),
                                    "confianca": float(it.get("confianca", 0))})
                return out
            except Exception as e:
                if tent == 3:
                    print("FALHA lote:", e)
                    return []
                time.sleep(2 * (tent + 1))
    with ThreadPoolExecutor(max_workers=6) as ex, io.open(RESP, "a", encoding="utf-8") as f:
        for res in ex.map(um_lote, lotes):
            for item in res or []:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    execu = "--exec" in sys.argv
    alvos = []
    for dp, dn, fn in os.walk(ROOT):
        for f in fn:
            if f.startswith("00 - ") or f.lower() == "desktop.ini":
                continue
            if defeituoso(os.path.splitext(f)[0]):
                alvos.append(os.path.join(dp, f))
    print("Nomes defeituosos:", len(alvos))

    dets = {p: corrigir_det(os.path.splitext(os.path.basename(p))[0]) for p in alvos}
    cab = {p: t for p, (k, t) in carregar_cab().items()}
    faltam = [p for p in alvos if len(cab.get(p, "")) < 120]
    if faltam:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for p, t in zip(faltam, ex.map(extrair_cabecalho, faltam)):
                cab[p] = t

    cache = {}
    if os.path.exists(RESP):
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass
    pend = [(p, dets[p]) for p in alvos if p not in cache]
    if pend and "--so-plano" not in sys.argv:
        print(f"IA reescrevendo {len(pend)}...")
        ia_reescrever(pend, cab)
        cache = {}
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass

    plano = []
    stats = Counter()
    for p in alvos:
        dirpath, nome = os.path.split(p)
        stem_atual, ext = os.path.splitext(nome)
        novo = dets[p]
        it = cache.get(p)
        if it and it["confianca"] >= 0.6 and it["novo_stem"].strip() \
                and preserva_prefixo_classe(stem_atual, it["novo_stem"].strip()):
            cand = unicodedata.normalize("NFD", it["novo_stem"])
            cand = "".join(c for c in cand if not unicodedata.combining(c)).encode("ascii", "ignore").decode()
            cand = re.sub(r'[<>:"/\\|?*]', " ", cand)
            cand = re.sub(r"\s+", " ", cand).strip(" -.,;")
            faltantes = runs_digitos(stem_atual) - runs_digitos(cand)
            if faltantes:
                orig_nums = [m.group(0) for m in re.finditer(r"[\d][\d.\-]*\d", stem_atual)
                             if runs_digitos(m.group(0), 6)]
                cand = cand + "".join(f" - {n}" for n in orig_nums
                                      if runs_digitos(n, 6) & faltantes)
            if cand:
                novo = cand
                stats["via IA"] += 1
            else:
                stats["via deterministico"] += 1
        else:
            stats["via deterministico"] += 1
        orcamento = MAXPATH - len(dirpath) - 1 - len(ext)
        if len(novo) > orcamento:
            novo = novo[: max(20, orcamento)].rstrip(" -,;")
        if novo != stem_atual:
            plano.append((p, os.path.join(dirpath, novo + ext)))
    print("Renomeacoes:", len(plano), dict(stats))
    import random
    random.seed(8)
    for de, para in random.sample(plano, min(10, len(plano))):
        print("  DE :", os.path.basename(de)[:105])
        print("  P/ :", os.path.basename(para)[:105])
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    if not execu:
        print("Dry-run. Rode com --exec para renomear.")
        return
    ok = 0
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
                print("ERRO:", de, e)
    print(f"Renomeados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
