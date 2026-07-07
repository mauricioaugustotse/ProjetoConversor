# -*- coding: utf-8 -*-
r"""Org2 etapa 13: auditoria da alocacao de pecas processuais pelo CONTEUDO (cabecalho).

- Extrai o cabecalho (~1.200 chars; PDFs: so 2 primeiras paginas) de todos os documentos
  de 01-05 (fora "Julgados e sumulas" e "Modelos e minutas"); cache org2_cabecalhos.tsv.
- Detecta assinatura de peca no TEXTO -> tipo provavel + materia (digito J do CNJ do texto).
- Fora de 01: candidato -> gpt-5.4 decide mover/manter (didatico que cita processo = manter);
  gate 0.7 (origem 02) / 0.6 (03-05).
- Dentro de 01: so divergencia FORTE entre tipo detectado e pasta (peca-de-parte x decisao
  judicial x parecer; "Outros processuais" diverge de deteccao especifica) -> IA confirma.
- Enderecamento fino: enderecar() do org2_12.

Uso: python org2_13_auditoria_conteudo.py [--exec] [--so-plano]  (log: log_org2_auditoria.csv)
"""
import csv, io, json, os, re, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_03_dedup_texto import extrair_completo
from org2_12_minutas import (enderecar, J_MATERIA, TIPOS_01, MATERIAS, materia_de,
                             TIPO_PECA, RX_MANTER)

ROOT = r"C:\Users\mauri\HD_Mau"
CAB = os.path.join(BASE, "org2_cabecalhos.tsv")
CACHE_FULL = os.path.join(BASE, "org2_cache_fulltext.tsv")
RESP = os.path.join(BASE, "org2_ia_auditoria.jsonl")
PLANO = os.path.join(BASE, "plano_org2_auditoria.csv")
LOG = os.path.join(BASE, "log_org2_auditoria.csv")
MODELO = "gpt-5.4"
PDFTOTEXT = "pdftotext"
# aceita CNJ moderno E numeracao antiga do TSE ("507.93.2016.6.00.0000")
RX_CNJ_TXT = re.compile(r"\d{1,7}\s?[.\-]\s?\d{2}\s?\.\s?\d{4}\s?\.\s?(\d)\s?\.\s?\d{2}\s?\.\s?\d{4}")

def norm_txt(t):
    """asciify + chars corrompidos viram '@' (regexes usam classes [x@])."""
    import unicodedata
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return "".join(c if ord(c) < 128 else "@" for c in t)

# peso 2: titulo ancorado no inicio OU titulo multi-palavra nos primeiros 400 chars
# (documentos comecam com cabecalho institucional antes do titulo); classes [x@]
# toleram o caractere corrompido tipico de acentuada perdida
ASSINATURAS = [
    ("Acordaos e decisoes", 2, re.compile(r"^\s*(SENTEN[C@]A|AC[O@]RD[A@]O)\b|\b(SENTEN[C@]A|AC[O@]RD[A@]O)\s+N", re.I)),
    ("Acordaos e decisoes", 1, re.compile(r"\bjulgo\s+(im)?procedente|\bEMENTA\s*[:\n ]", re.I)),
    ("Despachos e votos", 2, re.compile(r"^\s*(DESPACHO|DECIS[A@]O(\s+MONOCR[A@]TICA)?|VOTO)\b", re.I)),
    ("Despachos e votos", 1, re.compile(r"\bintime-se\b|\bnego seguimento\b|\bO SENHOR MINISTRO\b", re.I)),
    ("Recursos", 2, re.compile(r"(RECURSO (ESPECIAL|ORDIN[A@]RIO|DE REVISTA|EXTRAORDIN[A@]RIO)|AGRAVO (REGIMENTAL|DE INSTRUMENTO)|EMBARGOS DE DECLARA[C@][A@]O)\s+(ELEITORAL\s+)?N", re.I)),
    ("Contestacoes e manifestacoes", 2, re.compile(r"^\s*(CONTESTA[C@][A@]O|MANIFESTA[C@][A@]O|IMPUGNA[C@][A@]O|CONTRARRAZ[O@]ES|CONTRAMINUTA)", re.I)),
    ("Pareceres e notas tecnicas", 2, re.compile(r"^\s*(PARECER|NOTA (JUR[I@]DICA|T[E@]CNICA))\b", re.I)),
    ("Peticoes", 1, re.compile(r"\bEXCELENT[I@]SSIMO\s+SENHOR", re.I)),
    ("Outros processuais", 2, re.compile(r"(LISTA TR[I@]PLICE|PROCESSO ADMINISTRATIVO N|MANDADO DE SEGURAN[C@]A N|HABEAS CORPUS N|A[C@][A@]O CAUTELAR N|CUMPRIMENTO DE SENTEN[C@]A)", re.I)),
    ("Outros processuais", 1, re.compile(r"\bLISTA TR[I@]PLICE\b|\bRELATORA?\s*:\s*MINISTR", re.I)),
]
G_PARTE = {"Peticoes", "Contestacoes e manifestacoes", "Recursos"}
G_DECISAO = {"Acordaos e decisoes", "Despachos e votos"}
G_PARECER = {"Pareceres e notas tecnicas"}

def grupo(tipo):
    if tipo in G_PARTE:
        return "parte"
    if tipo in G_DECISAO:
        return "decisao"
    if tipo in G_PARECER:
        return "parecer"
    return "outros"

def detectar(texto):
    """(tipo, forca) da melhor assinatura, ou (None, 0)."""
    texto = norm_txt(texto)
    melhor = (None, 0)
    ini = texto[:400]
    for tipo, peso, rx in ASSINATURAS:
        alvo = ini if peso == 2 else texto
        if rx.search(alvo) and peso > melhor[1]:
            melhor = (tipo, peso)
    return melhor

def extrair_cabecalho(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            r = subprocess.run([PDFTOTEXT, "-q", "-l", "2", "-enc", "UTF-8", path, "-"],
                               capture_output=True, timeout=90)
            t = r.stdout.decode("utf-8", "ignore") if r.returncode == 0 else ""
        else:
            t = extrair_completo(path) or ""
    except Exception:
        t = ""
    return re.sub(r"\s+", " ", t).strip()[:1200]

def patologicos():
    ruins = set()
    if os.path.exists(CACHE_FULL):
        for linha in io.open(CACHE_FULL, encoding="utf-8"):
            p = linha.rstrip("\n").split("\t")
            if len(p) == 5 and p[3] == "FALHA":
                ruins.add(os.path.basename(p[0]).lower())
    return ruins

def carregar_cab():
    cab = {}
    if os.path.exists(CAB):
        for linha in io.open(CAB, encoding="utf-8"):
            p = linha.rstrip("\n").split("\t")
            if len(p) == 3:
                cab[p[0]] = (p[1], p[2])
    return cab

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

SYSTEM = """Voce audita a organizacao do acervo de um servidor do TSE (ex-advogado dos Correios).
Cada item traz: pasta atual, nome e o CABECALHO do documento. Estrutura do acervo:
"01 - Juridico\\<tipo>\\<materia>" guarda pecas/expedientes processuais REAIS e decisoes de processos concretos.
Pastas de estudo (02), administrativas (03), pessoais (04) e diversos (05) NAO devem conter pecas reais.
Decida pelo CABECALHO:
- Peca/expediente processual real fora do lugar -> acao "mover" + tipo + materia.
- Se ja esta em 01 mas o TIPO da pasta atual nao condiz com o documento -> "mover" + tipo certo.
- Material didatico, apostila, edital/gabarito de concurso, doutrina, documento administrativo interno,
  pessoal/financeiro, ou colecao de jurisprudencia para ESTUDO -> "manter".
tipos validos: {TIPOS}
materias validas: {MATERIAS}
Responda APENAS JSON: {{"itens": [{{"id": <n>, "acao": "mover"|"manter", "tipo": "<tipo|null>", "materia": "<materia|null>", "confianca": <0-1>}}]}}"""

def ia_decidir(itens, cab):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    system = SYSTEM.replace("{TIPOS}", ", ".join(TIPOS_01)).replace("{MATERIAS}", ", ".join(MATERIAS))
    lotes = [itens[i:i + 10] for i in range(0, len(itens), 10)]
    feitos = [0]
    def um_lote(lote):
        linhas = [f'id={i} | pasta="{os.path.dirname(os.path.relpath(p, ROOT))}" | nome="{os.path.basename(p)}" | cabecalho="{cab[p][1][:800]}"'
                  for i, p in enumerate(lote)]
        for tent in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODELO,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": "\n\n".join(linhas)}],
                    response_format={"type": "json_object"})
                data = json.loads(r.choices[0].message.content)
                out = []
                for it in data.get("itens", []):
                    i = int(it.get("id", -1))
                    if 0 <= i < len(lote):
                        out.append({"path": lote[i], "acao": it.get("acao", "manter"),
                                    "tipo": it.get("tipo"), "materia": it.get("materia"),
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
            feitos[0] += 1
            if feitos[0] % 40 == 0:
                print(f"  lotes {feitos[0]}/{len(lotes)}")
                f.flush()

def main():
    execu = "--exec" in sys.argv
    ruins = patologicos()
    alvos = []
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        if "Julgados e sumulas" in rel or "Modelos e minutas" in rel:
            continue
        for f in fn:
            if f.startswith("00 - ") or f.lower() in ruins:
                continue
            if os.path.splitext(f)[1].lower() in (".pdf", ".doc", ".docx", ".rtf"):
                alvos.append(os.path.join(dp, f))
    print("Alvos:", len(alvos))

    cab = carregar_cab()
    pendentes = []
    for p in alvos:
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = f"{st.st_size}|{int(st.st_mtime)}"
        if p not in cab or cab[p][0] != key:
            pendentes.append((p, key))
    print(f"Cabecalhos em cache: {len(alvos) - len(pendentes)}; extraindo: {len(pendentes)}")
    if pendentes:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex, io.open(CAB, "a", encoding="utf-8") as f:
            def w(item):
                p, key = item
                return p, key, extrair_cabecalho(p)
            for i, (p, key, t) in enumerate(ex.map(w, pendentes), 1):
                f.write(f"{p}\t{key}\t{t}\n")
                cab[p] = (key, t)
                if i % 2000 == 0:
                    print(f"  {i}/{len(pendentes)} ({time.time()-t0:.0f}s)")
                    f.flush()

    # deteccao
    candidatos = []  # (path, tipo_detectado, forca)
    for p in alvos:
        texto = cab.get(p, ("", ""))[1]
        if len(texto) < 120:
            continue
        tipo, forca = detectar(texto)
        rel = os.path.relpath(p, ROOT)
        cat = rel.split(os.sep)[0]
        if cat.startswith("01"):
            if not tipo:
                continue
            atual = rel.split(os.sep)[1]
            # "Recursos" e "Outros processuais" sao ancorados em CLASSE processual:
            # acordaos/votos/decisoes sobre recursos moram legitimamente la.
            if atual in ("Recursos", "Outros processuais", "Modelos e minutas"):
                continue
            if grupo(tipo) != grupo(atual) and forca == 2:
                candidatos.append(p)
        else:
            # fora de 01: qualquer assinatura forte, ou fraca + CNJ no texto
            if tipo and (forca == 2 or RX_CNJ_TXT.search(texto)):
                candidatos.append(p)
    print("Candidatos p/ IA:", len(candidatos))

    cache = {}
    if os.path.exists(RESP):
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass
    pend_ia = [p for p in candidatos if p not in cache]
    if pend_ia and "--so-plano" not in sys.argv:
        print(f"IA decidindo {len(pend_ia)}...")
        ia_decidir(pend_ia, cab)
        cache = {}
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass

    plano = []
    stats = Counter()
    for p in candidatos:
        it = cache.get(p)
        if not it:
            stats["sem resposta"] += 1
            continue
        rel = os.path.relpath(p, ROOT)
        cat = rel.split(os.sep)[0]
        stem = os.path.splitext(os.path.basename(p))[0]
        gate = 0.7 if cat.startswith("02") else 0.6
        if it["acao"] != "mover" or it["confianca"] < gate:
            stats["manter"] += 1
            continue
        t = it.get("tipo") if it.get("tipo") in TIPOS_01 else None
        if not t:
            stats["manter"] += 1
            continue
        if not cat.startswith("01") and RX_MANTER.search(stem):
            stats["manter (lista/pauta)"] += 1
            continue
        if cat.startswith("01"):
            if rel.split(os.sep)[1] == t:
                stats["manter"] += 1
                continue
            # docs de gabinete sao hibridos: so move se o NOME concorda com o texto/IA
            t_nome = next((tt for tt, rx in TIPO_PECA if rx.search(stem)), None)
            if t_nome != t:
                stats["manter (nome nao confirma)"] += 1
                continue
        mat = it.get("materia") if it.get("materia") in MATERIAS else None
        m = RX_CNJ_TXT.search(cab.get(p, ("", ""))[1])
        if m and m.group(1) in J_MATERIA:
            mat = J_MATERIA[m.group(1)]
        if not mat:
            mat = materia_de(os.path.splitext(os.path.basename(p))[0], rel)
        dest = enderecar(t, mat, os.path.basename(p))
        if os.path.normpath(dest) == os.path.normpath(os.path.dirname(p)):
            stats["manter"] += 1
            continue
        plano.append((p, os.path.join(dest, os.path.basename(p))))
        stats[f"mover -> {t}\\{mat}"] += 1

    print("\nPlano:", len(plano))
    for k, v in stats.most_common(25):
        print(f"  {v:5d}  {k}")
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    if not execu:
        print("Dry-run. Rode com --exec para mover.")
        return
    ok = 0
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
                ok += 1
            except OSError as e:
                print("ERRO:", de, e)
    print(f"Movidos: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
