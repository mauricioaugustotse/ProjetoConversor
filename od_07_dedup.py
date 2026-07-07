# -*- coding: utf-8 -*-
r"""OD etapa 7: deduplicacao criteriosa de OneDrive\Documentos (a pedido de 06/07/2026).

Camadas:
  1) BINARIA: grupos por tamanho -> MD5 (od_dups_bin.json da medicao previa, ou recalcula).
  2) TEXTUAL: texto INTEGRAL normalizado (sha1) dos textuais, extracao blindada em
     subprocess com timeout; salvaguardas do HD_Mau (>=300 chars normalizados).
Sobrevivente ("folder mais apropriado"):
  - dossie tematico vence (Unimed, Tese, Luiz Celso, Camara dos Deputados, Concursos);
  - mesma pasta: melhor nome (sem sufixo (N), mais descritivo);
  - cross-pasta ambiguo: gpt-5.4 escolhe a pasta certa (fallback: caminho mais especifico).
Perdedores -> LIXEIRA (send2trash) + log_od_dedup.csv. Pasta 12 fora de tudo.

Uso: python od_07_dedup.py [--exec] [--so-plano]
"""
import csv, hashlib, io, json, os, re, subprocess, sys, time, unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\mauri\OneDrive\Documentos"
DUPS_BIN = os.path.join(BASE, "od_dups_bin.json")
FULLTXT = os.path.join(BASE, "od_fulltext.tsv")
RESP = os.path.join(BASE, "od_ia_dedup.jsonl")
PLANO = os.path.join(BASE, "plano_od_dedup.csv")
LOG = os.path.join(BASE, "log_od_dedup.csv")
MODELO = "gpt-5.4"
PRE = "\\\\?\\"
MIN_NORM = 300
PROT = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
        "Modelos Personalizados do Office", "My Kindle Content")
DOSSIES = (r"02 - Saúde\Unimed", r"01 - Pessoal e identidade\Câmara dos Deputados",
           r"09 - Família, religião e mensagens\Luiz Celso Vieira",
           r"06 - Estudos, concursos e leitura\Tese",
           r"06 - Estudos, concursos e leitura\Concursos")
EXTS_TXT = (".pdf", ".doc", ".docx", ".rtf", ".txt", ".md", ".html", ".htm", ".csv")
RX_SUFIXO = re.compile(r"[ _]\(\d+\)$|[ _]\[\d+\]$|[ _]-[ _]?[Cc]ópia$")

def protegida(rel):
    return any(seg in PROT for seg in rel.split(os.sep))

def em_dossie(rel):
    return any(rel.startswith(d) for d in DOSSIES)

def normalizar(texto):
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]", "", t)

def extrair_full_isolado(path, timeout=120):
    cod = ("import sys; sys.path.insert(0, r'{b}'); "
           "from org2_03_dedup_texto import extrair_completo; "
           "sys.stdout.reconfigure(encoding='utf-8'); "
           "t = extrair_completo(sys.argv[1]) or ''; print(t)").format(b=BASE)
    try:
        r = subprocess.run([sys.executable, "-c", cod, path],
                           capture_output=True, timeout=timeout)
        return r.stdout.decode("utf-8", "ignore")
    except Exception:
        return ""

def coletar():
    alvos = []
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        if rel != "." and protegida(rel):
            dn[:] = []
            continue
        for f in fn:
            if f.startswith("00 - ") or f.lower() == "desktop.ini":
                continue
            alvos.append(os.path.join(dp, f))
    return alvos

def score_nome(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    return (0 if RX_SUFIXO.search(stem) else 1, min(len(stem), 150),
            1 if " - " in stem else 0)

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

SYS = """Você organiza o acervo pessoal de um servidor do TSE. Cada item é um GRUPO de arquivos
IDÊNTICOS em conteúdo, salvos em pastas diferentes. Escolha QUAL CÓPIA FICA (a pasta mais
apropriada ao conteúdo). Regras: dossiês temáticos vencem (o processo da Unimed fica no dossiê
Unimed; documentos do Luiz Celso ficam no acervo dele; bibliografia da Tese fica na Tese);
senão, a pasta tematicamente mais específica. As demais cópias serão removidas.
Responda APENAS JSON: {"itens": [{"id": <n>, "fica": <índice da cópia que fica, 0-based>, "confianca": <0-1>}]}"""

def ia_escolher(grupos_ambiguos, trechos):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    itens = list(grupos_ambiguos.items())
    lotes = [itens[i:i + 8] for i in range(0, len(itens), 8)]
    def um_lote(lote):
        linhas = []
        for i, (gid, grupo) in enumerate(lote):
            ops = "; ".join(f"[{j}] {os.path.relpath(p, ROOT)}" for j, p in enumerate(grupo))
            linhas.append(f'id={i} | copias: {ops} | trecho="{trechos.get(grupo[0], "")[:250]}"')
        for tent in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODELO,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": "\n\n".join(linhas)}],
                    response_format={"type": "json_object"})
                data = json.loads(r.choices[0].message.content)
                out = []
                for it in data.get("itens", []):
                    i = int(it.get("id", -1))
                    if 0 <= i < len(lote):
                        it["gid"] = lote[i][0]
                        out.append(it)
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
    so_plano = "--so-plano" in sys.argv
    alvos = coletar()
    print("Escopo:", len(alvos))

    # camada binaria
    grupos = []
    if os.path.exists(DUPS_BIN):
        dups_bin = json.load(io.open(DUPS_BIN, encoding="utf-8"))
        grupos.extend([v for v in dups_bin.values() if len(v) > 1])
    ja = {p for g in grupos for p in g}
    print("Grupos binarios:", len(grupos))

    # camada textual (texto integral)
    textuais = [p for p in alvos if os.path.splitext(p)[1].lower() in EXTS_TXT]
    cache = {}
    if os.path.exists(FULLTXT):
        for linha in io.open(FULLTXT, encoding="utf-8"):
            c = linha.rstrip("\n").split("\t")
            if len(c) == 4:
                cache[c[0]] = (c[1], c[2], int(c[3]))
    pend = []
    for p in textuais:
        try:
            st = os.stat(PRE + p)
        except OSError:
            continue
        key = f"{st.st_size}|{int(st.st_mtime)}"
        if p not in cache or cache[p][0] != key:
            pend.append((p, key))
    print(f"Texto integral: cache {len(textuais)-len(pend)}; extraindo {len(pend)}...")
    if pend:
        t0 = time.time()
        def w(item):
            p, key = item
            norm = normalizar(extrair_full_isolado(p))
            return p, key, hashlib.sha1(norm.encode()).hexdigest(), len(norm)
        with ThreadPoolExecutor(max_workers=6) as ex, io.open(FULLTXT, "a", encoding="utf-8") as f:
            for i, (p, key, h, ln) in enumerate(ex.map(w, pend), 1):
                f.write(f"{p}\t{key}\t{h}\t{ln}\n")
                cache[p] = (key, h, ln)
                if i % 500 == 0:
                    print(f"  {i}/{len(pend)} ({time.time()-t0:.0f}s)")
                    f.flush()
    por_sha = defaultdict(list)
    for p in textuais:
        if p in ja or p not in cache:
            continue
        _, h, ln = cache[p]
        if ln >= MIN_NORM:
            por_sha[(h, ln)].append(p)
    g_txt = [v for v in por_sha.values() if len(v) > 1]
    grupos.extend(g_txt)
    print("Grupos textuais adicionais:", len(g_txt))

    # trechos p/ IA (cabecalhos ja extraidos)
    trechos = {}
    cabp = os.path.join(BASE, "od_cabecalhos.tsv")
    if os.path.exists(cabp):
        for linha in io.open(cabp, encoding="utf-8"):
            c = linha.rstrip("\n").split("\t")
            if len(c) == 3:
                trechos[c[0]] = c[2]

    # sobrevivente por grupo
    plano = []
    stats = Counter()
    ambiguos = {}
    for gi, grupo in enumerate(grupos):
        grupo = [p for p in grupo if os.path.exists(PRE + p)]
        if len(grupo) < 2:
            continue
        rels = [os.path.relpath(p, ROOT) for p in grupo]
        dossies = [p for p, r in zip(grupo, rels) if em_dossie(os.path.dirname(r))]
        pastas = {os.path.dirname(p) for p in grupo}
        if len(dossies) == 1:
            keeper = dossies[0]
            stats["regra: dossiê"] += 1
        elif len(pastas) == 1:
            keeper = max(grupo, key=score_nome)
            stats["regra: mesma pasta, melhor nome"] += 1
        else:
            ambiguos[str(gi)] = grupo
            continue
        for p in grupo:
            if p != keeper:
                plano.append((p, keeper))
    print(f"Decididos por regra: {dict(stats)}; ambiguos p/ gpt-5.4: {len(ambiguos)}")

    cache_ia = {}
    if os.path.exists(RESP):
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache_ia[it["gid"]] = it
            except Exception:
                pass
    pend_ia = {k: v for k, v in ambiguos.items() if k not in cache_ia}
    if pend_ia and not so_plano:
        print(f"gpt-5.4 escolhendo pasta de {len(pend_ia)} grupos...")
        ia_escolher(pend_ia, trechos)
        cache_ia = {}
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache_ia[it["gid"]] = it
            except Exception:
                pass
    for gid, grupo in ambiguos.items():
        it = cache_ia.get(gid)
        if it and 0 <= int(it.get("fica", -1)) < len(grupo) and it.get("confianca", 0) >= 0.6:
            keeper = grupo[int(it["fica"])]
            stats["IA escolheu"] += 1
        else:
            keeper = max(grupo, key=lambda p: (len(os.path.dirname(p)), score_nome(p)))
            stats["fallback: mais específico"] += 1
        for p in grupo:
            if p != keeper:
                plano.append((p, keeper))

    assert not any("12 - Consultoria Legislativa" in a for a, b in plano), "GUARDA: pasta 12!"
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["remover", "sobrevivente"])
        w.writerows(plano)
    print(f"\nPlano: remover {len(plano)} copias. {dict(stats)}")
    import random
    random.seed(41)
    for de, keep in random.sample(plano, min(10, len(plano))):
        print("  REMOVE:", os.path.relpath(de, ROOT)[:95])
        print("  FICA  :", os.path.relpath(keep, ROOT)[:95])
    if not execu:
        print("Dry-run. Rode com --exec para enviar à Lixeira.")
        return

    from send2trash import send2trash
    ok = 0
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["removido_para_lixeira", "sobrevivente"])
        for de, keep in plano:
            try:
                send2trash(de)
                w.writerow([de, keep])
                ok += 1
            except Exception as e:
                print("ERRO:", de, e)
    print(f"Enviados à Lixeira: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
