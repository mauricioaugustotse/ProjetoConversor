# -*- coding: utf-8 -*-
r"""Org2 etapa 15: replica a estrategia da fase 4 para os NAO-processuais (02-05).

Escopo: tudo em 02/03/04/05, fora "Julgados e sumulas" (colecao ja tratada) e 00-*.
Dois estagios de IA:
  A) gpt-5.4-nano TRIAGEM (todos, ~15k): pasta certa? nome indica o conteudo?
  B) gpt-5.4 REFINO (so flagados): confirma destino na whitelist e/ou reescreve o nome
     na convencao da categoria.
Execucao: UM rename/move por arquivo (nome novo + pasta nova + enderecamento fino nas
subpastas de faixa/tipo existentes). Log unico log_org2_nao_proc.csv.

Uso: python org2_15_nao_processuais.py [--exec] [--so-plano]
"""
import csv, datetime, io, json, os, re, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_13_auditoria_conteudo import extrair_cabecalho, carregar_cab
from org2_04_quebra_pastas import ano_do_nome

ROOT = r"C:\Users\mauri\HD_Mau"
RESP_A = os.path.join(BASE, "org2_ia_triagem15.jsonl")
RESP_B = os.path.join(BASE, "org2_ia_refino15.jsonl")
PLANO = os.path.join(BASE, "plano_org2_nao_proc.csv")
LOG = os.path.join(BASE, "log_org2_nao_proc.csv")
MODELO_A = "gpt-5.4-nano"
MODELO_B = "gpt-5.4"
MAXPATH = 259
CATS = ["02 - Estudos e concursos", "03 - Administrativo",
        "04 - Pessoal e financeiro", "05 - Diversos (a revisar)"]

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def whitelist():
    dests = []
    for cat in CATS:
        pc = os.path.join(ROOT, cat)
        for d in sorted(os.listdir(pc)):
            if os.path.isdir(os.path.join(pc, d)):
                dests.append(f"{cat}\\{d}")
    return dests

def coletar():
    alvos = []
    for cat in CATS:
        for dp, dn, fn in os.walk(os.path.join(ROOT, cat)):
            if "Julgados e sumulas" in dp:
                continue
            for f in fn:
                if f.startswith("00 - ") or f.lower() == "desktop.ini":
                    continue
                alvos.append(os.path.join(dp, f))
    return alvos

def extrair_isolado(path, timeout=60):
    """Extrai o cabecalho num PROCESSO FILHO com timeout de verdade (python-docx e
    striprtf nao tem timeout inline e ja travaram 3x nesta base)."""
    import subprocess
    cod = ("import sys; sys.path.insert(0, r'{b}'); "
           "from org2_13_auditoria_conteudo import extrair_cabecalho; "
           "sys.stdout.reconfigure(encoding='utf-8'); "
           "print(extrair_cabecalho(sys.argv[1]))").format(b=BASE)
    try:
        r = subprocess.run([sys.executable, "-c", cod, path],
                           capture_output=True, timeout=timeout)
        return r.stdout.decode("utf-8", "ignore").strip()[:1200]
    except Exception:
        return ""

def textos_para(alvos):
    CABTSV = os.path.join(BASE, "org2_cabecalhos.tsv")
    cab = {p: t for p, (k, t) in carregar_cab().items()}
    out = {}
    faltam = []
    for p in alvos:
        t = cab.get(p, "")
        if len(t) >= 80:
            out[p] = t
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext == ".txt":
            try:
                raw = io.open(p, encoding="utf-8", errors="ignore").read(2000)
                out[p] = re.sub(r"\s+", " ", raw).strip()[:1200]
            except OSError:
                out[p] = ""
        elif ext in (".pdf", ".doc", ".docx", ".rtf") and p not in cab:
            faltam.append(p)
        else:
            out[p] = cab.get(p, "")
    if faltam:
        print(f"Extraindo {len(faltam)} cabecalhos (isolado c/ timeout)...")
        with ThreadPoolExecutor(max_workers=6) as ex, io.open(CABTSV, "a", encoding="utf-8") as f:
            for p, t in zip(faltam, ex.map(extrair_isolado, faltam)):
                out[p] = t
                try:
                    st = os.stat(p)
                    f.write(f"{p}\t{st.st_size}|{int(st.st_mtime)}\t{t}\n")
                except OSError:
                    pass
    return out

SYS_A = """Voce faz a triagem do acervo pessoal de um servidor do TSE. Categorias:
02 - Estudos e concursos (por disciplina), 03 - Administrativo (documentos de orgaos/empresas),
04 - Pessoal e financeiro, 05 - Diversos (nao identificaveis).
Destinos validos:
{DESTINOS}
Para cada item (pasta atual, nome, trecho do conteudo):
- "destino": o caminho da whitelist onde o arquivo deveria estar, ou "manter" se a pasta atual ja e adequada (na duvida, "manter").
- "nome_ok": false SOMENTE se o nome nao permite saber o que o arquivo e (ou esta muito fora do padrao "Tipo - assunto - pessoa/orgao - data"); caso contrario true.
Responda APENAS JSON: {{"itens": [{{"id": <n>, "destino": "<caminho|manter>", "nome_ok": true|false, "confianca": <0-1>}}]}}"""

SYS_B = """Voce organiza o acervo pessoal de um servidor do TSE (ex-advogado dos Correios).
Convencao de nomes (ASCII sem acento, segmentos " - ", minusculas exceto siglas/nomes proprios/inicio):
- 02 Estudos: "Tema ou titulo - autor - ano"
- 03 Administrativo: "Tipo doc - assunto - orgao ou pessoa - DD-MM-AAAA"
- 04 Pessoal/financeiro: "Tipo - pessoa - detalhe - data"
Destinos validos:
{DESTINOS}
Para cada item (pasta atual, nome, trecho): confirme o melhor destino ("manter" se a pasta atual serve)
e, se o nome nao indicar o conteudo, proponha "novo_stem" na convencao (senao devolva "").
PRESERVE numeros de documento e datas do nome atual; nada de inventar alem do trecho; max ~120 chars.
Responda APENAS JSON: {{"itens": [{{"id": <n>, "destino": "<caminho|manter>", "novo_stem": "<nome|vazio>", "confianca": <0-1>}}]}}"""

def rodar_ia(itens, textos, modelo, system, resp_path, lote_n=12):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    lotes = [itens[i:i + lote_n] for i in range(0, len(itens), lote_n)]
    feitos = [0]
    def um_lote(lote):
        linhas = [f'id={i} | pasta="{os.path.dirname(os.path.relpath(p, ROOT))}" | nome="{os.path.basename(p)}" | trecho="{textos.get(p, "")[:500]}"'
                  for i, p in enumerate(lote)]
        for tent in range(4):
            try:
                r = client.chat.completions.create(
                    model=modelo,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": "\n\n".join(linhas)}],
                    response_format={"type": "json_object"})
                data = json.loads(r.choices[0].message.content)
                out = []
                for it in data.get("itens", []):
                    i = int(it.get("id", -1))
                    if 0 <= i < len(lote):
                        it["path"] = lote[i]
                        out.append(it)
                return out
            except Exception as e:
                if tent == 3:
                    print("FALHA lote:", e)
                    return []
                time.sleep(2 * (tent + 1))
    with ThreadPoolExecutor(max_workers=6) as ex, io.open(resp_path, "a", encoding="utf-8") as f:
        for res in ex.map(um_lote, lotes):
            for item in res or []:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            feitos[0] += 1
            if feitos[0] % 80 == 0:
                print(f"  lotes {feitos[0]}/{len(lotes)}")
                f.flush()

def carregar_resp(path):
    cache = {}
    if os.path.exists(path):
        for linha in io.open(path, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass
    return cache

# ------------------- enderecamento fino nao-juridico -------------------
RX_FAIXA_ANO = re.compile(r"^(\d{4})(?:-(\d{4}))?$")
RX_FAIXA_ALFA = re.compile(r"^([0-9A-Za-z]{1,2})-([0-9A-Za-z]{1,2})$")
KW_PESSOAIS = [("Curriculos", ("curricul",)), ("Fichas cadastrais e financeiras", ("ficha cadastral", "ficha financeira")),
               ("Certificados, diplomas e certidoes", ("certificado", "diploma", "certidao")),
               ("Seguros e apolices", ("apolice", "seguro")), ("Saude e atestados", ("atestado", "exame", "receita", "vacina")),
               ("Identificacao", ("rg", "cpf", "identidade", "passaporte", "carteira", "cnh"))]

def enderecar_np(dest_rel, nome, mtime):
    base = os.path.join(ROOT, dest_rel)
    if not os.path.isdir(base):
        return base
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subs:
        return base
    anos = [d for d in subs if RX_FAIXA_ANO.match(d)]
    alfas = [d for d in subs if RX_FAIXA_ALFA.match(d)]
    if anos:
        a = ano_do_nome(nome, mtime)
        if a:
            for d in anos:
                m = RX_FAIXA_ANO.match(d)
                if int(m.group(1)) <= a <= int(m.group(2) or m.group(1)):
                    return os.path.join(base, d)
            return os.path.join(base, str(a))
        return os.path.join(base, "Sem ano") if "Sem ano" in subs else base
    if alfas:
        s = re.sub(r"[^0-9A-Za-z]", "", nome.upper())
        for d in sorted(alfas):
            m = RX_FAIXA_ALFA.match(d)
            larg = max(len(m.group(1)), len(m.group(2)))
            k = (s[:larg] or "0").ljust(larg, "0")
            if m.group(1).upper().ljust(larg, "0") <= k <= m.group(2).upper().ljust(larg, "Z"):
                return os.path.join(base, d)
        return os.path.join(base, sorted(alfas)[-1])
    lw = nome.lower()
    if dest_rel.endswith("Documentos pessoais"):
        for rot, kws in KW_PESSOAIS:
            if any(k in lw for k in kws) and rot in subs:
                return os.path.join(base, rot)
        return os.path.join(base, "Outros") if "Outros" in subs else base
    if dest_rel.startswith("02"):
        if any(k in lw for k in ("certificado", "curso", "aula", "slides")) and "Cursos e certificados" in subs:
            return os.path.join(base, "Cursos e certificados")
        if lw.endswith(".pdf") and "Doutrina e artigos" in subs:
            return os.path.join(base, "Doutrina e artigos")
        if "Outros" in subs:
            return os.path.join(base, "Outros")
    return base

def sanitizar(stem):
    s = unicodedata.normalize("NFD", stem)
    s = "".join(c for c in s if not unicodedata.combining(c)).encode("ascii", "ignore").decode()
    s = re.sub(r'[<>:"/\\|?*]', " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -.,;")
    # convencao: primeiro caractere alfabetico em maiuscula
    for i, c in enumerate(s):
        if c.isalpha():
            if c.islower():
                s = s[:i] + c.upper() + s[i + 1:]
            break
    return s

def runs_digitos(s, minimo=6):
    return {r for r in re.findall(r"\d+", re.sub(r"[.\-\s]", "", s)) if len(r) >= minimo}

def main():
    execu = "--exec" in sys.argv
    dests = whitelist()
    set_dests = set(dests)
    alvos = coletar()
    print("Alvos:", len(alvos))
    textos = textos_para(alvos)

    # Estagio A: triagem nano
    cache_a = carregar_resp(RESP_A)
    pend_a = [p for p in alvos if p not in cache_a]
    if pend_a and "--so-plano" not in sys.argv:
        print(f"Triagem nano: {len(pend_a)}...")
        rodar_ia(pend_a, textos, MODELO_A, SYS_A.replace("{DESTINOS}", "\n".join(dests)), RESP_A)
        cache_a = carregar_resp(RESP_A)

    flagados = []
    for p in alvos:
        it = cache_a.get(p)
        if not it:
            continue
        atual = os.path.dirname(os.path.relpath(p, ROOT))
        d = str(it.get("destino", "manter")).strip().strip("\\")
        move = d != "manter" and d in set_dests and not atual.startswith(d)
        if move or not it.get("nome_ok", True):
            flagados.append(p)
    print(f"Flagados pela triagem: {len(flagados)} de {len(alvos)}")

    # Estagio B: refino gpt-5.4
    cache_b = carregar_resp(RESP_B)
    pend_b = [p for p in flagados if p not in cache_b]
    if pend_b and "--so-plano" not in sys.argv:
        print(f"Refino gpt-5.4: {len(pend_b)}...")
        rodar_ia(pend_b, textos, MODELO_B, SYS_B.replace("{DESTINOS}", "\n".join(dests)), RESP_B, lote_n=10)
        cache_b = carregar_resp(RESP_B)

    plano = []
    stats = Counter()
    for p in flagados:
        it = cache_b.get(p)
        if not it or it.get("confianca", 0) < 0.6:
            stats["mantido (sem refino/conf)"] += 1
            continue
        dirpath, nome = os.path.split(p)
        stem_atual, ext = os.path.splitext(nome)
        atual_rel = os.path.dirname(os.path.relpath(p, ROOT))
        d = str(it.get("destino", "manter")).strip().strip("\\")
        novo_stem = sanitizar(str(it.get("novo_stem", "")).strip())
        if novo_stem and runs_digitos(stem_atual) - runs_digitos(novo_stem):
            for m in re.finditer(r"[\d][\d.\-]*\d", stem_atual):
                if runs_digitos(m.group(0), 6) - runs_digitos(novo_stem):
                    novo_stem += f" - {m.group(0)}"
        stem_final = novo_stem if novo_stem and novo_stem.lower() != stem_atual.lower() else stem_atual
        if d != "manter" and d in set_dests and not atual_rel.startswith(d):
            try:
                mt = os.path.getmtime(p)
            except OSError:
                mt = None
            destino_dir = enderecar_np(d, stem_final + ext, mt)
        else:
            destino_dir = dirpath
        orcamento = MAXPATH - len(destino_dir) - 1 - len(ext)
        if len(stem_final) > orcamento:
            stem_final = stem_final[: max(20, orcamento)].rstrip(" -,;")
        novo_path = os.path.join(destino_dir, stem_final + ext)
        if os.path.normpath(novo_path) == os.path.normpath(p):
            stats["mantido (igual)"] += 1
            continue
        acao = []
        if destino_dir != dirpath:
            acao.append("move")
        if stem_final != stem_atual:
            acao.append("rename")
        stats["+".join(acao)] += 1
        plano.append((p, novo_path))

    print("\nPlano:", len(plano))
    for k, v in stats.most_common():
        print(f"  {v:5d}  {k}")
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    import random
    random.seed(13)
    for de, para in random.sample(plano, min(10, len(plano))):
        print("  DE :", os.path.relpath(de, ROOT)[:105])
        print("  P/ :", os.path.relpath(para, ROOT)[:105])
    if not execu:
        print("Dry-run. Rode com --exec para aplicar.")
        return
    ok = 0
    pre = "\\\\?\\"
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for de, para in plano:
            try:
                os.makedirs(os.path.dirname(para), exist_ok=True)
                dest = para
                i = 2
                while os.path.exists(pre + dest):
                    s, e = os.path.splitext(para)
                    dest = f"{s} ({i}){e}"
                    i += 1
                os.rename(pre + de, pre + dest)
                w.writerow([de, dest])
                ok += 1
            except OSError as e:
                print("ERRO:", de, e)
    print(f"Aplicados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
