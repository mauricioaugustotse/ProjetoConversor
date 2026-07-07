# -*- coding: utf-8 -*-
r"""Org2 etapa 10: renomeacao descritiva pelo CONTEUDO (gpt-5.4) dos nomes fracos e medianos.

Fracos: stem <15 chars, generico (cookie/documento/scan...), so numeros, ALL CAPS criptico.
Medianos: identificador sem descricao (<2 palavras alfabeticas >3 letras), ex. "AgR-REspe 151-46 - SP".
So renomeia com texto extraivel (>=200 chars norm.) e confianca >= 0.6; identificadores
(classe, numero, CNJ) do nome atual sao preservados; validacao local + truncamento <=259.

Uso: python org2_10_nomes_ia.py            -> classifica (API) + plano
     python org2_10_nomes_ia.py --exec     -> tambem renomeia + log_org2_nomes_ia.csv
     python org2_10_nomes_ia.py --so-plano -> so re-deriva plano do cache
"""
import csv, io, json, os, re, sys, threading, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_03_dedup_texto import extrair_completo

ROOT = r"C:\Users\mauri\HD_Mau"
RESP = os.path.join(BASE, "org2_ia_nomes.jsonl")
PLANO = os.path.join(BASE, "plano_org2_nomes_ia.csv")
LOG = os.path.join(BASE, "log_org2_nomes_ia.csv")
CACHE_FULL = os.path.join(BASE, "org2_cache_fulltext.tsv")
MODELO = "gpt-5.4"
LOTE = 10
CONF_MIN = 0.6
MAXPATH = 259

RX_GEN = re.compile(r"^(documento?\d*|doc\d+|novo ?doc|sem ?titulo|untitled|scan|img|imagem|image|"
                    r"digitalizad|arquivo|file|copia de|attachment|anexo|temp|redireciona|sem nome)\b", re.I)
RX_CNJ = re.compile(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")

def eh_alvo(nome):
    stem = os.path.splitext(nome)[0]
    if stem.lower().startswith("cookie"):
        return None
    letras = [c for c in stem if c.isalpha()]
    allcaps = bool(letras) and all(c.isupper() for c in letras)
    if len(stem) < 15 or RX_GEN.match(stem) or not letras or (allcaps and len(stem) < 25):
        return "fraco"
    palavras = [w for w in re.split(r"[\s\-_.,]+", stem) if len(w) > 3 and w.isalpha()]
    if len(palavras) < 2:
        return "mediano"
    return None

def falhas_conhecidas():
    ruins = set()
    if os.path.exists(CACHE_FULL):
        for linha in io.open(CACHE_FULL, encoding="utf-8"):
            p = linha.rstrip("\n").split("\t")
            if len(p) == 5 and p[3] == "FALHA":
                ruins.add(os.path.basename(p[0]).lower())
    return ruins

def asciify(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.encode("ascii", "ignore").decode()

def sanitizar(stem):
    s = asciify(stem)
    s = re.sub(r'[<>:"/\\|?*]', " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -.,;")
    return s

SYSTEM = """Voce nomeia arquivos do acervo pessoal de um servidor do TSE (ex-advogado trabalhista dos Correios).
Para cada item (pasta, nome atual, trecho do conteudo), proponha um NOME DE ARQUIVO descritivo (sem extensao).

Convencao OBRIGATORIA:
- ASCII sem acentos; segmentos separados por " - "; maximo ~130 caracteres.
- Minusculas, EXCETO: siglas (REspe, AgR, CNJ, TSE, TRE-SP, OAB...), UFs e nomes proprios (Joao Silva).
- Primeiro segmento com inicial maiuscula quando for palavra (ex: "Contestacao", "Oficio", "Recibo").
- PRESERVE identificadores que ja estejam no nome atual: classe+numero de processo, numero CNJ (formato NNNNNNN-DD.AAAA.J.TR.OOOO), datas.
- Estruturas-alvo:
  Juridico: "CLASSE numero - UF - assunto - desfecho - MM-AAAA - CNJ ..."
  Administrativo: "Tipo doc - assunto - orgao ou pessoa - DD-MM-AAAA"
  Pessoal/financeiro: "Tipo - pessoa - detalhe - data"
  Estudos: "Tema ou titulo - autor - ano"
- O nome deve dizer O QUE o documento e (tipo + assunto + partes/pessoas relevantes). Nada de inventar: use so o que esta no trecho.
- Se o trecho nao permitir nome melhor que o atual, devolva confianca baixa (<0.5).

Responda APENAS JSON: {"itens": [{"id": <n>, "novo_stem": "<nome sem extensao>", "confianca": <0.0-1.0>}, ...]}"""

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("OPENAI_API_KEY nao encontrada")

def classificar(pendentes, textos):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    lock = threading.Lock()
    feitos = [0]
    lotes = [pendentes[i:i + LOTE] for i in range(0, len(pendentes), LOTE)]

    def um_lote(lote):
        linhas = []
        for i, p in enumerate(lote):
            rel = os.path.relpath(p, ROOT)
            linhas.append(f'id={i} | pasta="{os.path.dirname(rel)}" | nome_atual="{os.path.basename(p)}" | trecho="{textos.get(p, "")[:900]}"')
        msg = "\n\n".join(linhas)
        for tent in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODELO,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": msg}],
                    response_format={"type": "json_object"})
                data = json.loads(r.choices[0].message.content)
                out = []
                for it in data.get("itens", []):
                    i = int(it.get("id", -1))
                    if 0 <= i < len(lote):
                        out.append({"path": lote[i],
                                    "novo_stem": str(it.get("novo_stem", "")),
                                    "confianca": float(it.get("confianca", 0))})
                return out
            except Exception as e:
                if tent == 3:
                    print(f"FALHA lote: {type(e).__name__}: {e}")
                    return []
                time.sleep(2 * (tent + 1))

    with ThreadPoolExecutor(max_workers=6) as ex, io.open(RESP, "a", encoding="utf-8") as f:
        for res in ex.map(um_lote, lotes):
            with lock:
                for item in res or []:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                feitos[0] += 1
                if feitos[0] % 40 == 0:
                    print(f"  lotes {feitos[0]}/{len(lotes)}")
                    f.flush()

def montar_plano(cache):
    plano = []
    stats = Counter()
    for p, it in cache.items():
        if not os.path.exists(p):
            stats["sumiu"] += 1
            continue
        dirpath, nome = os.path.split(p)
        stem_atual, ext = os.path.splitext(nome)
        novo = sanitizar(it["novo_stem"])
        conf = it["confianca"]
        if not novo or conf < CONF_MIN or novo.lower() == stem_atual.lower():
            stats["mantido"] += 1
            continue
        m = RX_CNJ.search(stem_atual)
        if m and m.group(0) not in novo:
            novo = f"{novo} - CNJ {m.group(0)}"
        orcamento = MAXPATH - len(dirpath) - 1 - len(ext)
        if len(novo) > orcamento:
            sufixo = ""
            mm = re.search(r"( - CNJ \d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})$", novo)
            if mm:
                sufixo = mm.group(1)
                novo = novo[: len(novo) - len(sufixo)]
            novo = novo[: max(20, orcamento - len(sufixo))].rstrip(" -,;") + sufixo
        plano.append((p, os.path.join(dirpath, novo + ext)))
        stats["renomear"] += 1
    return plano, stats

def main():
    ruins = falhas_conhecidas()
    alvos = {}
    for dirpath, _, filenames in os.walk(ROOT):
        for n in filenames:
            if n.startswith("00 - ") or n.lower() == "desktop.ini" or n.lower() in ruins:
                continue
            t = eh_alvo(n)
            if t:
                alvos[os.path.join(dirpath, n)] = t
    print("Alvos:", len(alvos), dict(Counter(alvos.values())))

    cache = {}
    if os.path.exists(RESP):
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass
    pendentes = [p for p in alvos if p not in cache]
    print(f"Cache: {len(cache)}; pendentes: {len(pendentes)}")

    if pendentes and "--so-plano" not in sys.argv:
        print("Extraindo texto dos pendentes...")
        textos = {}
        def w(p):
            t = extrair_completo(p) or ""
            return p, re.sub(r"\s+", " ", t).strip()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, (p, t) in enumerate(ex.map(w, pendentes), 1):
                textos[p] = t
                if i % 1000 == 0:
                    print(f"  {i}/{len(pendentes)} ({time.time()-t0:.0f}s)")
        com_texto = [p for p in pendentes if len(textos.get(p, "")) >= 200]
        sem_texto = len(pendentes) - len(com_texto)
        print(f"Com texto suficiente: {len(com_texto)}; sem texto (mantidos): {sem_texto}")
        classificar(com_texto, textos)
        cache = {}
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass

    plano, stats = montar_plano(cache)
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    print("Resultado:", dict(stats))
    print("Amostra:")
    import random
    random.seed(4)
    for de, para in random.sample(plano, min(12, len(plano))):
        print("  DE :", os.path.basename(de)[:110])
        print("  P/ :", os.path.basename(para)[:110])
    if "--exec" not in sys.argv:
        print("Rode com --exec para renomear.")
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
