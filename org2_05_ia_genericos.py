# -*- coding: utf-8 -*-
r"""Org2 etapa 5: reclassificacao por conteudo (IA) dos arquivos em pastas genericas.

Alvos: 03\Outros, 05 - Diversos (a revisar)\*, 02\Atualidades e Outros, 04\Outros,
01\Outros processuais\Outros.
Modelo: gpt-5.4-nano (fallback aprovado; sem chave Anthropic disponivel para Sonnet 5).
A IA recebe nome + pasta atual + trecho do texto e escolhe um destino da taxonomia
real da arvore (whitelist) ou "manter". Respostas cacheadas em org2_ia_respostas.jsonl.

Uso: python org2_05_ia_genericos.py            -> classifica (API) + gera plano
     python org2_05_ia_genericos.py --exec     -> tambem move + log_org2_ia.csv
     python org2_05_ia_genericos.py --so-plano -> nao chama API; usa so o cache
"""
import csv, io, json, os, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_03_dedup_texto import parse_tsv, indexar_arvore, casar, extrair_completo

ROOT = r"C:\Users\mauri\HD_Mau"
RESP = os.path.join(BASE, "org2_ia_respostas.jsonl")
PLANO = os.path.join(BASE, "plano_org2_ia.csv")
LOG = os.path.join(BASE, "log_org2_ia.csv")
MODELO = "gpt-5.4-nano"
LOTE = 15
CONF_MIN = 0.6

ALVOS = [
    r"03 - Administrativo\Outros",
    r"05 - Diversos (a revisar)",
    r"02 - Estudos e concursos\Atualidades e Outros",
    r"04 - Pessoal e financeiro\Outros",
    r"01 - Juridico\Outros processuais\Outros",
]

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("OPENAI_API_KEY nao encontrada no .env")

def whitelist():
    """Destinos validos = pastas reais da taxonomia (nivel 2; nivel 3 no Juridico)."""
    dests = []
    for cat in sorted(os.listdir(ROOT)):
        pc = os.path.join(ROOT, cat)
        if not os.path.isdir(pc) or cat.startswith("05"):
            continue
        for tipo in sorted(os.listdir(pc)):
            pt = os.path.join(pc, tipo)
            if not os.path.isdir(pt):
                continue
            if cat.startswith("01"):
                for mat in sorted(os.listdir(pt)):
                    if os.path.isdir(os.path.join(pt, mat)):
                        dests.append(f"{cat}\\{tipo}\\{mat}")
            else:
                dests.append(f"{cat}\\{tipo}")
    return dests

def coletar_alvos():
    arquivos = []
    for rel in ALVOS:
        d = os.path.join(ROOT, rel)
        if not os.path.isdir(d):
            continue
        for dirpath, _, filenames in os.walk(d):
            for n in filenames:
                arquivos.append(os.path.join(dirpath, n))
    return arquivos

def textos_para(arquivos):
    """caminho -> trecho de texto (TSV da fase 1; extracao direta no residuo)."""
    regs = parse_tsv()
    caminhos, por_nome = indexar_arvore()
    casados = casar(regs, caminhos, por_nome)
    out = {}
    faltam = []
    for p in arquivos:
        t = casados.get(p, "")
        if len(t.strip()) >= 40:
            out[p] = t[:800]
        else:
            faltam.append(p)
    print(f"Texto via TSV: {len(out)}; extraindo residuo: {len(faltam)}")
    def w(p):
        t = extrair_completo(p) or ""
        return p, re.sub(r"\s+", " ", t)[:800]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for p, t in ex.map(w, faltam):
            out[p] = t
    return out

SYSTEM = """Voce organiza o acervo pessoal de documentos de um servidor do TSE (ex-advogado trabalhista dos Correios, estudioso de direito eleitoral).
Para cada item (nome do arquivo, pasta atual, trecho do conteudo), escolha o MELHOR destino na taxonomia abaixo, ou "manter" se o item ja esta bem colocado ou nao da para saber.

Destinos validos (use o caminho EXATAMENTE como escrito):
{DESTINOS}

Regras:
- "01 - Juridico" e para pecas processuais e decisoes REAIS (com partes/processo). Material de ESTUDO (julgados colecionados por tema, apostilas, resumos, doutrina) vai para "02 - Estudos e concursos\\<disciplina>".
- Documentos administrativos de orgao/empresa (oficios, atas, listas, contratos) -> "03 - Administrativo\\...".
- Documentos da vida pessoal/financeira -> "04 - Pessoal e financeiro\\...".
- Na duvida entre mover e manter, prefira "manter".
- confianca: 0.0 a 1.0 (use >=0.8 so quando o conteudo deixa claro).

Responda APENAS JSON: {{"itens": [{{"id": <n>, "destino": "<caminho ou manter>", "confianca": <x>}}, ...]}}"""

def classificar(pendentes, textos, dests):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    system = SYSTEM.replace("{DESTINOS}", "\n".join(dests))
    lock = threading.Lock()
    feitos = [0]

    lotes = [pendentes[i:i + LOTE] for i in range(0, len(pendentes), LOTE)]

    def um_lote(lote):
        linhas = []
        for i, p in enumerate(lote):
            rel = os.path.relpath(p, ROOT)
            linhas.append(f'id={i} | pasta="{os.path.dirname(rel)}" | nome="{os.path.basename(p)}" | texto="{textos.get(p, "")[:600]}"')
        msg = "\n\n".join(linhas)
        for tent in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODELO,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": msg}],
                    response_format={"type": "json_object"})
                data = json.loads(r.choices[0].message.content)
                out = []
                for it in data.get("itens", []):
                    i = int(it.get("id", -1))
                    if 0 <= i < len(lote):
                        out.append({"path": lote[i],
                                    "destino": str(it.get("destino", "manter")),
                                    "confianca": float(it.get("confianca", 0))})
                return out
            except Exception as e:
                if tent == 3:
                    print(f"FALHA lote: {e}")
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

def main():
    dests = whitelist()
    print(f"Whitelist: {len(dests)} destinos")
    arquivos = coletar_alvos()
    print(f"Alvos: {len(arquivos)} arquivos")

    cache = {}
    if os.path.exists(RESP):
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass
    pendentes = [p for p in arquivos if p not in cache]
    print(f"Cache: {len(cache)}; pendentes: {len(pendentes)}")

    if pendentes and "--so-plano" not in sys.argv:
        textos = textos_para(pendentes)
        classificar(pendentes, textos, dests)
        cache = {}
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass

    # plano: mover so com destino valido e confianca suficiente
    set_dests = set(dests)
    plano = []
    cont = Counter()
    for p in arquivos:
        it = cache.get(p)
        if not it:
            cont["(sem resposta)"] += 1
            continue
        d, conf = it["destino"].strip().strip("\\"), it["confianca"]
        if d.lower() == "manter" or conf < CONF_MIN or d not in set_dests:
            cont["manter"] += 1
            continue
        atual = os.path.relpath(os.path.dirname(p), ROOT)
        if atual == d:
            cont["manter"] += 1
            continue
        plano.append((p, os.path.join(ROOT, d, os.path.basename(p))))
        cont[d] += 1
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origem", "destino"])
        w.writerows(plano)
    print(f"\nPlano IA: mover {len(plano)}; manter {cont['manter']}; sem resposta {cont['(sem resposta)']}")
    for d, n in cont.most_common(25):
        if d not in ("manter", "(sem resposta)"):
            print(f"    {n:5d} -> {d}")
    if "--exec" not in sys.argv:
        print("Rode com --exec para mover.")
        return

    ok = 0
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for origem, destino in plano:
            try:
                base_d, nome_d = os.path.split(destino)
                os.makedirs(base_d, exist_ok=True)
                dest = destino
                i = 2
                while os.path.exists(dest):
                    stem, ext = os.path.splitext(nome_d)
                    dest = os.path.join(base_d, f"{stem} ({i}){ext}")
                    i += 1
                os.rename(origem, dest)
                w.writerow([origem, dest])
                ok += 1
            except OSError as e:
                print(f"ERRO: {origem}: {e}")
    print(f"Movidos: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
