# -*- coding: utf-8 -*-
r"""OD etapa 12: excelência em 06\Concursos — renomeação por tipo de material + alocação
por certame (gpt-5.4 lê o conteúdo). Dissolve pastas genéricas ("Arquivos", "DOCxs de
concurso", soltos da raiz) realocando pelo certame identificado no conteúdo.

Convenções:
  Edital - <órgão> - <cargo> - <ano>            Prova - <órgão> - <cargo/fase> - <ano>
  Gabarito - <órgão> - <fase> - <ano>           Recurso - questão <nn> - <matéria> - <órgão> <ano>
  Aula - <matéria> - <tema> - <professor/curso> Caderno - <matéria> - <banca> - <ano>
  Resumo/Mapa mental - <matéria> - <tema>       Simulado - <órgão> - <n ou data>
  Inscrição - <órgão> - <ano>                   Material de estudo - <matéria> - <tema>

Uso: python od_12_concursos.py [--exec] [--so-plano]  (log: log_od_concursos.csv)
"""
import csv, io, json, os, re, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_15_nao_processuais import extrair_isolado

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
CON = os.path.join(ROOT, "06 - Estudos, concursos e leitura", "Concursos")
RESP = os.path.join(BASE, "od_ia_concursos.jsonl")
PLANO = os.path.join(BASE, "plano_od_concursos.csv")
LOG = os.path.join(BASE, "log_od_concursos.csv")
MODELO = "gpt-5.4"
PRE = "\\\\?\\"
MAXPATH = 240
GENERICAS = ("Arquivos", "DOCxs de concurso")

SYS = """Você organiza a pasta "Concursos" (materiais de concursos públicos) do acervo de um servidor
do TSE que estuda para carreiras jurídicas. Subpastas por CERTAME/órgão:
{CERTAMES}
Para cada item (pasta atual, nome, trecho do conteúdo):
- "novo_stem" na convenção do TIPO de material (português com acentos):
  Edital - <órgão> - <cargo> - <ano> | Prova - <órgão> - <cargo ou fase> - <ano> |
  Prova de sentença - <órgão> - <ano> | Gabarito - <órgão> - <fase> - <ano> |
  Recurso - questão <nn> - <matéria> - <órgão> <ano> | Aula - <matéria> - <tema> - <professor ou curso> |
  Caderno - <matéria> - <banca> - <ano> | Resumo - <matéria> - <tema> | Mapa mental - <matéria> - <tema> |
  Simulado - <órgão> - <número ou data> | Inscrição - <órgão> - <ano> | Material de estudo - <matéria> - <tema> |
  Súmulas - <tribunal> - <tema> | Nota técnica - <órgão> - <tema>
  Se o nome atual já estiver adequado, devolva-o igual.
- "sub": subpasta de certame correta (use o caminho relativo, ex. "MPF\\Previdenciário", "TRF6\\Cadernos"),
  quando o item estiver em pasta genérica (Arquivos, DOCxs de concurso, raiz) ou claramente no certame errado;
  senão "manter". NUNCA invente subpasta nova.
Dados vêm do CONTEÚDO; não invente; preserve identificadores (nº edital, questão); máx ~110 chars.
Responda APENAS JSON: {{"itens": [{{"id": <n>, "novo_stem": "<nome>", "sub": "<caminho|manter>", "confianca": <0-1>}}]}}"""

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def nfc(s):
    return unicodedata.normalize("NFC", s)

def subpastas_validas():
    subs = []
    for dp, dn, fn in os.walk(CON):
        rel = os.path.relpath(dp, CON)
        if rel != "." and not any(g in rel.split(os.sep) for g in GENERICAS):
            subs.append(rel)
    return subs

def main():
    execu = "--exec" in sys.argv
    so_plano = "--so-plano" in sys.argv
    subs = subpastas_validas()
    set_subs = set(subs)
    alvos = []
    for dp, dn, fn in os.walk(CON):
        for f in fn:
            alvos.append(os.path.join(dp, f))
    print("Alvos:", len(alvos), "| subpastas válidas:", len(subs))

    print("Extraindo conteúdo...")
    textos = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for p, t in zip(alvos, ex.map(extrair_isolado, alvos)):
            textos[p] = t

    cache = {}
    if os.path.exists(RESP):
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass
    pend = [p for p in alvos if p not in cache]
    if pend and not so_plano:
        from openai import OpenAI
        client = OpenAI(api_key=chave_openai())
        system = SYS.replace("{CERTAMES}", ", ".join(sorted({s.split(os.sep)[0] for s in subs})))
        lotes = [pend[i:i + 10] for i in range(0, len(pend), 10)]
        feitos = [0]
        def um_lote(lote):
            linhas = [f'id={i} | pasta="{os.path.relpath(os.path.dirname(p), CON)}" | nome="{os.path.basename(p)}" | trecho="{textos.get(p, "")[:550]}"'
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
                            it["path"] = lote[i]
                            out.append(it)
                    return out
                except Exception as e:
                    if tent == 3:
                        print("FALHA:", e)
                        return []
                    time.sleep(2 * (tent + 1))
        with ThreadPoolExecutor(max_workers=6) as ex, io.open(RESP, "a", encoding="utf-8") as f:
            for res in ex.map(um_lote, lotes):
                for item in res or []:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                feitos[0] += 1
                if feitos[0] % 20 == 0:
                    print(f"  lotes {feitos[0]}/{len(lotes)}")
                    f.flush()
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
        it = cache.get(p)
        if not it or it.get("confianca", 0) < 0.6:
            stats["mantido"] += 1
            continue
        dirpath, nome = os.path.split(p)
        stem_atual, ext = os.path.splitext(nome)
        rel_atual = os.path.relpath(dirpath, CON)
        novo = nfc(str(it.get("novo_stem", "")).strip())
        novo = re.sub(r'[<>:"/\\|?*]', " ", novo)
        novo = re.sub(r"\s+", " ", novo).strip(" -.,;")
        if not novo or novo.lower() in ("manter", "mantido"):
            novo = stem_atual
        # numero da aula ordena o curso: preservar "Aula NN" do original
        m_aula = re.search(r"\bAula\s+(\d{1,3})\b", stem_atual, re.I)
        if m_aula and re.match(r"^Aula - ", novo):
            novo = re.sub(r"^Aula - ", f"Aula {int(m_aula.group(1)):02d} - ", novo)
        # "(Ok)" e marcador de progresso de estudo do usuario: preservar como sufixo
        if re.match(r"^\(\s*ok\s*\)", stem_atual, re.I) and "(ok)" not in novo.lower():
            novo += " (ok)"
        em_generica = rel_atual == "." or any(g in rel_atual.split(os.sep) for g in GENERICAS)
        sub = str(it.get("sub", "manter")).strip().strip("\\")
        destino_dir = dirpath
        gate_move = 0.65 if em_generica else 0.75
        if sub != "manter" and sub in set_subs and it.get("confianca", 0) >= gate_move:
            cand = os.path.join(CON, sub)
            if os.path.normpath(cand) != os.path.normpath(dirpath):
                destino_dir = cand
        orc = MAXPATH - len(destino_dir) - 1 - len(ext)
        if len(novo) > orc:
            novo = novo[: max(20, orc)].rstrip(" -.,;")
        novo_path = os.path.join(destino_dir, novo + ext.lower())
        if os.path.normpath(novo_path) == os.path.normpath(p):
            stats["ja bom"] += 1
            continue
        acao = ("move+" if destino_dir != dirpath else "") + ("rename" if novo != stem_atual else "move")
        stats[acao.rstrip("+")] += 1
        plano.append((p, novo_path))

    print("\nPlano:", dict(stats))
    with io.open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    import random
    random.seed(71)
    for de, para in random.sample(plano, min(12, len(plano))):
        print("  DE:", os.path.relpath(de, CON)[:100])
        print("  P/:", os.path.relpath(para, CON)[:100])
    if not execu:
        print("Dry-run. Rode com --exec.")
        return
    ok = 0
    vazias = set()
    with io.open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for de, para in plano:
            try:
                os.makedirs(os.path.dirname(para), exist_ok=True)
                dest = para
                i = 2
                while os.path.exists(PRE + dest):
                    s, e = os.path.splitext(para)
                    dest = f"{s} ({i}){e}"
                    i += 1
                os.rename(PRE + de, PRE + dest)
                w.writerow([de, dest])
                vazias.add(os.path.dirname(de))
                ok += 1
            except OSError as e:
                print("ERRO:", de, e)
    for d in sorted(vazias, key=lambda x: x.count(os.sep), reverse=True):
        try:
            os.rmdir(d)
        except OSError:
            pass
    print(f"Aplicados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
