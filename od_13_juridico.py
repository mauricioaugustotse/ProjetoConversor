# -*- coding: utf-8 -*-
r"""OD etapa 13: excelência em 05 - Jurídico e trabalho (gpt-5.4 lê o conteúdo).

Convenções por tipo documental (nº SEI/CNJ é identificador obrigatório quando existir):
  Despacho - <assunto> - SEI <nº> - <data>       Informação <nº> - <unidade> - <assunto> - <data>
  Memorando <nº> - <assunto> - <data>            Parecer - <assunto> - <órgão> - <data>
  Relatório - <assunto> - <órgão/unidade> - <data>   Ata - <reunião/assunto> - <data>
  Minuta - <ato> - <assunto> - <versão/data>     Resolução TSE <nº> - <assunto>
  Termo de Referência - <objeto> - <ano>         Processo SEI <nº> - <conteúdo> (.zip)
  Escritura - <ato> - <partes> - <data>          Certificado - <curso> - <pessoa> - <ano>
  Nota técnica/Ofício/Petição/Acórdão: padrões análogos.
Dossiês (rename APENAS, nunca mover p/ fora): Guillon, Inventário, Problema Aeed,
Processos\Azul, Processos\Unidas, TSE\13.23%. Capturas de tela sem conteúdo: mantidas.

Uso: python od_13_juridico.py [--exec] [--so-plano]  (log: log_od_juridico.csv)
"""
import csv, io, json, os, re, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_15_nao_processuais import extrair_isolado

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
JUR = os.path.join(ROOT, "05 - Jurídico e trabalho")
RESP = os.path.join(BASE, "od_ia_juridico.jsonl")
PLANO = os.path.join(BASE, "plano_od_juridico.csv")
LOG = os.path.join(BASE, "log_od_juridico.csv")
MODELO = "gpt-5.4"
PRE = "\\\\?\\"
MAXPATH = 240
DOSSIES_REL = ("Guillon", "Inventário", "Problema Aeed", os.path.join("Processos", "Azul"),
               os.path.join("Processos", "Unidas"), os.path.join("TSE", "13.23%"))
EXTS_SEM_TEXTO = (".png", ".jpg", ".jpeg", ".gif", ".zip", ".xlsb", ".xlsx", ".mp4")

SYS = """Você organiza a categoria "05 - Jurídico e trabalho" do acervo de um servidor do TSE
(assessor na AGEL; casos pessoais: inventário do pai, processos de locadoras Azul/Unidas, caso Guillon;
tema 13,23% = índice remuneratório VPI). Para cada item (pasta, nome, trecho), gere:
- "novo_stem" na convenção do tipo documental (português com acentos), SEMPRE preservando nº SEI
  (formato AAAA.00.000NNNNN-D ou 7 dígitos) e nº de processo CNJ quando existirem:
  Despacho - <assunto> - SEI <nº> - <data> | Informação <nº> - <unidade> - <assunto> - <data> |
  Memorando <nº> - <assunto> - <data> | Parecer - <assunto> - <órgão> - <data> |
  Relatório - <assunto> - <órgão> - <data ou período> | Ata - <reunião> - <data> |
  Minuta - <ato> - <assunto> - <versão> | Resolução TSE <nº> - <assunto> |
  Termo de Referência - <objeto> - <ano> | Processo SEI <nº> - <assunto> |
  Escritura - <ato> - <partes> - <data> | Certificado - <curso> - <pessoa> - <ano> |
  Ofício/Nota técnica/Petição/Acórdão/Informativo: padrão análogo "Tipo - assunto - órgão - data".
  Se o nome atual já estiver adequado, devolva-o igual.
- "sub": subpasta correta DENTRO de 05 quando o item estiver claramente deslocado
  (ex.: despacho da AGEL solto em TSE -> "TSE\\AGEL\\Despachos e Informações"); senão "manter".
  Itens de dossiês pessoais e qualquer coisa cujo lugar seja fora de 05: "manter".
Dados do CONTEÚDO; não invente; máx ~110 chars.
Responda APENAS JSON: {"itens": [{"id": <n>, "novo_stem": "<nome>", "sub": "<caminho|manter>", "confianca": <0-1>}]}"""

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def nfc(s):
    return unicodedata.normalize("NFC", s)

def em_dossie(rel):
    return any(rel == d or rel.startswith(d + os.sep) for d in DOSSIES_REL)

def runs_digitos(s, minimo=6):
    return {r for r in re.findall(r"\d+", re.sub(r"[.\-\s]", "", s)) if len(r) >= minimo}

def main():
    execu = "--exec" in sys.argv
    so_plano = "--so-plano" in sys.argv
    subs_validas = set()
    alvos = []
    for dp, dn, fn in os.walk(JUR):
        rel = os.path.relpath(dp, JUR)
        if rel != ".":
            subs_validas.add(rel)
        base = os.path.basename(dp).lower()
        if base.endswith(("_arquivos", "_files")):
            continue  # recursos de HTML salvo: intocaveis
        for f in fn:
            alvos.append(os.path.join(dp, f))
    print("Alvos:", len(alvos))

    print("Extraindo conteúdo...")
    textos = {}
    txt_alvos = [p for p in alvos if not p.lower().endswith(EXTS_SEM_TEXTO)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        for p, t in zip(txt_alvos, ex.map(extrair_isolado, txt_alvos)):
            textos[p] = t
    ia_alvos = [p for p in txt_alvos if len(textos.get(p, "")) >= 60]
    print(f"Com texto p/ IA: {len(ia_alvos)}; sem texto (mantidos): {len(alvos)-len(ia_alvos)}")

    cache = {}
    if os.path.exists(RESP):
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass
    pend = [p for p in ia_alvos if p not in cache]
    if pend and not so_plano:
        from openai import OpenAI
        client = OpenAI(api_key=chave_openai())
        lotes = [pend[i:i + 10] for i in range(0, len(pend), 10)]
        feitos = [0]
        def um_lote(lote):
            linhas = [f'id={i} | pasta="{os.path.relpath(os.path.dirname(p), JUR)}" | nome="{os.path.basename(p)}" | trecho="{textos.get(p, "")[:550]}"'
                      for i, p in enumerate(lote)]
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
    for p in ia_alvos:
        it = cache.get(p)
        if not it or it.get("confianca", 0) < 0.6:
            stats["mantido"] += 1
            continue
        dirpath, nome = os.path.split(p)
        stem_atual, ext = os.path.splitext(nome)
        rel_atual = os.path.relpath(dirpath, JUR)
        novo = nfc(str(it.get("novo_stem", "")).strip())
        novo = re.sub(r'[<>:"/\\|?*]', " ", novo)
        novo = re.sub(r"\s+", " ", novo).strip(" -.,;")
        if not novo or novo.lower() in ("manter", "mantido"):
            novo = stem_atual
        faltantes = runs_digitos(stem_atual) - runs_digitos(novo)
        if faltantes:
            for m in re.finditer(r"[\d][\d.\-/]*\d", stem_atual):
                if runs_digitos(m.group(0), 6) & faltantes:
                    novo += f" - {m.group(0)}"
        destino_dir = dirpath
        sub = str(it.get("sub", "manter")).strip().strip("\\")
        if (sub != "manter" and sub in subs_validas and not em_dossie(rel_atual)
                and it.get("confianca", 0) >= 0.75):
            cand = os.path.join(JUR, sub)
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
    random.seed(81)
    for de, para in random.sample(plano, min(12, len(plano))):
        print("  DE:", os.path.relpath(de, JUR)[:100])
        print("  P/:", os.path.relpath(para, JUR)[:100])
    if not execu:
        print("Dry-run. Rode com --exec.")
        return
    ok = 0
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
                ok += 1
            except OSError as e:
                print("ERRO:", de, e)
    print(f"Aplicados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
