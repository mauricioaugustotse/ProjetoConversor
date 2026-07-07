# -*- coding: utf-8 -*-
r"""OD etapa 9: reorganizacao cirurgica de 02 - Saúde\Dentista para comparacao de
tratamentos (pedido de 06/07/2026).

Subpastas-alvo:
  Orçamentos | Mensalidades | Notas fiscais e pagamentos | Contratos e planos | Materiais e orientações
Convencao (ordenavel e comparavel; gpt-5.4 le o conteudo p/ capturar valores/datas):
  Orçamento - <procedimento> - <clínica> - R$ <valor> - <AAAA ou data>.pdf
  Mensalidade NN - <boleto|comprovante> - <clínica> - R$ <valor> - DD-MM-AAAA.pdf   (NN zero-padded)
  NF <nº> - <clínica> - R$ <valor> - DD-MM-AAAA.pdf
  Pagamento - <meio/fim> - R$ <valor> - DD-MM-AAAA.pdf
  Contrato/Material - descricao - data.pdf

Uso: python od_09_dentista.py [--exec]   (log: log_od_dentista.csv)
"""
import csv, io, json, os, re, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_15_nao_processuais import extrair_isolado

PASTA = r"C:\Users\mauri\OneDrive\Documentos\02 - Saúde\Dentista"
RESP = os.path.join(BASE, "od_ia_dentista.jsonl")
LOG = os.path.join(BASE, "log_od_dentista.csv")
MODELO = "gpt-5.4"
PRE = "\\\\?\\"
SUBS = ["Orçamentos", "Mensalidades", "Notas fiscais e pagamentos",
        "Contratos e planos", "Materiais e orientações"]

SYS = """Você organiza a pasta "Dentista" do acervo de um servidor do TSE. Os dados serão usados
para COMPARAR tratamentos dentários — valores e datas no nome são essenciais.
Para cada item (nome atual + trecho do conteúdo), devolva:
- "sub": uma de: Orçamentos | Mensalidades | Notas fiscais e pagamentos | Contratos e planos | Materiais e orientações
- "novo_stem": nome na convenção EXATA da subpasta (português com acentos):
  Orçamentos:  "Orçamento - <procedimento> - <clínica> - R$ <valor> - <data ou ano>"
  Mensalidades: "Mensalidade NN - <boleto|comprovante> - <clínica abreviada> - R$ <valor> - DD-MM-AAAA"
     (NN com dois dígitos, ex. "Mensalidade 04"; use o número ordinal do nome atual; a data é o
      vencimento no boleto ou a data do pagamento no comprovante)
  Notas fiscais e pagamentos: "NF <nº> - <clínica> - R$ <valor> - <data>" ou "Pagamento - <meio ou fim> - R$ <valor> - <data>"
     (reembolsos: "Reembolso - <órgão> - R$ <valor> - <data>")
  Contratos e planos: "Contrato - <objeto> - <clínica> - <data>"
  Materiais e orientações: "Material - <item> - <detalhe>" ou "Orientações - <item>"
- Extraia valor (R$) e data DO CONTEÚDO quando não estiverem no nome; se realmente não houver, omita o segmento.
- "confianca": 0-1.
Responda APENAS JSON: {"itens": [{"id": <n>, "sub": "<subpasta>", "novo_stem": "<nome>", "confianca": <x>}]}"""

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def nfc(s):
    return unicodedata.normalize("NFC", s)

def main():
    execu = "--exec" in sys.argv
    alvos = []
    for dp, dn, fn in os.walk(PASTA):
        for f in fn:
            alvos.append(os.path.join(dp, f))
    print("Arquivos:", len(alvos))

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
    if pend:
        from openai import OpenAI
        client = OpenAI(api_key=chave_openai())
        lotes = [pend[i:i + 8] for i in range(0, len(pend), 8)]
        def um_lote(lote):
            linhas = [f'id={i} | nome="{os.path.basename(p)}" | trecho="{textos.get(p, "")[:600]}"'
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
        with ThreadPoolExecutor(max_workers=4) as ex, io.open(RESP, "a", encoding="utf-8") as f:
            for res in ex.map(um_lote, lotes):
                for item in res or []:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
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
        stem_atual, ext = os.path.splitext(os.path.basename(p))
        if not it or it.get("sub") not in SUBS or it.get("confianca", 0) < 0.6:
            stats["mantido (sem resposta/conf)"] += 1
            continue
        novo = nfc(str(it.get("novo_stem", "")).strip()) or stem_atual
        novo = re.sub(r'[<>:"/\\|?*]', " ", novo)
        novo = re.sub(r"\s+", " ", novo).strip(" -.,;")
        # zero-pad da mensalidade (garantia pós-IA)
        novo = re.sub(r"^(Mensalidade )(\d)\b", r"\g<1>0\g<2>", novo)
        sub = it["sub"]
        # "Mensalidade NN" so vale se o ordinal existe no nome ORIGINAL (a IA nao inventa numero)
        m_novo = re.match(r"^Mensalidade (\d{2})\b", novo)
        if m_novo or sub == "Mensalidades":
            m_orig = re.search(r"\b(\d{1,2})\s*[ªa°o]?\s*mensalidade|mensalidade\s+(\d{1,2})\b",
                               stem_atual, re.I)
            if not m_orig:
                sub = "Notas fiscais e pagamentos"
                resto = re.sub(r"^Mensalidade \d{2} - ", "", novo)
                tipo = "Boleto" if "boleto" in stem_atual.lower() else "Pagamento"
                desc = re.sub(r"^(boleto|comprovante) - ", "", resto)
                novo = f"{tipo} - {desc}"
                stats["rebaixado (ordinal inventado)"] += 1
            elif m_novo:
                n_orig = int(m_orig.group(1) or m_orig.group(2))
                if int(m_novo.group(1)) != n_orig:
                    novo = re.sub(r"^Mensalidade \d{2}", f"Mensalidade {n_orig:02d}", novo)
        destino = os.path.join(PASTA, sub, novo + ext.lower())
        if os.path.normpath(destino) == os.path.normpath(p):
            stats["já correto"] += 1
            continue
        plano.append((p, destino))
        stats[sub] += 1

    print("\nPlano:", len(plano), dict(stats))
    with io.open(os.path.join(BASE, "plano_od_dentista.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    for de, para in plano[:12]:
        print("  DE:", os.path.basename(de)[:95])
        print("  P/:", os.path.relpath(para, PASTA)[:95])
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
    # remove subpasta antiga vazia
    try:
        os.rmdir(os.path.join(PASTA, "Mensalidades")) if not os.listdir(os.path.join(PASTA, "Mensalidades")) else None
    except OSError:
        pass
    print(f"Aplicados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
