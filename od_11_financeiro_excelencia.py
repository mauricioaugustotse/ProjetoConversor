# -*- coding: utf-8 -*-
r"""OD etapa 11: excelência em 03 - Financeiro + contracheques da base toda.

Convenções POR TIPO (competência MM-AAAA é o eixo dos recorrentes):
  Contracheque - <Pessoa> - <Órgão> - MM-AAAA            (intervalos: "07-2024 a 09-2024")
  Extrato - <banco/app> - <conta ou cartão> - MM-AAAA
  Informe de rendimentos - <fonte> - ano-base AAAA - <Pessoa>
  Fatura - cartão final NNNN - MM-AAAA - R$ <valor>
  Declaração IRPF AAAA ano-base AAAA - <Pessoa> | Recibo - entrega IRPF AAAA - <Pessoa> | DARF - IRPF - ...
  Contas - <Pessoa> - MM-AAAA - R$ <total>               (planilhas de contas do mês)
  Exportação Mobills - <descrição> - MM-AAAA
  (transacionais seguem a convenção do od_10)
Alocação: só DENTRO de 03 (ex.: fatura solta -> Contas\Faturas; informe -> IRPF\<ano>);
contracheques em dossiês (Mykonos, Unimed, processos) ficam onde estão, só o nome muda.

Uso: python od_11_financeiro_excelencia.py [--exec] [--so-plano]  (log: log_od_fin11.csv)
"""
import csv, io, json, os, re, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_15_nao_processuais import extrair_isolado

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
FIN = os.path.join(ROOT, "03 - Financeiro")
RESP = os.path.join(BASE, "od_ia_fin11.jsonl")
PLANO = os.path.join(BASE, "plano_od_fin11.csv")
LOG = os.path.join(BASE, "log_od_fin11.csv")
MODELO = "gpt-5.4"
PRE = "\\\\?\\"
MAXPATH = 240
PROT = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
        "Modelos Personalizados do Office", "My Kindle Content")
RX_CC = re.compile(r"contra\s*-?cheque|holerite|demonstrativo de pagamento", re.I)

TIPOS = [
    ("contracheque", RX_CC),
    ("extrato", re.compile(r"\bextrato\b", re.I)),
    ("informe", re.compile(r"informe de rendimento", re.I)),
    ("fatura", re.compile(r"\bfatura\b|cart[aã]o.*final \d{4}|final \d{4}.*cart[aã]o", re.I)),
    ("irpf", re.compile(r"\birpf\b|imposto de renda|declara[cç][aã]o de ajuste|darf", re.I)),
    ("contas_mes", re.compile(r"^contas\b", re.I)),
    ("mobills", re.compile(r"mobills|exporta[cç][aã]o", re.I)),
    ("transacional", re.compile(r"boleto|comprovante|recibo|pagamento|nota fiscal|nfs?-?e", re.I)),
]

SYS = """Você padroniza documentos FINANCEIROS do acervo de um servidor do TSE (família: Maurício,
Ana Carolina, Matheus, Carol, Maria Isabel, Cel Mário). Cada item traz o TIPO detectado, pasta, nome
atual e trecho do conteúdo. Gere "novo_stem" na convenção EXATA do tipo (português com acentos):
- contracheque: "Contracheque - <Pessoa> - <Órgão> - MM-AAAA" (competência do CONTEÚDO; intervalo: "Contracheques - <Pessoa> - <Órgão> - 07-2024 a 09-2024")
- extrato: "Extrato - <banco ou app> - <conta/cartão se houver> - MM-AAAA"
- informe: "Informe de rendimentos - <fonte pagadora> - ano-base AAAA - <Pessoa>"
- fatura: "Fatura - cartão final NNNN - MM-AAAA - R$ <valor>" (ou banco/serviço no lugar do cartão)
- irpf: "Declaração IRPF AAAA ano-base AAAA - <Pessoa>" | "Recibo - entrega IRPF AAAA - <Pessoa>" | "DARF - IRPF <detalhe> - R$ <valor> - <venc>"
- contas_mes: "Contas - <Pessoa> - MM-AAAA - R$ <total se houver>"
- mobills: "Exportação Mobills - <o que é> - MM-AAAA"
- transacional: "<Boleto|Comprovante|Recibo|Pagamento|NF <nº>|Fatura> - <objeto> - <fornecedor> - R$ <valor> - DD-MM-AAAA"
Regras: dados vêm do CONTEÚDO; não invente; omita segmento sem dado; preserve identificadores; máx ~110 chars.
- "sub": se o arquivo estiver em subpasta ERRADA DE 03 - Financeiro, indique a certa (ex.: "Contas\\Faturas",
  "IRPF\\2023"); senão "manter". NUNCA sugerir mover o que está em pasta de outro assunto (dossiê).
Responda APENAS JSON: {"itens": [{"id": <n>, "novo_stem": "<nome>", "sub": "<subpasta|manter>", "confianca": <0-1>}]}"""

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def nfc(s):
    return unicodedata.normalize("NFC", s)

def tipo_de(stem):
    for t, rx in TIPOS:
        if rx.search(stem):
            return t
    return "outro"

def main():
    execu = "--exec" in sys.argv
    so_plano = "--so-plano" in sys.argv
    alvos = []
    # 03 - Financeiro inteira
    for dp, dn, fn in os.walk(FIN):
        for f in fn:
            if not f.startswith("00 - "):
                alvos.append(os.path.join(dp, f))
    # contracheques no resto da base
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        if rel != "." and any(seg in PROT for seg in rel.split(os.sep)):
            dn[:] = []
            continue
        if dp.startswith(FIN):
            continue
        for f in fn:
            if RX_CC.search(os.path.splitext(f)[0]):
                alvos.append(os.path.join(dp, f))
    print("Alvos:", len(alvos))

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
        lotes = [pend[i:i + 10] for i in range(0, len(pend), 10)]
        feitos = [0]
        def um_lote(lote):
            linhas = []
            for i, p in enumerate(lote):
                stem = os.path.splitext(os.path.basename(p))[0]
                linhas.append(f'id={i} | tipo={tipo_de(stem)} | pasta="{os.path.dirname(os.path.relpath(p, ROOT))}" | nome="{os.path.basename(p)}" | trecho="{textos.get(p, "")[:600]}"')
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
    for p in alvos:
        it = cache.get(p)
        if not it or it.get("confianca", 0) < 0.6:
            stats["mantido"] += 1
            continue
        dirpath, nome = os.path.split(p)
        stem_atual, ext = os.path.splitext(nome)
        novo = nfc(str(it.get("novo_stem", "")).strip())
        novo = re.sub(r'[<>:"/\\|?*]', " ", novo)
        novo = re.sub(r"\s+", " ", novo).strip(" -.,;")
        if not novo or novo.lower() in ("manter", "mantido"):
            stats["resposta invalida"] += 1
            continue
        # pessoa citada no nome original nao pode sumir do novo (evita "Fatura - 04-2026" anonima)
        PESSOAS = ("Maurício", "Mauricio", "Ana Carolina", "Carolina", "Carol", "Matheus",
                   "Maria Isabel", "Mário", "Mario", "Ana Lúcia", "Ana Lucia")
        pessoa = next((x for x in PESSOAS if re.search(rf"\b{re.escape(x)}\b", stem_atual, re.I)), None)
        if pessoa and not re.search(rf"\b{re.escape(pessoa)}\b", novo, re.I):
            partes = novo.split(" - ")
            partes.insert(1 if len(partes) > 1 else len(partes), pessoa)
            novo = " - ".join(partes)
        # alocacao: apenas dentro de 03 - Financeiro e apenas p/ arquivos que JA estao em 03
        destino_dir = dirpath
        sub = str(it.get("sub", "manter")).strip().strip("\\")
        if sub != "manter" and p.startswith(FIN) and it.get("confianca", 0) >= 0.7:
            cand = os.path.join(FIN, sub)
            if os.path.isdir(cand) and os.path.normpath(cand) != os.path.normpath(dirpath):
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

    assert not any("12 - Consultoria Legislativa" in a for a, b in plano), "GUARDA: pasta 12!"
    print("\nPlano:", dict(stats))
    with io.open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    import random
    random.seed(61)
    for de, para in random.sample(plano, min(12, len(plano))):
        print("  DE:", os.path.relpath(de, ROOT)[:105])
        print("  P/:", os.path.relpath(para, ROOT)[:105])
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
