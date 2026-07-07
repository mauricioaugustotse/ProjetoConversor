# -*- coding: utf-8 -*-
r"""OD etapa 10: padroniza TODOS os documentos transacionais (boletos, comprovantes,
recibos, NFs, faturas, mensalidades) na convencao comparavel, onde estiverem.

Convencao: "<Prefixo> - <objeto/servico> - <fornecedor/pessoa> - R$ <valor> - <data>"
Prefixos canonicos: Boleto | Comprovante | Recibo | Pagamento | NF <n> | Fatura | Mensalidade NN
Regras aprendidas: ordinal de mensalidade NUNCA inventado; conteudo e a fonte de valor/data;
dossies tematicos = rename no lugar (move so com gate 0.75 p/ fora de pasta inadequada).

Uso: python od_10_financeiro.py [--exec] [--so-plano]  (log: log_od_financeiro.csv)
"""
import csv, io, json, os, re, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_15_nao_processuais import extrair_isolado

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
RESP = os.path.join(BASE, "od_ia_financeiro.jsonl")
PLANO = os.path.join(BASE, "plano_od_financeiro.csv")
LOG = os.path.join(BASE, "log_od_financeiro.csv")
MODELO = "gpt-5.4"
PRE = "\\\\?\\"
MAXPATH = 240
PROT = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
        "Modelos Personalizados do Office", "My Kindle Content")
DOSSIES = (r"02 - Saúde\Unimed", r"01 - Pessoal e identidade\Câmara dos Deputados",
           r"09 - Família, religião e mensagens\Luiz Celso Vieira",
           r"06 - Estudos, concursos e leitura\Tese",
           r"06 - Estudos, concursos e leitura\Concursos")
RX_ALVO = re.compile(r"\b(boleto|comprovante|recibo|pagamento|fatura|nfs?-?e?\b|nota fiscal|mensalidade|transfer[eê]ncia|ted|pix)\b", re.I)
RX_OK = re.compile(r"^(Boleto|Comprovante|Recibo|Pagamento|NF \S+|Fatura|Mensalidade \d{2}|Mensalidade \(sem nº\))( - .+)? - R\$ [\d.,]+ - \d{2}-\d{2}-\d{4}$")
PREFIXOS = ("Boleto", "Comprovante", "Recibo", "Pagamento", "NF", "Fatura", "Mensalidade")

SYS = """Você padroniza documentos financeiros do acervo pessoal de um servidor do TSE, para permitir
COMPARAÇÃO de despesas. Para cada item (pasta, nome atual, trecho do conteúdo), gere o nome na convenção
EXATA (português com acentos):
"<Prefixo> - <objeto/serviço> - <fornecedor ou pessoa> - R$ <valor> - <data DD-MM-AAAA>"
Prefixos permitidos: Boleto | Comprovante | Recibo | Pagamento | NF <número> | Fatura | Mensalidade NN.
- Extraia VALOR e DATA do conteúdo quando não estiverem no nome (boleto: vencimento; comprovante: data paga).
- "Mensalidade NN" APENAS se o número ordinal já estiver no nome atual; nunca invente.
- Se não houver valor no documento, omita o segmento R$; idem data. Não invente nada.
- Mantenha identificadores (nº NF, protocolo, placa do veículo, competência "10-2024").
- máx ~110 caracteres. Se o nome atual já estiver perfeito na convenção, devolva-o igual.
Responda APENAS JSON: {"itens": [{"id": <n>, "novo_stem": "<nome>", "confianca": <0-1>}]}"""

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def nfc(s):
    return unicodedata.normalize("NFC", s)

def runs_digitos(s, minimo=6):
    return {r for r in re.findall(r"\d+", re.sub(r"[.\-\s]", "", s)) if len(r) >= minimo}

def main():
    execu = "--exec" in sys.argv
    so_plano = "--so-plano" in sys.argv
    alvos = []
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        if rel != "." and any(seg in PROT for seg in rel.split(os.sep)):
            dn[:] = []
            continue
        for f in fn:
            stem = os.path.splitext(f)[0]
            if f.startswith("00 - ") or not RX_ALVO.search(stem):
                continue
            if RX_OK.match(nfc(stem)):
                continue  # ja na convencao completa
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
            linhas = [f'id={i} | pasta="{os.path.dirname(os.path.relpath(p, ROOT))}" | nome="{os.path.basename(p)}" | trecho="{textos.get(p, "")[:600]}"'
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
        if not novo.startswith(PREFIXOS):
            stats["sem prefixo canonico"] += 1
            continue
        novo = re.sub(r"^(Mensalidade )(\d)\b", r"\g<1>0\g<2>", novo)
        m_novo = re.match(r"^Mensalidade (\d{2})\b", novo)
        if m_novo:
            m_orig = re.search(r"\b(\d{1,2})\s*[ªa°o]?\s*mensalidade|mensalidade\s+(\d{1,2})\b", stem_atual, re.I)
            if not m_orig:
                novo = re.sub(r"^Mensalidade \d{2}", "Pagamento", novo)
                stats["ordinal inventado -> Pagamento"] += 1
        # datas comparadas SEMANTICAMENTE (evita re-anexar data ja presente quando
        # os centavos ",00" se fundem com a data na comparacao de digitos)
        def datas_de(s):
            out = set()
            for d, mo, a in re.findall(r"\b(\d{2})-(\d{2})-(20\d{2})\b", s):
                out.add((d, mo, a))
            return out
        datas_novo = datas_de(novo)
        faltantes = runs_digitos(stem_atual) - runs_digitos(novo)
        if faltantes:
            for m in re.finditer(r"[\d][\d.\-/]*\d", stem_atual):
                trecho = m.group(0)
                md = re.fullmatch(r"(\d{2})-(\d{2})-(20\d{2})", trecho)
                if md and (md.group(1), md.group(2), md.group(3)) in datas_novo:
                    continue  # data ja presente: nao re-anexar
                if runs_digitos(trecho, 6) & faltantes:
                    novo += f" - {trecho}"
        # anos de 4 digitos do original devem constar do novo
        for ano in set(re.findall(r"\b(19\d{2}|20\d{2})\b", stem_atual)) - set(re.findall(r"\b(19\d{2}|20\d{2})\b", novo)):
            novo += f" - {ano}"
        if nfc(novo).lower() == nfc(stem_atual).lower():
            stats["ja bom"] += 1
            continue
        orc = MAXPATH - len(dirpath) - 1 - len(ext)
        if len(novo) > orc:
            novo = novo[: max(20, orc)].rstrip(" -.,;")
        plano.append((p, os.path.join(dirpath, novo + ext.lower())))
        stats["renomear"] += 1

    assert not any("12 - Consultoria Legislativa" in a for a, b in plano), "GUARDA: pasta 12!"
    print("\nPlano:", dict(stats))
    with io.open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    import random
    random.seed(51)
    for de, para in random.sample(plano, min(10, len(plano))):
        print("  DE:", os.path.basename(de)[:100])
        print("  P/:", os.path.basename(para)[:100])
    if not execu:
        print("Dry-run. Rode com --exec.")
        return
    ok = 0
    with io.open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for de, para in plano:
            try:
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
    print(f"Renomeados: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
