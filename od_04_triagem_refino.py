# -*- coding: utf-8 -*-
r"""OD etapa 4: triagem + refino com gpt-5.4 (AMBAS as fases) em OneDrive\Documentos.

- Triagem (todos os ~6,3k do escopo): destino na whitelist (categoria\subpasta) ou
  "manter"; nome_ok true/false.
- Refino (flagados): confirma destino e reescreve o nome na convencao COM ACENTOS:
  "Tipo - descrição - pessoa/órgão - data". NFC; identificadores preservados; <=240.
- Fotos/videos com nome fraco (IMG_*, VID_*, WhatsApp*, AAAAMMDD_HHMMSS): caminho
  deterministico "<Pasta> - <AAAA-MM-DD> (seq)" (data do nome ou mtime), sem IA.
- Nao-textuais sem texto: intocados.

Uso: python od_04_triagem_refino.py [--exec] [--so-plano]  (log: log_od_arquivos.csv)
"""
import csv, datetime, io, json, os, re, sys, time, unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

ROOT = r"C:\Users\mauri\OneDrive\Documentos"
CAB = os.path.join(BASE, "od_cabecalhos.tsv")
RESP_A = os.path.join(BASE, "od_ia_triagem.jsonl")
RESP_B = os.path.join(BASE, "od_ia_refino.jsonl")
PLANO = os.path.join(BASE, "plano_od_arquivos.csv")
LOG = os.path.join(BASE, "log_od_arquivos.csv")
MODELO = "gpt-5.4"
MAXPATH = 240
PRE = "\\\\?\\"
PROIBIDAS = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
             "Modelos Personalizados do Office", "My Kindle Content")
RX_FOTO_FRACA = re.compile(r"^(IMG[_\-]?\d+|VID[_\-]?\d+|DSC\d+|WhatsApp (Image|Video).*|"
                           r"\d{8}[_\-]\d{6}.*|Screenshot[_\- ].*|Captura de tela.*|PHOTO[_\-]?\d+)$", re.I)
# dossiês temáticos deliberados: o conteúdo NÃO sai deles (rename ok, move não) —
# o processo da Unimed pertence ao dossiê Unimed; os processos do pai, ao acervo dele;
# os acórdãos da Tese são a bibliografia da tese
DOSSIES = (r"02 - Saúde\Unimed", r"01 - Pessoal e identidade\Câmara dos Deputados",
           r"09 - Família, religião e mensagens\Luiz Celso Vieira",
           r"06 - Estudos, concursos e leitura\Tese",
           r"06 - Estudos, concursos e leitura\Concursos")

def em_dossie(rel_dir):
    return next((d for d in DOSSIES if rel_dir == d or rel_dir.startswith(d)), None)
EXTS_MIDIA = (".jpg", ".jpeg", ".png", ".heic", ".mp4", ".mov", ".gif", ".webp")

def protegida(path):
    rel = os.path.relpath(path, ROOT)
    return any(rel.startswith(p) or (os.sep + p + os.sep) in rel for p in PROIBIDAS)

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def whitelist():
    dests = []
    for cat in sorted(os.listdir(ROOT)):
        pc = os.path.join(ROOT, cat)
        if not os.path.isdir(pc) or protegida(pc) or not re.match(r"^(0\d|1[01]) - ", cat):
            continue
        dests.append(cat)
        for d in sorted(os.listdir(pc)):
            if os.path.isdir(os.path.join(pc, d)):
                dests.append(f"{cat}\\{d}")
    return dests

def carregar_cab():
    cab = {}
    if os.path.exists(CAB):
        for linha in io.open(CAB, encoding="utf-8"):
            p = linha.rstrip("\n").split("\t")
            if len(p) == 3:
                cab[p[0]] = p[2]
    return cab

def nfc(s):
    return unicodedata.normalize("NFC", s)

def sanitizar(stem):
    s = nfc(stem)
    s = re.sub(r'[<>:"/\\|?*]', " ", s)
    s = "".join(c for c in s if c.isprintable())
    return re.sub(r"\s+", " ", s).strip(" -.,;")

def runs_digitos(s, minimo=6):
    return {r for r in re.findall(r"\d+", re.sub(r"[.\-\s]", "", s)) if len(r) >= minimo}

def data_de_midia(nome, path):
    m = re.search(r"(20\d{2})[\-_]?(\d{2})[\-_]?(\d{2})[_\-]", nome)
    if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except OSError:
        return None

SYS_A = """Você faz a triagem do acervo pessoal de um servidor do TSE (OneDrive\\Documentos), em português.
Destinos válidos (categoria e, quando fizer sentido, subpasta):
{DESTINOS}
Para cada item (pasta atual, nome, trecho do conteúdo):
- "destino": onde o arquivo DEVERIA estar, ou "manter" se a pasta atual já é adequada (na dúvida, "manter").
- "nome_ok": false SOMENTE se o nome não permite saber o que o arquivo é, ou foge muito do padrão
  "Tipo - descrição - pessoa/órgão - data"; caso contrário true.
Responda APENAS JSON: {{"itens": [{{"id": <n>, "destino": "<caminho|manter>", "nome_ok": true|false, "confianca": <0-1>}}]}}"""

SYS_B = """Você organiza o acervo pessoal de um servidor do TSE. Convenção de nomes (PORTUGUÊS CORRETO,
COM acentos e cedilha): "Tipo - descrição do objeto - pessoa/órgão - data" (DD-MM-AAAA ou MM-AAAA).
Exemplos: "Boleto - condomínio Edifício Itarana - 10-2024.pdf"; "Certidão - inteiro teor OAB-MG nº 132.411 - 07-01-2026.pdf".
Destinos válidos:
{DESTINOS}
Para cada item (pasta atual, nome, trecho): confirme o melhor destino ("manter" se a atual serve) e,
se o nome não indicar o conteúdo, proponha "novo_stem" na convenção (senão devolva "").
PRESERVE números de processo/protocolo/lei/CPF e datas do nome atual; nada de inventar; máx ~120 caracteres.
Responda APENAS JSON: {{"itens": [{{"id": <n>, "destino": "<caminho|manter>", "novo_stem": "<nome|vazio>", "confianca": <0-1>}}]}}"""

def rodar_ia(itens, textos, system, resp_path, lote_n=10):
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
                    print("FALHA lote:", e)
                    return []
                time.sleep(2 * (tent + 1))
    with ThreadPoolExecutor(max_workers=6) as ex, io.open(resp_path, "a", encoding="utf-8") as f:
        for res in ex.map(um_lote, lotes):
            for item in res or []:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            feitos[0] += 1
            if feitos[0] % 60 == 0:
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

def main():
    execu = "--exec" in sys.argv
    so_plano = "--so-plano" in sys.argv
    dests = whitelist()
    set_dests = set(dests)
    print("Whitelist:", len(dests), "destinos")

    textos = carregar_cab()
    alvos, midias = [], []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if not protegida(os.path.join(dp, d))]
        for f in fn:
            if f.startswith("00 - ") or f.lower() == "desktop.ini":
                continue
            p = os.path.join(dp, f)
            stem, ext = os.path.splitext(f)
            if ext.lower() in EXTS_MIDIA:
                if RX_FOTO_FRACA.match(stem):
                    midias.append(p)
                continue
            alvos.append(p)
    com_texto = [p for p in alvos if len(textos.get(p, "")) >= 80]
    print(f"Alvos IA (com texto): {len(com_texto)} de {len(alvos)}; mídias fracas: {len(midias)}")

    cache_a = carregar_resp(RESP_A)
    pend_a = [p for p in com_texto if p not in cache_a]
    if pend_a and not so_plano:
        print(f"Triagem gpt-5.4: {len(pend_a)}...")
        rodar_ia(pend_a, textos, SYS_A.replace("{DESTINOS}", "\n".join(dests)), RESP_A)
        cache_a = carregar_resp(RESP_A)

    flagados = []
    for p in com_texto:
        it = cache_a.get(p)
        if not it:
            continue
        atual = os.path.dirname(os.path.relpath(p, ROOT))
        d = str(it.get("destino", "manter")).strip().strip("\\")
        move = d != "manter" and d in set_dests and not atual.startswith(d)
        if move or not it.get("nome_ok", True):
            flagados.append(p)
    print(f"Flagados: {len(flagados)}")

    cache_b = carregar_resp(RESP_B)
    pend_b = [p for p in flagados if p not in cache_b]
    if pend_b and not so_plano:
        print(f"Refino gpt-5.4: {len(pend_b)}...")
        rodar_ia(pend_b, textos, SYS_B.replace("{DESTINOS}", "\n".join(dests)), RESP_B)
        cache_b = carregar_resp(RESP_B)

    plano = []
    stats = Counter()
    for p in flagados:
        it = cache_b.get(p)
        if not it or it.get("confianca", 0) < 0.6:
            stats["mantido"] += 1
            continue
        dirpath, nome = os.path.split(p)
        stem_atual, ext = os.path.splitext(nome)
        d = str(it.get("destino", "manter")).strip().strip("\\")
        novo_stem = sanitizar(str(it.get("novo_stem", "")).strip())
        if novo_stem:
            faltantes = runs_digitos(stem_atual) - runs_digitos(novo_stem)
            if faltantes:
                for m in re.finditer(r"[\d][\d.\-/]*\d", stem_atual):
                    if runs_digitos(m.group(0), 6) & faltantes:
                        novo_stem += f" - {m.group(0)}"
        stem_final = novo_stem if novo_stem and nfc(novo_stem).lower() != nfc(stem_atual).lower() else stem_atual
        atual_rel = os.path.dirname(os.path.relpath(p, ROOT))
        quer_mover = d != "manter" and d in set_dests and not atual_rel.startswith(d)
        dossie = em_dossie(nfc(atual_rel))
        if quer_mover and dossie and not d.startswith(dossie.split(os.sep)[0]):
            quer_mover = False  # conteúdo de dossiê não sai dele
            stats["move bloqueado (dossiê)"] += 1
        destino_dir = os.path.join(ROOT, d) if quer_mover else dirpath
        orcamento = MAXPATH - len(destino_dir) - 1 - len(ext)
        if len(stem_final) > orcamento:
            stem_final = stem_final[: max(20, orcamento)].rstrip(" -.,;")
        novo_path = os.path.join(destino_dir, stem_final + ext)
        if os.path.normpath(novo_path) == os.path.normpath(p):
            stats["mantido"] += 1
            continue
        acao = ("move+" if destino_dir != dirpath else "") + ("rename" if stem_final != stem_atual else "move")
        stats[acao.rstrip("+")] += 1
        plano.append((p, novo_path))

    # midias fracas: deterministico
    seq = defaultdict(int)
    for p in midias:
        dirpath, nome = os.path.split(p)
        stem, ext = os.path.splitext(nome)
        data = data_de_midia(stem, p)
        if not data:
            continue
        chave = (dirpath, data)
        seq[chave] += 1
        pasta_rotulo = os.path.basename(dirpath)
        novo = f"{pasta_rotulo} - {data}" + (f" ({seq[chave]})" if seq[chave] > 1 else "")
        novo_path = os.path.join(dirpath, novo + ext.lower())
        if os.path.normpath(novo_path) != os.path.normpath(p):
            plano.append((p, novo_path))
            stats["mídia renomeada"] += 1

    assert not any("12 - Consultoria Legislativa" in a or "12 - Consultoria Legislativa" in b
                   for a, b in plano), "GUARDA: plano tocaria a pasta proibida!"
    print("\nPlano:", len(plano), dict(stats))
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    import random
    random.seed(31)
    for de, para in random.sample(plano, min(10, len(plano))):
        print("  DE :", os.path.relpath(de, ROOT)[:100])
        print("  P/ :", os.path.relpath(para, ROOT)[:100])
    if not execu:
        print("Dry-run. Rode com --exec para aplicar.")
        return

    ok, erros = 0, 0
    with open(LOG, "w", newline="", encoding="utf-8") as f:
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
                erros += 1
    print(f"Aplicados: {ok} (erros: {erros}). Log: {LOG}")

if __name__ == "__main__":
    main()
