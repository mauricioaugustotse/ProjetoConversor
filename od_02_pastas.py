# -*- coding: utf-8 -*-
r"""OD etapa 2: arvore de OneDrive\Documentos — mover Luiz Celso, achatar, uniformizar pastas.

Fases (executadas em sequencia):
  L) "LUIZ CELSO VIEIRA" -> "09 - Família, religião e mensagens\Luiz Celso Vieira".
  A) Achatamento: TODA pasta de nivel >=5 (categoria = nivel 1) funde seus ARQUIVOS ao
     ancestral de nivel 4 (regra dura: max 4 niveis); pastas com 1-2 arquivos em qualquer
     nivel -> gpt-5.4 decide achatar/manter (conjuntos tematicos ficam).
  R) Renomear pastas: Capitalizada pt-BR com acentos, sem prefixo tecnico; siglas
     conhecidas preservadas; tudo-CAPS ambiguos decididos por gpt-5.4 com contexto.
Protegidas (nunca tocar): 12 - Consultoria Legislativa, Arquivos do Outlook,
Modelos Personalizados do Office, My Kindle Content.

Uso: python od_02_pastas.py [--exec] [--so-plano]   (log: log_od_pastas.csv)
"""
import csv, io, json, os, re, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\mauri\OneDrive\Documentos"
LOG = os.path.join(BASE, "log_od_pastas.csv")
PLANO = os.path.join(BASE, "plano_od_pastas.csv")
RESP = os.path.join(BASE, "od_ia_pastas.jsonl")
MODELO = "gpt-5.4"
PRE = "\\\\?\\"

PROIBIDAS = ("12 - Consultoria Legislativa", "Arquivos do Outlook",
             "Modelos Personalizados do Office", "My Kindle Content")
SIGLAS_OK = {"TSE", "OAB", "CEF", "BRB", "IRPF", "CNH", "RG", "CPF", "ENFAM", "POUPEX",
             "REURB", "NFE", "BB", "TCE", "SQSW", "GPS", "USB", "PDF", "OCR", "CONLE",
             "ENAM", "STF", "STJ", "CNJ", "SEI", "PJE", "IPTU", "BNDES", "MPDFT", "SEDEG",
             "TRF1", "TRF2", "TRF3", "TRF4", "TRF5", "TRF6", "AGEL", "GSTI"}
FORMA_SIGLA = {"NFE": "NFe", "PJE": "PJe", "ESPCEX": "EsPCEx", "GOPRO": "GoPro", "PDFS": "PDFs"}
PREPOS = {"e", "de", "da", "do", "das", "dos", "em", "para", "com", "a", "o", "à", "às"}

def protegida(path):
    rel = os.path.relpath(path, ROOT)
    return any(rel == p or rel.startswith(p + os.sep) or (os.sep + p + os.sep) in (os.sep + rel + os.sep)
               for p in PROIBIDAS)

def nivel(path):
    return len(os.path.relpath(path, ROOT).split(os.sep))

PALAVRAS_PT = {"APTO", "CASA", "LIXO", "MAPA", "TESE", "AULA", "AULAS", "CONCURSOS", "CONCURSO",
               "CIRURGIA", "DENTISTA", "OFTALMO", "FITNESS", "REEMBOLSO", "CONTAS", "SEGUROS",
               "VIAGENS", "REVISOES", "CASAMENTO", "CATEQUESE", "JESUS", "MENSAGENS", "SENHAS",
               "ESTUDOS", "PROCESSOS", "INVENTARIO", "LEGISLACAO", "TEMAS", "SELECIONADOS",
               "TOTALIZACAO", "MAPAS", "OUTROS", "MATERIAIS", "RECURSOS", "PROCESSUAL"}

def sequencia_deliberada(path):
    """>=3 irmas com prefixo numerico = ordenacao intencional (nao remover prefixo)."""
    pai = os.path.dirname(path)
    try:
        irmas = [d for d in os.listdir(pai) if os.path.isdir(os.path.join(pai, d))]
    except OSError:
        return False
    numeradas = [d for d in irmas if re.match(r"^\d+[.\-_ ]+\S", d)]
    return len(numeradas) >= 2

MESES_ABREV = {"JAN": "Janeiro", "FEV": "Fevereiro", "MAR": "Março", "ABR": "Abril",
               "MAI": "Maio", "JUN": "Junho", "JUL": "Julho", "AGO": "Agosto",
               "SET": "Setembro", "OUT": "Outubro", "NOV": "Novembro", "DEZ": "Dezembro"}
NOMES_ESPECIAIS = {"TCE_MS": "TCE-MS"}

def nome_pasta_bonito(nome, path=None):
    """Capitalizada pt-BR; None se ambiguo (vai para IA)."""
    if re.search(r"_(arquivos|files)$", nome, re.I):
        return nome  # pasta de recursos de HTML salvo: rename quebra o vinculo
    if nome in NOMES_ESPECIAIS:
        return NOMES_ESPECIAIS[nome]
    m = re.fullmatch(r"(JAN|FEV|MAR|ABR|MAI|JUN|JUL|AGO|SET|OUT|NOV|DEZ)[_\- ](\d{4})", nome, re.I)
    if m:
        return f"{MESES_ABREV[m.group(1).upper()]} {m.group(2)}"
    s = nome
    # prefixo ordenador so cai se seguido de LETRA e fora de sequencia deliberada
    if re.match(r"^\d+[.\-_ ]+(?=[A-Za-zÀ-ÿ])", s) and not (path and sequencia_deliberada(path)):
        s = re.sub(r"^\d+[.\-_ ]+", "", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip() or nome
    letras = [c for c in s if c.isalpha()]
    if not letras or not all(c.isupper() for c in letras):
        # nome misto: aplica formas canonicas em tokens tudo-caps (PDFS -> PDFs) e inicial
        s = " ".join(FORMA_SIGLA.get(t.upper(), t) if t.isupper() else t for t in s.split(" "))
        return s[0].upper() + s[1:] if s and s[0].islower() else s
    # tudo-CAPS: sigla conhecida mantem; palavra pt conhecida capitaliza;
    # desconhecida MANTEM (presumida sigla) — nunca rebaixar orgao/sistema
    toks = s.split(" ")
    out = []
    tem_desconhecida_curta = False
    for i, t in enumerate(toks):
        up = unicodedata.normalize("NFD", t.upper())
        up = "".join(c for c in up if not unicodedata.combining(c))
        if up in FORMA_SIGLA:
            out.append(FORMA_SIGLA[up])
        elif up in SIGLAS_OK:
            out.append(up if t.isalpha() or any(c.isdigit() for c in t) else t)
        elif t.lower() in PREPOS and i > 0:
            out.append(t.lower())
        elif t.isdigit():
            out.append(t)
        elif up in PALAVRAS_PT:
            out.append(t.capitalize())
        elif len(t) <= 3:
            tem_desconhecida_curta = True
            out.append(t)
        else:
            out.append(t)  # sigla presumida: mantem
    if tem_desconhecida_curta and len(toks) == 1:
        return None  # "LE", "DH": so IA sabe expandir
    return " ".join(out)

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def ia_lote(system, linhas_por_item, resp_path, lote_n=12):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    itens = list(linhas_por_item.items())
    lotes = [itens[i:i + lote_n] for i in range(0, len(itens), lote_n)]
    def um_lote(lote):
        linhas = [f"id={i} | {desc}" for i, (chave, desc) in enumerate(lote)]
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
                        it["chave"] = lote[i][0]
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

def carregar_resp(path):
    cache = {}
    if os.path.exists(path):
        for linha in io.open(path, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["chave"]] = it
            except Exception:
                pass
    return cache

SYS_RESIDUO = """Você organiza o acervo pessoal de um servidor do TSE. Cada item é uma PASTA com 1-2 arquivos.
Decida: "achatar" (os arquivos sobem para a pasta-mãe; a pasta some) quando ela não agrega significado
(resíduo técnico, pasta de download, nome genérico), ou "manter" quando é um conjunto temático deliberado
(matéria de estudo, dossiê de um assunto, coleção que ainda vai crescer).
Responda APENAS JSON: {"itens": [{"id": <n>, "acao": "achatar"|"manter", "confianca": <0-1>}]}"""

SYS_NOME = """Você padroniza nomes de PASTAS do acervo pessoal de um servidor do TSE (português correto, com acentos).
Cada item: nome atual da pasta, pasta-mãe e exemplos de arquivos dentro. Proponha o nome ideal:
Capitalizada pt-BR ("Cirurgia", "Apto 216 Norte", "Mapas da Lulu"), acentos/cedilha corretos, sem prefixos
técnicos ("1_"), siglas reais em maiúsculas (TSE, OAB, CEF). Se o nome atual já estiver adequado, repita-o.
Responda APENAS JSON: {"itens": [{"id": <n>, "novo_nome": "<nome>", "confianca": <0-1>}]}"""

def main():
    execu = "--exec" in sys.argv
    so_plano = "--so-plano" in sys.argv
    plano = []  # (tipo, origem, destino)

    # ---- Fase L: Luiz Celso ----
    lc = os.path.join(ROOT, "LUIZ CELSO VIEIRA")
    lc_dest = os.path.join(ROOT, "09 - Família, religião e mensagens", "Luiz Celso Vieira")
    if os.path.isdir(lc):
        plano.append(("move_pasta", lc, lc_dest))

    # ---- Fase A: achatamento ----
    pastas = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if not protegida(os.path.join(dp, d))]
        if dp != ROOT and not protegida(dp):
            pastas.append(dp)

    # A1: arquivos em pastas nivel >=5 sobem para o ancestral nivel 4 (regra dura)
    n5 = [p for p in pastas if nivel(p) >= 5]
    for p in sorted(n5, key=nivel, reverse=True):
        anc = p
        while nivel(anc) > 4:
            anc = os.path.dirname(anc)
        # \\?\ obrigatorio: sem ele, isfile devolve False silencioso em caminho >260
        try:
            nomes = os.listdir(PRE + p)
        except OSError:
            nomes = []
        for f in nomes:
            fp = os.path.join(p, f)
            if os.path.isfile(PRE + fp):
                plano.append(("move_arq", fp, os.path.join(anc, f)))
    print(f"Pastas nivel >=5 (fundem ao nivel 4): {len(n5)}")

    # A2: pastas com 1-2 arquivos (e sem subpastas) em nivel <=4 -> IA decide
    residuos = {}
    for p in pastas:
        if nivel(p) > 4 or nivel(p) == 1:
            continue
        try:
            itens = os.listdir(p)
        except OSError:
            continue
        arqs = [i for i in itens if os.path.isfile(os.path.join(p, i))]
        subs = [i for i in itens if os.path.isdir(os.path.join(p, i))]
        if not subs and 1 <= len(arqs) <= 2:
            residuos[p] = (f'pasta="{os.path.relpath(p, ROOT)}" | arquivos={arqs}')
    print(f"Pastas-residuo (1-2 arquivos): {len(residuos)}")
    cache = carregar_resp(RESP)
    pend = {k: v for k, v in residuos.items() if k not in cache}
    if pend and not so_plano:
        print(f"IA decidindo {len(pend)} residuos...")
        ia_lote(SYS_RESIDUO, pend, RESP)
        cache = carregar_resp(RESP)
    achatadas = 0
    for p in residuos:
        it = cache.get(p)
        if it and it.get("acao") == "achatar" and it.get("confianca", 0) >= 0.6:
            for f in os.listdir(p):
                fp = os.path.join(p, f)
                if os.path.isfile(fp):
                    plano.append(("move_arq", fp, os.path.join(os.path.dirname(p), f)))
            achatadas += 1
    print(f"Residuos a achatar (IA): {achatadas}")

    # ---- Fase R: renomear pastas (apos achatamento; profunda -> rasa) ----
    ambiguas = {}
    renomes = []
    ja_achatadas = {o for t, o, d in plano if t == "move_arq"}
    for p in sorted(pastas, key=nivel, reverse=True):
        if nivel(p) == 1 or nivel(p) >= 5:
            continue
        nome = os.path.basename(p)
        novo = nome_pasta_bonito(nome, p)
        if novo is None:
            try:
                exemplos = [f for f in os.listdir(p)][:3]
            except OSError:
                exemplos = []
            ambiguas[p] = (f'nome="{nome}" | mae="{os.path.basename(os.path.dirname(p))}" | exemplos={exemplos}')
        elif novo != nome:
            renomes.append((p, novo))
    print(f"Pastas com rename deterministico: {len(renomes)}; ambiguas p/ IA: {len(ambiguas)}")
    cache2 = carregar_resp(RESP + ".nomes")
    pend2 = {k: v for k, v in ambiguas.items() if k not in cache2}
    if pend2 and not so_plano:
        print(f"IA nomeando {len(pend2)} pastas...")
        ia_lote(SYS_NOME, pend2, RESP + ".nomes")
        cache2 = carregar_resp(RESP + ".nomes")
    for p in ambiguas:
        it = cache2.get(p)
        if it and it.get("novo_nome", "").strip() and it.get("confianca", 0) >= 0.6:
            novo = unicodedata.normalize("NFC", it["novo_nome"].strip())
            novo = re.sub(r'[<>:"/\\|?*]', " ", novo).strip(" -.,;")
            if novo and novo != os.path.basename(p):
                renomes.append((p, novo))
    for p, novo in sorted(renomes, key=lambda x: nivel(x[0]), reverse=True):
        plano.append(("ren_pasta", p, os.path.join(os.path.dirname(p), novo)))

    # ---- seguranca + saida ----
    assert not any("12 - Consultoria Legislativa" in o or "12 - Consultoria Legislativa" in d
                   for t, o, d in plano), "GUARDA: plano tocaria a pasta proibida!"
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "origem", "destino"])
        w.writerows(plano)
    resumo = Counter(t for t, o, d in plano)
    print("\nPlano:", dict(resumo), "->", PLANO)
    for t, o, d in plano[:12]:
        print(f"  [{t}] {os.path.relpath(o, ROOT)[:70]} -> {os.path.relpath(d, ROOT)[:70]}")
    if not execu:
        print("Dry-run. Rode com --exec para aplicar.")
        return

    ok, erros = 0, 0
    vazias = set()
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "de", "para"])
        for t, origem, destino in plano:
            try:
                if t == "ren_pasta" and not os.path.isdir(origem):
                    continue  # pai ja renomeado? caminhos sao processados fundo->raso, ok
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                dest = destino
                i = 2
                while os.path.exists(PRE + dest):
                    if t == "move_arq":
                        s, e = os.path.splitext(destino)
                        dest = f"{s} ({i}){e}"
                    else:
                        dest = f"{destino} ({i})"
                    i += 1
                os.rename(PRE + origem, PRE + dest)
                w.writerow([t, origem, dest])
                if t == "move_arq":
                    vazias.add(os.path.dirname(origem))
                ok += 1
            except OSError as e:
                print(f"ERRO [{t}]: {origem}: {e}")
                erros += 1
    # remove cascas vazias (de baixo p/ cima)
    removidas = 0
    for d in sorted(vazias, key=nivel, reverse=True):
        atual = d
        while atual != ROOT and nivel(atual) >= 2:
            try:
                os.rmdir(atual)
                removidas += 1
                atual = os.path.dirname(atual)
            except OSError:
                break
    print(f"Aplicados: {ok} (erros: {erros}); cascas vazias removidas: {removidas}. Log: {LOG}")

if __name__ == "__main__":
    main()
