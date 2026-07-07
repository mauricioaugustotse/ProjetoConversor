# -*- coding: utf-8 -*-
r"""Org2 etapa 12: peca processual fora de "01 - Juridico" -> reordena para o folder certo.

Deteccao: CNJ no nome OU classe+numero com indicio de peca, fora de 01 e fora de
"Julgados e sumulas". Triagem em camadas:
  1) manter: listas/pautas/indices de sessao (administrativo legitimo);
  2) mover por regra: tipo de peca claro no nome -> 01\<Tipo>\<Materia (justica do CNJ)>;
  3) julgado de ESTUDO em "Cursos e certificados" (bug curso<recurso, sem CNJ/tipo)
     -> "Julgados e sumulas" da propria disciplina;
  4) duvidosos -> gpt-5.4 decide com o trecho do conteudo (peca real? tipo? materia? ou manter).
Enderecamento fino: encaixa na subpasta existente (grupo de classe -> faixa de ano; faixa
de ano; "Sem ano"; cria "AAAA" quando preciso).

Uso: python org2_12_minutas.py [--exec] [--so-plano]   (log: log_org2_minutas.csv)
"""
import csv, io, json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from org2_03_dedup_texto import extrair_completo
from org2_04_quebra_pastas import (GRUPO_ELEITORAL, GRUPO_TRABALHISTA, GRUPO_ACORDAOS,
                                   classe, ano_do_nome)

ROOT = r"C:\Users\mauri\HD_Mau"
RESP = os.path.join(BASE, "org2_ia_minutas.jsonl")
PLANO = os.path.join(BASE, "plano_org2_minutas.csv")
LOG = os.path.join(BASE, "log_org2_minutas.csv")
MODELO = "gpt-5.4"

RX_CNJ = re.compile(r"\d{7}-\d{2}\.\d{4}\.(\d)\.\d{2}\.\d{4}")
SIGLAS = ["AgR-REspe", "AgR-AI", "AgR-RO", "AgR-DREspe", "AgRg", "AIRR", "E-RR", "EDcl",
          "REspe", "ROEl", "RCED", "RMS", "AgR", "AC", "AI", "AR", "CC", "CTA", "ED",
          "HC", "RHC", "MS", "PA", "PC", "PP", "Pet", "RO", "RP", "RR", "Rcl", "RE"]
RX_CLASSE_NUM = re.compile(r"(?:^|[\s.\-])(" + "|".join(re.escape(s) for s in sorted(SIGLAS, key=len, reverse=True)) + r")[\s.\-]*\d{2,}")
RX_TIPO = re.compile(r"despacho|decisao|voto|acordao|nega seguimento|provimento|efeito suspensivo|(?<!pre)liminar|minuta", re.I)
RX_MANTER = re.compile(r"^(indice|lista|pauta|listas)\b|lista.{0,12}publicacao|gabarito|prova objetiva|prova oral|edital|resultado final|homologacao|concurso publico|delegacoes notariais", re.I)
RX_CURSO_REAL = re.compile(r"certificado|apostila|aula|slides|curso online|gabarito", re.I)
TIPO_PECA = [
    ("Despachos e votos", re.compile(r"despacho|decisao|voto|relatorio e voto", re.I)),
    ("Acordaos e decisoes", re.compile(r"acordao|sentenca", re.I)),
    ("Contestacoes e manifestacoes", re.compile(r"contestacao|manifestacao|contrarrazoes|contraminuta|impugnacao|defesa|razoes finais|Cr Ed", re.I)),
    ("Pareceres e notas tecnicas", re.compile(r"parecer|nota juridica|nota tecnica", re.I)),
    ("Peticoes", re.compile(r"peticao|\bpet\b", re.I)),
    ("Recursos", re.compile(r"recurso especial|recurso ordinario|agravo|embargos|apelacao|\bREspe\b|\bAgR\b|\bAIRR\b|\bRR\b", re.I)),
    ("Outros processuais", re.compile(r"habeas corpus|\bHC\b|mandado de seguranca|cautelar|cumprimento de sentenca|execucao|lista triplice|\bRVE\b|\bLT\b|forca federal|\bPA\b|\bPC\b", re.I)),
]
TIPOS_01 = [t for t, _ in TIPO_PECA]
MATERIAS = ["Eleitoral", "Trabalhista", "Civel", "Penal", "Administrativo", "Tributario",
            "Constitucional", "Previdenciario", "Outros"]
J_MATERIA = {"6": "Eleitoral", "5": "Trabalhista", "8": "Civel", "4": "Civel",
             "2": "Administrativo", "1": "Constitucional", "3": "Outros"}

def materia_de(stem, rel_origem):
    m = RX_CNJ.search(stem)
    if m and m.group(1) in J_MATERIA:
        return J_MATERIA[m.group(1)]
    lw = stem.lower()
    if re.search(r"eleicoes|eleitoral|rve|tse|candidat", lw):
        return "Eleitoral"
    if re.search(r"\bect\b|reclamante|trabalhist|correios", lw):
        return "Trabalhista"
    if "Direito Eleitoral" in rel_origem:
        return "Eleitoral"
    if "Direito do Trabalho" in rel_origem:
        return "Trabalhista"
    return "Outros"

# ---------------- enderecamento fino dentro de 01\Tipo\Materia ----------------
RX_FAIXA_ANO = re.compile(r"^(\d{4})(?:-(\d{4}))?$")

def enderecar(tipo, materia, nome):
    """Retorna caminho de pasta destino, encaixando nas subpastas existentes."""
    base = os.path.join(ROOT, "01 - Juridico", tipo, materia)
    if not os.path.isdir(base):
        base = os.path.join(ROOT, "01 - Juridico", tipo, "Outros")
        if not os.path.isdir(base):
            return os.path.join(ROOT, "01 - Juridico", tipo)
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subs:
        return base
    # grupos de classe? (rotulos como "REspe - Recurso especial")
    grupos = {}
    for d in subs:
        pref = d.split(" - ")[0]
        grupos[pref] = d
    c = classe(nome)
    mapa = GRUPO_ELEITORAL if materia == "Eleitoral" else GRUPO_TRABALHISTA
    grupo_rotulo = None
    if c:
        alvo = (GRUPO_ACORDAOS if tipo == "Acordaos e decisoes" else mapa).get(c)
        if alvo and alvo in subs:
            grupo_rotulo = alvo
        elif alvo and alvo.split(" - ")[0] in grupos:
            grupo_rotulo = grupos[alvo.split(" - ")[0]]
    if grupo_rotulo is None:
        for cand in ("Outras classes", "Outras classes e diversos", "Outros"):
            if cand in subs:
                grupo_rotulo = cand
                break
    nivel1 = os.path.join(base, grupo_rotulo) if grupo_rotulo else base
    # faixa de ano dentro do nivel atual?
    if os.path.isdir(nivel1):
        subs2 = [d for d in os.listdir(nivel1) if os.path.isdir(os.path.join(nivel1, d))]
        faixas = [d for d in subs2 if RX_FAIXA_ANO.match(d) or d.startswith(("Sem ano",)) or re.match(r"^\d{4} - ", d)]
        if faixas:
            ano = ano_do_nome(nome)
            if ano:
                for d in faixas:
                    m = RX_FAIXA_ANO.match(d)
                    if m and int(m.group(1)) <= ano <= int(m.group(2) or m.group(1)):
                        return os.path.join(nivel1, d)
                return os.path.join(nivel1, str(ano))  # cria pasta do ano
            if "Sem ano" in subs2:
                return os.path.join(nivel1, "Sem ano")
    return nivel1

SYSTEM = """Voce revisa a organizacao do acervo de um servidor do TSE (ex-advogado dos Correios).
Cada item esta HOJE numa pasta de estudos/administrativa/pessoal, mas o nome sugere processo judicial.
Decida pelo TRECHO DO CONTEUDO:
- Se e PECA/EXPEDIENTE PROCESSUAL REAL (minuta de decisao, despacho, voto, peticao, manifestacao, recurso, parecer em processo concreto): mover para o Juridico -> informe tipo e materia.
- Se e material de ESTUDO, edital/resultado de concurso, documento administrativo ou pessoal que apenas CITA um processo: manter.
tipos validos: {TIPOS}
materias validas: {MATERIAS}
Responda APENAS JSON: {{"itens": [{{"id": <n>, "acao": "mover"|"manter", "tipo": "<tipo|null>", "materia": "<materia|null>", "confianca": <0-1>}}]}}"""

def chave_openai():
    for linha in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
        if linha.startswith("OPENAI_API_KEY="):
            return linha.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("sem chave")

def ia_decidir(pendentes):
    from openai import OpenAI
    client = OpenAI(api_key=chave_openai())
    system = SYSTEM.replace("{TIPOS}", ", ".join(TIPOS_01)).replace("{MATERIAS}", ", ".join(MATERIAS))
    def texto(p):
        t = extrair_completo(p) or ""
        return re.sub(r"\s+", " ", t)[:800]
    textos = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for p, t in zip(pendentes, ex.map(texto, pendentes)):
            textos[p] = t
    lotes = [pendentes[i:i + 10] for i in range(0, len(pendentes), 10)]
    def um_lote(lote):
        linhas = [f'id={i} | pasta="{os.path.dirname(os.path.relpath(p, ROOT))}" | nome="{os.path.basename(p)}" | trecho="{textos.get(p, "")}"'
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
                        out.append({"path": lote[i], "acao": it.get("acao", "manter"),
                                    "tipo": it.get("tipo"), "materia": it.get("materia"),
                                    "confianca": float(it.get("confianca", 0))})
                return out
            except Exception as e:
                if tent == 3:
                    print("FALHA lote:", e)
                    return []
                time.sleep(2 * (tent + 1))
    with ThreadPoolExecutor(max_workers=6) as ex, io.open(RESP, "a", encoding="utf-8") as f:
        for res in ex.map(um_lote, lotes):
            for item in res or []:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    execu = "--exec" in sys.argv
    manter, regra, estudo, duvida = [], [], [], []
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        if rel.startswith("01 -") or "Julgados e sumulas" in rel:
            continue
        em_cursos = "Cursos e certificados" in rel
        for f in fn:
            if f.startswith("00 - "):
                continue
            stem = os.path.splitext(f)[0]
            p = os.path.join(dp, f)
            cnj = bool(RX_CNJ.search(stem))
            cls = bool(RX_CLASSE_NUM.search(stem))
            tipo_ind = bool(RX_TIPO.search(stem))
            if em_cursos and cls and not cnj and not RX_CURSO_REAL.search(stem):
                estudo.append(p)  # julgado de estudo preso no bug curso<recurso
                continue
            if not (cnj or (cls and tipo_ind)):
                continue
            if RX_MANTER.search(stem):
                manter.append(p)
                continue
            t = next((t for t, rx in TIPO_PECA if rx.search(stem)), None)
            if t:
                regra.append((p, t))
            else:
                duvida.append(p)
    print(f"manter: {len(manter)} | mover por regra: {len(regra)} | julgado estudo (bug curso): {len(estudo)} | IA: {len(duvida)}")

    cache = {}
    if os.path.exists(RESP):
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass
    pendentes = [p for p in duvida if p not in cache]
    if pendentes and "--so-plano" not in sys.argv:
        print(f"IA decidindo {len(pendentes)}...")
        ia_decidir(pendentes)
        cache = {}
        for linha in io.open(RESP, encoding="utf-8"):
            try:
                it = json.loads(linha)
                cache[it["path"]] = it
            except Exception:
                pass

    plano = []
    stats = Counter()
    for p, t in regra:
        rel = os.path.relpath(os.path.dirname(p), ROOT)
        mat = materia_de(os.path.splitext(os.path.basename(p))[0], rel)
        dest = enderecar(t, mat, os.path.basename(p))
        plano.append((p, os.path.join(dest, os.path.basename(p))))
        stats[f"regra -> {t}\\{mat}"] += 1
    for p in estudo:
        rel = os.path.relpath(os.path.dirname(p), ROOT)
        disciplina = rel.split(os.sep)[1] if len(rel.split(os.sep)) > 1 else ""
        base_j = os.path.join(ROOT, "02 - Estudos e concursos", disciplina, "Julgados e sumulas")
        nome = os.path.basename(p)
        if os.path.isdir(base_j):
            subs = [d for d in os.listdir(base_j) if os.path.isdir(os.path.join(base_j, d))]
            c = classe(nome)
            alvo = GRUPO_ELEITORAL.get(c, "Outras classes") if subs else None
            dest = os.path.join(base_j, alvo) if alvo and alvo in subs else base_j
        else:
            dest = base_j
        plano.append((p, os.path.join(dest, nome)))
        stats["estudo -> Julgados e sumulas"] += 1
    for p in duvida:
        it = cache.get(p)
        if not it or it["acao"] != "mover" or it["confianca"] < 0.6:
            stats["IA: manter"] += 1
            continue
        t = it.get("tipo") if it.get("tipo") in TIPOS_01 else "Outros processuais"
        mat = it.get("materia") if it.get("materia") in MATERIAS else None
        # a justica do CNJ e autoritativa sobre a materia sugerida pela IA
        m_cnj = RX_CNJ.search(os.path.splitext(os.path.basename(p))[0])
        if m_cnj and m_cnj.group(1) in J_MATERIA:
            mat = J_MATERIA[m_cnj.group(1)]
        if not mat:
            mat = materia_de(os.path.splitext(os.path.basename(p))[0], os.path.relpath(os.path.dirname(p), ROOT))
        dest = enderecar(t, mat, os.path.basename(p))
        plano.append((p, os.path.join(dest, os.path.basename(p))))
        stats[f"IA -> {t}\\{mat}"] += 1

    print("\nPlano:", len(plano), "movimentacoes")
    for k, v in stats.most_common(20):
        print(f"  {v:4d}  {k}")
    with open(PLANO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        w.writerows(plano)
    if not execu:
        print("Dry-run. Rode com --exec para mover.")
        return
    ok = 0
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["de", "para"])
        for de, para in plano:
            try:
                os.makedirs(os.path.dirname(para), exist_ok=True)
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
                print("ERRO:", de, e)
    print(f"Movidos: {ok}. Log: {LOG}")

if __name__ == "__main__":
    main()
