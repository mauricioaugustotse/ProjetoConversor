import csv
import os
import glob
import re
from typing import List, Set


# =========================================================
# 1. EXTRAÇÃO E NORMALIZAÇÃO DE RAMO DO DIREITO
# =========================================================

def extrair_ramos(texto: str) -> str:
    """
    Identifica o(s) ramo(s) do direito mencionados na ementa e
    devolve tags canônicas, separadas por vírgula, sem repetição.
    """

    patterns = [
        # Processual
        (r'\bPROCESSUAL\s+PENAL\b', 'DIREITO PROCESSUAL PENAL'),
        (r'\bPROCESSUAL\s+CIVIL\b', 'DIREITO PROCESSUAL CIVIL'),

        # Penal / Civil
        (r'\bPENAL\b', 'DIREITO PENAL'),
        (r'\bCIVIL\b', 'DIREITO CIVIL'),

        # Trabalho
        (r'\bTRABALHIST\w*', 'DIREITO DO TRABALHO'),
        (r'\bTRABALHO\b', 'DIREITO DO TRABALHO'),

        # Constitucional (inclui "constitucionalidade", "inconstitucionalidade")
        (r'\bCONSTITUCIONAL', 'DIREITO CONSTITUCIONAL'),

        # Administrativo
        (r'\bADMINISTRATIV', 'DIREITO ADMINISTRATIVO'),
        (r'\bSERVIDOR(?:ES)?\s+P[ÚU]BLIC', 'DIREITO ADMINISTRATIVO'),
        (r'\bPOLICIAL\s+MILITAR\b', 'DIREITO ADMINISTRATIVO'),

        # Tributário
        (r'\bTRIBUT[ÁA]R', 'DIREITO TRIBUTÁRIO'),
        (r'\bIPTU\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bIPVA\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bITBI\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bITCMD\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bICMS\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bIRPF\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bIRPJ\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bCOFINS\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bPIS\b', 'DIREITO TRIBUTÁRIO'),
        (r'\bCSLL\b', 'DIREITO TRIBUTÁRIO'),

        # Previdenciário
        (r'\bPREVIDENCI[ÁA]R', 'DIREITO PREVIDENCIÁRIO'),

        # Empresarial / Comercial
        (r'\bEMPRESAR', 'DIREITO EMPRESARIAL'),
        (r'\bCOMERCIAL', 'DIREITO EMPRESARIAL'),

        # Eleitoral
        (r'\bELEITORAL', 'DIREITO ELEITORAL'),

        # Ambiental
        (r'\bAMBIENTAL', 'DIREITO AMBIENTAL'),

        # Consumidor
        (r'\bCONSUMIDOR', 'DIREITO DO CONSUMIDOR'),

        # Internacional
        (r'\bINTERNACIONAL', 'DIREITO INTERNACIONAL'),

        # Financeiro / Econômico / Urbanístico
        (r'\bFINANCEIR', 'DIREITO FINANCEIRO'),
        (r'\bECONÔMIC', 'DIREITO ECONÔMICO'),
        (r'\bECONOMIC', 'DIREITO ECONÔMICO'),
        (r'\bURBAN[ÍI]STIC', 'DIREITO URBANÍSTICO'),
    ]

    txt = (texto or "").upper()
    encontrados = set()

    for pat, canon in patterns:
        if re.search(pat, txt, flags=re.IGNORECASE):
            encontrados.add(canon)

    return ", ".join(sorted(encontrados))


# =========================================================
# 2. EXTRAÇÃO E NORMALIZAÇÃO DE TESE
# =========================================================

TESE_PATTERNS = [
    r'tese\s+de\s+julgamento',                         # "Tese de julgamento:"
    r'teses\s+de\s+julgamento',                        # "Teses de julgamento:"
    r'tese\s+de\s+julgamento\s+para\s+o\s+tema',       # "Tese de julgamento para o Tema 1101:"
    r'fixaç[aã]o\s+de\s+tese',                         # "fixação de tese:"
    r'fixaç[aã]o\s+de\s+teses',                        # "fixação de teses:"
    r'fixaç[aã]o\s+das?\s+seguinte[s]?\s+teses?',      # "Fixação das seguintes teses:"
    r'fixaç[aã]o\s+da\s+seguinte\s+tese',              # "fixação da seguinte tese:"
    r'fixaç[aã]o\s+da\s+seguinte\s+tese\s+de',         # variação com repercussão geral
]

regex_quoted_tail = re.compile(r'["“](.{10,400}?)["”]', flags=re.DOTALL)


def limpar_quotes(txt: str) -> str:
    """
    Remove aspas externas e pontuação solta.
    """
    t = (txt or "").strip()
    t = re.sub(r'^[\'"“”]+', '', t)
    t = re.sub(r'[\'"“”]+$', '', t)
    t = t.strip(" \t\r\n.;:")
    t = t.replace('"', "'").replace("“", "'").replace("”", "'")
    return t


def _is_mere_repercussao(texto: str) -> bool:
    """
    Descartar frases que só dizem que há repercussão geral reconhecida,
    sem formular regra normativa.
    """
    low = texto.lower().strip()
    if "repercuss" in low and "geral" in low and ("reconhec" in low or "existên" in low or "existenc" in low):
        # Se começar com "É ..." (ex.: "É constitucional que..."), é tese normativa -> mantém
        if re.match(r'^\s*é\s', low):
            return False
        return True
    return False


def extrair_tese(ementa: str) -> str:
    """
    Captura a tese de julgamento (tese fixada).
    1) Padrões explícitos (tese de julgamento / fixação de tese / fixação das teses)
       seguidos de : ou - e aspas
    2) Última citação longa entre aspas nos 1200 últimos caracteres
    3) Frase que começa com "É ..." no final da ementa
    Ignora trechos que apenas falam em reconhecimento de repercussão geral.
    """
    if not ementa:
        return ""

    texto = ementa.strip()

    # 1) padrões explícitos seguidos de aspas
    candidatos = []
    for base in TESE_PATTERNS:
        rgx = re.compile(
            r'(?:' + base + r')\s*[:\-–]\s*(?P<quote>["“][^"”]+["”])',
            flags=re.IGNORECASE | re.DOTALL
        )
        m = rgx.search(texto)
        if m:
            cand = limpar_quotes(m.group("quote"))
            if cand and not _is_mere_repercussao(cand):
                candidatos.append(cand)

    if candidatos:
        return candidatos[0]

    # 2) fallback: aspas longas no final
    ultimos = texto[-1200:]
    quoted_blocks = regex_quoted_tail.findall(ultimos)
    for qb in reversed(quoted_blocks):
        cand = limpar_quotes(qb)
        if cand and not _is_mere_repercussao(cand):
            return cand

    # 3) fallback: frase que começa com "É ..."
    m2 = re.search(
        r'(?:^|[\.\n]\s*)(É\s+[^\.\n]{20,400}?)(?:[\.\n]|$)',
        ultimos,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m2:
        tentativa = m2.group(1).strip()
        tentativa = tentativa.strip(" .;:\n\t\"“”'")
        if tentativa and not _is_mere_repercussao(tentativa):
            return tentativa

    return ""


# =========================================================
# 3. EXTRAÇÃO E NORMALIZAÇÃO DE LEGISLAÇÃO
# =========================================================
#
# Formato final de cada TAG:
#   "{DIPLOMA} - art. X; §1º; inc. I; a"
#
# - Diploma SEM ponto final no fim (ex.: "Lei nº 8.213/91." -> "Lei nº 8.213/91")
# - Se art. está entre 1 e 9 e não tem "º", acrescentar "º":
#     "art. 3" -> "art. 3º"
# - Se parágrafo está entre 1 e 9 e não tem "º", acrescentar "º":
#     "§ 1" -> "§1º"
# - Dentro da MESMA TAG usamos "; " entre pedaços.
# - TAGS diferentes seguem separadas por ", " no CSV final.


def pre_normalizar_leg_text(t: str) -> str:
    """
    Limpeza preliminar:
    - "art.IGOS"/"art.IGO"/"artigo(s)" -> "art."
    - ", e 155" -> ", art. 155"
    - 'alínea “A”' -> 'alínea a'
    - "Lei 6.938/1981 art. 14 e § 1º"
      -> "Lei 6.938/1981 art. 14, § 1º"
    """
    if not t:
        return ""

    s = t

    # "art.IGOS" / "art.IGO" / "artigo(s)" -> "art."
    s = re.sub(r'art\.?\s*(?:igo|igos)\b', 'art.', s, flags=re.IGNORECASE)
    s = re.sub(r'\bartigos?\b', 'art.', s, flags=re.IGNORECASE)

    # ", e 155" -> ", art. 155"
    s = re.sub(r'([,;])\s*[eE]\s*(\d+)', r'\1 art. \2', s)

    # alínea “A” -> alínea a
    s = re.sub(
        r'al[ií]nea\s*[\"“]?([A-Za-z])[\"”]?',
        lambda m: f"alínea {m.group(1).lower()}",
        s,
        flags=re.IGNORECASE
    )

    # "Lei 6.938/1981 art. 14 e § 1º" -> "Lei 6.938/1981 art. 14, § 1º"
    s = re.sub(
        r'(Lei\s*(?:n[º°.]*\s*)?\d[\d\./\-]*\s*[/\d]*)\s*(art\.?\s*\d+[A-Za-z0-9\-]*)\s+e\s+§',
        r'\1 \2, §',
        s,
        flags=re.IGNORECASE
    )

    return s


def normalize_diploma(raw: str) -> str:
    """
    Normaliza o diploma em forma curta estável:
    CF/88
    CPC/2015 / CPC/1973 / CPC
    CPP
    CP
    CLT
    CTN
    CC
    Lei nº XXXX/AAAA

    Remove pontuação final como "." ";" ":" após o diploma.
    """
    txt = (raw or "").strip()
    txt = txt.rstrip(" .;:")

    lower = txt.lower()

    # Constituição Federal
    if 'constitui' in lower or 'cf/88' in lower or re.match(r'^cf\b', lower):
        return 'CF/88'

    # CPC / Código de Processo Civil
    if 'processo civil' in lower or re.match(r'^cpc', lower):
        my = re.search(r'20?15|1973', txt)
        if my:
            return f"CPC/{my.group(0)}"
        return "CPC"

    # Código de Processo Penal / CPP
    if 'processo penal' in lower or re.match(r'^cpp\b', lower):
        return 'CPP'

    # Código Penal / CP
    if re.match(r'^cp\b', lower) or 'código penal' in lower:
        return 'CP'

    # CLT
    if 'clt' in lower:
        return 'CLT'

    # CTN
    if 'ctn' in lower or ('tribut' in lower and 'nacional' in lower):
        return 'CTN'

    # Código Civil
    if 'código civil' in lower or re.match(r'\bcc\b', lower):
        return 'CC'

    # Lei nº XXXXX/AAAA
    if lower.startswith('lei'):
        m = re.search(r'(\d[\d\./\-]*)', txt)
        if m:
            numero = m.group(1).rstrip(" .;:")
            return f"Lei nº {numero}"
        return "Lei nº ?"

    # fallback
    return txt


def _normalizar_artigo_ordinal(artigo_raw: str) -> str:
    """
    Regras para artigo:
    - Se artigo está entre 1 e 9 e não tem 'º', acrescentar 'º'.
      "1" -> "1º"
      "3" -> "3º"
    - Se já vier "3º" ou "8º", mantém.
    - Corrige OCR '3o', '3°' -> '3º'.
    - Se >=10, mantém como está.
    """
    a = artigo_raw.strip()

    # corrigir "3o", "3°" -> "3º"
    m_sub = re.match(r'^([1-9])[o°]$', a, flags=re.IGNORECASE)
    if m_sub:
        return m_sub.group(1) + "º"

    # já tem "º" após dígito único?
    if re.match(r'^[1-9]º$', a):
        return a

    # dígito único 1-9 sem "º"?
    if re.match(r'^[1-9]$', a):
        return a + "º"

    # senão, mantém (10, 24, 62, 62-A etc.)
    return a


def _normalizar_paragrafo_ordinal(par_raw: str) -> str:
    """
    Regras para parágrafo (§):
    - Se parágrafo está entre 1 e 9 e não tem 'º', acrescentar 'º'.
      "1" -> "§1º"
      "2" -> "§2º"
    - Se já vier '1º', mantém.
    - Corrige "1o", "1°" -> "1º".
    - Se >=10, não adiciona 'º'.
    Retorna SEM o símbolo § na frente; quem chama recoloca '§'.
    """
    p = par_raw.strip()

    # corrigir "1o", "1°" → "1º"
    m_sub = re.match(r'^([1-9])[o°]$', p, flags=re.IGNORECASE)
    if m_sub:
        return m_sub.group(1) + "º"

    # já tem padrão "1º", "2º", etc.?
    if re.match(r'^[1-9]º$', p):
        return p

    # dígito único 1-9 sem "º"?
    if re.match(r'^[1-9]$', p):
        return p + "º"

    # 10, 11, etc.
    return p


def parse_artblock_single(part: str, diploma_norm: str) -> str:
    """
    Recebe subtrecho como:
        "art. 62 § 1º, I, i"
        "art. 487, III, b"
        "art. 3, § 1, VII"
    e devolve tag única:
        "{DIPLOMA} - art. 62; §1º; inc. I; i"
        "{DIPLOMA} - art. 487; inc. III; b"
        "{DIPLOMA} - art. 3º; §1º; inc. VII"
    """

    if not part:
        return ""

    p = part
    p = re.sub(r'\s+', ' ', p)
    p = p.strip(" .;:),(")

    # artigo
    mart = re.search(r'art\.?\s*([0-9]+[A-Za-z0-9\-ºo°]*)', p, flags=re.IGNORECASE)
    if not mart:
        return ""
    artigo_raw = mart.group(1).strip()
    artigo_norm = _normalizar_artigo_ordinal(artigo_raw)

    # parágrafo
    mpar = re.search(r'§\s*([0-9]+[ºo°]?)', p)
    paragrafo_fmt = None
    if mpar:
        par_raw = mpar.group(1).strip()
        # normaliza '1' -> '1º', '1o' -> '1º', etc.
        par_norm = _normalizar_paragrafo_ordinal(par_raw)
        paragrafo_fmt = f"§{par_norm}"  # sem espaço

    # inciso ("inciso III", "inc. III" ou apenas ", III")
    inciso_val = None
    minc = re.search(r'inc(?:\.|iso)?\s*([IVXLCDM]+)', p, flags=re.IGNORECASE)
    if minc:
        inciso_val = minc.group(1).upper()
    else:
        # romano solto depois de vírgula
        mrom = re.search(r',\s*([IVXLCDM]{1,10})(?:\b|,)', p)
        if mrom:
            inciso_val = mrom.group(1).upper()

    # alínea ("alínea a" ou vírgula + letra minúscula)
    alineia_val = None
    mali = re.search(r'al[ií]nea\s*([a-zA-Z])', p, flags=re.IGNORECASE)
    if mali:
        alineia_val = mali.group(1).lower()
    else:
        # pega possíveis letras soltas, mas ignora romanos
        mlet_all = re.findall(r',\s*([a-zA-Z])(?:\b|,)', p)
        for cand in mlet_all:
            if not re.match(r'^[IVXLCDM]+$', cand, flags=re.IGNORECASE):
                alineia_val = cand.lower()

    # montar TAG final:
    # base: "{DIPLOMA} - art. {artigo_norm}"
    parts_internos = [f"{diploma_norm} - art. {artigo_norm}"]
    if paragrafo_fmt:
        parts_internos.append(paragrafo_fmt)
    if inciso_val:
        parts_internos.append(f"inc. {inciso_val}")
    if alineia_val:
        parts_internos.append(alineia_val)

    tag = "; ".join(parts_internos)
    return tag.strip()


def split_artblocks(artblock: str) -> List[str]:
    """
    Divide bloco com múltiplos artigos do MESMO diploma,
    ex.: "art. 146 § 3º, III, a, art. 155 § 1º"
    -> ["art. 146 § 3º, III, a", "art. 155 § 1º"]
    """
    if not artblock:
        return []

    tmp = re.sub(r'\s+', ' ', artblock)
    parts = re.split(r'(?=(?:^|[\s,;])art\.?\s*\d)', tmp, flags=re.IGNORECASE)
    parts = [p.strip(" ,;") for p in parts if re.search(r'art\.?\s*\d', p, flags=re.IGNORECASE)]
    return parts


def extrair_legislacao(ementa: str) -> str:
    """
    Extrai referências legislativas e gera TAGS prontas pro Notion.

    Cada TAG segue:
      "{DIPLOMA} - art. X; §1º; inc. I; a"

    As TAGS diferentes são separadas por ", " na célula final.
    """
    if not ementa:
        return ""

    text = pre_normalizar_leg_text(ementa)

    # Diplomas possíveis
    DIPL_RE = (
        r'(?:CF/?\s*88|Constituiç[aã]o\s*Federal(?:\s*de\s*1988)?|CF)'
        r'|(?:CPC\s*/?\s*(?:2015|1973)?|C[oó]digo\s+de\s+Processo\s+Civil)'
        r'|(?:CPP|C[oó]digo\s+de\s+Processo\s+Penal)'
        r'|(?:CP|C[oó]digo\s+Penal)'
        r'|(?:CLT)'
        r'|(?:CTN|C[oó]digo\s+Tribut[áa]rio\s+Nacional)'
        r'|(?:C[oó]digo\s+Civil|CC\b)'
        r'|(?:Lei\s*(?:n[º°.]*\s*)?\d[\d\./\-]*\s*[/\d]*)'
    )

    # Suporta:
    # (A) artblock ... da|do|de DIPLOMA
    # (B) DIPLOMA , artblock
    # (C) artblock , DIPLOMA
    pattern_combo = re.compile(
        r'(?P<artblock1>art\.?\s*\d+[^\.;\)]{0,120}?)\s*(?:da|do|de)\s*(?P<diploma1>' + DIPL_RE + r')'
        r'|'
        r'(?P<diploma2>' + DIPL_RE + r')\s*,?\s*(?P<artblock2>art\.?\s*\d+[^\.;\)]{0,120})'
        r'|'
        r'(?P<artblock3>art\.?\s*\d+[^\.;\)]{0,120}?)\s*,\s*(?P<diploma3>' + DIPL_RE + r')',
        flags=re.IGNORECASE
    )

    tags_ordered: List[str] = []
    seen = set()

    for m in pattern_combo.finditer(text):
        if m.group("artblock1") and m.group("diploma1"):
            artblock_raw = m.group("artblock1")
            diploma_raw = m.group("diploma1")
        elif m.group("diploma2") and m.group("artblock2"):
            artblock_raw = m.group("artblock2")
            diploma_raw = m.group("diploma2")
        else:
            artblock_raw = m.group("artblock3")
            diploma_raw = m.group("diploma3")

        diploma_norm = normalize_diploma(diploma_raw)

        for subblock in split_artblocks(artblock_raw):
            tag = parse_artblock_single(subblock, diploma_norm)
            if tag:
                low = tag.lower()
                if low not in seen:
                    tags_ordered.append(tag)
                    seen.add(low)

    return ", ".join(tags_ordered)


# =========================================================
# 4. EXTRAÇÃO E NORMALIZAÇÃO DE JURISPRUDÊNCIA
# =========================================================

def normalizar_jur_item(item: str) -> str:
    """
    Normaliza referências jurisprudenciais p/ tags:
    "STF Tema 985", "Tema 1101", "Súmula 123"
    """
    if not item:
        return ""

    txt = item.strip()
    txt = re.sub(r'\s+', ' ', txt)
    txt = txt.strip('.,;:- ')
    txt = re.sub(r'^\s*STF\s*-\s*', 'STF ', txt, flags=re.IGNORECASE)

    mtema = re.search(r'^(?:STF\s+)?Tema\s*(?:n[º°.]*)?\s*(\d+)', txt, flags=re.IGNORECASE)
    if mtema:
        num = mtema.group(1)
        if re.search(r'^STF', txt, flags=re.IGNORECASE):
            return f"STF Tema {num}"
        else:
            return f"Tema {num}"

    msum = re.search(r'^S[úu]mula\s*(?:n[º°.]*)?\s*(\d+)', txt, flags=re.IGNORECASE)
    if msum:
        num = msum.group(1)
        return f"Súmula {num}"

    return txt


def extrair_jurisprudencia(ementa: str) -> str:
    """
    Captura "STF - Tema 985", "Tema 1101", "Tema nº 833", "Súmula 123".
    Deduplica e separa por vírgula e espaço.
    """
    if not ementa:
        return ""

    regex_jur = re.compile(
        r'('
        r'(?:STF\s*-\s*)?Tema\s*(?:n[.ºº]?)?\s*\d+'
        r'|S[úu]mula\s*(?:n[.ºº]?)?\s*\d+'
        r')',
        re.IGNORECASE
    )

    matches_jur = regex_jur.findall(ementa)

    seen_jur = set()
    cleaned_matches_jur = []
    for raw in matches_jur:
        cleaned = normalizar_jur_item(raw)
        if cleaned:
            low = cleaned.lower()
            if low not in seen_jur:
                cleaned_matches_jur.append(cleaned)
                seen_jur.add(low)

    return ", ".join(cleaned_matches_jur)


# =========================================================
# 5. RESULTADO
# =========================================================

def extrair_resultado(ementa: str, tese: str) -> str:
    """
    "Infraconstitucional" se constar "é infraconstitucional".
    Senão, se existe tese extraída -> "Tese fixada".
    Caso contrário -> "Aguardando julgamento".
    """
    ementa_local = ementa or ""

    if re.search(r'\bé\s+infraconstitucional', ementa_local, flags=re.IGNORECASE):
        return "Infraconstitucional"

    if tese.strip():
        return "Tese fixada"

    return "Aguardando julgamento"


# =========================================================
# 6. SUPORTE CSV / PROGRESSO / MAIN
# =========================================================

def detectar_csv_entrada() -> str:
    """
    Procura automaticamente um CSV na pasta atual.
    Ignora arquivos que já parecem ser saída (com 'processado' no nome).
    Retorna o primeiro em ordem alfabética.
    """
    candidates = [f for f in glob.glob("*.csv") if "processado" not in f.lower()]
    if not candidates:
        raise FileNotFoundError(
            "Nenhum arquivo .csv de entrada encontrado na pasta atual. "
            "Coloque o CSV na mesma pasta do script."
        )
    candidates.sort()
    return candidates[0]


def localizar_coluna_ementa(header: List[str]) -> int:
    """
    Tenta localizar a coluna 'Ementa' pelo nome.
    Se não achar, assume a coluna G (índice 6).
    Se não der, usa a última coluna.
    """
    for idx, col in enumerate(header):
        if col.strip().lower() == "ementa":
            return idx

    if len(header) > 6:
        return 6

    return len(header) - 1


def processar_csv(input_file: str, output_file: str, verbose: bool = True) -> None:
    """
    Lê o CSV de entrada, extrai:
      - Tese
      - Legislação
      - Jurisprudência
      - Ramo do direito
      - Resultado
    e grava um novo CSV com essas colunas extras.
    """
    if verbose:
        print(f"[INFO] Arquivo de entrada detectado: {input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_file}")

    with open(input_file, mode='r', encoding='utf-8', newline='') as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError("CSV vazio - nenhum cabeçalho encontrado.")

        rows = list(reader)

    ementa_idx = localizar_coluna_ementa(header)
    if verbose:
        print(f"[INFO] Coluna 'Ementa' identificada no índice {ementa_idx} ('{header[ementa_idx]}').")

    new_header = header + ['Tese', 'Legislação', 'Jurisprudência', 'Ramo do direito', 'Resultado']

    total = len(rows)
    if verbose:
        print(f"[INFO] Total de linhas de dados: {total}")

    processed_rows: List[List[str]] = []
    last_pct = -1

    for i, row in enumerate(rows, start=1):
        ementa_text = row[ementa_idx] if len(row) > ementa_idx else ""

        tese = extrair_tese(ementa_text)
        legislacao = extrair_legislacao(ementa_text)
        jurisprudencia = extrair_jurisprudencia(ementa_text)
        ramo = extrair_ramos(ementa_text)
        resultado = extrair_resultado(ementa_text, tese)

        processed_rows.append(row + [tese, legislacao, jurisprudencia, ramo, resultado])

        if verbose and total > 0:
            pct = int((i / total) * 100)
            if pct // 5 != last_pct // 5:
                print(f"[PROGRESS] Processando linha {i}/{total} ({pct}%)...")
            last_pct = pct

    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        writer.writerows(processed_rows)

    if verbose:
        print(f"[SUCESSO] {total} linhas processadas.")
        print(f"[SUCESSO] Arquivo de saída criado: {output_file}")


def main():
    try:
        input_filename = detectar_csv_entrada()

        base, ext = os.path.splitext(input_filename)
        if ext.lower() == ".csv":
            output_filename = f"{base}_processado.csv"
        else:
            output_filename = f"{base}_processado{ext}"

        processar_csv(input_filename, output_filename, verbose=True)

    except Exception as e:
        print("[ERRO] Ocorreu um problema durante o processamento:")
        print(e)


if __name__ == "__main__":
    main()