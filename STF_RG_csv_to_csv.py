"""
Consolida e normaliza CSVs de repercussão geral do STF.

Fluxo de trabalho:
1. Seleciona arquivos CSV (GUI) e define o arquivo consolidado de saída.
2. Para cada linha, extrai/normaliza ramo, tese, legislação, jurisprudência e resultado.
3. Padroniza datas e limpeza textual para reduzir variações de escrita.
4. Garante esquema final de colunas e alinha registros de entradas diferentes.
5. Escreve um CSV consolidado pronto para uso analítico.
"""

import csv
import os
import re
import unicodedata
from typing import Dict, List

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None


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
        (r"\bPROCESSUAL\s+PENAL\b", "DIREITO PROCESSUAL PENAL"),
        (r"\bPROCESSUAL\s+CIVIL\b", "DIREITO PROCESSUAL CIVIL"),
        # Penal / Civil
        (r"\bPENAL\b", "DIREITO PENAL"),
        (r"\bCIVIL\b", "DIREITO CIVIL"),
        # Trabalho
        (r"\bTRABALHIST\w*", "DIREITO DO TRABALHO"),
        (r"\bTRABALHO\b", "DIREITO DO TRABALHO"),
        # Constitucional
        (r"\bCONSTITUCIONAL", "DIREITO CONSTITUCIONAL"),
        # Administrativo
        (r"\bADMINISTRATIV", "DIREITO ADMINISTRATIVO"),
        (r"\bSERVIDOR(?:ES)?\s+P[ÚU]BLIC", "DIREITO ADMINISTRATIVO"),
        (r"\bPOLICIAL\s+MILITAR\b", "DIREITO ADMINISTRATIVO"),
        # Tributário
        (r"\bTRIBUT[ÁA]R", "DIREITO TRIBUTÁRIO"),
        (r"\bIPTU\b", "DIREITO TRIBUTÁRIO"),
        (r"\bIPVA\b", "DIREITO TRIBUTÁRIO"),
        (r"\bITBI\b", "DIREITO TRIBUTÁRIO"),
        (r"\bITCMD\b", "DIREITO TRIBUTÁRIO"),
        (r"\bICMS\b", "DIREITO TRIBUTÁRIO"),
        (r"\bIRPF\b", "DIREITO TRIBUTÁRIO"),
        (r"\bIRPJ\b", "DIREITO TRIBUTÁRIO"),
        (r"\bCOFINS\b", "DIREITO TRIBUTÁRIO"),
        (r"\bPIS\b", "DIREITO TRIBUTÁRIO"),
        (r"\bCSLL\b", "DIREITO TRIBUTÁRIO"),
        # Previdenciário
        (r"\bPREVIDENCI[ÁA]R", "DIREITO PREVIDENCIÁRIO"),
        # Empresarial / Comercial
        (r"\bEMPRESAR", "DIREITO EMPRESARIAL"),
        (r"\bCOMERCIAL", "DIREITO EMPRESARIAL"),
        # Eleitoral
        (r"\bELEITORAL", "DIREITO ELEITORAL"),
        # Ambiental
        (r"\bAMBIENTAL", "DIREITO AMBIENTAL"),
        # Consumidor
        (r"\bCONSUMIDOR", "DIREITO DO CONSUMIDOR"),
        # Internacional
        (r"\bINTERNACIONAL", "DIREITO INTERNACIONAL"),
        # Financeiro / Econômico / Urbanístico
        (r"\bFINANCEIR", "DIREITO FINANCEIRO"),
        (r"\bECONÔMIC", "DIREITO ECONÔMICO"),
        (r"\bECONOMIC", "DIREITO ECONÔMICO"),
        (r"\bURBAN[ÍI]STIC", "DIREITO URBANÍSTICO"),
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
    r"tese\s+de\s+julgamento",
    r"teses\s+de\s+julgamento",
    r"tese\s+de\s+julgamento\s+para\s+o\s+tema",
    r"fixaç[aã]o\s+de\s+tese",
    r"fixaç[aã]o\s+de\s+teses",
    r"fixaç[aã]o\s+das?\s+seguinte[s]?\s+teses?",
    r"fixaç[aã]o\s+da\s+seguinte\s+tese",
    r"fixaç[aã]o\s+da\s+seguinte\s+tese\s+de",
]

regex_quoted_tail = re.compile(r"[\"“](.{10,400}?)[\"”]", flags=re.DOTALL)


def limpar_quotes(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"^[\'\"“”]+", "", t)
    t = re.sub(r"[\'\"“”]+$", "", t)
    t = t.strip(" \t\r\n.;:")
    t = t.replace('"', "'").replace("“", "'").replace("”", "'")
    return t


def _is_mere_repercussao(texto: str) -> bool:
    low = texto.lower().strip()
    if "repercuss" in low and "geral" in low and (
        "reconhec" in low or "existên" in low or "existenc" in low
    ):
        if re.match(r"^\s*é\s", low):
            return False
        return True
    return False


def extrair_tese(ementa: str) -> str:
    """
    Captura a tese de julgamento (tese fixada).
    """
    if not ementa:
        return ""

    texto = ementa.strip()

    candidatos = []
    for base in TESE_PATTERNS:
        rgx = re.compile(
            r"(?:" + base + r")\s*[:\-–]\s*(?P<quote>[\"“][^\"”]+[\"”])",
            flags=re.IGNORECASE | re.DOTALL,
        )
        m = rgx.search(texto)
        if m:
            cand = limpar_quotes(m.group("quote"))
            if cand and not _is_mere_repercussao(cand):
                candidatos.append(cand)

    if candidatos:
        return candidatos[0]

    ultimos = texto[-1200:]
    quoted_blocks = regex_quoted_tail.findall(ultimos)
    for qb in reversed(quoted_blocks):
        cand = limpar_quotes(qb)
        if cand and not _is_mere_repercussao(cand):
            return cand

    m2 = re.search(
        r"(?:^|[\.\n]\s*)(É\s+[^\.\n]{20,400}?)(?:[\.\n]|$)",
        ultimos,
        flags=re.IGNORECASE | re.DOTALL,
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


def pre_normalizar_leg_text(t: str) -> str:
    if not t:
        return ""

    s = t
    s = re.sub(r"art\.?\s*(?:igo|igos)\b", "art.", s, flags=re.IGNORECASE)
    s = re.sub(r"\bartigos?\b", "art.", s, flags=re.IGNORECASE)
    s = re.sub(r"([,;])\s*[eE]\s*(\d+)", r"\1 art. \2", s)
    s = re.sub(
        r"al[ií]nea\s*[\"“]?([A-Za-z])[\"”]?",
        lambda m: f"alínea {m.group(1).lower()}",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"(Lei\s*(?:n[º°.]*\s*)?\d[\d\./\-]*\s*[/\d]*)\s*(art\.?\s*\d+[A-Za-z0-9\-]*)\s+e\s+§",
        r"\1 \2, §",
        s,
        flags=re.IGNORECASE,
    )

    return s


def normalize_diploma(raw: str) -> str:
    txt = (raw or "").strip()
    txt = txt.rstrip(" .;:")

    lower = txt.lower()

    if "constitui" in lower or "cf/88" in lower or re.match(r"^cf\b", lower):
        return "CF/88"

    if "processo civil" in lower or re.match(r"^cpc", lower):
        my = re.search(r"20?15|1973", txt)
        if my:
            return f"CPC/{my.group(0)}"
        return "CPC"

    if "processo penal" in lower or re.match(r"^cpp\b", lower):
        return "CPP"

    if re.match(r"^cp\b", lower) or "código penal" in lower:
        return "CP"

    if "clt" in lower:
        return "CLT"

    if "ctn" in lower or ("tribut" in lower and "nacional" in lower):
        return "CTN"

    if "código civil" in lower or re.match(r"\bcc\b", lower):
        return "CC"

    if lower.startswith("lei"):
        m = re.search(r"(\d[\d\./\-]*)", txt)
        if m:
            numero = m.group(1).rstrip(" .;:")
            return f"Lei nº {numero}"
        return "Lei nº ?"

    return txt


def _normalizar_artigo_ordinal(artigo_raw: str) -> str:
    a = artigo_raw.strip()

    m_sub = re.match(r"^([1-9])[o°]$", a, flags=re.IGNORECASE)
    if m_sub:
        return m_sub.group(1) + "º"

    if re.match(r"^[1-9]º$", a):
        return a

    if re.match(r"^[1-9]$", a):
        return a + "º"

    return a


def _normalizar_paragrafo_ordinal(par_raw: str) -> str:
    p = par_raw.strip()

    m_sub = re.match(r"^([1-9])[o°]$", p, flags=re.IGNORECASE)
    if m_sub:
        return m_sub.group(1) + "º"

    if re.match(r"^[1-9]º$", p):
        return p

    if re.match(r"^[1-9]$", p):
        return p + "º"

    return p


def parse_artblock_single(part: str, diploma_norm: str) -> str:
    if not part:
        return ""

    p = part
    p = re.sub(r"\s+", " ", p)
    p = p.strip(" .;:),(")

    mart = re.search(r"art\.?\s*([0-9]+[A-Za-z0-9\-ºo°]*)", p, flags=re.IGNORECASE)
    if not mart:
        return ""
    artigo_raw = mart.group(1).strip()
    artigo_norm = _normalizar_artigo_ordinal(artigo_raw)

    mpar = re.search(r"§\s*([0-9]+[ºo°]?)", p)
    paragrafo_fmt = None
    if mpar:
        par_raw = mpar.group(1).strip()
        par_norm = _normalizar_paragrafo_ordinal(par_raw)
        paragrafo_fmt = f"§{par_norm}"

    inciso_val = None
    minc = re.search(r"inc(?:\.|iso)?\s*([IVXLCDM]+)", p, flags=re.IGNORECASE)
    if minc:
        inciso_val = minc.group(1).upper()
    else:
        mrom = re.search(r",\s*([IVXLCDM]{1,10})(?:\b|,)", p)
        if mrom:
            inciso_val = mrom.group(1).upper()

    alineia_val = None
    mali = re.search(r"al[ií]nea\s*([a-zA-Z])", p, flags=re.IGNORECASE)
    if mali:
        alineia_val = mali.group(1).lower()
    else:
        mlet_all = re.findall(r",\s*([a-zA-Z])(?:\b|,)", p)
        for cand in mlet_all:
            if not re.match(r"^[IVXLCDM]+$", cand, flags=re.IGNORECASE):
                alineia_val = cand.lower()

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
    if not artblock:
        return []

    tmp = re.sub(r"\s+", " ", artblock)
    parts = re.split(r"(?=(?:^|[\s,;])art\.?\s*\d)", tmp, flags=re.IGNORECASE)
    parts = [p.strip(" ,;") for p in parts if re.search(r"art\.?\s*\d", p, flags=re.IGNORECASE)]
    return parts


def extrair_legislacao(ementa: str) -> str:
    if not ementa:
        return ""

    text = pre_normalizar_leg_text(ementa)

    dipl_re = (
        r"(?:CF/?\s*88|Constituiç[aã]o\s*Federal(?:\s*de\s*1988)?|CF)"
        r"|(?:CPC\s*/?\s*(?:2015|1973)?|C[oó]digo\s+de\s+Processo\s+Civil)"
        r"|(?:CPP|C[oó]digo\s+de\s+Processo\s+Penal)"
        r"|(?:CP|C[oó]digo\s+Penal)"
        r"|(?:CLT)"
        r"|(?:CTN|C[oó]digo\s+Tribut[áa]rio\s+Nacional)"
        r"|(?:C[oó]digo\s+Civil|CC\b)"
        r"|(?:Lei\s*(?:n[º°.]*\s*)?\d[\d\./\-]*\s*[/\d]*)"
    )

    pattern_combo = re.compile(
        r"(?P<artblock1>art\.?\s*\d+[^\.;\)]{0,120}?)\s*(?:da|do|de)\s*(?P<diploma1>"
        + dipl_re
        + r")"
        r"|(?P<diploma2>"
        + dipl_re
        + r")\s*,?\s*(?P<artblock2>art\.?\s*\d+[^\.;\)]{0,120})"
        r"|(?P<artblock3>art\.?\s*\d+[^\.;\)]{0,120}?)\s*,\s*(?P<diploma3>"
        + dipl_re
        + r")",
        flags=re.IGNORECASE,
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
# 4. EXTRAÇÃO E NORMALIZAÇÃO DE JURISPRUDÊNCIA (VERSÃO FINAL)
# =========================================================


def _remover_pontos_numero_tema(num: str) -> str:
    return num.replace(".", "")


def normalizar_tema(raw: str) -> str:
    txt = raw.strip()
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.strip(" .,:;–-")

    mnum = re.search(
        r"Tema\s*(?:n[º°.]*)?\s*([0-9]{1,4}(?:\.[0-9]{3})*)",
        txt,
        flags=re.IGNORECASE,
    )
    if not mnum:
        return ""

    numero_bruto = mnum.group(1)
    numero_limpo = _remover_pontos_numero_tema(numero_bruto)
    return f"Tema {numero_limpo}"


def normalizar_sumula(raw: str) -> str:
    txt = raw.strip()
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.strip(" .,:;–-")

    mvinc = re.search(
        r"(S[úu]mula\s+Vinculante)\s*(?:n[º°.]*)?\s*([0-9]+)",
        txt,
        flags=re.IGNORECASE,
    )
    if mvinc:
        return f"Súmula Vinculante {mvinc.group(2)}"

    msum = re.search(
        r"(S[úu]mula)\s*(?:n[º°.]*)?\s*([0-9]+)",
        txt,
        flags=re.IGNORECASE,
    )
    if msum:
        return f"Súmula {msum.group(2)}"

    return ""


def extrair_jurisprudencia(ementa: str) -> str:
    """
    Regras consolidadas:
    - Tema sem prefixo STF
    - Tema sem separador de milhar (1.010 -> 1010)
    - Deduplicação
    """
    if not ementa:
        return ""

    txt = ementa

    tema_pattern = re.compile(
        r"(?:STF\s*-\s*)?STF\s*TEMA\s*(?:n[º°.]*)?\s*[0-9]{1,4}(?:\.[0-9]{3})*"
        r"|(?:STF\s*-\s*)?TEMA\s*(?:n[º°.]*)?\s*[0-9]{1,4}(?:\.[0-9]{3})*",
        flags=re.IGNORECASE,
    )

    sumula_pattern = re.compile(
        r"S[úu]mula\s+Vinculante\s*(?:n[º°.]*)?\s*\d+"
        r"|S[úu]mula\s*(?:n[º°.]*)?\s*\d+",
        flags=re.IGNORECASE,
    )

    temas_brutos = tema_pattern.findall(txt)
    sumulas_brutas = sumula_pattern.findall(txt)

    resultados_norm = []
    vistos = set()

    for t in temas_brutos:
        norm_t = normalizar_tema(t)
        if norm_t:
            low = norm_t.lower()
            if low not in vistos:
                resultados_norm.append(norm_t)
                vistos.add(low)

    for s in sumulas_brutas:
        norm_s = normalizar_sumula(s)
        if norm_s:
            low = norm_s.lower()
            if low not in vistos:
                resultados_norm.append(norm_s)
                vistos.add(low)

    return ", ".join(resultados_norm)


# =========================================================
# 5. RESULTADO
# =========================================================


def extrair_resultado(ementa: str, tese: str) -> str:
    ementa_local = ementa or ""

    if re.search(r"\bé\s+infraconstitucional", ementa_local, flags=re.IGNORECASE):
        return "Infraconstitucional"

    if tese.strip():
        return "Tese fixada"

    return "Aguardando julgamento"


# =========================================================
# 6. CONVERSÃO DE DATAS PARA M/D/AAAA
# =========================================================

_MESES_PT: Dict[str, int] = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

_RE_DATE_PT = re.compile(
    r"(?<!\d)(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]+)\s+de\s+(\d{4})(?!\d)",
    flags=re.IGNORECASE,
)
_RE_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_RE_DATE_SLASH = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")


def _strip_acentos(texto: str) -> str:
    norm = unicodedata.normalize("NFD", texto)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def _validar_data(dia: int, mes: int, ano: int) -> bool:
    if ano < 1000 or ano > 9999:
        return False
    if mes < 1 or mes > 12:
        return False
    if dia < 1 or dia > 31:
        return False
    return True


def _format_m_d_aaaa(dia: int, mes: int, ano: int) -> str:
    return f"{mes}/{dia}/{ano}"


def normalizar_datas_m_d_aaaa(texto: str) -> str:
    """
    Converte datas para M/D/AAAA sem zeros à esquerda.
    Suporta:
    - DD de mês de AAAA
    - YYYY-MM-DD
    - D/M/AAAA ou M/D/AAAA (heurística quando necessário)
    """
    if texto is None:
        return texto
    if not isinstance(texto, str):
        texto = str(texto)
    if not texto.strip():
        return texto

    def repl_pt(m: re.Match) -> str:
        dia = int(m.group(1))
        mes_nome = _strip_acentos(m.group(2).lower())
        ano = int(m.group(3))
        mes = _MESES_PT.get(mes_nome)
        if not mes or not _validar_data(dia, mes, ano):
            return m.group(0)
        return _format_m_d_aaaa(dia, mes, ano)

    def repl_iso(m: re.Match) -> str:
        ano = int(m.group(1))
        mes = int(m.group(2))
        dia = int(m.group(3))
        if not _validar_data(dia, mes, ano):
            return m.group(0)
        return _format_m_d_aaaa(dia, mes, ano)

    def repl_slash(m: re.Match) -> str:
        a = int(m.group(1))
        b = int(m.group(2))
        ano = int(m.group(3))

        if a > 12 and b <= 12:
            dia, mes = a, b
        elif b > 12 and a <= 12:
            mes, dia = a, b
        elif a <= 12 and b <= 12:
            mes, dia = a, b
        else:
            return m.group(0)

        if not _validar_data(dia, mes, ano):
            return m.group(0)
        return _format_m_d_aaaa(dia, mes, ano)

    out = _RE_DATE_PT.sub(repl_pt, texto)
    out = _RE_DATE_ISO.sub(repl_iso, out)
    out = _RE_DATE_SLASH.sub(repl_slash, out)
    return out


# =========================================================
# 7. NORMALIZAÇÃO TEXTUAL FINAL (RG_d)
# =========================================================

_re_artigo = re.compile(r"\b(artigo|Artigo)\b")
_re_art_num_letra_ordinal = re.compile(r"(art\.\s+\d+º)([A-Z])", re.IGNORECASE)
_re_art_num_letra_no_ordinal = re.compile(r"(art\.\s+\d+)([A-Z])", re.IGNORECASE)
_re_paragrafo_espaco = re.compile(r"(§)\s*(\d+)")
_re_ellipsis_loose = re.compile(r"\.\s+\.\s+\.")
_re_ellipsis_paren = re.compile(r"\(\s*\.\s*\.\s*\.\s*\)")
_re_space_before_punct = re.compile(r"\s+([,\.;:])")
_re_multi_spaces = re.compile(r" {2,}")
_re_n_variantes = re.compile(r"\b[nN]\s*[\.\s]*º", re.UNICODE)
_re_n_ponto_digito = re.compile(r"\b[nN]\s*\.\s*(\d)")
_re_diploma_singular = re.compile(r"\b(Lei|Decreto|MP)\s+(\d[\d\.\-]*/\d{2,4})")
_re_diploma_singular_curto = re.compile(
    r"\b(LC|Decreto-?Lei)\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE,
)
_re_diploma_plural_basico = re.compile(r"\b(Leis|Decretos|MPs)\s+(\d[\d\.\-]*/\d{2,4})")
_re_lei_complementar_singular = re.compile(
    r"\bLei\s+Complementar\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE,
)
_re_medida_provisoria_singular = re.compile(
    r"\bMedida\s+Provis[oó]ria\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE,
)
_re_decreto_lei_singular = re.compile(
    r"\bDecreto-?Lei\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE,
)
_re_lei_complementar_plural = re.compile(
    r"\bLeis\s+Complementares\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE,
)
_re_medida_provisoria_plural = re.compile(
    r"\bMedidas\s+Provis[oó]rias\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE,
)
_re_decreto_lei_plural = re.compile(
    r"\bDecretos-?Leis\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE,
)
_re_diploma_plural_curto = re.compile(
    r"\b(LCs|Decretos-?Leis|MPs)\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE,
)
_re_singular_milhar_sem_ponto = re.compile(
    r"\b(Lei nº |Decreto nº |MP nº |LC nº |Decreto-?Lei nº )(\d{4,5})(/\d{2,4})",
    re.IGNORECASE,
)
_re_plural_block = re.compile(
    r"(Leis nºs |Decretos nºs |MPs nºs |LCs nºs |Decretos-?Leis nºs )([0-9/\s,e]+)",
    re.IGNORECASE,
)
_re_numero_sem_ponto = re.compile(r"\b(\d{4,5})(/\d{2,4})")
_re_arts_plural = re.compile(r"\barts(?!\.)\b", re.IGNORECASE)
_re_space_after_arts = re.compile(r"\b(arts\.)\s*(?=(\d|[IVXLCDM]))", re.IGNORECASE)
_re_inciso_word = re.compile(r"\binciso\b", re.IGNORECASE)
_re_inc_abbrev = re.compile(r"\binc(?!\.)\b", re.IGNORECASE)
_re_space_after_inc = re.compile(r"\b(inc\.)\s*(?=[IVXLCDM])", re.IGNORECASE)
_re_alinea_to_letter = re.compile(
    r'\b(?:al[ií]nea|al\.?|AL[IÍ]NEA|AL\.?)\s*["“\']?([A-Za-z])["”\']?',
    re.IGNORECASE,
)
_re_list_roman = re.compile(r"(^|\s)([IVXLCDM]+|[ivxlcdm]+)\s*[\)\-–—\.]\s+", re.MULTILINE)
_re_list_alpha = re.compile(r"(^|\s)([A-Za-z])\s*[\)\.]\s+", re.MULTILINE)
_re_caput = re.compile(r"\bcaput\b", re.IGNORECASE)


def _formatar_numero_norma(num: str, sufixo: str) -> str:
    if len(num) == 4:
        return num[0] + "." + num[1:] + sufixo
    if len(num) == 5:
        return num[0:2] + "." + num[2:] + sufixo
    return num + sufixo


def _inserir_ponto_milhar_singular(m: re.Match) -> str:
    prefixo = m.group(1)
    numero = m.group(2)
    sufixo = m.group(3)
    return prefixo + _formatar_numero_norma(numero, sufixo)


def _inserir_ponto_milhar_singular_sub(text: str) -> str:
    return _re_singular_milhar_sem_ponto.sub(_inserir_ponto_milhar_singular, text)


def _formatar_plural_block(m: re.Match) -> str:
    prefixo = m.group(1)
    lista = m.group(2)

    def _subplural(m2: re.Match) -> str:
        numero = m2.group(1)
        sufixo = m2.group(2)
        return _formatar_numero_norma(numero, sufixo)

    lista_fmt = _re_numero_sem_ponto.sub(_subplural, lista)
    return prefixo + lista_fmt


def _normalize_list_marker_spaces(text: str) -> str:
    def repl_roman(m: re.Match) -> str:
        prefix = m.group(1)
        token = m.group(2)
        pre = "" if prefix == "" else " "
        return f"{pre}{token}. "

    def repl_alpha(m: re.Match) -> str:
        prefix = m.group(1)
        token = m.group(2)
        pre = "" if prefix == "" else " "
        return f"{pre}{token}) "

    text = _re_list_roman.sub(repl_roman, text)
    text = _re_list_alpha.sub(repl_alpha, text)
    return text


def normalize_cell(text: str) -> str:
    """
    Normaliza ruído tipográfico e padroniza formas jurídicas visuais.
    """
    if text is None:
        return text
    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").replace("\t", " ")

    linhas = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(linhas).strip()

    text = _re_artigo.sub("art.", text)
    text = _re_art_num_letra_ordinal.sub(r"\1-\2", text)
    text = _re_art_num_letra_no_ordinal.sub(r"\1-\2", text)
    text = _re_paragrafo_espaco.sub(r"\1\2", text)
    text = text.replace(". ..", "...")
    text = _re_ellipsis_loose.sub("...", text)
    text = _re_ellipsis_paren.sub("(...)", text)

    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    text = text.replace("qü", "qu").replace("Qü", "Qu").replace("gü", "gu").replace("Gü", "Gu")

    text = text.replace("n°", "nº")
    text = _re_n_variantes.sub("nº", text)
    text = _re_n_ponto_digito.sub(r"nº \1", text)

    text = _re_lei_complementar_singular.sub(r"LC nº \1", text)
    text = _re_medida_provisoria_singular.sub(r"MP nº \1", text)
    text = _re_decreto_lei_singular.sub(r"Decreto-Lei nº \1", text)

    text = _re_lei_complementar_plural.sub(r"LCs nºs \1", text)
    text = _re_medida_provisoria_plural.sub(r"MPs nºs \1", text)
    text = _re_decreto_lei_plural.sub(r"Decretos-Leis nºs \1", text)

    text = _re_diploma_singular.sub(r"\1 nº \2", text)
    text = _re_diploma_singular_curto.sub(r"\1 nº \2", text)
    text = _re_diploma_plural_basico.sub(r"\1 nºs \2", text)
    text = _re_diploma_plural_curto.sub(r"\1 nºs \2", text)

    text = _inserir_ponto_milhar_singular_sub(text)
    text = _re_plural_block.sub(_formatar_plural_block, text)

    text = _re_arts_plural.sub("arts.", text)
    text = _re_space_after_arts.sub(r"\1 ", text)

    text = _re_inciso_word.sub("inc.", text)
    text = _re_inc_abbrev.sub("inc.", text)
    text = _re_space_after_inc.sub(r"\1 ", text)

    text = _re_alinea_to_letter.sub(r"\1", text)
    text = _re_caput.sub("caput", text)

    text = _normalize_list_marker_spaces(text)

    text = _re_space_before_punct.sub(r"\1", text)

    linhas = text.split("\n")
    linhas = [_re_multi_spaces.sub(" ", ln) for ln in linhas]
    text = "\n".join(linhas)

    if text.strip() == "-":
        return ""

    return text


# =========================================================
# 8. SUPORTE CSV + GUI
# =========================================================

NOME_ARQUIVO_SAIDA = "STF_RG_consolidado.csv"


def detectar_encoding(caminho_entrada: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(caminho_entrada, "r", encoding=enc) as f:
                f.read()
            return enc
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "encoding-desconhecido",
        b"",
        0,
        1,
        "Não foi possível decodificar o arquivo em UTF-8 nem em Latin-1.",
    )


def localizar_coluna_ementa(header: List[str]) -> int:
    for idx, col in enumerate(header):
        if col.strip().lower() == "ementa":
            return idx

    if len(header) > 6:
        return 6

    return len(header) - 1


def _normalizar_nome_coluna(nome: str) -> str:
    limpo = _strip_acentos((nome or "").strip().lower())
    limpo = re.sub(r"\s+", " ", limpo)
    return limpo


def _garantir_colunas_saida(header: List[str]) -> Dict[str, int]:
    desejadas = [
        ("tese", "Tese"),
        ("legislacao", "Legislação"),
        ("jurisprudencia", "Jurisprudência"),
        ("ramo_direito", "Ramo do direito"),
        ("resultado", "Resultado"),
    ]

    mapa_existente = {_normalizar_nome_coluna(col): i for i, col in enumerate(header)}
    indices: Dict[str, int] = {}

    for chave, nome_canonico in desejadas:
        normalizado = _normalizar_nome_coluna(nome_canonico)
        if normalizado in mapa_existente:
            indices[chave] = mapa_existente[normalizado]
        else:
            header.append(nome_canonico)
            novo_idx = len(header) - 1
            indices[chave] = novo_idx
            mapa_existente[normalizado] = novo_idx

    return indices


def _caminho_saida_padrao() -> str:
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pasta_script, NOME_ARQUIVO_SAIDA)


def _processar_csv_para_memoria(input_file: str, verbose: bool = False) -> tuple[List[str], List[List[str]]]:
    """
    Processa um CSV de entrada e devolve (header, linhas_processadas) sem gravar em disco.
    """
    if verbose:
        print(f"[INFO] Arquivo de entrada: {input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_file}")

    encoding_entrada = detectar_encoding(input_file)
    if verbose:
        print(f"[INFO] Encoding detectado: {encoding_entrada}")

    with open(input_file, mode="r", encoding=encoding_entrada, newline="") as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError(f"CSV vazio - nenhum cabeçalho encontrado: {input_file}")

        rows = list(reader)

    ementa_idx = localizar_coluna_ementa(header)
    idx_cols = _garantir_colunas_saida(header)

    total = len(rows)
    processed_rows: List[List[str]] = []
    last_pct = -1

    for i, row in enumerate(rows, start=1):
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))

        ementa_text = row[ementa_idx] if len(row) > ementa_idx else ""

        tese = extrair_tese(ementa_text)
        legislacao = extrair_legislacao(ementa_text)
        jurisprudencia = extrair_jurisprudencia(ementa_text)
        ramo = extrair_ramos(ementa_text)
        resultado = extrair_resultado(ementa_text, tese)

        row[idx_cols["tese"]] = tese
        row[idx_cols["legislacao"]] = legislacao
        row[idx_cols["jurisprudencia"]] = jurisprudencia
        row[idx_cols["ramo_direito"]] = ramo
        row[idx_cols["resultado"]] = resultado

        row = [normalizar_datas_m_d_aaaa(normalize_cell(normalizar_datas_m_d_aaaa(cel))) for cel in row]
        processed_rows.append(row)

        if verbose and total > 0:
            pct = int((i / total) * 100)
            if pct // 5 != last_pct // 5:
                print(f"[PROGRESS] {os.path.basename(input_file)}: linha {i}/{total} ({pct}%)...")
            last_pct = pct

    if verbose:
        print(f"[SUCESSO] {total} linhas processadas em {os.path.basename(input_file)}.")

    return header, processed_rows


def _expandir_header_consolidado(
    header_consolidado: List[str],
    linhas_consolidadas: List[List[str]],
    header_novo: List[str],
) -> None:
    mapa_existente = {_normalizar_nome_coluna(col): i for i, col in enumerate(header_consolidado)}

    for col in header_novo:
        norm = _normalizar_nome_coluna(col)
        if norm not in mapa_existente:
            mapa_existente[norm] = len(header_consolidado)
            header_consolidado.append(col)
            for linha in linhas_consolidadas:
                linha.append("")


def _alinhar_linha_para_header(
    row_origem: List[str],
    header_origem: List[str],
    header_destino: List[str],
) -> List[str]:
    mapa_valores: Dict[str, str] = {}
    for idx, col in enumerate(header_origem):
        norm = _normalizar_nome_coluna(col)
        if norm not in mapa_valores:
            mapa_valores[norm] = row_origem[idx] if idx < len(row_origem) else ""

    row_destino: List[str] = []
    for col_destino in header_destino:
        row_destino.append(mapa_valores.get(_normalizar_nome_coluna(col_destino), ""))

    return row_destino


def processar_csv_consolidado(input_files: List[str], output_file: str, verbose: bool = True) -> None:
    """
    Processa múltiplos CSVs e consolida em um único arquivo de saída.
    """
    if not input_files:
        raise RuntimeError("Nenhum arquivo CSV selecionado.")

    header_consolidado: List[str] = []
    linhas_consolidadas: List[List[str]] = []
    total_fontes = len(input_files)

    for idx_arquivo, input_file in enumerate(input_files, start=1):
        if verbose:
            print(f"[INFO] Processando fonte {idx_arquivo}/{total_fontes}: {input_file}")

        header_local, linhas_local = _processar_csv_para_memoria(input_file, verbose=verbose)

        if not header_consolidado:
            header_consolidado = header_local[:]
            linhas_consolidadas.extend(linhas_local)
            continue

        _expandir_header_consolidado(header_consolidado, linhas_consolidadas, header_local)
        for linha_local in linhas_local:
            linha_alinhada = _alinhar_linha_para_header(linha_local, header_local, header_consolidado)
            linhas_consolidadas.append(linha_alinhada)

    with open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header_consolidado)
        writer.writerows(linhas_consolidadas)

    if verbose:
        print(f"[SUCESSO] Consolidação finalizada: {len(linhas_consolidadas)} linhas.")
        print(f"[SUCESSO] Arquivo de saída criado: {output_file}")


def _listar_csv_em_pasta(pasta: str) -> List[str]:
    encontrados: List[str] = []
    for raiz, _, arquivos in os.walk(pasta):
        for nome in sorted(arquivos):
            if not nome.lower().endswith(".csv"):
                continue
            if nome.lower() == NOME_ARQUIVO_SAIDA.lower():
                continue
            encontrados.append(os.path.join(raiz, nome))
    return sorted(encontrados, key=lambda p: p.lower())


class STFConsolidadoGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("STF RG - Consolidador CSV")
        self.root.geometry("880x560")
        self.root.minsize(780, 500)

        self.arquivos_selecionados: List[str] = []
        self.output_path = _caminho_saida_padrao()
        self.status_var = tk.StringVar(value="Selecione arquivos CSV e/ou pastas para iniciar.")

        self._montar_interface()

    def _montar_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        titulo = tk.Label(
            self.root,
            text="Consolidar STF RG em CSV único",
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        )
        titulo.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))

        barra_acoes = tk.Frame(self.root)
        barra_acoes.grid(row=1, column=0, sticky="ew", padx=14)
        barra_acoes.columnconfigure(4, weight=1)

        tk.Button(barra_acoes, text="Adicionar arquivos CSV", command=self.adicionar_arquivos).grid(
            row=0, column=0, padx=(0, 8), pady=4
        )
        tk.Button(barra_acoes, text="Adicionar pasta", command=self.adicionar_pasta).grid(
            row=0, column=1, padx=(0, 8), pady=4
        )
        tk.Button(barra_acoes, text="Remover selecionado", command=self.remover_selecionado).grid(
            row=0, column=2, padx=(0, 8), pady=4
        )
        tk.Button(barra_acoes, text="Limpar lista", command=self.limpar_lista).grid(
            row=0, column=3, padx=(0, 8), pady=4
        )

        frame_lista = tk.Frame(self.root)
        frame_lista.grid(row=2, column=0, sticky="nsew", padx=14, pady=(8, 8))
        frame_lista.columnconfigure(0, weight=1)
        frame_lista.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(frame_lista, activestyle="dotbox")
        self.listbox.grid(row=0, column=0, sticky="nsew")

        scroll = tk.Scrollbar(frame_lista, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scroll.set)

        frame_saida = tk.Frame(self.root)
        frame_saida.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 4))
        frame_saida.columnconfigure(1, weight=1)

        tk.Label(frame_saida, text="Saída fixa: ", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(frame_saida, text=self.output_path, anchor="w", justify="left").grid(
            row=0, column=1, sticky="w"
        )

        tk.Label(self.root, textvariable=self.status_var, anchor="w").grid(
            row=4, column=0, sticky="ew", padx=14, pady=(2, 6)
        )

        tk.Button(
            self.root,
            text=f"Gerar {NOME_ARQUIVO_SAIDA}",
            command=self.executar_consolidacao,
            font=("Segoe UI", 11, "bold"),
            padx=12,
            pady=8,
        ).grid(row=5, column=0, sticky="e", padx=14, pady=(0, 14))

    def _atualizar_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for path in self.arquivos_selecionados:
            self.listbox.insert(tk.END, path)
        self.status_var.set(f"{len(self.arquivos_selecionados)} arquivo(s) pronto(s) para consolidação.")

    def _adicionar_caminhos(self, caminhos: List[str]) -> int:
        existentes = {os.path.normcase(os.path.abspath(p)) for p in self.arquivos_selecionados}
        adicionados = 0

        for caminho in caminhos:
            abs_path = os.path.abspath(caminho)
            if not os.path.isfile(abs_path):
                continue
            if not abs_path.lower().endswith(".csv"):
                continue
            if os.path.basename(abs_path).lower() == NOME_ARQUIVO_SAIDA.lower():
                continue

            key = os.path.normcase(abs_path)
            if key in existentes:
                continue

            self.arquivos_selecionados.append(abs_path)
            existentes.add(key)
            adicionados += 1

        self.arquivos_selecionados.sort(key=lambda p: p.lower())
        self._atualizar_listbox()
        return adicionados

    def adicionar_arquivos(self) -> None:
        arquivos = filedialog.askopenfilenames(
            title="Selecione um ou mais arquivos CSV",
            filetypes=[("Arquivos CSV", "*.csv")],
        )
        if not arquivos:
            return

        qtd = self._adicionar_caminhos(list(arquivos))
        if qtd == 0:
            messagebox.showinfo("Sem novos arquivos", "Nenhum novo CSV foi adicionado.")

    def adicionar_pasta(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione a pasta com CSVs")
        if not pasta:
            return

        csvs = _listar_csv_em_pasta(pasta)
        if not csvs:
            messagebox.showwarning("Pasta vazia", "Nenhum CSV foi encontrado nesta pasta.")
            return

        qtd = self._adicionar_caminhos(csvs)
        if qtd == 0:
            messagebox.showinfo("Sem novos arquivos", "Todos os CSVs dessa pasta já estavam na lista.")

    def remover_selecionado(self) -> None:
        selecionado = self.listbox.curselection()
        if not selecionado:
            return
        idx = selecionado[0]
        del self.arquivos_selecionados[idx]
        self._atualizar_listbox()

    def limpar_lista(self) -> None:
        self.arquivos_selecionados.clear()
        self._atualizar_listbox()
        self.status_var.set("Lista limpa. Selecione arquivos CSV e/ou pastas para iniciar.")

    def executar_consolidacao(self) -> None:
        if not self.arquivos_selecionados:
            messagebox.showwarning("Nenhuma entrada", "Selecione ao menos um arquivo CSV ou uma pasta.")
            return

        try:
            self.status_var.set("Processando arquivos, aguarde...")
            self.root.update_idletasks()

            processar_csv_consolidado(
                self.arquivos_selecionados,
                self.output_path,
                verbose=False,
            )

            self.status_var.set(f"Concluído: {self.output_path}")
            messagebox.showinfo(
                "Consolidação concluída",
                f"Arquivo gerado com sucesso:\n{self.output_path}",
            )
        except Exception as exc:
            self.status_var.set("Erro durante a consolidação.")
            messagebox.showerror("Erro", str(exc))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if tk is None:
        raise RuntimeError(
            "Tkinter não está disponível neste Python. "
            "Instale o pacote Tk/Tkinter para usar a interface GUI."
        )
    app = STFConsolidadoGUI()
    app.run()


if __name__ == "__main__":
    main()
