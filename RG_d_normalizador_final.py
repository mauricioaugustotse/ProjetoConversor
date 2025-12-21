import csv
import os
import re
import sys

# -------------------------------------------------
# REGEX PRÉ-COMPILADOS (performance e organização)
# -------------------------------------------------

# "artigo" -> "art."
_re_artigo = re.compile(r"\b(artigo|Artigo)\b")

# "art. 1ºF" -> "art. 1º-F"
_re_art_num_letra_ordinal = re.compile(r"(art\.\s+\d+º)([A-Z])", re.IGNORECASE)

# "art. 29C" -> "art. 29-C"
_re_art_num_letra_no_ordinal = re.compile(r"(art\.\s+\d+)([A-Z])", re.IGNORECASE)

# "§ 2º" -> "§2º"
_re_paragrafo_espaco = re.compile(r"(§)\s*(\d+)")

# reticências quebradas ". . ." etc. -> "..."
_re_ellipsis_loose = re.compile(r"\.\s+\.\s+\.")
_re_ellipsis_paren = re.compile(r"\(\s*\.\s*\.\s*\.\s*\)")

# remover espaço ANTES de pontuação , . ; :
_re_space_before_punct = re.compile(r"\s+([,\.;:])")

# colapsar espaços múltiplos numa mesma linha
_re_multi_spaces = re.compile(r" {2,}")

# variações "n. º", "n.º", "n .º", "N.º" etc -> "nº"
_re_n_variantes = re.compile(r"\b[nN]\s*[\.\s]*º", re.UNICODE)

# "n." / "N." seguido de dígito -> "nº <dígito>"
_re_n_ponto_digito = re.compile(r"\b[nN]\s*\.\s*(\d)")

# Diplomas normativos no singular (Lei, Decreto, MP)
# Ex.: "Lei 8622/1993" -> "Lei nº 8622/1993"
_re_diploma_singular = re.compile(
    r"\b(Lei|Decreto|MP)\s+(\d[\d\.\-]*/\d{2,4})"
)

# Diplomas normativos no singular curtos:
# "LC 105/2001" -> "LC nº 105/2001"
# "Decreto-Lei 911/1969" -> "Decreto-Lei nº 911/1969"
_re_diploma_singular_curto = re.compile(
    r"\b(LC|Decreto-?Lei)\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE
)

# Diplomas normativos no plural básicos:
# "Leis 8622/1993", "Decretos 3048/1999", "MPs 1127/2022"
_re_diploma_plural_basico = re.compile(
    r"\b(Leis|Decretos|MPs)\s+(\d[\d\.\-]*/\d{2,4})"
)

# Formas longas -> abreviações padronizadas (singular):
# "Lei Complementar 105/2001"   -> "LC nº 105/2001"
_re_lei_complementar_singular = re.compile(
    r"\bLei\s+Complementar\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE
)

# "Medida Provisória 1127/2022" -> "MP nº 1127/2022"
_re_medida_provisoria_singular = re.compile(
    r"\bMedida\s+Provis[oó]ria\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE
)

# "Decreto-Lei 911/1969" -> "Decreto-Lei nº 911/1969"
_re_decreto_lei_singular = re.compile(
    r"\bDecreto-?Lei\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE
)

# Formas longas -> abreviações padronizadas (plural):
# "Leis Complementares 105/2001 e 110/2001" -> "LCs nºs 105/2001 e 110/2001"
_re_lei_complementar_plural = re.compile(
    r"\bLeis\s+Complementares\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE
)

# "Medidas Provisórias 1127/2022 e 1185/2023"
# -> "MPs nºs 1127/2022 e 1185/2023"
_re_medida_provisoria_plural = re.compile(
    r"\bMedidas\s+Provis[oó]rias\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE
)

# "Decretos-Leis 911/1969 e 667/1969"
# -> "Decretos-Leis nºs 911/1969 e 667/1969"
_re_decreto_lei_plural = re.compile(
    r"\bDecretos-?Leis\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE
)

# Plural curto adicional:
# "LCs 105/2001", "Decretos-Leis 911/1969", "MPs 1127/2022"
_re_diploma_plural_curto = re.compile(
    r"\b(LCs|Decretos-?Leis|MPs)\s+(\d[\d\.\-]*/\d{2,4})",
    re.IGNORECASE
)

# Inserção do ponto nos milhares em citações SINGULARES padronizadas
# "Lei nº 9718/98"        -> "Lei nº 9.718/98"
# "Decreto nº 3048/1999"  -> "Decreto nº 3.048/1999"
# "MP nº 1127/2022"       -> "MP nº 1.127/2022"
# "LC nº 14133/2021"      -> "LC nº 14.133/2021"
# "Decreto-Lei nº 9111/1969" -> "Decreto-Lei nº 9.111/1969"
_re_singular_milhar_sem_ponto = re.compile(
    r"\b(Lei nº |Decreto nº |MP nº |LC nº |Decreto-?Lei nº )(\d{4,5})(/\d{2,4})",
    re.IGNORECASE
)

# Bloco plural pós "nºs":
# "Leis nºs 8622/1993 e 8627/1993"
# "MPs nºs 1127/2022, 1185/2023"
_re_plural_block = re.compile(
    r"(Leis nºs |Decretos nºs |MPs nºs |LCs nºs |Decretos-?Leis nºs )([0-9/\s,e]+)",
    re.IGNORECASE
)

# Dentro da lista plural, localizar cada número cru de 4 ou 5 dígitos
_re_numero_sem_ponto = re.compile(
    r"\b(\d{4,5})(/\d{2,4})"
)

# Padronização interna de citações de dispositivos:
# "arts" -> "arts."
_re_arts_plural = re.compile(r"\barts(?!\.)\b", re.IGNORECASE)
# garantir espaço depois de "arts." quando vem nº/romano
_re_space_after_arts = re.compile(
    r"\b(arts\.)\s*(?=(\d|[IVXLCDM]))",
    re.IGNORECASE
)

# "inciso" -> "inc." ; "inc" -> "inc."
_re_inciso_word = re.compile(r"\binciso\b", re.IGNORECASE)
_re_inc_abbrev  = re.compile(r"\binc(?!\.)\b", re.IGNORECASE)
_re_space_after_inc = re.compile(
    r"\b(inc\.)\s*(?=[IVXLCDM])",
    re.IGNORECASE
)

# "alínea"/"al."/ "al" -> apenas a letra da alínea
# Ex.: 'alínea "a"' -> 'a' ; 'al. b' -> 'b' ; 'al C' -> 'C'
_re_alinea_to_letter = re.compile(
    r'\b(?:al[ií]nea|al\.?|AL[IÍ]NEA|AL\.?)\s*["“\']?([A-Za-z])["”\']?',
    re.IGNORECASE
)

# Listas com ROMANOS:
# "I)", "II -", "III.", "iv)" etc.
# Devem virar "I. ", "II. ", "III. " ... (mantém caixa do original)
_re_list_roman = re.compile(
    r'(^|\s)([IVXLCDM]+|[ivxlcdm]+)\s*[\)\-–—\.]\s+',
    re.MULTILINE
)

# Listas com LETRAS:
# "a)", "b.", "c)" -> sempre "a) ", "b) ", "c) " (mantém caixa)
_re_list_alpha = re.compile(
    r'(^|\s)([A-Za-z])\s*[\)\.]\s+',
    re.MULTILINE
)

# "Caput"/"CAPUT" -> "caput"
_re_caput = re.compile(r"\bcaput\b", re.IGNORECASE)


def _formatar_numero_norma(num: str, sufixo: str) -> str:
    """
    '9718','/98'     -> '9.718/98'
    '14133','/2021' -> '14.133/2021'
    '1127','/2022'  -> '1.127/2022'
    """
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
    """
    - Romanos:
        "I)", "II -", "III.", "iv)" etc.
        => "I. ", "II. ", "III. ", "iv. " (mantém maiúsculo/minúsculo)
    - Letras:
        "a)", "b.", "c)" etc.
        => "a) ", "b) ", "c) " (mantém maiúsculo/minúsculo)

    Observações:
    - Garantimos exatamente um espaço depois do marcador final.
    - Se o marcador não está no começo da linha, garantimos um espaço
      antes dele, para não grudar na palavra anterior.
    """

    def repl_roman(m: re.Match) -> str:
        prefix = m.group(1)   # "" (início da linha) ou whitespace
        token  = m.group(2)   # romano (I, II, iii, iv...)
        # Se prefix == "" (início de linha), não colocamos espaço antes.
        pre = "" if prefix == "" else " "
        # Saída roman: "I. "
        return f"{pre}{token}. "

    def repl_alpha(m: re.Match) -> str:
        prefix = m.group(1)   # "" ou whitespace
        token  = m.group(2)   # letra (a, b, C...)
        pre = "" if prefix == "" else " "
        # Saída letra: "a) "
        return f"{pre}{token}) "

    text = _re_list_roman.sub(repl_roman, text)
    text = _re_list_alpha.sub(repl_alpha, text)
    return text


# -------------------------------------------------
# FUNÇÃO DE NORMALIZAÇÃO DE UMA CÉLULA
# -------------------------------------------------

def normalize_cell(text: str) -> str:
    """
    Normaliza ruído tipográfico e padroniza formas jurídicas visuais,
    SEM alterar o conteúdo normativo/material.
    """

    # 1) Garantir string
    if text is None:
        return text
    if not isinstance(text, str):
        text = str(text)

    # Se está vazia após strip → retorna ""
    if not text.strip():
        return ""

    # 2) Normalização bruta: fim de linha, NBSP, tabs
    text = (
        text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\xa0", " ")
            .replace("\t", " ")
    )

    # 3) Strip de borda em cada linha interna, mas preserva \n
    linhas = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(linhas).strip()

    # 4) Padronizações jurídicas / tipográficas

    # 4.1 "artigo" -> "art."
    text = _re_artigo.sub("art.", text)

    # 4.2 "art. 1ºF" -> "art. 1º-F"
    text = _re_art_num_letra_ordinal.sub(r"\1-\2", text)

    # 4.3 "art. 29C" -> "art. 29-C"
    text = _re_art_num_letra_no_ordinal.sub(r"\1-\2", text)

    # 4.4 "§ 2º" -> "§2º"
    text = _re_paragrafo_espaco.sub(r"\1\2", text)

    # 4.5 Reticências quebradas -> "..."
    text = text.replace(". ..", "...")
    text = _re_ellipsis_loose.sub("...", text)
    text = _re_ellipsis_paren.sub("(...)", text)

    # 4.6 Aspas curvas -> retas
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )

    # 4.7 Ortografia pré-acordo (qü/gü)
    text = (
        text.replace("qü", "qu")
            .replace("Qü", "Qu")
            .replace("gü", "gu")
            .replace("Gü", "Gu")
    )

    # 4.8 n° / n.º / n. º etc -> nº ; também "n. 123" -> "nº 123"
    text = text.replace("n°", "nº")
    text = _re_n_variantes.sub("nº", text)
    text = _re_n_ponto_digito.sub(r"nº \1", text)

    # 4.9 Formas longas para formas padronizadas (singular)
    text = _re_lei_complementar_singular.sub(r"LC nº \1", text)
    text = _re_medida_provisoria_singular.sub(r"MP nº \1", text)
    text = _re_decreto_lei_singular.sub(r"Decreto-Lei nº \1", text)

    # 4.10 Formas longas para plural
    text = _re_lei_complementar_plural.sub(r"LCs nºs \1", text)
    text = _re_medida_provisoria_plural.sub(r"MPs nºs \1", text)
    text = _re_decreto_lei_plural.sub(r"Decretos-Leis nºs \1", text)

    # 4.11 Diplomas curtos/simples no singular e plural
    text = _re_diploma_singular.sub(r"\1 nº \2", text)
    text = _re_diploma_singular_curto.sub(r"\1 nº \2", text)
    text = _re_diploma_plural_basico.sub(r"\1 nºs \2", text)
    text = _re_diploma_plural_curto.sub(r"\1 nºs \2", text)

    # 4.12 Inserir pontos de milhar (singular já padronizado)
    text = _inserir_ponto_milhar_singular_sub(text)

    # 4.13 Inserir pontos de milhar (plural "nºs ...")
    text = _re_plural_block.sub(_formatar_plural_block, text)

    # 4.14 Padronização adicional de citações normativas internas
    # arts -> arts. (+ espaço quando colado)
    text = _re_arts_plural.sub("arts.", text)
    text = _re_space_after_arts.sub(r"\1 ", text)

    # inciso -> inc. (+ espaço quando colado)
    text = _re_inciso_word.sub("inc.", text)
    text = _re_inc_abbrev.sub("inc.", text)
    text = _re_space_after_inc.sub(r"\1 ", text)

    # alínea/al./al -> só a letra (remove "al.", "alínea", etc.)
    text = _re_alinea_to_letter.sub(r"\1", text)

    # caput -> sempre "caput" minúsculo
    text = _re_caput.sub("caput", text)

    # 4.15 Organização hierárquica de listas
    # - Romanos => "I. "
    # - Letras  => "a) "
    text = _normalize_list_marker_spaces(text)

    # 5) Espaçamento e pontuação

    # 5.1 Remover espaço ANTES de , . ; :
    text = _re_space_before_punct.sub(r"\1", text)

    # 5.2 Colapsar espaços múltiplos por linha, mantendo \n
    linhas = text.split("\n")
    linhas = [_re_multi_spaces.sub(" ", ln) for ln in linhas]
    text = "\n".join(linhas)

    # 6) Regra final: célula "-" sozinha -> vazia
    if text.strip() == "-":
        return ""

    return text


# -------------------------------------------------
# SUPORTE: NOME DE SAÍDA E ENCODING
# -------------------------------------------------

def gerar_nome_saida(caminho_entrada: str) -> str:
    """
    'dados.csv' -> 'dados_normalizado.csv'
    '/pasta/planilha.csv' -> '/pasta/planilha_normalizado.csv'
    """
    base, ext = os.path.splitext(caminho_entrada)
    return f"{base}_normalizado{ext or '.csv'}"


def detectar_encoding(caminho_entrada: str) -> str:
    """
    Tenta abrir como UTF-8. Se falhar, tenta Latin-1.
    Se as duas falharem, levanta erro.
    """
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
        "Não foi possível decodificar o arquivo em UTF-8 nem em Latin-1."
    )


# -------------------------------------------------
# PIPELINE PRINCIPAL
# -------------------------------------------------

def processar_csv(caminho_entrada: str):
    if not os.path.exists(caminho_entrada):
        print(f"Erro: arquivo '{caminho_entrada}' não encontrado.")
        sys.exit(1)

    encoding_entrada = detectar_encoding(caminho_entrada)
    caminho_saida = gerar_nome_saida(caminho_entrada)

    print(f"Lendo '{caminho_entrada}' com encoding '{encoding_entrada}'...")
    print(f"Gerando arquivo normalizado em '{caminho_saida}' (UTF-8)...")

    linhas_processadas = []
    num_linhas = 0

    with open(caminho_entrada, mode="r", encoding=encoding_entrada, newline="") as f_in:
        reader = csv.reader(f_in)
        for row in reader:
            nova_linha = [normalize_cell(cel) for cel in row]
            linhas_processadas.append(nova_linha)
            num_linhas += 1

    with open(caminho_saida, mode="w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerows(linhas_processadas)

    print("-" * 40)
    print("Normalização concluída.")
    print(f"Total de linhas processadas: {num_linhas}")
    print(f"Arquivo salvo como: {caminho_saida}")


# -------------------------------------------------
# PONTO DE ENTRADA
# -------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        caminho_in = sys.argv[1]
        processar_csv(caminho_in)
    else:
        caminho_padrao = "Temas_STJ_todos.csv"  # ajuste se quiser outro padrão
        if os.path.exists(caminho_padrao):
            print(f"Nenhum argumento recebido. Usando arquivo padrão: {caminho_padrao}")
            processar_csv(caminho_padrao)
        else:
            print("Uso:")
            print("    python RG_normalizador_final.py caminho/do/arquivo.csv")
            print()
            print(
                "Nenhum argumento recebido e o arquivo padrão "
                f"'{caminho_padrao}' não foi encontrado neste diretório."
            )
            sys.exit(1)