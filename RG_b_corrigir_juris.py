import csv
import glob
import os
import re
from typing import List


def detectar_csv_processado() -> str:
    """
    Localiza automaticamente um CSV de entrada.
    Preferência: arquivos que contenham 'processado' no nome.
    Se houver mais de um, pega o primeiro em ordem alfabética.
    Se não houver, usa qualquer .csv na pasta.
    """
    candidates = [f for f in glob.glob("*.csv") if "processado" in f.lower()]
    if not candidates:
        candidates = glob.glob("*.csv")

    if not candidates:
        raise FileNotFoundError(
            "Nenhum arquivo .csv encontrado na pasta atual. "
            "Coloque o CSV processado na mesma pasta deste script."
        )

    candidates.sort()
    return candidates[0]


def localizar_indices_colunas(header: List[str]) -> dict:
    """
    Descobre os índices das colunas relevantes pelo nome.
    Retorna dict com:
      - 'ementa_idx'
      - 'juris_idx'

    Se não encontrar algum nome esperado, aplica heurísticas.
    """
    nome_para_idx = {col.strip().lower(): i for i, col in enumerate(header)}

    # Ementa
    if "ementa" in nome_para_idx:
        ementa_idx = nome_para_idx["ementa"]
    else:
        # fallback: coluna G padrão (índice 6) ou última coluna
        ementa_idx = 6 if len(header) > 6 else len(header) - 1

    # Jurisprudência (com ou sem acento)
    if "jurisprudência" in nome_para_idx:
        juris_idx = nome_para_idx["jurisprudência"]
    elif "jurisprudencia" in nome_para_idx:
        juris_idx = nome_para_idx["jurisprudencia"]
    else:
        juris_idx = None  # vamos criar depois se não existir

    return {
        "ementa_idx": ementa_idx,
        "juris_idx": juris_idx,
    }


# ---------------------------------------------------------
# EXTRAÇÃO / NORMALIZAÇÃO DA JURISPRUDÊNCIA
# ---------------------------------------------------------

def _remover_pontos_numero_tema(num: str) -> str:
    """
    Recebe algo tipo '1.010' ou '1.234' e retorna '1010' / '1234'.
    Se receber '985', retorna '985'.
    """
    return num.replace('.', '')


def normalizar_tema(raw: str) -> str:
    """
    Normaliza referências de "Tema", padronizando SEM prefixo "STF"
    e SEM pontos no número.

    Exemplos:
      "STF - Tema 1.010"      -> "Tema 1010"
      "STF Tema nº 1.234"     -> "Tema 1234"
      "Tema nº 1.010"         -> "Tema 1010"
      "Tema 985"              -> "Tema 985"
    """
    txt = raw.strip()

    # colapsa espaços e limpa pontuação solta no final
    txt = re.sub(r'\s+', ' ', txt)
    txt = txt.strip(' .,:;–-')

    # extrai o número completo (com possíveis pontos internos)
    mnum = re.search(
        r'Tema\s*(?:n[º°.]*)?\s*([0-9]{1,4}(?:\.[0-9]{3})*)',
        txt,
        flags=re.IGNORECASE
    )
    if not mnum:
        return ""

    numero_bruto = mnum.group(1)
    numero_limpo = _remover_pontos_numero_tema(numero_bruto)

    # padronização única, sem "STF"
    return f"Tema {numero_limpo}"


def normalizar_sumula(raw: str) -> str:
    """
    Normaliza referências a Súmula / Súmula Vinculante.
    Exemplos:
      "Súmula Vinculante 13"
      "Súmula 279"
      "Súmula n.º 279"
      "Súmula Vinculante n° 14"
    """
    txt = raw.strip()
    txt = re.sub(r'\s+', ' ', txt)
    txt = txt.strip(' .,:;–-')

    # Súmula Vinculante primeiro
    mvinc = re.search(
        r'(S[úu]mula\s+Vinculante)\s*(?:n[º°.]*)?\s*([0-9]+)',
        txt,
        flags=re.IGNORECASE
    )
    if mvinc:
        base_norm = "Súmula Vinculante"
        num = mvinc.group(2)
        return f"{base_norm} {num}"

    # Súmula simples
    msum = re.search(
        r'(S[úu]mula)\s*(?:n[º°.]*)?\s*([0-9]+)',
        txt,
        flags=re.IGNORECASE
    )
    if msum:
        base_norm = "Súmula"
        num = msum.group(2)
        return f"{base_norm} {num}"

    return ""


def extrair_jurisprudencia_corrigida(ementa: str) -> str:
    """
    A partir do texto da ementa, extrai:
      - Temas (ex.: "Tema 1010")
      - Súmulas (ex.: "Súmula 279")
      - Súmula Vinculante (ex.: "Súmula Vinculante 13")

    Regras:
      - "Tema 1.010" -> "Tema 1010"
      - "STF Tema 1.010" -> "Tema 1010"
      - Nunca manter ".", nunca manter "STF"
      - Deduplicação
      - Join final com ", "
    """
    if not ementa:
        return ""

    txt = ementa

    # Captura candidatos a "Tema ..."
    # Aceita:
    #   "STF - Tema 1.010"
    #   "STF Tema 1.234"
    #   "Tema nº 1.010"
    #   "Tema 985"
    tema_pattern = re.compile(
        r'(?:STF\s*-\s*)?STF\s*TEMA\s*(?:n[º°.]*)?\s*[0-9]{1,4}(?:\.[0-9]{3})*'
        r'|'
        r'(?:STF\s*-\s*)?TEMA\s*(?:n[º°.]*)?\s*[0-9]{1,4}(?:\.[0-9]{3})*',
        flags=re.IGNORECASE
    )

    # Captura candidatos a Súmula / Súmula Vinculante
    sumula_pattern = re.compile(
        r'S[úu]mula\s+Vinculante\s*(?:n[º°.]*)?\s*\d+'
        r'|'
        r'S[úu]mula\s*(?:n[º°.]*)?\s*\d+',
        flags=re.IGNORECASE
    )

    temas_brutos = tema_pattern.findall(txt)
    sumulas_brutas = sumula_pattern.findall(txt)

    resultados_norm = []
    vistos = set()

    # Normaliza Temas -> "Tema 1234"
    for t in temas_brutos:
        norm_t = normalizar_tema(t)
        if norm_t:
            low = norm_t.lower()
            if low not in vistos:
                resultados_norm.append(norm_t)
                vistos.add(low)

    # Normaliza Súmulas
    for s in sumulas_brutas:
        norm_s = normalizar_sumula(s)
        if norm_s:
            low = norm_s.lower()
            if low not in vistos:
                resultados_norm.append(norm_s)
                vistos.add(low)

    return ", ".join(resultados_norm)


# ---------------------------------------------------------
# PROCESSAMENTO DO CSV
# ---------------------------------------------------------

def corrigir_jurisprudencia_csv(input_file: str, output_file: str, verbose: bool = True) -> None:
    """
    Lê o CSV processado existente, recalcula a coluna 'Jurisprudência'
    a partir da coluna 'Ementa', e salva um novo CSV com sufixo
    '_juris_corrigido'.
    """
    if verbose:
        print(f"[INFO] Lendo arquivo: {input_file}")

    with open(input_file, mode='r', encoding='utf-8', newline='') as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError("CSV vazio - nenhum cabeçalho encontrado.")

        rows = list(reader)

    idxs = localizar_indices_colunas(header)
    ementa_idx = idxs["ementa_idx"]
    juris_idx = idxs["juris_idx"]

    if verbose:
        print(f"[INFO] Índice da coluna 'Ementa': {ementa_idx} ('{header[ementa_idx]}').")
        if juris_idx is not None:
            print(f"[INFO] Índice da coluna 'Jurisprudência': {juris_idx} ('{header[juris_idx]}').")
        else:
            print("[WARN] Coluna 'Jurisprudência' não encontrada. Ela será criada no final.")

    # Se a coluna "Jurisprudência" não existir, adiciona ao cabeçalho
    if juris_idx is None:
        header.append("Jurisprudência")
        juris_idx = len(header) - 1

    total = len(rows)
    if verbose:
        print(f"[INFO] Total de linhas de dados: {total}")

    last_pct = -1
    new_rows = []
    for i, row in enumerate(rows, start=1):
        ementa_texto = row[ementa_idx] if len(row) > ementa_idx else ""

        nova_juris = extrair_jurisprudencia_corrigida(ementa_texto)

        # Garante que a linha tem colunas suficientes
        if len(row) <= juris_idx:
            # completa com strings vazias até chegar em juris_idx
            row = row + [""] * (juris_idx - len(row) + 1)

        row[juris_idx] = nova_juris
        new_rows.append(row)

        if verbose and total > 0:
            pct = int((i / total) * 100)
            if pct // 5 != last_pct // 5:
                print(f"[PROGRESS] Corrigindo linha {i}/{total} ({pct}%)...")
            last_pct = pct

    if verbose:
        print(f"[INFO] Salvando arquivo corrigido em: {output_file}")

    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(new_rows)

    if verbose:
        print("[SUCESSO] Coluna 'Jurisprudência' atualizada: Temas sem pontos e sem prefixo 'STF'.")
        print(f"[SUCESSO] Arquivo gerado: {output_file}")


def main():
    try:
        input_csv = detectar_csv_processado()

        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_juris_corrigido{ext}"

        corrigir_jurisprudencia_csv(input_csv, output_csv, verbose=True)

    except Exception as e:
        print("[ERRO] Falha ao corrigir jurisprudência:")
        print(e)


if __name__ == "__main__":
    main()