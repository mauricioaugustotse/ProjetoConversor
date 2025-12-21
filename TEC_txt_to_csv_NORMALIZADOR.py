import pandas as pd
import re
import os

def corrigir_bloco_texto(texto_bloco):
    """
    Junta linhas de um bloco de texto (enunciado ou alternativa) que foram 
    quebradas no meio de uma frase. A heurística é substituir uma quebra de linha 
    seguida por uma letra minúscula por um espaço.
    """
    if not isinstance(texto_bloco, str):
        return texto_bloco
        
    # Garante que as quebras de linha sejam uniformes (padrão Unix \n)
    texto_bloco = texto_bloco.replace('\r\n', '\n')
    
    # Regex: Substitui uma quebra de linha `\n` (e possíveis espaços) 
    # seguida por uma letra minúscula (incluindo acentuadas) por um espaço.
    texto_corrigido = re.sub(r'\n\s*(?=[a-zà-ú])', ' ', texto_bloco, flags=re.UNICODE)
    
    return texto_corrigido.strip()

def normalizar_questao(corpo):
    """
    Normaliza o texto da coluna 'questao' com regras estritas:
      - Colapsa quebras indevidas dentro de parágrafos.
      - Coloca cada item romano (i/ii/III/IV...) e cada "( ) ..." em linha própria.
      - Insere exatamente 1 linha em branco entre:
          * narrativa ↔ primeiro enunciado (I./( )) e
          * comando explícito (linha que termina com ':') ↔ alternativas.
      - Mantém 1 linha por alternativa (a), b) ...), sem quebras internas.
      - NÃO 'inventa' comando: só reconhece se a última linha não vazia antes das alternativas termina com ':'.
    """
    import re

    if not isinstance(corpo, str):
        return corpo

    t = corpo.replace('\r\n', '\n').strip()

    # ---- Separa bloco anterior às alternativas e o bloco de alternativas ----
    m = re.search(r'(?mi)^\s*[a-e]\)', t)
    if m:
        antes_alts = t[:m.start()].rstrip()
        alts_txt   = t[m.start():].strip()
    else:
        antes_alts = t
        alts_txt   = ""

    # ---------- ENUNCIADO/NARRATIVA/COMANDO (antes das alternativas) ----------
    # (1) Normalização básica de parágrafos
    S = antes_alts
    S = re.sub(r'\n{3,}', '\n\n', S)                   # no máx. parágrafo duplo
    S = re.sub(r'(?<!\n)\n(?!\n)', ' ', S)             # quebra simples -> espaço
    S = re.sub(r'[ \t]{2,}', ' ', S).strip()           # espaços duplos+

    # (1.1) Conserta dupla quebra no meio de frase (ex.: "pensão por morte\n\ncom base")
    S = re.sub(r'([^\.\!\?\:])\n\n(?=[a-zà-ú])', r'\1 ', S, flags=re.UNICODE)

    # (2) Em linha própria: romanos e "( ) ..."
    roman_enum = r'(?:[ivxlcdm]{1,6})'  # cobre I..XX e mais (case-insensitive)
    # Espaço -> nova linha antes de item romano
    S = re.sub(rf'(?<!\n)\s+(?={roman_enum}[\)\.\:](?:\s|$))', '\n', S, flags=re.IGNORECASE)
    # Espaço -> nova linha antes de "( )"
    S = re.sub(r'(?<!\n)\s+(?=\(\s*\))', '\n', S)

    # (2.1) GARANTE 1 linha em branco ANTES do PRIMEIRO enunciado (roman OU "( )")
    m_first_roman = re.search(rf'(?m)^\s*{roman_enum}[\)\.\:]\s', S, flags=re.IGNORECASE)
    m_first_vf    = re.search(r'(?m)^\s*\(\s*\)', S)
    # escolhe o mais cedo (se existir)
    first_idx = None
    if m_first_roman:
        first_idx = m_first_roman.start()
    if m_first_vf and (first_idx is None or m_first_vf.start() < first_idx):
        first_idx = m_first_vf.start()

    if first_idx is not None:
        pre  = S[:first_idx].rstrip()
        rest = S[first_idx:].lstrip()
        if pre:  # só insere linha em branco se houver narrativa anterior
            S = pre + '\n\n' + rest
        else:
            S = rest  # já inicia com enunciado

    # (2.2) Colapsa brancos APENAS ENTRE enunciados sucessivos (preserva o branco antes do 1º)
    pattern_between_enunciados = rf'(^\s*(?:{roman_enum}[\)\.\:]\s|\(\s*\)).*?)\n\s*\n+(?=^\s*(?:{roman_enum}[\)\.\:]\s|\(\s*\)))'
    S = re.sub(pattern_between_enunciados, r'\1\n', S, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    # (3) COMANDO explícito: última linha não vazia terminando com ':'
    linhas = S.split('\n')
    while linhas and not linhas[-1].strip():
        linhas.pop()
    i_last = len(linhas) - 1
    if i_last >= 0 and linhas[i_last].rstrip().endswith(':'):
        # remove brancos imediatamente antes
        j = i_last - 1
        while j >= 0 and not linhas[j].strip():
            linhas.pop(j)
            i_last -= 1
            j -= 1
        # insere 1 linha em branco antes do comando (se houver conteúdo antes)
        if i_last > 0:
            linhas.insert(i_last, '')
    S = '\n'.join(linhas).strip()

    # ---------- ALTERNATIVAS ----------
    alternativas_norm = ""
    if alts_txt:
        linhas = alts_txt.split('\n')
        alts, atual = [], ""
        for linha in linhas:
            if re.match(r'^\s*[a-eA-E]\)', linha):  # nova alternativa
                if atual:
                    alts.append(atual.strip())
                atual = linha.strip()
            else:
                if linha.strip():
                    atual += ' ' + linha.strip()
        if atual:
            alts.append(atual.strip())

        alternativas_norm = '\n'.join(alts)  # uma linha por alternativa

    # ---------- JUNÇÃO (exatamente 1 linha em branco antes das alternativas) ----------
    return (S + ('\n\n' + alternativas_norm if alternativas_norm else '')).strip()

def normalizar_tribunal(valor_atual, valor_anterior):
    """
    Normaliza a coluna 'tribunal' com base em um conjunto de regras.
    Usa o valor da linha anterior se o valor atual for um placeholder.
    """
    valor_str = str(valor_atual).strip()

    if "Questões para concursos" in valor_str:
        return valor_anterior

    if "NeR (ENAC) ENAC" in valor_str:
        return "ENAC"
    
    if "Magis (ENAM) ENAM" in valor_str:
        return "ENAM"
    
    match_trf = re.search(r'TRF\s*(\d)', valor_str)
    if match_trf:
        return f"TRF{match_trf.group(1)}"

    match_tj_join = re.search(r'TJ\s*([A-Z]{2})', valor_str)
    if match_tj_join and "JS (" in valor_str:
         return f"TJ{match_tj_join.group(1)}"

    match_tj_find = re.search(r'\b(TJ[A-Z]{2,})\b', valor_str)
    if match_tj_find:
        return match_tj_find.group(1)

    return valor_str

def normalizar_ramo(ramo):
    """
    Normaliza a coluna 'ramo'. Se contiver URL ('www') ou uma data, esvazia a célula.
    """
    ramo_str = str(ramo).strip()

    if 'www' in ramo_str.lower():
        return ""

    is_date_like = bool(re.match(r'\d{1,2}/\d{1,2}/\d{4}', ramo_str))
    if is_date_like:
        return ""

    return ramo_str

# -------------------- Extração de cabeçalho embutido na 'questao' --------------------
def extrair_cabecalho_embutido(texto):
    """
    Detecta e extrai cabeçalho indevido embutido no início da string 'questao', com formato:
      <...>/<SIGLA>/<ANO> <RAMO> - <SUBRAMO> <corpo...>

    Critérios anti-falso-positivo (evita "Lei 13.848/2019", "CF/1988"):
      - Só considera a 1ª linha (até ~200 chars).
      - <ANO> deve vir como '/<SIGLA>/<ANO>' (SIGLA aceita letras latinas, dígitos, espaço, 'º', 'ª', '().-').
      - <RAMO> deve começar por termos esperados (Direito, Legislação, Processo, Noções, Regime, Estatuto...).
    """
    if not isinstance(texto, str):
        return None

    t = texto.strip()
    if not t:
        return None

    # 1) Só a primeira linha (até 200 chars)
    primeira_linha = t.split('\n', 1)[0]
    if len(primeira_linha) > 200:
        primeira_linha = primeira_linha[:200]

    # 2) '/<SIGLA>/<ANO>' — SIGLA começa com letra latina; pode conter letras/dígitos/espaços/() . - º ª
    LAT = r"A-Za-zÀ-ÖØ-öø-ÿ"
    # CORREÇÃO: ano = (19|20)\d{2}  (4 dígitos no total)
    padrao_sigla_ano = rf'/([{LAT}][{LAT}0-9()\.\-ºª ]{{0,40}})/((?:19|20)\d{{2}})\b'
    m = re.search(padrao_sigla_ano, primeira_linha)
    if not m:
        return None

    ano = int(m.group(2))

    # 3) Tudo antes de '/SIGLA/ANO' -> banca (primeiro segmento antes da 1ª '/')
    pos0 = t.find(m.group(0))
    prefixo = t[:pos0].strip()
    partes = [p.strip() for p in prefixo.split('/') if p.strip()]
    banca = partes[0] if partes else ''
    if len(banca) > 80:  # anti-lixo
        return None

    # 4) Após o ANO: "<RAMO> - <SUBRAMO>"
    tail = t[pos0 + len(m.group(0)):].lstrip()
    if ' - ' not in tail:
        return None

    ramo_part, sub_and_rest = tail.split(' - ', 1)
    ramo = ramo_part.strip()

    termos_ramo = (
        'Direito', 'Legislação', 'Processo', 'Noções',
        'Regime', 'Estatuto', 'Legislações', 'Normas', 'Princípios'
    )
    if not any(ramo.startswith(tk) for tk in termos_ramo):
        return None

    # 5) Delimita SUBRAMO até início do corpo (ponto/fecha-parêntese/dois-pontos + espaço + Maiúscula)
    upper_pt = r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ]"
    m_boundary = re.search(rf'(?<=[\.\)\:])\s+(?={upper_pt})', sub_and_rest)

    if not m_boundary:
        # fallback: início de romanos, V/F ou alternativas
        m_boundary = re.search(
            r'(?=(?:^\s*[ivxlcdm]{1,6}[\)\.\:]\s|^\s*\(\s*\)|^\s*[a-e]\)))',
            sub_and_rest,
            flags=re.IGNORECASE | re.MULTILINE
        )

    if m_boundary:
        subramo = sub_and_rest[:m_boundary.start()].strip()
        corpo   = sub_and_rest[m_boundary.end():].lstrip()
    else:
        subramo = sub_and_rest.strip()
        corpo   = ""

    # 6) Corpo final (fallback conservador)
    if not corpo:
        reconstruido = f"{ramo} - {sub_and_rest}"
        pos = t.find(reconstruido)
        corpo_final = t[pos + len(reconstruido):].lstrip() if pos >= 0 else t
    else:
        corpo_final = corpo

    return {
        "ano": ano,
        "banca": banca,
        "ramo": ramo,
        "subramo": subramo,
        "corpo": corpo_final
    }

# -------------------- colocação de # antes do número da questão --------------------

def fixar_hashtag_questao_tec(valor):
    """Em 'questao_tec', garante '#'+número (ex.: '12457' -> '#12457', '# 12457' -> '#12457')."""
    import re
    import pandas as pd
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return valor
    s = str(valor).strip()
    s = re.sub(r'#\s+(?=\d)', '#', s)        # remove espaço entre '#' e dígito
    s = re.sub(r'(?<!#)(\d+)', r'#\1', s, 1) # acrescenta '#' só antes do 1º grupo de dígitos
    return s

# -------------------------------------------------------------------------------------------

def processar_csv(input_filepath, output_filepath):
    """
    Função principal que carrega, processa e salva o arquivo CSV com lógica aprimorada.
    """
    if not os.path.exists(input_filepath):
        print(f"Erro: O arquivo de entrada '{input_filepath}' não foi encontrado.")
        return

    try:
        df = pd.read_csv(input_filepath)
        
        processed_data = []
        tribunal_anterior = ""

        for index, row in df.iterrows():
            new_row = row.to_dict()

            # --- Substituição global de caracteres ---
            for key, value in new_row.items():
                if isinstance(value, str):
                    # Trata os artefatos cp1252 mal decodificados:
                    #  \u0096 (), \u0094 () e \u0093 () -> ' - '
                    for bad in ('\u0096', '\u0094', '\u0093'):
                        value = value.replace(bad, ' - ')
                    new_row[key] = value
                     # --- Normalização específica: coluna 'questao_tec' ---
            if 'questao_tec' in new_row:
                new_row['questao_tec'] = fixar_hashtag_questao_tec(new_row['questao_tec'])
            
            # --- 1. EXTRAÇÃO DE CABEÇALHO EMBUTIDO NA COLUNA 'questao' (NOVA) ---
            questao_str = str(new_row.get('questao', ''))
            corpo_da_questao = questao_str
            header_from_embedded = False

            cab = extrair_cabecalho_embutido(questao_str)
            if cab:
                # Popula colunas solicitadas
                new_row['ano']     = cab['ano']
                new_row['banca']   = cab['banca']
                new_row['ramo']    = cab['ramo']
                new_row['subramo'] = cab['subramo']
                # Limpa 'questao' removendo o cabeçalho
                corpo_da_questao   = cab['corpo']
                header_from_embedded = True

            # --- 2. (ANTIGO) EXTRAÇÃO DE CABEÇALHO EM DUAS LINHAS (TRIBUNAL/ANO + RAMO/SUBRAMO) ---
            # Só tenta se NÃO encontrou cabeçalho embutido
            if not header_from_embedded:
                lines = questao_str.strip().split('\n')
                cabecalho_processado = False

                # Tenta identificar o padrão de 2 linhas (tribunal/ano + ramo/subramo)
                if len(lines) >= 2:
                    match_tribunal_ano = re.search(r'(TJ[A-Z]{2,}|TRF\d)\s*/.*?/(\d{4})', lines[0])
                    match_ramo_subramo = re.search(r'(.+?)\s*-\s*.+', lines[1])

                    if match_tribunal_ano and match_ramo_subramo:
                        new_row['tribunal'] = match_tribunal_ano.group(1)
                        new_row['ano'] = int(match_tribunal_ano.group(2))
                        new_row['ramo'] = match_ramo_subramo.group(1).lower().strip()
                        new_row['subramo'] = lines[1].strip()
                        corpo_da_questao = '\n'.join(lines[2:])
                        cabecalho_processado = True

                # Se não encontrou, tenta o padrão de 1 linha (apenas ramo/subramo)
                if not cabecalho_processado and len(lines) >= 1:
                    match_ramo_sozinho = re.search(r'^(Direito|Processo|Legislação|Regime|Estatuto|Noções)\s+.*?\s*-\s*.+', lines[0])
                    if match_ramo_sozinho:
                        match_extracao = re.search(r'(.+?)\s*-\s*.+', lines[0])
                        if match_extracao:
                            new_row['ramo'] = match_extracao.group(1).lower().strip()
                            new_row['subramo'] = lines[0].strip()
                            corpo_da_questao = '\n'.join(lines[1:])

            # --- 3. Normalização das QUEBRAS (função já ajustada anteriormente) ---
            new_row['questao'] = normalizar_questao(corpo_da_questao)

            # --- 4. Limpeza da Coluna 'subramo' se placeholder conhecido ---
            if "Questões para concursos" in str(new_row.get('subramo', '')):
                new_row['subramo'] = ""

            # --- 5. Normalização da Coluna 'ramo' ---
            new_row['ramo'] = normalizar_ramo(new_row.get('ramo', ''))

            # --- 6. Normalização da Coluna 'tribunal' (inalterado) ---
            valor_original_tribunal = str(row.get('tribunal', ''))
            new_row['tribunal'] = normalizar_tribunal(new_row.get('tribunal', ''), tribunal_anterior)
            if "Questões para concursos" not in valor_original_tribunal:
                 tribunal_anterior = new_row['tribunal']
            
            processed_data.append(new_row)

        new_df = pd.DataFrame(processed_data)
        new_df = new_df[df.columns]

        # --- 7. Formatação da coluna 'ano' para remover '.0' ---
        if 'ano' in new_df.columns:
            new_df['ano'] = pd.to_numeric(new_df['ano'], errors='coerce').fillna(0)
            new_df['ano'] = new_df['ano'].astype(int)
            new_df['ano'] = new_df['ano'].apply(lambda x: '' if x == 0 else x)

        new_df.to_csv(output_filepath, index=False, encoding='utf-8')
        
        print(f"Processamento concluído com sucesso!")
        print(f"O arquivo normalizado foi salvo em: '{output_filepath}'")

    except Exception as e:
        print(f"Ocorreu um erro durante o processamento: {e}")

# --- Execução do Script ---
if __name__ == "__main__":
    arquivo_entrada = 'questoes_compiladas_IA.csv'
    arquivo_saida = 'questoes_normalizadas.csv'
    
    processar_csv(arquivo_entrada, arquivo_saida)