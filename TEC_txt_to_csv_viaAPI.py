# -*- coding: utf-8 -*-
"""
Este script processa arquivos de texto (.txt) contendo questões de concurso,
extrai informações estruturadas, enriquece os dados com uma IA generativa
e salva o resultado em um único arquivo CSV.

"""
import os
import re
import csv
import json
import time
import traceback
from typing import List, Dict, Any, Tuple, Optional

try:
    from dotenv import load_dotenv  # opcional
    load_dotenv()
except ImportError:
    print("dotenv não instalado, pulando o carregamento de variáveis de .env")
except Exception as e:
    print(f"Erro ao carregar .env: {e}")

# --- CONFIGURAÇÕES ---

INPUT_DIR = 'arquivos_txt'
OUTPUT_FILE = 'questoes_compiladas_IA.csv'
CSV_HEADER = [
    'questao_TEC', 'ano', 'banca', 'tribunal', 'gabarito',
    'ramo', 'subramo', 'questao', 'punchline', 'bullet_points', 'tema_comum'
]

# --- CONFIGURAÇÕES DA IA ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODELO_IA = "gpt-5-mini"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 3

# Pausa curta entre chamadas, apenas para suavizar tráfego
PAUSA_ENTRE_CHAMADAS_S = 0.25

# Verbosidade (apenas sinalização; não altera fluxo/resultado)
VERBOSE = True

# Arquivo de checkpoint (backup)
CHECKPOINT_FILE = OUTPUT_FILE + ".checkpoint.json"

# --- DEPENDÊNCIAS ---
try:
    import openai
except ImportError:
    print("\033[91m\033[1mERRO: Dependências não encontradas.\033[0m")
    print("\033[93mPor favor, execute o comando no seu terminal: pip install openai\033[0m")
    openai = None

# --- EXCEÇÕES ESPECÍFICAS (se disponíveis nesta versão do SDK) ---
try:
    from openai import APIError, RateLimitError, APITimeoutError
except Exception:
    APIError = RateLimitError = APITimeoutError = Exception

# --- CLASSES DE ESTILO ---
class C:
    BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
    RED = '\033[91m'; END = '\033[0m'; BOLD = '\033[1m'

def vprint(msg: str):
    if VERBOSE:
        print(msg, flush=True)

# ==============================================================================
# FUNÇÕES DE PROCESSAMENTO DE IA
# ==============================================================================

def _chunked(seq, size):
    """Divide uma sequência em pedaços de um tamanho específico."""
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def _montar_prompt_lote(lote):
    """
    Gera um prompt compacto pedindo um ÚNICO objeto JSON com campo "itens": [...]
    Cada item deve conter: id, punchline, bullet_points, tema_comum.
    (AJUSTE: inclui 'gabarito' em cada item e reforça que a alternativa CORRETA é a fornecida.)
    """
    blocos = []
    for q in lote:
        texto_questao_escapado = q.get("questao", "").replace('"""', '\\"\\"\\"')
        gabarito_val = str(q.get("gabarito", "")).strip()
        blocos.append(
            f'{{"id":"{q.get("questao_TEC","")}", "gabarito":"{gabarito_val}", "texto":"""{texto_questao_escapado}"""}}'
        )
    itens = ",\n".join(blocos)
    return (
        'Analise juridicamente cada questão a seguir e RETORNE um ÚNICO objeto JSON com o formato:\n'
        '{ "itens": [ {"id": "...", "punchline": "...", "bullet_points": "...", "tema_comum": "..."} ] }\n'
        "REGRAS:\n"
        "- 'id' deve repetir exatamente o id recebido.\n"
        "- ATENÇÃO: considere EXCLUSIVAMENTE como correta a alternativa indicada em 'gabarito' recebida no LOTE; NÃO deduza o gabarito por conta própria e NÃO o altere.\n"
        "- 'punchline': Elabore uma punchline técnica, clara e precisa, em um parágrafo de 60 a 100 palavras. Use linguagem técnico-jurídica formal. Sintetize a explicação jurídica da alternativa CORRETA (conforme 'gabarito') e a razão da inadequação das demais alternativas incorretas; evite metáforas.\n"
        "- 'bullet_points': Crie bullet points sobre o contexto e a alternativa correta da questão. Acrescente bullets que facilitem a distinção com as outras alternativas, as quais são necessariamente incorretas. Use o caractere '•' para cada ponto. Cada ponto deve ter no máximo 20 palavras.\n"
        "- 'tema_comum': Gere uma LISTA de 1 a 3 rótulos canônicos, separados por vírgulas (formato multiselect do Notion). NÃO repita o macrotema já extraído como 'ramo' nem use macrotemas genéricos ('Direito Administrativo', 'Direito Constitucional', etc.). Use termos estáveis, no singular, sem sinônimos/abreviações/nomes próprios e sem pontuação final; capitalização consistente (iniciais maiúsculas, exceto preposições). Exemplos: 'Ações Possessórias'; 'Licitações, Dispensa'; 'Direitos Fundamentais, Direito à Intimidade'.\n"
        "- Não inclua nada além do JSON final.\n"
        "LOTE:\n"
        f'{{"itens_de_entrada":[\n{itens}\n]}}'
    )

def _chamada_openai_sincrona(prompt_usuario: str):
    """
    Chamada síncrona com response_format JSON (estilo adotado previamente).
    """
    mensagens = [
        {"role": "system", "content": "Você é um assistente especialista em análise jurídico-concursal. Responda exclusivamente com JSON válido que obedeça ao formato solicitado."},
        {"role": "user", "content": prompt_usuario}
    ]
    openai.api_key = OPENAI_API_KEY
    resp = openai.chat.completions.create(
        model=MODELO_IA,
        messages=mensagens,
        response_format={"type": "json_object"},
    )
    return resp

def _extrair_campos_do_conteudo(conteudo: str) -> Dict[str, Dict[str, str]]:
    """
    Recebe o conteúdo textual (JSON) da IA e devolve um dicionário:
      { id: {"punchline":..., "bullet_points":..., "tema_comum":...}, ... }
    """
    dados = json.loads(conteudo)
    saida = {}
    for item in dados.get("itens", []):
        item_id = item.get("id")
        if item_id:
            saida[item_id] = {
                "punchline": item.get("punchline", ""),
                "bullet_points": item.get("bullet_points", ""),
                "tema_comum": item.get("tema_comum", "")
            }
    return saida

# -------------------- Funções de Checkpoint (backup) -------------------------

def carregar_checkpoint(filepath: str) -> Tuple[int, List[Dict]]:
    """
    Carrega o estado de um arquivo de checkpoint.
    Retorna o índice do próximo item a ser processado e a lista de registros já salvos.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ultimo_indice = data.get('ultimo_item_processado_idx', -1)
                registros = data.get('registros_salvos', [])
                print(f"  -> Checkpoint encontrado. Retomando da questão #{ultimo_indice + 2}.")
                return ultimo_indice + 1, registros
        except (json.JSONDecodeError, IOError) as e:
            print(f"  -> AVISO: Não foi possível ler o checkpoint '{filepath}'. Começando do zero. Erro: {e}")
            return 0, []
    return 0, []

def salvar_checkpoint(filepath: str, indice: int, registros: List[Dict]):
    """Salva o progresso atual em um arquivo de checkpoint."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            checkpoint_data = {
                'ultimo_item_processado_idx': indice,
                'registros_salvos': registros
            }
            json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"  -> ERRO: Falha ao salvar o checkpoint em '{filepath}'. Erro: {e}")

def _aplicar_registros_salvos(lista_questoes: List[Dict[str, Any]], registros_salvos: List[Dict[str, Any]]):
    """Aplica os campos já salvos (do checkpoint) na lista de questões atual."""
    if not registros_salvos:
        return 0
    mapa_salvos = {str(r.get('questao_TEC','')): r for r in registros_salvos if r.get('questao_TEC') is not None}
    aplicados = 0
    for i, q in enumerate(lista_questoes):
        qid = str(q.get('questao_TEC',''))
        if qid in mapa_salvos:
            # Atualiza apenas campos de IA (mantém demais intactos)
            campos = mapa_salvos[qid]
            for k in ('punchline','bullet_points','tema_comum'):
                if k in campos:
                    q[k] = campos[k]
            aplicados += 1
    return aplicados

def processar_questoes_com_ia(lista_questoes: List[Dict[str, Any]]):
    """Processamento sequencial com verbose e checkpoint (preservado)."""
    if not openai or not OPENAI_API_KEY:
        print(f"\n{C.RED}{C.BOLD}ERRO CRÍTICO: Chave da API não configurada ou dependências ausentes.{C.END}")
        return

    total_itens = len(lista_questoes)
    print(f"\n{C.CYAN}{C.BOLD}🤖 Enriquecendo {total_itens} questões (modelo {MODELO_IA})...{C.END}")
    print(f"{C.CYAN}   Modo sequencial • verbose por item • backoff com checkpoint{C.END}")

    # Carrega progresso anterior, se houver
    indice_inicial, registros_salvos = carregar_checkpoint(CHECKPOINT_FILE)

    # Aplica campos já salvos à lista atual
    if registros_salvos:
        aplicados = _aplicar_registros_salvos(lista_questoes, registros_salvos)
        if aplicados:
            print(f"  -> {aplicados} registros aplicados a partir do checkpoint.")

    # Mapeia id -> posição
    index_por_id = {str(q.get("questao_TEC", "")): i for i, q in enumerate(lista_questoes)}

    # Inicia registros finais com os salvos
    registros_finais = list(registros_salvos)

    preenchidas = 0
    for q in lista_questoes[:indice_inicial]:
        if q.get('punchline') and "[ERRO" not in q['punchline']:
            preenchidas += 1

    for i in range(indice_inicial, total_itens):
        q = lista_questoes[i]
        q_id = str(q.get("questao_TEC", "")).strip()
        print(f" - Processando Questão #{i + 1}/{total_itens} (id={q_id})...")
        prompt = _montar_prompt_lote([q])  # mantém EXATAMENTE o mesmo prompt
        delay = INITIAL_RETRY_DELAY

        sucesso = False
        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                resp = _chamada_openai_sincrona(prompt)
                dt = time.perf_counter() - t0
                conteudo = resp.choices[0].message.content
                campos_por_id = _extrair_campos_do_conteudo(conteudo)

                if q_id in campos_por_id:
                    lista_questoes[i].update(campos_por_id[q_id])
                    preenchidas += 1
                    print(f"{C.GREEN}   -> OK Questão #{i + 1} em {dt:.2f}s.{C.END}")
                    sucesso = True
                    break
                else:
                    if campos_por_id:
                        primeiro_id, campos = next(iter(campos_por_id.items()))
                        lista_questoes[i].update(campos)
                        preenchidas += 1
                        print(f"{C.YELLOW}   -> AVISO: id retornado='{primeiro_id}' difere do esperado='{q_id}'. Campos aplicados. ({dt:.2f}s){C.END}")
                        sucesso = True
                        break
                    raise ValueError("JSON retornado sem 'itens' válidos para esta questão.")

            except (APITimeoutError, APIError, RateLimitError) as e:
                print(f"{C.YELLOW}   -> AVISO: {type(e).__name__} na tentativa {tentativa}/{MAX_RETRIES}. Aguardando {delay:.2f}s...{C.END}")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"{C.YELLOW}   -> AVISO: Erro inesperado ({type(e).__name__}) na tentativa {tentativa}/{MAX_RETRIES}. Aguardando {delay:.2f}s...{C.END}")
                time.sleep(delay)
                delay *= 2

        if not sucesso:
            print(f"{C.RED}   -> ERRO IA: Não foi possível enriquecer a Questão #{i + 1}. Campos marcados como [ERRO IA].{C.END}")
            lista_questoes[i].update({
                "punchline": "[ERRO IA]",
                "bullet_points": "[ERRO IA]",
                "tema_comum": "[ERRO IA]"
            })

        # Atualiza/Anexa aos salvos e persiste checkpoint
        if q_id:
            registros_finais = [r for r in registros_finais if str(r.get('questao_TEC','')) != q_id]
        registros_finais.append(dict(lista_questoes[i]))
        salvar_checkpoint(CHECKPOINT_FILE, i, registros_finais)

        time.sleep(PAUSA_ENTRE_CHAMADAS_S)

    print(f"{C.GREEN}✓ Análise por IA concluída ({preenchidas}/{total_itens} preenchidas com sucesso).{C.END}")

# ==============================================================================
# FUNÇÕES DE PARSING E MANIPULAÇÃO DE ARQUIVOS  (CORREÇÃO PONTUAL NA EXTRAÇÃO)
# ==============================================================================

def parse_question_block(block_text: str, forced_id: Optional[str] = None):
    """
    Analisa um bloco de texto contendo uma única questão e extrai os dados.
    Retorna um dicionário com os dados ou None se o bloco for inválido.

    CORREÇÃO PONTUAL:
    - Aceita 'forced_id' extraído da própria URL do bloco (padrão Tec Concursos),
      garantindo que o campo 'questao_TEC' seja preenchido mesmo quando o ID
      não aparece como primeira linha do bloco.
    - Ajuste específico para arquivos em que a LINHA DE ASSUNTO já vem como
      "Direito ... - Subtema ...": nesses casos, 'subramo' passa a ser a
      LINHA COMPLETA (ex.: "Direito Administrativo - Do Procedimento ..."),
      preservando 'ramo' apenas com o macrotema em minúsculas.
    """
    data: Dict[str, Any] = {}
    lines = [line.strip() for line in block_text.strip().split('\n') if line.strip()]
    if len(lines) < 3:
        return None  # Bloco muito curto para ser uma questão válida.

    # 1) questao_TEC
    if forced_id:
        data['questao_TEC'] = forced_id
    else:
        match_id = re.match(r'(\d+)', lines[0])
        if not match_id:
            return None
        data['questao_TEC'] = match_id.group(1)

    # 2) banca, tribunal, ano
    info_line_idx = 0
    ano_re = re.compile(r'(\d{4})$')
    for idx in (0, 1):
        if idx < len(lines) and ano_re.search(lines[idx]):
            info_line_idx = idx
            break
    info_line = lines[info_line_idx]
    ano_match = ano_re.search(info_line)
    ano = ano_match.group(1) if ano_match else ''
    banca_tribunal_part = info_line[:ano_match.start()].strip('/') if ano_match else info_line
    parts = banca_tribunal_part.split(' - ', 1)
    data['banca'] = parts[0].strip()
    data['tribunal'] = parts[1].strip().replace('/', ' ') if len(parts) > 1 else ''
    data['ano'] = ano

    # 3) ramo e subramo
    subject_idx = info_line_idx + 1
    if subject_idx >= len(lines):
        return None
    subject_line = lines[subject_idx]

    # Ajuste: se começar com "Direito " e tiver " - ", subramo = linha completa
    if subject_line.startswith("Direito ") and " - " in subject_line:
        left, right = subject_line.split(" - ", 1)
        data['ramo'] = left.strip().lower()
        data['subramo'] = subject_line.strip()
    else:
        subject_parts = subject_line.split(' - ', 1)
        if len(subject_parts) == 2:
            data['ramo'] = subject_parts[0].strip().lower()
            data['subramo'] = subject_parts[1].strip()
        else:
            data['ramo'] = subject_line.strip().lower()
            data['subramo'] = ''

    # 4) texto da questão + gabarito (filtros de rodapé/paginação)
    question_text_lines: List[str] = []
    gabarito_found = False
    for raw in lines[subject_idx+1:]:
        if raw.lower().startswith('gabarito:'):
            data['gabarito'] = raw.split(':', 1)[-1].strip()
            gabarito_found = True
            break
        if re.search(r'\d{2}/\d{2}/\d{4}.*,?\s*Tec Concursos', raw):
            continue
        if 'https://www.tecconcursos.com.br/questoes/cadernos' in raw:
            continue
        if re.match(r'^\d+\)\s*$', raw):
            continue
        question_text_lines.append(raw)

    if not gabarito_found:
        return None

    data['questao'] = '\n'.join(question_text_lines).strip()
    data.update({'punchline': '', 'bullet_points': '', 'tema_comum': ''})
    return data

def main():
    print(f"{C.BOLD}{C.BLUE}{'='*80}{C.END}")
    print(f"{C.BOLD}{C.BLUE}  EXTRATOR E ANALISADOR DE QUESTÕES COM IA v3.0 (Lotes){C.END}")
    print(f"{C.BOLD}{C.BLUE}{'='*80}{C.END}\n")
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Pasta '{INPUT_DIR}' criada. Adicione seus arquivos .txt e rode novamente.")
        return

    # Lista os .txt de entrada (com tratamento de erros)
    try:
        txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    except OSError as e:
        print(f"Erro ao acessar a pasta '{INPUT_DIR}': {e}")
        return

    if not txt_files:
        print(f"Nenhum arquivo .txt encontrado na pasta '{INPUT_DIR}'.")
        print("Por favor, adicione seus arquivos e tente novamente.")
        return

    all_questions_data: List[Dict[str, Any]] = []

    # Padrão da URL/ID do Tec Concursos
    url_id_pattern = re.compile(r'www\.tecconcursos\.com\.br/questoes/(\d+)')

    print("Iniciando o processamento dos arquivos de texto...")
    for filename in txt_files:
        filepath = os.path.join(INPUT_DIR, filename)
        print(f"  - Lendo '{filename}'...")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"    -> Aviso: Falha ao ler '{filename}' como UTF-8. Tentando com 'latin-1'...")
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                print(f"    -> Erro: Não foi possível ler o arquivo '{filename}' com nenhuma codificação: {e}")
                continue
        except Exception as e:
            print(f"    -> Erro ao ler o arquivo '{filename}': {e}")
            continue

        matches = list(url_id_pattern.finditer(content))
        if not matches:
            print(f"    -> Aviso: Nenhuma URL de questão encontrada em '{filename}'. O arquivo pode estar em formato inesperado.")
            continue

        for idx, m in enumerate(matches):
            qid = m.group(1)
            start = m.end()
            end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(content)
            block = content[start:end]

            parsed_data = parse_question_block(block, forced_id=qid)
            if parsed_data:
                all_questions_data.append(parsed_data)
            else:
                first_line = block.strip().split('\n')[0] if block.strip() else ''
                print(f"    -> Aviso: Bloco com id={qid} em '{filename}' não pôde ser processado e foi ignorado. Início: '{first_line[:80]}...'")

    if not all_questions_data:
        print("\nNenhuma questão válida foi extraída.")
        return
    print(f"\n{C.GREEN}✓ {len(all_questions_data)} questões extraídas com sucesso.{C.END}")

    # Retomada: se já existe OUTPUT_FILE e não há checkpoint, pula IA
    if os.path.exists(OUTPUT_FILE) and not os.path.exists(CHECKPOINT_FILE):
        print(f"{C.YELLOW}AVISO: Arquivo de resultado '{OUTPUT_FILE}' já existe e não há checkpoint pendente. Pulando etapa de IA...{C.END}")
    else:
        # Enriquecimento por IA com checkpoint (preservado)
        processar_questoes_com_ia(all_questions_data)

    print(f"\n{C.CYAN}💾 Escrevendo {len(all_questions_data)} questões no arquivo '{OUTPUT_FILE}'...{C.END}")
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
            writer.writeheader()
            writer.writerows(all_questions_data)
        print(f"\n{C.GREEN}{C.BOLD}--- Processo Concluído com Sucesso! ---{C.END}")
        print(f"O arquivo '{OUTPUT_FILE}' foi gerado na mesma pasta deste script.")
        # Limpa o arquivo de checkpoint após sucesso
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    except IOError as e:
        print(f"\n{C.RED}ERRO ao escrever no arquivo '{OUTPUT_FILE}': {e}{C.END}")
    except Exception as e:
        print(f"\n{C.RED}{C.BOLD}Ocorreu um erro inesperado:{C.END}")
        print(f"{C.RED}{traceback.format_exc()}{C.END}")

if __name__ == "__main__":
    if openai:
        main()
