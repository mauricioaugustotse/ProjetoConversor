# -*- coding: utf-8 -*-
"""
Converte atas `.docx` de sessões do TSE em CSV estruturado via OpenAI.

Fluxo de trabalho:
1. Descobre arquivos de entrada por CLI/GUI e valida dependências/chave da API.
2. Lê cada `.docx` e segmenta o conteúdo em blocos de sessão processáveis.
3. Extrai metadados da sessão (data, tipo e número) e prepara o contexto.
4. Envia blocos em paralelo para a OpenAI, com retries e controle de concorrência.
5. Pós-processa os resultados e grava um CSV por arquivo de origem.
"""
from __future__ import annotations
import re, csv, os, json, time, sys, traceback, asyncio, argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from gui_intuitiva import dedupe_files, list_files_in_directory, open_file_panel
from openai_progress_utils import (
    build_file_signature,
    make_backup,
    read_json_dict,
    same_file_signature,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
)
from openai_log_utils import configure_standard_logging, install_print_logger_bridge

# --- Bloco de Cores para o Terminal (para um log mais sofisticado) ---
class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
# --------------------------------------------------------------------

# --- Bloco de Instalação de Dependências ---
try:
    from docx import Document
except ImportError:
    raise SystemExit(f"{C.RED}ERRO: python-docx é necessário. Execute 'pip install python-docx'{C.END}")

try:
    import openai
except ImportError:
    raise SystemExit(f"{C.RED}ERRO: openai é necessário. Execute 'pip install openai'{C.END}")

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    raise SystemExit(f"{C.RED}ERRO: tqdm é necessário. Execute 'pip install tqdm'{C.END}")
# -----------------------------------------

try:
    from dotenv import load_dotenv  # opcional
    load_dotenv()
except ImportError:
    print("dotenv não instalado, pulando o carregamento de variáveis de .env")
except Exception as e:
    print(f"Erro ao carregar .env: {e}")

# --- Configuração da API OpenAI ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(f"{C.RED}A variável OPENAI_API_KEY não está definida no ambiente.{C.END}")
client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
# -----------------------------------

# --- Constantes e Listas Globais ---
OPENAI_DEFAULT_MODEL = "gpt-5-mini"
OPENAI_DEFAULT_BATCH_SIZE = 40
OPENAI_DEFAULT_MAX_WORKERS = 10
OPENAI_DEFAULT_DELAY = 0.05
OPENAI_DEFAULT_RETRIES = 3
OPENAI_DEFAULT_TIMEOUT = 45
OPENAI_DEFAULT_TARGET_RPM = 180
OPENAI_DEFAULT_RETRY_BASE_DELAY = 1.0
CHECKPOINT_VERSION = 1
LOGGER = logging.getLogger("TSJE_docx_to_csv_viaAPI")


@dataclass(frozen=True)
class OpenAIRuntimeConfig:
    model: str = OPENAI_DEFAULT_MODEL
    batch_size: int = OPENAI_DEFAULT_BATCH_SIZE
    max_workers: int = OPENAI_DEFAULT_MAX_WORKERS
    delay: float = OPENAI_DEFAULT_DELAY
    retries: int = OPENAI_DEFAULT_RETRIES
    timeout: int = OPENAI_DEFAULT_TIMEOUT
    target_rpm: int = OPENAI_DEFAULT_TARGET_RPM
    retry_base_delay: float = OPENAI_DEFAULT_RETRY_BASE_DELAY


class AsyncRequestPacer:
    def __init__(self, target_rpm: int) -> None:
        rpm = max(0, int(target_rpm))
        self._min_interval = (60.0 / float(rpm)) if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait_turn(self) -> None:
        if self._min_interval <= 0.0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                if now >= self._next_at:
                    self._next_at = now + self._min_interval
                    return
                sleep_for = max(0.0, self._next_at - now)
            if sleep_for > 0:
                await asyncio.sleep(min(sleep_for, 0.2))

MINISTROS_LISTA_OFICIAL = sorted([
    "Affonso Celso de Assis Figueiredo Júnior", "Affonso Augusto Moreira Penna Júnior",
    "Antônio de Sampaio Dória", "Armando Prado", "Arthur Quadros Collares Moreira",
    "Cândido Luiz Maria de Oliveira Filho", "Eduardo Espinola", "Francisco Carneiro Monteiro de Salles",
    "Francisco Cezário Alvim", "Hermenegildo Rodrigues de Barros", "João Chrisóstomo da Rocha Cabral",
    "João Martins de Carvalho Mourão", "José de Miranda Valverde", "José Linhares",
    "José Lisboa", "José Maria Mac Dowell da Costa", "José Ovídio Marcondes Romeiro",
    "José Soriano de Souza Filho", "Laudo Ferreira de Camargo", "Levi Fernandes Carneiro",
    "Mário da Lima Rocha", "Plínio de Castro Casado", "Prudente de Moraes Filho",
    "Renato de Carvalho Tavares"
])

MODEL_COLS = ["classe","punchline","decisão","numero_do_processo","data_da_decisão","numero_da_sessão","UF",
              "relator","parte_recorrente","parte_recorrida","interessado","inteiro_teor","ementa","quorum", "votação"]

# --- Funções Auxiliares ---
def _norm_spaces(s: str) -> str: return re.sub(r"\s{2,}", " ", (s or "").replace("\xa0", " ")).strip()

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions, deletions, substitutions = previous_row[j + 1] + 1, current_row[j] + 1, previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def find_best_match(name: str, name_list: List[str]) -> str:
    if not name or not name_list: return ""
    return min(name_list, key=lambda official_name: levenshtein_distance(name.upper(), official_name.upper()))

def normalize_date_pt(text: str) -> str:
    meses = {"JANEIRO":1,"FEVEREIRO":2,"MARÇO":3,"MARCO":3,"ABRIL":4,"MAIO":5,"JUNHO":6,"JULHO":7, "AGOSTO":8,"SETEMBRO":9,"OUTUBRO":10,"NOVEMBRO":11,"DEZEMBRO":12}
    m = re.search(r"(\d{1,2})\s*(?:DE)?\s*([A-ZÇÃÕÉÍÓÚÂÊÔÜÀ]+)\s*(?:DE)?\s*(\d{4})", (text or "").upper())
    if not m: return ""
    d, mes, ano = int(m.group(1)), m.group(2), int(m.group(3))
    mesn = meses.get(mes)
    return f"{mesn}/{d}/{ano:04d}" if mesn else ""

def normalize_date_output_mdy(text: str) -> str:
    """Normaliza datas para o formato M/D/AAAA."""
    s = _norm_spaces(str(text or ""))
    if not s:
        return ""

    meses = {"JANEIRO":1,"FEVEREIRO":2,"MARÇO":3,"MARCO":3,"ABRIL":4,"MAIO":5,"JUNHO":6,"JULHO":7, "AGOSTO":8,"SETEMBRO":9,"OUTUBRO":10,"NOVEMBRO":11,"DEZEMBRO":12}

    # Ex.: "30 de julho de 1932" -> "7/30/1932"
    m_ext = re.search(r"(\d{1,2})\s*(?:DE)?\s*([A-ZÇÃÕÉÍÓÚÂÊÔÜÀ]+)\s*(?:DE)?\s*(\d{4})", s.upper())
    if m_ext:
        day, mes, year = int(m_ext.group(1)), m_ext.group(2), int(m_ext.group(3))
        month = meses.get(mes)
        if month and 1 <= day <= 31:
            return f"{month}/{day}/{year:04d}"

    # Ex.: "1932-07-30" -> "7/30/1932"
    m_iso = re.search(r"\b(\d{4})[\\/.-](\d{1,2})[\\/.-](\d{1,2})\b", s)
    if m_iso:
        year, month, day = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{month}/{day}/{year:04d}"

    # Ex.: "30/07/1932" ou "7/30/1932" -> "7/30/1932"
    m_num = re.search(r"\b(\d{1,2})[\\/.-](\d{1,2})[\\/.-](\d{4})\b", s)
    if m_num:
        first, second, year = int(m_num.group(1)), int(m_num.group(2)), int(m_num.group(3))

        if first > 12 and 1 <= second <= 12:
            # D/M/AAAA
            day, month = first, second
        elif second > 12 and 1 <= first <= 12:
            # M/D/AAAA
            month, day = first, second
        else:
            # Ambíguo: prioriza M/D/AAAA.
            month, day = first, second

        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{month}/{day}/{year:04d}"

    return ""

def extract_sessao_num(text: str) -> str:
    uc_text = (text or "").upper()
    if "SESSÃO PREPARATÓRIA" in uc_text: return "sessão preparatória"
    if "SESSÃO DE INSTALAÇÃO" in uc_text: return "sessão de instalação"
    m_extra = re.search(r"(\d+)\s*ª\s*SESSÃO\s+EXTRAORDINÁRIA", uc_text)
    if m_extra: return f"{m_extra.group(1)}E"
    m_ord = re.search(r"(\d+)\s*ª\s*SESSÃO\s+ORDINÁRIA", uc_text)
    if m_ord: return m_ord.group(1)
    m_any = re.search(r"(\d+)\s*ª\s*SESS[ÃA]O", uc_text)
    if m_any: return m_any.group(1)
    return "sessão s/nº"

def extract_sessions_from_docx(path:Path)->list[str]:
    doc=Document(str(path));sessions,current=[],[]
    session_heading_pattern=re.compile(r"^\s*(?:ATA DA\s+)?(?:\d+ª\s*)?SESS[ÃA]O.*",re.I)
    for p in doc.paragraphs:
        t=(p.text or "").strip();
        if not t:continue
        is_sep="***"in t;is_head=session_heading_pattern.match(t)
        if(is_sep or is_head)and current:sessions.append("\n".join(current));current=[]
        if not is_sep:current.append(p.text)
    if current:sessions.append("\n".join(current))
    return[s.strip()for s in sessions if len(s.strip())>50]


def select_docx_files_gui() -> List[Path]:
    """Abre painel GUI completo e retorna os .docx escolhidos."""
    gui = open_file_panel(
        title="TSJE DOCX para CSV",
        subtitle="Selecione um ou mais arquivos .docx para processar.",
        filetypes=[("Documentos Word", "*.docx"), ("Todos os arquivos", "*.*")],
        extensions=[".docx"],
        initial_files=[],
        allow_add_dir=True,
        recursive_dir=True,
        min_files=1,
    )
    if not gui or not gui.get("confirmed"):
        return []
    files = dedupe_files(gui.get("files") or [], [".docx"])
    return [Path(p) for p in files]

async def extract_data_with_llm(
    session_index: int,
    session_text: str,
    session_date: str,
    session_num: str,
    ministros_list: List[str],
    cfg: OpenAIRuntimeConfig,
    pacer: Optional[AsyncRequestPacer],
) -> List[Dict[str, str]]:
    """Função assíncrona que envia a sessão inteira para a IA decompor e extrair."""
    ministros_prompt_part = f"O nome do relator DEVE ser um dos seguintes: [{', '.join(ministros_list)}]. Corrija para o nome mais próximo da lista." if ministros_list else ""

    date_instruction = f'Use o valor "{session_date}".'
    if not session_date:
        date_instruction = 'Extraia a data completa por extenso (ex: "30 de julho de 1932") do "TEXTO COMPLETO DA SESSÃO" e formate-a estritamente como M/D/AAAA (Mês/Dia/Ano). Por exemplo, "30 de julho de 1932" deve se tornar "7/30/1932".'

    prompt = f"""
    Sua tarefa principal é analisar o TEXTO COMPLETO DE UMA SESSÃO do Tribunal Superior Eleitoral de 1932 e decompor seu conteúdo.

    TEXTO COMPLETO DA SESSÃO:
    ---
    {session_text}
    ---
    INSTRUÇÃO PRINCIPAL:
    1. DECOMPONHA O TEXTO: Leia o texto completo e identifique CADA deliberação, processo, consulta ou ato administrativo individual. Cada assunto distinto deve se tornar um registro separado.
    2. CRIE UM OBJETO JSON PARA CADA DELIBERAÇÃO: Para cada assunto que você identificar, crie um objeto JSON.
    3. RETORNE UMA LISTA: Sua resposta final DEVE ser um objeto JSON com uma chave "deliberacoes", cujo valor é a LISTA de todos os objetos JSON que você criou.

    Para CADA objeto JSON, extraia os seguintes campos:
    - "classe": (ESSENCIAL) Use "CONSULTA", "REPRESENTAÇÃO", etc. para processos formais. Para outros atos, discussões ou deliberações, use "ATOS E EXPEDIENTES".
    - "punchline": (ESSENCIAL) Um resumo de UMA ÚNICA FRASE, clara e direta, da deliberação.
    - "decisão": (ESSENCIAL E DETALHADO) Um resumo completo e informativo, compreensível por si só.
    - "numero_do_processo": Apenas os dígitos do número do processo, se houver.
    - "data_da_decisão": (ESSENCIAL) {date_instruction}
    - "numero_da_sessão": Use o valor "{session_num}".
    - "UF": A sigla da Unidade Federativa (UF), se mencionada.
    - "relator": Identifique o nome do ministro que relata o processo, apresenta a consulta ou propõe a deliberação principal. O nome deve ser completo. {ministros_prompt_part}. Se nenhum ministro for claramente identificado nesta função, deixe este campo em branco.
    - "parte_recorrente": Identifique o nome da pessoa, partido ou entidade que inicia ou apresenta o processo (recorrente, representante, consulente, etc). Em 'ATOS E EXPEDIENTES', pode ser o proponente principal da matéria, se houver. Se não houver ou não for claro, deixe este campo em branco. Não acrescente  '(', '/', '-' ou outros detalhes ao nome detectado.
    - "parte_recorrida": Identifique o nome da pessoa, partido ou entidade contra quem o processo é apresentado (recorrido, representado, impetrado, etc). Este campo geralmente não se aplica a 'CONSULTAS' ou 'ATOS E EXPEDIENTES'. Se não houver ou não for aplicável, deixe este campo em branco. Não acrescente  '(', '/', '-' ou outros detalhes ao nome detectado.
    - "interessado": Liste os nomes de quaisquer outras partes que participam do processo mas não são recorrente ou recorrida (ex: terceiros interessados, litisconsortes, ou outras pessoas/entidades indetificadas no feito). Separe múltiplos nomes por vírgula. Se não houver, deixe este campo em branco. Não confundir com os juízes/ministros responsáveis pela deliberação.
    - "ementa": (ESSENCIAL) O trecho de texto exato e literal, extraído do "TEXTO COMPLETO DA SESSÃO" acima, que corresponde especificamente a esta única deliberação.
    - "quorum": (ESSENCIAL E CONSISTENTE) Encontre a lista de todos os ministros presentes na sessão (geralmente no início do texto) e REPITA essa mesma lista de nomes para TODOS os objetos JSON que você gerar. Todos os registros desta sessão devem ter o mesmo quorum. NECESSARIAMENTE individualize os nomes e separe múltiplos nomes por vírgula. Normalize os nomes para o formato abreviado, por exemplo, "Carvalho Mourão". Ignore títulos como "Dr.", "Sr.", "Ministro", "desembargador", etc. Todas as sessões devem contar com o nome do presidente "Hermenegildo de Barros", que sempre presidia as sessões.
    - "votação": (SE APLICÁVEL) Se o texto indicar claramente como cada ministro votou (por exemplo, "o ministro X votou contra" ou "o desembargador Y divergiu"), liste os nomes abreviados e padronizados dos que votaram contra, separados por vírgula. Caso a decisão tenha sido por unanimidade ou sem divergência, registre "Unânime". Se não houver detalhes de votação ou tratar-se de um simples registro sem cunho decisório, deixe este campo em branco.
    """
    
    max_retries = max(1, int(cfg.retries))
    for attempt in range(max_retries):
        try:
            if pacer is not None:
                await pacer.wait_turn()
            response = await client.chat.completions.create(
                model=cfg.model,
                timeout=max(5, int(cfg.timeout)),
                messages=[
                    {"role": "system", "content": "Você é um especialista em documentos jurídicos antigos do Brasil. Sua tarefa principal é decompor o texto completo de uma ata de sessão em múltiplas deliberações individuais e retornar um array JSON sob a chave 'deliberacoes'."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("deliberacoes", [])
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            delay = min(10.0, max(0.2, float(cfg.retry_base_delay)) * (2 ** attempt))
            tqdm_asyncio.write(
                f"{C.YELLOW}AVISO: Erro de conexão/timeout na API para a sessão #{session_index+1}. "
                f"Tentativa {attempt + 1}/{max_retries}. Tentando novamente em {delay:.2f}s...{C.END}"
            )
            await asyncio.sleep(delay)
        except Exception as e:
            msg = str(e).lower()
            retryable = any(tok in msg for tok in ("rate limit", "429", "timeout", "timed out", "connection", "503", "502", "500"))
            if retryable and attempt < max_retries - 1:
                delay = min(10.0, max(0.2, float(cfg.retry_base_delay)) * (2 ** attempt))
                tqdm_asyncio.write(
                    f"{C.YELLOW}AVISO: Erro transitório na API para a sessão #{session_index+1}. "
                    f"Tentativa {attempt + 1}/{max_retries}. Retry em {delay:.2f}s...{C.END}"
                )
                await asyncio.sleep(delay)
                continue
            tqdm_asyncio.write(f"{C.RED}ERRO INESPERADO na API para a sessão #{session_index+1}: {e}{C.END}")
            return [{"error": f"{type(e).__name__}: {e}", "session_text": session_text}]
    
    return [{"error": f"Erro de API: a requisição falhou após {max_retries} tentativas.", "session_text": session_text}]


async def process_session_worker(
    job: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    ministros_list: List[str],
    cfg: OpenAIRuntimeConfig,
    pacer: Optional[AsyncRequestPacer],
) -> Dict[str, Any]:
    """Worker que processa uma sessão inteira."""
    async with semaphore:
        session_text = job['session_text']
        llm_results = await extract_data_with_llm(
            job['index'],
            session_text,
            job['date'],
            job['num'],
            ministros_list,
            cfg,
            pacer,
        )
        
        # Pós-processamento para garantir quórum consistente e limpo
        if llm_results and "error" not in llm_results[0]:
            master_quorum = ""
            for res in llm_results:
                quorum_value = res.get("quorum")
                if quorum_value:
                    if isinstance(quorum_value, list):
                        # Converte lista para string
                        master_quorum = ", ".join(map(str, quorum_value))
                    else:
                        # Garante que é uma string limpa
                        master_quorum = _norm_spaces(str(quorum_value))
                    break  # Para após encontrar o primeiro quórum válido
            
            if master_quorum:
                # Aplica o quórum limpo e consistente a todos os resultados da sessão
                for res in llm_results:
                    res["quorum"] = master_quorum
        
        processed_rows = []
        for llm_data in llm_results:
            if "error" in llm_data:
                 return {"original_index": job["index"], "rows": [], "error": llm_data["error"]}

            extracted_relator = llm_data.get("relator", "")
            official_relator = find_best_match(extracted_relator, ministros_list) if extracted_relator else ""
            
            llm_raw_date = llm_data.get("data_da_decisão", "")
            final_date = normalize_date_output_mdy(llm_raw_date)
            if not final_date:
                final_date = normalize_date_output_mdy(job['date'])

            row_data = {
                "classe": llm_data.get("classe", ""),
                "punchline": llm_data.get("punchline", ""),
                "decisão": llm_data.get("decisão", ""),
                "numero_do_processo": llm_data.get("numero_do_processo", ""),
                "data_da_decisão": final_date,
                "numero_da_sessão": llm_data.get("numero_da_sessão", job['num']),
                "UF": llm_data.get("UF", ""), "relator": official_relator,
                "parte_recorrente": llm_data.get("parte_recorrente", ""),
                "parte_recorrida": llm_data.get("parte_recorrida", ""),
                "interessado": llm_data.get("interessado", ""),
                "inteiro_teor": _norm_spaces(session_text),
                "ementa": _norm_spaces(llm_data.get("ementa", "")),
                "quorum": llm_data.get("quorum", ""),
                "votação": llm_data.get("votação", ""),
            }
            processed_rows.append(row_data)

        return {"original_index": job["index"], "rows": processed_rows, "error": None}


async def process_docx_to_csv(
    input_path: Path,
    out_csv: Path,
    ministros_list: List[str],
    openai_cfg: OpenAIRuntimeConfig,
) -> None:
    started_at = time.time()
    checkpoint_path = out_csv.with_name(f".{out_csv.stem}.openai.checkpoint.json")
    report_path = out_csv.with_name(f".{out_csv.stem}.openai.report.json")
    source_sig = build_file_signature(input_path)

    print(f"{C.CYAN}  → Lendo e preparando as sessões do arquivo...{C.END}")
    sessions = extract_sessions_from_docx(input_path)

    jobs = []
    for i, sess in enumerate(sessions):
        session_date = normalize_date_pt(sess)
        session_num = extract_sessao_num(sess)
        jobs.append({"index": i, "session_text": sess, "date": session_date, "num": session_num})

    if not jobs:
        print(f"{C.YELLOW}  AVISO: Nenhuma sessão encontrada no arquivo.{C.END}")
        return

    results_by_index: Dict[int, Dict[str, Any]] = {}
    checkpoint = read_json_dict(checkpoint_path)
    cp_sig = checkpoint.get("source_signature", {})
    cp_model = str(checkpoint.get("openai_model", "")).strip()
    cp_total = int(checkpoint.get("total_sessions", 0) or 0)
    cp_results = checkpoint.get("results_by_index", {})
    if (
        int(checkpoint.get("version", 0) or 0) == CHECKPOINT_VERSION
        and same_file_signature(cp_sig, source_sig)
        and cp_model == openai_cfg.model
        and cp_total == len(jobs)
        and isinstance(cp_results, dict)
    ):
        for raw_idx, raw_result in cp_results.items():
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if not isinstance(raw_result, dict):
                continue
            rows = raw_result.get("rows", [])
            err = raw_result.get("error")
            results_by_index[idx] = {
                "original_index": idx,
                "rows": rows if isinstance(rows, list) else [],
                "error": str(err) if err else None,
            }
        if results_by_index:
            print(
                f"{C.CYAN}  -> [resume] checkpoint carregado: sessões concluídas "
                f"{len(results_by_index)}/{len(jobs)}{C.END}"
            )

    workers = min(max(1, int(openai_cfg.max_workers)), len(jobs))
    semaphore = asyncio.Semaphore(workers)
    pacer = AsyncRequestPacer(openai_cfg.target_rpm)

    def _save_progress(status: str) -> None:
        ordered = [results_by_index[idx] for idx in sorted(results_by_index.keys())]
        final_rows: List[Dict[str, Any]] = []
        errors_found = 0
        for result in ordered:
            if result.get("error"):
                errors_found += 1
            final_rows.extend(result.get("rows", []))
        write_csv_atomic(out_csv, MODEL_COLS, final_rows)
        write_json_atomic(
            checkpoint_path,
            {
                "version": CHECKPOINT_VERSION,
                "source_signature": source_sig,
                "input_docx": str(input_path.resolve()),
                "output_csv": str(out_csv.resolve()),
                "openai_model": openai_cfg.model,
                "total_sessions": len(jobs),
                "results_by_index": {
                    str(k): v for k, v in sorted(results_by_index.items(), key=lambda item: item[0])
                },
                "status": status,
                "updated_at": utc_now_iso(),
            },
        )
        write_json_atomic(
            report_path,
            {
                "script": "TSJE_docx_to_csv_viaAPI.py",
                "input_docx": str(input_path.resolve()),
                "output_csv": str(out_csv.resolve()),
                "checkpoint_file": str(checkpoint_path.resolve()),
                "status": status,
                "model": openai_cfg.model,
                "total_sessions": len(jobs),
                "done_sessions": len(results_by_index),
                "error_sessions": errors_found,
                "rows_written": len(final_rows),
                "elapsed_seconds": round(max(0.0, time.time() - started_at), 2),
                "updated_at": utc_now_iso(),
            },
        )

    pending_jobs = [job for job in jobs if job["index"] not in results_by_index]
    if not pending_jobs:
        print(f"{C.CYAN}  -> Nenhuma sessão pendente (checkpoint completo).{C.END}")
        _save_progress("completed")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"{C.GREEN}  ✔ Arquivo salvo com sucesso.{C.END}")
        return

    print(f"\n{C.BOLD}{C.BLUE}--- INICIANDO ANÁLISE CONCORRENTE DE {len(jobs)} SESSÕES ---{C.END}")
    print(
        f"{C.CYAN}  → workers={workers} | batch={openai_cfg.batch_size} | "
        f"delay={openai_cfg.delay:.2f}s | rpm={openai_cfg.target_rpm}{C.END}"
    )

    start_time = time.time()
    batch_size = max(1, int(openai_cfg.batch_size))
    total_pending = len(pending_jobs)
    for start in range(0, total_pending, batch_size):
        end = min(start + batch_size, total_pending)
        batch_jobs = pending_jobs[start:end]
        tasks = [
            process_session_worker(job, semaphore, ministros_list, openai_cfg, pacer)
            for job in batch_jobs
        ]
        batch_results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Analisando sessões pendentes {start + 1}-{end}/{total_pending}"
        )
        for result in batch_results:
            idx = int(result.get("original_index", -1))
            if idx >= 0:
                results_by_index[idx] = result
        _save_progress("running")
        if end < total_pending and openai_cfg.delay > 0:
            await asyncio.sleep(max(0.0, float(openai_cfg.delay)))
    end_time = time.time()

    print(f"\n{C.GREEN}✔ Análise de todas as sessões concluída em {end_time - start_time:.2f} segundos.{C.END}")

    results = [results_by_index[idx] for idx in sorted(results_by_index.keys())]
    final_rows = []
    errors_found = 0
    for result in results:
        if result["error"]:
            tqdm_asyncio.write(f"{C.RED}  ERRO na sessão {result['original_index']+1}: {result['error']}{C.END}")
            errors_found += 1
        final_rows.extend(result["rows"])

    print(f"{C.CYAN}  → Salvando {len(final_rows)} registros em {C.BOLD}{out_csv}{C.END}...")
    if errors_found > 0:
        print(f"{C.YELLOW}  AVISO: {errors_found} sessões falharam e não foram incluídas no CSV.{C.END}")

    write_csv_atomic(out_csv, MODEL_COLS, final_rows)
    final_status = "completed" if len(results_by_index) >= len(jobs) else "partial"
    _save_progress(final_status)
    if final_status == "completed" and checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"{C.GREEN}  ✔ Arquivo salvo com sucesso.{C.END}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSJE DOCX -> CSV via OpenAI.")
    parser.add_argument("--input-files", nargs="*", default=[], help="Arquivos .docx especificos.")
    parser.add_argument("--input-dirs", nargs="*", default=[], help="Pastas para buscar .docx.")
    parser.add_argument("--openai-model", default=OPENAI_DEFAULT_MODEL)
    parser.add_argument("--openai-batch-size", type=int, default=OPENAI_DEFAULT_BATCH_SIZE)
    parser.add_argument("--openai-max-workers", type=int, default=OPENAI_DEFAULT_MAX_WORKERS)
    parser.add_argument("--openai-delay", type=float, default=OPENAI_DEFAULT_DELAY)
    parser.add_argument("--openai-retries", type=int, default=OPENAI_DEFAULT_RETRIES)
    parser.add_argument("--openai-timeout", type=int, default=OPENAI_DEFAULT_TIMEOUT)
    parser.add_argument("--openai-target-rpm", type=int, default=OPENAI_DEFAULT_TARGET_RPM)
    parser.add_argument("--openai-retry-base-delay", type=float, default=OPENAI_DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--verbose", action="store_true", help="Exibe logs detalhados.")
    parser.add_argument("--quiet", action="store_true", help="Exibe apenas avisos/erros.")
    parser.add_argument("--debug", action="store_true", help="Ativa modo debug técnico.")
    parser.add_argument("--log-file", default="", help="Arquivo opcional para salvar logs.")
    parser.add_argument("--no-gui", action="store_true", help="Desativa o painel GUI.")
    return parser


def discover_docx_inputs(input_files: List[str], input_dirs: List[str]) -> List[Path]:
    found: List[str] = []
    for folder in input_dirs or []:
        found.extend(list_files_in_directory(folder, [".docx"], recursive=True))
    found.extend(input_files or [])
    return [Path(p) for p in dedupe_files(found, [".docx"])]


async def main():
    args = build_arg_parser().parse_args()
    logger = configure_standard_logging(
        "TSJE_docx_to_csv_viaAPI",
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
    )
    install_print_logger_bridge(globals(), logger)
    openai_cfg = OpenAIRuntimeConfig(
        model=str(args.openai_model or OPENAI_DEFAULT_MODEL).strip() or OPENAI_DEFAULT_MODEL,
        batch_size=max(1, int(args.openai_batch_size)),
        max_workers=max(1, int(args.openai_max_workers)),
        delay=max(0.0, float(args.openai_delay)),
        retries=max(1, int(args.openai_retries)),
        timeout=max(5, int(args.openai_timeout)),
        target_rpm=max(0, int(args.openai_target_rpm)),
        retry_base_delay=max(0.1, float(args.openai_retry_base_delay)),
    )
    start_total_time = time.time()
    print(f"{C.BOLD}{C.BLUE}="*70)
    print(f"AUTOMATIZADOR DE ANÁLISE DE ATAS JURÍDICAS v27.0 (GUI de seleção)")
    print(f"{C.BLUE}-{C.END}"*70)

    script_dir = Path(__file__).parent
    ministros_list = MINISTROS_LISTA_OFICIAL
    print(f"{C.GREEN}INFO: Usando lista interna com {len(ministros_list)} nomes de ministros.{C.END}")

    docx_files = discover_docx_inputs(list(args.input_files or []), list(args.input_dirs or []))
    if not docx_files and not args.no_gui:
        print(f"{C.CYAN}INFO: Abrindo interface para seleção de arquivos .docx...{C.END}")
        docx_files = select_docx_files_gui()
    if not docx_files:
        print(f"{C.YELLOW}AVISO: Nenhum arquivo .docx selecionado.{C.END}"); return

    print(f"{C.GREEN}INFO: {len(docx_files)} arquivo(s) .docx selecionado(s).{C.END}")
    
    for idx, docx_path in enumerate(docx_files):
        print(f"\n{C.BOLD}{C.CYAN}Iniciando processamento do arquivo {idx+1}/{len(docx_files)}: {docx_path.name}{C.END}")
        csv_path = script_dir / f"{docx_path.stem}.csv"
        checkpoint_path = csv_path.with_name(f".{csv_path.stem}.openai.checkpoint.json")
        if csv_path.exists() and not checkpoint_path.exists():
            backup = make_backup(csv_path, label="startup_backup")
            if backup is not None:
                print(f"{C.CYAN}  -> Backup inicial criado: {backup}{C.END}")
        try:
            await process_docx_to_csv(docx_path, csv_path, ministros_list, openai_cfg)
        except Exception as e:
            print(f"\n{C.RED}ERRO INESPERADO NO FLUXO PRINCIPAL: {e}{C.END}")
            traceback.print_exc()
    
    end_total_time = time.time()
    print(f"\n{C.BLUE}-{C.END}"*70)
    print(f"{C.BOLD}{C.GREEN}Processamento de todos os arquivos concluído!{C.END}")
    print(f"Tempo total de execução: {end_total_time - start_total_time:.2f} segundos.")
    print(f"{C.BOLD}{C.BLUE}="*70 + C.END)

if __name__ == "__main__":
    asyncio.run(main())
