# -*- coding: utf-8 -*-
"""
Converte arquivos `.txt` de Temas Selecionados em CSV via OpenAI.

Fluxo de trabalho:
1. Lê `*.txt` de `--input-dir` e/ou `--input-files` e carrega taxonomia de subramos.
2. Separa o conteúdo em julgados e aplica retomada por checkpoint por arquivo.
3. Para cada julgado, chama a OpenAI e valida o JSON com casos extraídos.
4. Corrige subramos com a taxonomia e acumula registros estruturados.
5. Salva CSV/checkpoint/report na pasta do script e remove checkpoint ao concluir com sucesso.
"""

import os
import re
import json
import pandas as pd
import openai
import time
import logging
import argparse
import random
import threading
from datetime import date
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Set, Optional
import traceback

from gui_intuitiva import dedupe_files, list_files_in_directory, open_file_panel
from openai_progress_utils import (
    build_file_signature,
    make_backup,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
)
from openai_log_utils import configure_standard_logging, install_print_logger_bridge

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_DEFAULT_BATCH_SIZE = 40
OPENAI_DEFAULT_MAX_WORKERS = 10
OPENAI_DEFAULT_MAX_WORKERS_CAP = 14
OPENAI_DEFAULT_DELAY = 0.05
OPENAI_DEFAULT_RETRIES = 3
OPENAI_DEFAULT_TIMEOUT = 45
OPENAI_DEFAULT_TARGET_RPM = 180
CHECKPOINT_VERSION = 1
LOGGER = logging.getLogger("TEMAS_SELC_txt_to_csv")
SCRIPT_DIR = Path(__file__).resolve().parent

# -------------------- Regex de final de julgado ------------------------------
PADRAO_FINAL_JULGADO = re.compile(
    r"""
    \(
      \s*
      (?:Ac\.|Res\.|EDcl|Embargos?|AgR(?:-[A-Za-z]+)?|MS|REspE?|REspe|REsp|RO-?El|RMS|RCEd|AE|TutCautAnt|MC|AAg|Rec\.?)
      [\s\S]*?
      rel\.\s*(?:Min\.|Des\.|Juiz(?:a)?|in\.)[\s\S]*?
    \)
    \s*
    """,
    re.IGNORECASE | re.VERBOSE
)

# -------------------- Prompt para IA (v6) -----------------------------------
PROMPT_EXTRACAO = """
Você é um assistente de jurimetria especializado em jurisprudência eleitoral brasileira.
Sua tarefa é extrair informações de um trecho de um julgado do TSE.

REGRA GERAL:
1.  Analise a citação final do julgado. Retorne uma LISTA de objetos JSON, um para cada processo encontrado. Cada objeto deve ter: "tipo_de_processo", "numero_processo", "relator", "data_julgamento". Se houver apenas um, retorne uma lista com um único objeto. PREENCHA TODOS OS CAMPOS PARA CADA CASO.

REGRAS ESPECÍFICAS (SCHEMA JSON):
1.  **tipo_de_processo**: A classificação processual do expediente (CAMPO CRÍTICO). É uma sigla (ex: "AREspE", "ED-REspe", "AgRgAg", "AgRgREspe") que aparece próximo ao número do processo na referência final. (Tipo: string)
2.  **numero_processo**: O número do processo. (CAMPO CRÍTICO). Procure por um padrão numérico, como "060023641", "82911", etc. Está na referência final entre parênteses. (Tipo: string)
3.  **relator**: O nome do relator ou relatora da decisão. Sempre usar "Min." antes do nome (ex.: Min. Luiz Fux). (Tipo: string)
4.  **data_julgamento**: A data do julgamento, no formato M/D/AAAA. (Tipo: string)
5.  **subramo**: Liste os subramos jurídicos discutidos, com base na taxonomia fornecida. Separe-os por vírgulas. (Tipo: string)
6.  **contexto**: Descreva o caso concreto, com menção à legislação citada e à jurisprudência/precedentes mencionados. (Tipo: string)
7.  **tese**: Extraia a tese jurídica principal ou a 'ratio decidendi', com a referência à legislação e à jurisprudência adotada. (Tipo: string)
8.  **bullet_points**: Liste os principais pontos e fundamentos legais em 3 a 5 bullets. (Tipo: string)
9.  **punchline**: Crie uma frase de efeito, curta e de alto impacto, resumindo a decisão. (Tipo: string)
10. **tema**: O título principal que resume o assunto tratado em até 20 palavras. (Tipo: string)
11. **texto_original**: O texto exato do julgado. (Tipo: string)

# REGRAS DE FORMATAÇÃO E EXCEÇÕES
1. Preste atenção especial para distinguir "contexto" (a história do caso) de "tese" (a decisão/regra jurídica).

# REGRAS DE FOCO ESPECIAL
- Os campos `tipo_de_processo`, `numero_processo`, `relator` e `data_julgamento` são muito importantes. Eles quase sempre estão juntos na referência final do julgado, dentro dos parênteses. Analise essa referência com atenção máxima.
- Se você não encontrar um valor claro para eles, e apenas para eles, faça uma segunda verificação no texto antes de retornar `null`.

Julgado a ser analisado:
---
{texto_julgado}
---

Retorne a resposta APENAS em formato JSON.
Exemplo de saída para um julgado com múltiplas citações:
{{
  "casos": [
    {{
      "tipo_de_processo": "AgR-MS",
      "numero_processo": "57264",
      "relator": "Marcelo Ribeiro",
      "data_julgamento": "5/12/2011"
    }},
    {{
      "tipo_de_processo": "MS",
      "numero_processo": "168383",
      "relator": "Cármen Lúcia",
      "data_julgamento": "2/14/2012"
    }}
  ],
  "tema": "Mitigação de prazos em eleições suplementares",
  "subramo": ["Prazo processual"],
  "contexto": "Descrição do caso...",
  "tese": "A tese jurídica firmada foi...",
  "bullet_points": "• Ponto 1\\n• Ponto 2",
  "punchline": "Frase de efeito.",
  "texto_original": "Texto exato do julgado..."
}}
"""

# -------------------- Funções de Checkpoint -------------------------

def carregar_checkpoint(filepath: str) -> Tuple[Set[int], List[Dict[str, Any]]]:
    """
    Carrega estado do checkpoint.
    Compatível com o formato antigo (ultimo_julgado_processado_idx) e novo (done_indices).
    """
    if not os.path.exists(filepath):
        return set(), []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  -> AVISO: Não foi possível ler o checkpoint '{filepath}'. Começando do zero. Erro: {e}")
        return set(), []

    registros_raw = data.get('registros_salvos', [])
    registros = registros_raw if isinstance(registros_raw, list) else []
    done_indices: Set[int] = set()

    raw_done = data.get("done_indices", [])
    if isinstance(raw_done, list):
        for idx in raw_done:
            try:
                i = int(idx)
            except Exception:
                continue
            if i >= 0:
                done_indices.add(i)
    else:
        try:
            ultimo_indice = int(data.get('ultimo_julgado_processado_idx', -1))
        except Exception:
            ultimo_indice = -1
        if ultimo_indice >= 0:
            done_indices = set(range(0, ultimo_indice + 1))

    if done_indices:
        print(
            f"  -> Checkpoint encontrado. Julgados já processados: {len(done_indices)} "
            f"(último índice: {max(done_indices) + 1})."
        )
    else:
        print("  -> Checkpoint encontrado, mas sem índices concluídos. Retomando do início.")
    return done_indices, registros


def salvar_checkpoint(filepath: str, done_indices: Set[int], registros: List[Dict[str, Any]]) -> None:
    """Salva progresso atual em checkpoint."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            max_done = max(done_indices) if done_indices else -1
            checkpoint_data = {
                'version': CHECKPOINT_VERSION,
                'done_indices': sorted(done_indices),
                'ultimo_julgado_processado_idx': max_done,
                'registros_salvos': registros,
                'updated_at': utc_now_iso(),
            }
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"  -> ERRO: Falha ao salvar o checkpoint em '{filepath}'. Erro: {e}")

# -------------------- Funções Auxiliares ------------------------------------

def carregar_taxonomia(filepath: str) -> Dict[str, str]:
    """Lê o arquivo taxonomia.json e cria um mapa de 'filho' -> 'mãe'."""
    child_to_parent_map = {}
    print(f"Carregando taxonomia de '{filepath}'...")

    def _traverse_json(data_node, parent=None):
        if isinstance(data_node, dict):
            for key, value in data_node.items():
                if parent:
                    child_to_parent_map[key] = parent
                _traverse_json(value, parent=key)
        elif isinstance(data_node, list):
            if parent:
                for child in data_node:
                    child_to_parent_map[child] = parent

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            taxonomia_data = json.load(f)
        _traverse_json(taxonomia_data)
    except FileNotFoundError:
        print(f"AVISO: Arquivo de taxonomia '{filepath}' não encontrado.")
        return {}
    except json.JSONDecodeError:
        print(f"AVISO: Erro ao decodificar o arquivo JSON '{filepath}'.")
        return {}

    if not child_to_parent_map:
        print(f"AVISO: Nenhum mapeamento de taxonomia foi carregado.")
    else:
        print(f"Carregamento da taxonomia concluído. {len(child_to_parent_map)} mapeamentos carregados.")
    return child_to_parent_map

def corrigir_subramo_com_taxonomia(subramos: List[str], taxonomia_mapa: Dict[str, str]) -> List[List[str]]:
    """Corrige a lista de subramos usando o mapa de taxonomia."""
    if not isinstance(subramos, list): return []
    subramos_corrigidos, pais_ja_incluidos = [], set()
    for subramo in subramos:
        pai = taxonomia_mapa.get(subramo)
        item_final = [pai, subramo] if pai else [subramo]
        if pai: pais_ja_incluidos.add(pai)
        if not (len(item_final) == 1 and item_final[0] in pais_ja_incluidos):
            subramos_corrigidos.append(item_final)
    return [list(item) for item in set(map(tuple, [tuple(i) for i in subramos_corrigidos]))]

def extrair_julgados(texto: str) -> List[str]:
    """Divide o texto em julgados individuais."""
    julgados, ultimo_ponto_corte = [], 0
    for match in PADRAO_FINAL_JULGADO.finditer(texto):
        ponto_corte_atual = match.end()
        julgado_completo = texto[ultimo_ponto_corte:ponto_corte_atual].strip()
        if julgado_completo: julgados.append(julgado_completo)
        ultimo_ponto_corte = ponto_corte_atual
    return [j for j in julgados if j]


def gravar_csv_resultado_atomic(saida_csv: str, registros_finais: List[Dict[str, Any]]) -> int:
    col_order_lower = [
        "tipo_de_processo", "numero_processo", "relator", "data_julgamento",
        "tema", "subramo", "contexto", "tese", "bullet_points", "punchline", "texto_original"
    ]
    if not registros_finais:
        write_csv_atomic(Path(saida_csv), col_order_lower, [])
        return 0

    df = pd.DataFrame(registros_finais)
    for col in col_order_lower:
        if col not in df.columns:
            df[col] = None
    if "data_julgamento" in df.columns:
        df["data_julgamento"] = df["data_julgamento"].apply(normalizar_data_para_mdy)
    df = df[col_order_lower + [c for c in df.columns if c not in col_order_lower]]
    rows = df.to_dict(orient="records")
    write_csv_atomic(Path(saida_csv), list(df.columns), rows)
    return len(rows)


def normalizar_data_para_mdy(value: Any) -> str:
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    if not raw:
        return ""
    text = re.sub(r"(?i)^dje\s+de\s+", "", raw).strip()

    match_iso = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", text)
    if match_iso:
        year, month, day = map(int, match_iso.groups())
        try:
            date(year, month, day)
            return f"{month}/{day}/{year}"
        except ValueError:
            return raw

    match = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b", text)
    if not match:
        return raw

    a, b, year = map(int, match.groups())
    mdy_ok = True
    dmy_ok = True
    try:
        date(year, a, b)
    except ValueError:
        mdy_ok = False
    try:
        date(year, b, a)
    except ValueError:
        dmy_ok = False

    if a > 12 and dmy_ok:
        return f"{b}/{a}/{year}"
    if b > 12 and mdy_ok:
        return f"{a}/{b}/{year}"
    if dmy_ok:
        return f"{b}/{a}/{year}"
    if mdy_ok:
        return f"{a}/{b}/{year}"
    return raw


@dataclass(frozen=True)
class OpenAIRuntimeConfig:
    model: str = "gpt-5-mini"
    batch_size: int = OPENAI_DEFAULT_BATCH_SIZE
    max_workers: int = OPENAI_DEFAULT_MAX_WORKERS
    max_workers_cap: int = OPENAI_DEFAULT_MAX_WORKERS_CAP
    delay: float = OPENAI_DEFAULT_DELAY
    retries: int = OPENAI_DEFAULT_RETRIES
    timeout: int = OPENAI_DEFAULT_TIMEOUT
    target_rpm: int = OPENAI_DEFAULT_TARGET_RPM


class RequestPacer:
    def __init__(self, target_rpm: int) -> None:
        rpm = max(0, int(target_rpm))
        self._min_interval = (60.0 / float(rpm)) if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait_turn(self) -> None:
        if self._min_interval <= 0.0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_at:
                    self._next_at = now + self._min_interval
                    return
                sleep_for = max(0.0, self._next_at - now)
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.2))


def _classify_openai_error(exc: Exception) -> str:
    msg = str(exc or "").lower()
    if "rate limit" in msg or "429" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg:
        return "connection"
    if any(tok in msg for tok in ("500", "502", "503", "504", "overloaded", "bad gateway")):
        return "upstream"
    if any(tok in msg for tok in ("api key", "authentication", "unauthorized", "forbidden", "401", "403")):
        return "auth"
    return "unknown"


def _retry_wait(attempt: int, kind: str) -> float:
    base = 1.0 if kind in {"rate_limit", "timeout", "connection", "upstream"} else 0.5
    jitter = random.uniform(0.0, 0.3)
    return min(10.0, (base * (2 ** max(0, attempt - 1))) + jitter)


def _parse_and_normalize_json_response(raw_text: str) -> Dict:
    """Localiza, analisa e normaliza um JSON dentro de um texto."""
    if not raw_text: return {}
    json_str = ""
    if "```json" in raw_text:
        try: json_str = raw_text.split("```json")[1].split("```")[0].strip()
        except IndexError: pass
    if not json_str:
        start, end = raw_text.find('{'), raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = raw_text[start:end+1]
        else:
            print("      -> AVISO: Nenhum objeto JSON encontrado na resposta da IA.")
            return {}
    try: data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"      -> AVISO: Falha no parsing do JSON. Erro: {e}. Conteúdo: '{json_str[:200]}...'")
        return {}
    if not isinstance(data, dict):
        print(f"      -> AVISO: Resposta da IA não é um dicionário. Tipo: {type(data)}")
        return {}
    def normalize_keys(obj: Any) -> Any:
        if isinstance(obj, dict): return {str(k).strip().lower(): normalize_keys(v) for k, v in obj.items()}
        if isinstance(obj, list): return [normalize_keys(elem) for elem in obj]
        return obj
    normalized_data = normalize_keys(data)
    if 'casos' in normalized_data:
        casos_value = normalized_data['casos']
        if isinstance(casos_value, dict): normalized_data['casos'] = [casos_value]
        elif not isinstance(casos_value, list):
            print(f"      -> AVISO: O campo 'casos' não é válido. Removendo. Conteúdo: {casos_value}")
            normalized_data.pop('casos')
    return normalized_data

def extrair_dados_com_ia(
    julgado: str,
    *,
    cfg: OpenAIRuntimeConfig,
    pacer: Optional[RequestPacer],
) -> Dict[str, Any]:
    """Envia um julgado para a API e retorna os dados extraídos."""
    mensagens = [
        {"role": "system", "content": "Você é um assistente de jurimetria especializado em jurisprudência eleitoral brasileira."},
        {"role": "user", "content": PROMPT_EXTRACAO.format(texto_julgado=julgado)}
    ]
    total_attempts = max(1, int(cfg.retries))
    last_error = ""
    for attempt in range(1, total_attempts + 1):
        try:
            if pacer is not None:
                pacer.wait_turn()
            payload = {
                "model": cfg.model,
                "messages": mensagens,
                "response_format": {"type": "json_object"},
                "timeout": max(5, int(cfg.timeout)),
            }
            try:
                response = openai.chat.completions.create(**payload)
            except TypeError:
                # Compatibilidade com SDKs que não aceitam timeout por parâmetro.
                payload.pop("timeout", None)
                response = openai.chat.completions.create(**payload)
            raw_response = response.choices[0].message.content or ""
            if raw_response:
                return _parse_and_normalize_json_response(raw_response)
            last_error = "Resposta vazia da API."
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            kind = _classify_openai_error(exc)
            retryable = kind in {"rate_limit", "timeout", "connection", "upstream", "unknown"}
            if attempt >= total_attempts or not retryable:
                break
            time.sleep(_retry_wait(attempt, kind))
    if last_error:
        print(f"      -> ERRO: Chamada à API falhou após {total_attempts} tentativa(s): {last_error}")
    return {}

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Temas Selecionados TXT -> CSV via OpenAI.")
    parser.add_argument("--input-dir", default=".", help="Pasta com arquivos .txt.")
    parser.add_argument("--input-files", nargs="*", default=[], help="Arquivos .txt especificos.")
    parser.add_argument("--taxonomia", default="taxonomia.json", help="Arquivo JSON de taxonomia.")
    parser.add_argument("--pause-sec", type=float, default=None, help="(Legado) pausa entre lotes.")
    parser.add_argument("--openai-model", default="gpt-5-mini")
    parser.add_argument("--openai-batch-size", type=int, default=OPENAI_DEFAULT_BATCH_SIZE)
    parser.add_argument("--openai-max-workers", type=int, default=OPENAI_DEFAULT_MAX_WORKERS)
    parser.add_argument("--openai-max-workers-cap", type=int, default=OPENAI_DEFAULT_MAX_WORKERS_CAP)
    parser.add_argument("--openai-delay", type=float, default=OPENAI_DEFAULT_DELAY)
    parser.add_argument("--openai-retries", type=int, default=OPENAI_DEFAULT_RETRIES)
    parser.add_argument("--openai-timeout", type=int, default=OPENAI_DEFAULT_TIMEOUT)
    parser.add_argument("--openai-target-rpm", type=int, default=OPENAI_DEFAULT_TARGET_RPM)
    parser.add_argument("--verbose", action="store_true", help="Exibe logs detalhados.")
    parser.add_argument("--quiet", action="store_true", help="Exibe apenas avisos/erros.")
    parser.add_argument("--debug", action="store_true", help="Ativa modo debug técnico.")
    parser.add_argument("--log-file", default="", help="Arquivo opcional para salvar logs.")
    parser.add_argument("--no-gui", action="store_true", help="Desativa painel GUI.")
    return parser


def resolve_input_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return str(path.resolve())


def maybe_collect_gui_inputs(args):
    input_dir_norm = os.path.normpath(str(args.input_dir or "."))
    if args.no_gui or args.input_files or input_dir_norm not in (".", ""):
        return args

    gui = open_file_panel(
        title="Temas Selecionados - TXT para CSV",
        subtitle="Selecione os arquivos .txt de Temas Selecionados.",
        filetypes=[("TXT", "*.txt"), ("Todos os arquivos", "*.*")],
        extensions=[".txt"],
        initial_files=[],
        allow_add_dir=True,
        recursive_dir=True,
        min_files=1,
        output_label="",
        extra_bools=[
            ("verbose", "Logs detalhados", bool(args.verbose)),
            ("quiet", "Apenas avisos/erros", bool(args.quiet)),
            ("debug", "Debug técnico", bool(args.debug)),
        ],
        extra_texts=[
            ("taxonomia", "Arquivo de taxonomia (.json)", str(args.taxonomia or "taxonomia.json")),
            ("openai_model", "Modelo OpenAI", str(args.openai_model or "gpt-5-mini")),
            ("openai_workers", "OpenAI workers", str(args.openai_max_workers)),
            ("openai_batch", "OpenAI batch size", str(args.openai_batch_size)),
            ("openai_delay", "OpenAI delay entre lotes (s)", str(args.openai_delay)),
            ("log_file", "Arquivo de log (opcional)", str(args.log_file or "")),
        ],
    )
    if not gui or not gui.get("confirmed"):
        return args
    files = list(gui.get("files") or [])
    if files:
        args.input_files = files
    texts = gui.get("texts") or {}
    args.taxonomia = str(texts.get("taxonomia", args.taxonomia)).strip() or args.taxonomia
    try:
        args.openai_max_workers = max(
            1, int(str(texts.get("openai_workers", args.openai_max_workers)).strip() or args.openai_max_workers)
        )
    except Exception:
        pass
    try:
        args.openai_batch_size = max(
            1, int(str(texts.get("openai_batch", args.openai_batch_size)).strip() or args.openai_batch_size)
        )
    except Exception:
        pass
    try:
        args.openai_delay = max(
            0.0, float(str(texts.get("openai_delay", args.openai_delay)).strip() or args.openai_delay)
        )
    except Exception:
        pass
    args.openai_model = str(texts.get("openai_model", args.openai_model)).strip() or args.openai_model
    args.log_file = str(texts.get("log_file", args.log_file)).strip()
    bools = gui.get("bools") or {}
    args.verbose = bool(bools.get("verbose", args.verbose))
    args.quiet = bool(bools.get("quiet", args.quiet))
    args.debug = bool(bools.get("debug", args.debug))
    return args


def discover_input_files(input_dir: str, input_files: List[str]) -> List[str]:
    found: List[str] = []
    if input_dir and os.path.isdir(input_dir):
        found.extend(list_files_in_directory(input_dir, [".txt"], recursive=False))
    found.extend(input_files or [])
    files = dedupe_files(found, [".txt"])
    files = [f for f in files if os.path.basename(f).lower() != "taxonomia.txt"]
    return files


# -------------------- Main (com lógica de checkpoint corrigida) -----------------------
def main():
    """Função principal para orquestrar o processo."""
    args = build_arg_parser().parse_args()
    args = maybe_collect_gui_inputs(args)
    logger = configure_standard_logging(
        "TEMAS_SELC_txt_to_csv",
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
    )
    install_print_logger_bridge(globals(), logger)

    effective_delay = max(0.0, float(args.openai_delay))
    if args.pause_sec is not None:
        effective_delay = max(0.0, float(args.pause_sec))
    openai_cfg = OpenAIRuntimeConfig(
        model=str(args.openai_model or "gpt-5-mini").strip() or "gpt-5-mini",
        batch_size=max(1, int(args.openai_batch_size)),
        max_workers=max(1, int(args.openai_max_workers)),
        max_workers_cap=max(1, int(args.openai_max_workers_cap)),
        delay=effective_delay,
        retries=max(1, int(args.openai_retries)),
        timeout=max(5, int(args.openai_timeout)),
        target_rpm=max(0, int(args.openai_target_rpm)),
    )
    taxonomia_mapa = carregar_taxonomia(resolve_input_path(args.taxonomia))
    resolved_input_dir = resolve_input_path(args.input_dir)
    resolved_input_files = [resolve_input_path(p) for p in list(args.input_files or [])]
    arquivos_txt = discover_input_files(resolved_input_dir, resolved_input_files)
    if not arquivos_txt:
        print("Nenhum arquivo .txt encontrado para processamento.")
        return

    for arquivo_txt in arquivos_txt:
        print(f"\n--- Processando arquivo: {arquivo_txt} ---")
        input_path = Path(arquivo_txt).resolve()
        base_name = input_path.stem
        saida_csv = str((SCRIPT_DIR / f"{base_name}.csv").resolve())
        checkpoint_file = str((SCRIPT_DIR / f".{base_name}.checkpoint.json").resolve())
        report_file = str((SCRIPT_DIR / f".{base_name}.openai.report.json").resolve())
        file_started_at = time.time()
        source_sig = build_file_signature(input_path)

        def _write_report(status: str, *, rows_written: Optional[int] = None) -> None:
            write_json_atomic(
                Path(report_file),
                {
                    "script": "TEMAS_SELC_txt_to_csv.py",
                    "input_file": str(input_path),
                    "output_csv": str(Path(saida_csv).resolve()),
                    "checkpoint_file": str(Path(checkpoint_file).resolve()),
                    "source_signature": source_sig,
                    "status": status,
                    "openai_model": openai_cfg.model,
                    "total_julgados": len(julgados) if "julgados" in locals() else 0,
                    "done_julgados": len(done_indices) if "done_indices" in locals() else 0,
                    "registros_extraidos": len(registros_finais) if "registros_finais" in locals() else 0,
                    "rows_written": rows_written if rows_written is not None else 0,
                    "elapsed_seconds": round(max(0.0, time.time() - file_started_at), 2),
                    "updated_at": utc_now_iso(),
                },
            )
        
        # LÓGICA DE RETOMADA CORRIGIDA:
        # Pula o arquivo SOMENTE se o CSV de resultado já existe E não há um checkpoint.
        # A ausência do checkpoint indica que o processo foi concluído com sucesso.
        if os.path.exists(saida_csv) and not os.path.exists(checkpoint_file):
            print(f"  -> AVISO: Arquivo de resultado '{saida_csv}' já foi gerado com sucesso anteriormente. Pulando...")
            _write_report("skipped_existing")
            continue

        try:
            with input_path.open("r", encoding="utf-8") as f:
                conteudo = f.read()
        except Exception as e:
            print(f"  -> ERRO: Não foi possível ler o arquivo '{arquivo_txt}'. Erro: {e}")
            continue

        if not conteudo or not conteudo.strip():
            print(f"  -> AVISO: O arquivo '{arquivo_txt}' está vazio. Pulando...")
            _write_report("empty_input")
            continue
        
        julgados = extrair_julgados(conteudo)
        print(f"Encontrados {len(julgados)} julgados para análise.")
        if os.path.exists(saida_csv) and os.path.exists(checkpoint_file):
            backup = make_backup(Path(saida_csv), label="resume_backup")
            if backup is not None:
                print(f"  -> Backup de retomada criado: {backup}")
        
        # Carrega o progresso anterior (se houver)
        done_indices, registros_finais = carregar_checkpoint(checkpoint_file)
        pending_indices = [i for i in range(len(julgados)) if i not in done_indices]
        _write_report("running", rows_written=(len(registros_finais) if registros_finais else 0))

        if not pending_indices:
            print("  -> Nenhum julgado pendente neste arquivo.")
        else:
            workers = min(
                max(1, int(openai_cfg.max_workers)),
                max(1, int(openai_cfg.max_workers_cap)),
                len(pending_indices),
            )
            pacer = RequestPacer(openai_cfg.target_rpm)
            print(
                f"  -> OpenAI: pendentes={len(pending_indices)} | workers={workers} "
                f"| batch={openai_cfg.batch_size} | rpm={openai_cfg.target_rpm}"
            )

            def _worker(idx: int) -> Tuple[int, Dict[str, Any]]:
                return idx, extrair_dados_com_ia(julgados[idx], cfg=openai_cfg, pacer=pacer)

            for start in range(0, len(pending_indices), max(1, int(openai_cfg.batch_size))):
                end = min(start + max(1, int(openai_cfg.batch_size)), len(pending_indices))
                lote_indices = pending_indices[start:end]
                print(f" - Processando lote {start + 1}-{end}/{len(pending_indices)}...")
                lote_resultados: Dict[int, Dict[str, Any]] = {}

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    fut_map = {ex.submit(_worker, idx): idx for idx in lote_indices}
                    for fut in as_completed(fut_map):
                        idx = fut_map[fut]
                        try:
                            idx_result, dados = fut.result()
                            lote_resultados[idx_result] = dados
                        except Exception as e:  # noqa: BLE001
                            print(f"   -> ERRO CRÍTICO no Julgado #{idx + 1}: {type(e).__name__}: {e}")
                            traceback.print_exc()
                            lote_resultados[idx] = {}

                for i in sorted(lote_resultados.keys()):
                    dados = lote_resultados.get(i, {})
                    casos_processados_neste_julgado = 0
                    if not dados or 'casos' not in dados or not isinstance(dados.get('casos'), list) or not dados['casos']:
                        print(f"   -> AVISO Julgado #{i + 1}: nenhum caso válido extraído.")
                        if dados:
                            print(f"      -> Debug: Chaves recebidas: {list(dados.keys())}")
                    else:
                        dados_comuns = {k: v for k, v in dados.items() if k != 'casos'}
                        for caso_especifico in dados['casos']:
                            if not isinstance(caso_especifico, dict):
                                continue
                            novo_registro = dados_comuns.copy()
                            novo_registro.update(caso_especifico)
                            subramo_val = novo_registro.get("subramo", [])
                            if isinstance(subramo_val, str):
                                novo_registro["subramo"] = [subramo_val]
                            elif not isinstance(subramo_val, list):
                                novo_registro["subramo"] = []
                            novo_registro["subramo"] = corrigir_subramo_com_taxonomia(
                                novo_registro.get("subramo", []), taxonomia_mapa
                            )
                            novo_registro["data_julgamento"] = normalizar_data_para_mdy(
                                novo_registro.get("data_julgamento", "")
                            )
                            registros_finais.append(novo_registro)
                            casos_processados_neste_julgado += 1
                        print(f"   -> OK Julgado #{i + 1} | Extraídos {casos_processados_neste_julgado} caso(s).")

                    done_indices.add(i)
                    salvar_checkpoint(checkpoint_file, done_indices, registros_finais)
                    _write_report("running", rows_written=(len(registros_finais) if registros_finais else 0))

                rows_snapshot = gravar_csv_resultado_atomic(saida_csv, registros_finais) if registros_finais else 0
                _write_report("running", rows_written=rows_snapshot)
                if end < len(pending_indices) and openai_cfg.delay > 0:
                    time.sleep(max(0.0, float(openai_cfg.delay)))

        if not registros_finais:
            print(f"Nenhum dado foi extraído com sucesso do arquivo '{arquivo_txt}'.")
            # Se não há registros, mas o checkpoint existe, remove-o para evitar confusão.
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            _write_report("empty_output", rows_written=0)
            continue

        rows_written = gravar_csv_resultado_atomic(saida_csv, registros_finais)
        print(f"\nArquivo '{saida_csv}' gerado com sucesso!")
        _write_report("completed", rows_written=rows_written)

        # Limpa o arquivo de checkpoint após o sucesso
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

if __name__ == "__main__":
    main()
