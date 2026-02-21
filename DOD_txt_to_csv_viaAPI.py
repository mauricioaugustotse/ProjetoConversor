"""
Converte informativos jurídicos em `.txt` para `.csv` via OpenAI.

Fluxo de trabalho:
1. Carrega a chave da API e escolhe prompt conforme tribunal (auto/STF/STJ).
2. Divide o texto em blocos de julgados por regex de citação final.
3. Envia cada bloco para a API, valida o JSON e normaliza campos de lista.
4. Consolida os registros extraídos e ordena colunas do esquema.
5. Gera um CSV por arquivo de entrada (processamento em lote via glob).
"""

import os
import re
import json
import time
import glob
import logging
import argparse
import random
import threading
from datetime import date
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Any, Dict, Pattern, Optional

import pandas as pd
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

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

# ---- OpenAI (compatível com versões antigas e novas) ----
try:
    from dotenv import load_dotenv  # opcional
    load_dotenv()
except ImportError:
    print("dotenv não instalado, pulando o carregamento de variáveis de .env")
except Exception as e:
    print(f"Erro ao carregar .env: {e}")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não encontrado nas variáveis de ambiente (.env).")

SCRIPT_DIR = Path(__file__).resolve().parent

OPENAI_DEFAULT_BATCH_SIZE = 40
OPENAI_DEFAULT_MAX_WORKERS = 10
OPENAI_DEFAULT_MAX_WORKERS_CAP = 14
OPENAI_DEFAULT_DELAY = 0.05
OPENAI_DEFAULT_RETRIES = 3
OPENAI_DEFAULT_TIMEOUT = 45
OPENAI_DEFAULT_TARGET_RPM = 180
CHECKPOINT_VERSION = 1
LOGGER = logging.getLogger("DOD_txt_to_csv_viaAPI")

# Tenta importar a nova versão da biblioteca openai, senão usa a antiga
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
except ImportError:
    import openai
    openai.api_key = OPENAI_API_KEY
    client = None # Flag para usar a sintaxe antiga


# ===================== PROMPTS =====================
PROMPT_STF = """
# PAPEL E OBJETIVO
Você é um assistente de jurimetria treinado para analisar informativos do STF e extrair metadados estruturados em JSON.

# TAREFA
Analise o texto a seguir (um bloco do informativo) e extraia os dados abaixo.
Se o bloco contiver mais de um acórdão (ex.: duas linhas de referência final distintas, como “RE 1.057.258/MG” e “RE 1.037.396/SP”), RETORNE uma LISTA de objetos JSON, um por acórdão.
Se houver apenas um acórdão, retorne um ÚNICO objeto JSON.

# ESQUEMA (CHAVES E TIPOS)
Cada objeto deve conter EXATAMENTE estas chaves:
- "tipo_de_processo": string | null                      # Classe processual (ex.: "ADI", "ADPF", "RE", "ARE")
- "numero_processo": string | null                       # Apenas número (ex.: "7.459/ES" será "7.459", "631.363/SP" será "631.363", "1.057.258/MG" será "1.057.258")
- "UF_Origem_do_caso": string | null                     # Sigla da UF ou origem do caso (ex.: "ES", "MG", "SP", "DF"); null se não houver
- "tribunal": "STF" | "STJ" | "TST" | "TSE" | "STM" | null
- "orgao_julgador": string | null                        # Plenário, 1ª Turma, 2ª Turma etc.
- "relator": string | null                               # Ex.: "Min. Dias Toffoli"
- "data_julgamento": string | null                       # Formato M/D/AAAA
- "informativo": string | null                           # Ex.: "Info 1184"
- "titulo": string | null                                # Título principal (até 20 palavras)
- "tema_repercussao": string | null                      # Ex.: "Tema 284", "Tema 533", "Tema 987" (se houver)
- "contexto": string | null                              # Um resumo robusto do caso prático ou da situação que deu origem à discussão, com menção à legislação citada e à jurisprudência/precedentes mencionados
- "tese": string | null                                  # Regra jurídica/ratio decidendi (clareza e concisão)
- "modulacao": string | null                             # Se houver modulação/limitação de efeitos, resuma; senão null
- "dispositivo": string | null                           # Comandos finais (se existentes); senão null
- "referencias_legais": array of string                  # Normas citadas normalizadas (ex.: "CF/88, art. 170"; "Lei 12.965/2014, art. 19")
- "precedentes_citados": array of string                 # Ex.: "ADPF 165", "RE 631.363". Liste aqui os precedentes mencionados que serviram como panorama/fundamento do caso, explicando-o brevemente no contexto do julgado principal.
- "ramo_direito": string | null                           # Ex.: "Direito Constitucional", "Direito Civil", "Direito Administrativo", "Direito Penal", "Direito Tributário" etc.

# REGRAS DE NORMALIZAÇÃO
- Extraia "tipo_de_processo" e "numero_processo" a partir da(s) linha(s) de referência final do STF (que começam com "STF. ...").
- "tribunal" e "orgao_julgador" saem da mesma linha (ex.: "STF. Plenário. ...").
- Converta datas para M/D/AAAA.
- "informativo": normalize como "Info NNNN" quando houver "(Info NNNN)".
- "tema_repercussao": capte "Tema NNN" quando houver “(Repercussão Geral – Tema NNN)”.
- "referencias_legais": normalize diploma+dispositivo (ex.: "CF/88, art. 5º, XXXV"; "Lei 12.965/2014, art. 19").
- "precedentes_citados": inclua apenas referências explícitas.
- Se não houver informação inequívoca para uma chave, use null (exceto arrays, que devem ser []).

# REGRAS DE FOCO ESPECIAL
- Os campos "tipo_de_processo", "numero_processo", "tribunal", "orgao_julgador", "relator", "data_julgamento", "informativo" e "UF_Origem_do_caso" são muito importantes. Eles quase sempre estão juntos na referência final do julgado. Analise essa referência com atenção máxima.
- Se você não encontrar um valor claro para eles, e apenas para eles, faça uma segunda verificação no texto antes de retornar `null`.

# DIFERENÇA ENTRE "CONTEXTO" E "TESE"
- "contexto" = narrativa fático-processual (o que motivou a controvérsia).
- "tese"     = a regra/ratio decidendi aplicada pelo STF, preferindo formulação geral.

# FORMATO DE SAÍDA
- Se houver mais de um acórdão no bloco: retorne uma LISTA de objetos.
- Caso contrário: retorne um ÚNICO objeto.
- SEM texto fora do JSON.
- A saída DEVE ser JSON válido.

--- TEXTO PARA ANÁLISE ---
{TEXTO_PARA_ANALISE}
--- FIM ---
"""

PROMPT_STJ = """
# PAPEL E OBJETIVO
Você é um assistente de jurimetria treinado para analisar informativos do STJ e extrair metadados estruturados em JSON.

# TAREFA
Analise o texto a seguir (um bloco do informativo) e extraia os dados abaixo.
Se o bloco contiver mais de um acórdão (ex.: várias linhas de referência final do tipo “STJ. 2ª Turma. REsp 2.029.719-RJ, ... (Info 857).”), RETORNE uma LISTA de objetos JSON, um por acórdão.
Se houver apenas um acórdão, retorne um ÚNICO objeto JSON.

# ESQUEMA (CHAVES E TIPOS)
Cada objeto deve conter EXATAMENTE estas chaves:
- "tipo_de_processo": string | null                      # Classe processual (ex.: "REsp", "AREsp", "AgInt no REsp", "AgInt no AREsp", "EDcl", "AgRg")
- "numero_processo": string | null                       # Apenas número (ex.: "2.029.719-RJ" será "2.029.719", "2.163.612-PR" será "2.163.612")
- "UF_Origem_do_caso": string | null                     # Sigla da UF ou origem do caso (ex.: "RJ", "PR", "AL", "SP", "DF"); null se não houver
- "tribunal": "STF" | "STJ" | "TST" | "TSE" | "STM" | null
- "orgao_julgador": string | null                        # Corte Especial, 1ª/2ª/3ª/4ª Turma etc.
- "relator": string | null                               # Ex.: "Min. Marco Aurélio Bellizze"
- "data_julgamento": string | null                       # Formato M/D/AAAA
- "informativo": string | null                           # Ex.: "Info 857"
- "titulo": string | null                                # Título principal (até 20 palavras)
- "tema_repetitivo": string | null                       # Ex.: "Tema 1201" (se houver “Recurso Repetitivo – Tema NNN”). Se não houver, insira a classe processual do recurso (ex.: "REsp", "AREsp" etc.) e número do processo (ex.: "2.029.719")
- "contexto": string | null                              # Um resumo robusto do caso prático ou da situação que deu origem à discussão, com menção à legislação citada e à jurisprudência/precedentes mencionados
- "tese": string | null                                  # Regra jurídica/ratio decidendi
- "modulacao": string | null                             # Se houver modulação/limitação de efeitos, resuma; senão null
- "dispositivo": string | null                           # Comandos finais (se existentes); senão null
- "referencias_legais": array of string                  # Normas citadas normalizadas (ex.: "CF/88, art. 170"; "Lei 8.666/1993, art. 25, III"; "DL 911/1969, art. 2º")
- "precedentes_citados": array of string                 # Ex.: "EDcl no AgInt no AREsp 1.252.262", "REsp 1.924.164". Liste aqui os precedentes mencionados que serviram como panorama/fundamento do caso, explicando-o brevemente no contexto do julgado principal.
- "ramo_direito": string | null                           # Ex.: "Direito Constitucional", "Direito Civil", "Direito Administrativo", "Direito Penal", "Direito Tributário" etc.

# REGRAS DE EXTRAÇÃO E NORMALIZAÇÃO
- Extraia "tipo_de_processo" e "numero_processo" a partir da(s) linha(s) final(is) do bloco (as que começam com "STJ. ...").
- Em "numero_processo", traga exatamente o número.
- "tribunal" e "orgao_julgador" estão na mesma linha (ex.: "STJ. 2ª Turma. ...", "STJ. Corte Especial. ...").
- Converta datas para M/D/AAAA, inclusive quando vierem como "DJe de DD/MM/AAAA".
- "informativo": normalize como "Info XXX" quando houver "(Info XXX)".
- "tema_repetitivo": capte "Tema NNN" quando houver “(Recurso Repetitivo – Tema NNN)” ou forma equivalente.
- "referencias_legais": normalize cada diploma+dispositivo.
- "precedentes_citados": inclua apenas referências jurisprudenciais explicitamente citadas.
- Se não houver informação inequívoca para uma chave, use null (exceto arrays, que devem ser []).

# REGRAS DE FOCO ESPECIAL
- Os campos "tipo_de_processo", "numero_processo", "tribunal", "orgao_julgador", "relator", "data_julgamento", "informativo" e "UF_Origem_do_caso" são muito importantes. Eles quase sempre estão juntos na referência final do julgado. Analise essa referência com atenção máxima.
- Se você não encontrar um valor claro para eles, e apenas para eles, faça uma segunda verificação no texto antes de retornar `null`.

# DIFERENÇA ENTRE "CONTEXTO" E "TESE"
- "contexto" = narrativa fático-processual (o que motivou a controvérsia).
- "tese"     = a regra/ratio decidendi aplicada pelo STJ, preferindo formulação geral.

# FORMATO DE SAÍDA
- Se houver mais de um acórdão no bloco: retorne uma LISTA de objetos.
- Caso contrário: retorne um ÚNICO objeto.
- SEM texto fora do JSON.
- A saída DEVE ser JSON válido.

--- TEXTO PARA ANÁLISE ---
{TEXTO_PARA_ANALISE}
--- FIM ---
"""

# ===================== PADRÕES DE REFERÊNCIA FINAL =====================
PADRAO_FINAL_STF: Pattern[str] = re.compile(
    r"""
    ^[\s\d\.\-—]*STF\.\s+
    (?P<orgao>[^.]+)\.\s+
    (?P<classe>[\w\s]+?)\s+
    (?P<numero>[\d\.\-]+(?:/\w{2})?)\s*
    ,\s*Rel\.\s*[^,]+,\s*
    julgado\s+em\s+(?P<data>\d{1,2}/\d{1,2}/\d{4})
    [^()]*?
    \(Info\s*\d+\)
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE
)

PADRAO_FINAL_STJ: Pattern[str] = re.compile(
    r"""
    ^[\s\d\.\-—]*STJ\.\s+
    (?P<orgao>[^.]+)\.?\s*\.\s+
    (?P<texto>.+?)
    \(
        (?:(?:Recurso\s+Repetitivo\s*[-–]\s*Tema\s*\d+)\s*\)\s*\()?
        Info\s*\d+
    \)\.?\s*$
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE
)


# ===================== Utilitários =====================
def _normalize_date_to_mdy(value: Any) -> str:
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


def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 1)[1]
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()

def _coerce_lists(obj: Dict[str, Any]) -> None:
    for k in ("referencias_legais", "precedentes_citados"):
        if k in obj:
            if obj[k] is None:
                obj[k] = []
            elif isinstance(obj[k], str):
                parts = [p.strip() for p in re.split(r"[;,#]|\s{2,}|,\s*", obj[k]) if p.strip()]
                obj[k] = parts
    if "data_julgamento" in obj and obj["data_julgamento"] is not None:
        obj["data_julgamento"] = _normalize_date_to_mdy(obj["data_julgamento"])


def _listar_txt_em_pasta(pasta: str) -> List[str]:
    encontrados: List[str] = []
    pasta_abs = os.path.abspath(os.path.expanduser(pasta))
    if not os.path.isdir(pasta_abs):
        return encontrados

    for raiz, _, arquivos in os.walk(pasta_abs):
        for nome in sorted(arquivos):
            if nome.lower().endswith(".txt"):
                encontrados.append(os.path.join(raiz, nome))
    return sorted(encontrados, key=lambda p: p.lower())


def _deduplicar_arquivos_txt(caminhos: List[str]) -> List[str]:
    vistos = set()
    unicos: List[str] = []
    for caminho in caminhos:
        if not caminho:
            continue
        abs_path = os.path.abspath(os.path.expanduser(caminho))
        if not os.path.isfile(abs_path):
            continue
        if not abs_path.lower().endswith(".txt"):
            continue
        key = os.path.normcase(abs_path)
        if key in vistos:
            continue
        vistos.add(key)
        unicos.append(abs_path)
    return sorted(unicos, key=lambda p: p.lower())


class DODBatchInputGUI:
    def __init__(self) -> None:
        if tk is None or filedialog is None or messagebox is None or ttk is None:
            raise RuntimeError("Tkinter indisponível.")

        self.root = tk.Tk()
        self.root.title("DOD TXT -> CSV via API")
        self.root.geometry("900x560")
        self.root.minsize(760, 460)

        self.arquivos_selecionados: List[str] = []
        self.selecao_confirmada = False
        self.status_var = tk.StringVar(
            value="Selecione arquivos .txt e/ou pastas para iniciar o processamento."
        )

        self._montar_interface()

    def _montar_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        cabecalho = ttk.Frame(self.root, padding=(12, 10, 12, 0))
        cabecalho.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            cabecalho,
            text="Conversor DOD: TXT para CSV",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            cabecalho,
            text="Adicione arquivos .txt ou pastas com .txt (busca recursiva).",
        ).pack(anchor="w", pady=(2, 0))

        barra_acoes = ttk.Frame(self.root, padding=(12, 8))
        barra_acoes.grid(row=1, column=0, sticky="ew")
        ttk.Button(barra_acoes, text="Adicionar arquivos .txt", command=self.adicionar_arquivos).pack(side="left")
        ttk.Button(barra_acoes, text="Adicionar pasta", command=self.adicionar_pasta).pack(side="left", padx=(8, 0))
        ttk.Button(barra_acoes, text="Remover selecionado(s)", command=self.remover_selecionados).pack(side="left", padx=(8, 0))
        ttk.Button(barra_acoes, text="Limpar lista", command=self.limpar_lista).pack(side="left", padx=(8, 0))

        frame_lista = ttk.Frame(self.root, padding=(12, 0))
        frame_lista.grid(row=2, column=0, sticky="nsew")
        frame_lista.columnconfigure(0, weight=1)
        frame_lista.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(frame_lista, selectmode=tk.EXTENDED, activestyle="dotbox")
        self.listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(frame_lista, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scrollbar.set)

        rodape = ttk.Frame(self.root, padding=(12, 8))
        rodape.grid(row=3, column=0, sticky="ew")
        ttk.Label(rodape, textvariable=self.status_var).pack(anchor="w")

        botoes_finais = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        botoes_finais.grid(row=4, column=0, sticky="ew")
        ttk.Button(botoes_finais, text="Cancelar", command=self.cancelar).pack(side="right")
        ttk.Button(botoes_finais, text="Iniciar processamento", command=self.iniciar_processamento).pack(side="right", padx=(0, 8))

    def _atualizar_lista(self) -> None:
        self.listbox.delete(0, tk.END)
        for path in self.arquivos_selecionados:
            self.listbox.insert(tk.END, path)
        self.status_var.set(f"{len(self.arquivos_selecionados)} arquivo(s) .txt selecionado(s).")

    def _adicionar_caminhos(self, caminhos: List[str]) -> int:
        existentes = {os.path.normcase(p) for p in self.arquivos_selecionados}
        adicionados = 0
        for caminho in _deduplicar_arquivos_txt(caminhos):
            key = os.path.normcase(caminho)
            if key in existentes:
                continue
            self.arquivos_selecionados.append(caminho)
            existentes.add(key)
            adicionados += 1
        self.arquivos_selecionados.sort(key=lambda p: p.lower())
        self._atualizar_lista()
        return adicionados

    def adicionar_arquivos(self) -> None:
        arquivos = filedialog.askopenfilenames(
            title="Selecione um ou mais arquivos TXT",
            filetypes=[("Arquivos TXT", "*.txt"), ("Todos os arquivos", "*.*")],
        )
        if not arquivos:
            return
        qtd = self._adicionar_caminhos(list(arquivos))
        if qtd == 0:
            messagebox.showinfo("Sem novos arquivos", "Nenhum novo arquivo .txt foi adicionado.")

    def adicionar_pasta(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione uma pasta com arquivos TXT")
        if not pasta:
            return
        txts = _listar_txt_em_pasta(pasta)
        if not txts:
            messagebox.showwarning("Pasta vazia", "Nenhum arquivo .txt foi encontrado nesta pasta.")
            return
        qtd = self._adicionar_caminhos(txts)
        if qtd == 0:
            messagebox.showinfo("Sem novos arquivos", "Todos os arquivos dessa pasta já estavam na lista.")

    def remover_selecionados(self) -> None:
        selecionados = list(self.listbox.curselection())
        if not selecionados:
            return
        for idx in reversed(selecionados):
            del self.arquivos_selecionados[idx]
        self._atualizar_lista()

    def limpar_lista(self) -> None:
        self.arquivos_selecionados.clear()
        self._atualizar_lista()
        self.status_var.set("Lista limpa. Adicione arquivos .txt e/ou pastas para continuar.")

    def iniciar_processamento(self) -> None:
        if not self.arquivos_selecionados:
            messagebox.showwarning("Nenhuma entrada", "Selecione ao menos um arquivo .txt ou uma pasta.")
            return
        self.selecao_confirmada = True
        self.root.destroy()

    def cancelar(self) -> None:
        self.selecao_confirmada = False
        self.root.destroy()

    def run(self) -> List[str]:
        self.root.mainloop()
        if self.selecao_confirmada:
            return list(self.arquivos_selecionados)
        return []


def gui_select_inputs() -> List[str]:
    if tk is None or filedialog is None or messagebox is None or ttk is None:
        print("Aviso: GUI indisponível (tkinter). Prosseguindo sem seleção gráfica.")
        return []
    try:
        app = DODBatchInputGUI()
        return app.run()
    except Exception as e:
        print(f"Aviso: falha ao abrir GUI ({e}). Prosseguindo sem seleção gráfica.")
        return []


def descobrir_caminhos_txt(
    input_glob: str = "*.txt",
    input_dirs: Optional[List[str]] = None,
    input_files: Optional[List[str]] = None,
) -> List[str]:
    candidatos: List[str] = []
    explicit_inputs = bool(input_dirs or input_files)

    for pasta in (input_dirs or []):
        candidatos.extend(_listar_txt_em_pasta(pasta))

    candidatos.extend(input_files or [])
    caminhos = _deduplicar_arquivos_txt(candidatos)
    if caminhos or explicit_inputs:
        return caminhos

    padrao = (input_glob or "").strip() or "*.txt"
    return _deduplicar_arquivos_txt(glob.glob(padrao))


@dataclass(frozen=True)
class OpenAIRuntimeConfig:
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
    if "connection" in msg or "temporarily unavailable" in msg:
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


def _call_openai(
    prompt: str,
    model: str = "gpt-5-mini",
    *,
    timeout: int = OPENAI_DEFAULT_TIMEOUT,
    retries: int = OPENAI_DEFAULT_RETRIES,
    pacer: Optional[RequestPacer] = None,
) -> str:
    """Chama a OpenAI com retries, timeout e pacing simples por RPM."""
    last_error: Optional[Exception] = None
    total_attempts = max(1, int(retries))
    for attempt in range(1, total_attempts + 1):
        try:
            if pacer is not None:
                pacer.wait_turn()
            if client:  # Nova sintaxe (openai >= 1.0)
                resp = client.chat.completions.create(
                    model=model,
                    timeout=timeout,
                    messages=[
                        {"role": "system", "content": "Responda APENAS com JSON válido."},
                        {"role": "user", "content": prompt},
                    ],
                )
                txt = resp.choices[0].message.content
            else:  # Sintaxe antiga (openai < 1.0)
                resp = openai.ChatCompletion.create(
                    model=model,
                    request_timeout=timeout,
                    messages=[
                        {"role": "system", "content": "Responda APENAS com JSON válido."},
                        {"role": "user", "content": prompt},
                    ],
                )
                txt = resp["choices"][0]["message"]["content"]

            if txt:
                return txt
            raise RuntimeError("Resposta vazia da OpenAI.")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            kind = _classify_openai_error(exc)
            retryable = kind in {"rate_limit", "timeout", "connection", "upstream", "unknown"}
            if attempt >= total_attempts or not retryable:
                break
            time.sleep(_retry_wait(attempt, kind))

    raise RuntimeError(
        f"Falha na chamada à API OpenAI após {total_attempts} tentativa(s): {last_error}"
    )

def _auto_detect_tribunal(conteudo: str) -> str:
    stf_hits = len(list(PADRAO_FINAL_STF.finditer(conteudo)))
    stj_hits = len(list(PADRAO_FINAL_STJ.finditer(conteudo)))
    if stf_hits == 0 and stj_hits == 0:
        if "supremo tribunal federal" in conteudo.lower() or "\nSTF." in conteudo:
            return "STF"
        if "superior tribunal de justiça" in conteudo.lower() or "\nSTJ." in conteudo:
            return "STJ"
        return "STF" # Default
    return "STF" if stf_hits >= stj_hits else "STJ"

def _split_em_blocos(conteudo: str, tribunal: str) -> List[str]:
    padrao = PADRAO_FINAL_STF if tribunal == "STF" else PADRAO_FINAL_STJ
    matches = list(padrao.finditer(conteudo))
    if not matches:
        return [conteudo.strip()] if conteudo.strip() else []

    blocos = []
    cursor = 0
    for match in matches:
        # Adiciona o texto ANTES da referência final encontrada
        bloco_anterior = conteudo[cursor:match.start()].strip()
        if bloco_anterior:
            blocos.append(bloco_anterior)
        cursor = match.start()

    # Adiciona o último bloco, que contém a última referência final
    ultimo_bloco = conteudo[cursor:].strip()
    if ultimo_bloco:
        blocos.append(ultimo_bloco)

    # Lógica para agrupar múltiplos acórdãos no mesmo bloco
    blocos_agrupados = []
    temp_bloco = ""
    for bloco in blocos:
        if temp_bloco and not padrao.search(temp_bloco):
            temp_bloco += "\n\n" + bloco
        else:
            if temp_bloco:
                blocos_agrupados.append(temp_bloco)
            temp_bloco = bloco
    if temp_bloco:
        blocos_agrupados.append(temp_bloco)

    return blocos_agrupados

def _rows_para_csv(registros: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
    schema_cols = [
        "Tipo_de_processo","numero_processo","tribunal","orgao_julgador",
        "relator","data_julgamento","informativo","tema",
        "tema_repercussao","tema_repetitivo",
        "contexto","tese","modulacao","dispositivo",
        "referencias_legais","precedentes_citados","fonte_referencia_final"]

    if not registros:
        return schema_cols, []

    df = pd.DataFrame(registros)
    outras_cols = [c for c in df.columns if c not in schema_cols]
    df = df[[c for c in schema_cols if c in df.columns] + outras_cols]

    def _list_to_str(v: Any) -> str:
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return str(v) if v is not None else ""

    for col in ("referencias_legais", "precedentes_citados"):
        if col in df.columns:
            df[col] = df[col].apply(_list_to_str)
    if "data_julgamento" in df.columns:
        df["data_julgamento"] = df["data_julgamento"].apply(_normalize_date_to_mdy)
    return list(df.columns), df.to_dict(orient="records")


def processar_arquivo_txt(caminho_txt: str,
                          modelo: str = "gpt-5-mini",
                          openai_cfg: Optional[OpenAIRuntimeConfig] = None,
                          tribunal_cli: str = "auto",
                          checkpoint_path: Optional[Path] = None,
                          output_csv_path: Optional[Path] = None,
                          report_path: Optional[Path] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(caminho_txt, "r", encoding="utf-8") as f:
        bruto = f.read()

    started_at = time.time()
    tribunal_detectado = tribunal_cli.upper()
    if tribunal_detectado == "AUTO":
        tribunal_detectado = _auto_detect_tribunal(bruto)

    prompt_base = PROMPT_STF if tribunal_detectado == "STF" else PROMPT_STJ

    blocos = _split_em_blocos(bruto, tribunal_detectado)
    resultados: List[Dict[str, Any]] = []
    cfg = openai_cfg or OpenAIRuntimeConfig()
    blocos_validos = [(i, b) for i, b in enumerate(blocos, start=1) if b.strip()]
    total_blocos = len(blocos_validos)
    file_sig = build_file_signature(Path(caminho_txt))
    done_blocos: set[int] = set()
    blocos_falha: set[int] = set()

    if checkpoint_path is not None:
        cp = read_json_dict(checkpoint_path)
        cp_sig = cp.get("source_signature", {})
        cp_model = str(cp.get("model", "")).strip()
        cp_tribunal = str(cp.get("tribunal", "")).strip().upper()
        cp_rows = cp.get("rows", [])
        cp_done = cp.get("done_blocos", [])
        cp_version = int(cp.get("version", 0) or 0)
        cp_total = int(cp.get("total_blocos", 0) or 0)
        if (
            cp_version == CHECKPOINT_VERSION
            and same_file_signature(cp_sig, file_sig)
            and cp_model == modelo
            and cp_tribunal == tribunal_detectado
            and cp_total == total_blocos
            and isinstance(cp_rows, list)
            and isinstance(cp_done, list)
        ):
            resultados = [row for row in cp_rows if isinstance(row, dict)]
            for raw_idx in cp_done:
                try:
                    idx = int(raw_idx)
                except Exception:
                    continue
                if idx >= 1:
                    done_blocos.add(idx)
            print(
                f"  [resume] checkpoint carregado: blocos concluídos={len(done_blocos)}/{total_blocos} "
                f"| registros={len(resultados)}"
            )

    def _save_progress(status: str) -> None:
        if output_csv_path is not None:
            cols, rows = _rows_para_csv(resultados)
            write_csv_atomic(output_csv_path, cols, rows)
        if checkpoint_path is not None:
            write_json_atomic(
                checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "source_signature": file_sig,
                    "model": modelo,
                    "tribunal": tribunal_detectado,
                    "total_blocos": total_blocos,
                    "done_blocos": sorted(done_blocos),
                    "blocos_falha": sorted(blocos_falha),
                    "rows": resultados,
                    "status": status,
                    "updated_at": utc_now_iso(),
                },
            )
        if report_path is not None:
            write_json_atomic(
                report_path,
                {
                    "script": "DOD_txt_to_csv_viaAPI.py",
                    "input_file": str(Path(caminho_txt).resolve()),
                    "output_csv": str(output_csv_path.resolve()) if output_csv_path is not None else "",
                    "checkpoint_file": str(checkpoint_path.resolve()) if checkpoint_path is not None else "",
                    "status": status,
                    "model": modelo,
                    "tribunal": tribunal_detectado,
                    "total_blocos": total_blocos,
                    "done_blocos": len(done_blocos),
                    "blocos_falha": len(blocos_falha),
                    "registros_extraidos": len(resultados),
                    "elapsed_seconds": round(max(0.0, time.time() - started_at), 2),
                    "updated_at": utc_now_iso(),
                },
            )

    print(
        f"  Tribunal: {tribunal_detectado} | Blocos: {len(blocos_validos)} "
        f"| workers={min(cfg.max_workers, cfg.max_workers_cap)} "
        f"| batch={cfg.batch_size} | rpm={cfg.target_rpm}"
    )

    if not blocos_validos:
        _save_progress(status="completed")
        return resultados, {
            "total_blocos": 0,
            "done_blocos": 0,
            "blocos_falha": 0,
            "registros": len(resultados),
            "status": "completed",
        }

    pending_blocos = [item for item in blocos_validos if item[0] not in done_blocos]
    if not pending_blocos:
        print("  -> Nenhum bloco pendente (checkpoint já completo).")
        _save_progress(status="completed")
        return resultados, {
            "total_blocos": total_blocos,
            "done_blocos": len(done_blocos),
            "blocos_falha": len(blocos_falha),
            "registros": len(resultados),
            "status": "completed",
        }

    workers = min(max(1, int(cfg.max_workers)), max(1, int(cfg.max_workers_cap)), len(pending_blocos))
    pacer = RequestPacer(cfg.target_rpm)

    def _processar_bloco(item: tuple[int, str]) -> tuple[int, List[Dict[str, Any]], str, bool]:
        bloco_idx, bloco_txt = item
        prompt = prompt_base.replace("{TEXTO_PARA_ANALISE}", bloco_txt)
        try:
            raw = _call_openai(
                prompt,
                model=modelo,
                timeout=max(5, int(cfg.timeout)),
                retries=max(1, int(cfg.retries)),
                pacer=pacer,
            )
            raw = _strip_code_fences(raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                raw_clean = raw.strip().rstrip(",").replace("'", '"')
                parsed = json.loads(raw_clean)

            out: List[Dict[str, Any]] = []
            if isinstance(parsed, dict):
                _coerce_lists(parsed)
                out.append(parsed)
            elif isinstance(parsed, list):
                for row in parsed:
                    if isinstance(row, dict):
                        _coerce_lists(row)
                        out.append(row)
                if not out:
                    return bloco_idx, [], f"    AVISO: Bloco {bloco_idx} sem objetos válidos.", True
            else:
                return bloco_idx, [], f"    AVISO: Bloco {bloco_idx} retornou tipo inesperado: {type(parsed)}.", True
            return bloco_idx, out, "", True
        except Exception as exc:  # noqa: BLE001
            return bloco_idx, [], f"    ERRO: Bloco {bloco_idx} falhou ({exc}).", False

    total = len(pending_blocos)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, total, max(1, int(cfg.batch_size))):
            end = min(start + max(1, int(cfg.batch_size)), total)
            lote = pending_blocos[start:end]
            print(f"  - OpenAI lote {start + 1}-{end}/{total} (pendentes)...")
            fut_map = {executor.submit(_processar_bloco, item): item[0] for item in lote}
            lote_resultados: Dict[int, List[Dict[str, Any]]] = {}
            lote_status: Dict[int, bool] = {}
            for fut in as_completed(fut_map):
                bloco_idx, rows, aviso, ok = fut.result()
                if aviso:
                    print(aviso)
                lote_resultados[bloco_idx] = rows
                lote_status[bloco_idx] = ok

            for bloco_idx in sorted(lote_resultados.keys()):
                if lote_status.get(bloco_idx, False):
                    done_blocos.add(bloco_idx)
                    blocos_falha.discard(bloco_idx)
                else:
                    blocos_falha.add(bloco_idx)
                resultados.extend(lote_resultados[bloco_idx])
            _save_progress(status="running")

            if end < total and cfg.delay > 0:
                time.sleep(max(0.0, float(cfg.delay)))

    status = "completed" if len(done_blocos) >= total_blocos else "partial"
    _save_progress(status=status)
    return resultados, {
        "total_blocos": total_blocos,
        "done_blocos": len(done_blocos),
        "blocos_falha": len(blocos_falha),
        "registros": len(resultados),
        "status": status,
    }

def processar_em_lote(
    input_glob: str = "*.txt",
    input_dirs: Optional[List[str]] = None,
    input_files: Optional[List[str]] = None,
    modelo: str = "gpt-5-mini",
    openai_cfg: Optional[OpenAIRuntimeConfig] = None,
    tribunal: str = "auto",
) -> None:
    caminhos = descobrir_caminhos_txt(
        input_glob=input_glob,
        input_dirs=input_dirs,
        input_files=input_files,
    )
    if not caminhos:
        origem = []
        if input_dirs:
            origem.append(f"{len(input_dirs)} pasta(s)")
        if input_files:
            origem.append(f"{len(input_files)} arquivo(s)")
        if not origem:
            origem.append(f"padrão '{(input_glob or '').strip() or '*.txt'}'")
        raise FileNotFoundError(f"Nenhum arquivo .txt encontrado para: {', '.join(origem)}.")

    print(f"Encontrados {len(caminhos)} arquivos para processar.")

    for caminho in caminhos:
        print(f"\nProcessando arquivo: {caminho}")
        try:
            output_csv = (SCRIPT_DIR / f"{Path(caminho).stem}.csv").resolve()
            checkpoint_path = (SCRIPT_DIR / f".{Path(caminho).stem}.openai.checkpoint.json").resolve()
            report_path = (SCRIPT_DIR / f".{Path(caminho).stem}.openai.report.json").resolve()
            if output_csv.exists() and not checkpoint_path.exists():
                backup = make_backup(output_csv, label="startup_backup")
                if backup is not None:
                    print(f"  -> Backup inicial: {backup}")

            res_arquivo, stats = processar_arquivo_txt(
                caminho,
                modelo=modelo,
                openai_cfg=openai_cfg,
                tribunal_cli=tribunal,
                checkpoint_path=checkpoint_path,
                output_csv_path=output_csv,
                report_path=report_path,
            )

            if not res_arquivo:
                print(f"  => Nenhum registro extraído de '{caminho}'.")
                continue

            print(f"  => {len(res_arquivo)} registro(s) extraído(s).")

            cols, rows_csv = _rows_para_csv(res_arquivo)
            write_csv_atomic(output_csv, cols, rows_csv)
            print(f"  => Salvo em: {output_csv}")
            print(
                f"  => Progresso final: blocos {stats.get('done_blocos', 0)}/{stats.get('total_blocos', 0)} "
                f"| falhas={stats.get('blocos_falha', 0)}"
            )
            if stats.get("status") == "completed" and checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"  => Checkpoint removido: {checkpoint_path}")

        except Exception as e:
            print(f"   ERRO GERAL ao processar '{caminho}': {e}")

    print("\nProcessamento em lote concluído.")


def main():
    parser = argparse.ArgumentParser(description="Extrai metadados de Informativos (STF/STJ) e gera um CSV por arquivo.")
    parser.add_argument("--input", required=False, default="*.txt",
                        help="Caminho ou glob para arquivos .txt (ex.: 'Informativo*.txt' ou 'C:/docs/*.txt')")
    parser.add_argument("--input-files", nargs="*", default=[],
                        help="Lista de arquivos .txt específicos para processar.")
    parser.add_argument("--input-dirs", nargs="*", default=[],
                        help="Lista de pastas para busca recursiva de arquivos .txt.")
    parser.add_argument("--no-gui", action="store_true",
                        help="Desativa a interface GUI de seleção de arquivos/pastas.")
    parser.add_argument("--model", required=False, default="gpt-5-mini",
                        help="Modelo OpenAI (ex.: gpt-5-mini, gpt-4o)")
    parser.add_argument("--sleep", required=False, type=float, default=None,
                        help="(Legado) pausa entre chamadas. Se definido, substitui --openai-delay.")
    parser.add_argument("--openai-batch-size", type=int, default=OPENAI_DEFAULT_BATCH_SIZE)
    parser.add_argument("--openai-max-workers", type=int, default=OPENAI_DEFAULT_MAX_WORKERS)
    parser.add_argument("--openai-max-workers-cap", type=int, default=OPENAI_DEFAULT_MAX_WORKERS_CAP)
    parser.add_argument("--openai-delay", type=float, default=OPENAI_DEFAULT_DELAY)
    parser.add_argument("--openai-retries", type=int, default=OPENAI_DEFAULT_RETRIES)
    parser.add_argument("--openai-timeout", type=int, default=OPENAI_DEFAULT_TIMEOUT)
    parser.add_argument("--openai-target-rpm", type=int, default=OPENAI_DEFAULT_TARGET_RPM)
    parser.add_argument("--tribunal", required=False, default="auto", choices=["auto","STF","STJ"],
                        help="Força o padrão de análise: STF, STJ ou auto-detecção.")
    parser.add_argument("--verbose", action="store_true", help="Exibe logs detalhados.")
    parser.add_argument("--quiet", action="store_true", help="Exibe apenas avisos/erros.")
    parser.add_argument("--debug", action="store_true", help="Ativa modo debug técnico.")
    parser.add_argument("--log-file", default="", help="Arquivo opcional para salvar logs.")
    args = parser.parse_args()

    logger = configure_standard_logging(
        "DOD_txt_to_csv_viaAPI",
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
    )
    install_print_logger_bridge(globals(), logger)

    input_files = list(args.input_files or [])
    input_dirs = list(args.input_dirs or [])
    input_glob = args.input

    deve_tentar_gui = (
        not args.no_gui
        and not input_files
        and not input_dirs
        and ((args.input or "").strip() in ("", "*.txt"))
    )
    if deve_tentar_gui:
        selecionados_gui = gui_select_inputs()
        if selecionados_gui:
            input_files = selecionados_gui
        else:
            print("GUI sem seleção. Usando o padrão informado em --input.")

    effective_delay = max(0.0, float(args.openai_delay))
    if args.sleep is not None:
        effective_delay = max(0.0, float(args.sleep))
    openai_cfg = OpenAIRuntimeConfig(
        batch_size=max(1, int(args.openai_batch_size)),
        max_workers=max(1, int(args.openai_max_workers)),
        max_workers_cap=max(1, int(args.openai_max_workers_cap)),
        delay=effective_delay,
        retries=max(1, int(args.openai_retries)),
        timeout=max(5, int(args.openai_timeout)),
        target_rpm=max(0, int(args.openai_target_rpm)),
    )

    processar_em_lote(
        input_glob=input_glob,
        input_dirs=input_dirs,
        input_files=input_files,
        modelo=args.model,
        openai_cfg=openai_cfg,
        tribunal=args.tribunal
    )

if __name__ == "__main__":
    main()
