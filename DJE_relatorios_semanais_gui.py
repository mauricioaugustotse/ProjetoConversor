#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI para processar CSVs DJe, importar para Notion e gerar relatorios semanais."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import NOTION_DJE_csv_to_notion as importer
import NOTION_relatoriodeIA_v2 as report
import SJUR_csv_to_csv_NOTIONfriendly_v2 as sjur
from Artefatos.scripts.openai_progress_utils import utc_now_iso, write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = PROJECT_ROOT / "DJE"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Artefatos" / "dados" / "csv"
DEFAULT_REPORTS_PARENT_PAGE_ID = "31772195-5c64-803f-902d-d2cb8600b4dd"
DEFAULT_REPORTS_PARENT_URL = f"https://www.notion.so/{DEFAULT_REPORTS_PARENT_PAGE_ID.replace('-', '')}"
MANIFEST_FILE = PROJECT_ROOT / "Artefatos" / "reports" / "dje_relatorios_semanais_manifest.json"
PIPELINE_VERSION = 3
URL_PRESERVE_COLUMNS = ("noticia_TSE", "noticia_TRE", "noticia_geral_1", "noticia_geral_2")
THEME_PUNCHLINE_MODEL = os.getenv("DJE_THEME_PUNCHLINE_MODEL", "gpt-5.4-nano") or "gpt-5.4-nano"
THEME_OPENAI_MAX_WORKERS = int(os.getenv("DJE_THEME_OPENAI_MAX_WORKERS", "4") or "4")
THEME_OPENAI_BATCH_SIZE = int(os.getenv("DJE_THEME_OPENAI_BATCH_SIZE", "25") or "25")
THEME_OPENAI_TARGET_RPM = int(os.getenv("DJE_THEME_OPENAI_TARGET_RPM", "120") or "120")
THEME_OPENAI_TIMEOUT = int(os.getenv("DJE_THEME_OPENAI_TIMEOUT", "60") or "60")
REPORT_MAX_OPENAI_CASES = int(os.getenv("DJE_REPORT_MAX_OPENAI_CASES", "1000") or "1000")
REPORT_OPENAI_TRIAGE_THRESHOLD = int(os.getenv("DJE_REPORT_OPENAI_TRIAGE_THRESHOLD", "0") or "0")


@dataclass(frozen=True)
class WeeklyPeriod:
    start: date
    end: date

    @property
    def title(self) -> str:
        return f"DJe do TSE - {self.start:%d/%m/%Y} a {self.end:%d/%m/%Y}"


def _normalize_ws(value: Any) -> str:
    return report._normalize_ws(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def read_manifest() -> Dict[str, Any]:
    try:
        with MANIFEST_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def write_manifest(payload: Mapping[str, Any]) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(MANIFEST_FILE, dict(payload))


def _files_signature_map(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    return {str(path.resolve()): file_signature(path) for path in paths}


def _same_file_signatures(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key, left_sig in left.items():
        right_sig = right.get(key)
        if not isinstance(left_sig, Mapping) or not isinstance(right_sig, Mapping):
            return False
        if left_sig.get("sha256") != right_sig.get("sha256"):
            return False
    return True


def find_pending_combined_csv(
    paths: Sequence[Path],
    *,
    require_current_pipeline: bool = True,
) -> Optional[Path]:
    manifest = read_manifest()
    pending = manifest.get("pending_run") or {}
    if not isinstance(pending, Mapping):
        return None
    if require_current_pipeline and int(pending.get("pipeline_version", 0) or 0) != PIPELINE_VERSION:
        return None
    pending_files = pending.get("input_files") or {}
    if not isinstance(pending_files, Mapping):
        return None
    try:
        current_files = _files_signature_map(paths)
    except Exception:
        return None
    if not _same_file_signatures(current_files, pending_files):
        return None
    combined = Path(str(pending.get("combined_csv") or "")).expanduser()
    if not combined.is_absolute():
        combined = PROJECT_ROOT / combined
    combined = combined.resolve()
    return combined if combined.exists() and combined.is_file() else None


def mark_pending_run(paths: Sequence[Path], *, combined_csv: Path, periods: Sequence[WeeklyPeriod]) -> None:
    manifest = read_manifest()
    manifest["pending_run"] = {
        "pipeline_version": PIPELINE_VERSION,
        "input_files": _files_signature_map(paths),
        "combined_csv": str(combined_csv.resolve()),
        "periods": [period.title for period in periods],
        "updated_at_utc": utc_now_iso(),
    }
    manifest["updated_at_utc"] = utc_now_iso()
    write_manifest(manifest)


def mark_files_processed(paths: Sequence[Path], *, combined_csv: Path, periods: Sequence[WeeklyPeriod]) -> None:
    manifest = read_manifest()
    processed_files = dict(manifest.get("processed_files") or {})
    processed_hashes = dict(manifest.get("processed_hashes") or {})
    for path in paths:
        sig = file_signature(path)
        item = {
            **sig,
            "processed_at_utc": utc_now_iso(),
            "combined_csv": str(combined_csv.resolve()),
            "periods": [period.title for period in periods],
        }
        processed_files[str(path.resolve())] = item
        processed_hashes[sig["sha256"]] = item
    manifest["processed_files"] = processed_files
    manifest["processed_hashes"] = processed_hashes
    manifest.pop("pending_run", None)
    manifest["updated_at_utc"] = utc_now_iso()
    write_manifest(manifest)


def is_new_csv(path: Path) -> bool:
    manifest = read_manifest()
    processed_hashes = manifest.get("processed_hashes") or {}
    try:
        return _sha256(path) not in processed_hashes
    except Exception:
        return True


def mark_report_generated(period: WeeklyPeriod, *, case_count: int) -> None:
    manifest = read_manifest()
    generated = dict(manifest.get("generated_reports") or {})
    generated[period.title] = {
        "case_count": int(case_count),
        "generated_at_utc": utc_now_iso(),
    }
    manifest["generated_reports"] = generated
    manifest["updated_at_utc"] = utc_now_iso()
    write_manifest(manifest)


def existing_report_is_stale(period: WeeklyPeriod, *, current_count: int) -> bool:
    manifest = read_manifest()
    generated = manifest.get("generated_reports") or {}
    entry = generated.get(period.title)
    if not isinstance(entry, Mapping):
        return True
    try:
        return int(entry.get("case_count", -1)) != int(current_count)
    except Exception:
        return True


def count_cases_in_period(data_source_id: str, period: WeeklyPeriod) -> int:
    existing = importer.query_existing_pages_by_period(
        data_source_id,
        period.start.isoformat(),
        period.end.isoformat(),
    )
    return len(set(existing.values()))


def parse_treated_csv_date(value: Any) -> Optional[date]:
    return importer.parse_csv_date(value, date_order="mdy")


def read_treated_rows(path: Path) -> List[Dict[str, str]]:
    encoding, delimiter = sjur.detect_csv_format(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows: List[Dict[str, str]] = []
        for raw in reader:
            if not raw:
                continue
            rows.append({str(key or ""): str(value or "") for key, value in raw.items()})
    return rows


def week_for_decision_day(day: date) -> WeeklyPeriod:
    # Semana civil completa (seg-dom): decisoes de sabado/domingo pertencem
    # a semana que termina naquele domingo, e a query do relatorio cobre os 7 dias.
    start = day - timedelta(days=day.weekday())
    return WeeklyPeriod(start=start, end=start + timedelta(days=6))


def periods_from_treated_csv(path: Path) -> List[WeeklyPeriod]:
    periods: Dict[tuple[date, date], WeeklyPeriod] = {}
    for row in read_treated_rows(path):
        parsed = parse_treated_csv_date(row.get("dataDecisao"))
        if parsed is None:
            continue
        period = week_for_decision_day(parsed)
        periods[(period.start, period.end)] = period
    return sorted(periods.values(), key=lambda item: item.start)


def _child_page_title(block: Mapping[str, Any]) -> str:
    child_page = block.get("child_page") if isinstance(block, dict) else {}
    return _normalize_ws((child_page or {}).get("title"))


def _normalize_title_key(title: str) -> str:
    return _normalize_ws(title).casefold()


def list_report_pages(parent_page_id: str) -> Dict[str, str]:
    pages: Dict[str, str] = {}
    cursor = ""
    while True:
        params: Dict[str, Any] = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        payload = report.notion_request("GET", f"/v1/blocks/{parent_page_id}/children", params=params)
        for block in payload.get("results", []) or []:
            if not isinstance(block, dict) or block.get("type") != "child_page":
                continue
            title = _child_page_title(block)
            if title:
                pages[_normalize_title_key(title)] = report._normalize_notion_id(str(block.get("id", "")))
        if not payload.get("has_more"):
            break
        cursor = _normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return pages


def create_report_page(parent_page_id: str, title: str) -> str:
    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "properties": {"title": {"title": [{"type": "text", "text": {"content": title}}]}},
    }
    created = report.notion_request("POST", "/v1/pages", json_body=payload)
    return report._normalize_notion_id(str(created.get("id", "")))


def ensure_report_page(parent_page_id: str, title: str, *, overwrite_existing: bool) -> tuple[str, bool]:
    pages = list_report_pages(parent_page_id)
    existing_id = pages.get(_normalize_title_key(title), "")
    if existing_id:
        return report._notion_page_url_from_id(existing_id), False
    created_id = create_report_page(parent_page_id, title)
    return report._notion_page_url_from_id(created_id), True


def run_command(command: Sequence[str], *, log: Any, cwd: Path = PROJECT_ROOT) -> None:
    log("$ " + " ".join(str(part) for part in command))
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Comando falhou com codigo {return_code}: {' '.join(command)}")


def build_combined_name() -> str:
    return f"DJE_gui_{datetime.now():%Y%m%d_%H%M%S}_consolidado.csv"


def process_import_and_generate(
    files: Sequence[Path],
    *,
    database_url: str,
    reports_parent_url: str,
    log: Any,
    force_regenerate: bool = False,
) -> None:
    if not files:
        raise RuntimeError("Nenhum CSV selecionado.")

    report_model = report.DEFAULT_OPENAI_MODEL
    theme_model = THEME_PUNCHLINE_MODEL
    max_openai_cases = REPORT_MAX_OPENAI_CASES
    openai_triage_threshold = REPORT_OPENAI_TRIAGE_THRESHOLD

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_csv = find_pending_combined_csv(files)
    if combined_csv is not None:
        log(f"Reaproveitando CSV consolidado de execucao anterior: {combined_csv}")
    else:
        previous_combined = find_pending_combined_csv(files, require_current_pipeline=False)
        combined_name = build_combined_name()
        combined_csv = DEFAULT_OUTPUT_DIR / combined_name
        cmd = [
            sys.executable,
            "SJUR_csv_to_csv_NOTIONfriendly_v2.py",
            *[str(path) for path in files],
            "--no-gui",
            "--out-dir",
            str(DEFAULT_OUTPUT_DIR),
            "--combined-name",
            combined_name,
            "--max-texto-chars",
            "9000",
            "--openai-model",
            theme_model,
            "--openai-max-workers",
            str(THEME_OPENAI_MAX_WORKERS),
            "--openai-batch-size",
            str(THEME_OPENAI_BATCH_SIZE),
            "--openai-target-rpm",
            str(THEME_OPENAI_TARGET_RPM),
            "--openai-timeout",
            str(THEME_OPENAI_TIMEOUT),
            "--verbose",
            "--gerar-tema-punchline-chatgpt",
        ]
        log(f"Gerando tema/punchline com OpenAI economico: {theme_model}")
        if previous_combined is not None:
            log(f"Reprocessando com tema contextual e preservando URLs ja encontradas: {previous_combined}")
            cmd.extend(
                [
                    "--preserve-from-csv",
                    str(previous_combined),
                    "--preserve-columns",
                    ",".join(URL_PRESERVE_COLUMNS),
                ]
            )
        else:
            log(
                "Busca Gemini em massa desativada no preparo do CSV; "
                "as noticias serao buscadas apenas para os casos publicados nos relatorios."
            )
        run_command(cmd, log=log)
    if not combined_csv.exists():
        raise RuntimeError(f"CSV consolidado nao foi criado: {combined_csv}")

    periods = periods_from_treated_csv(combined_csv)
    if not periods:
        raise RuntimeError("Nenhum periodo semanal foi identificado no CSV consolidado.")
    mark_pending_run(files, combined_csv=combined_csv, periods=periods)

    run_command(
        [
            sys.executable,
            "NOTION_DJE_csv_to_notion.py",
            "--csv",
            str(combined_csv),
            "--database-url",
            database_url,
            "--mode",
            "upsert",
            "--model",
            report_model,
            "--verbose",
        ],
        log=log,
    )

    parent_page_id = report.extract_notion_id_from_url(reports_parent_url)

    class InitArgs:
        openai_max_workers = report.DEFAULT_OPENAI_MAX_WORKERS
        max_cases_per_batch = report.DEFAULT_MAX_CASES_PER_BATCH
        openai_target_rpm = report.DEFAULT_OPENAI_TARGET_RPM
        max_openai_cases = report.DEFAULT_MAX_OPENAI_CASES
        openai_triage_threshold = report.DEFAULT_OPENAI_TRIAGE_THRESHOLD
        disable_analysis_cache = False
        model = report.DEFAULT_OPENAI_MODEL
        force_openai_fallback = False

    InitArgs.model = report_model
    InitArgs.force_openai_fallback = False
    InitArgs.max_openai_cases = max_openai_cases
    InitArgs.openai_triage_threshold = openai_triage_threshold
    report.initialize_clients(InitArgs)

    database_id = report.extract_notion_id_from_url(database_url)
    data_source_id = report.retrieve_database_and_datasource_id(database_id)

    log("Periodos detectados: " + "; ".join(period.title for period in periods))
    for period in periods:
        page_url, created = ensure_report_page(parent_page_id, period.title, overwrite_existing=False)
        current_count = count_cases_in_period(data_source_id, period)
        if not created and not force_regenerate and not existing_report_is_stale(period, current_count=current_count):
            log(f"Relatorio ja existe e esta atualizado; pulando: {period.title}")
            continue
        if not created:
            reason = "regeneracao forcada" if force_regenerate else "casos novos detectados na base"
            log(f"Relatorio existente sera regenerado ({reason}): {period.title}")
        report_cmd = [
            sys.executable,
            "NOTION_relatoriodeIA_v2.py",
            "--no-gui",
            "--page-url",
            page_url,
            "--source-database-url",
            database_url,
            "--start-date",
            period.start.isoformat(),
            "--end-date",
            period.end.isoformat(),
            "--force-clear",
            "--no-resume",
            "--model",
            report_model,
            "--max-openai-cases",
            str(max_openai_cases),
            "--openai-triage-threshold",
            str(openai_triage_threshold),
            "--enrich-news-gemini",
            "--verbose",
        ]
        run_command(report_cmd, log=log)
        mark_report_generated(period, case_count=current_count)

    mark_files_processed(files, combined_csv=combined_csv, periods=periods)
    log("Fluxo concluido.")


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tkinter nao esta disponivel neste Python.") from exc

    root = tk.Tk()
    root.title("DJE/TSE - Relatorios semanais")
    root.geometry("1080x720")
    root.minsize(900, 560)

    csv_dir_var = tk.StringVar(value=str(DEFAULT_CSV_DIR))
    database_url_var = tk.StringVar(value=report.DEFAULT_SOURCE_DATABASE_URL)
    parent_url_var = tk.StringVar(value=DEFAULT_REPORTS_PARENT_URL)
    status_var = tk.StringVar(value="")
    selected: Dict[str, tk.BooleanVar] = {}
    log_queue: "queue.Queue[str]" = queue.Queue()
    busy = tk.BooleanVar(value=False)

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(2, weight=1)

    top = ttk.LabelFrame(main, text="Entradas", padding=10)
    top.grid(row=0, column=0, sticky="ew")
    top.columnconfigure(1, weight=1)

    ttk.Label(top, text="Pasta DJE").grid(row=0, column=0, sticky="w")
    ttk.Entry(top, textvariable=csv_dir_var).grid(row=0, column=1, sticky="ew", padx=8)

    def browse_dir() -> None:
        path = filedialog.askdirectory(title="Selecione a pasta com CSVs", initialdir=csv_dir_var.get())
        if path:
            csv_dir_var.set(path)
            refresh_files()

    ttk.Button(top, text="Selecionar...", command=browse_dir).grid(row=0, column=2, sticky="e")
    ttk.Label(top, text="Base Notion").grid(row=1, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(top, textvariable=database_url_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
    ttk.Label(top, text="Pagina-mae dos relatorios").grid(row=2, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(top, textvariable=parent_url_var).grid(row=2, column=1, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
    force_regen_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        top,
        text="Regerar relatorios existentes (mesmo sem casos novos)",
        variable=force_regen_var,
    ).grid(row=3, column=1, sticky="w", padx=8, pady=(8, 0))
    ttk.Label(
        top,
        text=(
            f"Config: semana civil seg-dom | modelo {report.DEFAULT_OPENAI_MODEL} | "
            "todos os casos da semana analisados | altos cargos da Republica sempre em destaque"
        ),
        foreground="#555555",
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

    files_box = ttk.LabelFrame(main, text="CSVs", padding=8)
    files_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    files_box.columnconfigure(0, weight=1)
    toolbar = ttk.Frame(files_box)
    toolbar.grid(row=0, column=0, sticky="ew")
    files_frame = ttk.Frame(files_box)
    files_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def log(message: str) -> None:
        log_queue.put(message)

    def refresh_files() -> None:
        for child in files_frame.winfo_children():
            child.destroy()
        selected.clear()
        base = Path(csv_dir_var.get()).expanduser()
        if not base.exists():
            ttk.Label(files_frame, text="Pasta nao encontrada.").grid(row=0, column=0, sticky="w")
            return
        paths = sorted(path for path in base.glob("*.csv") if path.is_file())
        if not paths:
            ttk.Label(files_frame, text="Nenhum CSV encontrado.").grid(row=0, column=0, sticky="w")
            return
        for row, path in enumerate(paths):
            var = tk.BooleanVar(value=is_new_csv(path))
            selected[str(path.resolve())] = var
            label = path.name + ("  [novo]" if var.get() else "  [ja processado]")
            ttk.Checkbutton(files_frame, text=label, variable=var).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        status_var.set(f"{sum(1 for item in selected.values() if item.get())} CSV(s) selecionado(s).")

    def select_all() -> None:
        for var in selected.values():
            var.set(True)

    def select_new() -> None:
        for path, var in selected.items():
            var.set(is_new_csv(Path(path)))

    def add_csvs() -> None:
        paths = filedialog.askopenfilenames(
            title="Selecione CSVs",
            initialdir=csv_dir_var.get(),
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        for raw in paths:
            path = str(Path(raw).resolve())
            if path not in selected:
                selected[path] = tk.BooleanVar(value=True)
                ttk.Checkbutton(files_frame, text=Path(path).name, variable=selected[path]).grid(
                    row=len(selected), column=0, sticky="w", padx=4, pady=2
                )

    ttk.Button(toolbar, text="Atualizar lista", command=refresh_files).pack(side="left")
    ttk.Button(toolbar, text="Selecionar novos", command=select_new).pack(side="left", padx=(8, 0))
    ttk.Button(toolbar, text="Selecionar todos", command=select_all).pack(side="left", padx=(8, 0))
    ttk.Button(toolbar, text="Adicionar CSVs...", command=add_csvs).pack(side="left", padx=(8, 0))

    log_box = ttk.LabelFrame(main, text="Execucao", padding=8)
    log_box.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
    log_box.rowconfigure(0, weight=1)
    log_box.columnconfigure(0, weight=1)
    log_widget = tk.Text(log_box, height=18, wrap="word", state="disabled")
    log_widget.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(log_box, orient="vertical", command=log_widget.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    log_widget.configure(yscrollcommand=scroll.set)

    bottom = ttk.Frame(main)
    bottom.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    ttk.Label(bottom, textvariable=status_var).pack(side="left")

    def drain_log() -> None:
        drained = False
        while True:
            try:
                message = log_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            log_widget.configure(state="normal")
            log_widget.insert("end", message + "\n")
            log_widget.see("end")
            log_widget.configure(state="disabled")
        if drained:
            root.update_idletasks()
        root.after(150, drain_log)

    def run_selected() -> None:
        if busy.get():
            return
        paths = [Path(path) for path, var in selected.items() if var.get()]
        if not paths:
            messagebox.showerror("Sem CSV", "Selecione ao menos um CSV.")
            return
        try:
            report.extract_notion_id_from_url(database_url_var.get())
            report.extract_notion_id_from_url(parent_url_var.get())
        except Exception as exc:
            messagebox.showerror("Configuracao", str(exc))
            return

        busy.set(True)
        status_var.set("Executando...")

        force_regenerate = bool(force_regen_var.get())

        def worker() -> None:
            try:
                process_import_and_generate(
                    paths,
                    database_url=database_url_var.get().strip(),
                    reports_parent_url=parent_url_var.get().strip(),
                    log=log,
                    force_regenerate=force_regenerate,
                )
                root.after(0, lambda: status_var.set("Concluido."))
            except Exception as exc:  # pylint: disable=broad-except
                log(f"ERRO: {exc}")
                root.after(0, lambda: status_var.set("Falhou."))
            finally:
                root.after(0, lambda: busy.set(False))

        threading.Thread(target=worker, daemon=True).start()

    def run_maintenance(extra_args: Sequence[str], descricao: str) -> None:
        if busy.get():
            return
        busy.set(True)
        status_var.set(descricao + "...")

        def worker() -> None:
            try:
                run_command(
                    [
                        sys.executable,
                        "-X",
                        "utf8",
                        "DJE_refazer_relatorios.py",
                        "--parent-page-url",
                        parent_url_var.get().strip() or DEFAULT_REPORTS_PARENT_URL,
                        "--database-url",
                        database_url_var.get().strip() or report.DEFAULT_SOURCE_DATABASE_URL,
                        *extra_args,
                    ],
                    log=log,
                )
                root.after(0, lambda: status_var.set("Concluido."))
            except Exception as exc:  # pylint: disable=broad-except
                log(f"ERRO: {exc}")
                root.after(0, lambda: status_var.set("Falhou."))
            finally:
                root.after(0, lambda: busy.set(False))

        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(
        bottom,
        text="Auditar relatorios (dry-run)",
        command=lambda: run_maintenance(["--dry-run"], "Auditando relatorios"),
    ).pack(side="left", padx=(12, 0))
    ttk.Button(
        bottom,
        text="Refazer todos os relatorios",
        command=lambda: run_maintenance([], "Refazendo relatorios"),
    ).pack(side="left", padx=(8, 0))
    ttk.Button(bottom, text="Processar, importar e gerar relatorios", command=run_selected).pack(side="right")

    refresh_files()
    drain_log()
    root.mainloop()


def main() -> int:
    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
