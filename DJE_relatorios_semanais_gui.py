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
DEFAULT_CSV_DIR = PROJECT_ROOT / "dje"
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
    return len({page_id for page_ids in existing.values() for page_id in page_ids})


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

    # Etapa final automatica: garante que altos cargos e casos marcados
    # 'Dep. Federal' estejam nos destaques — apenas nas semanas processadas
    # nesta execucao (economia: nao varre a colecao inteira).
    for period in periods:
        log(f"Etapa final: varredura do alvo em {period.title}...")
        run_command(
            [
                sys.executable,
                "-X",
                "utf8",
                "DJE_refazer_relatorios.py",
                "--parent-page-url",
                reports_parent_url,
                "--database-url",
                database_url,
                "--varredura-alvo",
                "--fix",
                "--only-title",
                period.title,
            ],
            log=log,
        )

    # So marca os CSVs como processados quando TODO o fluxo (inclusive a
    # varredura) concluiu: uma falha no meio preserva o pending_run e a
    # re-execucao retoma barato, sem reconsolidar nem repetir chamadas de IA.
    mark_files_processed(files, combined_csv=combined_csv, periods=periods)
    log("Fluxo concluido.")


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tkinter nao esta disponivel neste Python.") from exc

    import webbrowser

    root = tk.Tk()
    root.title("DJe/TSE — Relatórios semanais")
    root.geometry("880x620")
    root.minsize(780, 540)

    def _add_tooltip(widget: Any, text: str) -> None:
        tip: Dict[str, Any] = {"win": None}

        def _show(_event: Any) -> None:
            if tip["win"] is not None:
                return
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{widget.winfo_rootx() + 10}+{widget.winfo_rooty() + widget.winfo_height() + 6}")
            tk.Label(
                win,
                text=text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                wraplength=420,
                font=("Segoe UI", 9),
            ).pack(ipadx=6, ipady=4)
            tip["win"] = win

        def _hide(_event: Any) -> None:
            if tip["win"] is not None:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", _show)
        widget.bind("<Leave>", _hide)

    csv_dir_var = tk.StringVar(value=str(DEFAULT_CSV_DIR))
    database_url_var = tk.StringVar(value=report.DEFAULT_SOURCE_DATABASE_URL)
    parent_url_var = tk.StringVar(value=DEFAULT_REPORTS_PARENT_URL)
    status_var = tk.StringVar(value="Pronto.")
    count_var = tk.StringVar(value="Nenhum CSV carregado.")
    force_regen_var = tk.BooleanVar(value=False)
    show_advanced_var = tk.BooleanVar(value=False)
    log_queue: "queue.Queue[str]" = queue.Queue()
    busy = tk.BooleanVar(value=False)
    row_state: Dict[str, Dict[str, Any]] = {}

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(1, weight=2)
    main.rowconfigure(4, weight=3)

    # ------------------------------------------------------------------ 1) Arquivos
    src_box = ttk.LabelFrame(main, text="1) Arquivos do DJe (CSV)", padding=10)
    src_box.grid(row=0, column=0, sticky="ew")
    src_box.columnconfigure(1, weight=1)

    ttk.Label(src_box, text="Pasta:").grid(row=0, column=0, sticky="w")
    ttk.Entry(src_box, textvariable=csv_dir_var).grid(row=0, column=1, sticky="ew", padx=8)

    tree_box = ttk.Frame(main)
    tree_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
    tree_box.columnconfigure(0, weight=1)
    tree_box.rowconfigure(0, weight=1)

    tree = ttk.Treeview(
        tree_box,
        columns=("status", "modificado", "tamanho"),
        show="tree headings",
        selectmode="none",
        height=7,
    )
    tree.heading("#0", text="Arquivo  (clique para marcar)")
    tree.heading("status", text="Situação")
    tree.heading("modificado", text="Modificado em")
    tree.heading("tamanho", text="Tamanho")
    tree.column("#0", width=330, minwidth=220, anchor="w")
    tree.column("status", width=105, minwidth=90, anchor="center", stretch=False)
    tree.column("modificado", width=125, minwidth=110, anchor="center", stretch=False)
    tree.column("tamanho", width=75, minwidth=65, anchor="e", stretch=False)
    tree.grid(row=0, column=0, sticky="nsew")
    tree.tag_configure("novo", foreground="#0a6b22")
    tree.tag_configure("processado", foreground="#666666")

    tree_scroll = ttk.Scrollbar(tree_box, orient="vertical", command=tree.yview)
    tree_scroll.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=tree_scroll.set)

    def _update_count() -> None:
        total = len(row_state)
        marcados = sum(1 for info in row_state.values() if info["selected"])
        novos = sum(1 for info in row_state.values() if info["new"])
        count_var.set(f"{total} CSV(s) na pasta | {novos} novo(s) | {marcados} selecionado(s) para processar")

    def _set_row_mark(item_id: str, value: bool) -> None:
        info = row_state.get(item_id)
        if not info:
            return
        info["selected"] = value
        prefix = "☑" if value else "☐"
        tree.item(item_id, text=f"{prefix}  {info['name']}")

    def refresh_files() -> None:
        for item_id in tree.get_children(""):
            tree.delete(item_id)
        row_state.clear()
        base = Path(csv_dir_var.get()).expanduser()
        if not base.exists():
            count_var.set("Pasta nao encontrada.")
            return
        paths = sorted((p for p in base.glob("*.csv") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths:
            novo = is_new_csv(path)
            stat = path.stat()
            item_id = tree.insert(
                "",
                "end",
                text="",
                values=(
                    "novo" if novo else "ja processado",
                    datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m/%Y %H:%M"),
                    f"{stat.st_size / (1024 * 1024):.1f} MB",
                ),
                tags=("novo" if novo else "processado",),
            )
            row_state[item_id] = {"path": str(path.resolve()), "name": path.name, "new": novo, "selected": False}
            _set_row_mark(item_id, novo)
        _update_count()

    def _on_tree_click(event: Any) -> None:
        item_id = tree.identify_row(event.y)
        if item_id in row_state:
            _set_row_mark(item_id, not row_state[item_id]["selected"])
            _update_count()

    tree.bind("<Button-1>", _on_tree_click)

    def select_only_new() -> None:
        for item_id, info in row_state.items():
            _set_row_mark(item_id, info["new"])
        _update_count()

    def select_all() -> None:
        for item_id in row_state:
            _set_row_mark(item_id, True)
        _update_count()

    def select_none() -> None:
        for item_id in row_state:
            _set_row_mark(item_id, False)
        _update_count()

    def browse_dir() -> None:
        path = filedialog.askdirectory(title="Selecione a pasta com os CSVs do DJe", initialdir=csv_dir_var.get())
        if path:
            csv_dir_var.set(path)
            refresh_files()

    ttk.Button(src_box, text="Escolher pasta...", command=browse_dir).grid(row=0, column=2)
    ttk.Button(src_box, text="Atualizar lista", command=refresh_files).grid(row=0, column=3, padx=(8, 0))

    sel_bar = ttk.Frame(main)
    sel_bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
    btn_sel_new = ttk.Button(sel_bar, text="Marcar novos", command=select_only_new)
    btn_sel_new.pack(side="left")
    _add_tooltip(btn_sel_new, "Marca apenas os CSVs que ainda não foram processados (situação 'novo').\nÉ a escolha do dia a dia.")
    btn_sel_all = ttk.Button(sel_bar, text="Marcar todos", command=select_all)
    btn_sel_all.pack(side="left", padx=(8, 0))
    _add_tooltip(btn_sel_all, "Marca todos os CSVs da pasta, inclusive os já processados.")
    btn_sel_none = ttk.Button(sel_bar, text="Desmarcar tudo", command=select_none)
    btn_sel_none.pack(side="left", padx=(8, 0))
    ttk.Label(sel_bar, textvariable=count_var, foreground="#444444").pack(side="left", padx=(16, 0))

    # ------------------------------------------------------------------ 2) Acoes
    actions = ttk.LabelFrame(main, text="2) Ações", padding=10)
    actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    actions.columnconfigure(6, weight=1)

    action_buttons: List[Any] = []

    def log(message: str) -> None:
        log_queue.put(message)

    def set_busy(value: bool, status_text: str = "") -> None:
        busy.set(value)
        state = "disabled" if value else "normal"
        for button in action_buttons:
            button.configure(state=state)
        if value:
            progress.start(12)
        else:
            progress.stop()
            progress["value"] = 0
        if status_text:
            status_var.set(status_text)

    def _run_in_thread(target: Any, done_message: str) -> None:
        def worker() -> None:
            try:
                target()
                root.after(0, lambda: set_busy(False, done_message))
            except Exception as exc:  # pylint: disable=broad-except
                log(f"ERRO: {exc}")
                root.after(0, lambda: set_busy(False, "Falhou. Veja o log."))
            finally:
                root.after(0, refresh_files)

        threading.Thread(target=worker, daemon=True).start()

    def run_selected() -> None:
        if busy.get():
            return
        paths = [Path(info["path"]) for info in row_state.values() if info["selected"]]
        if not paths:
            messagebox.showinfo("Nada selecionado", "Marque ao menos um CSV na lista (clique na linha).")
            return
        try:
            report.extract_notion_id_from_url(database_url_var.get())
            report.extract_notion_id_from_url(parent_url_var.get())
        except Exception as exc:
            messagebox.showerror("Configuracao", str(exc))
            return

        force_regenerate = bool(force_regen_var.get())
        set_busy(True, f"Processando {len(paths)} CSV(s)...")
        _run_in_thread(
            lambda: process_import_and_generate(
                paths,
                database_url=database_url_var.get().strip(),
                reports_parent_url=parent_url_var.get().strip(),
                log=log,
                force_regenerate=force_regenerate,
            ),
            "Concluido.",
        )

    def run_maintenance(extra_args: Sequence[str], descricao: str) -> None:
        if busy.get():
            return
        set_busy(True, descricao + "...")
        command = [
            sys.executable,
            "-X",
            "utf8",
            "DJE_refazer_relatorios.py",
            "--parent-page-url",
            parent_url_var.get().strip() or DEFAULT_REPORTS_PARENT_URL,
            "--database-url",
            database_url_var.get().strip() or report.DEFAULT_SOURCE_DATABASE_URL,
            *extra_args,
        ]
        _run_in_thread(lambda: run_command(command, log=log), "Concluido.")

    btn_process = ttk.Button(actions, text="▶  Gerar relatórios", command=run_selected)
    btn_process.grid(row=0, column=0, sticky="w")
    action_buttons.append(btn_process)
    _add_tooltip(
        btn_process,
        "Fluxo principal: consolida os CSVs marcados na lista, atualiza a base no Notion\n"
        "e cria/atualiza os relatórios semanais. Use sempre que baixar CSVs novos do DJe.",
    )

    chk_regen = ttk.Checkbutton(actions, text="Regerar existentes", variable=force_regen_var)
    chk_regen.grid(row=0, column=1, sticky="w", padx=(14, 0))

    ttk.Label(actions, text="Manutenção:", foreground="#555555").grid(row=1, column=0, sticky="w", pady=(10, 0))
    _add_tooltip(
        chk_regen,
        "Marque para reescrever também as semanas que já têm relatório,\n"
        "mesmo sem casos novos na base.",
    )

    btn_audit = ttk.Button(
        actions,
        text="Auditar (só verificar)",
        command=lambda: run_maintenance(["--dry-run"], "Auditando relatorios"),
    )
    btn_audit.grid(row=1, column=1, sticky="w", padx=(14, 0), pady=(10, 0))
    action_buttons.append(btn_audit)
    _add_tooltip(
        btn_audit,
        "Apenas verifica: compara cada relatório com a base e lista pendências no log.\n"
        "Não altera nada no Notion.",
    )

    btn_alvo = ttk.Button(
        actions,
        text="Corrigir omissões do alvo",
        command=lambda: run_maintenance(["--varredura-alvo", "--fix"], "Varrendo altos cargos"),
    )
    btn_alvo.grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(10, 0))
    action_buttons.append(btn_alvo)
    _add_tooltip(
        btn_alvo,
        "Procura processos de altos cargos da República — ou marcados com o checkbox\n"
        "'Dep. Federal' na base — que ficaram fora dos destaques, e refaz somente\n"
        "as semanas com omissão.",
    )

    btn_stale = ttk.Button(
        actions,
        text="Atualizar desatualizados",
        command=lambda: run_maintenance(["--somente-desatualizados"], "Atualizando semanas desatualizadas"),
    )
    btn_stale.grid(row=1, column=3, sticky="w", padx=(8, 0), pady=(10, 0))
    action_buttons.append(btn_stale)
    _add_tooltip(
        btn_stale,
        "Regenera apenas as semanas cuja quantidade de casos na base mudou desde\n"
        "o último relatório (ex.: após edições/correções na base). Econômico:\n"
        "reaproveita as análises já feitas.",
    )

    btn_refazer = ttk.Button(
        actions,
        text="Refazer todos",
        command=lambda: run_maintenance([], "Refazendo relatorios"),
    )
    btn_refazer.grid(row=1, column=4, sticky="w", padx=(8, 0), pady=(10, 0))
    action_buttons.append(btn_refazer)
    _add_tooltip(
        btn_refazer,
        "Apaga e regenera TODOS os relatórios semanais do zero.\n"
        "Demorado (horas); use apenas após mudanças grandes.",
    )

    links_bar = ttk.Frame(actions)
    links_bar.grid(row=2, column=0, columnspan=7, sticky="w", pady=(8, 0))
    ttk.Button(
        links_bar,
        text="Abrir relatórios no Notion",
        command=lambda: webbrowser.open(parent_url_var.get().strip() or DEFAULT_REPORTS_PARENT_URL),
    ).pack(side="left")
    ttk.Button(
        links_bar,
        text="Abrir base de casos no Notion",
        command=lambda: webbrowser.open(database_url_var.get().strip() or report.DEFAULT_SOURCE_DATABASE_URL),
    ).pack(side="left", padx=(8, 0))
    ttk.Checkbutton(links_bar, text="Configurações avançadas", variable=show_advanced_var).pack(side="left", padx=(20, 0))

    ttk.Label(
        actions,
        text=(
            f"Semana seg-dom | modelo {report.DEFAULT_OPENAI_MODEL} | altos cargos e casos marcados "
            "'Dep. Federal' sempre em destaque | passe o mouse nos botões para ver o que cada um faz"
        ),
        foreground="#555555",
    ).grid(row=3, column=0, columnspan=7, sticky="w", pady=(8, 0))

    adv = ttk.Frame(actions)
    adv.columnconfigure(1, weight=1)
    ttk.Label(adv, text="Base Notion:").grid(row=0, column=0, sticky="w")
    ttk.Entry(adv, textvariable=database_url_var).grid(row=0, column=1, sticky="ew", padx=8)
    ttk.Label(adv, text="Página-mãe dos relatórios:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(adv, textvariable=parent_url_var).grid(row=1, column=1, sticky="ew", padx=8, pady=(6, 0))

    def _toggle_advanced(*_args: Any) -> None:
        if show_advanced_var.get():
            adv.grid(row=4, column=0, columnspan=7, sticky="ew", pady=(8, 0))
        else:
            adv.grid_remove()

    show_advanced_var.trace_add("write", _toggle_advanced)

    # ------------------------------------------------------------------ 3) Execucao
    run_box = ttk.LabelFrame(main, text="3) Execução", padding=10)
    run_box.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
    run_box.columnconfigure(0, weight=1)
    run_box.rowconfigure(2, weight=1)

    progress = ttk.Progressbar(run_box, mode="indeterminate")
    progress.grid(row=0, column=0, columnspan=2, sticky="ew")

    status_bar = ttk.Frame(run_box)
    status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 4))
    ttk.Label(status_bar, textvariable=status_var, font=("Segoe UI", 9, "bold")).pack(side="left")

    def clear_log() -> None:
        log_widget.configure(state="normal")
        log_widget.delete("1.0", "end")
        log_widget.configure(state="disabled")

    ttk.Button(status_bar, text="Limpar log", command=clear_log).pack(side="right")

    log_widget = tk.Text(run_box, height=12, wrap="word", state="disabled")
    log_widget.grid(row=2, column=0, sticky="nsew")
    log_scroll = ttk.Scrollbar(run_box, orient="vertical", command=log_widget.yview)
    log_scroll.grid(row=2, column=1, sticky="ns")
    log_widget.configure(yscrollcommand=log_scroll.set)

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

    def _on_close() -> None:
        if busy.get():
            confirmar = messagebox.askyesno(
                "Execução em andamento",
                "Há um processamento em andamento. Fechar agora pode deixar um relatório pela metade.\n\nFechar mesmo assim?",
            )
            if not confirmar:
                return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    refresh_files()
    drain_log()
    root.mainloop()


def main() -> int:
    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
