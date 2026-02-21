#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline final consolidado para TRF1 (PDF -> CSV).

Etapas:
1) Extração determinística dos PDFs.
2) Enriquecimento OpenAI (campos analíticos) + geração principal de tema.
3) Enriquecimento Perplexity (notícias) com política de domínio e checkpoint.
4) Pós-padronização final:
   - classe
   - relator(a)
   - ramo/subramo
   - legislação

Saída padrão:
- boletins_de_jurisprudencia_TRF1_with_news.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import TRF1_pdf_to_csv_viaAPI_openai as core

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore[assignment]


DEFAULT_OUTPUT_NAME = "boletins_de_jurisprudencia_TRF1_with_news.csv"
DEFAULT_CHECKPOINT_NAME = ".boletins_de_jurisprudencia_TRF1_with_news.checkpoint.json"
DEFAULT_QUALITY_REPORT_NAME = ".boletins_de_jurisprudencia_TRF1_with_news.quality_report.json"

# Mantém o layout já consolidado no CSV ideal atual.
FINAL_CSV_COLUMNS: List[str] = [
    "tema",
    "bullet_points",
    "classe",
    "contexto",
    "data_julgamento",
    "data_publicacao",
    "decisao",
    "informativo",
    "jurisprudência",
    "legislacao",
    "noticia_1",
    "noticia_2",
    "numero_processo",
    "orgao_julgador",
    "punchline",
    "ramo_do_direito",
    "relator(a)",
    "subramo_do_direito",
    "tese",
    "texto_do_boletim",
    "tribunal",
]


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


RE_WS = re.compile(r"\s+")


def normalize_ws_local(value: Any) -> str:
    return RE_WS.sub(" ", str(value or "")).strip()


def fold_text(value: str) -> str:
    text = normalize_ws_local(value).casefold()
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def slug_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", fold_text(value))


def load_text_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def load_classe_rules(config_path: Path, logger: logging.Logger) -> Tuple[List[str], Dict[str, str]]:
    if not config_path.exists() or not config_path.is_file():
        logger.warning("Padronização de classe ignorada: arquivo de regras ausente (%s).", config_path)
        return [], {}

    try:
        payload = load_text_or_json(config_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Padronização de classe ignorada: não foi possível ler %s (%s).", config_path, exc)
        return [], {}

    if not isinstance(payload, dict):
        logger.warning("Padronização de classe ignorada: estrutura inválida em %s.", config_path)
        return [], {}

    canonical = [normalize_ws_local(x) for x in payload.get("canonical", []) if normalize_ws_local(x)]
    aliases = {
        slug_alnum(str(k)): normalize_ws_local(v)
        for k, v in dict(payload.get("aliases", {})).items()
        if slug_alnum(str(k)) and normalize_ws_local(v)
    }
    return canonical, aliases


def normalize_classe_with_rules(value: str, canonical: Sequence[str], aliases: Mapping[str, str]) -> str:
    text = normalize_ws_local(value)
    if not text:
        return ""

    canonical_set = set(canonical)
    if text in canonical_set:
        return text

    key = slug_alnum(text)
    if key in aliases:
        return aliases[key]

    for cand in canonical:
        if slug_alnum(cand) == key:
            return cand
    return text


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    tmp.replace(path)


def apply_classe_standardization(
    *,
    output_csv: Path,
    report_dir: Path,
    classe_config_file: Path,
    logger: logging.Logger,
) -> None:
    canonical, aliases = load_classe_rules(classe_config_file, logger)
    if not canonical:
        return

    if not output_csv.exists() or not output_csv.is_file():
        logger.warning("Padronização de classe ignorada: CSV ausente (%s).", output_csv)
        return

    with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            logger.warning("Padronização de classe ignorada: CSV sem cabeçalho (%s).", output_csv)
            return
        if "classe" not in fieldnames:
            logger.warning("Padronização de classe ignorada: coluna 'classe' não encontrada.")
            return
        rows = [dict(row) for row in reader]

    changed_rows: List[Dict[str, str]] = []
    changed = 0
    for idx, row in enumerate(rows, start=2):
        before = normalize_ws_local(row.get("classe", ""))
        after = normalize_classe_with_rules(before, canonical, aliases)
        if after != before:
            row["classe"] = after
            changed += 1
            changed_rows.append(
                {
                    "row_number": str(idx),
                    "numero_processo": normalize_ws_local(row.get("numero_processo", "")),
                    "before": before,
                    "after": after,
                }
            )

    if changed <= 0:
        logger.info("Padronização de classe: nenhuma alteração necessária.")
        return

    tag = timestamp_tag()
    backup_path = output_csv.with_name(f"{output_csv.name}.backup_classe_{tag}")
    shutil.copy2(output_csv, backup_path)
    write_csv_atomic(output_csv, fieldnames, rows)

    report_applied = report_dir / f"pipeline_classe_{tag}_aplicadas.csv"
    report_summary = report_dir / f"pipeline_classe_{tag}_summary.json"
    write_csv_atomic(
        report_applied,
        ("row_number", "numero_processo", "before", "after"),
        changed_rows,
    )
    report_summary.write_text(
        json.dumps(
            {
                "timestamp": tag,
                "input_csv": str(output_csv),
                "backup_csv": str(backup_path),
                "rows_total": len(rows),
                "rows_changed": changed,
                "rules_canonical_count": len(canonical),
                "rules_alias_count": len(aliases),
                "report_applied": str(report_applied),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Padronização de classe concluída: %d linhas alteradas | backup=%s | relatório=%s",
        changed,
        backup_path,
        report_applied,
    )


def add_final_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = (
        "Pipeline consolidado TRF1: PDF -> CSV final com OpenAI + Perplexity + padronizações."
    )

    parser.add_argument(
        "--pipeline-profile",
        choices=("balanced", "economico", "velocidade"),
        default="balanced",
        help="Preset de custo/velocidade para parâmetros de API.",
    )

    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--quality-report-name", default=DEFAULT_QUALITY_REPORT_NAME)
    parser.add_argument("--checkpoint-file", default="", help="Caminho customizado do checkpoint.")
    parser.add_argument("--quality-report-file", default="", help="Caminho customizado do quality report.")

    parser.add_argument("--backup-dir", default="", help="Diretório para backups de segurança no início do run.")
    parser.add_argument("--no-startup-backup", action="store_true", help="Desativa backup preventivo inicial.")

    parser.add_argument("--disable-perplexity", action="store_true")
    parser.add_argument("--perplexity-api-key", default="", help="API key Perplexity.")
    parser.add_argument("--perplexity-model", default="sonar", help="Modelo Perplexity.")
    parser.add_argument("--perplexity-batch-size", type=int, default=3)
    parser.add_argument("--perplexity-max-workers", type=int, default=core.PERPLEXITY_DEFAULT_MAX_WORKERS)
    parser.add_argument("--perplexity-max-workers-cap", type=int, default=6)
    parser.add_argument("--perplexity-delay", type=float, default=core.PERPLEXITY_DEFAULT_DELAY)
    parser.add_argument("--perplexity-retries", type=int, default=core.PERPLEXITY_DEFAULT_RETRIES)
    parser.add_argument("--perplexity-timeout", type=int, default=40)
    parser.add_argument(
        "--perplexity-scaling-mode",
        choices=list(core.PERPLEXITY_SCALING_MODE_CHOICES),
        default=core.PERPLEXITY_DEFAULT_SCALING_MODE,
    )
    parser.add_argument("--perplexity-target-rpm", type=int, default=core.PERPLEXITY_DEFAULT_TARGET_RPM)
    parser.add_argument("--perplexity-resume-rate-state", action="store_true")
    parser.add_argument("--perplexity-max-tokens", type=int, default=core.PERPLEXITY_DEFAULT_MAX_TOKENS)
    parser.add_argument("--perplexity-text-max-chars", type=int, default=core.PERPLEXITY_DEFAULT_TEXT_MAX_CHARS)
    parser.add_argument("--perplexity-min-score-mainstream", type=int, default=2)
    parser.add_argument("--perplexity-min-score-official", type=int, default=1)
    parser.add_argument(
        "--perplexity-domain-policy",
        choices=list(core.PERPLEXITY_DOMAIN_POLICY_CHOICES),
        default=core.PERPLEXITY_DEFAULT_DOMAIN_POLICY,
    )
    parser.add_argument("--perplexity-fallback-enabled", action="store_true")
    parser.add_argument(
        "--perplexity-fallback-max-fraction",
        type=float,
        default=core.PERPLEXITY_DEFAULT_FALLBACK_MAX_FRACTION,
    )
    parser.add_argument(
        "--perplexity-allow-terminal-retry",
        action="store_true",
        help="Permite tentar novamente linhas com razões terminais do Perplexity.",
    )

    parser.add_argument(
        "--enable-final-tema-review-pass",
        action="store_true",
        help="Executa um passe extra opcional de revisão de tema via script dedicado.",
    )
    parser.add_argument(
        "--final-tema-review-mode",
        choices=("invalid_only", "all"),
        default="invalid_only",
    )

    parser.add_argument("--disable-post-processing", action="store_true")
    parser.add_argument("--disable-classe-standardization", action="store_true")
    parser.add_argument("--disable-relator-standardization", action="store_true")
    parser.add_argument("--disable-ramo-subramo-standardization", action="store_true")
    parser.add_argument("--disable-legislacao-standardization", action="store_true")
    parser.add_argument("--classe-config-file", default="canon_config/classe_canonical.yaml")

    parser.add_argument("--report-dir", default="reports")

    parser.add_argument(
        "--ramo-subramo-scope",
        choices=("smart", "problematic", "all"),
        default="smart",
    )
    parser.add_argument("--ramo-subramo-max-rows", type=int, default=0)
    parser.add_argument("--ramo-subramo-cache-file", default=".checkpoint/ramo_subramo_openai_cache.json")
    parser.add_argument("--disable-ramo-subramo-openai", action="store_true")
    parser.add_argument("--ramo-subramo-openai-model", default="gpt-5-mini")
    parser.add_argument("--ramo-subramo-openai-max-workers", type=int, default=16)
    parser.add_argument("--ramo-subramo-openai-batch-size", type=int, default=80)
    parser.add_argument("--ramo-subramo-openai-delay", type=float, default=0.0)
    parser.add_argument(
        "--gui-panel",
        action="store_true",
        help="Força abertura do painel GUI completo para configurar a execução.",
    )

    return parser


def should_open_gui_panel(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_gui", False)):
        return False
    if bool(getattr(args, "gui_panel", False)):
        return True
    # Sem input explícito, o painel completo vira o fluxo GUI padrão.
    return not bool(list(args.input_dirs or []) or list(args.input_files or []))


def _listbox_values(listbox: Any) -> List[str]:
    return [str(x) for x in listbox.get(0, "end")]


def _set_listbox_values(listbox: Any, values: Sequence[str]) -> None:
    listbox.delete(0, "end")
    for item in values:
        listbox.insert("end", str(item))


def _dedupe_paths(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in paths:
        txt = normalize_ws_local(raw)
        if not txt:
            continue
        key = txt.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)
    return out


def open_gui_control_panel(args: argparse.Namespace) -> bool:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:  # noqa: BLE001
        print(f"Aviso: GUI indisponível ({exc}). Prosseguindo sem painel.")
        return True

    root = tk.Tk()
    root.title("TRF1 Pipeline Final - Painel de Execução")
    root.geometry("1180x760")
    root.minsize(1020, 680)

    output_dir_default = str(Path(args.output_dir).resolve())
    mode_default = "auto"
    if list(args.input_files or []):
        mode_default = "files"
    elif list(args.input_dirs or []):
        mode_default = "dirs"

    mode_var = tk.StringVar(value=mode_default)
    profile_var = tk.StringVar(value=str(args.pipeline_profile))
    output_dir_var = tk.StringVar(value=output_dir_default)
    verbosity_var = tk.StringVar(
        value="verbose" if bool(args.verbose) else ("quiet" if bool(args.quiet) else "normal")
    )

    openai_var = tk.BooleanVar(value=(not bool(args.disable_openai)))
    perplexity_var = tk.BooleanVar(value=(not bool(args.disable_perplexity)))
    post_var = tk.BooleanVar(value=(not bool(args.disable_post_processing)))
    classe_var = tk.BooleanVar(value=(not bool(args.disable_classe_standardization)))
    relator_var = tk.BooleanVar(value=(not bool(args.disable_relator_standardization)))
    ramo_sub_var = tk.BooleanVar(value=(not bool(args.disable_ramo_subramo_standardization)))
    legislacao_var = tk.BooleanVar(value=(not bool(args.disable_legislacao_standardization)))
    tema_pass_var = tk.BooleanVar(value=bool(args.enable_final_tema_review_pass))
    startup_backup_var = tk.BooleanVar(value=(not bool(args.no_startup_backup)))

    summary_var = tk.StringVar(
        value=(
            "Selecione o modo de entrada e as etapas do pipeline. "
            "Clique em Iniciar para executar."
        )
    )

    top_frame = ttk.Frame(root, padding=(12, 10))
    top_frame.pack(fill="x")
    ttk.Label(
        top_frame,
        text="Processador Definitivo TRF1",
        font=("Segoe UI", 14, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        top_frame,
        text=(
            "Painel para configurar entrada, perfil, APIs, pós-padronizações e logs "
            "antes de iniciar o processamento."
        ),
    ).pack(anchor="w", pady=(2, 0))

    body = ttk.Frame(root, padding=(12, 8))
    body.pack(fill="both", expand=True)

    # Entrada
    input_box = ttk.LabelFrame(body, text="Entrada", padding=10)
    input_box.pack(fill="x", pady=(0, 8))

    mode_row = ttk.Frame(input_box)
    mode_row.pack(fill="x", pady=(0, 8))
    ttk.Label(mode_row, text="Modo de entrada:").pack(side="left")
    ttk.Radiobutton(
        mode_row,
        text="Auto (pastas 2020..2026)",
        variable=mode_var,
        value="auto",
    ).pack(side="left", padx=(12, 0))
    ttk.Radiobutton(
        mode_row,
        text="Selecionar pastas",
        variable=mode_var,
        value="dirs",
    ).pack(side="left", padx=(12, 0))
    ttk.Radiobutton(
        mode_row,
        text="Selecionar PDFs",
        variable=mode_var,
        value="files",
    ).pack(side="left", padx=(12, 0))

    lists_row = ttk.Frame(input_box)
    lists_row.pack(fill="both", expand=True)

    dirs_col = ttk.LabelFrame(lists_row, text="Pastas selecionadas", padding=8)
    dirs_col.pack(side="left", fill="both", expand=True, padx=(0, 8))
    dirs_list = tk.Listbox(dirs_col, height=6, exportselection=False)
    dirs_list.pack(fill="both", expand=True)
    for d in _dedupe_paths([str(Path(x).expanduser()) for x in list(args.input_dirs or [])]):
        dirs_list.insert("end", d)

    dirs_btn_row = ttk.Frame(dirs_col)
    dirs_btn_row.pack(fill="x", pady=(8, 0))

    files_col = ttk.LabelFrame(lists_row, text="PDFs selecionados", padding=8)
    files_col.pack(side="left", fill="both", expand=True)
    files_list = tk.Listbox(files_col, height=6, exportselection=False)
    files_list.pack(fill="both", expand=True)
    for f in _dedupe_paths([str(Path(x).expanduser()) for x in list(args.input_files or [])]):
        files_list.insert("end", f)

    files_btn_row = ttk.Frame(files_col)
    files_btn_row.pack(fill="x", pady=(8, 0))

    def add_dir() -> None:
        picked = filedialog.askdirectory(title="Selecione uma pasta com PDFs do TRF1")
        if not picked:
            return
        current = _listbox_values(dirs_list)
        current.append(str(Path(picked).expanduser()))
        _set_listbox_values(dirs_list, _dedupe_paths(current))

    def remove_selected_dir() -> None:
        selected = list(dirs_list.curselection())
        selected.reverse()
        for idx in selected:
            dirs_list.delete(idx)

    def clear_dirs() -> None:
        dirs_list.delete(0, "end")

    ttk.Button(dirs_btn_row, text="Adicionar pasta", command=add_dir).pack(side="left")
    ttk.Button(dirs_btn_row, text="Remover selecionada", command=remove_selected_dir).pack(side="left", padx=6)
    ttk.Button(dirs_btn_row, text="Limpar", command=clear_dirs).pack(side="left")

    def add_files() -> None:
        picked = filedialog.askopenfilenames(
            title="Selecione PDFs do TRF1",
            filetypes=[("PDF", "*.pdf"), ("Todos os arquivos", "*.*")],
        )
        if not picked:
            return
        current = _listbox_values(files_list)
        current.extend(str(Path(x).expanduser()) for x in picked)
        _set_listbox_values(files_list, _dedupe_paths(current))

    def remove_selected_files() -> None:
        selected = list(files_list.curselection())
        selected.reverse()
        for idx in selected:
            files_list.delete(idx)

    def clear_files() -> None:
        files_list.delete(0, "end")

    ttk.Button(files_btn_row, text="Adicionar PDFs", command=add_files).pack(side="left")
    ttk.Button(files_btn_row, text="Remover selecionados", command=remove_selected_files).pack(side="left", padx=6)
    ttk.Button(files_btn_row, text="Limpar", command=clear_files).pack(side="left")

    # Configuração geral
    general_box = ttk.LabelFrame(body, text="Configuração Geral", padding=10)
    general_box.pack(fill="x", pady=(0, 8))

    general_row_1 = ttk.Frame(general_box)
    general_row_1.pack(fill="x")
    ttk.Label(general_row_1, text="Perfil:").pack(side="left")
    profile_combo = ttk.Combobox(
        general_row_1,
        values=["balanced", "economico", "velocidade"],
        textvariable=profile_var,
        state="readonly",
        width=18,
    )
    profile_combo.pack(side="left", padx=(8, 18))

    ttk.Label(general_row_1, text="Log:").pack(side="left")
    ttk.Radiobutton(general_row_1, text="Normal", variable=verbosity_var, value="normal").pack(side="left", padx=(8, 0))
    ttk.Radiobutton(general_row_1, text="Verbose", variable=verbosity_var, value="verbose").pack(side="left", padx=(8, 0))
    ttk.Radiobutton(general_row_1, text="Quiet", variable=verbosity_var, value="quiet").pack(side="left", padx=(8, 0))
    ttk.Checkbutton(general_row_1, text="Backup inicial", variable=startup_backup_var).pack(side="right")

    general_row_2 = ttk.Frame(general_box)
    general_row_2.pack(fill="x", pady=(10, 0))
    ttk.Label(general_row_2, text="Pasta de saída:").pack(side="left")
    output_entry = ttk.Entry(general_row_2, textvariable=output_dir_var)
    output_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))

    def pick_output_dir() -> None:
        picked = filedialog.askdirectory(title="Selecione a pasta de saída")
        if picked:
            output_dir_var.set(str(Path(picked).expanduser()))

    ttk.Button(general_row_2, text="Selecionar...", command=pick_output_dir).pack(side="left")

    # Etapas
    stage_box = ttk.LabelFrame(body, text="Etapas do Pipeline", padding=10)
    stage_box.pack(fill="x", pady=(0, 8))

    row_a = ttk.Frame(stage_box)
    row_a.pack(fill="x")
    ttk.Checkbutton(row_a, text="Habilitar OpenAI (enriquecimento + tema)", variable=openai_var).pack(side="left")
    ttk.Checkbutton(row_a, text="Habilitar Perplexity (notícias)", variable=perplexity_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(row_a, text="Passe final opcional de tema", variable=tema_pass_var).pack(side="left", padx=(12, 0))

    row_b = ttk.Frame(stage_box)
    row_b.pack(fill="x", pady=(8, 0))
    ttk.Checkbutton(row_b, text="Executar pós-processamento", variable=post_var).pack(side="left")
    ttk.Checkbutton(row_b, text="Padronizar classe", variable=classe_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(row_b, text="Padronizar relator(a)", variable=relator_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(row_b, text="Padronizar ramo/subramo", variable=ramo_sub_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(row_b, text="Padronizar legislação", variable=legislacao_var).pack(side="left", padx=(12, 0))

    status_box = ttk.LabelFrame(body, text="Resumo", padding=10)
    status_box.pack(fill="x")
    ttk.Label(status_box, textvariable=summary_var).pack(anchor="w")

    button_row = ttk.Frame(root, padding=(12, 8))
    button_row.pack(fill="x")

    result: Dict[str, bool] = {"ok": False}

    def _validate_mode_selection(mode: str, dirs: Sequence[str], files: Sequence[str]) -> bool:
        if mode == "dirs" and not dirs:
            messagebox.showerror("TRF1", "Selecione ao menos uma pasta ou troque para modo Auto.")
            return False
        if mode == "files" and not files:
            messagebox.showerror("TRF1", "Selecione ao menos um PDF ou troque para modo Auto.")
            return False
        return True

    def on_start() -> None:
        mode = mode_var.get().strip()
        selected_dirs = _dedupe_paths(_listbox_values(dirs_list))
        selected_files = _dedupe_paths(_listbox_values(files_list))
        if not _validate_mode_selection(mode, selected_dirs, selected_files):
            return

        out_dir = normalize_ws_local(output_dir_var.get())
        if not out_dir:
            messagebox.showerror("TRF1", "Informe uma pasta de saída válida.")
            return

        args.pipeline_profile = profile_var.get().strip() or "balanced"
        args.output_dir = out_dir

        if mode == "dirs":
            args.input_dirs = selected_dirs
            args.input_files = []
        elif mode == "files":
            args.input_dirs = []
            args.input_files = selected_files
        else:
            args.input_dirs = []
            args.input_files = []

        args.disable_openai = not bool(openai_var.get())
        args.disable_perplexity = not bool(perplexity_var.get())
        args.enable_final_tema_review_pass = bool(tema_pass_var.get())

        args.disable_post_processing = not bool(post_var.get())
        args.disable_classe_standardization = not bool(classe_var.get())
        args.disable_relator_standardization = not bool(relator_var.get())
        args.disable_ramo_subramo_standardization = not bool(ramo_sub_var.get())
        args.disable_legislacao_standardization = not bool(legislacao_var.get())

        args.no_startup_backup = not bool(startup_backup_var.get())

        verbosity = verbosity_var.get().strip().lower()
        args.verbose = verbosity == "verbose"
        args.quiet = verbosity == "quiet"

        # Evita abertura de GUI secundária de seleção.
        args.no_gui = True
        args.gui_panel = False

        result["ok"] = True
        root.destroy()

    def on_cancel() -> None:
        result["ok"] = False
        root.destroy()

    ttk.Button(button_row, text="Cancelar", command=on_cancel).pack(side="right")
    ttk.Button(button_row, text="Iniciar Pipeline", command=on_start).pack(side="right", padx=(0, 8))

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return bool(result["ok"])


def apply_pipeline_profile(args: argparse.Namespace, logger: logging.Logger) -> None:
    profile = (args.pipeline_profile or "balanced").strip().lower()
    if profile == "balanced":
        return

    if profile == "economico":
        args.openai_tema_reprocess_mode = "invalid_only"
        if args.openai_text_max_chars <= 0 or args.openai_text_max_chars > 1800:
            args.openai_text_max_chars = 1800
        args.openai_max_workers = min(max(1, args.openai_max_workers), 8)
        args.openai_batch_size = min(max(1, args.openai_batch_size), 24)

        args.perplexity_scaling_mode = "fixed"
        args.perplexity_max_workers = min(max(1, args.perplexity_max_workers), 2)
        args.perplexity_target_rpm = min(max(1, args.perplexity_target_rpm), 20)
        args.perplexity_fallback_enabled = False

        if args.ramo_subramo_scope == "all":
            args.ramo_subramo_scope = "problematic"

    elif profile == "velocidade":
        args.openai_max_workers = max(args.openai_max_workers, 12)
        args.openai_batch_size = max(args.openai_batch_size, 36)
        args.openai_delay = min(max(0.0, args.openai_delay), 0.03)

        args.perplexity_scaling_mode = "adaptive"
        args.perplexity_max_workers = max(args.perplexity_max_workers, 3)
        args.perplexity_target_rpm = max(args.perplexity_target_rpm, 35)
        args.perplexity_delay = min(max(0.0, args.perplexity_delay), 0.6)

    logger.info(
        "Pipeline profile '%s' aplicado | openai_workers=%d openai_batch=%d perplexity_workers=%d perplexity_rpm=%d",
        profile,
        int(args.openai_max_workers),
        int(args.openai_batch_size),
        int(args.perplexity_max_workers),
        int(args.perplexity_target_rpm),
    )


def create_startup_backup(path: Path, backup_dir: Path, label: str, logger: logging.Logger) -> None:
    if not path.exists() or not path.is_file():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    tag = timestamp_tag()
    target = backup_dir / f"{path.name}.backup_{label}_{tag}"
    shutil.copy2(path, target)
    logger.info("Backup inicial criado: %s", target)


def run_command(cmd: Sequence[str], cwd: Path, env: Dict[str, str], logger: logging.Logger) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    logger.info("Executando: %s", pretty)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando falhou (exit={proc.returncode}): {pretty}")


def load_rows_from_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV sem cabeçalho: {path}")
        for raw in reader:
            row = core.base_empty_row()
            for col in core.CSV_COLUMNS:
                row[col] = str(raw.get(col, "") or "")
            row["_row_id"] = core.generate_row_id(row)
            rows.append(row)
    core.ensure_output_rows_format(rows)
    return rows


def checkpoint_matches_manifest(checkpoint: Dict[str, Any], manifest: Sequence[Dict[str, str]]) -> bool:
    if not checkpoint:
        return False
    cp_version = int(checkpoint.get("version", 0) or 0)
    cp_manifest = checkpoint.get("manifest", [])
    if cp_version != core.CHECKPOINT_VERSION:
        return False
    if not isinstance(cp_manifest, list):
        return False
    return core.same_manifest(cp_manifest, manifest)


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path, Path]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / (args.output_name or DEFAULT_OUTPUT_NAME)
    checkpoint_path = (
        Path(args.checkpoint_file).resolve()
        if (args.checkpoint_file or "").strip()
        else output_dir / (args.checkpoint_name or DEFAULT_CHECKPOINT_NAME)
    )
    quality_report_path = (
        Path(args.quality_report_file).resolve()
        if (args.quality_report_file or "").strip()
        else output_dir / (args.quality_report_name or DEFAULT_QUALITY_REPORT_NAME)
    )

    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = Path(args.backup_dir).resolve() if (args.backup_dir or "").strip() else output_dir
    backup_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, output_csv, checkpoint_path, quality_report_path, report_dir, backup_dir


def discover_explicit_input_files(
    input_dirs: Sequence[str],
    input_files: Sequence[str],
    logger: logging.Logger,
) -> List[Path]:
    files: List[Path] = []

    for raw in input_files:
        p = Path(raw).expanduser()
        if not p.exists() or not p.is_file() or p.suffix.lower() != ".pdf":
            logger.warning("Arquivo ignorado (não é PDF válido): %s", raw)
            continue
        files.append(p.resolve())

    for raw in input_dirs:
        d = Path(raw).expanduser()
        if not d.exists() or not d.is_dir():
            logger.warning("Pasta ignorada (não encontrada): %s", raw)
            continue
        files.extend(sorted(x.resolve() for x in d.glob("*.pdf")))

    files = sorted(files, key=lambda p: (p.parent.name, p.name))
    if not files:
        return []

    # dedupe por hash SHA-1 para evitar repetição de conteúdos idênticos.
    unique: List[Path] = []
    seen_hash: Dict[str, Path] = {}
    for f in files:
        try:
            h = core.sha1_file(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha hash %s: %s", f, exc)
            continue
        if h in seen_hash:
            logger.info("PDF duplicado ignorado: %s (igual a %s)", f, seen_hash[h])
            continue
        seen_hash[h] = f
        unique.append(f)
    return unique


def resolve_inputs(args: argparse.Namespace, logger: logging.Logger) -> List[Path]:
    input_dirs = list(args.input_dirs or [])
    input_files = list(args.input_files or [])

    if not args.no_gui and not input_dirs and not input_files:
        gui_dirs, gui_files = core.gui_select_inputs(logger)
        input_dirs = gui_dirs or input_dirs
        input_files = gui_files or input_files

    explicit_selection = bool(input_dirs or input_files)
    if explicit_selection:
        files = discover_explicit_input_files(input_dirs=input_dirs, input_files=input_files, logger=logger)
        if not files:
            raise SystemExit(
                "Nenhum PDF válido encontrado nos caminhos informados em --input-dirs/--input-files."
            )
        return files

    files = core.discover_input_files(input_dirs=input_dirs, input_files=input_files, logger=logger)
    if not files:
        raise SystemExit("Nenhum PDF encontrado para processar.")
    return files


def run_final_tema_review_if_enabled(
    *,
    args: argparse.Namespace,
    output_csv: Path,
    script_dir: Path,
    env: Dict[str, str],
    openai_key: str,
    logger: logging.Logger,
) -> None:
    if not args.enable_final_tema_review_pass:
        return
    if not openai_key:
        logger.warning("Passe final de tema ignorado: API key OpenAI ausente.")
        return

    script_path = script_dir / "TRF1_csv_revisar_tema_viaAPI_openai.py"
    if not script_path.exists():
        logger.warning("Passe final de tema ignorado: script não encontrado (%s).", script_path)
        return

    cmd = [
        sys.executable,
        str(script_path),
        "--input-csv",
        str(output_csv),
        "--inplace",
        "--no-gui",
        "--reprocess-mode",
        str(args.final_tema_review_mode),
        "--openai-api-key",
        openai_key,
        "--openai-model",
        str(args.openai_model),
        "--openai-batch-size",
        "40",
        "--openai-max-workers",
        "10",
        "--openai-delay",
        "0.05",
    ]
    run_command(cmd, cwd=script_dir, env=env, logger=logger)


def run_post_processing(
    *,
    args: argparse.Namespace,
    output_csv: Path,
    report_dir: Path,
    script_dir: Path,
    env: Dict[str, str],
    openai_key: str,
    logger: logging.Logger,
) -> None:
    if args.disable_post_processing:
        logger.info("Pós-processamento: desabilitado por flag.")
        return

    tag = timestamp_tag()

    if not args.disable_classe_standardization:
        apply_classe_standardization(
            output_csv=output_csv,
            report_dir=report_dir,
            classe_config_file=Path(str(args.classe_config_file)).resolve(),
            logger=logger,
        )

    if not args.disable_relator_standardization:
        relator_script = script_dir / "padronizar_relator_trf1.py"
        if relator_script.exists():
            cmd_relator = [
                sys.executable,
                str(relator_script),
                "--input",
                str(output_csv),
                "--inplace",
                "--report-dir",
                str(report_dir),
                "--report-prefix",
                f"pipeline_relator_{tag}",
            ]
            run_command(cmd_relator, cwd=script_dir, env=env, logger=logger)
        else:
            logger.warning("Padronização de relator ignorada: script ausente (%s).", relator_script)

    if not args.disable_ramo_subramo_standardization:
        ramo_script = script_dir / "padronizar_ramo_subramo_trf1.py"
        if ramo_script.exists():
            cmd_ramo = [
                sys.executable,
                str(ramo_script),
                "--input",
                str(output_csv),
                "--inplace",
                "--report-dir",
                str(report_dir),
                "--report-prefix",
                f"pipeline_ramo_subramo_{tag}",
                "--scope",
                str(args.ramo_subramo_scope),
                "--cache-file",
                str(args.ramo_subramo_cache_file),
                "--adaptive-throttle",
            ]
            if int(args.ramo_subramo_max_rows) > 0:
                cmd_ramo.extend(["--max-rows", str(int(args.ramo_subramo_max_rows))])

            if args.disable_ramo_subramo_openai or not openai_key:
                cmd_ramo.append("--disable-openai")
            else:
                cmd_ramo.extend(
                    [
                        "--openai-api-key",
                        openai_key,
                        "--openai-model",
                        str(args.ramo_subramo_openai_model),
                        "--openai-max-workers",
                        str(max(1, int(args.ramo_subramo_openai_max_workers))),
                        "--openai-batch-size",
                        str(max(1, int(args.ramo_subramo_openai_batch_size))),
                        "--openai-delay",
                        str(max(0.0, float(args.ramo_subramo_openai_delay))),
                    ]
                )
            run_command(cmd_ramo, cwd=script_dir, env=env, logger=logger)
        else:
            logger.warning("Padronização de ramo/subramo ignorada: script ausente (%s).", ramo_script)

    if not args.disable_legislacao_standardization:
        legislacao_script = script_dir / "padronizar_legislacao_trf1.py"
        if legislacao_script.exists():
            cmd_leg = [
                sys.executable,
                str(legislacao_script),
                "--input",
                str(output_csv),
                "--inplace",
                "--report-dir",
                str(report_dir),
                "--report-prefix",
                f"pipeline_legislacao_{tag}",
            ]
            run_command(cmd_leg, cwd=script_dir, env=env, logger=logger)
        else:
            logger.warning("Padronização de legislação ignorada: script ausente (%s).", legislacao_script)


def main() -> None:
    parser = add_final_args(core.build_arg_parser())
    args = parser.parse_args()

    if should_open_gui_panel(args):
        proceed = open_gui_control_panel(args)
        if not proceed:
            raise SystemExit("Execução cancelada pelo usuário no painel GUI.")

    logger = core.setup_logger(args.verbose, args.quiet)
    apply_pipeline_profile(args, logger)

    # Ajusta ordem final do CSV para o padrão consolidado.
    core.CSV_COLUMNS = list(FINAL_CSV_COLUMNS)

    script_dir = Path(__file__).resolve().parent
    output_dir, output_csv, checkpoint_path, quality_report_path, report_dir, backup_dir = resolve_paths(args)

    if not args.no_startup_backup:
        create_startup_backup(output_csv, backup_dir, "pre_pipeline_csv", logger)
        create_startup_backup(checkpoint_path, backup_dir, "pre_pipeline_checkpoint", logger)
        create_startup_backup(quality_report_path, backup_dir, "pre_pipeline_quality", logger)

    files = resolve_inputs(args, logger)
    logger.info("PDFs selecionados: %d", len(files))

    manifest = core.compute_manifest(files)
    rows: List[Dict[str, str]] = []
    perplexity_state: Dict[str, Any] = {}

    checkpoint = core.read_checkpoint(checkpoint_path)
    if checkpoint_matches_manifest(checkpoint, manifest):
        saved_rows = checkpoint.get("rows", [])
        if isinstance(saved_rows, list) and saved_rows:
            rows = [core.row_from_checkpoint(r) for r in saved_rows if isinstance(r, dict)]
            raw_state = checkpoint.get("perplexity_state", {})
            perplexity_state = raw_state if isinstance(raw_state, dict) else {}
            logger.info("Checkpoint compatível carregado: %d linhas.", len(rows))

    if not rows:
        logger.info("Extração base: iniciando leitura de PDFs.")
        extracted: List[Dict[str, str]] = []
        for idx, pdf in enumerate(files, start=1):
            logger.info("Extraindo %d/%d: %s", idx, len(files), pdf.name)
            extracted.extend(core.extract_rows_from_pdf(pdf, logger))
        rows = core.deduplicate_rows(extracted, logger)
        core.ensure_output_rows_format(rows)
        core.write_csv(output_csv, rows)
        core.write_checkpoint(
            checkpoint_path,
            core.checkpoint_payload(manifest=manifest, rows=rows, perplexity_state=perplexity_state),
        )
        logger.info("Extração base concluída: %d julgados.", len(rows))

    core.ensure_output_rows_format(rows)

    openai_key = core.resolve_openai_key(args.openai_api_key)
    openai_cfg = core.OpenAIConfig(
        enabled=(not args.disable_openai) and bool(openai_key),
        api_key=openai_key,
        model=args.openai_model.strip() or "gpt-5-mini",
        batch_size=max(1, int(args.openai_batch_size)),
        max_workers=max(1, int(args.openai_max_workers)),
        max_workers_cap=max(1, int(args.openai_max_workers_cap)),
        delay=max(0.0, float(args.openai_delay)),
        retries=max(1, int(args.openai_retries)),
        timeout=max(5, int(args.openai_timeout)),
        max_completion_tokens=max(0, int(args.openai_max_completion_tokens)),
        max_completion_tokens_cap=max(0, int(args.openai_max_completion_tokens_cap)),
        reasoning_effort=(args.openai_reasoning_effort or "medium").strip().lower(),
        verbosity=(args.openai_verbosity or "medium").strip().lower(),
        length_fallback_policy=(args.openai_length_fallback_policy or "auto_downgrade").strip().lower(),
        length_error_threshold=max(0.0, float(args.openai_length_error_threshold)),
        text_max_chars=max(0, int(args.openai_text_max_chars)),
        tema_refine_enabled=(not args.disable_openai_tema_refine),
        tema_reprocess_mode=(args.openai_tema_reprocess_mode or "invalid_only").strip().lower(),
        tema_input_profile=(args.openai_tema_input_profile or "lean").strip().lower(),
        tema_profile_fallback=(args.openai_tema_profile_fallback or "auto").strip().lower(),
        tema_contexto_max_chars=max(0, int(args.openai_tema_contexto_max_chars)),
        tema_bullets_max_items=max(0, int(args.openai_tema_bullets_max_items)),
        tema_target_rpm=max(0, int(args.openai_tema_target_rpm)),
        tema_max_completion_tokens=max(0, int(args.openai_tema_max_completion_tokens)),
        tema_max_completion_tokens_cap=max(0, int(args.openai_tema_max_completion_tokens_cap)),
        tema_reasoning_effort=(args.openai_tema_reasoning_effort or "minimal").strip().lower(),
        tema_verbosity=(args.openai_tema_verbosity or "low").strip().lower(),
        tema_length_fallback_policy=(args.openai_tema_length_fallback_policy or "auto_downgrade").strip().lower(),
        tema_length_error_threshold=max(0.0, float(args.openai_tema_length_error_threshold)),
        tema_text_max_chars=max(0, int(args.openai_tema_text_max_chars)),
    )

    if not openai_cfg.enabled and not args.disable_openai:
        logger.warning("OpenAI desabilitado por ausência de API key.")

    if openai_cfg.enabled:
        core.run_openai_enrichment(
            rows=rows,
            config=openai_cfg,
            output_csv=output_csv,
            checkpoint_path=checkpoint_path,
            manifest=manifest,
            perplexity_state=perplexity_state,
            logger=logger,
        )
        if openai_cfg.tema_refine_enabled:
            core.run_openai_tema_refinement(
                rows=rows,
                config=openai_cfg,
                output_csv=output_csv,
                checkpoint_path=checkpoint_path,
                manifest=manifest,
                perplexity_state=perplexity_state,
                logger=logger,
            )
        else:
            logger.info("OpenAI tema (geração principal): etapa desabilitada por flag.")
    elif args.disable_openai:
        logger.info("OpenAI: etapa desabilitada por flag.")

    perplexity_key = core.resolve_perplexity_key(args.perplexity_api_key)
    perplexity_cfg = core.PerplexityConfig(
        enabled=(not args.disable_perplexity) and bool(perplexity_key),
        api_key=perplexity_key,
        model=(args.perplexity_model or "sonar").strip() or "sonar",
        batch_size=max(1, int(args.perplexity_batch_size)),
        max_workers=max(1, int(args.perplexity_max_workers)),
        delay=max(0.0, float(args.perplexity_delay)),
        retries=max(1, int(args.perplexity_retries)),
        timeout=max(5, int(args.perplexity_timeout)),
        max_workers_cap=max(1, int(args.perplexity_max_workers_cap)),
        fallback_enabled=bool(args.perplexity_fallback_enabled),
        fallback_max_fraction=max(0.0, min(1.0, float(args.perplexity_fallback_max_fraction))),
        min_score_mainstream=max(0, int(args.perplexity_min_score_mainstream)),
        min_score_official=max(0, int(args.perplexity_min_score_official)),
        domain_policy=str(args.perplexity_domain_policy),
        scaling_mode=str(args.perplexity_scaling_mode),
        target_rpm=max(0, int(args.perplexity_target_rpm)),
        resume_rate_state=bool(args.perplexity_resume_rate_state),
        max_tokens=max(0, int(args.perplexity_max_tokens)),
        text_max_chars=max(0, int(args.perplexity_text_max_chars)),
        skip_terminal_reasons=(not bool(args.perplexity_allow_terminal_retry)),
    )

    if not perplexity_cfg.enabled and not args.disable_perplexity:
        logger.warning("Perplexity desabilitado por ausência de API key.")

    if perplexity_cfg.enabled:
        perplexity_state = core.run_perplexity_enrichment(
            rows=rows,
            config=perplexity_cfg,
            output_csv=output_csv,
            checkpoint_path=checkpoint_path,
            manifest=manifest,
            perplexity_state=perplexity_state,
            logger=logger,
        )
    elif args.disable_perplexity:
        logger.info("Perplexity: etapa desabilitada por flag.")

    core.ensure_output_rows_format(rows)
    core.write_csv(output_csv, rows)
    core.write_checkpoint(
        checkpoint_path,
        core.checkpoint_payload(manifest=manifest, rows=rows, perplexity_state=perplexity_state),
    )

    env = os.environ.copy()
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key
    if perplexity_key:
        env["PERPLEXITY_API_KEY"] = perplexity_key

    run_final_tema_review_if_enabled(
        args=args,
        output_csv=output_csv,
        script_dir=script_dir,
        env=env,
        openai_key=openai_key,
        logger=logger,
    )

    run_post_processing(
        args=args,
        output_csv=output_csv,
        report_dir=report_dir,
        script_dir=script_dir,
        env=env,
        openai_key=openai_key,
        logger=logger,
    )

    # Recarrega após pós-processamentos para registrar estado final real.
    rows = load_rows_from_csv(output_csv)
    core.write_checkpoint(
        checkpoint_path,
        core.checkpoint_payload(manifest=manifest, rows=rows, perplexity_state=perplexity_state),
    )
    core.write_quality_report(quality_report_path, rows)

    report = core.build_quality_report(rows)
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    openai_summary = core.summarize_openai_progress(rows)
    perplexity_summary = core.summarize_perplexity_progress(rows, perplexity_cfg.skip_terminal_reasons)

    logger.info(
        "Métricas-chave: tema_curto=%s | openai_done=%s | noticias=%s",
        metrics.get("tema_too_short_count", 0),
        metrics.get("openai_done", 0),
        metrics.get("perplexity_done_with_url", 0),
    )
    core.validate_rows(rows, logger)

    logger.info(
        "Concluído. Linhas: %d | OpenAI completas: %d | OpenAI pendentes: %d | Perplexity com URL: %d",
        len(rows),
        openai_summary.get("done", 0),
        openai_summary.get("pending_retryable", 0),
        perplexity_summary.get("done_with_url", 0),
    )
    logger.info("CSV final: %s", output_csv)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Quality report: %s", quality_report_path)
    logger.info("Relatórios de padronização: %s", report_dir)


if __name__ == "__main__":
    main()
