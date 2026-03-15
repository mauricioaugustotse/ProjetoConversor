#!/usr/bin/env python3
"""Enriquece apenas colunas de API vazias em CSVs do pipeline SJUR.

Fluxo:
1. Le CSV(s) ja processado(s) no formato Notion-friendly.
2. Identifica apenas linhas com colunas de API vazias.
3. Chama OpenAI (tema/punchline e opcionalmente assuntos) e Perplexity (noticias).
4. Mantem valores existentes e preenche somente vazios.
5. Grava CSV final (novo arquivo ou in-place).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import SJUR_csv_to_csv_NOTIONfriendly as sjur
from Artefatos.scripts.openai_log_utils import configure_standard_logging, install_print_logger_bridge
from Artefatos.scripts.openai_progress_utils import make_backup, write_csv_atomic


@dataclass
class FileEnrichmentResult:
    input_path: Path
    output_path: Path
    rows: int
    tema_added: int
    punchline_added: int
    assuntos_added: int
    noticias_added: dict[str, int]


def _is_blank(value: object) -> bool:
    return not bool(str(value or "").strip())


def _count_filled(rows: Sequence[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if not _is_blank(row.get(column, "")))


def _ensure_columns(rows: list[dict[str, str]], fieldnames: list[str], columns: Sequence[str]) -> None:
    for col in columns:
        if col not in fieldnames:
            fieldnames.append(col)
    for row in rows:
        for col in columns:
            if col not in row:
                row[col] = ""


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    encoding, delimiter = sjur.detect_csv_format(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"{path.name}: cabecalho CSV ausente ou invalido.")
        rows: list[dict[str, str]] = []
        for raw in reader:
            row: dict[str, str] = {}
            for key in fieldnames:
                value = raw.get(key, "") if raw is not None else ""
                row[key] = str(value or "")
            rows.append(row)
    return fieldnames, rows


def _resolve_output_path(input_path: Path, *, out_dir: str, suffix: str, in_place: bool) -> Path:
    if in_place:
        return input_path
    chosen_suffix = suffix if suffix else "_APIenriched"
    base_dir = Path(out_dir).expanduser().resolve() if out_dir.strip() else input_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{input_path.stem}{chosen_suffix}{input_path.suffix}"


def _build_tema_config(args: argparse.Namespace, api_key: str) -> sjur.TemaPunchlineConfig:
    return sjur.TemaPunchlineConfig(
        enabled=True,
        api_key=api_key,
        model=args.openai_model.strip() or "gpt-5.1",
        timeout_seconds=args.openai_timeout,
        max_workers=args.openai_max_workers,
        batch_size=args.openai_batch_size,
        delay_between_batches=args.openai_delay,
        retries=args.openai_retries,
        target_rpm=args.openai_target_rpm,
    )


def _build_assuntos_config(args: argparse.Namespace, api_key: str) -> sjur.AssuntosEnrichmentConfig:
    return sjur.AssuntosEnrichmentConfig(
        enabled=True,
        api_key=api_key,
        model=args.openai_model.strip() or "gpt-5.1",
        timeout_seconds=args.openai_timeout,
        max_workers=args.openai_max_workers,
        batch_size=args.openai_batch_size,
        delay_between_batches=args.openai_delay,
        retries=args.openai_retries,
        target_rpm=args.openai_target_rpm,
        max_items=args.assuntos_max_itens,
        taxonomy_mode=args.assuntos_taxonomy_mode.strip().lower(),
    )


def _build_web_config(args: argparse.Namespace, api_key: str) -> sjur.WebLookupConfig:
    return sjur.WebLookupConfig(
        enabled=True,
        api_key=api_key,
        model=args.perplexity_model.strip() or "sonar",
        timeout_seconds=args.perplexity_timeout,
        max_workers=args.perplexity_max_workers,
        batch_size=args.perplexity_batch_size,
        delay_between_batches=args.perplexity_delay,
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    use_tema_punchline = not args.skip_openai_tema_punchline
    use_assuntos = args.enriquecer_assuntos_openai
    use_urls = not args.skip_perplexity_urls

    if not (use_tema_punchline or use_assuntos or use_urls):
        parser.error(
            "Nada para executar. Ative ao menos uma frente de API "
            "(OpenAI tema/punchline, OpenAI assuntos ou Perplexity URLs)."
        )

    if use_tema_punchline or use_assuntos:
        if args.openai_max_workers <= 0:
            parser.error("--openai-max-workers deve ser maior que zero.")
        if args.openai_timeout <= 0:
            parser.error("--openai-timeout deve ser maior que zero.")
        if args.openai_batch_size <= 0:
            parser.error("--openai-batch-size deve ser maior que zero.")
        if args.openai_retries <= 0:
            parser.error("--openai-retries deve ser maior que zero.")
        if args.openai_target_rpm < 0:
            parser.error("--openai-target-rpm nao pode ser negativo.")
        if args.openai_delay < 0:
            parser.error("--openai-delay nao pode ser negativo.")

    if use_assuntos:
        if args.assuntos_max_itens <= 0:
            parser.error("--assuntos-max-itens deve ser maior que zero.")
        mode = args.assuntos_taxonomy_mode.strip().lower()
        if mode not in sjur.ASSUNTOS_TAXONOMY_CHOICES:
            parser.error(
                "--assuntos-taxonomy-mode invalido. "
                f"Use: {', '.join(sjur.ASSUNTOS_TAXONOMY_CHOICES)}."
            )

    if use_urls:
        if args.perplexity_max_workers <= 0:
            parser.error("--perplexity-max-workers deve ser maior que zero.")
        if args.perplexity_timeout <= 0:
            parser.error("--perplexity-timeout deve ser maior que zero.")
        if args.perplexity_batch_size <= 0:
            parser.error("--perplexity-batch-size deve ser maior que zero.")
        if args.perplexity_delay < 0:
            parser.error("--perplexity-delay nao pode ser negativo.")


def _validate_namespace(args: argparse.Namespace) -> None:
    use_tema_punchline = not args.skip_openai_tema_punchline
    use_assuntos = bool(args.enriquecer_assuntos_openai)
    use_urls = not args.skip_perplexity_urls

    if not (use_tema_punchline or use_assuntos or use_urls):
        raise ValueError(
            "Nada para executar. Ative ao menos uma frente de API "
            "(OpenAI tema/punchline, OpenAI assuntos ou Perplexity URLs)."
        )

    if use_tema_punchline or use_assuntos:
        if int(args.openai_max_workers) <= 0:
            raise ValueError("OpenAI: --openai-max-workers deve ser maior que zero.")
        if int(args.openai_timeout) <= 0:
            raise ValueError("OpenAI: --openai-timeout deve ser maior que zero.")
        if int(args.openai_batch_size) <= 0:
            raise ValueError("OpenAI: --openai-batch-size deve ser maior que zero.")
        if int(args.openai_retries) <= 0:
            raise ValueError("OpenAI: --openai-retries deve ser maior que zero.")
        if int(args.openai_target_rpm) < 0:
            raise ValueError("OpenAI: --openai-target-rpm nao pode ser negativo.")
        if float(args.openai_delay) < 0:
            raise ValueError("OpenAI: --openai-delay nao pode ser negativo.")

    if use_assuntos:
        if int(args.assuntos_max_itens) <= 0:
            raise ValueError("Assuntos: --assuntos-max-itens deve ser maior que zero.")
        mode = str(args.assuntos_taxonomy_mode).strip().lower()
        if mode not in sjur.ASSUNTOS_TAXONOMY_CHOICES:
            raise ValueError(
                "Assuntos: --assuntos-taxonomy-mode invalido. "
                f"Use: {', '.join(sjur.ASSUNTOS_TAXONOMY_CHOICES)}."
            )

    if use_urls:
        if int(args.perplexity_max_workers) <= 0:
            raise ValueError("Perplexity: --perplexity-max-workers deve ser maior que zero.")
        if int(args.perplexity_timeout) <= 0:
            raise ValueError("Perplexity: --perplexity-timeout deve ser maior que zero.")
        if int(args.perplexity_batch_size) <= 0:
            raise ValueError("Perplexity: --perplexity-batch-size deve ser maior que zero.")
        if float(args.perplexity_delay) < 0:
            raise ValueError("Perplexity: --perplexity-delay nao pode ser negativo.")


def enrich_one_file(
    input_path: Path,
    *,
    output_path: Path,
    use_tema_punchline: bool,
    use_assuntos: bool,
    use_urls: bool,
    tema_config: sjur.TemaPunchlineConfig | None,
    assuntos_config: sjur.AssuntosEnrichmentConfig | None,
    web_config: sjur.WebLookupConfig | None,
    backup_before_write: bool,
    logger: Callable[[str], None],
) -> FileEnrichmentResult:
    fieldnames, rows = _load_csv(input_path)

    needed_columns: list[str] = []
    if use_tema_punchline:
        needed_columns.extend(sjur.THEME_COLUMNS)
    if use_assuntos:
        needed_columns.append("assuntos")
    if use_urls:
        needed_columns.extend(sjur.URL_COLUMNS)
    _ensure_columns(rows, fieldnames, needed_columns)

    before_tema = _count_filled(rows, "tema")
    before_punchline = _count_filled(rows, "punchline")
    before_assuntos = _count_filled(rows, "assuntos")
    before_noticias = {col: _count_filled(rows, col) for col in sjur.URL_COLUMNS}

    lookup_payloads = [sjur.build_lookup_payload(row) for row in rows]
    log = lambda message: logger(f"[{input_path.name}] {message}")

    if use_tema_punchline and tema_config is not None:
        idx_tema = [
            idx
            for idx, row in enumerate(rows)
            if _is_blank(row.get("tema", "")) or _is_blank(row.get("punchline", ""))
        ]
        log(
            f"[ChatGPT] Linhas com tema/punchline pendentes: {len(idx_tema)} de {len(rows)}."
        )
        if idx_tema:
            asyncio.run(
                sjur.enriquecer_rows_com_tema_punchline_async(
                    [rows[idx] for idx in idx_tema],
                    log,
                    tema_config,
                    lookup_payloads=[lookup_payloads[idx] for idx in idx_tema],
                )
            )

    if use_assuntos and assuntos_config is not None:
        idx_assuntos = [idx for idx, row in enumerate(rows) if _is_blank(row.get("assuntos", ""))]
        log(f"[Assuntos/OpenAI] Linhas com assuntos pendentes: {len(idx_assuntos)} de {len(rows)}.")
        if idx_assuntos:
            asyncio.run(
                sjur.enriquecer_rows_com_assuntos_openai_async(
                    [rows[idx] for idx in idx_assuntos],
                    log,
                    assuntos_config,
                    lookup_payloads=[lookup_payloads[idx] for idx in idx_assuntos],
                )
            )

    if use_urls and web_config is not None:
        idx_urls = [
            idx
            for idx, row in enumerate(rows)
            if any(_is_blank(row.get(col, "")) for col in sjur.URL_COLUMNS)
        ]
        log(f"[Perplexity] Linhas com noticias pendentes: {len(idx_urls)} de {len(rows)}.")
        if idx_urls:
            asyncio.run(
                sjur.enriquecer_rows_com_urls_async(
                    [rows[idx] for idx in idx_urls],
                    log,
                    web_config,
                    lookup_payloads=[lookup_payloads[idx] for idx in idx_urls],
                )
            )

    after_tema = _count_filled(rows, "tema")
    after_punchline = _count_filled(rows, "punchline")
    after_assuntos = _count_filled(rows, "assuntos")
    after_noticias = {col: _count_filled(rows, col) for col in sjur.URL_COLUMNS}

    if backup_before_write and output_path.exists():
        backup_path = make_backup(output_path, label="apienriched_backup")
        if backup_path is not None:
            log(f"[Backup] {backup_path.name}")

    write_csv_atomic(output_path, fieldnames, rows)
    log(f"[Gravado] {output_path}")

    return FileEnrichmentResult(
        input_path=input_path,
        output_path=output_path,
        rows=len(rows),
        tema_added=max(0, after_tema - before_tema),
        punchline_added=max(0, after_punchline - before_punchline),
        assuntos_added=max(0, after_assuntos - before_assuntos),
        noticias_added={col: max(0, after_noticias[col] - before_noticias[col]) for col in sjur.URL_COLUMNS},
    )


def run_enrichment(args: argparse.Namespace, logger: Callable[[str], None]) -> list[FileEnrichmentResult]:
    _validate_namespace(args)
    use_tema_punchline = not args.skip_openai_tema_punchline
    use_assuntos = bool(args.enriquecer_assuntos_openai)
    use_urls = not args.skip_perplexity_urls

    openai_key = ""
    if use_tema_punchline or use_assuntos:
        openai_key = sjur.resolve_openai_api_key(args.openai_api_key.strip(), args.openai_key_file)
        if not openai_key:
            raise ValueError(
                "OpenAI habilitado, mas sem chave. Use --openai-api-key, OPENAI_API_KEY "
                f"ou arquivo {args.openai_key_file}."
            )

    perplexity_key = ""
    if use_urls:
        perplexity_key = sjur.resolve_perplexity_api_key(args.perplexity_api_key.strip(), args.perplexity_key_file)
        if not perplexity_key:
            raise ValueError(
                "Perplexity habilitado, mas sem chave. Use --perplexity-api-key, PERPLEXITY_API_KEY "
                f"ou arquivo {args.perplexity_key_file}."
            )

    tema_config = _build_tema_config(args, openai_key) if use_tema_punchline else None
    assuntos_config = _build_assuntos_config(args, openai_key) if use_assuntos else None
    web_config = _build_web_config(args, perplexity_key) if use_urls else None

    input_paths = sjur.normalize_input_paths(args.files)

    results: list[FileEnrichmentResult] = []
    for input_path in input_paths:
        output_path = _resolve_output_path(
            input_path,
            out_dir=args.out_dir,
            suffix=args.suffix,
            in_place=bool(args.in_place),
        )
        result = enrich_one_file(
            input_path=input_path,
            output_path=output_path,
            use_tema_punchline=use_tema_punchline,
            use_assuntos=use_assuntos,
            use_urls=use_urls,
            tema_config=tema_config,
            assuntos_config=assuntos_config,
            web_config=web_config,
            backup_before_write=not args.no_backup,
            logger=logger,
        )
        results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preenche somente colunas de API vazias (OpenAI/Perplexity) em CSVs SJUR."
    )
    parser.add_argument("files", nargs="*", help="Arquivos CSV de entrada (ja no layout final).")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Pasta de saida. Se vazio, usa a pasta do CSV de entrada.",
    )
    parser.add_argument(
        "--suffix",
        default="_APIenriched",
        help="Sufixo para o nome de saida quando nao usar --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Sobrescreve o proprio CSV de entrada.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Nao cria backup quando o arquivo de saida ja existir.",
    )

    parser.add_argument(
        "--skip-openai-tema-punchline",
        action="store_true",
        help="Desativa enriquecimento de tema/punchline.",
    )
    parser.add_argument(
        "--enriquecer-assuntos-openai",
        action="store_true",
        help="Ativa enriquecimento de assuntos vazios via OpenAI.",
    )
    parser.add_argument(
        "--skip-perplexity-urls",
        action="store_true",
        help="Desativa enriquecimento de noticias (TSE/TRE/Gerais).",
    )

    parser.add_argument("--openai-api-key", default="", help="API key da OpenAI.")
    parser.add_argument(
        "--openai-key-file",
        default=sjur.DEFAULT_OPENAI_KEY_FILE,
        help="Arquivo local com a chave OpenAI.",
    )
    parser.add_argument("--openai-model", default="gpt-5.1", help="Modelo OpenAI.")
    parser.add_argument(
        "--openai-max-workers",
        type=int,
        default=sjur.OPENAI_DEFAULT_MAX_WORKERS,
        help="Numero maximo de workers OpenAI.",
    )
    parser.add_argument(
        "--openai-timeout",
        type=int,
        default=sjur.OPENAI_DEFAULT_TIMEOUT,
        help="Timeout OpenAI em segundos.",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=sjur.OPENAI_DEFAULT_BATCH_SIZE,
        help="Tamanho do lote OpenAI.",
    )
    parser.add_argument(
        "--openai-delay",
        type=float,
        default=sjur.OPENAI_DEFAULT_DELAY,
        help="Delay entre lotes OpenAI (segundos).",
    )
    parser.add_argument(
        "--openai-retries",
        type=int,
        default=sjur.OPENAI_DEFAULT_RETRIES,
        help="Maximo de retries OpenAI.",
    )
    parser.add_argument(
        "--openai-target-rpm",
        type=int,
        default=sjur.OPENAI_DEFAULT_TARGET_RPM,
        help="Limite alvo de requests/min OpenAI (0 desativa pacing).",
    )

    parser.add_argument(
        "--assuntos-max-itens",
        type=int,
        default=sjur.DEFAULT_ASSUNTOS_MAX_ITEMS,
        help="Numero maximo de assuntos por linha.",
    )
    parser.add_argument(
        "--assuntos-taxonomy-mode",
        default="mixed",
        choices=list(sjur.ASSUNTOS_TAXONOMY_CHOICES),
        help="Modo de taxonomia para assuntos.",
    )

    parser.add_argument("--perplexity-api-key", default="", help="API key da Perplexity.")
    parser.add_argument(
        "--perplexity-key-file",
        default=sjur.DEFAULT_PERPLEXITY_KEY_FILE,
        help="Arquivo local com a chave Perplexity.",
    )
    parser.add_argument("--perplexity-model", default="sonar", help="Modelo Perplexity.")
    parser.add_argument("--perplexity-max-workers", type=int, default=4, help="Workers Perplexity.")
    parser.add_argument("--perplexity-timeout", type=int, default=15, help="Timeout Perplexity em segundos.")
    parser.add_argument("--perplexity-batch-size", type=int, default=20, help="Tamanho do lote Perplexity.")
    parser.add_argument("--perplexity-delay", type=float, default=0.3, help="Delay entre lotes Perplexity.")

    parser.add_argument("--verbose", action="store_true", help="Exibe logs detalhados.")
    parser.add_argument("--quiet", action="store_true", help="Exibe apenas avisos/erros.")
    parser.add_argument("--debug", action="store_true", help="Ativa logs tecnicos.")
    parser.add_argument("--log-file", default="", help="Arquivo opcional para salvar logs.")
    parser.add_argument("--no-gui", action="store_true", help="Forca modo CLI mesmo sem arquivos posicionais.")
    return parser


def _summarize_results(results: Sequence[FileEnrichmentResult]) -> list[str]:
    total_rows = sum(item.rows for item in results)
    total_tema = sum(item.tema_added for item in results)
    total_punchline = sum(item.punchline_added for item in results)
    total_assuntos = sum(item.assuntos_added for item in results)
    total_urls = {col: sum(item.noticias_added.get(col, 0) for item in results) for col in sjur.URL_COLUMNS}
    lines = [
        "Resumo API enrichment:",
        f"- Arquivos processados: {len(results)}",
        f"- Linhas processadas: {total_rows}",
        f"- tema adicionados: {total_tema}",
        f"- punchline adicionados: {total_punchline}",
        f"- assuntos adicionados: {total_assuntos}",
    ]
    for col in sjur.URL_COLUMNS:
        lines.append(f"- {col} adicionados: {total_urls[col]}")
    if results:
        lines.append(f"- Ultimo arquivo gerado: {results[-1].output_path}")
    return lines


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Tkinter nao esta disponivel neste Python. "
            "Instale/suporte tkinter ou use modo CLI com --no-gui."
        ) from exc

    class App:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("SJUR API Enriched (Lote)")
            self.root.geometry("920x760")

            self.file_vars: dict[str, tk.BooleanVar] = {}
            self.output_dir_var = tk.StringVar(value="")
            self.suffix_var = tk.StringVar(value="_APIenriched")
            self.in_place_var = tk.BooleanVar(value=False)
            self.no_backup_var = tk.BooleanVar(value=False)
            self.skip_openai_tema_punchline_var = tk.BooleanVar(value=False)
            self.enriquecer_assuntos_var = tk.BooleanVar(value=False)
            self.skip_perplexity_urls_var = tk.BooleanVar(value=False)

            self.openai_key_file_var = tk.StringVar(
                value=sjur._resolve_existing_key_file_path(  # pylint: disable=protected-access
                    sjur.DEFAULT_OPENAI_KEY_FILE,
                    sjur.OPENAI_KEY_FALLBACK_FILES,
                )
            )
            self.openai_api_key_var = tk.StringVar(
                value=sjur.resolve_openai_api_key("", self.openai_key_file_var.get())
            )
            self.openai_model_var = tk.StringVar(value="gpt-5.1")
            self.openai_workers_var = tk.StringVar(value=str(sjur.OPENAI_DEFAULT_MAX_WORKERS))
            self.openai_timeout_var = tk.StringVar(value=str(sjur.OPENAI_DEFAULT_TIMEOUT))
            self.openai_batch_size_var = tk.StringVar(value=str(sjur.OPENAI_DEFAULT_BATCH_SIZE))
            self.openai_delay_var = tk.StringVar(value=str(sjur.OPENAI_DEFAULT_DELAY))
            self.openai_retries_var = tk.StringVar(value=str(sjur.OPENAI_DEFAULT_RETRIES))
            self.openai_target_rpm_var = tk.StringVar(value=str(sjur.OPENAI_DEFAULT_TARGET_RPM))
            self.assuntos_max_itens_var = tk.StringVar(value=str(sjur.DEFAULT_ASSUNTOS_MAX_ITEMS))
            self.assuntos_taxonomy_mode_var = tk.StringVar(value="mixed")

            self.perplexity_key_file_var = tk.StringVar(
                value=sjur._resolve_existing_key_file_path(  # pylint: disable=protected-access
                    sjur.DEFAULT_PERPLEXITY_KEY_FILE,
                    sjur.PERPLEXITY_KEY_FALLBACK_FILES,
                )
            )
            self.perplexity_api_key_var = tk.StringVar(
                value=sjur.resolve_perplexity_api_key("", self.perplexity_key_file_var.get())
            )
            self.perplexity_model_var = tk.StringVar(value="sonar")
            self.perplexity_workers_var = tk.StringVar(value="4")
            self.perplexity_timeout_var = tk.StringVar(value="15")
            self.perplexity_batch_size_var = tk.StringVar(value="20")
            self.perplexity_delay_var = tk.StringVar(value="0.3")
            self.verbose_terminal_var = tk.BooleanVar(value=True)
            self._build_ui()

        def _build_ui(self) -> None:
            main = ttk.Frame(self.root, padding=12)
            main.pack(fill="both", expand=True)

            top_controls = ttk.Frame(main)
            top_controls.pack(fill="x")
            ttk.Button(top_controls, text="Processar selecionados", command=self.process_selected).pack(
                side="right", padx=(8, 0)
            )
            ttk.Button(top_controls, text="Selecionar CSVs", command=self.add_files).pack(side="left", padx=(0, 8))
            ttk.Button(top_controls, text="Adicionar pasta", command=self.add_folder).pack(side="left", padx=(0, 8))
            ttk.Button(top_controls, text="Marcar todos", command=self.check_all).pack(side="left", padx=(0, 8))
            ttk.Button(top_controls, text="Desmarcar todos", command=self.uncheck_all).pack(side="left", padx=(0, 8))
            ttk.Button(top_controls, text="Limpar lista", command=self.clear_files).pack(side="left")

            list_box = ttk.LabelFrame(main, text="Arquivos selecionados (checkbox)", padding=8)
            list_box.pack(fill="both", expand=False, pady=(10, 10))
            self.canvas = tk.Canvas(list_box, height=220)
            scrollbar = ttk.Scrollbar(list_box, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=scrollbar.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            self.files_frame = ttk.Frame(self.canvas)
            self.files_frame_window = self.canvas.create_window((0, 0), window=self.files_frame, anchor="nw")
            self.files_frame.bind(
                "<Configure>",
                lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
            )
            self.canvas.bind("<Configure>", self._on_files_canvas_configure)

            options = ttk.LabelFrame(main, text="Opcoes", padding=10)
            options.pack(fill="x", pady=(0, 10))

            ttk.Label(options, text="Pasta de saida (vazio = pasta do CSV):").grid(row=0, column=0, sticky="w")
            ttk.Entry(options, textvariable=self.output_dir_var, width=70).grid(row=0, column=1, sticky="ew", padx=8)
            ttk.Button(options, text="Selecionar...", command=self.choose_output_dir).grid(row=0, column=2, sticky="e")

            ttk.Label(options, text="Sufixo de saida:").grid(row=1, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.suffix_var, width=25).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))
            ttk.Checkbutton(options, text="Sobrescrever arquivo de entrada (--in-place)", variable=self.in_place_var).grid(
                row=2, column=0, columnspan=3, sticky="w", pady=(8, 0)
            )
            ttk.Checkbutton(options, text="Sem backup (--no-backup)", variable=self.no_backup_var).grid(
                row=3, column=0, columnspan=3, sticky="w", pady=(2, 0)
            )

            ttk.Checkbutton(
                options,
                text="PREENCHER tema/punchline vazios (OpenAI)",
                variable=self.skip_openai_tema_punchline_var,
                onvalue=False,
                offvalue=True,
            ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))
            ttk.Checkbutton(
                options,
                text="PREENCHER assuntos vazios (OpenAI)",
                variable=self.enriquecer_assuntos_var,
            ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 0))
            ttk.Checkbutton(
                options,
                text="PREENCHER noticias vazias (Perplexity)",
                variable=self.skip_perplexity_urls_var,
                onvalue=False,
                offvalue=True,
            ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(2, 0))

            assuntos_frame = ttk.Frame(options)
            assuntos_frame.grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))
            ttk.Label(assuntos_frame, text="Max assuntos/linha").grid(row=0, column=0, sticky="w")
            ttk.Entry(assuntos_frame, textvariable=self.assuntos_max_itens_var, width=6).grid(row=0, column=1, padx=(4, 14))
            ttk.Label(assuntos_frame, text="Taxonomia").grid(row=0, column=2, sticky="w")
            ttk.Combobox(
                assuntos_frame,
                textvariable=self.assuntos_taxonomy_mode_var,
                values=list(sjur.ASSUNTOS_TAXONOMY_CHOICES),
                state="readonly",
                width=12,
            ).grid(row=0, column=3, padx=(4, 0))

            ttk.Label(options, text="OpenAI API key:").grid(row=8, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.openai_api_key_var, width=70, show="*").grid(
                row=8, column=1, sticky="ew", padx=8, pady=(8, 0)
            )
            ttk.Label(options, text="Arquivo da chave OpenAI:").grid(row=9, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.openai_key_file_var, width=70).grid(
                row=9, column=1, sticky="ew", padx=8, pady=(8, 0)
            )

            openai_frame = ttk.Frame(options)
            openai_frame.grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 0))
            ttk.Label(openai_frame, text="Modelo OpenAI").grid(row=0, column=0, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_model_var, width=12).grid(row=0, column=1, padx=(4, 14))
            ttk.Label(openai_frame, text="Workers").grid(row=0, column=2, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_workers_var, width=5).grid(row=0, column=3, padx=(4, 14))
            ttk.Label(openai_frame, text="Timeout(s)").grid(row=0, column=4, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_timeout_var, width=6).grid(row=0, column=5, padx=(4, 14))
            ttk.Label(openai_frame, text="Batch").grid(row=0, column=6, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_batch_size_var, width=6).grid(row=0, column=7, padx=(4, 14))
            ttk.Label(openai_frame, text="Delay(s)").grid(row=0, column=8, sticky="w")
            ttk.Entry(openai_frame, textvariable=self.openai_delay_var, width=6).grid(row=0, column=9, padx=(4, 0))
            ttk.Label(openai_frame, text="Retries").grid(row=1, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(openai_frame, textvariable=self.openai_retries_var, width=6).grid(row=1, column=1, padx=(4, 14), pady=(6, 0))
            ttk.Label(openai_frame, text="Target RPM").grid(row=1, column=2, sticky="w", pady=(6, 0))
            ttk.Entry(openai_frame, textvariable=self.openai_target_rpm_var, width=8).grid(row=1, column=3, padx=(4, 14), pady=(6, 0))

            ttk.Label(options, text="Perplexity API key:").grid(row=11, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.perplexity_api_key_var, width=70, show="*").grid(
                row=11, column=1, sticky="ew", padx=8, pady=(8, 0)
            )
            ttk.Label(options, text="Arquivo da chave Perplexity:").grid(row=12, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(options, textvariable=self.perplexity_key_file_var, width=70).grid(
                row=12, column=1, sticky="ew", padx=8, pady=(8, 0)
            )
            perf_frame = ttk.Frame(options)
            perf_frame.grid(row=13, column=0, columnspan=3, sticky="w", pady=(8, 0))
            ttk.Label(perf_frame, text="Modelo Perplexity").grid(row=0, column=0, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_model_var, width=10).grid(row=0, column=1, padx=(4, 14))
            ttk.Label(perf_frame, text="Workers").grid(row=0, column=2, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_workers_var, width=5).grid(row=0, column=3, padx=(4, 14))
            ttk.Label(perf_frame, text="Timeout(s)").grid(row=0, column=4, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_timeout_var, width=6).grid(row=0, column=5, padx=(4, 14))
            ttk.Label(perf_frame, text="Batch").grid(row=0, column=6, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_batch_size_var, width=6).grid(row=0, column=7, padx=(4, 14))
            ttk.Label(perf_frame, text="Delay(s)").grid(row=0, column=8, sticky="w")
            ttk.Entry(perf_frame, textvariable=self.perplexity_delay_var, width=6).grid(row=0, column=9, padx=(4, 0))

            ttk.Checkbutton(
                options,
                text="Verbose no terminal",
                variable=self.verbose_terminal_var,
            ).grid(row=14, column=0, columnspan=3, sticky="w", pady=(8, 0))

            options.columnconfigure(1, weight=1)

            ttk.Button(main, text="Processar selecionados", command=self.process_selected).pack(anchor="w")

            log_box = ttk.LabelFrame(main, text="Log", padding=8)
            log_box.pack(fill="both", expand=True, pady=(10, 0))
            self.log_widget = tk.Text(log_box, height=12, wrap="word")
            self.log_widget.pack(fill="both", expand=True)
            self.log_widget.configure(state="disabled")
            self.refresh_file_list()

        def _on_files_canvas_configure(self, event: tk.Event) -> None:
            self.canvas.itemconfigure(self.files_frame_window, width=event.width)

        def log(self, message: str) -> None:
            self.log_widget.configure(state="normal")
            self.log_widget.insert("end", f"{message}\n")
            self.log_widget.see("end")
            self.log_widget.configure(state="disabled")
            if self.verbose_terminal_var.get():
                print(message, flush=True)
            self.root.update_idletasks()

        def add_files(self) -> None:
            raw_selection = filedialog.askopenfilenames(
                title="Selecione os arquivos CSV",
                filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
            )
            if not raw_selection:
                return
            try:
                file_paths = list(self.root.tk.splitlist(raw_selection))
            except Exception:
                file_paths = list(raw_selection) if isinstance(raw_selection, (list, tuple)) else [str(raw_selection)]
            for raw_path in file_paths:
                file_path = str(Path(raw_path).expanduser().resolve())
                if file_path not in self.file_vars:
                    self.file_vars[file_path] = tk.BooleanVar(value=True)
            self.refresh_file_list()

        def add_folder(self) -> None:
            folder = filedialog.askdirectory(title="Selecione a pasta com CSVs")
            if not folder:
                return
            for file_path in sorted(Path(folder).expanduser().rglob("*.csv")):
                file_str = str(file_path.resolve())
                if file_str not in self.file_vars:
                    self.file_vars[file_str] = tk.BooleanVar(value=True)
            self.refresh_file_list()

        def clear_files(self) -> None:
            self.file_vars.clear()
            self.refresh_file_list()

        def check_all(self) -> None:
            for variable in self.file_vars.values():
                variable.set(True)

        def uncheck_all(self) -> None:
            for variable in self.file_vars.values():
                variable.set(False)

        def choose_output_dir(self) -> None:
            selected_dir = filedialog.askdirectory(title="Selecione a pasta de saida")
            if selected_dir:
                self.output_dir_var.set(selected_dir)

        def refresh_file_list(self) -> None:
            for child in self.files_frame.winfo_children():
                child.destroy()
            if not self.file_vars:
                ttk.Label(self.files_frame, text="Nenhum CSV selecionado.").grid(
                    row=0,
                    column=0,
                    sticky="w",
                    padx=4,
                    pady=2,
                )
            for row_index, file_path in enumerate(sorted(self.file_vars)):
                label = Path(file_path).name
                ttk.Checkbutton(self.files_frame, text=label, variable=self.file_vars[file_path]).grid(
                    row=row_index,
                    column=0,
                    sticky="w",
                    padx=4,
                    pady=2,
                )
            self.files_frame.update_idletasks()
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)

        def process_selected(self) -> None:
            selected_files = [path for path, var in self.file_vars.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("Aviso", "Selecione ao menos um arquivo CSV.")
                return
            try:
                args = argparse.Namespace(
                    files=selected_files,
                    out_dir=self.output_dir_var.get().strip(),
                    suffix=self.suffix_var.get(),
                    in_place=self.in_place_var.get(),
                    no_backup=self.no_backup_var.get(),
                    skip_openai_tema_punchline=self.skip_openai_tema_punchline_var.get(),
                    enriquecer_assuntos_openai=self.enriquecer_assuntos_var.get(),
                    skip_perplexity_urls=self.skip_perplexity_urls_var.get(),
                    openai_api_key=self.openai_api_key_var.get().strip(),
                    openai_key_file=self.openai_key_file_var.get().strip(),
                    openai_model=self.openai_model_var.get().strip() or "gpt-5.1",
                    openai_max_workers=int(self.openai_workers_var.get().strip()),
                    openai_timeout=int(self.openai_timeout_var.get().strip()),
                    openai_batch_size=int(self.openai_batch_size_var.get().strip()),
                    openai_delay=float(self.openai_delay_var.get().strip()),
                    openai_retries=int(self.openai_retries_var.get().strip()),
                    openai_target_rpm=int(self.openai_target_rpm_var.get().strip()),
                    assuntos_max_itens=int(self.assuntos_max_itens_var.get().strip()),
                    assuntos_taxonomy_mode=self.assuntos_taxonomy_mode_var.get().strip().lower() or "mixed",
                    perplexity_api_key=self.perplexity_api_key_var.get().strip(),
                    perplexity_key_file=self.perplexity_key_file_var.get().strip(),
                    perplexity_model=self.perplexity_model_var.get().strip() or "sonar",
                    perplexity_max_workers=int(self.perplexity_workers_var.get().strip()),
                    perplexity_timeout=int(self.perplexity_timeout_var.get().strip()),
                    perplexity_batch_size=int(self.perplexity_batch_size_var.get().strip()),
                    perplexity_delay=float(self.perplexity_delay_var.get().strip()),
                )
            except ValueError:
                messagebox.showerror(
                    "Erro",
                    "Parametros invalidos (workers/timeout/batch/delay/retries/rpm/assuntos).",
                )
                return
            try:
                results = run_enrichment(args, logger=self.log)
            except Exception as exc:  # pylint: disable=broad-except
                messagebox.showerror("Erro no processamento", str(exc))
                return
            messagebox.showinfo("Concluido", "\n".join(_summarize_results(results)))

    root = tk.Tk()
    App(root)
    root.mainloop()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_cli = bool(args.files) or bool(args.no_gui)
    if not run_cli:
        try:
            launch_gui()
            return 0
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Erro ao iniciar GUI: {exc}", file=sys.stderr)
            return 1
    if not args.files:
        parser.error("No modo --no-gui, informe ao menos um arquivo CSV.")
    _validate_args(parser, args)

    logger = configure_standard_logging(
        "SJUR_csv_to_csv_APIenriched",
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
    )
    install_print_logger_bridge(globals(), logger)
    log = logger.info

    try:
        results = run_enrichment(args, logger=log)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    print()
    for line in _summarize_results(results):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
