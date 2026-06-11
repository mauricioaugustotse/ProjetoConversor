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

# Script movido da raiz do projeto para conversores/. O bloco abaixo mantem
# imports (Artefatos.*, gui_intuitiva e scripts irmaos) e caminhos relativos
# ancorados na raiz do projeto.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "conversores")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_PROJECT_ROOT)

import argparse
import asyncio
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import Artefatos.legado.SJUR_csv_to_csv_NOTIONfriendly as sjur
from Artefatos.scripts.openai_log_utils import configure_standard_logging, install_print_logger_bridge
from Artefatos.scripts.openai_progress_utils import (
    build_file_signature,
    make_backup,
    read_json_dict,
    same_file_signature,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
)


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
    checkpoint_path = sjur.resolve_checkpoint_artifact_path(output_path)
    report_path = sjur.resolve_report_artifact_path(output_path)
    cache_path = sjur.resolve_web_lookup_cache_path(output_path.parent)
    input_sig = build_file_signature(input_path)
    started_at = time.time()
    row_progress: dict[str, dict[str, object]] = {}
    perplexity_metrics = {
        "perplexity_api_calls": 0,
        "perplexity_cache_hits": 0,
        "perplexity_no_match": 0,
        "perplexity_retryable_errors": 0,
        "perplexity_skipped_existing": 0,
    }

    needed_columns: list[str] = []
    if use_tema_punchline:
        needed_columns.extend(sjur.THEME_COLUMNS)
    if use_assuntos:
        needed_columns.append("assuntos")
    if use_urls:
        needed_columns.extend(sjur.URL_COLUMNS)
    _ensure_columns(rows, fieldnames, needed_columns)

    def _sanitize_official_news(stage: str) -> None:
        cleared = sjur.sanitize_official_news_columns(rows)
        if cleared:
            logger(
                f"[{input_path.name}] [sanitize] {stage}: removidos {cleared} valores genericos/inuteis "
                "em noticia_TSE/noticia_TRE."
            )

    _sanitize_official_news("entrada")

    if backup_before_write and output_path.exists() and not checkpoint_path.exists():
        backup_path = make_backup(
            output_path,
            backup_dir=sjur.resolve_backup_artifacts_dir(),
            label="apienriched_backup",
        )
        if backup_path is not None:
            logger(f"[{input_path.name}] [Backup] {backup_path.name}")

    cp = read_json_dict(checkpoint_path)
    cp_sig = cp.get("source_signature", {})
    cp_rows = cp.get("processed_rows", [])
    if (
        int(cp.get("version", 0) or 0) in (1, sjur.CHECKPOINT_VERSION)
        and same_file_signature(cp_sig, input_sig)
        and isinstance(cp_rows, list)
        and len(cp_rows) == len(rows)
    ):
        restored = 0
        for idx, saved_row in enumerate(cp_rows):
            if not isinstance(saved_row, dict):
                continue
            for col in fieldnames:
                if col in saved_row:
                    rows[idx][col] = str(saved_row.get(col, "") or "")
            restored += 1
        cp_row_progress = cp.get("row_progress", {})
        if isinstance(cp_row_progress, dict):
            for idx_key, saved_progress in cp_row_progress.items():
                if not isinstance(saved_progress, dict):
                    continue
                row_progress[str(idx_key)] = {
                    "stage": str(saved_progress.get("stage", "") or ""),
                    "request_key": str(saved_progress.get("request_key", "") or ""),
                    "news_status": str(saved_progress.get("news_status", "") or ""),
                }
        _sanitize_official_news("checkpoint")
        logger(f"[{input_path.name}] [resume] checkpoint aplicado: {restored}/{len(rows)} linhas.")
    elif output_path.exists() and output_path.resolve() != input_path.resolve() and not checkpoint_path.exists():
        preserved_stats = sjur.preserve_columns_from_reference_rows(
            target_rows=rows,
            reference_csv=output_path,
            columns=[*sjur.DEFAULT_PRESERVE_COLUMNS, "assuntos"],
        )
        _sanitize_official_news("csv_existente")
        applied_total = int(preserved_stats.get("applied_total", 0) or 0)
        if applied_total > 0:
            logger(
                f"[{input_path.name}] [resume] reaproveitado do CSV existente: "
                f"{applied_total} celulas em {output_path.name}."
            )

    before_tema = _count_filled(rows, "tema")
    before_punchline = _count_filled(rows, "punchline")
    before_assuntos = _count_filled(rows, "assuntos")
    before_noticias = {col: _count_filled(rows, col) for col in sjur.URL_COLUMNS}

    lookup_payloads = [sjur.build_lookup_payload(row) for row in rows]
    log = lambda message: logger(f"[{input_path.name}] {message}")

    def _refresh_lookup_payloads() -> None:
        for idx, row in enumerate(rows):
            lookup_payloads[idx]["assuntos"] = row.get("assuntos", "") or ""
            lookup_payloads[idx]["partes"] = row.get("partes", "") or ""
            lookup_payloads[idx]["advogados"] = row.get("advogados", "") or ""
            lookup_payloads[idx]["tema"] = row.get("tema", "") or ""
            lookup_payloads[idx]["punchline"] = row.get("punchline", "") or ""

    def _save_state(status: str, stage: str, extra: dict[str, int] | None = None) -> None:
        completed_at = utc_now_iso() if status == "completed" else ""
        write_csv_atomic(output_path, fieldnames, rows)
        source_signature = (
            build_file_signature(output_path)
            if output_path.exists() and output_path.resolve() == input_path.resolve()
            else input_sig
        )
        write_json_atomic(
            checkpoint_path,
            {
                "version": sjur.CHECKPOINT_VERSION,
                "source_signature": source_signature,
                "input_csv": str(input_path.resolve()),
                "output_csv": str(output_path.resolve()),
                "status": status,
                "stage": stage,
                "tema_enabled": bool(use_tema_punchline),
                "assuntos_openai_enabled": bool(use_assuntos),
                "perplexity_enabled": bool(use_urls),
                "strategy_version": sjur.PERPLEXITY_NEWS_STRATEGY_VERSION,
                "cache_file": str(cache_path.resolve()),
                "row_progress": row_progress,
                "processed_rows": rows,
                "completed_at": completed_at,
                "updated_at": utc_now_iso(),
            },
        )
        write_json_atomic(
            report_path,
            {
                "script": "SJUR_csv_to_csv_APIenriching.py",
                "input_csv": str(input_path.resolve()),
                "output_csv": str(output_path.resolve()),
                "checkpoint_file": str(checkpoint_path.resolve()),
                "cache_file": str(cache_path.resolve()),
                "status": status,
                "stage": stage,
                "rows_total": len(rows),
                "tema_filled": _count_filled(rows, "tema"),
                "punchline_filled": _count_filled(rows, "punchline"),
                "assuntos_filled": _count_filled(rows, "assuntos"),
                "url_tse_filled": _count_filled(rows, "noticia_TSE"),
                "url_tre_filled": _count_filled(rows, "noticia_TRE"),
                "url_gerais_filled_total": sum(_count_filled(rows, col) for col in sjur.GENERAL_NEWS_COLUMNS),
                "perplexity_api_calls": perplexity_metrics["perplexity_api_calls"],
                "perplexity_cache_hits": perplexity_metrics["perplexity_cache_hits"],
                "perplexity_no_match": perplexity_metrics["perplexity_no_match"],
                "perplexity_retryable_errors": perplexity_metrics["perplexity_retryable_errors"],
                "perplexity_skipped_existing": perplexity_metrics["perplexity_skipped_existing"],
                "elapsed_seconds": round(max(0.0, time.time() - started_at), 2),
                "completed_at": completed_at,
                "batch_stats": extra or {},
                "updated_at": utc_now_iso(),
            },
        )

    _save_state("running", "prepared")

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
                    on_batch_done=lambda s, e, t, stats: _save_state(
                        "running",
                        f"openai_batch_{s + 1}_{e}_of_{t}",
                        stats,
                    ),
                )
            )
            _refresh_lookup_payloads()
            _save_state("running", "after_openai")

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
                    on_batch_done=lambda s, e, t, stats: _save_state(
                        "running",
                        f"openai_assuntos_batch_{s + 1}_{e}_of_{t}",
                        stats,
                    ),
                )
            )
            _refresh_lookup_payloads()
            _save_state("running", "after_openai_assuntos")

    if use_urls and web_config is not None:
        idx_urls = [
            idx
            for idx, row in enumerate(rows)
            if any(_is_blank(row.get(col, "")) for col in sjur.URL_COLUMNS)
        ]
        log(f"[Perplexity] Linhas com noticias pendentes: {len(idx_urls)} de {len(rows)}.")
        if idx_urls:
            log(
                "[Perplexity] Politica balanceada: aceita confidence=high ou medium quando houver "
                "matched_fields suficientes e validacao local de pagina noticiosa."
            )
            estimate = sjur.estimate_news_api_calls(
                [rows[idx] for idx in idx_urls],
                lookup_payloads=[lookup_payloads[idx] for idx in idx_urls],
                model=web_config.model,
                cache_path=cache_path,
            )
            log(
                "[Perplexity] Estimativa antes do run: "
                f"api={estimate['estimated_api_calls']} | "
                f"cache_terminal={estimate['estimated_cache_hits']} | "
                f"ja_preenchidas={estimate['estimated_skipped_existing']}"
            )
            perplexity_metrics = asyncio.run(
                sjur.enriquecer_rows_com_urls_async(
                    [rows[idx] for idx in idx_urls],
                    log,
                    web_config,
                    lookup_payloads=[lookup_payloads[idx] for idx in idx_urls],
                    on_batch_done=lambda s, e, t, stats: _save_state(
                        "running",
                        f"perplexity_batch_{s + 1}_{e}_of_{t}",
                        stats,
                    ),
                    cache_path=cache_path,
                    row_indices=idx_urls,
                    row_progress=row_progress,
                )
            )
            _save_state("running", "after_perplexity")

    after_tema = _count_filled(rows, "tema")
    after_punchline = _count_filled(rows, "punchline")
    after_assuntos = _count_filled(rows, "assuntos")
    after_noticias = {col: _count_filled(rows, col) for col in sjur.URL_COLUMNS}

    write_csv_atomic(output_path, fieldnames, rows)
    _save_state("completed", "final")
    sjur.cleanup_processing_artifacts(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        logger=log,
    )
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
        help="Pasta de saida do CSV final. Checkpoints/reports/cache/backups vao para Artefatos/.",
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
            self._configure_window()

            self.style = ttk.Style(self.root)
            self.style.configure("GuiTitle.TLabel", font=("Segoe UI", 12, "bold"))
            self.style.configure("Hint.TLabel", foreground="#555555")

            self.file_vars: dict[str, tk.BooleanVar] = {}
            self.managed_widget_states: dict[tk.Widget, str] = {}
            self.feature_groups: dict[str, list[tk.Widget]] = {
                "openai": [],
                "perplexity": [],
                "assuntos": [],
                "output": [],
            }
            self.secret_entries: list[tk.Widget] = []

            self.files_summary_var = tk.StringVar(value="Nenhum CSV selecionado.")
            self.run_summary_var = tk.StringVar(value="Selecione arquivos e ajuste o fluxo.")
            self.openai_status_var = tk.StringVar(value="")
            self.perplexity_status_var = tk.StringVar(value="")
            self.show_secrets_var = tk.BooleanVar(value=False)
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
            self._attach_state_traces()
            self._update_secret_visibility()
            self._sync_feature_state()

        def _configure_window(self) -> None:
            self.root.update_idletasks()
            screen_width = max(1024, int(self.root.winfo_screenwidth()))
            screen_height = max(720, int(self.root.winfo_screenheight()))
            width = min(1080, screen_width - 80)
            height = min(820, screen_height - 120)
            width = max(880, width)
            height = max(620, height)
            pos_x = max(20, (screen_width - width) // 2)
            pos_y = max(20, (screen_height - height) // 5)
            self.root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
            self.root.minsize(860, 620)

        def _register_managed_widget(
            self,
            widget: tk.Widget,
            *,
            group: str | None = None,
            enabled_state: str = "normal",
        ) -> tk.Widget:
            self.managed_widget_states[widget] = enabled_state
            if group is not None:
                self.feature_groups[group].append(widget)
            return widget

        def _register_secret_entry(self, widget: tk.Widget) -> tk.Widget:
            self.secret_entries.append(widget)
            return widget

        def _set_widgets_state(self, group: str, enabled: bool) -> None:
            for widget in self.feature_groups.get(group, []):
                try:
                    state = self.managed_widget_states.get(widget, "normal") if enabled else "disabled"
                    widget.configure(state=state)
                except Exception:
                    continue

        def _build_ui(self) -> None:
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)

            main = ttk.Frame(self.root, padding=12)
            main.grid(row=0, column=0, sticky="nsew")
            main.columnconfigure(0, weight=1)
            main.rowconfigure(2, weight=1)

            header = ttk.Frame(main)
            header.grid(row=0, column=0, sticky="ew")
            header.columnconfigure(0, weight=1)
            ttk.Label(header, text="SJUR API Enriched", style="GuiTitle.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(
                header,
                text="Selecione os CSVs, defina o fluxo e execute o enriquecimento por lote.",
                style="Hint.TLabel",
            ).grid(row=1, column=0, sticky="w", pady=(2, 0))
            ttk.Label(header, textvariable=self.files_summary_var).grid(row=0, column=1, rowspan=2, sticky="e")

            self._build_files_box(main)

            notebook = ttk.Notebook(main)
            notebook.grid(row=2, column=0, sticky="nsew", pady=(12, 0))

            flow_tab = ttk.Frame(notebook, padding=12)
            openai_tab = ttk.Frame(notebook, padding=12)
            perplexity_tab = ttk.Frame(notebook, padding=12)
            log_tab = ttk.Frame(notebook, padding=12)
            notebook.add(flow_tab, text="Fluxo")
            notebook.add(openai_tab, text="OpenAI")
            notebook.add(perplexity_tab, text="Perplexity")
            notebook.add(log_tab, text="Log")

            self._build_flow_tab(flow_tab)
            self._build_openai_tab(openai_tab)
            self._build_perplexity_tab(perplexity_tab)
            self._build_log_tab(log_tab)

            footer = ttk.Frame(main)
            footer.grid(row=3, column=0, sticky="ew", pady=(12, 0))
            footer.columnconfigure(0, weight=1)
            ttk.Label(footer, textvariable=self.run_summary_var, style="Hint.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Button(footer, text="Processar selecionados", command=self.process_selected).grid(
                row=0, column=1, sticky="e", padx=(12, 0)
            )

            self.root.bind("<Control-Return>", lambda _event: self.process_selected())
            self.refresh_file_list()

        def _build_files_box(self, parent: tk.Widget) -> None:
            files_box = ttk.LabelFrame(parent, text="Arquivos selecionados", padding=8)
            files_box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
            files_box.columnconfigure(0, weight=1)
            files_box.rowconfigure(1, weight=1)

            controls = ttk.Frame(files_box)
            controls.grid(row=0, column=0, sticky="ew")
            controls.columnconfigure(6, weight=1)
            ttk.Button(controls, text="Selecionar CSVs", command=self.add_files).grid(row=0, column=0, padx=(0, 8))
            ttk.Button(controls, text="Adicionar pasta", command=self.add_folder).grid(row=0, column=1, padx=(0, 8))
            ttk.Button(controls, text="Marcar todos", command=self.check_all).grid(row=0, column=2, padx=(0, 8))
            ttk.Button(controls, text="Desmarcar todos", command=self.uncheck_all).grid(row=0, column=3, padx=(0, 8))
            ttk.Button(controls, text="Limpar lista", command=self.clear_files).grid(row=0, column=4, padx=(0, 8))
            ttk.Label(
                controls,
                text="Use as caixas para incluir/excluir arquivos da execução.",
                style="Hint.TLabel",
            ).grid(row=0, column=6, sticky="e")

            canvas_wrap = ttk.Frame(files_box)
            canvas_wrap.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            canvas_wrap.columnconfigure(0, weight=1)
            self.canvas = tk.Canvas(canvas_wrap, height=160, highlightthickness=0)
            scrollbar = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=scrollbar.set)
            self.canvas.grid(row=0, column=0, sticky="ew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            self.files_frame = ttk.Frame(self.canvas)
            self.files_frame_window = self.canvas.create_window((0, 0), window=self.files_frame, anchor="nw")
            self.files_frame.bind(
                "<Configure>",
                lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
            )
            self.canvas.bind("<Configure>", self._on_files_canvas_configure)

        def _build_flow_tab(self, parent: ttk.Frame) -> None:
            parent.columnconfigure(0, weight=1)

            output_box = ttk.LabelFrame(parent, text="Saida", padding=10)
            output_box.grid(row=0, column=0, sticky="ew")
            output_box.columnconfigure(1, weight=1)
            ttk.Label(output_box, text="Pasta de saida (vazio = pasta do CSV):").grid(row=0, column=0, sticky="w")
            self.output_dir_entry = self._register_managed_widget(
                ttk.Entry(output_box, textvariable=self.output_dir_var),
                group="output",
            )
            self.output_dir_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
            self.output_dir_button = self._register_managed_widget(
                ttk.Button(output_box, text="Selecionar...", command=self.choose_output_dir),
                group="output",
            )
            self.output_dir_button.grid(row=0, column=2, sticky="e")
            ttk.Label(output_box, text="Sufixo de saida:").grid(row=1, column=0, sticky="w", pady=(10, 0))
            self.suffix_entry = self._register_managed_widget(
                ttk.Entry(output_box, textvariable=self.suffix_var, width=24),
                group="output",
            )
            self.suffix_entry.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
            ttk.Checkbutton(
                output_box,
                text="Sobrescrever o proprio CSV (--in-place)",
                variable=self.in_place_var,
            ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))
            ttk.Checkbutton(
                output_box,
                text="Nao criar backup quando houver arquivo de saida (--no-backup)",
                variable=self.no_backup_var,
            ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

            fill_box = ttk.LabelFrame(parent, text="O que preencher", padding=10)
            fill_box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
            fill_box.columnconfigure(0, weight=1)
            ttk.Checkbutton(
                fill_box,
                text="Tema e punchline vazios (OpenAI)",
                variable=self.skip_openai_tema_punchline_var,
                onvalue=False,
                offvalue=True,
            ).grid(row=0, column=0, sticky="w")
            ttk.Checkbutton(
                fill_box,
                text="Assuntos vazios (OpenAI)",
                variable=self.enriquecer_assuntos_var,
            ).grid(row=1, column=0, sticky="w", pady=(4, 0))
            ttk.Checkbutton(
                fill_box,
                text="Noticias vazias (Perplexity)",
                variable=self.skip_perplexity_urls_var,
                onvalue=False,
                offvalue=True,
            ).grid(row=2, column=0, sticky="w", pady=(4, 0))
            ttk.Label(
                fill_box,
                text=(
                    "As abas OpenAI e Perplexity abaixo guardam as chaves e parametros. "
                    "Campos irrelevantes ficam desativados automaticamente."
                ),
                style="Hint.TLabel",
                wraplength=760,
                justify="left",
            ).grid(row=3, column=0, sticky="w", pady=(8, 0))

            assuntos_box = ttk.LabelFrame(parent, text="Detalhes de assuntos", padding=10)
            assuntos_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
            ttk.Label(assuntos_box, text="Maximo por linha").grid(row=0, column=0, sticky="w")
            self.assuntos_max_entry = self._register_managed_widget(
                ttk.Entry(assuntos_box, textvariable=self.assuntos_max_itens_var, width=6),
                group="assuntos",
            )
            self.assuntos_max_entry.grid(row=0, column=1, sticky="w", padx=(8, 20))
            ttk.Label(assuntos_box, text="Taxonomia").grid(row=0, column=2, sticky="w")
            self.assuntos_taxonomy_combo = self._register_managed_widget(
                ttk.Combobox(
                    assuntos_box,
                    textvariable=self.assuntos_taxonomy_mode_var,
                    values=list(sjur.ASSUNTOS_TAXONOMY_CHOICES),
                    state="readonly",
                    width=12,
                ),
                group="assuntos",
                enabled_state="readonly",
            )
            self.assuntos_taxonomy_combo.grid(row=0, column=3, sticky="w", padx=(8, 0))
            ttk.Label(
                assuntos_box,
                text="Use somente quando “Assuntos vazios” estiver marcado.",
                style="Hint.TLabel",
            ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        def _build_openai_tab(self, parent: ttk.Frame) -> None:
            parent.columnconfigure(0, weight=1)
            ttk.Label(parent, textvariable=self.openai_status_var, style="Hint.TLabel").grid(
                row=0, column=0, sticky="w"
            )

            credentials_box = ttk.LabelFrame(parent, text="Chaves OpenAI", padding=10)
            credentials_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
            credentials_box.columnconfigure(1, weight=1)
            ttk.Checkbutton(
                credentials_box,
                text="Mostrar chaves na tela",
                variable=self.show_secrets_var,
            ).grid(row=0, column=0, columnspan=3, sticky="w")
            ttk.Label(credentials_box, text="API key").grid(row=1, column=0, sticky="w", pady=(10, 0))
            self.openai_key_entry = self._register_secret_entry(
                self._register_managed_widget(
                    ttk.Entry(credentials_box, textvariable=self.openai_api_key_var),
                    group="openai",
                )
            )
            self.openai_key_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
            ttk.Label(credentials_box, text="Arquivo da chave").grid(row=2, column=0, sticky="w", pady=(8, 0))
            self.openai_key_file_entry = self._register_managed_widget(
                ttk.Entry(credentials_box, textvariable=self.openai_key_file_var),
                group="openai",
            )
            self.openai_key_file_entry.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
            self.openai_key_file_button = self._register_managed_widget(
                ttk.Button(credentials_box, text="Selecionar...", command=self.choose_openai_key_file),
                group="openai",
            )
            self.openai_key_file_button.grid(row=2, column=2, sticky="e", pady=(8, 0))

            tuning_box = ttk.LabelFrame(parent, text="Parametros OpenAI", padding=10)
            tuning_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
            ttk.Label(tuning_box, text="Modelo").grid(row=0, column=0, sticky="w")
            self.openai_model_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.openai_model_var, width=12),
                group="openai",
            )
            self.openai_model_entry.grid(row=0, column=1, padx=(8, 18), sticky="w")
            ttk.Label(tuning_box, text="Workers").grid(row=0, column=2, sticky="w")
            self.openai_workers_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.openai_workers_var, width=6),
                group="openai",
            )
            self.openai_workers_entry.grid(row=0, column=3, padx=(8, 18), sticky="w")
            ttk.Label(tuning_box, text="Timeout(s)").grid(row=0, column=4, sticky="w")
            self.openai_timeout_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.openai_timeout_var, width=7),
                group="openai",
            )
            self.openai_timeout_entry.grid(row=0, column=5, padx=(8, 18), sticky="w")
            ttk.Label(tuning_box, text="Batch").grid(row=0, column=6, sticky="w")
            self.openai_batch_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.openai_batch_size_var, width=7),
                group="openai",
            )
            self.openai_batch_entry.grid(row=0, column=7, padx=(8, 0), sticky="w")
            ttk.Label(tuning_box, text="Delay(s)").grid(row=1, column=0, sticky="w", pady=(8, 0))
            self.openai_delay_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.openai_delay_var, width=7),
                group="openai",
            )
            self.openai_delay_entry.grid(row=1, column=1, padx=(8, 18), pady=(8, 0), sticky="w")
            ttk.Label(tuning_box, text="Retries").grid(row=1, column=2, sticky="w", pady=(8, 0))
            self.openai_retries_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.openai_retries_var, width=7),
                group="openai",
            )
            self.openai_retries_entry.grid(row=1, column=3, padx=(8, 18), pady=(8, 0), sticky="w")
            ttk.Label(tuning_box, text="Target RPM").grid(row=1, column=4, sticky="w", pady=(8, 0))
            self.openai_rpm_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.openai_target_rpm_var, width=8),
                group="openai",
            )
            self.openai_rpm_entry.grid(row=1, column=5, padx=(8, 0), pady=(8, 0), sticky="w")

        def _build_perplexity_tab(self, parent: ttk.Frame) -> None:
            parent.columnconfigure(0, weight=1)
            ttk.Label(parent, textvariable=self.perplexity_status_var, style="Hint.TLabel").grid(
                row=0, column=0, sticky="w"
            )

            credentials_box = ttk.LabelFrame(parent, text="Chaves Perplexity", padding=10)
            credentials_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
            credentials_box.columnconfigure(1, weight=1)
            ttk.Label(credentials_box, text="API key").grid(row=0, column=0, sticky="w")
            self.perplexity_key_entry = self._register_secret_entry(
                self._register_managed_widget(
                    ttk.Entry(credentials_box, textvariable=self.perplexity_api_key_var),
                    group="perplexity",
                )
            )
            self.perplexity_key_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
            ttk.Label(credentials_box, text="Arquivo da chave").grid(row=1, column=0, sticky="w", pady=(8, 0))
            self.perplexity_key_file_entry = self._register_managed_widget(
                ttk.Entry(credentials_box, textvariable=self.perplexity_key_file_var),
                group="perplexity",
            )
            self.perplexity_key_file_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
            self.perplexity_key_file_button = self._register_managed_widget(
                ttk.Button(credentials_box, text="Selecionar...", command=self.choose_perplexity_key_file),
                group="perplexity",
            )
            self.perplexity_key_file_button.grid(row=1, column=2, sticky="e", pady=(8, 0))

            tuning_box = ttk.LabelFrame(parent, text="Parametros Perplexity", padding=10)
            tuning_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
            ttk.Label(tuning_box, text="Modelo").grid(row=0, column=0, sticky="w")
            self.perplexity_model_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.perplexity_model_var, width=12),
                group="perplexity",
            )
            self.perplexity_model_entry.grid(row=0, column=1, padx=(8, 18), sticky="w")
            ttk.Label(tuning_box, text="Workers").grid(row=0, column=2, sticky="w")
            self.perplexity_workers_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.perplexity_workers_var, width=6),
                group="perplexity",
            )
            self.perplexity_workers_entry.grid(row=0, column=3, padx=(8, 18), sticky="w")
            ttk.Label(tuning_box, text="Timeout(s)").grid(row=0, column=4, sticky="w")
            self.perplexity_timeout_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.perplexity_timeout_var, width=7),
                group="perplexity",
            )
            self.perplexity_timeout_entry.grid(row=0, column=5, padx=(8, 18), sticky="w")
            ttk.Label(tuning_box, text="Batch").grid(row=0, column=6, sticky="w")
            self.perplexity_batch_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.perplexity_batch_size_var, width=7),
                group="perplexity",
            )
            self.perplexity_batch_entry.grid(row=0, column=7, padx=(8, 0), sticky="w")
            ttk.Label(tuning_box, text="Delay(s)").grid(row=1, column=0, sticky="w", pady=(8, 0))
            self.perplexity_delay_entry = self._register_managed_widget(
                ttk.Entry(tuning_box, textvariable=self.perplexity_delay_var, width=7),
                group="perplexity",
            )
            self.perplexity_delay_entry.grid(row=1, column=1, padx=(8, 0), pady=(8, 0), sticky="w")
            ttk.Label(
                tuning_box,
                text="Usado apenas para a busca de noticias TSE/TRE/gerais.",
                style="Hint.TLabel",
            ).grid(row=2, column=0, columnspan=8, sticky="w", pady=(10, 0))

        def _build_log_tab(self, parent: ttk.Frame) -> None:
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(1, weight=1)

            controls = ttk.Frame(parent)
            controls.grid(row=0, column=0, sticky="ew")
            controls.columnconfigure(0, weight=1)
            ttk.Checkbutton(
                controls,
                text="Espelhar log no terminal",
                variable=self.verbose_terminal_var,
            ).grid(row=0, column=0, sticky="w")
            ttk.Button(controls, text="Limpar log", command=self.clear_log).grid(row=0, column=1, sticky="e")

            log_box = ttk.Frame(parent)
            log_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
            log_box.columnconfigure(0, weight=1)
            log_box.rowconfigure(0, weight=1)
            self.log_widget = tk.Text(log_box, height=14, wrap="word")
            self.log_widget.grid(row=0, column=0, sticky="nsew")
            log_scrollbar = ttk.Scrollbar(log_box, orient="vertical", command=self.log_widget.yview)
            log_scrollbar.grid(row=0, column=1, sticky="ns")
            self.log_widget.configure(yscrollcommand=log_scrollbar.set, state="disabled")

        def _on_files_canvas_configure(self, event: tk.Event) -> None:
            self.canvas.itemconfigure(self.files_frame_window, width=event.width)

        def _attach_state_traces(self) -> None:
            for var in (
                self.show_secrets_var,
                self.in_place_var,
                self.skip_openai_tema_punchline_var,
                self.enriquecer_assuntos_var,
                self.skip_perplexity_urls_var,
            ):
                var.trace_add("write", lambda *_args: self._sync_feature_state())

        def _update_secret_visibility(self) -> None:
            mask = "" if self.show_secrets_var.get() else "*"
            for entry in self.secret_entries:
                try:
                    entry.configure(show=mask)
                except Exception:
                    continue

        def _sync_feature_state(self) -> None:
            self._update_secret_visibility()
            in_place = bool(self.in_place_var.get())
            openai_enabled = (not self.skip_openai_tema_punchline_var.get()) or self.enriquecer_assuntos_var.get()
            perplexity_enabled = not self.skip_perplexity_urls_var.get()
            assuntos_enabled = bool(self.enriquecer_assuntos_var.get())

            self._set_widgets_state("output", not in_place)
            self._set_widgets_state("openai", openai_enabled)
            self._set_widgets_state("perplexity", perplexity_enabled)
            self._set_widgets_state("assuntos", assuntos_enabled)

            if in_place:
                self.run_summary_var.set(
                    "Saida: sobrescrevendo os arquivos de entrada. Sufixo e pasta de saida ficam desativados."
                )
            else:
                parts: list[str] = []
                if not self.skip_openai_tema_punchline_var.get():
                    parts.append("tema/punchline")
                if assuntos_enabled:
                    parts.append("assuntos")
                if perplexity_enabled:
                    parts.append("noticias")
                summary = ", ".join(parts) if parts else "nada selecionado"
                self.run_summary_var.set(f"Fluxo ativo: {summary}.")

            self.openai_status_var.set(
                "OpenAI ativo para os campos marcados no fluxo."
                if openai_enabled
                else "OpenAI desativado. Marque tema/punchline e/ou assuntos na aba Fluxo."
            )
            self.perplexity_status_var.set(
                "Perplexity ativa para preenchimento de noticias."
                if perplexity_enabled
                else "Perplexity desativada. Marque noticias na aba Fluxo."
            )

        def log(self, message: str) -> None:
            self.log_widget.configure(state="normal")
            self.log_widget.insert("end", f"{message}\n")
            self.log_widget.see("end")
            self.log_widget.configure(state="disabled")
            if self.verbose_terminal_var.get():
                print(message, flush=True)
            self.root.update_idletasks()

        def clear_log(self) -> None:
            self.log_widget.configure(state="normal")
            self.log_widget.delete("1.0", "end")
            self.log_widget.configure(state="disabled")

        def choose_openai_key_file(self) -> None:
            selected_path = filedialog.askopenfilename(
                title="Selecione o arquivo da chave OpenAI",
                filetypes=[("TXT", "*.txt"), ("Todos os arquivos", "*.*")],
            )
            if selected_path:
                self.openai_key_file_var.set(selected_path)

        def choose_perplexity_key_file(self) -> None:
            selected_path = filedialog.askopenfilename(
                title="Selecione o arquivo da chave Perplexity",
                filetypes=[("TXT", "*.txt"), ("Todos os arquivos", "*.*")],
            )
            if selected_path:
                self.perplexity_key_file_var.set(selected_path)

        def _add_file_path(self, raw_path: str) -> None:
            file_path = str(Path(raw_path).expanduser().resolve())
            if file_path in self.file_vars:
                return
            variable = tk.BooleanVar(value=True)
            variable.trace_add("write", lambda *_args: self._update_file_summary())
            self.file_vars[file_path] = variable

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
                self._add_file_path(str(raw_path))
            self.refresh_file_list()

        def add_folder(self) -> None:
            folder = filedialog.askdirectory(title="Selecione a pasta com CSVs")
            if not folder:
                return
            for file_path in sorted(Path(folder).expanduser().rglob("*.csv")):
                self._add_file_path(str(file_path))
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

        def _update_file_summary(self) -> None:
            total = len(self.file_vars)
            selected = sum(1 for variable in self.file_vars.values() if variable.get())
            if total == 0:
                self.files_summary_var.set("Nenhum CSV selecionado.")
            else:
                self.files_summary_var.set(f"{selected} de {total} arquivo(s) marcados.")

        def _format_file_caption(self, file_path: str) -> tuple[str, str]:
            path = Path(file_path)
            parent_label = path.parent.name or str(path.parent)
            return f"{path.name}  [{parent_label}]", str(path.parent)

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
            else:
                for row_index, file_path in enumerate(sorted(self.file_vars)):
                    row = ttk.Frame(self.files_frame)
                    row.grid(row=row_index, column=0, sticky="ew", padx=4, pady=3)
                    row.columnconfigure(0, weight=1)
                    title, subtitle = self._format_file_caption(file_path)
                    ttk.Checkbutton(row, text=title, variable=self.file_vars[file_path]).grid(
                        row=0,
                        column=0,
                        sticky="w",
                    )
                    ttk.Label(row, text=subtitle, style="Hint.TLabel").grid(
                        row=1,
                        column=0,
                        sticky="w",
                        padx=(26, 0),
                    )
            self.files_frame.update_idletasks()
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)
            self._update_file_summary()

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
