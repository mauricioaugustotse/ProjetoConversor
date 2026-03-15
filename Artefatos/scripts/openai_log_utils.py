"""
Utilitários de logging padronizado para pipelines de extração/enriquecimento.

Fluxo de trabalho:
1. Resolve nível de log com base em `verbose`, `quiet` e `debug`.
2. Configura handlers de console/arquivo com formatter único.
3. Reduz verbosidade de bibliotecas ruidosas (`openai`, `httpx`, `urllib3`).
4. Expõe bridge opcional para redirecionar `print` para o logger.
"""

from __future__ import annotations

import builtins
import logging
import sys
from pathlib import Path
from typing import Any, MutableMapping, Optional


NOISY_LOGGERS = (
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "urllib3",
)


class StandardFormatter(logging.Formatter):
    LEVEL_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "AVISO",
        "ERROR": "ERRO",
        "CRITICAL": "ERRO",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = self.LEVEL_MAP.get(record.levelname, record.levelname)
        return super().format(record)


def _resolve_level(*, verbose: bool, quiet: bool, debug: bool) -> int:
    if debug:
        return logging.DEBUG
    if verbose:
        return logging.DEBUG
    if quiet:
        return logging.WARNING
    return logging.INFO


def configure_standard_logging(
    script_name: str,
    *,
    verbose: bool = False,
    quiet: bool = False,
    debug: bool = False,
    log_file: str = "",
) -> logging.Logger:
    level = _resolve_level(verbose=verbose, quiet=quiet, debug=debug)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = StandardFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    resolved_log_path: Optional[Path] = None
    if (log_file or "").strip():
        resolved_log_path = Path(log_file).expanduser().resolve()
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(script_name)
    logger.debug(
        "Logger configurado: level=%s | quiet=%s | verbose=%s | debug=%s",
        logging.getLevelName(level),
        quiet,
        verbose,
        debug,
    )
    if resolved_log_path is not None:
        logger.info("Arquivo de log: %s", resolved_log_path)
    return logger


def install_print_logger_bridge(module_globals: MutableMapping[str, Any], logger: logging.Logger) -> None:
    builtin_print = builtins.print

    def _logger_print(*args: Any, **kwargs: Any) -> None:
        target = kwargs.get("file", sys.stdout)
        if target not in (None, sys.stdout, sys.stderr):
            builtin_print(*args, **kwargs)
            return

        sep = kwargs.get("sep", " ")
        message = sep.join(str(arg) for arg in args).strip()
        if not message:
            return

        level = logging.ERROR if target is sys.stderr else logging.INFO
        logger.log(level, message)

        if kwargs.get("flush", False):
            for handler in logging.getLogger().handlers:
                try:
                    handler.flush()
                except Exception:
                    pass

    module_globals["print"] = _logger_print
