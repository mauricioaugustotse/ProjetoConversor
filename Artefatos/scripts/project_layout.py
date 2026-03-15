from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable


SCRIPTS_DIR = Path(__file__).resolve().parent
ARTEFATOS_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = ARTEFATOS_DIR.parent

LOGS_DIR = ARTEFATOS_DIR / "logs"
REPORTS_DIR = ARTEFATOS_DIR / "reports"
CHECKPOINTS_DIR = ARTEFATOS_DIR / "checkpoints"
KEYS_DIR = ARTEFATOS_DIR / "chaves"

PHASE2_DIR = ARTEFATOS_DIR / "phase2"
PHASE2_NOTION_CHAT_DIR = PHASE2_DIR / "notion_chat"
PHASE2_PAYLOADS_DIR = PHASE2_DIR / "payloads"
PHASE2_PROMPTS_DIR = PHASE2_DIR / "prompts"
PHASE2_QUEUES_DIR = PHASE2_DIR / "queues"
PHASE2_BLOCKS_DIR = PHASE2_DIR / "blocos"
PHASE2_MANUAL_PLAN_DIR = PHASE2_DIR / "manual_plan"

DATA_DIR = ARTEFATOS_DIR / "dados"
DATA_CSV_DIR = DATA_DIR / "csv"
DATA_JSON_DIR = DATA_DIR / "json"
DATA_HTML_DIR = DATA_DIR / "html"
DATA_PDF_DIR = DATA_DIR / "pdf"
LEGACY_DIR = ARTEFATOS_DIR / "legado"
MISC_DIR = ARTEFATOS_DIR / "diversos"

STANDARD_DIRS = (
    SCRIPTS_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    CHECKPOINTS_DIR,
    KEYS_DIR,
    PHASE2_DIR,
    PHASE2_NOTION_CHAT_DIR,
    PHASE2_PAYLOADS_DIR,
    PHASE2_PROMPTS_DIR,
    PHASE2_QUEUES_DIR,
    PHASE2_BLOCKS_DIR,
    PHASE2_MANUAL_PLAN_DIR,
    DATA_DIR,
    DATA_CSV_DIR,
    DATA_JSON_DIR,
    DATA_HTML_DIR,
    DATA_PDF_DIR,
    LEGACY_DIR,
    MISC_DIR,
)


def ensure_standard_layout() -> None:
    for path in STANDARD_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(raw_path: str, *, default: Path | None = None, base_dir: Path = PROJECT_ROOT) -> Path:
    candidate = str(raw_path or "").strip()
    path = Path(candidate) if candidate else (default if default is not None else base_dir)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve()


def default_report_path(script_stem: str) -> Path:
    ensure_standard_layout()
    return REPORTS_DIR / f".{script_stem}.report.json"


def default_checkpoint_path(script_stem: str) -> Path:
    ensure_standard_layout()
    return CHECKPOINTS_DIR / f".{script_stem}.checkpoint.json"


def named_log_path(filename: str) -> Path:
    ensure_standard_layout()
    return (LOGS_DIR / filename).resolve()


def default_script_log_path(script_stem: str) -> Path:
    return named_log_path(f"{script_stem}.log")


def default_manual_plan_path(filename: str = "notion_etiquetas_plano_manual_fase2.csv") -> Path:
    ensure_standard_layout()
    return (PHASE2_MANUAL_PLAN_DIR / filename).resolve()


def default_manual_plan_timestamped_path(prefix: str = "notion_etiquetas_plano_manual") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return default_manual_plan_path(f"{prefix}_{stamp}.csv")


def default_phase2_blocks_dir(*, stamp: str | None = None) -> Path:
    ensure_standard_layout()
    final_stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PHASE2_BLOCKS_DIR / f"notion_etiquetas_fase2_blocos_{final_stamp}").resolve()


def default_phase2_queue_path(filename: str = "phase2_chat_queue.csv") -> Path:
    ensure_standard_layout()
    return (PHASE2_QUEUES_DIR / filename).resolve()


def phase2_payload_path(prop_token: str) -> Path:
    ensure_standard_layout()
    return (PHASE2_PAYLOADS_DIR / f"phase2_{prop_token}.json").resolve()


def phase2_prompt_path(prop_token: str) -> Path:
    ensure_standard_layout()
    return (PHASE2_PROMPTS_DIR / f"phase2_{prop_token}.prompt.txt").resolve()


def notion_secret_candidates(filename: str = "Chave_Notion.txt") -> Iterable[Path]:
    seen: set[str] = set()
    candidates = (
        Path.cwd() / filename,
        KEYS_DIR / filename,
        PROJECT_ROOT / filename,
        SCRIPTS_DIR / filename,
    )
    for path in candidates:
        resolved = str(path.expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        yield Path(resolved)
