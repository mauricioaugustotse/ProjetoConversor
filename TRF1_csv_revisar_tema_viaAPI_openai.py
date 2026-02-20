#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRF1 CSV -> CSV (revisão da coluna "tema" via OpenAI).

Objetivo:
- Reprocessar apenas a coluna `tema` de um CSV existente.
- Permitir seleção do CSV por CLI ou GUI.
- Rodar em lotes com workers + controle de ritmo (RPM), retries e autoajuste.
- Manter segurança operacional com backup + checkpoint/resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
    import openai
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERRO: openai não encontrado. Execute: pip install openai") from exc


CHECKPOINT_VERSION = 1
DEFAULT_OUTPUT_SUFFIX = "_tema_revisado"
DEFAULT_CHECKPOINT_PREFIX = ".tema_review_openai"
MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a"}

TEMA_MIN_WORDS = 8
TEMA_MAX_WORDS = 30
TEMA_TRAILING_CONNECTORS = {
    "a",
    "ao",
    "aos",
    "à",
    "às",
    "com",
    "contra",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "na",
    "nas",
    "no",
    "nos",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "que",
    "se",
    "sem",
    "sob",
    "sobre",
}
SPACE_RE = re.compile(r"\s+")
PROFILE_CHOICES = ("lean", "balanced", "full")
PROFILE_FALLBACK_CHOICES = ("auto", "off")
PROFILE_AUTO_CHAIN = {
    "lean": ["lean", "balanced"],
    "balanced": ["balanced", "full"],
    "full": ["full"],
}
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]{3,}")
PT_STOPWORDS = {
    "para",
    "com",
    "sobre",
    "pela",
    "pelas",
    "pelos",
    "entre",
    "contra",
    "apenas",
    "caso",
    "julgado",
    "julgamento",
    "direito",
    "processo",
    "acao",
    "ação",
    "tema",
    "boletim",
    "jurisprudencia",
    "jurisprudência",
    "tribunal",
    "federal",
    "regiao",
    "região",
    "publico",
    "público",
    "publica",
    "pública",
    "administracao",
    "administração",
    "administrativo",
    "administrativa",
    "civil",
    "penal",
    "constitucional",
    "processual",
    "recurso",
    "decisao",
    "decisão",
    "acordao",
    "acórdão",
    "sentenca",
    "sentença",
}

TEMA_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tema": {"type": "string"},
    },
    "required": ["tema"],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_ws(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in MISSING_TOKENS


def cleanup_text(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\u00ad", "")
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\s+,", ",", s)
    s = normalize_ws(s)
    return s.strip(" \n\t")


def sanitize_boletim_text(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    return cleanup_text(s)


def tema_word_count(tema: str) -> int:
    return len([t for t in normalize_ws(tema).split(" ") if t])


def strip_tema_trailing_connectors(tokens: List[str]) -> List[str]:
    out = list(tokens)
    while out:
        tail = out[-1].strip(" \t\n\r.,;:!?()[]{}\"'")
        if not tail:
            out.pop()
            continue
        if tail.casefold() in TEMA_TRAILING_CONNECTORS:
            out.pop()
            continue
        break
    return out


def tema_looks_truncated(tema: str) -> bool:
    text = normalize_ws(tema)
    if not text:
        return True
    if re.search(r"[,:;/\-]\s*$", text):
        return True
    if re.search(r"\b(?:ap[oó]s|at[eé]|desde|de|em|entre)\s+\d{1,2}\s*$", text, flags=re.IGNORECASE):
        return True
    tokens = [t for t in text.split(" ") if t]
    if not tokens:
        return True
    tail = tokens[-1].strip(" \t\n\r.,;:!?()[]{}\"'").casefold()
    if not tail:
        return True
    return tail in TEMA_TRAILING_CONNECTORS


def tema_is_telegraphic(tema: str) -> bool:
    text = normalize_ws(tema)
    if not text:
        return False
    return len(re.findall(r"[.;:]", text)) >= 2


def normalize_tema_text(tema: str, *, max_words: int = TEMA_MAX_WORDS) -> str:
    text = normalize_ws(str(tema or ""))
    if not text:
        return ""
    lead_in = re.search(
        r"(?:^|[.;:])\s*O\s+(?:Superior Tribunal de Justiça|Supremo Tribunal Federal)\b",
        text,
        flags=re.IGNORECASE,
    )
    if lead_in and lead_in.start() > 0:
        text = normalize_ws(text[: lead_in.start()])
    tokens = [t for t in text.split(" ") if t]
    if max_words > 0 and len(tokens) > max_words:
        tokens = tokens[:max_words]
    tokens = strip_tema_trailing_connectors(tokens)
    text = " ".join(tokens).strip(" ,;:-")
    text = re.sub(r"[/:;\-]+\s*$", "", text).strip(" ,;:-")
    return normalize_ws(text)


def is_tema_detailed(tema: str) -> bool:
    wc = tema_word_count(tema)
    if not (TEMA_MIN_WORDS <= wc <= TEMA_MAX_WORDS):
        return False
    if tema_looks_truncated(tema):
        return False
    return not tema_is_telegraphic(tema)


def compact_for_prompt(text: str, max_chars: int, *, keep_tail: int = 420) -> str:
    cleaned = sanitize_boletim_text(text)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    keep_tail = max(120, min(keep_tail, max_chars // 2))
    head_len = max(120, max_chars - keep_tail - 24)
    head = cleaned[:head_len].rstrip()
    tail = cleaned[-keep_tail:].lstrip()
    return f"{head} ... [trecho omitido] ... {tail}"


def truncate_chars(text: str, max_chars: int) -> str:
    cleaned = sanitize_boletim_text(text)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip(" ,;:-")


def split_bullet_points_for_prompt(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value or "")
    if not text:
        return []
    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    parts: List[str] = []
    if "•" in raw:
        parts = [normalize_ws(x) for x in re.split(r"\s*•\s*", raw) if normalize_ws(x)]
    else:
        parts = [normalize_ws(x) for x in re.split(r"[\n;|]+", raw) if normalize_ws(x)]
    cleaned: List[str] = []
    seen: set[str] = set()
    for p in parts:
        item = normalize_ws(re.sub(r"^[\-\*\d\.\)\s]+", "", p))
        item = item.strip(" ,;:-")
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def compact_bullet_points_for_prompt(value: Any, max_items: int, max_chars_each: int = 120) -> str:
    items = split_bullet_points_for_prompt(value)
    if max_items > 0:
        items = items[:max_items]
    items = [truncate_chars(x, max_chars_each) for x in items if truncate_chars(x, max_chars_each)]
    if not items:
        return ""
    return " | ".join(items)


def split_jurisprudencia_for_prompt(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value or "")
    if not text:
        return []
    parts = [normalize_ws(x) for x in re.split(r"[,;\n\r|]+", text) if normalize_ws(x)]
    cleaned: List[str] = []
    seen: set[str] = set()
    for p in parts:
        item = normalize_ws(p).strip(" ,;:-")
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def compact_jurisprudencia_numerica_for_prompt(
    value: Any,
    max_items: int = 8,
    max_chars_each: int = 120,
) -> str:
    items = split_jurisprudencia_for_prompt(value)
    # Usa apenas entradas com referência numérica (REsp, Tema, súmula etc. com dígitos).
    numericas = [x for x in items if re.search(r"\d", x)]
    if max_items > 0:
        numericas = numericas[:max_items]
    numericas = [truncate_chars(x, max_chars_each) for x in numericas if truncate_chars(x, max_chars_each)]
    if not numericas:
        return ""
    return " | ".join(numericas)


def keyword_tokens(text: str) -> set[str]:
    tokens = {t.casefold() for t in TOKEN_RE.findall(text or "")}
    return {t for t in tokens if t not in PT_STOPWORDS}


def tema_has_min_alignment(tema: str, row: Dict[str, str]) -> bool:
    """
    Verificação leve para reduzir temas desconectados do núcleo do caso.
    """
    tema_kw = keyword_tokens(tema)
    if not tema_kw:
        return False
    source = normalize_ws(f"{row.get('tese', '')} {row.get('punchline', '')}")
    source_kw = keyword_tokens(source)
    if len(source_kw) < 3:
        # Evita falso negativo quando tese/punchline vierem pobres.
        return True
    return bool(tema_kw & source_kw)


def build_profile_chain(base_profile: str, fallback_policy: str) -> List[str]:
    base = (base_profile or "lean").strip().lower()
    if base not in PROFILE_CHOICES:
        base = "lean"
    policy = (fallback_policy or "auto").strip().lower()
    if policy not in PROFILE_FALLBACK_CHOICES:
        policy = "auto"
    if policy == "off":
        return [base]
    return list(PROFILE_AUTO_CHAIN.get(base, [base]))


def build_profile_inputs(
    row: Dict[str, str],
    *,
    profile: str,
    text_max_chars: int,
    contexto_max_chars: int,
    bullets_max_items: int,
) -> Dict[str, str]:
    p = (profile or "lean").strip().lower()
    if p not in PROFILE_CHOICES:
        p = "lean"
    out: Dict[str, str] = {
        "ramo_do_direito": normalize_ws(row.get("ramo_do_direito", "")),
        "subramo_do_direito": normalize_ws(row.get("subramo_do_direito", "")),
        "punchline": normalize_ws(row.get("punchline", "")),
        "tese": normalize_ws(row.get("tese", "")),
        "bullet_points": compact_bullet_points_for_prompt(row.get("bullet_points", ""), bullets_max_items),
        "jurisprudencia_numerica": compact_jurisprudencia_numerica_for_prompt(
            row.get("jurisprudência", row.get("jurisprudencia", ""))
        ),
        "contexto": "",
        "texto_do_boletim": "",
    }
    if p in {"balanced", "full"}:
        out["contexto"] = truncate_chars(row.get("contexto", ""), contexto_max_chars)
    if p == "full":
        out["texto_do_boletim"] = compact_for_prompt(row.get("texto_do_boletim", ""), text_max_chars)
    return out


def read_secret_from_file(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not raw:
        return ""
    first_line = raw.splitlines()[0].strip()
    if "=" in first_line:
        first_line = first_line.split("=", 1)[1].strip()
    return first_line.strip("\"' ")


def resolve_openai_key(cli_value: str) -> str:
    if cli_value.strip():
        return cli_value.strip()
    env = os.getenv("OPENAI_API_KEY", "").strip()
    if env:
        return env
    for fallback in ("CHAVE_SECRETA_API_Mauricio_local.txt",):
        val = read_secret_from_file(fallback)
        if val:
            return val
    return ""


def setup_logger(verbose: bool, quiet: bool) -> logging.Logger:
    logger = logging.getLogger("trf1_tema_review_openai")
    logger.handlers.clear()
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def select_input_csv_via_gui(logger: logging.Logger) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        logger.warning("GUI indisponível (tkinter): %s", exc)
        return ""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Selecione o CSV para revisão da coluna tema",
        filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
    )
    root.destroy()
    return selected or ""


def ensure_tema_column(fieldnames: List[str], rows: List[Dict[str, str]]) -> List[str]:
    if "tema" in fieldnames:
        return fieldnames
    out = list(fieldnames)
    if "tese" in out:
        idx = out.index("tese") + 1
        out.insert(idx, "tema")
    else:
        out.append("tema")
    for row in rows:
        row.setdefault("tema", "")
    return out


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        cleaned_iter = (line.replace("\x00", "") for line in f)
        reader = csv.DictReader(cleaned_iter)
        if not reader.fieldnames:
            return [], []
        fieldnames = [str(x) for x in reader.fieldnames]
        for raw in reader:
            row = {k: str((raw or {}).get(k, "") or "") for k in fieldnames}
            rows.append(row)
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def generate_row_id(row: Dict[str, str]) -> str:
    texto_fingerprint = cleanup_text(row.get("texto_do_boletim", ""))[:180]
    relator_value = row.get("relator(a)", "") or row.get("relator", "")
    base = "|".join(
        [
            row.get("numero_processo", ""),
            row.get("informativo", ""),
            relator_value,
            row.get("data_julgamento", ""),
            texto_fingerprint,
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def build_openai_request_key(
    row: Dict[str, str],
    *,
    profile: str,
    text_max_chars: int,
    contexto_max_chars: int,
    bullets_max_items: int,
) -> str:
    profile_inputs = build_profile_inputs(
        row,
        profile=profile,
        text_max_chars=text_max_chars,
        contexto_max_chars=contexto_max_chars,
        bullets_max_items=bullets_max_items,
    )
    base = "|".join(
        [
            normalize_ws(profile),
            normalize_ws(row.get("numero_processo", "")),
            normalize_ws(row.get("classe", "")),
            normalize_ws(row.get("relator(a)", row.get("relator", ""))),
            normalize_ws(row.get("tema", "")),
            normalize_ws(profile_inputs.get("ramo_do_direito", "")),
            normalize_ws(profile_inputs.get("subramo_do_direito", "")),
            normalize_ws(profile_inputs.get("punchline", "")),
            normalize_ws(profile_inputs.get("tese", "")),
            normalize_ws(profile_inputs.get("bullet_points", "")),
            normalize_ws(profile_inputs.get("jurisprudencia_numerica", "")),
            normalize_ws(profile_inputs.get("contexto", "")),
            normalize_ws(profile_inputs.get("texto_do_boletim", "")),
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def compute_dataset_signature(rows: Sequence[Dict[str, str]], fieldnames: Sequence[str], source_path: Path) -> Dict[str, Any]:
    h = hashlib.sha1()
    h.update(str(source_path.resolve()).encode("utf-8", errors="ignore"))
    h.update("\n".join(fieldnames).encode("utf-8", errors="ignore"))
    h.update(str(len(rows)).encode("utf-8", errors="ignore"))
    for row in rows:
        h.update((row.get("_row_id", "") + "\n").encode("utf-8", errors="ignore"))
    return {
        "path": str(source_path.resolve()),
        "rows": int(len(rows)),
        "digest": h.hexdigest(),
    }


def same_signature(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        normalize_ws(str(a.get("path", ""))) == normalize_ws(str(b.get("path", "")))
        and int(a.get("rows", -1) or -1) == int(b.get("rows", -2) or -2)
        and normalize_ws(str(a.get("digest", ""))) == normalize_ws(str(b.get("digest", "")))
    )


def read_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def checkpoint_payload(
    *,
    source_signature: Dict[str, Any],
    output_csv: Path,
    reprocess_mode: str,
    processed_rows: Dict[str, Dict[str, str]],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "source_signature": source_signature,
        "output_csv": str(output_csv.resolve()),
        "reprocess_mode": reprocess_mode,
        "processed_rows": processed_rows,
        "stats": stats,
        "updated_at": utc_now_iso(),
    }


def extract_processed_rows(cp: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    raw = cp.get("processed_rows", {})
    if not isinstance(raw, dict):
        return out
    for rid, item in raw.items():
        if not isinstance(rid, str) or not isinstance(item, dict):
            continue
        tema = normalize_tema_text(str(item.get("tema", "") or ""))
        if not tema:
            continue
        out[rid] = {
            "tema": tema,
            "updated_at": normalize_ws(str(item.get("updated_at", "") or "")),
        }
    return out


def make_backup(path: Path, backup_dir: Optional[Path], logger: logging.Logger) -> Optional[Path]:
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = backup_dir if backup_dir else path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    backup_name = f"{path.name}.backup_{ts}"
    dst = base_dir / backup_name
    shutil.copy2(path, dst)
    logger.info("Backup criado: %s", dst)
    return dst


def count_invalid_temas(rows: Sequence[Dict[str, str]]) -> int:
    return sum(1 for row in rows if needs_tema_review(row, mode="invalid_only"))


def ensure_rows_ready(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        row.setdefault("tema", "")
        row["tema"] = normalize_tema_text(row.get("tema", ""))
        row["_row_id"] = row.get("_row_id", "") or generate_row_id(row)


def apply_checkpoint_to_rows(rows: List[Dict[str, str]], processed_rows: Dict[str, Dict[str, str]]) -> int:
    by_id = {row.get("_row_id", ""): row for row in rows}
    applied = 0
    for rid, item in processed_rows.items():
        row = by_id.get(rid)
        if not row:
            continue
        tema = normalize_tema_text(item.get("tema", ""))
        if not tema:
            continue
        row["tema"] = tema
        applied += 1
    return applied


def needs_tema_review(row: Dict[str, str], mode: str) -> bool:
    tema = normalize_tema_text(row.get("tema", ""))
    if mode == "all":
        return True
    if not tema:
        return True
    if not is_tema_detailed(tema):
        return True
    if re.search(
        r"\b(?:O\s+Superior Tribunal de Justiça|O\s+Supremo Tribunal Federal)\b",
        tema,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def build_openai_tema_prompt(
    row: Dict[str, str],
    *,
    profile: str,
    text_max_chars: int,
    contexto_max_chars: int,
    bullets_max_items: int,
) -> str:
    insumos = build_profile_inputs(
        row,
        profile=profile,
        text_max_chars=text_max_chars,
        contexto_max_chars=contexto_max_chars,
        bullets_max_items=bullets_max_items,
    )
    contexto_block = insumos.get("contexto", "")
    boletim_block = insumos.get("texto_do_boletim", "")
    contexto_line = f"- contexto: {contexto_block}" if contexto_block else "- contexto: [não fornecido neste perfil]"
    boletim_line = (
        f"- texto_do_boletim: {boletim_block}"
        if boletim_block
        else "- texto_do_boletim: [não fornecido neste perfil]"
    )
    return f"""
Você é um assistente jurídico especializado em taxonomia de temas.
Sua tarefa é revisar e reescrever SOMENTE a coluna `tema` para indexação jurídica.

REGRAS ESTRITAS PARA `tema`:
1) Entre 8 e 30 palavras.
2) Frase completa, sem interrupções.
3) Não terminar em preposição/conjunção (ex.: "de", "em", "sob", "para", "e").
4) Não iniciar narrativa longa de precedente (evite: "O Superior Tribunal de Justiça, sob ...").
5) Descrever objeto jurídico + controvérsia central do caso.
6) Não copiar períodos inteiros do texto do boletim.
7) Português claro, técnico e conciso.
8) Priorize precisão usando os insumos disponíveis no perfil atual.
9) Se faltarem dados no perfil atual, use com parcimônia os metadados do julgado.

Contexto do julgado:
- perfil_de_insumo: {profile}
- classe: {row.get('classe', '')}
- numero_processo: {row.get('numero_processo', '')}
- relator(a): {row.get('relator(a)', row.get('relator', ''))}
- orgao_julgador: {row.get('orgao_julgador', '')}
- data_julgamento: {row.get('data_julgamento', '')}
- tema_atual: {row.get('tema', '')}
- ramo_do_direito: {insumos.get('ramo_do_direito', '')}
- subramo_do_direito: {insumos.get('subramo_do_direito', '')}
- punchline: {insumos.get('punchline', '')}
- tese: {insumos.get('tese', '')}
- bullet_points: {insumos.get('bullet_points', '')}
- jurisprudência_numerica: {insumos.get('jurisprudencia_numerica', '')}
{contexto_line}
{boletim_line}

Retorne APENAS JSON válido: {{"tema":"..."}}
""".strip()


def parse_openai_message_content(content_obj: Any) -> str:
    if isinstance(content_obj, str):
        return content_obj.strip()
    if isinstance(content_obj, list):
        chunks: List[str] = []
        for item in content_obj:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks).strip()
    return ""


def classify_openai_error(err_text: str) -> str:
    low = (err_text or "").lower()
    if "empty_json_due_length" in low:
        return "length"
    if (
        "authentication" in low
        or "invalid api key" in low
        or "incorrect api key" in low
        or "permission" in low
        or "401" in low
        or "403" in low
    ):
        return "auth"
    if "429" in low or "rate limit" in low:
        return "rate_limit"
    if "timeout" in low or "timed out" in low or "read timed out" in low:
        return "timeout"
    if "503" in low or "502" in low:
        return "upstream_unavailable"
    return "error"


class RequestPacer:
    def __init__(self, target_rpm: int) -> None:
        rpm = max(0, int(target_rpm))
        self._min_interval = (60.0 / float(rpm)) if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_slot_at = 0.0

    def wait_turn(self) -> None:
        if self._min_interval <= 0.0:
            return
        wait_for = 0.0
        with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_slot_at - now)
            slot_start = now + wait_for
            self._next_slot_at = slot_start + self._min_interval
        if wait_for > 0.0:
            time.sleep(wait_for)


@dataclass
class OpenAIConfig:
    api_key: str
    model: str
    batch_size: int
    max_workers: int
    max_workers_cap: int
    delay: float
    retries: int
    timeout: int
    target_rpm: int
    max_completion_tokens: int
    max_completion_tokens_cap: int
    reasoning_effort: str
    verbosity: str
    length_fallback_policy: str
    length_error_threshold: float
    text_max_chars: int
    contexto_max_chars: int
    bullets_max_items: int
    input_profile: str
    profile_fallback: str


def openai_tema_call_single(
    client: OpenAI,
    model: str,
    row: Dict[str, str],
    *,
    profile: str,
    timeout: int,
    retries: int,
    max_completion_tokens: int,
    max_completion_tokens_cap: int,
    reasoning_effort: str,
    verbosity: str,
    length_fallback_policy: str,
    text_max_chars: int,
    contexto_max_chars: int,
    bullets_max_items: int,
    pacer: Optional[RequestPacer],
    logger: logging.Logger,
) -> Tuple[bool, str, str]:
    prompt = build_openai_tema_prompt(
        row,
        profile=profile,
        text_max_chars=text_max_chars,
        contexto_max_chars=contexto_max_chars,
        bullets_max_items=bullets_max_items,
    )
    last_err = ""
    completion_tokens = max(0, int(max_completion_tokens))
    completion_cap = max(completion_tokens, int(max_completion_tokens_cap))
    effort_current = (reasoning_effort or "minimal").strip().lower()
    verbosity_current = (verbosity or "low").strip().lower()
    if effort_current not in {"minimal", "low", "medium", "high", "xhigh"}:
        effort_current = "minimal"
    if verbosity_current not in {"low", "medium", "high"}:
        verbosity_current = "low"
    fallback_policy = (length_fallback_policy or "auto_downgrade").strip().lower()
    if fallback_policy not in {"auto_downgrade", "keep_deep", "fail_fast"}:
        fallback_policy = "auto_downgrade"

    fatal_error_classes = (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.BadRequestError,
    )
    recoverable_error_classes = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    for attempt in range(1, retries + 1):
        try:
            if pacer is not None:
                pacer.wait_turn()
            req_payload: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Você é um assistente jurídico. Responda somente com JSON válido.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trf1_tema_review",
                        "strict": True,
                        "schema": TEMA_JSON_SCHEMA,
                    },
                },
                "reasoning_effort": effort_current,
                "verbosity": verbosity_current,
                "timeout": timeout,
            }
            if completion_tokens > 0:
                req_payload["max_completion_tokens"] = completion_tokens
            resp = client.chat.completions.create(**req_payload)
            choice = resp.choices[0]
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            raw = parse_openai_message_content(choice.message.content)
            if not raw:
                if finish_reason == "length":
                    raise ValueError("empty_json_due_length")
                refusal = getattr(choice.message, "refusal", None)
                if refusal:
                    raise ValueError(f"refusal: {refusal}")
                raise ValueError("empty_json")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("JSON de resposta não é objeto")
            raw_tema = normalize_ws(str(parsed.get("tema", "") or ""))
            if not raw_tema:
                raise ValueError("empty_tema")
            # Confia no conteúdo retornado pela API; aplica apenas limpeza leve.
            tema = normalize_tema_text(raw_tema)
            if not tema:
                tema = raw_tema
            return True, tema, ""
        except fatal_error_classes as exc:
            last_err = str(exc)
            return False, "", last_err
        except recoverable_error_classes as exc:
            last_err = str(exc)
            if attempt < retries:
                wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.35)
                logger.debug("OpenAI retry %d/%d em %.2fs: %s", attempt, retries, wait, exc)
                time.sleep(wait)
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if "empty_json_due_length" in last_err:
                if completion_tokens > 0:
                    completion_tokens = min(
                        completion_cap,
                        max(completion_tokens + 80, int(completion_tokens * 1.4)),
                    )
                if fallback_policy == "auto_downgrade" and effort_current != "minimal":
                    effort_current = "minimal"
                elif fallback_policy == "fail_fast":
                    return (
                        False,
                        "",
                        (
                            "empty_json_due_length; interrompido por policy fail_fast. "
                            "Sugestão: aumente --openai-max-completion-tokens "
                            "ou use --openai-length-fallback-policy auto_downgrade."
                        ),
                    )
            if attempt < retries:
                wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.35)
                logger.debug("OpenAI retry %d/%d em %.2fs: %s", attempt, retries, wait, exc)
                time.sleep(wait)
    return False, "", last_err


def run_tema_review(
    rows: List[Dict[str, str]],
    *,
    cfg: OpenAIConfig,
    reprocess_mode: str,
    output_csv: Path,
    checkpoint_path: Path,
    source_signature: Dict[str, Any],
    processed_rows: Dict[str, Dict[str, str]],
    fieldnames: Sequence[str],
    logger: logging.Logger,
) -> None:
    if reprocess_mode == "all":
        pending_seed = [row for row in rows if row.get("_row_id", "") not in processed_rows]
    else:
        pending_seed = [row for row in rows if needs_tema_review(row, reprocess_mode)]

    if not pending_seed:
        logger.info("OpenAI tema review: nada pendente.")
        return

    profile_chain = build_profile_chain(cfg.input_profile, cfg.profile_fallback)
    logger.info(
        "OpenAI tema review: perfis de insumo em sequência: %s",
        " -> ".join(profile_chain),
    )
    client = OpenAI(api_key=cfg.api_key, max_retries=0)
    pacer = RequestPacer(cfg.target_rpm)
    workers = max(1, int(cfg.max_workers))
    workers_cap = max(workers, int(cfg.max_workers_cap))
    delay = max(0.0, float(cfg.delay))
    total_seed = len(pending_seed)
    api_calls_done = 0
    global_reasoning_effort = (cfg.reasoning_effort or "minimal").strip().lower()
    stage_rows = list(pending_seed)

    def apply_success(row: Dict[str, str], tema: str, profile: str) -> None:
        row["tema"] = normalize_tema_text(tema)
        rid = row.get("_row_id", "")
        if rid:
            processed_rows[rid] = {
                "tema": row["tema"],
                "updated_at": utc_now_iso(),
                "profile": profile,
            }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    for stage_no, profile in enumerate(profile_chain, start=1):
        if not stage_rows:
            break
        dedupe_map: Dict[str, Dict[str, str]] = {}
        duplicates_by_primary: Dict[str, List[Dict[str, str]]] = {}
        pending_unique: List[Dict[str, str]] = []
        for row in stage_rows:
            key = build_openai_request_key(
                row,
                profile=profile,
                text_max_chars=cfg.text_max_chars,
                contexto_max_chars=cfg.contexto_max_chars,
                bullets_max_items=cfg.bullets_max_items,
            )
            primary = dedupe_map.get(key)
            if primary is None:
                dedupe_map[key] = row
                pending_unique.append(row)
                continue
            primary_id = primary.get("_row_id") or str(id(primary))
            duplicates_by_primary.setdefault(primary_id, []).append(row)

        duplicates_count = len(stage_rows) - len(pending_unique)
        logger.info(
            (
                "OpenAI tema estágio %d/%d (%s): pendentes=%d "
                "(chamadas únicas=%d; duplicadas reaproveitadas=%d)"
            ),
            stage_no,
            len(profile_chain),
            profile,
            len(stage_rows),
            len(pending_unique),
            max(0, duplicates_count),
        )

        stage_success_ids: set[str] = set()
        stage_call_total = len(pending_unique)
        stage_done_calls = 0

        for start in range(0, stage_call_total, cfg.batch_size):
            end = min(start + cfg.batch_size, stage_call_total)
            batch = pending_unique[start:end]
            batch_started = time.perf_counter()
            logger.info(
                "OpenAI tema estágio %s lote %d-%d/%d | workers=%d delay=%.2fs rpm=%d",
                profile,
                start + 1,
                end,
                stage_call_total,
                workers,
                delay,
                cfg.target_rpm,
            )
            futures = {}
            batch_errors = 0
            batch_rate_limited = 0
            batch_timeouts = 0
            batch_length_errors = 0
            batch_auth_errors = 0
            batch_upstream_unavailable = 0

            with ThreadPoolExecutor(max_workers=workers) as ex:
                for row in batch:
                    futures[
                        ex.submit(
                            openai_tema_call_single,
                            client,
                            cfg.model,
                            row,
                            profile=profile,
                            timeout=cfg.timeout,
                            retries=cfg.retries,
                            max_completion_tokens=cfg.max_completion_tokens,
                            max_completion_tokens_cap=cfg.max_completion_tokens_cap,
                            reasoning_effort=global_reasoning_effort,
                            verbosity=cfg.verbosity,
                            length_fallback_policy=cfg.length_fallback_policy,
                            text_max_chars=cfg.text_max_chars,
                            contexto_max_chars=cfg.contexto_max_chars,
                            bullets_max_items=cfg.bullets_max_items,
                            pacer=pacer,
                            logger=logger,
                        )
                    ] = row

                for fut in as_completed(futures):
                    row = futures[fut]
                    ok, tema_out, err = fut.result()
                    stage_done_calls += 1
                    api_calls_done += 1

                    row_id = row.get("_row_id", "")
                    if ok:
                        apply_success(row, tema_out, profile)
                        if row_id:
                            stage_success_ids.add(row_id)
                    else:
                        batch_errors += 1
                        kind = classify_openai_error(err)
                        if kind == "rate_limit":
                            batch_rate_limited += 1
                        elif kind == "timeout":
                            batch_timeouts += 1
                        elif kind == "length":
                            batch_length_errors += 1
                        elif kind == "auth":
                            batch_auth_errors += 1
                        elif kind == "upstream_unavailable":
                            batch_upstream_unavailable += 1
                        logger.debug(
                            "OpenAI tema falha row=%s stage=%s: %s",
                            row.get("_row_id"),
                            profile,
                            err,
                        )

                    primary_id = row_id or str(id(row))
                    for dup in duplicates_by_primary.get(primary_id, []):
                        if ok:
                            apply_success(dup, tema_out, profile)
                            dup_id = dup.get("_row_id", "")
                            if dup_id:
                                stage_success_ids.add(dup_id)

            elapsed = max(0.001, time.perf_counter() - batch_started)
            throughput = len(batch) / elapsed
            length_ratio = batch_length_errors / max(1, len(batch))
            transient_error_ratio = (
                batch_rate_limited + batch_timeouts + batch_upstream_unavailable
            ) / max(1, len(batch))

            if batch_auth_errors > 0:
                raise RuntimeError(
                    "OpenAI retornou erro de autenticação/permissão. Verifique API key e permissões da conta."
                )

            if (
                cfg.length_fallback_policy == "auto_downgrade"
                and global_reasoning_effort != "minimal"
                and length_ratio >= max(0.0, float(cfg.length_error_threshold))
            ):
                global_reasoning_effort = "minimal"
                logger.info("OpenAI tema fallback: reasoning_effort -> minimal por excesso de length")

            if batch_rate_limited > 0 or transient_error_ratio >= 0.25:
                workers = max(1, workers - 1)
                delay = min(2.5, delay * 1.35 + 0.05)
                logger.info("OpenAI tema autoajuste: reduzindo workers=%d delay=%.2f", workers, delay)
            elif batch_errors == 0 and workers < workers_cap:
                workers += 1
                delay = max(0.0, delay * 0.9)
                logger.info("OpenAI tema autoajuste: aumentando workers=%d delay=%.2f", workers, delay)

            stage_remaining_estimate = max(0, len(stage_rows) - len(stage_success_ids))
            write_csv_rows(output_csv, fieldnames, rows)
            snapshot_stats = {
                "total_rows": len(rows),
                "pending_invalid_only": count_invalid_temas(rows),
                "processed_rows_count": len(processed_rows),
                "api_calls_done": api_calls_done,
                "stage_no": stage_no,
                "stage_profile": profile,
                "stage_rows_total": len(stage_rows),
                "stage_rows_remaining_estimate": stage_remaining_estimate,
                "stage_unique_calls_done": stage_done_calls,
                "stage_unique_calls_total": stage_call_total,
                "profile_chain": profile_chain,
                "target_rows_initial": total_seed,
            }
            write_checkpoint(
                checkpoint_path,
                checkpoint_payload(
                    source_signature=source_signature,
                    output_csv=output_csv,
                    reprocess_mode=reprocess_mode,
                    processed_rows=processed_rows,
                    stats=snapshot_stats,
                ),
            )

            logger.info(
                (
                    "OpenAI tema lote concluído (%s): %.2f req/s | erros=%d timeout=%d "
                    "rate_limit=%d upstream=%d length=%d | chamadas_stage=%d/%d"
                ),
                profile,
                throughput,
                batch_errors,
                batch_timeouts,
                batch_rate_limited,
                batch_upstream_unavailable,
                batch_length_errors,
                stage_done_calls,
                stage_call_total,
            )

            if end < stage_call_total and delay > 0:
                time.sleep(delay)

        unresolved = [row for row in stage_rows if row.get("_row_id", "") not in stage_success_ids]
        resolved_stage = len(stage_rows) - len(unresolved)
        logger.info(
            "OpenAI tema estágio %s concluído: resolvidas=%d | remanescentes=%d",
            profile,
            resolved_stage,
            len(unresolved),
        )
        stage_rows = unresolved

    if stage_rows:
        logger.warning(
            (
                "OpenAI tema: %d linhas permaneceram sem revisão válida após %d estágio(s). "
                "Sugestão: usar --openai-input-profile full ou aumentar --openai-text-max-chars."
            ),
            len(stage_rows),
            len(profile_chain),
        )


def default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}{DEFAULT_OUTPUT_SUFFIX}{input_csv.suffix}")


def default_checkpoint_path(input_csv: Path, output_csv: Path) -> Path:
    base = output_csv.parent if output_csv.parent else input_csv.parent
    name = f"{DEFAULT_CHECKPOINT_PREFIX}_{input_csv.stem}.json"
    return base / name


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Revisão da coluna 'tema' em CSV do TRF1 via OpenAI, com checkpoint/backup.",
    )
    p.add_argument("--input-csv", default="", help="CSV de entrada para reprocessar.")
    p.add_argument("--output-csv", default="", help="CSV de saída. Padrão: <input>_tema_revisado.csv")
    p.add_argument("--inplace", action="store_true", help="Sobrescreve o próprio CSV de entrada.")
    p.add_argument("--checkpoint-file", default="", help="Arquivo de checkpoint customizado.")
    p.add_argument("--backup-dir", default="", help="Diretório para backups (opcional).")
    p.add_argument("--no-backup", action="store_true", help="Desativa criação de backup.")
    p.add_argument("--no-gui", action="store_true", help="Desativa seletor GUI de CSV.")

    p.add_argument("--reprocess-mode", choices=("invalid_only", "all"), default="invalid_only")

    p.add_argument("--openai-api-key", default="", help="API key OpenAI.")
    p.add_argument("--openai-model", default="gpt-5-mini", help="Modelo OpenAI.")
    p.add_argument("--openai-batch-size", type=int, default=40)
    p.add_argument("--openai-max-workers", type=int, default=10)
    p.add_argument("--openai-max-workers-cap", type=int, default=14)
    p.add_argument("--openai-delay", type=float, default=0.05)
    p.add_argument("--openai-target-rpm", type=int, default=180)
    p.add_argument(
        "--openai-input-profile",
        default="lean",
        choices=PROFILE_CHOICES,
        help="Perfil base de insumo para gerar tema (lean recomendado para custo/velocidade).",
    )
    p.add_argument(
        "--openai-profile-fallback",
        default="auto",
        choices=PROFILE_FALLBACK_CHOICES,
        help="Fallback automático de perfil entre estágios (auto/off).",
    )
    p.add_argument(
        "--openai-contexto-max-chars",
        type=int,
        default=700,
        help="Máximo de caracteres de contexto no perfil balanced/full.",
    )
    p.add_argument(
        "--openai-bullets-max-items",
        type=int,
        default=3,
        help="Quantidade máxima de bullet_points enviados no perfil (padrão: 3).",
    )
    p.add_argument("--openai-retries", type=int, default=3)
    p.add_argument("--openai-timeout", type=int, default=45)
    p.add_argument("--openai-max-completion-tokens", type=int, default=140)
    p.add_argument("--openai-max-completion-tokens-cap", type=int, default=700)
    p.add_argument(
        "--openai-reasoning-effort",
        default="minimal",
        choices=("minimal", "low", "medium", "high", "xhigh"),
    )
    p.add_argument(
        "--openai-verbosity",
        default="low",
        choices=("low", "medium", "high"),
    )
    p.add_argument(
        "--openai-length-fallback-policy",
        default="auto_downgrade",
        choices=("auto_downgrade", "keep_deep", "fail_fast"),
    )
    p.add_argument(
        "--openai-length-error-threshold",
        type=float,
        default=0.20,
        help="Limiar de erros length por lote para fallback global de reasoning.",
    )
    p.add_argument(
        "--openai-text-max-chars",
        type=int,
        default=2600,
        help="Máximo de caracteres de texto_do_boletim no prompt.",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logger = setup_logger(args.verbose, args.quiet)

    input_csv_str = normalize_ws(args.input_csv)
    if not input_csv_str and not args.no_gui:
        input_csv_str = normalize_ws(select_input_csv_via_gui(logger))
    if not input_csv_str:
        raise SystemExit("Nenhum CSV informado para reprocessar.")

    input_csv = Path(input_csv_str).expanduser().resolve()
    if not input_csv.exists() or not input_csv.is_file():
        raise SystemExit(f"CSV de entrada não encontrado: {input_csv}")

    if args.inplace and normalize_ws(args.output_csv):
        raise SystemExit("Use apenas uma estratégia: --inplace OU --output-csv.")

    if args.inplace:
        output_csv = input_csv
    elif normalize_ws(args.output_csv):
        output_csv = Path(args.output_csv).expanduser().resolve()
    else:
        output_csv = default_output_path(input_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        Path(args.checkpoint_file).expanduser().resolve()
        if normalize_ws(args.checkpoint_file)
        else default_checkpoint_path(input_csv, output_csv)
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    backup_dir: Optional[Path] = None
    if normalize_ws(args.backup_dir):
        backup_dir = Path(args.backup_dir).expanduser().resolve()

    fieldnames, rows = read_csv_rows(input_csv)
    if not rows:
        raise SystemExit("CSV sem linhas úteis para processar.")
    if not fieldnames:
        raise SystemExit("CSV inválido: sem cabeçalho.")
    fieldnames = ensure_tema_column(fieldnames, rows)
    ensure_rows_ready(rows)

    source_signature = compute_dataset_signature(rows, fieldnames, input_csv)
    logger.info("CSV carregado: %d linhas | inválidas em tema (pré): %d", len(rows), count_invalid_temas(rows))

    processed_rows: Dict[str, Dict[str, str]] = {}
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint:
        cp_version = int(checkpoint.get("version", 0) or 0)
        cp_sig = checkpoint.get("source_signature", {})
        if cp_version == CHECKPOINT_VERSION and same_signature(cp_sig, source_signature):
            processed_rows = extract_processed_rows(checkpoint)
            applied = apply_checkpoint_to_rows(rows, processed_rows)
            logger.info("Checkpoint restaurado: temas aplicados=%d | itens checkpoint=%d", applied, len(processed_rows))
        else:
            logger.warning("Checkpoint ignorado por incompatibilidade (versão ou assinatura da base).")

    if args.reprocess_mode == "all":
        pending_estimate = sum(1 for row in rows if row.get("_row_id", "") not in processed_rows)
    else:
        pending_estimate = count_invalid_temas(rows)
    logger.info("Pendentes estimadas para API: %d", pending_estimate)

    api_key = resolve_openai_key(args.openai_api_key)
    if pending_estimate > 0 and not api_key:
        raise SystemExit("API key OpenAI ausente. Informe --openai-api-key ou OPENAI_API_KEY.")

    if not args.no_backup:
        # Backup do destino existente e, em modo inplace, também do arquivo de entrada.
        if output_csv.exists():
            make_backup(output_csv, backup_dir, logger)
        elif args.inplace and input_csv.exists():
            make_backup(input_csv, backup_dir, logger)

    profile_chain = build_profile_chain(args.openai_input_profile, args.openai_profile_fallback)
    logger.info("Perfis planejados para revisão de tema: %s", " -> ".join(profile_chain))

    cfg = OpenAIConfig(
        api_key=api_key,
        model=args.openai_model,
        batch_size=max(1, int(args.openai_batch_size)),
        max_workers=max(1, int(args.openai_max_workers)),
        max_workers_cap=max(1, int(args.openai_max_workers_cap)),
        delay=max(0.0, float(args.openai_delay)),
        retries=max(1, int(args.openai_retries)),
        timeout=max(5, int(args.openai_timeout)),
        target_rpm=max(0, int(args.openai_target_rpm)),
        max_completion_tokens=max(0, int(args.openai_max_completion_tokens)),
        max_completion_tokens_cap=max(0, int(args.openai_max_completion_tokens_cap)),
        reasoning_effort=args.openai_reasoning_effort,
        verbosity=args.openai_verbosity,
        length_fallback_policy=args.openai_length_fallback_policy,
        length_error_threshold=max(0.0, float(args.openai_length_error_threshold)),
        text_max_chars=max(0, int(args.openai_text_max_chars)),
        contexto_max_chars=max(0, int(args.openai_contexto_max_chars)),
        bullets_max_items=max(0, int(args.openai_bullets_max_items)),
        input_profile=args.openai_input_profile,
        profile_fallback=args.openai_profile_fallback,
    )

    # Snapshot inicial de segurança.
    write_csv_rows(output_csv, fieldnames, rows)
    write_checkpoint(
        checkpoint_path,
        checkpoint_payload(
            source_signature=source_signature,
            output_csv=output_csv,
            reprocess_mode=args.reprocess_mode,
            processed_rows=processed_rows,
            stats={
                "total_rows": len(rows),
                "pending_invalid_only": count_invalid_temas(rows),
                "processed_rows_count": len(processed_rows),
                "profile_chain": profile_chain,
                "input_profile": args.openai_input_profile,
                "profile_fallback": args.openai_profile_fallback,
            },
        ),
    )

    run_tema_review(
        rows,
        cfg=cfg,
        reprocess_mode=args.reprocess_mode,
        output_csv=output_csv,
        checkpoint_path=checkpoint_path,
        source_signature=source_signature,
        processed_rows=processed_rows,
        fieldnames=fieldnames,
        logger=logger,
    )

    write_csv_rows(output_csv, fieldnames, rows)
    final_stats = {
        "total_rows": len(rows),
        "pending_invalid_only": count_invalid_temas(rows),
        "processed_rows_count": len(processed_rows),
        "profile_chain": profile_chain,
        "input_profile": args.openai_input_profile,
        "profile_fallback": args.openai_profile_fallback,
    }
    write_checkpoint(
        checkpoint_path,
        checkpoint_payload(
            source_signature=source_signature,
            output_csv=output_csv,
            reprocess_mode=args.reprocess_mode,
            processed_rows=processed_rows,
            stats=final_stats,
        ),
    )

    logger.info(
        "Concluído. saída=%s | checkpoint=%s | inválidas_restantes=%d",
        output_csv,
        checkpoint_path,
        final_stats["pending_invalid_only"],
    )


if __name__ == "__main__":
    main()
