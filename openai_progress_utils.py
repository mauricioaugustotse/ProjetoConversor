"""
Funções auxiliares para checkpoint, backup e escrita atômica de JSON/CSV.

Fluxo de trabalho:
1. Gera timestamps UTC e assinaturas de arquivo para rastreamento.
2. Compara assinaturas para validar retomada de processamento.
3. Lê payloads JSON com fallback seguro para dicionário vazio.
4. Escreve JSON e CSV de forma atômica via arquivo temporário.
5. Cria backups versionados com carimbo de data/hora.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_file_signature(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def same_file_signature(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return (
        str(a.get("path", "")) == str(b.get("path", ""))
        and int(a.get("size", -1) or -1) == int(b.get("size", -2) or -2)
        and int(a.get("mtime_ns", -1) or -1) == int(b.get("mtime_ns", -2) or -2)
    )


def read_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_json_atomic(path: Path, payload: Mapping[str, Any], *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if pretty:
        body = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def make_backup(path: Path, backup_dir: Optional[Path] = None, *, label: str = "backup") -> Optional[Path]:
    if not path.exists():
        return None
    target_dir = (backup_dir or path.parent).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = target_dir / f"{path.name}.{label}_{stamp}"
    shutil.copy2(path, dst)
    return dst
