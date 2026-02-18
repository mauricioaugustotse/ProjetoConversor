#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preenche omissões de um CSV usando API em lotes (batch), sem tocar em colunas de link.

Objetivo:
- Reprocessar apenas células vazias.
- Preservar valores já preenchidos.
- Chamar gpt-5-mini (com fallback opcional).
- Permitir retomada por checkpoint.

Exemplo:
  python preencher_omissoes_batch_api.py \
      --csv-in temas_selec_TSE_all_3.csv \
      --csv-out temas_selec_TSE_all_3_preenchido.csv \
      --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI


MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a"}
DEFAULT_EXCLUDED = {"ID", "link_1", "link_2", "link_3", "link_de_acesso"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in MISSING_TOKENS


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {}
    json_str = ""
    if "```json" in raw_text:
        try:
            json_str = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
        except IndexError:
            json_str = ""
    if not json_str:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            json_str = raw_text[start : end + 1]
    if not json_str:
        return {}
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_date_to_mdy(text: str) -> str:
    raw = normalize_ws(text)
    if not raw:
        return ""

    s = raw.replace(".", "/").replace("-", "/")
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Heurística:
        # - se primeira parte > 12, tratamos como D/M.
        # - caso contrário, tratamos como M/D.
        if a > 12 and b <= 12:
            d, mm = a, b
        else:
            mm, d = a, b
        try:
            dt = datetime(y, mm, d)
            return f"{dt.month}/{dt.day}/{dt.year}"
        except ValueError:
            return raw

    m2 = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s)
    if m2:
        y, mm, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        try:
            dt = datetime(y, mm, d)
            return f"{dt.month}/{dt.day}/{dt.year}"
        except ValueError:
            return raw

    return raw


def normalize_subramo(value: Any, max_items: int = 3) -> str:
    if isinstance(value, list):
        parts = [normalize_ws(str(x)) for x in value if normalize_ws(str(x))]
    else:
        text = normalize_ws(str(value or ""))
        if not text:
            return ""
        parts = [normalize_ws(p) for p in re.split(r"[,;\n\r|]+", text) if normalize_ws(p)]

    seen = set()
    final: List[str] = []
    for p in parts:
        key = re.sub(r"\W+", " ", p.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        final.append(p)
        if len(final) >= max_items:
            break
    return ", ".join(final)


def normalize_bullets(value: Any) -> str:
    if isinstance(value, list):
        lines = [normalize_ws(str(x)) for x in value if normalize_ws(str(x))]
    else:
        raw = str(value or "")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    if not lines:
        return ""

    out: List[str] = []
    for ln in lines:
        ln = normalize_ws(ln.lstrip("•- ").strip())
        if not ln:
            continue
        out.append(f"• {ln}")
    return "\n".join(out)


def normalize_field(col: str, value: Any) -> str:
    if value is None:
        return ""
    if col == "subramo":
        return normalize_subramo(value, max_items=3)
    if col == "bullet_points":
        return normalize_bullets(value)
    text = normalize_ws(str(value))
    if col == "data_julgamento":
        return normalize_date_to_mdy(text)
    return text


def build_messages(batch_items: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = (
        "Você é um assistente de jurimetria especializado em jurisprudência eleitoral brasileira. "
        "Retorne apenas JSON válido."
    )
    user = (
        "Preencha apenas campos faltantes de cada linha.\n"
        "NÃO altere valores já existentes.\n"
        "NÃO inclua colunas de link.\n"
        "Faça melhor esforço para completar todos os campos faltantes.\n"
        "Se o texto estiver limitado à citação final, use metadados disponíveis (ramo, subramo, processo, relator, data)\n"
        "para gerar resumo conservador sem inventar fatos específicos do mérito.\n"
        "Só retorne string vazia quando for realmente impossível inferir algo mínimo útil.\n\n"
        "Regras de formato:\n"
        "1) data_julgamento: M/D/AAAA.\n"
        "2) subramo: string com até 3 itens separados por vírgula.\n"
        "3) bullet_points: 3 a 5 linhas, cada linha começando com '• '.\n"
        "4) relator: manter prefixo 'Min.' quando aplicável.\n\n"
        "Saída obrigatória (JSON):\n"
        "{\n"
        '  "updates": [\n'
        "    {\n"
        '      "row_index": 123,\n'
        '      "fields": {\n'
        '        "campo_faltante": "valor"\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Entrada do lote (JSON):\n"
        f"{json.dumps(batch_items, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_batch(
    client: OpenAI,
    batch_items: Sequence[Dict[str, Any]],
    model: str,
    fallback_model: str,
    retries: int,
    retry_wait_s: float,
) -> Tuple[Dict[str, Any], str]:
    messages = build_messages(batch_items)
    models_to_try = [model]
    if fallback_model and fallback_model != model:
        models_to_try.append(fallback_model)

    last_error = None
    for chosen_model in models_to_try:
        for attempt in range(1, retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                content = (resp.choices[0].message.content or "").strip()
                parsed = parse_json_response(content)
                if isinstance(parsed.get("updates"), list):
                    return parsed, chosen_model
                last_error = RuntimeError("Resposta sem lista 'updates'.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                msg = str(exc).lower()
                model_unavailable = (
                    "model_not_found" in msg
                    or "does not exist" in msg
                    or "do not have access" in msg
                )
                if model_unavailable:
                    # Não insiste no mesmo modelo quando ele claramente não está disponível.
                    break
            if attempt < retries:
                time.sleep(retry_wait_s * attempt)
    if last_error:
        raise RuntimeError(f"Falha no lote após retries. Último erro: {last_error}")
    raise RuntimeError("Falha no lote sem detalhes.")


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pending(
    rows: Sequence[Dict[str, str]],
    target_cols: Sequence[str],
) -> List[Dict[str, Any]]:
    pending: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        missing_cols = [c for c in target_cols if c in row and is_missing(row.get(c))]
        if not missing_cols:
            continue
        if is_missing(row.get("texto_original")):
            continue
        pending.append({"row_index": i, "missing_cols": missing_cols})
    return pending


def summarize_missing(rows: Sequence[Dict[str, str]], target_cols: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in target_cols:
        out[c] = sum(1 for r in rows if is_missing(r.get(c)))
    return out


def build_batch_items(
    rows: Sequence[Dict[str, str]],
    pending_slice: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in pending_slice:
        idx = int(p["row_index"])
        row = rows[idx]
        missing_cols = list(p["missing_cols"])
        context = {
            "ID": row.get("ID", ""),
            "numero_no_ramo": row.get("numero_no_ramo", ""),
            "ramo": row.get("ramo", ""),
            "dados_existentes": {
                k: row.get(k, "")
                for k in (
                    "tema",
                    "punchline",
                    "tese",
                    "bullet_points",
                    "data_julgamento",
                    "numero_processo",
                    "relator",
                    "subramo",
                    "contexto",
                    "tipo_de_processo",
                )
                if not is_missing(row.get(k))
            },
            "campos_faltantes": missing_cols,
            "texto_original": row.get("texto_original", ""),
        }
        items.append({"row_index": idx, **context})
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preenche omissões do CSV em lote via OpenAI (gpt-5-mini)."
    )
    parser.add_argument("--csv-in", default="temas_selec_TSE_all_3.csv")
    parser.add_argument("--csv-out", default="temas_selec_TSE_all_4.csv")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--pause-s", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait-s", type=float, default=2.0)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fallback-model", default="")
    parser.add_argument(
        "--exclude-cols",
        default="ID,link_1,link_2,link_3,link_de_acesso",
        help="Colunas ignoradas no preenchimento (separadas por vírgula).",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Arquivo de checkpoint. Padrão: <csv-out>.omissoes.checkpoint.json",
    )
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrada no ambiente/.env.")

    args = parse_args()
    input_path = Path(args.csv_in)
    output_path = Path(args.csv_out)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(f"{args.csv_out}.omissoes.checkpoint.json")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"CSV de entrada não encontrado: {input_path}")

    if checkpoint_path.exists() and output_path.exists():
        rows, fieldnames = read_csv(output_path)
        with checkpoint_path.open("r", encoding="utf-8") as f:
            ckpt = json.load(f)
        pending = ckpt.get("pending", [])
        cursor = int(ckpt.get("cursor", 0))
        excluded_cols = set(ckpt.get("excluded_cols", []))
        print(f"[resume] checkpoint encontrado: cursor={cursor}, pendentes={len(pending)}")
    else:
        rows, fieldnames = read_csv(input_path)
        excluded_cols = set(
            normalize_ws(c) for c in args.exclude_cols.split(",") if normalize_ws(c)
        )
        excluded_cols = excluded_cols | DEFAULT_EXCLUDED

        target_cols = [c for c in fieldnames if c not in excluded_cols]
        pending = build_pending(rows, target_cols)
        if args.max_rows and args.max_rows > 0:
            pending = pending[: args.max_rows]
        cursor = 0
        ckpt = {
            "created_from": str(input_path),
            "excluded_cols": sorted(excluded_cols),
            "target_cols": target_cols,
            "pending": pending,
            "cursor": cursor,
        }
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2)
        write_csv(output_path, rows, fieldnames)
        print(f"[start] pendentes identificados: {len(pending)}")

    if not pending:
        print("Nenhuma omissão encontrada para processamento.")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return

    # Reconstroi target_cols do checkpoint para manter consistência em retomada.
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            ckpt_data = json.load(f)
        target_cols = list(ckpt_data.get("target_cols", []))
    else:
        target_cols = [c for c in fieldnames if c not in excluded_cols]

    client = OpenAI(api_key=api_key)
    total = len(pending)
    applied_updates = 0
    batches_done = 0
    active_model = args.model
    active_fallback = args.fallback_model

    while cursor < total:
        end = min(cursor + max(1, args.batch_size), total)
        batch_pending = pending[cursor:end]
        batch_items = build_batch_items(rows, batch_pending)

        print(f"[batch] {cursor + 1}-{end}/{total} (size={len(batch_items)})")

        try:
            response, used_model = call_batch(
                client=client,
                batch_items=batch_items,
                model=active_model,
                fallback_model=active_fallback,
                retries=max(1, args.retries),
                retry_wait_s=max(0.5, args.retry_wait_s),
            )
            if used_model != active_model:
                print(f"[model] fallback ativado: {active_model} -> {used_model}")
                active_model = used_model
                active_fallback = ""
        except Exception as exc:  # noqa: BLE001
            print(f"[erro] lote falhou: {exc}")
            # Salva estado e encerra para retomada manual.
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "excluded_cols": sorted(excluded_cols),
                        "target_cols": target_cols,
                        "pending": pending,
                        "cursor": cursor,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            write_csv(output_path, rows, fieldnames)
            raise

        updates = response.get("updates", [])
        missing_map = {int(p["row_index"]): set(p["missing_cols"]) for p in batch_pending}

        for upd in updates:
            try:
                idx = int(upd.get("row_index"))
            except Exception:  # noqa: BLE001
                continue
            fields = upd.get("fields")
            if idx not in missing_map or not isinstance(fields, dict):
                continue
            for col, val in fields.items():
                if col not in missing_map[idx]:
                    continue
                if col not in rows[idx]:
                    continue
                if not is_missing(rows[idx].get(col)):
                    continue
                normalized_val = normalize_field(col, val)
                if is_missing(normalized_val):
                    continue
                rows[idx][col] = normalized_val
                applied_updates += 1

        cursor = end
        batches_done += 1

        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "excluded_cols": sorted(excluded_cols),
                    "target_cols": target_cols,
                    "pending": pending,
                    "cursor": cursor,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        write_csv(output_path, rows, fieldnames)

        if args.pause_s > 0:
            time.sleep(args.pause_s)

    final_missing = summarize_missing(rows, target_cols)
    write_csv(output_path, rows, fieldnames)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\nConcluído.")
    print(f"Arquivo de saída: {output_path}")
    print(f"Lotes processados: {batches_done}")
    print(f"Células preenchidas: {applied_updates}")
    print("Omissões restantes por coluna:")
    for col in target_cols:
        print(f"  - {col}: {final_missing.get(col, 0)}")


if __name__ == "__main__":
    main()
