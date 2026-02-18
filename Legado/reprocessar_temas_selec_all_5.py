#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrige `texto_original` dos casos novos e reprocessa colunas analíticas via API.

Saída principal:
- temas_selec_TSE_all_5.csv

Auditorias:
- <prefix>_reprocessados.csv
- <prefix>_texto_original_nao_recuperado.csv
- <prefix>_api_falhas.csv
- <prefix>_resumo.txt
"""

from __future__ import annotations

import argparse
import bisect
import csv
import html
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI


TARGET_CONTEXT = "Incluído automaticamente por auditoria de omissões."
SJUR_PREFIX = "sjur-servicos.tse.jus.br/sjur-servicos/rest/download/pdf/"
ANALYTIC_COLS = (
    "texto_original",
    "tema",
    "subramo",
    "contexto",
    "tese",
    "bullet_points",
    "punchline",
)

ANCHOR_RE = re.compile(
    r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>.*?</a>",
    re.IGNORECASE | re.DOTALL,
)
P_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
CANONICAL_RE = re.compile(
    r"<link\s+rel=[\"']canonical[\"']\s+href=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
CITATION_LIKE_RE = re.compile(
    r"^\s*\((?:Ac\.|Res\.|EDcl|Embargos?|AgR(?:-[A-Za-z]+)?|MS|REspE?|REspe|REsp|RO-?El|RMS|RCEd|AE|TutCautAnt|MC|AAg|Rec\.?)",
    re.IGNORECASE,
)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_nul(value: str) -> str:
    return (value or "").replace("\x00", "")


def strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_token(text: str) -> str:
    s = strip_accents((text or "").lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return normalize_ws(s)


def clean_html_text(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(TAG_RE.sub(" ", text)).replace("\xa0", " ")
    t = clean_nul(t)
    return normalize_ws(t)


def load_prompt_from_temas_script(script_path: Path) -> str:
    if not script_path.exists():
        raise FileNotFoundError(f"Script base não encontrado: {script_path}")
    txt = script_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(
        r"PROMPT_EXTRACAO\s*=\s*([\"']{3})([\s\S]*?)\1",
        txt,
        re.MULTILINE,
    )
    if not m:
        raise RuntimeError("Não foi possível extrair PROMPT_EXTRACAO de TEMAS_SELC_txt_to_csv.py")
    return m.group(2)


def normalize_subramo_value(value: Any, max_items: int = 3) -> str:
    flat: List[str] = []

    def _walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            s = clean_nul(normalize_ws(v))
            if not s:
                return
            parts = [normalize_ws(p) for p in re.split(r"[,;\n\r|]+", s) if normalize_ws(p)]
            flat.extend(parts)
            return
        if isinstance(v, list):
            for x in v:
                _walk(x)
            return
        _walk(str(v))

    _walk(value)
    out: List[str] = []
    seen: Set[str] = set()
    for item in flat:
        key = normalize_token(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_items:
            break
    return ", ".join(out)


def normalize_bullet_points(value: Any) -> str:
    if value is None:
        return ""
    lines: List[str] = []
    if isinstance(value, list):
        for x in value:
            s = normalize_ws(clean_nul(str(x)))
            if s:
                lines.append(s)
    else:
        raw = clean_nul(str(value))
        for ln in raw.splitlines():
            s = normalize_ws(ln)
            if s:
                lines.append(s)
    if not lines:
        return ""
    out: List[str] = []
    for ln in lines:
        if ln.startswith("•"):
            out.append(ln)
        else:
            out.append(f"• {ln.lstrip('- ').strip()}")
    return "\n".join(out)


def normalize_api_field(col: str, value: Any) -> str:
    if col == "subramo":
        return normalize_subramo_value(value, max_items=3)
    if col == "bullet_points":
        return normalize_bullet_points(value)
    if value is None:
        return ""
    return normalize_ws(clean_nul(str(value)))


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
    if not isinstance(data, dict):
        return {}

    def normalize_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k).strip().lower(): normalize_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize_keys(x) for x in obj]
        return obj

    return normalize_keys(data)


def score_text_candidate(text: str, source_priority: int) -> Tuple[int, int, int, int]:
    t = normalize_ws(text)
    quote_bonus = 1 if any(ch in t for ch in ("“", "”", '"', "‘", "’")) else 0
    length_bonus = min(len(t), 2000)
    citation_penalty = 1 if CITATION_LIKE_RE.search(t) else 0
    return (
        source_priority,
        quote_bonus - citation_penalty,
        0 if citation_penalty else 1,
        length_bonus,
    )


def is_housekeeping_text(text: str) -> bool:
    t = normalize_ws(text)
    if not t:
        return True
    if re.match(r"^(Atualizado em\b|NE\s*:|Vide\b|Nota explicativa\b)", t, re.IGNORECASE):
        return True
    return False


def map_href_to_preceding_text(html_text: str, source_label: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    p_matches = list(P_RE.finditer(html_text))
    p_end_positions = [m.end() for m in p_matches]

    paragraphs = [clean_html_text(m.group(1)) for m in p_matches]
    for a in ANCHOR_RE.finditer(html_text):
        href = normalize_ws(html.unescape(a.group(1)))
        if SJUR_PREFIX not in href:
            continue

        idx = bisect.bisect_left(p_end_positions, a.start()) - 1
        if idx < 0:
            continue

        candidates: List[str] = []
        j = idx
        while j >= 0 and len(candidates) < 8:
            ptxt = paragraphs[j]
            if ptxt:
                candidates.append(ptxt)
            j -= 1

        if not candidates:
            continue

        # Regra principal: usar o parágrafo imediatamente anterior (texto realmente "acima" do link).
        # Fallback: se o mais próximo for metadado/nota, avança para o próximo parágrafo substantivo.
        best = candidates[0]
        if is_housekeeping_text(best) or CITATION_LIKE_RE.search(best):
            for cand in candidates[1:]:
                if not is_housekeeping_text(cand) and not CITATION_LIKE_RE.search(cand):
                    best = cand
                    break
        out.setdefault(href, []).append(best)
    return out


def build_href_text_map(html_paths: Sequence[Path], use_live_fallback: bool) -> Tuple[Dict[str, str], Dict[str, Any]]:
    # Política: para cada href, prioriza primeira ocorrência local; se não houver, primeira live.
    # Dentro de cada ocorrência, o melhor parágrafo precedente é escolhido por score determinístico.
    href_best: Dict[str, str] = {}
    href_source: Dict[str, str] = {}
    canonical_urls: List[str] = []
    href_candidates_total = 0

    for path in html_paths:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        m = CANONICAL_RE.search(txt)
        if m:
            canonical_urls.append(normalize_ws(m.group(1)))
        partial = map_href_to_preceding_text(txt, source_label="local")
        for href, texts in partial.items():
            href_candidates_total += len(texts)
            if href in href_best:
                continue
            if texts:
                href_best[href] = texts[0]
                href_source[href] = "local"

    live_ok = 0
    live_fail = 0
    if use_live_fallback:
        for url in sorted(set(canonical_urls)):
            if not url:
                continue
            try:
                with urllib.request.urlopen(url, timeout=45) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                partial = map_href_to_preceding_text(raw, source_label="live")
                for href, texts in partial.items():
                    href_candidates_total += len(texts)
                    if href in href_best:
                        continue
                    if texts:
                        href_best[href] = texts[0]
                        href_source[href] = "live"
                live_ok += 1
            except Exception:
                live_fail += 1

    stats = {
        "href_candidates_total": href_candidates_total,
        "href_best_total": len(href_best),
        "live_pages_ok": live_ok,
        "live_pages_fail": live_fail,
        "canonical_urls_total": len(set(canonical_urls)),
        "href_source": href_source,
    }
    return href_best, stats


def row_is_target(row: Dict[str, str]) -> bool:
    ctx = normalize_ws(clean_nul(row.get("contexto", "")))
    return ctx == TARGET_CONTEXT


def parse_api_payload(data: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in ("tema", "subramo", "contexto", "tese", "bullet_points", "punchline"):
        if col in data:
            out[col] = normalize_api_field(col, data.get(col))
    return out


def call_api_single(
    api_key: str,
    model: str,
    prompt_base: str,
    texto_julgado: str,
    retries: int,
    retry_wait_s: float,
) -> Tuple[bool, Dict[str, str], str]:
    messages = [
        {
            "role": "system",
            "content": "Você é um assistente de jurimetria especializado em jurisprudência eleitoral brasileira.",
        },
        {
            "role": "user",
            "content": prompt_base.format(texto_julgado=texto_julgado),
        },
    ]
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = parse_json_response(content)
            fields = parse_api_payload(parsed)
            required = ("tema", "contexto", "tese", "bullet_points", "punchline")
            if not all(normalize_ws(fields.get(k, "")) for k in required):
                last_error = "Resposta incompleta da API (campos analíticos obrigatórios vazios)."
            else:
                return True, fields, ""
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if attempt < retries:
            time.sleep(retry_wait_s * attempt)
    return False, {}, last_error or "Falha desconhecida na chamada da API."


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_dict_csv(path: Path, rows: Sequence[Dict[str, Any]], ordered_cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ordered_cols))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ordered_cols})


def ensure_fields(rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    for row in rows:
        for f in fieldnames:
            if f not in row:
                row[f] = ""


def build_audit_prefix(user_prefix: str) -> str:
    if user_prefix:
        return user_prefix
    stamp = datetime.now().strftime("%Y%m%d")
    return f"auditoria_temas_all_5_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrige texto_original e reprocessa analíticas para gerar temas_selec_TSE_all_5.csv."
    )
    parser.add_argument("--csv-in", default="temas_selec_TSE_all_4.csv")
    parser.add_argument("--csv-out", default="temas_selec_TSE_all_5.csv")
    parser.add_argument("--html-glob", default="* — Temas Selecionados.html")
    parser.add_argument("--audit-prefix", default="")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait-s", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-target-rows", type=int, default=0, help="0 = sem limite")
    parser.add_argument(
        "--extra-numero-processo",
        default="",
        help="Lista de número_processo extra (separados por vírgula) para forçar reprocessamento.",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--temas-script", default="TEMAS_SELC_txt_to_csv.py")
    parser.add_argument(
        "--use-live-fallback",
        action="store_true",
        default=True,
        help="Usa páginas online canônicas como fallback para recuperar texto_original.",
    )
    parser.add_argument(
        "--no-use-live-fallback",
        dest="use_live_fallback",
        action="store_false",
        help="Desativa fallback online.",
    )
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrada.")

    csv_in = Path(args.csv_in)
    csv_out = Path(args.csv_out)
    if not csv_in.exists():
        raise FileNotFoundError(f"CSV de entrada não encontrado: {csv_in}")

    html_paths = sorted(Path(".").glob(args.html_glob))
    if not html_paths:
        raise RuntimeError(f"Nenhum HTML encontrado com glob: {args.html_glob}")

    audit_prefix = build_audit_prefix(args.audit_prefix)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(f"{csv_out}.reprocess.checkpoint.json")
    prompt_base = load_prompt_from_temas_script(Path(args.temas_script))

    resume_mode = checkpoint_path.exists() and csv_out.exists()
    load_path = csv_out if resume_mode else csv_in
    with load_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise RuntimeError("CSV de entrada vazio.")
    ensure_fields(rows, fieldnames)

    extra_proc = {
        normalize_ws(x)
        for x in (args.extra_numero_processo or "").split(",")
        if normalize_ws(x)
    }
    target_indices = [
        i
        for i, row in enumerate(rows)
        if row_is_target(row) or normalize_ws(row.get("numero_processo", "")) in extra_proc
    ]
    target_expected = 932
    if args.max_target_rows > 0:
        target_indices = target_indices[: args.max_target_rows]

    href_text_map, href_stats = build_href_text_map(
        html_paths=html_paths,
        use_live_fallback=bool(args.use_live_fallback),
    )

    text_unrecovered: List[Dict[str, Any]] = []
    text_recovered_count = 0
    text_updated_indices: Set[int] = set()
    for idx in target_indices:
        row = rows[idx]
        links = [normalize_ws(row.get(c, "")) for c in ("link_1", "link_2", "link_3")]
        links = [l for l in links if l]

        chosen_href = ""
        chosen_text = ""
        for href in links:
            cand = href_text_map.get(href, "")
            if cand:
                chosen_href = href
                chosen_text = cand
                break

        if chosen_text:
            new_texto = clean_nul(chosen_text)
            if new_texto != row.get("texto_original", ""):
                row["texto_original"] = new_texto
                text_updated_indices.add(idx)
            text_recovered_count += 1
        else:
            text_unrecovered.append(
                {
                    "row_idx": str(idx),
                    "ID": row.get("ID", ""),
                    "numero_processo": row.get("numero_processo", ""),
                    "ramo": row.get("ramo", ""),
                    "link_1": row.get("link_1", ""),
                    "link_2": row.get("link_2", ""),
                    "link_3": row.get("link_3", ""),
                    "texto_original_atual": row.get("texto_original", "")[:800],
                    "motivo": "Sem parágrafo precedente recuperável por href.",
                }
            )

    write_csv(csv_out, rows, fieldnames)

    done_indices: Set[int] = set()
    api_failures: List[Dict[str, Any]] = []
    reprocessados: List[Dict[str, Any]] = []

    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            done_indices = set(int(x) for x in ckpt.get("done_indices", []))
            saved_fail = ckpt.get("api_failures", [])
            if isinstance(saved_fail, list):
                api_failures = saved_fail
            saved_rep = ckpt.get("reprocessados", [])
            if isinstance(saved_rep, list):
                reprocessados = saved_rep
            saved_target = ckpt.get("target_indices", [])
            if isinstance(saved_target, list) and saved_target:
                target_indices = [int(x) for x in saved_target]
            print(f"[resume] checkpoint carregado: done={len(done_indices)}")
        except Exception:
            print("[resume] checkpoint inválido, ignorando e recomeçando API.")

    remaining = [i for i in target_indices if i not in done_indices]
    total_target = len(target_indices)
    print(f"[info] linhas alvo: {total_target} (esperado=932)")
    print(f"[info] texto_original recuperado por HTML: {text_recovered_count}/{total_target}")
    print(f"[info] pendentes para API: {len(remaining)}")

    lock = threading.Lock()
    processed_since_save = 0

    def _save_progress() -> None:
        write_csv(csv_out, rows, fieldnames)
        payload = {
            "done_indices": sorted(done_indices),
            "api_failures": api_failures,
            "reprocessados": reprocessados,
            "target_indices": target_indices,
        }
        checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _worker(row_idx: int) -> Dict[str, Any]:
        row = rows[row_idx]
        texto = row.get("texto_original", "") or ""
        ok, fields, err = call_api_single(
            api_key=api_key,
            model=args.model,
            prompt_base=prompt_base,
            texto_julgado=texto,
            retries=max(1, int(args.retries)),
            retry_wait_s=max(0.5, float(args.retry_wait_s)),
        )
        return {
            "row_idx": row_idx,
            "ok": ok,
            "fields": fields,
            "error": err,
        }

    if remaining:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            future_map = {ex.submit(_worker, i): i for i in remaining}
            for fut in as_completed(future_map):
                result = fut.result()
                idx = int(result["row_idx"])
                row = rows[idx]
                ok = bool(result["ok"])

                with lock:
                    if ok:
                        fields = result.get("fields", {}) or {}
                        # texto_original foi corrigido a partir da fonte HTML e é preservado aqui.
                        row["texto_original"] = clean_nul(row.get("texto_original", ""))
                        for col in ("tema", "subramo", "contexto", "tese", "bullet_points", "punchline"):
                            row[col] = clean_nul(normalize_api_field(col, fields.get(col, "")))

                        reprocessados.append(
                            {
                                "row_idx": str(idx),
                                "ID": row.get("ID", ""),
                                "numero_processo": row.get("numero_processo", ""),
                                "ramo": row.get("ramo", ""),
                                "link_1": row.get("link_1", ""),
                                "status_api": "ok",
                                "texto_original_recuperado_html": "sim" if idx in text_updated_indices else "nao",
                                "tema_len": str(len(row.get("tema", ""))),
                                "contexto_len": str(len(row.get("contexto", ""))),
                                "tese_len": str(len(row.get("tese", ""))),
                                "bullet_points_len": str(len(row.get("bullet_points", ""))),
                                "punchline_len": str(len(row.get("punchline", ""))),
                            }
                        )
                    else:
                        api_failures.append(
                            {
                                "row_idx": str(idx),
                                "ID": row.get("ID", ""),
                                "numero_processo": row.get("numero_processo", ""),
                                "ramo": row.get("ramo", ""),
                                "link_1": row.get("link_1", ""),
                                "erro": result.get("error", "Erro desconhecido"),
                            }
                        )
                    done_indices.add(idx)
                    processed_since_save += 1

                    if processed_since_save >= max(1, int(args.save_every)):
                        _save_progress()
                        processed_since_save = 0
                        print(f"[progress] API concluído: {len(done_indices)}/{total_target}")

    if processed_since_save > 0:
        _save_progress()

    write_csv(csv_out, rows, fieldnames)

    reprocessados_path = Path(f"{audit_prefix}_reprocessados.csv")
    unrecovered_path = Path(f"{audit_prefix}_texto_original_nao_recuperado.csv")
    falhas_path = Path(f"{audit_prefix}_api_falhas.csv")
    resumo_path = Path(f"{audit_prefix}_resumo.txt")

    write_dict_csv(
        reprocessados_path,
        reprocessados,
        [
            "row_idx",
            "ID",
            "numero_processo",
            "ramo",
            "link_1",
            "status_api",
            "texto_original_recuperado_html",
            "tema_len",
            "contexto_len",
            "tese_len",
            "bullet_points_len",
            "punchline_len",
        ],
    )
    write_dict_csv(
        unrecovered_path,
        text_unrecovered,
        [
            "row_idx",
            "ID",
            "numero_processo",
            "ramo",
            "link_1",
            "link_2",
            "link_3",
            "motivo",
            "texto_original_atual",
        ],
    )
    write_dict_csv(
        falhas_path,
        api_failures,
        ["row_idx", "ID", "numero_processo", "ramo", "link_1", "erro"],
    )

    output_count = len(rows)
    with csv_in.open("r", encoding="utf-8-sig", newline="") as f:
        input_count = sum(1 for _ in csv.DictReader(f))

    context_auto_after = sum(1 for i in target_indices if normalize_ws(rows[i].get("contexto", "")) == TARGET_CONTEXT)

    resumo_lines = [
        f"CSV entrada: {csv_in}",
        f"CSV saída: {csv_out}",
        f"Linhas input: {input_count}",
        f"Linhas output: {output_count}",
        f"Linhas alvo detectadas: {total_target}",
        f"Linhas alvo esperadas (referência): {target_expected}",
        f"Texto_original recuperado por HTML: {text_recovered_count}",
        f"Texto_original não recuperado: {len(text_unrecovered)}",
        f"API sucesso: {len(reprocessados)}",
        f"API falhas: {len(api_failures)}",
        f"contexto automático remanescente nas linhas alvo: {context_auto_after}",
        f"HTML local lidos: {len(html_paths)}",
        f"Href com texto recuperável (global): {href_stats.get('href_best_total', 0)}",
        f"Páginas live OK: {href_stats.get('live_pages_ok', 0)}",
        f"Páginas live falha: {href_stats.get('live_pages_fail', 0)}",
        f"Auditoria reprocessados: {reprocessados_path}",
        f"Auditoria texto não recuperado: {unrecovered_path}",
        f"Auditoria API falhas: {falhas_path}",
    ]
    resumo_path.write_text("\n".join(resumo_lines) + "\n", encoding="utf-8")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\nConcluído.")
    print(f"Saída: {csv_out}")
    print(f"Auditoria: {resumo_path}")
    print(f"API sucesso={len(reprocessados)} | falhas={len(api_failures)}")


if __name__ == "__main__":
    main()
