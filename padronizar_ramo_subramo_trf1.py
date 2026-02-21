#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmoniza `ramo_do_direito` e `subramo_do_direito` em CSV do TRF1 com OpenAI.

Objetivos:
- garantir 1 ramo canônico;
- garantir 2 subramos (separados por virgula);
- reduzir typos/aliases e desconexoes entre conteudo juridico e etiquetas;
- executar em alta vazao (workers + batches), com cache e relatorios.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore[assignment]

try:
    from openai import OpenAI
    import openai
except Exception:
    OpenAI = None  # type: ignore[assignment]
    openai = None  # type: ignore[assignment]


RE_WHITESPACE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z0-9]{3,}")
CSV_SPLIT_RE = re.compile(r"[,;\n\r|]+")

DEFAULT_CONTEXT_COLUMNS = (
    "tema",
    "contexto",
    "tese",
    "texto_do_boletim",
)


@dataclass
class CanonicalRules:
    ramo_canonical: List[str]
    ramo_aliases: Dict[str, str]
    ramo_drop: Set[str]
    sub_aliases: Dict[str, str]
    sub_drop: Set[str]
    sub_defaults_by_ramo: Dict[str, List[str]]


@dataclass
class OpenAIConfig:
    enabled: bool
    api_key: str
    model: str
    timeout: int
    retries: int
    max_workers: int
    batch_size: int
    delay: float
    max_sub_candidates: int
    candidate_by_ramo: int
    candidate_global: int
    text_max_chars: int
    fuzzy_threshold_sub: float
    fuzzy_threshold_ramo: float


@dataclass
class RequestItem:
    key: str
    row_index: int
    row_number: int
    ramo_hint: str
    current_ramo: str
    current_subs: List[str]
    sub_candidates: List[str]
    context: Dict[str, str]
    reason: str


@dataclass
class RequestResult:
    ok: bool
    payload: Dict[str, Any]
    error: str
    source: str
    attempts: int


@dataclass
class RowPlan:
    index: int
    row_number: int
    before_ramo: str
    before_sub: str
    base_ramo: str
    base_subs: List[str]
    context: Dict[str, str]
    context_blob: str
    needs_llm: bool = False
    reason: str = ""
    request_key: str = ""
    request_ramo_hint: str = ""
    request_sub_candidates: List[str] = field(default_factory=list)


@dataclass
class SubramoCatalog:
    values: List[str]
    by_key: Dict[str, str]
    keys: List[str]
    global_freq: Counter[str]
    by_ramo_freq: Dict[str, Counter[str]]
    tokens_by_value: Dict[str, Set[str]]
    _map_cache: MutableMapping[str, str] = field(default_factory=dict)

    def has(self, value: str) -> bool:
        return normalize_ws(value) in set(self.values)

    def top_global(self, limit: int) -> List[str]:
        return [x for x, _ in self.global_freq.most_common(max(0, int(limit)))]

    def top_by_ramo(self, ramo: str, limit: int) -> List[str]:
        if not ramo or ramo not in self.by_ramo_freq:
            return []
        return [x for x, _ in self.by_ramo_freq[ramo].most_common(max(0, int(limit)))]

    def map_item(self, value: str, rules: CanonicalRules, fuzzy_threshold: float) -> str:
        norm = normalize_subramo_item(value, rules)
        if not norm:
            return ""
        key = slug_alnum(norm)
        if not key:
            return ""
        if key in self.by_key:
            return self.by_key[key]
        if key in self._map_cache:
            return self._map_cache[key]
        best = ""
        best_score = 0.0
        for cand_key in self.keys:
            score = similarity_key(key, cand_key)
            if score > best_score:
                best_score = score
                best = cand_key
        mapped = self.by_key.get(best, "") if best and best_score >= fuzzy_threshold else ""
        self._map_cache[key] = mapped
        return mapped


def normalize_ws(value: Any) -> str:
    return RE_WHITESPACE.sub(" ", str(value or "")).strip()


def fold_text(value: str) -> str:
    text = normalize_ws(value).casefold()
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def slug_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", fold_text(value))


def similarity_key(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        value = normalize_ws(item)
        if not value:
            continue
        key = fold_text(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def load_text_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def read_csv_rows(path: Path, encoding: str = "utf-8-sig") -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]], encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_rules(config_dir: Path) -> CanonicalRules:
    ramo_path = config_dir / "ramo_canonical.yaml"
    sub_path = config_dir / "subramo_overrides.yaml"

    ramo_cfg: Dict[str, Any] = {}
    if ramo_path.exists():
        obj = load_text_or_json(ramo_path)
        if isinstance(obj, dict):
            ramo_cfg = obj

    sub_cfg: Dict[str, Any] = {}
    if sub_path.exists():
        obj = load_text_or_json(sub_path)
        if isinstance(obj, dict):
            sub_cfg = obj

    ramo_canonical = [normalize_ws(x) for x in ramo_cfg.get("canonical", []) if normalize_ws(x)]
    ramo_aliases = {
        slug_alnum(str(k)): normalize_ws(v)
        for k, v in dict(ramo_cfg.get("aliases", {})).items()
        if slug_alnum(str(k)) and normalize_ws(v)
    }
    ramo_drop = {slug_alnum(str(x)) for x in ramo_cfg.get("drop_exact", []) if slug_alnum(str(x))}

    sub_aliases = {
        slug_alnum(str(k)): normalize_ws(v)
        for k, v in dict(sub_cfg.get("aliases", {})).items()
        if slug_alnum(str(k)) and normalize_ws(v)
    }
    sub_drop = {slug_alnum(str(x)) for x in sub_cfg.get("drop_exact", []) if slug_alnum(str(x))}
    sub_defaults_by_ramo = {
        normalize_ws(k): [normalize_ws(x) for x in v if normalize_ws(x)]
        for k, v in dict(sub_cfg.get("forced_two_by_ramo", {})).items()
        if normalize_ws(k) and isinstance(v, list)
    }

    return CanonicalRules(
        ramo_canonical=ramo_canonical,
        ramo_aliases=ramo_aliases,
        ramo_drop=ramo_drop,
        sub_aliases=sub_aliases,
        sub_drop=sub_drop,
        sub_defaults_by_ramo=sub_defaults_by_ramo,
    )


def normalize_ramo(value: str, rules: CanonicalRules) -> str:
    text = normalize_ws(value)
    if not text:
        return ""
    key = slug_alnum(text)
    if key in rules.ramo_drop:
        return ""
    if text in set(rules.ramo_canonical):
        return text
    if key in rules.ramo_aliases:
        return rules.ramo_aliases[key]
    for cand in rules.ramo_canonical:
        if slug_alnum(cand) == key:
            return cand
    return ""


def normalize_subramo_item(value: str, rules: CanonicalRules) -> str:
    text = normalize_ws(value)
    if not text:
        return ""
    key = slug_alnum(text)
    if key in rules.sub_drop:
        return ""
    if key in rules.sub_aliases:
        return rules.sub_aliases[key]
    return text


def split_subramo_cell(value: str) -> List[str]:
    text = normalize_ws(value)
    if not text:
        return []
    return [normalize_ws(x) for x in CSV_SPLIT_RE.split(text) if normalize_ws(x)]


def join_subramo_items(items: Sequence[str]) -> str:
    return ", ".join(dedupe_keep_order(items))


def choose_display_variant(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return max(counter.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def build_subramo_catalog(rows: Sequence[Mapping[str, str]], ramo_col: str, sub_col: str, rules: CanonicalRules) -> SubramoCatalog:
    variants_by_key: Dict[str, Counter[str]] = defaultdict(Counter)
    global_key_freq: Counter[str] = Counter()
    by_ramo_key_freq: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        ramo_norm = normalize_ramo(str(row.get(ramo_col, "")), rules)
        for raw_sub in split_subramo_cell(str(row.get(sub_col, ""))):
            norm_sub = normalize_subramo_item(raw_sub, rules)
            key = slug_alnum(norm_sub)
            if not key:
                continue
            variants_by_key[key][norm_sub] += 1
            global_key_freq[key] += 1
            if ramo_norm:
                by_ramo_key_freq[ramo_norm][key] += 1

    for alias_target in rules.sub_aliases.values():
        key = slug_alnum(alias_target)
        if key:
            variants_by_key[key][normalize_ws(alias_target)] += 0
    for defaults in rules.sub_defaults_by_ramo.values():
        for item in defaults:
            key = slug_alnum(item)
            if key:
                variants_by_key[key][normalize_ws(item)] += 0

    by_key: Dict[str, str] = {}
    for key, variants in variants_by_key.items():
        best = choose_display_variant(variants)
        if best:
            by_key[key] = best

    global_freq: Counter[str] = Counter()
    for key, count in global_key_freq.items():
        label = by_key.get(key, "")
        if label:
            global_freq[label] += int(count)

    by_ramo_freq: Dict[str, Counter[str]] = defaultdict(Counter)
    for ramo, key_counter in by_ramo_key_freq.items():
        out_counter: Counter[str] = Counter()
        for key, count in key_counter.items():
            label = by_key.get(key, "")
            if label:
                out_counter[label] += int(count)
        if out_counter:
            by_ramo_freq[ramo] = out_counter

    values = [x for x, _ in global_freq.most_common()]
    for value in sorted(set(by_key.values())):
        if value not in values:
            values.append(value)

    tokens_by_value: Dict[str, Set[str]] = {}
    for item in values:
        tokens_by_value[item] = {tok for tok in WORD_RE.findall(fold_text(item)) if len(tok) >= 4}

    return SubramoCatalog(
        values=values,
        by_key=by_key,
        keys=list(by_key.keys()),
        global_freq=global_freq,
        by_ramo_freq=dict(by_ramo_freq),
        tokens_by_value=tokens_by_value,
    )


def normalize_ramo_to_allowed(raw_value: str, rules: CanonicalRules, allowed: Sequence[str], fuzzy_threshold: float) -> str:
    text = normalize_ws(raw_value)
    if not text:
        return ""
    norm = normalize_ramo(text, rules)
    if norm and norm in set(allowed):
        return norm

    key = slug_alnum(text)
    allowed_by_key = {slug_alnum(x): x for x in allowed if slug_alnum(x)}
    if key in allowed_by_key:
        return allowed_by_key[key]

    best = ""
    best_score = 0.0
    for cand in allowed:
        cand_key = slug_alnum(cand)
        score = similarity_key(key, cand_key)
        if score > best_score:
            best = cand
            best_score = score
    return best if best and best_score >= fuzzy_threshold else ""


def infer_ramo_from_text(text: str) -> str:
    t = fold_text(text)
    rules: List[Tuple[str, Sequence[str]]] = [
        ("Direito Tributário", ("tribut", "imposto", "cofins", "pis", "csll", "icms", "ipi", "irpf", "execucao fiscal")),
        ("Direito Previdenciário", ("previd", "inss", "aposent", "pensao", "beneficio", "seguridade", "assistencial")),
        ("Direito Processual Penal", ("habeas corpus", "prisao preventiva", "processo penal", "acao penal", "prova penal")),
        ("Direito Penal", ("crime", "penal", "reu", "denuncia", "pena", "criminal")),
        ("Direito Processual Civil", ("apela", "agravo", "cumprimento de sentenca", "processo civil", "tutela provisoria", "execucao")),
        ("Direito Ambiental", ("ambient", "ibama", "licenciamento", "dano ambiental")),
        ("Direito Agrário", ("agrar", "incra", "reforma agraria", "imovel rural")),
        ("Direito do Trabalho", ("trabalh", "clt", "empregado", "jornada", "verba trabalhista")),
        ("Direito Administrativo", ("servidor", "concurso publico", "licit", "improbidade", "ato administrativo", "administracao publica")),
        ("Direito Civil", ("contrato", "responsabilidade civil", "familia", "sucess", "obrigacao", "dano moral")),
    ]
    for ramo, markers in rules:
        if any(marker in t for marker in markers):
            return ramo
    return ""


def infer_ramo_from_subs(subs: Sequence[str]) -> str:
    text = " ".join(fold_text(x) for x in subs if normalize_ws(x))
    if not text:
        return ""
    if "tribut" in text or "fiscal" in text or "imposto" in text:
        return "Direito Tributário"
    if "previd" in text or "beneficio" in text or "seguridade" in text:
        return "Direito Previdenciário"
    if "penal" in text or "criminal" in text:
        if "processo penal" in text or "processual penal" in text:
            return "Direito Processual Penal"
        return "Direito Penal"
    if "processo civil" in text or "processual civil" in text or "tutela provisoria" in text:
        return "Direito Processual Civil"
    if "administrativo" in text or "servidor publico" in text or "improbidade" in text:
        return "Direito Administrativo"
    if "ambiental" in text:
        return "Direito Ambiental"
    if "agrar" in text:
        return "Direito Agrário"
    if "trabalh" in text:
        return "Direito do Trabalho"
    if "civil" in text or "contrat" in text or "responsabilidade civil" in text:
        return "Direito Civil"
    return ""


def pick_default_subs(ramo: str, catalog: SubramoCatalog, rules: CanonicalRules) -> List[str]:
    defaults: List[str] = []
    for item in rules.sub_defaults_by_ramo.get(ramo, []):
        mapped = catalog.map_item(item, rules, fuzzy_threshold=0.92)
        if mapped:
            defaults.append(mapped)
    return dedupe_keep_order(defaults)


def build_context(row: Mapping[str, str], columns: Sequence[str], text_max_chars: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in columns:
        raw = normalize_ws(str(row.get(col, "")))
        if not raw:
            continue
        if col == "texto_do_boletim" and text_max_chars > 0 and len(raw) > text_max_chars:
            out[col] = raw[:text_max_chars]
        else:
            out[col] = raw
    return out


def build_context_blob(context: Mapping[str, str]) -> str:
    parts: List[str] = []
    for k, v in context.items():
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


def looks_domain_mismatch(ramo: str, sub: str) -> bool:
    r = fold_text(ramo)
    s = fold_text(sub)
    if not r or not s:
        return False
    if "eleitoral" in s and "eleitoral" not in r:
        return True
    if "tribut" in s and "tribut" not in r and "publico" not in r:
        return True
    if ("penal" in s or "criminal" in s) and "penal" not in r and "processual penal" not in r:
        return True
    if "previd" in s and "previd" not in r and "publico" not in r:
        return True
    return False


def should_review_with_llm(
    plan: RowPlan,
    scope: str,
    pair_freq: Counter[Tuple[str, str]],
    global_sub_freq: Counter[str],
    suspicious_pair_threshold: int,
    suspicious_global_min: int,
) -> Tuple[bool, str]:
    problematic = (not plan.base_ramo) or (len(plan.base_subs) != 2)
    if problematic:
        return True, "problematic_structure"
    if scope == "problematic":
        return False, ""
    if scope == "all":
        return True, "scope_all"

    # smart mode
    for sub in plan.base_subs:
        pair = pair_freq.get((plan.base_ramo, sub), 0)
        global_count = global_sub_freq.get(sub, 0)
        if global_count >= suspicious_global_min and pair <= suspicious_pair_threshold:
            return True, "rare_pair"
        if looks_domain_mismatch(plan.base_ramo, sub):
            return True, "domain_mismatch"
    return False, ""


def build_sub_candidates(
    plan: RowPlan,
    ramo_hint: str,
    catalog: SubramoCatalog,
    rules: CanonicalRules,
    cfg: OpenAIConfig,
) -> List[str]:
    pool: List[str] = []
    pool.extend(plan.base_subs)
    pool.extend(pick_default_subs(ramo_hint, catalog, rules))
    pool.extend(catalog.top_by_ramo(ramo_hint, cfg.candidate_by_ramo))
    pool.extend(catalog.top_global(cfg.candidate_global))

    text_tokens = {tok for tok in WORD_RE.findall(fold_text(plan.context_blob)) if len(tok) >= 5}
    if text_tokens:
        lexical_hits: List[Tuple[int, str]] = []
        for cand in catalog.top_global(350):
            cand_tokens = catalog.tokens_by_value.get(cand, set())
            overlap = len(text_tokens & cand_tokens)
            if overlap > 0:
                lexical_hits.append((overlap, cand))
        lexical_hits.sort(key=lambda item: (-item[0], -catalog.global_freq.get(item[1], 0)))
        pool.extend([cand for _, cand in lexical_hits[:140]])

    out = dedupe_keep_order(pool)
    if len(out) > cfg.max_sub_candidates:
        out = out[: cfg.max_sub_candidates]
    return out


def build_request_key(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def parse_chat_content(content_obj: Any) -> str:
    if isinstance(content_obj, str):
        return content_obj.strip()
    if isinstance(content_obj, list):
        chunks: List[str] = []
        for item in content_obj:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    chunks.append(txt)
        return "".join(chunks).strip()
    return ""


def build_llm_messages(item: RequestItem, ramo_options: Sequence[str]) -> Tuple[str, str]:
    ramo_line = "; ".join(ramo_options)
    sub_line = "; ".join(item.sub_candidates)
    context_json = json.dumps(item.context, ensure_ascii=False)
    user = (
        "Tarefa: harmonizar classificacao juridica de jurisprudencia.\n"
        "Regras obrigatorias:\n"
        "1) Escolha exatamente 1 ramo_do_direito da lista permitida.\n"
        "2) Escolha exatamente 2 subramos distintos da lista permitida.\n"
        "3) Os subramos devem ser hierarquicamente coerentes com o ramo e com o conteudo juridico.\n"
        "4) Nao use termos fora da taxonomia fornecida.\n"
        "5) Corrija typos e evite rotulos desconectados do caso.\n\n"
        f"ramo_permitido={ramo_line}\n"
        f"subramo_permitido={sub_line}\n"
        f"ramo_atual={item.current_ramo}\n"
        f"subramo_atual={', '.join(item.current_subs)}\n"
        f"conteudo={context_json}\n"
    )
    system = "Retorne somente JSON estrito no schema solicitado."
    return system, user


def call_openai_json_schema(
    client: Any,
    cfg: OpenAIConfig,
    item: RequestItem,
    ramo_options: Sequence[str],
) -> RequestResult:
    if not cfg.enabled or client is None:
        return RequestResult(ok=False, payload={}, error="openai_disabled", source="disabled", attempts=0)

    schema = {
        "type": "object",
        "properties": {
            "ramo": {"type": "string"},
            "subramos": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "required": ["ramo", "subramos"],
        "additionalProperties": False,
    }

    fatal_error_classes = ()
    recoverable_error_classes = ()
    if openai is not None:
        fatal_error_classes = (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
        )
        recoverable_error_classes = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )

    system, user = build_llm_messages(item, ramo_options)

    attempts = 0
    last_error = ""
    use_reasoning_fields = True
    for attempt in range(1, cfg.retries + 1):
        attempts = attempt
        try:
            req: Dict[str, Any] = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ramo_subramo_choice",
                        "strict": True,
                        "schema": schema,
                    },
                },
                "timeout": cfg.timeout,
            }
            if use_reasoning_fields:
                req["reasoning_effort"] = "minimal"
                req["verbosity"] = "low"

            resp = client.chat.completions.create(**req)
            choice = resp.choices[0]
            content_text = parse_chat_content(choice.message.content)
            if not content_text:
                raise ValueError("empty_json_response")
            payload = json.loads(content_text)
            if not isinstance(payload, dict):
                raise ValueError("json_not_object")
            return RequestResult(ok=True, payload=payload, error="", source="api", attempts=attempts)
        except fatal_error_classes as exc:  # type: ignore[arg-type]
            return RequestResult(ok=False, payload={}, error=str(exc), source="fatal", attempts=attempts)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            low = message.casefold()
            last_error = message

            if "reasoning_effort" in low or "verbosity" in low or "unsupported parameter" in low:
                if use_reasoning_fields:
                    use_reasoning_fields = False
                    continue
                return RequestResult(ok=False, payload={}, error=message, source="bad_request", attempts=attempts)

            is_recoverable = any(marker in low for marker in ("rate limit", "timeout", "temporar", "connection", "server error"))
            if recoverable_error_classes and isinstance(exc, recoverable_error_classes):  # type: ignore[arg-type]
                is_recoverable = True

            if attempt < cfg.retries and is_recoverable:
                wait = (2 ** (attempt - 1)) + random.uniform(0.05, 0.35)
                time.sleep(wait)
                continue
            return RequestResult(ok=False, payload={}, error=message, source="recoverable" if is_recoverable else "error", attempts=attempts)

    return RequestResult(ok=False, payload={}, error=last_error or "unknown_error", source="error", attempts=attempts)


def classify_error_kind(error_text: str) -> str:
    low = normalize_ws(error_text).casefold()
    if not low:
        return "unknown"
    if "rate limit" in low:
        return "rate_limit"
    if "auth" in low or "permission" in low or "api key" in low:
        return "auth"
    if "timeout" in low:
        return "timeout"
    if "connection" in low:
        return "connection"
    if "json" in low:
        return "json"
    if "bad request" in low or "unsupported parameter" in low:
        return "bad_request"
    return "other"


def resolve_openai_key(cli_value: str, key_file: str) -> str:
    if normalize_ws(cli_value):
        return normalize_ws(cli_value)
    env = normalize_ws(os.getenv("OPENAI_API_KEY", ""))
    if env:
        return env
    path = Path(normalize_ws(key_file))
    if path.exists():
        text = normalize_ws(path.read_text(encoding="utf-8", errors="ignore"))
        if text:
            return text
    return ""


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    items = payload.get("items", {})
    if isinstance(items, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for key, value in items.items():
            if isinstance(key, str) and isinstance(value, dict):
                out[key] = dict(value)
        return out
    return {}


def save_cache(path: Path, items: Mapping[str, Mapping[str, Any]], model: str) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(),
        "model": model,
        "items": items,
    }
    write_json_atomic(path, payload)


def determine_output_path(input_path: Path, output_arg: str, inplace: bool) -> Path:
    if inplace:
        return input_path
    if normalize_ws(output_arg):
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_ramo_subramo_harmonizado{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harmoniza ramo_do_direito e subramo_do_direito via OpenAI (alta vazao)."
    )
    parser.add_argument("--input", default="boletins_de_jurisprudencia_TRF1_with_news.csv")
    parser.add_argument("--output", default="")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ramo-column", default="ramo_do_direito")
    parser.add_argument("--subramo-column", default="subramo_do_direito")
    parser.add_argument("--config-dir", default="canon_config")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--report-prefix", default="ramo_subramo_harmonizacao")
    parser.add_argument(
        "--context-columns",
        default=",".join(DEFAULT_CONTEXT_COLUMNS),
        help="Colunas usadas no contexto da classificacao (separadas por virgula).",
    )
    parser.add_argument(
        "--scope",
        choices=("smart", "problematic", "all"),
        default="all",
        help="all=todas linhas; smart=problemas + sinais de desconexao; problematic=apenas problemas estruturais.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Limita linhas revisadas pelo LLM (0 = sem limite).")
    parser.add_argument("--suspicious-pair-threshold", type=int, default=1)
    parser.add_argument("--suspicious-global-min", type=int, default=12)

    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-key-file", default="CHAVE_SECRETA_API_Mauricio_local.txt")
    parser.add_argument("--openai-model", default="gpt-5-mini")
    parser.add_argument("--openai-timeout", type=int, default=50)
    parser.add_argument("--openai-retries", type=int, default=3)
    parser.add_argument("--openai-max-workers", type=int, default=24)
    parser.add_argument("--openai-batch-size", type=int, default=120)
    parser.add_argument("--openai-delay", type=float, default=0.00)
    parser.add_argument("--openai-max-sub-candidates", type=int, default=70)
    parser.add_argument("--openai-candidate-by-ramo", type=int, default=45)
    parser.add_argument("--openai-candidate-global", type=int, default=25)
    parser.add_argument("--openai-text-max-chars", type=int, default=1000)
    parser.add_argument("--openai-fuzzy-threshold-sub", type=float, default=0.90)
    parser.add_argument("--openai-fuzzy-threshold-ramo", type=float, default=0.84)
    parser.add_argument("--disable-openai", action="store_true")
    parser.add_argument("--adaptive-throttle", action="store_true")

    parser.add_argument("--cache-file", default=".checkpoint/ramo_subramo_openai_cache.json")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--cache-save-every-batches", type=int, default=1)
    return parser.parse_args()


def build_row_plans(
    rows: Sequence[Mapping[str, str]],
    ramo_col: str,
    sub_col: str,
    context_columns: Sequence[str],
    rules: CanonicalRules,
    catalog: SubramoCatalog,
    cfg: OpenAIConfig,
) -> List[RowPlan]:
    plans: List[RowPlan] = []
    for idx, row in enumerate(rows):
        before_ramo = normalize_ws(str(row.get(ramo_col, "")))
        before_sub = normalize_ws(str(row.get(sub_col, "")))

        ramo_norm = normalize_ramo(before_ramo, rules)
        if not ramo_norm:
            ramo_norm = before_ramo if before_ramo in set(rules.ramo_canonical) else ""

        subs_norm: List[str] = []
        for item in split_subramo_cell(before_sub):
            mapped = catalog.map_item(item, rules, cfg.fuzzy_threshold_sub)
            if mapped:
                subs_norm.append(mapped)
        subs_norm = dedupe_keep_order(subs_norm)

        context = build_context(row, context_columns, cfg.text_max_chars)
        context_blob = build_context_blob(context)
        plans.append(
            RowPlan(
                index=idx,
                row_number=idx + 2,
                before_ramo=before_ramo,
                before_sub=before_sub,
                base_ramo=ramo_norm,
                base_subs=subs_norm,
                context=context,
                context_blob=context_blob,
            )
        )
    return plans


def resolve_final_labels(
    plan: RowPlan,
    result: RequestResult | None,
    rules: CanonicalRules,
    ramo_options: Sequence[str],
    dominant_ramo: str,
    catalog: SubramoCatalog,
    cfg: OpenAIConfig,
) -> Tuple[str, List[str], str, str]:
    ramo = ""
    pred_subs: List[str] = []
    llm_error = ""
    method = "baseline"

    if result is not None and result.ok:
        ramo = normalize_ramo_to_allowed(str(result.payload.get("ramo", "")), rules, ramo_options, cfg.fuzzy_threshold_ramo)
        raw_subs = result.payload.get("subramos", [])
        if isinstance(raw_subs, list):
            for item in raw_subs:
                mapped = catalog.map_item(str(item), rules, cfg.fuzzy_threshold_sub)
                if mapped:
                    pred_subs.append(mapped)
        method = result.source
    elif result is not None:
        llm_error = result.error
        method = "fallback"

    if not ramo:
        ramo = plan.base_ramo
    if not ramo:
        ramo = infer_ramo_from_subs(pred_subs or plan.base_subs)
    if not ramo:
        ramo = infer_ramo_from_text(plan.context_blob)
    ramo = normalize_ramo_to_allowed(ramo, rules, ramo_options, cfg.fuzzy_threshold_ramo)
    if not ramo:
        ramo = dominant_ramo if dominant_ramo in set(ramo_options) else (ramo_options[0] if ramo_options else "")

    final_subs: List[str] = []
    final_subs.extend(pred_subs)
    final_subs.extend(plan.base_subs)
    final_subs.extend(pick_default_subs(ramo, catalog, rules))
    final_subs.extend(catalog.top_by_ramo(ramo, 25))
    final_subs.extend(catalog.top_global(40))
    final_subs = dedupe_keep_order(final_subs)[:2]

    if len(final_subs) < 2:
        for item in dedupe_keep_order(catalog.values):
            if item not in final_subs:
                final_subs.append(item)
            if len(final_subs) >= 2:
                break

    return ramo, final_subs[:2], method, llm_error


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = determine_output_path(input_path, args.output, args.inplace)
    config_dir = Path(args.config_dir)
    cache_path = Path(args.cache_file)

    fieldnames, rows = read_csv_rows(input_path)
    if args.ramo_column not in fieldnames:
        raise KeyError(f'Coluna "{args.ramo_column}" nao encontrada.')
    if args.subramo_column not in fieldnames:
        raise KeyError(f'Coluna "{args.subramo_column}" nao encontrada.')

    rules = load_rules(config_dir)
    if not rules.ramo_canonical:
        raise RuntimeError("Nao foi possivel carregar ramos canonicos em canon_config/ramo_canonical.yaml")

    openai_key = ""
    if not args.disable_openai:
        openai_key = resolve_openai_key(args.openai_api_key, args.openai_key_file)

    cfg = OpenAIConfig(
        enabled=(not args.disable_openai) and bool(openai_key) and OpenAI is not None and openai is not None,
        api_key=openai_key,
        model=normalize_ws(args.openai_model) or "gpt-5-mini",
        timeout=max(5, int(args.openai_timeout)),
        retries=max(1, int(args.openai_retries)),
        max_workers=max(1, int(args.openai_max_workers)),
        batch_size=max(1, int(args.openai_batch_size)),
        delay=max(0.0, float(args.openai_delay)),
        max_sub_candidates=max(20, int(args.openai_max_sub_candidates)),
        candidate_by_ramo=max(5, int(args.openai_candidate_by_ramo)),
        candidate_global=max(5, int(args.openai_candidate_global)),
        text_max_chars=max(300, int(args.openai_text_max_chars)),
        fuzzy_threshold_sub=max(0.70, min(0.99, float(args.openai_fuzzy_threshold_sub))),
        fuzzy_threshold_ramo=max(0.70, min(0.99, float(args.openai_fuzzy_threshold_ramo))),
    )

    if (not args.disable_openai) and (not cfg.enabled):
        print("Aviso: OpenAI desabilitado por ausencia de chave ou dependencia; usando fallback deterministico.")

    context_columns = [normalize_ws(x) for x in args.context_columns.split(",") if normalize_ws(x) and normalize_ws(x) in fieldnames]
    if not context_columns:
        context_columns = [col for col in DEFAULT_CONTEXT_COLUMNS if col in fieldnames]

    catalog = build_subramo_catalog(rows, args.ramo_column, args.subramo_column, rules)
    if not catalog.values:
        raise RuntimeError("Catalogo de subramos vazio; revise dados de entrada.")

    plans = build_row_plans(
        rows=rows,
        ramo_col=args.ramo_column,
        sub_col=args.subramo_column,
        context_columns=context_columns,
        rules=rules,
        catalog=catalog,
        cfg=cfg,
    )

    pair_freq: Counter[Tuple[str, str]] = Counter()
    ramo_freq: Counter[str] = Counter()
    for plan in plans:
        if plan.base_ramo:
            ramo_freq[plan.base_ramo] += 1
        if plan.base_ramo and plan.base_subs:
            for sub in plan.base_subs:
                pair_freq[(plan.base_ramo, sub)] += 1

    dominant_ramo = ramo_freq.most_common(1)[0][0] if ramo_freq else rules.ramo_canonical[0]
    ramo_options = list(rules.ramo_canonical)

    targets: List[int] = []
    for idx, plan in enumerate(plans):
        needs_llm, reason = should_review_with_llm(
            plan=plan,
            scope=args.scope,
            pair_freq=pair_freq,
            global_sub_freq=catalog.global_freq,
            suspicious_pair_threshold=max(0, int(args.suspicious_pair_threshold)),
            suspicious_global_min=max(1, int(args.suspicious_global_min)),
        )
        plans[idx].needs_llm = needs_llm
        plans[idx].reason = reason
        if needs_llm:
            targets.append(idx)

    if args.max_rows > 0:
        targets = targets[: args.max_rows]
        targets_set = set(targets)
        for idx in range(len(plans)):
            if idx not in targets_set and plans[idx].needs_llm:
                plans[idx].needs_llm = False
                plans[idx].reason = "out_of_max_rows"

    request_by_key: Dict[str, RequestItem] = {}
    groups_by_key: Dict[str, List[int]] = defaultdict(list)
    for idx in targets:
        plan = plans[idx]
        ramo_hint = plan.base_ramo or infer_ramo_from_subs(plan.base_subs) or infer_ramo_from_text(plan.context_blob) or dominant_ramo
        ramo_hint = normalize_ramo_to_allowed(ramo_hint, rules, ramo_options, cfg.fuzzy_threshold_ramo) or dominant_ramo
        sub_candidates = build_sub_candidates(plan, ramo_hint, catalog, rules, cfg)

        req_payload = {
            "ramo_hint": ramo_hint,
            "current_ramo": plan.base_ramo,
            "current_subs": plan.base_subs,
            "context": plan.context,
            "sub_candidates": sub_candidates,
            "model": cfg.model,
            "scope_reason": plan.reason,
        }
        key = build_request_key(req_payload)
        plans[idx].request_key = key
        plans[idx].request_ramo_hint = ramo_hint
        plans[idx].request_sub_candidates = sub_candidates

        groups_by_key[key].append(idx)
        if key not in request_by_key:
            request_by_key[key] = RequestItem(
                key=key,
                row_index=idx,
                row_number=plan.row_number,
                ramo_hint=ramo_hint,
                current_ramo=plan.base_ramo,
                current_subs=plan.base_subs,
                sub_candidates=sub_candidates,
                context=plan.context,
                reason=plan.reason,
            )

    cache_items: Dict[str, Dict[str, Any]] = {}
    if not args.disable_cache:
        cache_items = load_cache(cache_path)

    resolved_requests: Dict[str, RequestResult] = {}
    cache_hits = 0
    for key, req in request_by_key.items():
        c = cache_items.get(key)
        if not isinstance(c, dict):
            continue
        ramo = normalize_ws(str(c.get("ramo", "")))
        subramos = c.get("subramos", [])
        if ramo and isinstance(subramos, list) and len(subramos) >= 2:
            resolved_requests[key] = RequestResult(
                ok=True,
                payload={"ramo": ramo, "subramos": subramos[:2]},
                error="",
                source="cache",
                attempts=0,
            )
            cache_hits += 1

    pending_keys = [k for k in request_by_key if k not in resolved_requests]
    total_unique_requests = len(request_by_key)

    client = OpenAI(api_key=cfg.api_key, max_retries=0) if cfg.enabled and OpenAI is not None else None
    api_calls = 0
    api_failures = 0
    batch_counter = 0
    error_kind_counter: Counter[str] = Counter()

    current_workers = cfg.max_workers
    current_delay = cfg.delay
    t0_api = time.perf_counter()

    for start in range(0, len(pending_keys), cfg.batch_size):
        batch_counter += 1
        batch_keys = pending_keys[start : start + cfg.batch_size]
        reqs = [request_by_key[k] for k in batch_keys]
        t_batch = time.perf_counter()

        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            futures = {executor.submit(call_openai_json_schema, client, cfg, req, ramo_options): req for req in reqs}
            for future in as_completed(futures):
                req = futures[future]
                result = future.result()
                resolved_requests[req.key] = result
                if result.source == "api":
                    api_calls += 1
                if not result.ok and result.source != "disabled":
                    api_failures += 1
                    error_kind_counter[classify_error_kind(result.error)] += 1

                if not args.disable_cache and result.ok:
                    cache_items[req.key] = {
                        "ramo": normalize_ws(str(result.payload.get("ramo", ""))),
                        "subramos": list(result.payload.get("subramos", []))[:2],
                        "updated_at": datetime.now().isoformat(),
                        "model": cfg.model,
                    }

        elapsed_batch = max(1e-6, time.perf_counter() - t_batch)
        elapsed_total = max(1e-6, time.perf_counter() - t0_api)
        done_pending = min(start + len(batch_keys), len(pending_keys))
        batch_rate = len(batch_keys) / elapsed_batch
        total_rate = done_pending / elapsed_total if pending_keys else 0.0
        print(
            f"OpenAI lote {start + 1}-{start + len(batch_keys)}/{len(pending_keys)} "
            f"| workers={current_workers} | req/s lote={batch_rate:.2f} | req/s total={total_rate:.2f}"
        )

        if (not args.disable_cache) and (batch_counter % max(1, int(args.cache_save_every_batches)) == 0):
            save_cache(cache_path, cache_items, cfg.model)

        if args.adaptive_throttle and len(batch_keys) > 0:
            rate_limit_errors = error_kind_counter.get("rate_limit", 0)
            timeout_errors = error_kind_counter.get("timeout", 0)
            if (rate_limit_errors + timeout_errors) >= max(2, len(batch_keys) // 4):
                current_workers = max(4, int(current_workers * 0.80))
                current_delay = min(1.50, current_delay + 0.05)
            elif (rate_limit_errors + timeout_errors) == 0 and current_workers < cfg.max_workers:
                current_workers = min(cfg.max_workers, current_workers + 1)
                current_delay = max(0.0, current_delay - 0.01)

        if current_delay > 0:
            time.sleep(current_delay)

    if not args.disable_cache:
        save_cache(cache_path, cache_items, cfg.model)

    changed_rows_report: List[Dict[str, Any]] = []
    reviewed_rows_report: List[Dict[str, Any]] = []
    unresolved_requests_report: List[Dict[str, Any]] = []

    for key in pending_keys:
        res = resolved_requests.get(key)
        if res is None:
            continue
        if not res.ok:
            unresolved_requests_report.append(
                {
                    "request_key": key,
                    "error_kind": classify_error_kind(res.error),
                    "error": res.error,
                    "rows": len(groups_by_key.get(key, [])),
                    "row_numbers": ", ".join(str(plans[i].row_number) for i in groups_by_key.get(key, [])),
                }
            )

    rows_changed = 0
    rows_reviewed = 0
    rows_reviewed_changed = 0
    rows_fallback = 0
    rows_cache = 0

    for idx, row in enumerate(rows):
        plan = plans[idx]
        result = resolved_requests.get(plan.request_key) if plan.needs_llm else None

        ramo_final, subs_final, method, llm_error = resolve_final_labels(
            plan=plan,
            result=result,
            rules=rules,
            ramo_options=ramo_options,
            dominant_ramo=dominant_ramo,
            catalog=catalog,
            cfg=cfg,
        )
        sub_final_text = join_subramo_items(subs_final)

        before_ramo = plan.before_ramo
        before_sub = plan.before_sub

        row[args.ramo_column] = ramo_final
        row[args.subramo_column] = sub_final_text

        changed = (normalize_ws(before_ramo) != normalize_ws(ramo_final)) or (
            normalize_ws(before_sub) != normalize_ws(sub_final_text)
        )

        if changed:
            rows_changed += 1
            changed_rows_report.append(
                {
                    "row_number": plan.row_number,
                    "numero_processo": normalize_ws(str(row.get("numero_processo", ""))),
                    "tema": normalize_ws(str(row.get("tema", "")))[:180],
                    "before_ramo": before_ramo,
                    "after_ramo": ramo_final,
                    "before_subramo": before_sub,
                    "after_subramo": sub_final_text,
                    "method": method,
                    "reason": plan.reason,
                    "llm_error": llm_error,
                }
            )

        if plan.needs_llm:
            rows_reviewed += 1
            if changed:
                rows_reviewed_changed += 1
            if method == "cache":
                rows_cache += 1
            if method == "fallback":
                rows_fallback += 1
            reviewed_rows_report.append(
                {
                    "row_number": plan.row_number,
                    "numero_processo": normalize_ws(str(row.get("numero_processo", ""))),
                    "tema": normalize_ws(str(row.get("tema", "")))[:180],
                    "before_ramo": before_ramo,
                    "after_ramo": ramo_final,
                    "before_subramo": before_sub,
                    "after_subramo": sub_final_text,
                    "changed": "1" if changed else "0",
                    "method": method,
                    "reason": plan.reason,
                    "llm_error": llm_error,
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_base = f"{normalize_ws(args.report_prefix) or 'ramo_subramo_harmonizacao'}_{timestamp}"
    report_applied = report_dir / f"{report_base}_aplicadas.csv"
    report_reviewed = report_dir / f"{report_base}_revisadas.csv"
    report_unresolved = report_dir / f"{report_base}_erros_requisicao.csv"
    report_summary = report_dir / f"{report_base}_summary.json"

    write_csv_rows(
        report_applied,
        fieldnames=(
            "row_number",
            "numero_processo",
            "tema",
            "before_ramo",
            "after_ramo",
            "before_subramo",
            "after_subramo",
            "method",
            "reason",
            "llm_error",
        ),
        rows=changed_rows_report,
    )
    write_csv_rows(
        report_reviewed,
        fieldnames=(
            "row_number",
            "numero_processo",
            "tema",
            "before_ramo",
            "after_ramo",
            "before_subramo",
            "after_subramo",
            "changed",
            "method",
            "reason",
            "llm_error",
        ),
        rows=reviewed_rows_report,
    )
    write_csv_rows(
        report_unresolved,
        fieldnames=("request_key", "error_kind", "error", "rows", "row_numbers"),
        rows=unresolved_requests_report,
    )

    if not args.dry_run:
        if args.inplace:
            backup_path = input_path.with_name(f"{input_path.name}.backup_ramo_subramo_{timestamp}")
            shutil.copy2(input_path, backup_path)
            print(f"Backup criado: {backup_path}")
        write_csv_rows(output_path, fieldnames, rows)

    summary = {
        "timestamp": timestamp,
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "output_written": not bool(args.dry_run),
        "rows_total": len(rows),
        "rows_changed": rows_changed,
        "scope": args.scope,
        "rows_reviewed_by_llm_or_cache": rows_reviewed,
        "rows_reviewed_changed": rows_reviewed_changed,
        "rows_fallback": rows_fallback,
        "rows_cache": rows_cache,
        "unique_requests_total": total_unique_requests,
        "unique_requests_pending_api": len(pending_keys),
        "unique_requests_cache_hit": cache_hits,
        "api_calls": api_calls,
        "api_failures": api_failures,
        "error_kind_counter": dict(error_kind_counter),
        "openai_enabled": cfg.enabled,
        "openai_model": cfg.model,
        "openai_max_workers": cfg.max_workers,
        "openai_batch_size": cfg.batch_size,
        "openai_delay": cfg.delay,
        "context_columns": context_columns,
        "reports": {
            "aplicadas": str(report_applied),
            "revisadas": str(report_reviewed),
            "erros_requisicao": str(report_unresolved),
            "summary": str(report_summary),
        },
        "cache_file": str(cache_path),
        "cache_enabled": not bool(args.disable_cache),
    }
    report_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("Resumo da harmonizacao ramo/subramo:")
    print(f"- Linhas totais: {summary['rows_total']}")
    print(f"- Linhas alteradas: {summary['rows_changed']}")
    print(f"- Linhas revisadas por LLM/cache: {summary['rows_reviewed_by_llm_or_cache']}")
    print(f"- Requests unicos (total): {summary['unique_requests_total']}")
    print(f"- Requests cache hit: {summary['unique_requests_cache_hit']}")
    print(f"- Chamadas API realizadas: {summary['api_calls']}")
    print(f"- Falhas API: {summary['api_failures']}")
    print(f"- Modo de escopo: {summary['scope']}")
    print(f"- OpenAI ativo: {'sim' if summary['openai_enabled'] else 'nao'}")
    print(f"- CSV de saida gravado: {'sim' if summary['output_written'] else 'nao (dry-run)'}")
    print(f"- Relatorio aplicadas: {report_applied}")
    print(f"- Relatorio revisadas: {report_reviewed}")
    print(f"- Relatorio erros requisicao: {report_unresolved}")
    print(f"- Resumo JSON: {report_summary}")


if __name__ == "__main__":
    main()
