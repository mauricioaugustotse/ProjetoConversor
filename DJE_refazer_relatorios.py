#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renomeia, regenera e audita os relatorios DJe/TSE existentes no Notion.

Fluxo:
1. Enumera as paginas filhas da pagina-mae de relatorios.
2. Classifica cada titulo (semanal "DD/MM/YYYY a DD/MM/YYYY" ou mensal "NOV/2025").
3. Semanais antigas (seg-sex) sao renomeadas para a semana civil completa (seg-dom),
   preservando a URL da pagina; colisoes de titulo arquivam a pagina redundante.
4. Regenera cada pagina via NOTION_relatoriodeIA_v2.py (--force-clear --no-resume).
5. Audita omissoes: contagem na base vs analisados vs publicados (tabela + inventario).
6. Salva auditoria em Artefatos/reports/refazer/ (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import DJE_relatorios_semanais_gui as gui
import NOTION_DJE_csv_to_notion as importer
import NOTION_relatoriodeIA_v2 as report
from Artefatos.scripts.openai_progress_utils import utc_now_iso, write_json_atomic

PROJECT_ROOT = Path(__file__).resolve().parent
REFAZER_DIR = PROJECT_ROOT / "Artefatos" / "reports" / "refazer"
AUDIT_JSON = REFAZER_DIR / "refazer_relatorios_audit.json"
AUDIT_MD = REFAZER_DIR / "refazer_relatorios_audit.md"

WEEKLY_TITLE_RE = re.compile(
    r"^DJe do TSE\s*-\s*(\d{1,2}/\d{1,2}/\d{2,4})\s+a\s+(\d{1,2}/\d{1,2}/\d{2,4})\s*$"
)
MONTHLY_TITLE_RE = re.compile(r"^DJe do TSE\s*-\s*([A-ZÇ]{3})/(\d{4})\s*$")
MONTHS_PT = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
}


def _parse_br_date(raw: str) -> Optional[date]:
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", raw.strip())
    if not match:
        return None
    day, month, year = (int(part) for part in match.groups())
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


@dataclass
class ReportPagePlan:
    page_id: str
    current_title: str
    kind: str  # weekly | monthly | unknown
    start: Optional[date] = None
    end: Optional[date] = None
    target_title: str = ""
    actions: List[str] = field(default_factory=list)
    status: str = "pending"
    audit: Dict[str, Any] = field(default_factory=dict)


def list_report_pages_ordered(parent_page_id: str) -> List[tuple[str, str]]:
    pages: List[tuple[str, str]] = []
    cursor = ""
    while True:
        params: Dict[str, Any] = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        payload = report.notion_request("GET", f"/v1/blocks/{parent_page_id}/children", params=params)
        for block in payload.get("results", []) or []:
            if not isinstance(block, dict) or block.get("type") != "child_page":
                continue
            title = gui._normalize_ws((block.get("child_page") or {}).get("title"))
            block_id = report._normalize_notion_id(str(block.get("id", "")))
            if title and block_id:
                pages.append((title, block_id))
        if not payload.get("has_more"):
            break
        cursor = gui._normalize_ws(payload.get("next_cursor"))
        if not cursor:
            break
    return pages


def classify_page(title: str, page_id: str) -> ReportPagePlan:
    weekly = WEEKLY_TITLE_RE.match(title)
    if weekly:
        start = _parse_br_date(weekly.group(1))
        end = _parse_br_date(weekly.group(2))
        if start and end and start <= end:
            period = gui.week_for_decision_day(start)
            return ReportPagePlan(
                page_id=page_id,
                current_title=title,
                kind="weekly",
                start=period.start,
                end=period.end,
                target_title=period.title,
            )
    monthly = MONTHLY_TITLE_RE.match(title)
    if monthly:
        month = MONTHS_PT.get(monthly.group(1).upper())
        year = int(monthly.group(2))
        if month:
            last_day = calendar.monthrange(year, month)[1]
            return ReportPagePlan(
                page_id=page_id,
                current_title=title,
                kind="monthly",
                start=date(year, month, 1),
                end=date(year, month, last_day),
                target_title=title.strip(),
            )
    return ReportPagePlan(page_id=page_id, current_title=title, kind="unknown")


def rename_page(page_id: str, new_title: str) -> None:
    report.notion_request(
        "PATCH",
        f"/v1/pages/{page_id}",
        json_body={"properties": {"title": {"title": [{"type": "text", "text": {"content": new_title}}]}}},
    )


def archive_page(page_id: str) -> None:
    report.notion_request("PATCH", f"/v1/pages/{page_id}", json_body={"archived": True})


def _mark_weekly_manifest(plan: ReportPagePlan, *, case_count: int) -> None:
    # Mantem o manifesto da GUI em dia para o staleness nao regenerar a mesma
    # semana de novo sem necessidade.
    if plan.kind != "weekly" or not plan.start or not plan.end:
        return
    try:
        period = gui.WeeklyPeriod(start=plan.start, end=plan.end)
        gui.mark_report_generated(period, case_count=case_count)
    except Exception:
        pass


def regenerate_page(plan: ReportPagePlan, *, database_url: str, model: str, log: Any) -> None:
    page_url = report._notion_page_url_from_id(plan.page_id)
    cmd = [
        sys.executable,
        "NOTION_relatoriodeIA_v2.py",
        "--no-gui",
        "--page-url",
        page_url,
        "--source-database-url",
        database_url,
        "--start-date",
        plan.start.isoformat(),
        "--end-date",
        plan.end.isoformat(),
        "--force-clear",
        "--no-resume",
        "--model",
        model,
        "--max-openai-cases",
        "1000",
        "--openai-triage-threshold",
        "0",
        "--enrich-news-gemini",
        "--verbose",
    ]
    gui.run_command(cmd, log=log)


def _safe_filename(title: str) -> str:
    return re.sub(r"[^\w\-]+", "_", title).strip("_")[:120] or "relatorio"


def snapshot_report_json(plan: ReportPagePlan) -> Dict[str, Any]:
    source = Path(report.REPORT_FILE)
    if not source.exists():
        return {}
    target = REFAZER_DIR / f"{_safe_filename(plan.target_title or plan.current_title)}.report.json"
    shutil.copyfile(source, target)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def audit_page(plan: ReportPagePlan, payload: Dict[str, Any], *, data_source_id: str) -> Dict[str, Any]:
    # Mesma query do gerador (todas as paginas do periodo), e nao a indexada por
    # chave numeroUnico/numeroProcesso, que subconta paginas sem essas propriedades.
    base_cases = report.query_cases_by_period(
        data_source_id, plan.start.isoformat(), plan.end.isoformat()
    )
    base_ids = {
        report._normalize_notion_id(str(case.page_id))
        for case in base_cases
        if gui._normalize_ws(case.page_id)
    }
    analyzed_ids = {
        report._normalize_notion_id(str(item.get("page_id") or item.get("case_id") or ""))
        for item in (payload.get("analyses") or [])
        if isinstance(item, dict)
    }
    analyzed_ids.discard("")
    publish_stats = payload.get("publish_stats") or {}
    published = int(publish_stats.get("case_rows_created", 0) or 0)
    inventory = int(publish_stats.get("inventory_rows_created", 0) or 0)
    missing = sorted(base_ids - analyzed_ids)
    # Varredura do alvo: nenhum caso com alto cargo da Republica pode ficar
    # abaixo do piso 8/alto (e, portanto, fora dos destaques).
    target_total = 0
    target_below_floor: List[str] = []
    for item in payload.get("analyses") or []:
        if not isinstance(item, dict):
            continue
        signal = " | ".join(
            str(item.get(key, "") or "")
            for key in ("title", "what_happened", "why_relevant", "legal_grounds", "strategic_comment")
        )
        signal += " | " + " | ".join(str(x) for x in (item.get("public_figures") or []))
        if not report.FEDERAL_HIGH_OFFICE_RE.search(signal):
            continue
        target_total += 1
        score = int(item.get("relevance_score", 0) or 0)
        risk = str(item.get("risk_level", "") or "").casefold()
        if score < 8 or risk not in {"alto", "critico"}:
            target_below_floor.append(str(item.get("case_id") or item.get("page_id") or "?"))
    audit = {
        "base_count": len(base_ids),
        "report_case_count": int(payload.get("case_count", 0) or 0),
        "analyzed_count": len(analyzed_ids),
        "published_count": published,
        "inventory_count": inventory,
        "page_coverage": published + inventory,
        "missing_case_ids": missing,
        "target_office_total": target_total,
        "target_office_below_floor": target_below_floor,
    }
    audit["complete"] = (
        not missing
        and not target_below_floor
        and audit["base_count"] == audit["report_case_count"]
        and (audit["base_count"] == 0 or audit["page_coverage"] >= audit["base_count"])
    )
    return audit


def _case_target_signal(case: Any) -> str:
    return " | ".join(
        [
            gui._normalize_ws(case.tema),
            gui._normalize_ws(case.punchline),
            " ; ".join(case.partes or []),
            " ; ".join(case.assuntos or []),
        ]
    )


def varredura_alvo(
    plans: List[ReportPagePlan],
    *,
    data_source_id: str,
    database_url: str,
    model: str,
    fix: bool,
) -> List[Dict[str, Any]]:
    """Para cada relatorio existente, detecta casos com alto cargo da Republica
    (FEDERAL_HIGH_OFFICE_RE) presentes na base mas fora da tabela de destaques.
    Com fix=True, regenera apenas as paginas com omissao."""
    resultados: List[Dict[str, Any]] = []
    for plan in plans:
        if plan.kind not in {"weekly", "monthly"} or not plan.start or not plan.end:
            continue
        cases = report.query_cases_by_period(data_source_id, plan.start.isoformat(), plan.end.isoformat())
        alvo = [
            case
            for case in cases
            if getattr(case, "dep_federal", False) or report.FEDERAL_HIGH_OFFICE_RE.search(_case_target_signal(case))
        ]
        if not alvo:
            resultados.append({"title": plan.current_title, "alvo": 0, "omitidos": [], "regenerado": False})
            print(f"[ok] {plan.current_title}: sem casos-alvo no periodo")
            continue
        published = report.load_cases_from_published_summary_table(plan.page_id)
        published_ids = {report._normalize_notion_id(str(c.page_id)) for c in published if gui._normalize_ws(c.page_id)}
        omitidos = [
            case
            for case in alvo
            if report._normalize_notion_id(str(case.page_id)) not in published_ids
        ]
        item = {
            "title": plan.current_title,
            "alvo": len(alvo),
            "omitidos": [case.process_label() for case in omitidos],
            "regenerado": False,
        }
        if omitidos:
            print(
                f"[OMISSAO] {plan.current_title}: {len(omitidos)}/{len(alvo)} caso(s)-alvo fora dos destaques: "
                + "; ".join(case.process_label() for case in omitidos[:6])
            )
            if fix:
                regenerate_page(plan, database_url=database_url, model=model, log=print)
                snapshot_report_json(plan)
                _mark_weekly_manifest(plan, case_count=len(cases))
                item["regenerado"] = True
        else:
            print(f"[ok] {plan.current_title}: {len(alvo)} caso(s)-alvo, todos nos destaques")
        resultados.append(item)
    payload = {"finished_at_utc": utc_now_iso(), "resultados": resultados}
    write_json_atomic(REFAZER_DIR / "varredura_alvo.json", payload)
    print(f"\nVarredura salva em: {REFAZER_DIR / 'varredura_alvo.json'}")
    return resultados


def regenerar_somente_desatualizados(
    plans: List[ReportPagePlan],
    *,
    data_source_id: str,
    database_url: str,
    model: str,
    dry_run: bool,
) -> None:
    """Regenera apenas as semanas cuja contagem de casos na base divergiu do
    manifesto (ex.: decisoes restauradas/importadas depois do relatorio),
    reaproveitando o cache de analises para nao reanalisar o que ja foi feito."""
    manifest = gui.read_manifest()
    generated = manifest.get("generated_reports") or {}
    desatualizadas = 0
    for plan in plans:
        if plan.kind != "weekly" or not plan.start or not plan.end:
            continue
        cases = report.query_cases_by_period(data_source_id, plan.start.isoformat(), plan.end.isoformat())
        current = len(cases)
        entry = generated.get(plan.target_title or plan.current_title)
        previous = None
        if isinstance(entry, dict):
            try:
                previous = int(entry.get("case_count"))
            except Exception:
                previous = None
        if previous is not None and previous == current:
            print(f"[ok] {plan.current_title}: {current} casos (sem mudanca)")
            continue
        desatualizadas += 1
        print(f"[DESATUALIZADA] {plan.current_title}: base={current} vs manifesto={previous} — regenerando...")
        if dry_run:
            continue
        regenerate_page(plan, database_url=database_url, model=model, log=print)
        snapshot_report_json(plan)
        _mark_weekly_manifest(plan, case_count=current)
    print(f"\nSemanas desatualizadas: {desatualizadas}" + (" (dry-run, nada regenerado)" if dry_run else ""))


def load_previous_audit() -> Dict[str, Any]:
    try:
        with AUDIT_JSON.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_audit(plans: List[ReportPagePlan]) -> None:
    REFAZER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "finished_at_utc": utc_now_iso(),
        "pages": [
            {
                "page_id": plan.page_id,
                "current_title": plan.current_title,
                "target_title": plan.target_title,
                "kind": plan.kind,
                "period": (
                    f"{plan.start.isoformat()} a {plan.end.isoformat()}" if plan.start and plan.end else ""
                ),
                "actions": plan.actions,
                "status": plan.status,
                **plan.audit,
            }
            for plan in plans
        ],
    }
    write_json_atomic(AUDIT_JSON, payload)

    lines = [
        "# Auditoria — refazer relatórios DJe/TSE",
        "",
        f"Gerado em {payload['finished_at_utc']} (UTC).",
        "",
        "| Relatório | Período | Status | Base | Analisados | Destaques | Inventário | Faltando | Alvo | Alvo<8 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["pages"]:
        lines.append(
            "| {title} | {period} | {status} | {base} | {analyzed} | {published} | {inventory} | {missing} | {alvo} | {alvo_baixo} |".format(
                title=item.get("target_title") or item.get("current_title"),
                period=item.get("period") or "-",
                status=item.get("status"),
                base=item.get("base_count", "-"),
                analyzed=item.get("analyzed_count", "-"),
                published=item.get("published_count", "-"),
                inventory=item.get("inventory_count", "-"),
                missing=len(item.get("missing_case_ids") or []),
                alvo=item.get("target_office_total", "-"),
                alvo_baixo=len(item.get("target_office_below_floor") or []),
            )
        )
    pendentes = [item for item in payload["pages"] if item.get("status") not in {"ok", "skipped"}]
    lines.append("")
    lines.append(
        f"{len(payload['pages'])} página(s) processada(s); {len(pendentes)} com pendência."
    )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Renomeia, regenera e audita relatorios DJe/TSE no Notion.")
    parser.add_argument("--parent-page-url", default=gui.DEFAULT_REPORTS_PARENT_URL)
    parser.add_argument("--database-url", default=report.DEFAULT_SOURCE_DATABASE_URL)
    parser.add_argument("--model", default=report.DEFAULT_OPENAI_MODEL)
    parser.add_argument("--only-title", default="", help="Processa apenas titulos que contenham esta substring.")
    parser.add_argument("--limit", type=int, default=0, help="Limita o numero de paginas processadas.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra o plano sem escrever nada no Notion.")
    parser.add_argument("--rename-only", action="store_true", help="So renomeia titulos; nao regenera conteudo.")
    parser.add_argument("--audit-only", action="store_true", help="So audita as paginas usando snapshots existentes.")
    parser.add_argument("--skip-done", action="store_true", help="Pula paginas com status ok na auditoria anterior.")
    parser.add_argument("--archive-unrecognized", action="store_true", help="Arquiva paginas de titulo nao reconhecido.")
    parser.add_argument(
        "--varredura-alvo",
        action="store_true",
        help="So varre os relatorios procurando altos cargos fora dos destaques (nao regenera nada sem --fix).",
    )
    parser.add_argument("--fix", action="store_true", help="Com --varredura-alvo: regenera as paginas com omissao detectada.")
    parser.add_argument(
        "--somente-desatualizados",
        action="store_true",
        help="Regenera apenas semanas cuja contagem de casos na base divergiu do manifesto (economico).",
    )
    args = parser.parse_args()

    class InitArgs:
        openai_max_workers = report.DEFAULT_OPENAI_MAX_WORKERS
        max_cases_per_batch = report.DEFAULT_MAX_CASES_PER_BATCH
        openai_target_rpm = report.DEFAULT_OPENAI_TARGET_RPM
        max_openai_cases = report.DEFAULT_MAX_OPENAI_CASES
        openai_triage_threshold = report.DEFAULT_OPENAI_TRIAGE_THRESHOLD
        disable_analysis_cache = False
        model = args.model
        force_openai_fallback = False

    report.initialize_clients(InitArgs)

    parent_page_id = report.extract_notion_id_from_url(args.parent_page_url)
    database_id = report.extract_notion_id_from_url(args.database_url)
    data_source_id = report.retrieve_database_and_datasource_id(database_id)

    pages = list_report_pages_ordered(parent_page_id)
    plans = [classify_page(title, page_id) for title, page_id in pages]
    if args.only_title:
        needle = args.only_title.casefold()
        plans = [plan for plan in plans if needle in plan.current_title.casefold() or needle in plan.target_title.casefold()]
    plans.sort(key=lambda plan: (plan.start or date.max, plan.current_title))

    # Colisoes: mais de uma pagina com o mesmo titulo-alvo. A que ja tem o titulo
    # exato e mantida; as demais sao arquivadas.
    by_target: Dict[str, List[ReportPagePlan]] = {}
    for plan in plans:
        if plan.kind != "unknown" and plan.target_title:
            by_target.setdefault(plan.target_title, []).append(plan)
    for target_title, group in by_target.items():
        if len(group) <= 1:
            continue
        keeper = next((plan for plan in group if plan.current_title.strip() == target_title), group[0])
        for plan in group:
            if plan is not keeper:
                plan.kind = "duplicate"
                plan.actions.append("archive_duplicate")

    if args.somente_desatualizados:
        regenerar_somente_desatualizados(
            [plan for plan in plans if plan.kind == "weekly"],
            data_source_id=data_source_id,
            database_url=args.database_url,
            model=args.model,
            dry_run=bool(args.dry_run),
        )
        return 0

    if args.varredura_alvo:
        varredura_alvo(
            [plan for plan in plans if plan.kind in {"weekly", "monthly"}],
            data_source_id=data_source_id,
            database_url=args.database_url,
            model=args.model,
            fix=bool(args.fix),
        )
        return 0

    previous = load_previous_audit()
    done_titles = {
        item.get("target_title") or item.get("current_title")
        for item in (previous.get("pages") or [])
        if item.get("status") == "ok"
    }

    processed = 0
    for plan in plans:
        if args.limit and processed >= args.limit:
            plan.status = "skipped"
            plan.actions.append("limit_reached")
            continue

        if plan.kind == "unknown":
            plan.status = "skipped"
            plan.actions.append("unrecognized_title")
            if args.archive_unrecognized and not args.dry_run:
                archive_page(plan.page_id)
                plan.actions.append("archived")
            print(f"[pulado] titulo nao reconhecido: {plan.current_title}")
            continue

        if plan.kind == "duplicate":
            plan.status = "archived_duplicate"
            if not args.dry_run:
                archive_page(plan.page_id)
            print(f"[duplicado] arquivando: {plan.current_title} -> mantido {plan.target_title}")
            continue

        if args.skip_done and plan.target_title in done_titles:
            plan.status = "skipped"
            plan.actions.append("already_ok")
            print(f"[pulado] ja concluido: {plan.target_title}")
            continue

        needs_rename = plan.current_title.strip() != plan.target_title
        print(
            f"[{plan.kind}] {plan.current_title} | periodo {plan.start} a {plan.end}"
            + (f" | renomear para: {plan.target_title}" if needs_rename else "")
        )

        if args.dry_run:
            plan.status = "dry_run"
            processed += 1
            continue

        try:
            if needs_rename:
                rename_page(plan.page_id, plan.target_title)
                plan.actions.append("renamed")
            if not args.rename_only and not args.audit_only:
                regenerate_page(plan, database_url=args.database_url, model=args.model, log=print)
                plan.actions.append("regenerated")
                payload = snapshot_report_json(plan)
            else:
                snapshot = REFAZER_DIR / f"{_safe_filename(plan.target_title)}.report.json"
                try:
                    payload = json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.exists() else {}
                except Exception:
                    payload = {}
            if not args.rename_only:
                plan.audit = audit_page(plan, payload, data_source_id=data_source_id)
                plan.status = "ok" if plan.audit.get("complete") else "incomplete"
            else:
                plan.status = "renamed_only"
        except Exception as exc:
            plan.status = "error"
            plan.audit["error"] = str(exc)
            print(f"[ERRO] {plan.current_title}: {exc}")
        processed += 1
        write_audit(plans)

    write_audit(plans)
    print(f"\nAuditoria salva em: {AUDIT_JSON}")
    print(f"Resumo legivel em: {AUDIT_MD}")
    pendentes = [plan for plan in plans if plan.status not in {"ok", "skipped", "dry_run", "renamed_only", "archived_duplicate"}]
    if pendentes:
        print(f"{len(pendentes)} pagina(s) com pendencia: " + "; ".join(plan.target_title or plan.current_title for plan in pendentes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
