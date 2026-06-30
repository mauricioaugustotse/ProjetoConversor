#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrige o ruido da coluna 'advogados' RE-EXTRAINDO do CSV bruto com o parser
BLINDADO (que agora rejeita cabecalho de processo/PJe, numero CNJ, classe etc.).

Motivo: o valor gravado no Notion, nos casos ruins, tem o cabecalho do PJe ou o
proprio texto da decisao colado. Tentar "limpar" esse valor e inviavel (inventa
nomes). A fonte correta e o bruto: re-extrair advogados de
partes+publicacoes+textoDecisao+textoEmenta com o parser blindado da o resultado
certo (advogado real sem ruido, ou vazio quando nao ha advogado).

So mexe em paginas cujo 'advogados' atual tem ruido (cabecalho de processo OU
texto de decisao colado) e cujo numero existe no bruto. Varre POR ANO (paginacao
do Notion para em ~10k) e cruza com o bruto do ano por numeroUnico. --dry-run
nao escreve."""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

csv.field_size_limit(2_000_000_000)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import NOTION_relatoriodeIA_v2 as report  # noqa: E402
import SJUR_csv_to_csv_NOTIONfriendly_v2 as sjur  # noqa: E402
import Artefatos.legado.SJUR_csv_to_csv_NOTIONfriendly as base  # noqa: E402

ANOS = list(range(2010, 2028))
BRUTOS = PROJECT_ROOT / "Artefatos" / "intermediarios" / "historico_brutos"
_CFG = sjur.MetadataExtractionConfig()

RUIDO_DECISAO = re.compile(
    r"(?i)\b(?:decis[ãa]o|ementa|trata-se|cuida-se|relat[óo]rio|ante\s+o\s+exposto|"
    r"agravad[oa]|recorrid[oa]|recorrente|agravante|embargante|impetrante|impetrad[oa]|"
    r"requerente|requerid[oa]|r[ée]u|interessad[oa]|noticiad[oa]|"
    r"minist[ée]rio\s+p[úu]blico|elei[çc][õo]es\s+\d)"
)


def tem_ruido(valor: str) -> bool:
    return bool(base.PROCESS_HEADER_NOISE_RE.search(valor) or RUIDO_DECISAO.search(valor))


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _rich(texto: str):
    texto = texto or ""
    blocos = [texto[i:i + 1900] for i in range(0, len(texto), 1900)] or [""]
    return [{"type": "text", "text": {"content": b}} for b in blocos]


class A:
    force_openai_fallback = True
    openai_max_workers = 4
    model = report.DEFAULT_OPENAI_MODEL


def _query_retry(ds, body, tentativas=6, pausa=15):
    for t in range(1, tentativas + 1):
        try:
            return report.notion_request("POST", f"/v1/data_sources/{ds}/query", json_body=body)
        except Exception as exc:
            if t >= tentativas:
                raise
            print(f"   ! query falhou ({t}/{tentativas}): {exc}; aguardando {pausa}s", flush=True)
            time.sleep(pausa)


def paginas_com_ruido_do_ano(ds, ano):
    """{numero_digits: [(page_id, adv_atual), ...]} das paginas com ruido no ano
    (lista por numero: um processo pode ter varias decisoes/paginas no ano)."""
    out = {}
    cursor = ""
    while True:
        body = {"page_size": 100, "filter": {"and": [
            {"property": "dataDecisao", "date": {"on_or_after": f"{ano}-01-01"}},
            {"property": "dataDecisao", "date": {"on_or_before": f"{ano}-12-31"}},
            {"property": "advogados", "rich_text": {"is_not_empty": True}},
        ]}}
        if cursor:
            body["start_cursor"] = cursor
        q = _query_retry(ds, body)
        for it in q.get("results", []) or []:
            props = it.get("properties", {})
            adv = report._property_rich_text(props.get("advogados", {}))
            if not adv.strip() or not tem_ruido(adv):
                continue
            num = _digits(report._property_rich_text(props.get("numeroUnico", {})))
            if num:
                out.setdefault(num, []).append((report._normalize_notion_id(str(it.get("id", ""))), adv))
        if not q.get("has_more"):
            break
        cursor = q.get("next_cursor") or ""
        if not cursor:
            break
    return out


def reextrair_do_bruto(ano, alvos):
    """Re-extrai advogados 1x por numero (primeira linha do bruto) e aplica a
    TODAS as paginas daquele numero. Retorna [(page_id, antes, depois)]."""
    path = BRUTOS / f"bruto_{ano}.csv"
    correcoes = []
    if not path.exists():
        return correcoes, set()
    novo_por_numero = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            num = _digits(row.get("numeroUnico", ""))
            if num not in alvos or num in novo_por_numero:
                continue
            novo_por_numero[num] = sjur.extract_advogados_multiselect(
                row.get("partes", ""), row.get("publicacoes", ""),
                row.get("textoDecisao", ""), row.get("textoEmenta", ""),
                config=_CFG,
            )
    for num, paginas in alvos.items():
        if num not in novo_por_numero:
            continue
        novo = novo_por_numero[num]
        for page_id, antes in paginas:
            if report._normalize_ws(novo) != report._normalize_ws(antes):
                correcoes.append((page_id, antes, novo))
    return correcoes, set(novo_por_numero)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report.initialize_clients(A)
    ds = report.retrieve_database_and_datasource_id(
        report.extract_notion_id_from_url(report.DEFAULT_SOURCE_DATABASE_URL))

    todas = []          # (page_id, antes, depois)
    numeros_sem_bruto = 0
    paginas_ruido = 0
    numeros_ruido = 0
    for ano in ANOS:
        alvos = paginas_com_ruido_do_ano(ds, ano)
        if not alvos:
            continue
        paginas_ruido += sum(len(v) for v in alvos.values())
        numeros_ruido += len(alvos)
        correcoes, achados = reextrair_do_bruto(ano, alvos)
        numeros_sem_bruto += len(alvos) - len(achados)
        todas.extend(correcoes)
        print(f"   ... {ano}: numeros c/ ruido={len(alvos)} | correcoes={len(correcoes)} | sem bruto={len(alvos)-len(achados)}",
              file=sys.stderr, flush=True)

    vazias = sum(1 for _, _, d in todas if not d.strip())
    print("\n========================= RESUMO =========================")
    print(f"paginas com RUIDO em advogados:  {paginas_ruido} (em {numeros_ruido} processos)")
    print(f"paginas a corrigir (valor muda): {len(todas)}")
    print(f"   - ficam vazias (sem advogado real no bruto): {vazias}")
    print(f"   - ficam com advogado(s) limpos:              {len(todas) - vazias}")
    print(f"processos nao encontrados no bruto:             {numeros_sem_bruto}")
    print("==========================================================")
    print("\nAmostra (antes -> depois re-extraido do bruto):")
    for _pid, a, d in todas[:12]:
        print(f"   ANTES : {a[:90]}")
        print(f"   DEPOIS: {d[:90]!r}\n")

    if args.dry_run:
        print("[DRY-RUN] Nada foi escrito.")
        return 0

    print(f"Aplicando {len(todas)} correcoes...")
    ok, falhas = 0, []
    for i, (pid, _a, depois) in enumerate(todas, 1):
        try:
            report.notion_request("PATCH", f"/v1/pages/{pid}",
                                  json_body={"properties": {"advogados": {"rich_text": _rich(depois)}}})
            ok += 1
        except Exception as exc:
            falhas.append((pid, depois))
            print(f"   ! falha {pid}: {exc}", flush=True)
        if i % 50 == 0:
            print(f"   {i}/{len(todas)}...", flush=True)
    if falhas:
        print(f"Repetindo {len(falhas)}...")
        time.sleep(15)
        ainda = []
        for pid, depois in falhas:
            try:
                report.notion_request("PATCH", f"/v1/pages/{pid}",
                                      json_body={"properties": {"advogados": {"rich_text": _rich(depois)}}})
                ok += 1
            except Exception:
                ainda.append((pid, depois))
        falhas = ainda
    print(f"\nConcluido | corrigidas: {ok} | falhas: {len(falhas)}")
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
