#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marca o checkbox 'Dep. Federal' nas decisoes que envolvem deputado federal.

Criterio (validado contra a curadoria manual: ~96% de recall): procura a
expressao 'deputado(a) federal' / 'deputados federais' no conteudo da decisao
(tema, punchline, assuntos, classe, partes e -- por escolha do usuario -- o
texto da decisao). Alem da deteccao direta, aplica a REGRA DE PROPAGACAO POR
PROCESSO do usuario: todo processo (numeroUnico) com ao menos uma linha
detectada OU ja marcada tem TODAS as suas linhas marcadas.

NUNCA desmarca nada: so adiciona marcacoes (preserva a curadoria existente).
Use --dry-run para apenas contar, sem escrever no Notion.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import NOTION_relatoriodeIA_v2 as report  # noqa: E402

# Variacoes linguisticas de "deputado federal":
#  - genero/numero: deputado/deputada/deputados/deputadas + federal/federais
#  - flexao explicita: deputado(a) federal, deputado/a federal
#  - abreviacoes: Dep. Federal, Dep Federal, Dep. Fed., Dep Fed
# Evita falsos positivos comuns (Distrito/Policia/Justica Federal, deputado
# estadual/distrital, camara dos deputados), validado em casos sinteticos.
RE_DEPFED = re.compile(
    r"(?:deputad[oa]s?|dep\.?)\s*(?:\([oa]s?\)|/[oa]s?)?\s*federa(?:l|is)\b"
    r"|\bdep\.?\s*fed\.?\b",
    re.IGNORECASE,
)


class InitArgs:
    openai_max_workers = report.DEFAULT_OPENAI_MAX_WORKERS
    max_cases_per_batch = report.DEFAULT_MAX_CASES_PER_BATCH
    openai_target_rpm = report.DEFAULT_OPENAI_TARGET_RPM
    max_openai_cases = report.DEFAULT_MAX_OPENAI_CASES
    openai_triage_threshold = report.DEFAULT_OPENAI_TRIAGE_THRESHOLD
    disable_analysis_cache = False
    model = report.DEFAULT_OPENAI_MODEL
    force_openai_fallback = True


def montar_signal(case, incluir_texto: bool) -> str:
    partes = [
        case.tema,
        case.punchline,
        report._safe_join(case.assuntos),
        case.descricao_classe,
        case.sigla_classe,
        report._safe_join(case.partes),
    ]
    if incluir_texto:
        partes.append(case.texto_decisao or "")
    return report._safe_join(partes, sep=" | ")


def _query_com_retry(ds_id: str, body: dict, *, tentativas: int = 6, pausa_s: int = 15):
    """Query de paginacao tolerante a quedas de rede (alem dos retries internos
    do notion_request). Sem isso, uma ConnectionReset aborta a varredura toda."""
    for tentativa in range(1, tentativas + 1):
        try:
            return report.notion_request("POST", f"/v1/data_sources/{ds_id}/query", json_body=body)
        except Exception as exc:
            if tentativa >= tentativas:
                raise
            print(f"   ! query falhou (tentativa {tentativa}/{tentativas}): {exc}; aguardando {pausa_s}s", flush=True)
            time.sleep(pausa_s)


ANOS_VARREDURA = list(range(2010, 2028))


def varrer_base(ds_id: str, incluir_texto: bool):
    """Retorna (linhas, por_processo). Cada linha: dict id/numero/marcado/detectado.

    Varre POR ANO (filtro dataDecisao) porque a paginacao da query do Notion para
    em ~10.000 resultados; cada ano tem < 10k, entao a base toda (~75k) e coberta.
    A propagacao por processo continua correta: por_processo acumula por
    numeroUnico mesmo que um processo tenha decisoes em anos diferentes."""
    linhas = []
    por_processo = defaultdict(list)
    estado = {"total": 0}

    def _processa(raw):
        if not isinstance(raw, dict) or raw.get("object") != "page":
            return
        estado["total"] += 1
        try:
            case = report.build_case_record(raw)
        except Exception:
            return
        detectado = bool(RE_DEPFED.search(montar_signal(case, incluir_texto)))
        numero = re.sub(r"\D", "", case.numero_unico or "") or re.sub(r"\D", "", case.numero_processo or "")
        item = {
            "id": report._normalize_notion_id(str(raw.get("id", ""))),
            "numero": numero,
            "marcado": bool(case.dep_federal),
            "detectado": detectado,
            "data": case.data_decisao,
            "tema": (case.tema or "")[:60],
        }
        linhas.append(item)
        if numero:
            por_processo[numero].append(item)

    for ano in ANOS_VARREDURA:
        cursor = ""
        ano_count = 0
        while True:
            body = {
                "page_size": 100,
                "filter": {"and": [
                    {"property": "dataDecisao", "date": {"on_or_after": f"{ano}-01-01"}},
                    {"property": "dataDecisao", "date": {"on_or_before": f"{ano}-12-31"}},
                ]},
                "sorts": [{"property": "dataDecisao", "direction": "ascending"}],
            }
            if cursor:
                body["start_cursor"] = cursor
            payload = _query_com_retry(ds_id, body)
            for raw in payload.get("results", []) or []:
                _processa(raw)
                ano_count += 1
            if not payload.get("has_more"):
                break
            cursor = str(payload.get("next_cursor") or "")
            if not cursor:
                break
        if ano_count:
            print(f"   ... {ano}: {ano_count} | total {estado['total']}", file=sys.stderr, flush=True)
    return linhas, por_processo, estado["total"]


def marcar(page_id: str) -> bool:
    try:
        report.notion_request(
            "PATCH",
            f"/v1/pages/{page_id}",
            json_body={"properties": {"Dep. Federal": {"checkbox": True}}},
        )
        return True
    except Exception as exc:
        print(f"   ! falha ao marcar {page_id}: {exc}", flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="So conta; nao escreve no Notion.")
    parser.add_argument("--sem-texto", action="store_true", help="NAO usa o texto da decisao no criterio (so campos curados).")
    parser.add_argument("--sem-propagacao", action="store_true", help="NAO propaga por processo (so marca as linhas detectadas).")
    args = parser.parse_args()
    incluir_texto = not args.sem_texto
    propagar = not args.sem_propagacao

    report.initialize_clients(InitArgs)
    db_id = report.extract_notion_id_from_url(report.DEFAULT_SOURCE_DATABASE_URL)
    ds_id = report.retrieve_database_and_datasource_id(db_id)

    print(f"Varrendo a base (criterio: {'campos+texto' if incluir_texto else 'so campos'} | "
          f"propagacao por processo: {'sim' if propagar else 'nao'})...")
    linhas, por_processo, total = varrer_base(ds_id, incluir_texto)

    ja_marcadas = sum(1 for l in linhas if l["marcado"])
    detectadas = sum(1 for l in linhas if l["detectado"])

    # processos-alvo: tem >=1 linha detectada OU (se propagar) ja marcada
    a_marcar = []  # ids de linhas nao-marcadas que devem ser marcadas
    proc_alvo = 0
    for numero, grupo in por_processo.items():
        tem_detectada = any(g["detectado"] for g in grupo)
        tem_marcada = any(g["marcado"] for g in grupo)
        if propagar:
            alvo = tem_detectada or tem_marcada
        else:
            alvo = tem_detectada
        if not alvo:
            continue
        proc_alvo += 1
        for g in grupo:
            if not g["marcado"]:
                a_marcar.append(g)

    # linhas sem numero de processo: marca as detectadas isoladamente
    sem_numero = [l for l in linhas if not l["numero"] and l["detectado"] and not l["marcado"]]
    a_marcar.extend(sem_numero)

    print("\n========================= RESUMO =========================")
    print(f"paginas na base:            {total}")
    print(f"ja marcadas (curadoria):    {ja_marcadas}")
    print(f"detectadas pela regex:      {detectadas}")
    print(f"processos-alvo:             {proc_alvo}")
    print(f"NOVAS marcacoes a aplicar:  {len(a_marcar)}")
    novas_por_deteccao = sum(1 for l in a_marcar if l['detectado'])
    print(f"   - por deteccao direta:   {novas_por_deteccao}")
    print(f"   - por propagacao:        {len(a_marcar) - novas_por_deteccao}")
    print(f"total marcadas apos rodar:  {ja_marcadas + len(a_marcar)}")
    print("==========================================================")
    print("\nAmostra de novas marcacoes (deteccao direta):")
    for l in [x for x in a_marcar if x["detectado"]][:10]:
        print(f"   {l['data']} | {l['tema']}")

    if args.dry_run:
        print("\n[DRY-RUN] Nada foi escrito no Notion.")
        return 0

    print(f"\nAplicando {len(a_marcar)} marcacoes...")
    ok = 0
    falhas = []
    for i, l in enumerate(a_marcar, 1):
        if marcar(l["id"]):
            ok += 1
        else:
            falhas.append(l)
        if i % 500 == 0:
            print(f"   {i}/{len(a_marcar)} marcadas...", flush=True)
    # uma nova passada nas que falharam (rede transitoria)
    if falhas:
        print(f"Repetindo {len(falhas)} marcacoes que falharam...")
        time.sleep(10)
        ainda = []
        for l in falhas:
            if marcar(l["id"]):
                ok += 1
            else:
                ainda.append(l)
        falhas = ainda
    print(f"\nConcluido | marcadas com sucesso: {ok} | falhas: {len(falhas)}")
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
