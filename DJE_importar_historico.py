#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Importa o dump historico do DJe/TSE para a base do Notion, ANO A ANO.

Contexto (28/06/2026): o export `TSE_decisoes_consolidado_*.csv` da pasta da
Consultoria no OneDrive e um CSV BRUTO (mesmas colunas da busca de
jurisprudencia do TSE) cobrindo ~10 anos (2016-2026, ~75 mil decisoes). Este
orquestrador popula a base do Notion APENAS com os registros que faltam, sem
gerar relatorios e sem chamadas de IA do relatorio.

Por que ano a ano:
  * limita a memoria do tratamento (o SJUR carrega o CSV inteiro);
  * cada ano e uma unidade que conclui sozinha -> retomada barata;
  * a query de existentes do importer e por periodo (fica restrita ao ano).

Etapas por ano (com checkpoint em disco, idempotente):
  1) filtra o bruto -> Artefatos/intermediarios/historico_brutos/bruto_<ano>.csv
  2) trata via SJUR (tema/punchline + assuntos canonicos via gpt-5.4-nano)
     -> Artefatos/dados/csv/DJE_hist_<ano>_consolidado.csv
  3) importa via NOTION_DJE_csv_to_notion.py em modo create-only (so faltantes)

Reexecutar retoma de onde parou. O modo create-only NAO toca paginas ja
existentes; reimportar so cria o que ainda falta.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_ORIGEM_PADRAO = Path(
    r"C:\Users\mauri\OneDrive\Documentos\12 - Consultoria Legislativa\DJe"
    r"\TSE_decisoes_consolidado_2026-06-28.csv"
)
INTERMEDIARIOS = PROJECT_ROOT / "Artefatos" / "intermediarios"
BRUTOS_DIR = INTERMEDIARIOS / "historico_brutos"
CONSOLIDADOS_DIR = PROJECT_ROOT / "Artefatos" / "dados" / "csv"
ESTADO_FILE = INTERMEDIARIOS / "historico_import_estado.json"
LOG_FILE = INTERMEDIARIOS / "historico_import.log"

# Do mais recente ao mais antigo: popula primeiro o que mais importa.
ANOS_PADRAO = [str(a) for a in range(2026, 2015, -1)]

# Mesmos parametros que a GUI passa ao tratamento (DJE_relatorios_semanais_gui).
THEME_MODEL = os.getenv("DJE_THEME_PUNCHLINE_MODEL", "gpt-5.4-nano") or "gpt-5.4-nano"
DATABASE_URL = "https://www.notion.so/328721955c648020988af0654c3554a2?v=cfbaf74128e14ad1a210db1be8fc68c1"

csv.field_size_limit(2_000_000_000)


# --------------------------------------------------------------------------- log
def _log(msg: str) -> None:
    carimbo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = f"[{carimbo}] {msg}"
    print(linha, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(linha + "\n")
    except Exception:
        pass


# ------------------------------------------------------------------------- estado
def carregar_estado() -> Dict:
    try:
        with ESTADO_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def salvar_estado(estado: Dict) -> None:
    ESTADO_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ESTADO_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(estado, fh, ensure_ascii=False, indent=2)
    tmp.replace(ESTADO_FILE)


def _ano_estado(estado: Dict, ano: str) -> Dict:
    bloco = estado.get(ano)
    if not isinstance(bloco, dict):
        bloco = {"filtered": False, "treated": False, "imported": False}
        estado[ano] = bloco
    return bloco


# ----------------------------------------------------------------------- filtragem
def _ano_da_data(valor: str) -> Optional[str]:
    valor = (valor or "").strip()
    # formato esperado DD/MM/YYYY; tolera YYYY-MM-DD por seguranca
    if len(valor) >= 10 and valor[2] == "/" and valor[5] == "/":
        return valor[6:10]
    if len(valor) >= 10 and valor[4] == "-" and valor[7] == "-":
        return valor[0:4]
    return None


def bruto_do_ano(ano: str) -> Path:
    return BRUTOS_DIR / f"bruto_{ano}.csv"


def particionar_tudo(origem: Path, anos: List[str]) -> Dict[str, int]:
    """Le o bruto UMA vez e separa as linhas por ano (um arquivo por ano).

    Evita reler o CSV de ~696 MB a cada ano. So os anos pedidos sao gravados;
    linhas de outros anos sao ignoradas. Escreve em arquivos .tmp e renomeia ao
    final (atomico) para que uma interrupcao nao deixe um ano pela metade.
    """
    BRUTOS_DIR.mkdir(parents=True, exist_ok=True)
    alvo = set(anos)
    tmp_paths: Dict[str, Path] = {a: bruto_do_ano(a).with_suffix(".csv.tmp") for a in anos}
    handles: Dict[str, object] = {}
    writers: Dict[str, object] = {}
    contagem: Dict[str, int] = {a: 0 for a in anos}
    try:
        with origem.open("r", encoding="utf-8-sig", newline="") as fin:
            reader = csv.reader(fin)
            header = next(reader)
            try:
                idx_data = header.index("dataDecisao")
            except ValueError:
                raise RuntimeError("Coluna 'dataDecisao' nao encontrada no cabecalho do CSV bruto.")
            for a in anos:
                fh = tmp_paths[a].open("w", encoding="utf-8-sig", newline="")
                handles[a] = fh
                w = csv.writer(fh)
                w.writerow(header)
                writers[a] = w
            for campos in reader:
                if idx_data >= len(campos):
                    continue
                ano = _ano_da_data(campos[idx_data])
                if ano in alvo:
                    writers[ano].writerow(campos)  # type: ignore[attr-defined]
                    contagem[ano] += 1
    finally:
        for fh in handles.values():
            try:
                fh.close()  # type: ignore[attr-defined]
            except Exception:
                pass
    for a in anos:
        tmp_paths[a].replace(bruto_do_ano(a))
    return contagem


# ----------------------------------------------------------------------- subprocess
def _rodar(cmd: List[str]) -> int:
    _log("$ " + " ".join(str(p) for p in cmd))
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [str(p) for p in cmd],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    ultimas: List[str] = []
    for linha in proc.stdout:
        linha = linha.rstrip()
        ultimas.append(linha)
        if len(ultimas) > 400:
            ultimas.pop(0)
        print(linha, flush=True)
    rc = proc.wait()
    # guarda as ultimas linhas (uteis para extrair o JSON do importer)
    _rodar.ultimas_linhas = ultimas  # type: ignore[attr-defined]
    return rc


def _rodar_com_retry(cmd: List[str], *, tentativas: int = 5, pausa_base_s: int = 30) -> int:
    """Roda um comando idempotente (import upsert/create-only) repetindo se falhar.
    Cobre quedas de rede transitorias (ConnectionReset etc.) que, sem retry,
    derrubariam o ano inteiro. Como o import e idempotente, repetir e seguro.
    Usa BACKOFF EXPONENCIAL (30, 60, 120, 240s) para escapar de janelas de
    instabilidade de varios minutos, que um retry de pausa fixa nao cobre."""
    rc = 1
    for tentativa in range(1, tentativas + 1):
        rc = _rodar(cmd)
        if rc == 0:
            return 0
        if tentativa < tentativas:
            pausa = min(pausa_base_s * (2 ** (tentativa - 1)), 300)
            _log(f"  ... tentativa {tentativa}/{tentativas} falhou (rc={rc}); aguardando {pausa}s e repetindo")
            time.sleep(pausa)
        else:
            _log(f"  ... tentativa {tentativa}/{tentativas} falhou (rc={rc}); desistindo deste ano")
    return rc


def _extrair_resumo_import(linhas: List[str]) -> Dict:
    # o importer imprime asdict(result) como JSON indentado no final
    texto = "\n".join(linhas)
    fim = texto.rfind("}")
    ini = texto.rfind("{", 0, texto.find('"created"') + 1) if '"created"' in texto else -1
    # estrategia robusta: tenta achar o ultimo bloco {...} parseavel
    for abre in range(len(texto)):
        if texto[abre] == "{":
            trecho = texto[abre : fim + 1]
            try:
                obj = json.loads(trecho)
                if isinstance(obj, dict) and "created" in obj:
                    return obj
            except Exception:
                continue
    return {}


def _cmd_sjur(bruto: Path, combined_name: str) -> List[str]:
    return [
        sys.executable,
        "SJUR_csv_to_csv_NOTIONfriendly_v2.py",
        str(bruto),
        "--no-gui",
        "--out-dir",
        str(CONSOLIDADOS_DIR),
        "--combined-name",
        combined_name,
        "--max-texto-chars",
        "9000",
        "--openai-model",
        THEME_MODEL,
        # Config acelerada (29/06/2026): o gargalo do tratamento era o OpenAI em
        # lotes ~sequenciais limitados por rpm=120/4 workers (~2 chamadas/s).
        # Subindo para ~10 workers / rpm 400 / batch 100 fica ~3-4x mais rapido,
        # mantendo workers <= pool HTTP (10) p/ nao saturar conexoes ("Connection
        # pool is full").
        "--openai-max-workers",
        str(int(os.getenv("DJE_HIST_OPENAI_WORKERS", "10") or "10")),
        "--openai-batch-size",
        str(int(os.getenv("DJE_HIST_OPENAI_BATCH", "100") or "100")),
        "--openai-target-rpm",
        str(int(os.getenv("DJE_HIST_OPENAI_RPM", "400") or "400")),
        "--openai-timeout",
        "60",
        "--verbose",
        "--gerar-tema-punchline-chatgpt",
        "--enriquecer-assuntos-openai",
        "--assuntos-taxonomy-mode",
        "controlled",
        "--assuntos-max-itens",
        "2",
    ]


def _cmd_import(consolidado: Path, dry_run: bool, import_mode: str = "create-only") -> List[str]:
    cmd = [
        sys.executable,
        "NOTION_DJE_csv_to_notion.py",
        "--csv",
        str(consolidado),
        "--database-url",
        DATABASE_URL,
        "--mode",
        import_mode,
        "--model",
        THEME_MODEL,
        "--verbose",
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


# Acima desta fracao de linhas sem 'tema' o tratamento e considerado degradado
# (sinal classico de OpenAI recusando por falta de credito/rate-limit): o ano NAO
# e importado e fica marcado para refazer, em vez de gravar lacunas no Notion.
LIMIAR_TEMA_VAZIO = float(os.getenv("DJE_HIST_LIMIAR_TEMA_VAZIO", "0.05") or "0.05")


def taxa_tema_vazio(consolidado: Path) -> tuple[int, int]:
    total = 0
    vazio = 0
    with consolidado.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            if not (row.get("tema") or "").strip():
                vazio += 1
    return vazio, total


def reset_ano(ano: str, estado: Dict) -> None:
    """Forca o re-tratamento limpo de um ano: apaga consolidado e intermediarios
    *_notion.csv (senao o SJUR reaproveita o tratamento degradado preservado) e
    zera treated/imported no estado. NAO apaga nada no Notion."""
    alvos = [
        CONSOLIDADOS_DIR / f"DJE_hist_{ano}_consolidado.csv",
        INTERMEDIARIOS / "csv" / f"bruto_{ano}_notion.csv",
    ]
    # remove tambem eventuais CSVs de etapa do mesmo ano
    etapa_dir = INTERMEDIARIOS / "csv"
    if etapa_dir.is_dir():
        for p in etapa_dir.glob(f"bruto_{ano}_*.csv"):
            alvos.append(p)
    removidos = []
    for p in alvos:
        try:
            if p.exists():
                p.unlink()
                removidos.append(p.name)
        except Exception as exc:  # pragma: no cover
            _log(f"[{ano}] aviso: nao consegui remover {p.name}: {exc}")
    bloco = _ano_estado(estado, ano)
    bloco["treated"] = False
    bloco["imported"] = False
    bloco.pop("resumo", None)
    bloco.pop("resumo_parcial", None)
    _log(f"[{ano}] reset p/ re-tratamento limpo | removidos: {removidos or 'nada'}")


# --------------------------------------------------------------------- processa ano
def processar_ano(ano: str, estado: Dict, *, dry_run_import: bool, import_mode: str = "create-only") -> bool:
    bloco = _ano_estado(estado, ano)
    bruto = bruto_do_ano(ano)
    combined_name = f"DJE_hist_{ano}_consolidado.csv"
    consolidado = CONSOLIDADOS_DIR / combined_name

    # 1) particionamento (normalmente ja feito globalmente no main; aqui so um
    # fallback caso o arquivo do ano nao exista).
    if not (bloco.get("filtered") and bruto.exists()):
        _log(f"[{ano}] particionando o bruto deste ano...")
        cont = particionar_tudo(CSV_ORIGEM, [ano])
        bloco["linhas_brutas"] = cont.get(ano, 0)
        bloco["filtered"] = True
        salvar_estado(estado)
    n = int(bloco.get("linhas_brutas", 0) or 0)
    _log(f"[{ano}] {n} decisoes brutas -> {bruto.name}")
    if n == 0:
        bloco["treated"] = True
        bloco["imported"] = True
        bloco["resumo"] = {"created": 0, "unchanged": 0, "rows_seen": 0, "vazio": True}
        salvar_estado(estado)
        _log(f"[{ano}] nenhuma decisao neste ano; pulando.")
        return True

    # 2) tratamento (SJUR + OpenAI)
    if bloco.get("treated") and consolidado.exists():
        _log(f"[{ano}] tratamento ja feito -> reaproveitando {consolidado.name}")
    else:
        _log(f"[{ano}] tratando com SJUR (tema/punchline + assuntos via {THEME_MODEL})...")
        rc = _rodar(_cmd_sjur(bruto, combined_name))
        if rc != 0 or not consolidado.exists():
            _log(f"[{ano}] ERRO no tratamento SJUR (rc={rc}). Consolidado nao gerado.")
            bloco["treated"] = False
            salvar_estado(estado)
            return False
        # Salvaguarda anti-degradacao: se muitas linhas ficaram sem 'tema', a
        # OpenAI provavelmente recusou (credito/rate-limit). Nao importa lacunas.
        vazio, total = taxa_tema_vazio(consolidado)
        frac = (vazio / total) if total else 0.0
        if total and frac > LIMIAR_TEMA_VAZIO:
            _log(
                f"[{ano}] ALERTA: {vazio}/{total} ({frac:.1%}) linhas SEM tema -> tratamento "
                f"DEGRADADO (provavel falta de credito/limite OpenAI). NAO sera importado; "
                f"apagando consolidado p/ forcar novo tratamento limpo ao reexecutar."
            )
            bloco["treated"] = False
            bloco["tema_vazio"] = {"vazio": vazio, "total": total}
            try:
                consolidado.unlink()
            except Exception:
                pass
            salvar_estado(estado)
            return False
        bloco["treated"] = True
        bloco["tema_vazio"] = {"vazio": vazio, "total": total}
        salvar_estado(estado)
        _log(f"[{ano}] tratamento concluido ({vazio}/{total} sem tema) -> {consolidado.name}")

    # 3) import (create-only)
    if bloco.get("imported"):
        _log(f"[{ano}] import ja concluido: {bloco.get('resumo')}")
        return True
    _log(f"[{ano}] importando para o Notion ({import_mode}{' / DRY-RUN' if dry_run_import else ''})...")
    rc = _rodar_com_retry(_cmd_import(consolidado, dry_run_import, import_mode))
    resumo = _extrair_resumo_import(getattr(_rodar, "ultimas_linhas", []))
    if rc != 0:
        _log(f"[{ano}] ERRO no import (rc={rc}).")
        bloco["resumo_parcial"] = resumo
        salvar_estado(estado)
        return False
    bloco["imported"] = not dry_run_import  # dry-run nao marca como concluido
    bloco["resumo"] = resumo
    salvar_estado(estado)
    _log(
        f"[{ano}] import OK | criados={resumo.get('created', '?')} "
        f"| atualizados={resumo.get('updated', '?')} "
        f"| inalterados={resumo.get('unchanged', '?')} "
        f"| linhas={resumo.get('rows_seen', '?')} | pulados={resumo.get('rows_skipped', '?')}"
    )
    return True


# ------------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_ORIGEM_PADRAO), help="CSV bruto consolidado de origem.")
    parser.add_argument("--anos", default="", help="Lista de anos separados por virgula (default: 2026..2016).")
    parser.add_argument("--dry-run-import", action="store_true", help="Passa --dry-run ao importer (nao escreve no Notion).")
    parser.add_argument("--reset", action="store_true", help="Ignora o estado salvo e recomeca o controle (nao apaga nada no Notion).")
    parser.add_argument("--import-mode", default="create-only", choices=("create-only", "upsert", "update-only"),
                        help="Modo do importer. 'upsert' ATUALIZA paginas existentes (use p/ corrigir anos ja importados degradados).")
    parser.add_argument("--retratar", action="store_true",
                        help="Apaga consolidado+intermediarios e zera treated/imported dos anos pedidos, forcando tratamento limpo (nao apaga nada no Notion).")
    args = parser.parse_args()

    global CSV_ORIGEM
    CSV_ORIGEM = Path(args.csv).expanduser().resolve()
    if not CSV_ORIGEM.exists():
        _log(f"CSV de origem nao encontrado: {CSV_ORIGEM}")
        return 2

    anos = [a.strip() for a in args.anos.split(",") if a.strip()] if args.anos else list(ANOS_PADRAO)

    estado = {} if args.reset else carregar_estado()
    estado.setdefault("_meta", {})
    estado["_meta"]["csv_origem"] = str(CSV_ORIGEM)
    estado["_meta"]["iniciado_em"] = estado["_meta"].get("iniciado_em") or datetime.now().isoformat(timespec="seconds")
    salvar_estado(estado)

    _log("=" * 78)
    _log(f"Importacao historica DJe -> Notion | origem={CSV_ORIGEM.name}")
    _log(f"Anos: {', '.join(anos)} | modo import={args.import_mode} | retratar={args.retratar} | relatorios=NAO")
    _log(f"Base Notion: {DATABASE_URL}")
    _log("=" * 78)

    # Re-tratamento limpo: apaga consolidado+intermediarios degradados e zera o
    # estado dos anos pedidos (ex.: anos afetados por falta de credito OpenAI).
    if args.retratar:
        for ano in anos:
            reset_ano(ano, estado)
        salvar_estado(estado)

    # Particiona o bruto de ~696 MB UMA unica vez (uma leitura -> um arquivo por
    # ano), em vez de relê-lo a cada ano.
    falta = [a for a in anos if not (_ano_estado(estado, a).get("filtered") and bruto_do_ano(a).exists())]
    if falta:
        _log(f"Particionando o bruto por ano (1 leitura) para: {', '.join(falta)} ...")
        cont = particionar_tudo(CSV_ORIGEM, falta)
        for a in falta:
            b = _ano_estado(estado, a)
            b["linhas_brutas"] = cont.get(a, 0)
            b["filtered"] = True
        salvar_estado(estado)
        _log("Particionamento concluido: " + " | ".join(f"{a}={cont.get(a, 0)}" for a in falta))
    else:
        _log("Particionamento ja feito; reaproveitando os arquivos por ano.")

    sucessos = 0
    falhas: List[str] = []
    for ano in anos:
        bloco = _ano_estado(estado, ano)
        if bloco.get("imported"):
            _log(f"[{ano}] ja concluido anteriormente -> {bloco.get('resumo')}")
            sucessos += 1
            continue
        try:
            ok = processar_ano(ano, estado, dry_run_import=args.dry_run_import, import_mode=args.import_mode)
        except Exception as exc:  # pragma: no cover
            _log(f"[{ano}] EXCECAO: {exc}")
            ok = False
        if ok:
            sucessos += 1
        else:
            falhas.append(ano)
            # aborta se nada deu certo ainda (provavel erro sistemico: chave,
            # schema, rede). Apos o 1o sucesso, falhas isoladas nao abortam.
            if sucessos == 0:
                _log(f"[{ano}] falha sem nenhum sucesso previo -> abortando para investigar.")
                break

    _log("-" * 78)
    _log(f"FIM | anos OK={sucessos} | falhas={falhas or 'nenhuma'}")
    total_criados = sum(
        int((estado.get(a, {}) or {}).get("resumo", {}).get("created", 0) or 0)
        for a in anos
        if isinstance(estado.get(a), dict)
    )
    _log(f"Total criados (soma dos anos concluidos): {total_criados}")
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
