# -*- coding: utf-8 -*-
"""GUI para atualizar a base Notion 'stf' (acórdãos STF de Repercussão Geral) a partir de
um novo CSV de export. Insere os processos ausentes, enriquece os novos com IA e reindexa
a RAG. Abra pelo atalho "STF - Atualizar base" ou: python STF_atualizar_gui.py
"""
from __future__ import annotations

from pathlib import Path

import _atualizador_gui as G
import _stf_lib as L

ICONE = str(Path(__file__).resolve().parent / "icones" / "stf.ico")


def montar(csv: str, py: str, enriquecer: bool, reindexar: bool):
    passos = [("Inserir acórdãos novos no Notion",
               [py, "_stf_inserir_novos.py", "--apply", "--csv", csv])]
    if enriquecer:
        passos.append(("Enriquecer os novos (Resumo IA + Palavras-chave)",
                       [py, "_stf_enriquecer.py", "--faltantes"]))
    if reindexar:
        passos.append(("Reindexar a RAG (base stf)",
                       [py, "-m", "conle_gerador.notion_rag", "--indexar", "--bases", "stf"]))
    return passos


def verificar(csv: str, py: str):
    return [py, "_stf_inserir_novos.py", "--csv", csv]  # sem --apply = dry-run


if __name__ == "__main__":
    G.rodar(
        titulo="STF — Atualizar base (acórdãos RG)",
        insumo_label="CSV de export de acórdãos do STF",
        insumo_default=str(L.CSV_PATH), insumo_is_dir=False,
        montar_comandos=montar, verificar_comando=verificar, icone=ICONE,
    )
