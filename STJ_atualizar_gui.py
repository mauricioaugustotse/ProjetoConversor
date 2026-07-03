# -*- coding: utf-8 -*-
"""GUI para atualizar a base Notion 'STJ — Precedentes Qualificados (RAG)'. Os 5 CSVs do
painel NUGEPNAC devem estar na pasta stj/. A GUI consolida por tema, insere os precedentes
novos (idempotente), enriquece os novos com IA e reindexa a RAG.
Abra pelo atalho "STJ - Atualizar base" ou: python STJ_atualizar_gui.py
"""
from __future__ import annotations

from pathlib import Path

import _atualizador_gui as G
import _stj_lib as L


ICONE = str(Path(__file__).resolve().parent / "icones" / "stj.ico")


def montar(pasta: str, py: str, enriquecer: bool, reindexar: bool):
    # Os scripts _stj_* leem os CSVs da pasta fixa stj/ (L.STJ_DIR). Mantenha os CSVs
    # NUGEPNAC lá; a pasta mostrada serve de lembrete do local.
    passos = [
        ("Consolidar CSVs por tema (preview)", [py, "_stj_consolidar.py"]),
        ("Inserir precedentes novos no Notion (idempotente)", [py, "_stj_popular.py", "--apply"]),
    ]
    if enriquecer:
        passos.append(("Enriquecer os novos (Resumo IA + Palavras-chave)",
                       [py, "_stj_enriquecer.py", "--faltantes"]))
    if reindexar:
        passos.append(("Reindexar a RAG (base stj)",
                       [py, "-m", "conle_gerador.notion_rag", "--indexar", "--bases", "stj"]))
    return passos


def verificar(pasta: str, py: str):
    return [py, "_stj_consolidar.py"]


if __name__ == "__main__":
    G.rodar(
        titulo="STJ — Atualizar base (precedentes qualificados)",
        insumo_label="Pasta com os CSVs NUGEPNAC (mantê-los em stj/)",
        insumo_default=str(L.STJ_DIR), insumo_is_dir=True,
        montar_comandos=montar, verificar_comando=verificar, icone=ICONE,
    )
