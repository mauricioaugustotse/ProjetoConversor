# -*- coding: utf-8 -*-
"""Metadados de identificação do documento (campos que não vêm do Notion)."""
from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class MetaDocumento:
    ano: int
    vocativo: str = config.VOCATIVO_DEFAULT
    consultor: str = config.CONSULTOR_NOME
    consultor_cargo: str = config.CONSULTOR_CARGO
    sisconle: str = ""                    # ex.: "2026-XXXX" (em branco => placeholder)
    deputado_nome: str = ""               # solicitante; vazio => placeholder
    data_fecho_it: str = ""               # por extenso; vazio => placeholder na assinatura
    data_fecho_prop: str = ""             # por extenso; vazio => lacuna "      de            "

    # ---- derivados ----
    @property
    def sisconle_txt(self) -> str:
        return self.sisconle.strip() or f"{self.ano}-____"

    @property
    def solicitante_capa(self) -> str:
        nome = self.deputado_nome.strip()
        return f"Deputado(a) {nome}" if nome else "[NOME DO(A) SOLICITANTE]"

    @property
    def autor_linhas(self) -> list:
        return [self.consultor, config.CONSULTOR_CARGO_CAPA, config.CONSULTOR_AREA]

    @property
    def autoria_prop(self) -> str:
        nome = self.deputado_nome.strip()
        return f"(Do(a) Sr.(a) {nome or '[NOME DO(A) AUTOR(A)]'})"

    @property
    def assinatura_prop(self) -> str:
        nome = self.deputado_nome.strip()
        return f"Deputado(a) {nome or '[NOME DO(A) AUTOR(A)]'}"

    @property
    def fecho_it_txt(self) -> str:
        data = self.data_fecho_it.strip() or "____ de __________ de %d" % self.ano
        return f"{config.LOCAL_FECHO_IT}, em {data}."

    def fecho_prop_txt(self, local: str) -> str:
        data = self.data_fecho_prop.strip() or "        de                      de %d" % self.ano
        return f"{local}, em {data}."
