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
    def genero(self) -> str:
        """Gênero do parlamentar — "M", "F" ou "" (neutro) — inferido do vocativo
        escolhido na GUI: "Senhor Deputado," ⇒ M; "Senhora Deputada," ⇒ F;
        "Senhor(a) Deputado(a)," ⇒ neutro (mantém as formas com "(a)")."""
        v = (self.vocativo or "").lower()
        if "(a)" in v:
            return ""
        if "deputada" in v or "senhora" in v:
            return "F"
        if "deputado" in v or "senhor" in v:
            return "M"
        return ""

    @property
    def tratamento_deputado(self) -> str:
        """"Deputado"/"Deputada"/"Deputado(a)" conforme o gênero do vocativo."""
        return {"M": "Deputado", "F": "Deputada"}.get(self.genero, "Deputado(a)")

    @property
    def solicitante_capa(self) -> str:
        nome = self.deputado_nome.strip()
        return f"{self.tratamento_deputado} {nome}" if nome else "[NOME DO(A) SOLICITANTE]"

    @property
    def autor_linhas(self) -> list:
        return [self.consultor, config.CONSULTOR_CARGO_CAPA, config.CONSULTOR_AREA]

    @property
    def autoria_prop(self) -> str:
        nome = self.deputado_nome.strip()
        artigo = {"M": "Do", "F": "Da"}.get(self.genero, "Do(a)")
        senhor = {"M": "Sr.", "F": "Sra."}.get(self.genero, "Sr.(a)")
        return f"({artigo} {senhor} {nome or '[NOME DO(A) AUTOR(A)]'})"

    @property
    def assinatura_prop(self) -> str:
        nome = self.deputado_nome.strip()
        return f"{self.tratamento_deputado} {nome or '[NOME DO(A) AUTOR(A)]'}"

    @property
    def fecho_it_txt(self) -> str:
        data = self.data_fecho_it.strip() or "____ de __________ de %d" % self.ano
        return f"{config.LOCAL_FECHO_IT}, em {data}."

    def fecho_prop_txt(self, local: str) -> str:
        # lacunas de dia/mês com 8 e 21 espaços após "em" — idêntico ao modelo
        # oficial da Câmara (o parágrafo precisa de xml:space="preserve", ver _p)
        data = self.data_fecho_prop.strip() or "       de                     de %d" % self.ano
        return f"{local}, em {data}."
