# -*- coding: utf-8 -*-
"""Detecta o tipo de proposição legislativa a partir do texto da minuta."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class TipoProposicao:
    sigla: str
    nome_extenso: str          # usado na epígrafe (MAIÚSCULAS)
    preambulo: str             # fórmula de promulgação
    local_fecho: str           # "Sala das Sessões" etc.


# Catálogo dos tipos suportados
PLP = TipoProposicao(
    "PLP", "PROJETO DE LEI COMPLEMENTAR",
    "O Congresso Nacional decreta:", "Sala das Sessões",
)
PL = TipoProposicao(
    "PL", "PROJETO DE LEI",
    "O Congresso Nacional decreta:", "Sala das Sessões",
)
PEC = TipoProposicao(
    "PEC", "PROPOSTA DE EMENDA À CONSTITUIÇÃO",
    "As Mesas da Câmara dos Deputados e do Senado Federal, nos termos do § 3º do art. 60 da "
    "Constituição Federal, promulgam a seguinte emenda ao texto constitucional:",
    "Sala das Sessões",
)
PRC = TipoProposicao(
    "PRC", "PROJETO DE RESOLUÇÃO",
    "A Câmara dos Deputados resolve:", "Sala das Sessões",
)
PDL = TipoProposicao(
    "PDL", "PROJETO DE DECRETO LEGISLATIVO",
    "O Congresso Nacional decreta:", "Sala das Sessões",
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).upper().strip()


def detectar_tipo(*textos: str) -> TipoProposicao:
    """Escolhe o tipo pelo primeiro indício forte encontrado nos textos fornecidos
    (título da página, heading da minuta, epígrafe). Ordem importa: do mais
    específico para o mais genérico."""
    blob = _norm(" || ".join(t for t in textos if t))

    if "EMENDA A CONSTITUICAO" in blob or re.search(r"\bPEC\b", blob):
        return PEC
    if "LEI COMPLEMENTAR" in blob or re.search(r"\bPLP\b", blob):
        return PLP
    if "DECRETO LEGISLATIVO" in blob or re.search(r"\bPDL\b", blob):
        return PDL
    if "RESOLUCAO" in blob or re.search(r"\bPRC\b", blob):
        return PRC
    if "PROJETO DE LEI" in blob or re.search(r"\bPL\b", blob):
        return PL
    # padrão seguro: lei ordinária
    return PL
