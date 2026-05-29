from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Codigo_Eleitoral_2026_pdf_to_csv_rag as codigo


PDF_PATH = Path(__file__).resolve().parents[1] / codigo.DEFAULT_INPUT_NAME


@lru_cache(maxsize=1)
def _load_rows_by_norm():
    if not PDF_PATH.exists():
        pytest.skip("PDF Código Eleitoral 2026 anotado não encontrado no repositório.")
    if codigo.fitz is None:
        pytest.skip("PyMuPDF não está disponível no ambiente de teste.")
    return codigo.extract_document_rows_by_norm(str(PDF_PATH))


def test_page_27_note_is_linked_to_art_1_single_paragraph():
    _, rows_by_norm = _load_rows_by_norm()
    rows = rows_by_norm["lei_4737_1965"]

    target = next(row for row in rows if row["dispositivo"] == "Art. 1º, Parágrafo único")

    assert target["tipo_dispositivo"] == "paragrafo"
    assert "O Tribunal Superior Eleitoral expedirá instruções" in target["texto_dispositivo"]
    assert "V. art. 23-A deste código" in target["notas_texto"]
    assert "organização dos partidos políticos" in target["notas_texto"]


def test_page_28_running_header_art_6_does_not_capture_art_5_inciso_iv():
    _, rows_by_norm = _load_rows_by_norm()
    rows = rows_by_norm["lei_4737_1965"]

    art_5_iv = next(row for row in rows if row["dispositivo"] == "Art. 5º, IV")
    art_6 = next(row for row in rows if row["dispositivo"] == "Art. 6º")

    assert "pessoas recolhidas a estabelecimento prisional" in art_5_iv["texto_dispositivo"]
    assert "Inciso IV acrescido pelo art. 40 da Lei n. 15.358/2026" in art_5_iv["notas_texto"]
    assert "pessoas recolhidas a estabelecimento prisional" not in art_6["texto_dispositivo"]
    assert art_6["texto_dispositivo"].startswith("Art. 6º O alistamento")


def test_page_491_federacoes_notes_attach_to_structure_not_neighbors():
    _, rows_by_norm = _load_rows_by_norm()
    rows = rows_by_norm["lei_9504_1997"]

    estrutura = next(row for row in rows if row["tipo_dispositivo"] == "estrutura" and row["dispositivo"] == "DAS FEDERAÇÕES")
    art_6a = next(row for row in rows if row["dispositivo"] == "Art. 6º-A")
    par_5 = next(row for row in rows if row["dispositivo"] == "Art. 6º, §5º")

    assert "Título acrescido pelo art. 2º da Lei n. 14.208/2021" in estrutura["notas_texto"]
    assert "ADI n. 7021" in estrutura["notas_texto"]
    assert "Título acrescido pelo art. 2º da Lei n. 14.208/2021" not in art_6a["notas_texto"]
    assert "Título acrescido pelo art. 2º da Lei n. 14.208/2021" not in par_5["notas_texto"]


def test_page_1313_sumulas_are_separate_norms():
    starts, rows_by_norm = _load_rows_by_norm()
    norm_ids = {start.norm_id for start in starts}

    assert {"sumula_tse_1_cancelada", "sumula_tse_2", "sumula_tse_3"}.issubset(norm_ids)
    assert rows_by_norm["sumula_tse_1_cancelada"][0]["dispositivo"] == "Súmula-TSE n. 1 (cancelada)"
    assert rows_by_norm["sumula_tse_2"][0]["dispositivo"] == "Súmula-TSE n. 2"
    assert rows_by_norm["sumula_tse_3"][0]["dispositivo"] == "Súmula-TSE n. 3"
    assert "Súmula-TSE n. 3" not in rows_by_norm["sumula_tse_2"][0]["texto_dispositivo"]


def test_live_pdf_smoke_generates_norm_csv_rows_with_rag_text():
    starts, rows_by_norm = _load_rows_by_norm()

    assert len(starts) >= 200
    assert len(rows_by_norm) == len(starts)
    assert all(rows for rows in rows_by_norm.values())
    assert all(row["texto_rag"] for rows in rows_by_norm.values() for row in rows)
    assert all("Sumário" not in row["texto_dispositivo"] for rows in rows_by_norm.values() for row in rows)
    assert any(row["qtd_notas"] for rows in rows_by_norm.values() for row in rows)
