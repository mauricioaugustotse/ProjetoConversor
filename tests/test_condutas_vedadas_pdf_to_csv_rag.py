from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

import Condutas_vedadas_pdf_to_csv_rag as condutas


PDF_PATH = Path(__file__).resolve().parents[1] / condutas.DEFAULT_INPUT_NAME


@lru_cache(maxsize=1)
def _load_rows():
    if not PDF_PATH.exists():
        pytest.skip("PDF Condutas Vedadas não encontrado no repositório.")
    if condutas.fitz is None:
        pytest.skip("PyMuPDF não está disponível no ambiente de teste.")
    return condutas.extract_document_rows(str(PDF_PATH))


def test_merge_multiline_headings_reassembles_subsection_title():
    lines = [
        condutas.PageLine(
            41,
            "5.1.7. Propaganda eleitoral em sítios oficiais",
            73.2,
            86.0,
            16.0,
            "Alice-Regular",
            False,
            False,
        ),
        condutas.PageLine(
            41,
            "ou hospedados por órgãos ou entidades da",
            73.2,
            105.2,
            16.0,
            "Alice-Regular",
            False,
            False,
        ),
        condutas.PageLine(
            41,
            "administração pública direta ou indireta",
            73.2,
            124.4,
            16.0,
            "Alice-Regular",
            False,
            False,
        ),
    ]

    merged = condutas.merge_multiline_headings(lines)

    assert len(merged) == 1
    assert (
        merged[0].text
        == "5.1.7. Propaganda eleitoral em sítios oficiais ou hospedados por órgãos ou entidades da administração pública direta ou indireta"
    )


def test_find_chapter_number_detects_number_on_chapter_page():
    raw_lines = [
        condutas.PageLine(9, "Capítulo", 523.7, 252.7, 12.0, "Rawline-Regular", False, False),
        condutas.PageLine(9, "01", 523.5, 256.0, 50.0, "Alice-Regular", False, False),
    ]

    assert condutas.find_chapter_number(raw_lines) == "01"


def test_build_text_from_lines_keeps_wrapped_observation_in_single_paragraph():
    lines = [
        condutas.PageLine(41, "OBSERVAÇÕES:", 84.3, 405.1, 11.0, "Rawline-Bold", True, False),
        condutas.PageLine(41, "Link em página oficial: Para o TSE", 103.5, 429.8, 11.0, "Rawline-Italic", False, True),
        condutas.PageLine(41, "a utilização de página mantida por órgão", 89.4, 446.6, 11.1, "Rawline-Italic", False, True),
        condutas.PageLine(41, "da administração pública do município", 89.4, 463.4, 11.1, "Rawline-Italic", False, True),
    ]

    text = condutas.build_text_from_lines(lines)

    assert "OBSERVAÇÕES:\n\nLink em página oficial" in text
    assert "órgão da administração pública do município" in text
    assert text.count("\n\n") == 1


def test_live_pdf_smoke_rows_have_required_rag_fields():
    rows = _load_rows()

    assert rows
    assert any(row["parte"] == "Apresentação" for row in rows)
    assert any(row["parte"].startswith("Capítulo 05") for row in rows)
    assert any(row["parte"].startswith("Capítulo 09") for row in rows)
    assert all("Sumário" not in row["texto_limpo"] for row in rows)
    assert all("Expediente" not in row["texto_limpo"] for row in rows)
    assert all(row["tipo_chunk"] == "narrativo" for row in rows)
    assert all(row["titulo_chunk"] for row in rows)
    assert all(row["texto_limpo"] for row in rows)
    assert all(row["texto_rag"] for row in rows)
    assert all(row["pagina_inicial"] for row in rows)
    assert all(row["pagina_final"] for row in rows)


def test_live_pdf_covers_key_ranges_and_structural_markers():
    rows = _load_rows()

    assert any(
        row["pagina_inicial"] <= 41 <= row["pagina_final"]
        and "5.1.7" in row["caminho_hierarquico"]
        for row in rows
    )
    assert any(
        row["pagina_inicial"] <= 61 <= row["pagina_final"]
        and "1º de janeiro" in row["caminho_hierarquico"]
        for row in rows
    )
    assert any(
        row["pagina_inicial"] <= 67 <= row["pagina_final"]
        and "7.2" in row["caminho_hierarquico"]
        for row in rows
    )
    assert any(
        row["pagina_inicial"] <= 75 <= row["pagina_final"]
        and row["parte"].startswith("Capítulo 08")
        for row in rows
    )
    assert any(
        row["pagina_inicial"] <= 79 <= row["pagina_final"]
        and row["parte"].startswith("Capítulo 09")
        for row in rows
    )


def test_write_rows_to_csv_round_trip(tmp_path):
    rows = _load_rows()
    output = tmp_path / "condutas_rag.csv"

    condutas.write_rows_to_csv(str(output), rows)

    content = output.read_text(encoding="utf-8-sig")
    assert "titulo_chunk" in content
    assert "texto_rag" in content
