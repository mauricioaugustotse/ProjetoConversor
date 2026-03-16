from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

import IDP_relatorio_pdf_to_csv_rag as idp


PDF_PATH = Path(__file__).resolve().parents[1] / idp.DEFAULT_INPUT_NAME


@lru_cache(maxsize=1)
def _load_rows():
    if not PDF_PATH.exists():
        pytest.skip("PDF do relatório IDP não encontrado no repositório.")
    if idp.fitz is None:
        pytest.skip("PyMuPDF não está disponível no ambiente de teste.")
    return idp.extract_document_rows(str(PDF_PATH))


def test_merge_multiline_headings_reassembles_broken_title():
    lines = [
        idp.PageLine(7, "1.1 INTEGRIDADE DA INFORMAÇÃO E CORREGULA-", 56.7, 51.2, 15.0, "Conthrax-SemiBold", True, False),
        idp.PageLine(7, "ÇÃO NO CONTEXTO ELEITORAL DIGITAL", 56.7, 75.2, 15.0, "Conthrax-SemiBold", True, False),
        idp.PageLine(7, "A transformação do ambiente informacional.", 92.7, 110.1, 11.0, "ProximaNova-Regular", False, False),
    ]

    merged = idp.merge_multiline_headings(lines)

    assert len(merged) == 2
    assert merged[0].text == "1.1 INTEGRIDADE DA INFORMAÇÃO E CORREGULAÇÃO NO CONTEXTO ELEITORAL DIGITAL"


def test_merge_text_fragments_removes_soft_hyphenation():
    text = ""
    text = idp.merge_text_fragments(text, "corregula-")
    text = idp.merge_text_fragments(text, "ção normativa")

    assert text == "corregulação normativa"


def test_split_text_with_overlap_creates_multiple_chunks():
    sentence = "A integridade informacional exige métricas objetivas e transparência auditável."
    text = " ".join([sentence] * 80)

    chunks = idp.split_text_with_overlap(text, max_chars=600, min_chars=200, overlap_chars=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 700 for chunk in chunks)
    assert chunks[1][:40] in chunks[0][-120:]


def test_live_pdf_smoke_rows_have_required_rag_fields():
    rows = _load_rows()

    assert rows
    assert any(row["parte"].startswith("PARTE I") for row in rows)
    assert any(row["parte"].startswith("PARTE IV") for row in rows)
    assert {"Bloco 1", "Bloco 2", "Bloco 3", "Bloco 4"}.issubset(
        {row["bloco_tabela"] for row in rows if row["tipo_chunk"] == "tabela_recomendacao"}
    )
    assert all("SUMÁRIO" not in row["texto_limpo"] for row in rows)

    rag_rows = [row for row in rows if row["incluir_no_rag"]]
    assert rag_rows
    assert all(row["titulo_chunk"] for row in rag_rows)
    assert all(row["texto_limpo"] for row in rag_rows)
    assert all(row["texto_rag"] for row in rag_rows)
    assert all(row["pagina_inicial"] for row in rag_rows)
    assert all(row["pagina_final"] for row in rag_rows)


def test_live_pdf_covers_key_ranges_and_reference_chunks():
    rows = _load_rows()

    assert any(
        row["tipo_chunk"] == "narrativo"
        and row["pagina_inicial"] <= 7 <= row["pagina_final"]
        and "1.1" in row["caminho_hierarquico"]
        for row in rows
    )
    assert any(
        row["tipo_chunk"] == "narrativo"
        and row["pagina_inicial"] <= 50 <= row["pagina_final"]
        and "2.1.1" in row["caminho_hierarquico"]
        for row in rows
    )
    assert any(
        row["tipo_chunk"] == "tabela_recomendacao"
        and row["pagina_inicial"] <= 273 <= row["pagina_final"]
        and row["bloco_tabela"] == "Bloco 3"
        for row in rows
    )
    assert any(
        row["tipo_chunk"] == "narrativo"
        and row["pagina_inicial"] <= 277 <= row["pagina_final"]
        and "4.1" in row["caminho_hierarquico"]
        for row in rows
    )
    assert any(row["tipo_chunk"] == "referencia_bibliografica" and not row["incluir_no_rag"] for row in rows)


def test_write_rows_to_csv_round_trip(tmp_path):
    rows = _load_rows()
    output = tmp_path / "idp_rag.csv"

    idp.write_rows_to_csv(str(output), rows)

    content = output.read_text(encoding="utf-8-sig")
    assert "titulo_chunk" in content
    assert "texto_rag" in content
