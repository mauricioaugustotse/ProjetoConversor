from NOTION_relatoriodeIA_v2 import (
    CaseAnalysis,
    CaseRecord,
    HEADING_ACCEPTABLE_SCORE,
    build_publishable_case_pairs,
    build_published_summary_rows,
    build_published_toggle_blocks,
    build_semantic_microheadline,
    heading_summary_looks_incomplete,
    is_publishable_analysis,
    looks_generic_case_heading_summary,
    normalize_heading_candidate,
    score_heading_candidate,
    select_best_heading_summary,
)


def make_case(**overrides):
    payload = {
        "case_id": "case-1",
        "page_id": "page-1",
        "source_url": "https://www.notion.so/317721955c64801e98bee86ccfdf8b17",
        "data_decisao": "2026-02-03",
        "numero_unico": "0600086-02.2021.6.10.0000",
        "numero_processo": "",
        "ano_eleicao": "2024",
        "sigla_classe": "AREspEl",
        "descricao_classe": "",
        "sigla_uf": "MA",
        "nome_municipio": "SÃO LUÍS",
        "descricao_tipo_decisao": "acórdão",
        "assuntos": [],
        "partes": ["UNIÃO BRASIL"],
        "partidos": ["UNIÃO BRASIL"],
        "relator": "Ministro",
        "advogados": [],
        "resultado": [],
        "tema": "",
        "punchline": "",
        "texto_decisao": "",
        "noticias": [],
    }
    payload.update(overrides)
    return CaseRecord(**payload)


def make_analysis(**overrides):
    payload = {
        "case_id": "case-1",
        "title": "",
        "relevance_score": 9,
        "display_score": 9,
        "risk_level": "alto",
        "includes_public_figure": False,
        "includes_party": True,
        "public_figures": [],
        "parties": ["UNIÃO BRASIL"],
        "lawyers_signal": "",
        "what_happened": "",
        "legal_grounds": "",
        "consequence": "",
        "strategic_comment": "",
        "why_relevant": "",
        "source_notes": [],
        "page_id": "page-1",
    }
    payload.update(overrides)
    return CaseAnalysis(**payload)


def test_build_semantic_microheadline_keeps_closed_accounts_title():
    text = (
        "Prestação de contas partidárias anuais de diretório estadual – exercício financeiro de 2020 – "
        "recebimento de recursos de fonte vedada, uso irregular do Fundo Partidário, falhas na comprovação documental"
    )
    headline = build_semantic_microheadline(text)
    assert headline == (
        "Prestação de contas partidárias anuais de diretório estadual (exercício 2020), "
        "fonte vedada, uso irregular do Fundo Partidário"
    )
    assert not heading_summary_looks_incomplete(headline)


def test_build_semantic_microheadline_converts_temporal_marker_to_parentheses():
    text = "Fraude à cota de gênero – Eleições de 2024 – AIJE"
    headline = build_semantic_microheadline(text)
    assert headline == "Fraude à cota de gênero (Eleições 2024), AIJE"


def test_build_semantic_microheadline_preserves_essential_legal_reference():
    text = "Captação ilícita de sufrágio em AIJE (art. 41-A da Lei 9.504/1997)"
    headline = build_semantic_microheadline(text)
    assert "art. 41-A" in headline
    assert "AIJE" in headline
    assert not heading_summary_looks_incomplete(headline)


def test_build_semantic_microheadline_removes_nonessential_legal_tail():
    text = "Prestação de contas eleitorais: recebimento de doação em espécie acima do limite previsto no art. 21"
    headline = build_semantic_microheadline(text)
    assert "art. 21" not in headline
    assert headline == "Prestação de contas eleitorais, doação em espécie acima do limite legal"


def test_normalize_heading_candidate_does_not_remove_word_prefix_for_party_abbreviation():
    normalized = normalize_heading_candidate(
        "PR | Prestação de contas partidárias anuais de diretório estadual",
        extra_markers=["PR"],
    )
    assert normalized == "Prestação de contas partidárias anuais de diretório estadual"


def test_select_best_heading_summary_prefers_theme_over_narrative_ai_title():
    case = make_case(
        tema="Prestação de contas partidárias anuais de diretório estadual – exercício financeiro de 2020 – recebimento de recursos de fonte vedada, uso irregular do Fundo Partidário",
        punchline="O relator negou seguimento ao agravo e manteve a desaprovação das contas.",
    )
    analysis = make_analysis(
        title="Em decisão monocrática, negou seguimento ao agravo interposto contra acórdão regional.",
        what_happened="Em decisão monocrática, negou seguimento ao agravo interposto contra acórdão regional.",
    )
    summary = select_best_heading_summary(case, analysis, extra_markers=["UNIÃO BRASIL"])
    assert summary.startswith("Prestação de contas partidárias anuais de diretório estadual")
    assert "decisão monocrática" not in summary.casefold()
    assert score_heading_candidate(summary) >= HEADING_ACCEPTABLE_SCORE


def test_looks_generic_case_heading_summary_flags_bare_subjects():
    assert looks_generic_case_heading_summary("Obrigatoriedade")
    assert looks_generic_case_heading_summary("Prestação de contas eleitorais")
    assert looks_generic_case_heading_summary("Eleições 2024")
    assert not looks_generic_case_heading_summary(
        "Prestação de contas eleitorais, uso de recursos de origem não identificada (RONI)"
    )


def test_select_best_heading_summary_avoids_generic_theme_and_narrative_punchline():
    case = make_case(
        tema="Prestação de contas eleitorais",
        punchline="Tese firmada, a ausência do relatório semanal impede a verificação da destinação das despesas com combustíveis.",
    )
    analysis = make_analysis(
        title="Prestação de contas eleitorais",
        what_happened=(
            "Ausência de relatório semanal de combustíveis custeados com recursos públicos, "
            "o que impediu o controle da destinação das despesas."
        ),
    )
    summary = select_best_heading_summary(case, analysis)
    assert "combust" in summary.casefold()
    assert "tese firmada" not in summary.casefold()
    assert not looks_generic_case_heading_summary(summary)


def test_build_semantic_microheadline_shortens_accounts_limit_case():
    text = (
        "Prestação de contas de campanha – extrapolação do limite global de gastos nas eleições de 2024 "
        "e aplicação de multa vinculada de 100% sobre o valor excedente"
    )
    headline = build_semantic_microheadline(text)
    assert headline == "Prestação de contas de campanha, extrapolação do limite de gastos, multa de 100% sobre o excedente"
    assert not looks_generic_case_heading_summary(headline)


def test_build_semantic_microheadline_promotes_better_primary_when_legal_stub_is_generic():
    text = (
        "Aplicação do art. 73, V, da Lei nº 9.504/1997 — demissão sem justa causa de servidores temporários "
        "no período entre as eleições e a posse e valoração da sanção pecuniária"
    )
    headline = build_semantic_microheadline(text)
    assert headline == (
        "Aplicação do art. 73, V, da Lei nº 9.504/1997, demissão sem justa causa de temporários "
        "entre eleições e posse, sanção pecuniária"
    )
    assert not looks_generic_case_heading_summary(headline)


def test_is_publishable_analysis_accepts_only_critico_e_alto():
    assert is_publishable_analysis(make_analysis(risk_level="critico"))
    assert is_publishable_analysis(make_analysis(risk_level="alto"))
    assert not is_publishable_analysis(make_analysis(risk_level="medio"))
    assert not is_publishable_analysis(make_analysis(risk_level="baixo"))


def test_build_publishable_case_pairs_orders_by_prominence_inside_same_score():
    case_public = make_case(case_id="case-public", page_id="page-public", partes=["Prefeito João"], partidos=["MDB/Nacional"])
    analysis_public = make_analysis(
        case_id="case-public",
        page_id="page-public",
        relevance_score=10,
        display_score=10,
        risk_level="critico",
        includes_public_figure=True,
        public_figures=["JOÃO"],
        parties=["MDB/Nacional"],
        what_happened="Cassação de mandato de prefeito por AIJE.",
        consequence="Perda de mandato e inelegibilidade.",
    )
    case_accounts = make_case(case_id="case-accounts", page_id="page-accounts", partes=["Diretório estadual"], partidos=["MDB/MA"])
    analysis_accounts = make_analysis(
        case_id="case-accounts",
        page_id="page-accounts",
        relevance_score=10,
        display_score=10,
        risk_level="critico",
        includes_public_figure=False,
        public_figures=[],
        parties=["MDB/MA"],
        what_happened="Prestação de contas partidárias com recursos do Fundo Partidário.",
        consequence="Ressarcimento ao erário.",
    )
    medium_case = make_case(case_id="case-medium", page_id="page-medium")
    medium_analysis = make_analysis(case_id="case-medium", page_id="page-medium", risk_level="medio", relevance_score=6, display_score=6)

    pairs = build_publishable_case_pairs(
        [case_accounts, case_public, medium_case],
        [analysis_accounts, analysis_public, medium_analysis],
    )

    assert [case.case_id for case, _analysis in pairs] == ["case-public", "case-accounts"]


def test_build_publishable_case_pairs_keeps_critico_before_alto():
    case_critical = make_case(case_id="case-critical", page_id="page-critical", partes=["Diretório estadual"], partidos=["PSD/MA"])
    analysis_critical = make_analysis(
        case_id="case-critical",
        page_id="page-critical",
        relevance_score=9,
        display_score=9,
        risk_level="critico",
        includes_party=True,
        parties=["PSD/MA"],
        what_happened="Prestação de contas partidárias com Fundo Partidário.",
        consequence="Ressarcimento ao Tesouro.",
    )
    case_high = make_case(case_id="case-high", page_id="page-high", partes=["Prefeito João"], partidos=["MDB/Nacional"])
    analysis_high = make_analysis(
        case_id="case-high",
        page_id="page-high",
        relevance_score=10,
        display_score=10,
        risk_level="alto",
        includes_public_figure=True,
        public_figures=["JOÃO"],
        parties=["MDB/Nacional"],
        what_happened="Cassação de mandato de prefeito por AIJE.",
        consequence="Perda de mandato e inelegibilidade.",
    )

    pairs = build_publishable_case_pairs(
        [case_high, case_critical],
        [analysis_high, analysis_critical],
    )

    assert [case.case_id for case, _analysis in pairs] == ["case-critical", "case-high"]


def test_published_rows_and_toggles_are_aligned_and_exclude_medium():
    case_1 = make_case(case_id="case-1", page_id="page-1", partes=["Prefeito João"], partidos=["MDB/Nacional"])
    analysis_1 = make_analysis(
        case_id="case-1",
        page_id="page-1",
        relevance_score=10,
        display_score=10,
        risk_level="critico",
        includes_public_figure=True,
        public_figures=["JOÃO"],
        parties=["MDB/Nacional"],
        what_happened="Cassação de mandato.",
        consequence="Perda de mandato.",
    )
    case_2 = make_case(case_id="case-2", page_id="page-2", partes=["Diretório estadual"], partidos=["MDB/MA"])
    analysis_2 = make_analysis(
        case_id="case-2",
        page_id="page-2",
        relevance_score=8,
        display_score=8,
        risk_level="alto",
        includes_party=True,
        parties=["MDB/MA"],
        what_happened="Prestação de contas com FEFC.",
        consequence="Devolução ao Tesouro.",
    )
    case_3 = make_case(case_id="case-3", page_id="page-3")
    analysis_3 = make_analysis(case_id="case-3", page_id="page-3", risk_level="medio", relevance_score=6, display_score=6)

    rows = build_published_summary_rows([case_1, case_2, case_3], [analysis_1, analysis_2, analysis_3])
    toggles = build_published_toggle_blocks([case_1, case_2, case_3], [analysis_1, analysis_2, analysis_3])

    assert len(rows) == 2
    assert rows[0][0] == "1"
    assert rows[1][0] == "2"
    assert len(toggles) == 2
    assert toggles[0]["toggle"]["rich_text"][0]["text"]["content"].startswith("1.")
    assert toggles[1]["toggle"]["rich_text"][0]["text"]["content"].startswith("2.")
