from collections import Counter
from dataclasses import replace

import NOTION_relatoriodeIA_v2 as notion


def _make_case() -> notion.CaseRecord:
    return notion.CaseRecord(
        case_id="case-1",
        page_id="page-1",
        source_url="https://www.notion.so/324721955c6480d09adae26a3def3ade",
        data_decisao="2026-02-27",
        numero_unico="0600001-00.2024.6.00.0000",
        numero_processo="",
        ano_eleicao="2024",
        sigla_classe="AIJE",
        descricao_classe="",
        sigla_uf="MA",
        nome_municipio="SAO LUIS",
        descricao_tipo_decisao="acórdão",
        assuntos=["cassação de diploma"],
        partes=["Prefeito"],
        partidos=["MDB"],
        relator="Ministro",
        advogados=[],
        resultado=["deferido"],
        tema="Cassação de diploma em eleição municipal",
        punchline="Mantida decisão com impacto direto sobre o mandato.",
        texto_decisao="Texto integral da decisão.",
        noticias=[],
    )


def _make_analysis() -> notion.CaseAnalysis:
    return notion.CaseAnalysis(
        case_id="case-1",
        page_id="page-1",
        title="Cassação de diploma em eleição municipal",
        relevance_score=9,
        display_score=9,
        risk_level="alto",
        includes_public_figure=True,
        includes_party=True,
        public_figures=["Prefeito"],
        parties=["MDB"],
        lawyers_signal="Sem recorrência material.",
        what_happened="O Tribunal manteve decisão com impacto direto sobre o mandato.",
        legal_grounds="Abuso de poder político e prova documental.",
        consequence="Risco de vacância e de eleição suplementar.",
        strategic_comment="Sinalização de reforço do controle sobre estabilidade do mandato.",
        why_relevant="Afeta diretamente a permanência no cargo.",
        source_notes=[],
    )


def test_summarize_report_compacts_long_overview_callout_from_openai(monkeypatch):
    full_text = (
        "No intervalo de 23/02/2026 a 27/02/2026, o contencioso eleitoral monitorado mostrou dois vetores dominantes: "
        "(i) forte incidência de litígios cassatórios e cautelares em eleições municipais de 2024, com impacto direto "
        "sobre manutenção ou retorno a mandatos e risco de eleições suplementares; e (ii) consolidação de uma linha "
        "rigorosa em contas eleitorais e partidárias, sobretudo quanto ao uso do FEFC, circulação de recursos entre "
        "diretórios e fornecedores e exigência de lastro documental completo para afastar devolução ao erário."
    )

    monkeypatch.setattr(notion, "OPENAI_CFG", object())
    monkeypatch.setattr(notion, "OPENAI_SESSION", object())
    monkeypatch.setattr(
        notion,
        "openai_json_call",
        lambda *args, **kwargs: {
            "overview_callout": full_text,
            "executive_highlights": ["Destaque 1"],
            "party_alerts": [],
            "lawyer_signals": [],
            "watchpoints": [],
            "closing_note": "Fechamento.",
        },
    )

    summary = notion.summarize_report(
        [_make_case()],
        [_make_analysis()],
        party_counter=Counter({"MDB": 1}),
        lawyer_counter=Counter(),
        start_iso="2026-02-23",
        end_iso="2026-02-27",
    )

    assert len(full_text) > 420
    assert len(summary.overview_callout) <= notion.MAX_OVERVIEW_CALLOUT_CHARS
    assert not notion.summary_text_looks_incomplete(summary.overview_callout)
    assert summary.overview_callout.endswith(".")


def test_summarize_report_compacts_long_openai_executive_highlight(monkeypatch):
    long_highlight = (
        "O TSE consolidou linha rigorosa em registros de candidatura e inelegibilidades, com ênfase na exigência de "
        "suspensão judicial específica para afastar efeitos de condenações por improbidade e AIJE, o que aumenta a "
        "relevância estratégica do contencioso preventivo antes da fase final do registro e reduz a utilidade de "
        "medidas reativas apresentadas apenas às vésperas da diplomação."
    )

    monkeypatch.setattr(notion, "OPENAI_CFG", object())
    monkeypatch.setattr(notion, "OPENAI_SESSION", object())
    monkeypatch.setattr(
        notion,
        "openai_json_call",
        lambda *args, **kwargs: {
            "overview_callout": "Visão geral.",
            "executive_highlights": [long_highlight],
            "party_alerts": [],
            "lawyer_signals": [],
            "watchpoints": [],
            "closing_note": "Fechamento.",
        },
    )

    summary = notion.summarize_report(
        [_make_case()],
        [_make_analysis()],
        party_counter=Counter({"MDB": 1}),
        lawyer_counter=Counter(),
        start_iso="2026-02-23",
        end_iso="2026-02-27",
    )

    assert len(long_highlight) > notion.MAX_EXECUTIVE_BULLET_CHARS
    assert len(summary.executive_highlights[0]) <= notion.MAX_EXECUTIVE_BULLET_CHARS
    assert not notion.summary_text_looks_incomplete(summary.executive_highlights[0])


def test_build_callout_block_preserves_full_long_text_across_rich_text_chunks():
    full_text = " ".join(f"trecho-{idx:04d}" for idx in range(600))

    block = notion.build_callout_block(full_text)

    assert notion._plain_rich_text(block["callout"]["rich_text"]) == full_text


def test_compact_report_field_text_avoids_abrupt_fragments():
    relevance = (
        "A relevância é elevada por envolver ação relativa ao Senado Federal e por reafirmar uma regra processual "
        "com efeito institucional sobre julgamentos eleitorais cassatórios. A decisão orienta TREs a não desempatarem, "
        "por voto de qualidade, ações capazes de gerar cassação de mandato."
    )
    facts = (
        "O TSE examinou embargos de declaração contra decisão que reconhecera erro de procedimento no TRE/PA. "
        "O vício decorreu do uso de voto de qualidade pelo presidente da Corte Regional, após empate em embargos "
        "de declaração em ações com potencial de cassação. A Corte manteve a nulidade do julgamento regional."
    )

    compact_relevance = notion._compact_report_field_text(relevance, 180)
    compact_facts = notion._compact_report_field_text(facts, 240)

    assert compact_relevance.endswith("cassatórios.")
    assert compact_facts.endswith("cassação.")
    assert not compact_relevance.endswith((" A", " a", " após", " com"))
    assert not compact_facts.endswith((" A", " a", " após", " com"))


def test_compact_report_field_text_removes_interrupted_effect_tail():
    effect = (
        "monocrática. O Ministro relator indeferiu o pedido de atribuição de efeito suspensivo ao agravo em "
        "recurso especial por ausência de plausibilidade recursal, ante a falta de impugnação específica dos "
        "fundamentos e a incidência de óbices sumulares do TSE, vedado o reexame de fatos e provas. "
        "Com isso, permaneceram"
    )

    compact_effect = notion._compact_report_field_text(effect, notion.MAX_TOGGLE_CONSEQUENCE_CHARS)

    assert compact_effect.endswith("reexame de fatos e provas.")
    assert "permaneceram" not in compact_effect
    assert not notion.summary_text_looks_incomplete(compact_effect)


def test_finalize_report_summary_replaces_generic_count_alerts_with_material_alerts():
    case_1 = notion.CaseRecord(
        case_id="case-1",
        page_id="page-1",
        source_url="https://www.notion.so/page-1",
        data_decisao="2026-02-27",
        numero_unico="0600001-00.2024.6.00.0000",
        numero_processo="",
        ano_eleicao="2024",
        sigla_classe="AIJE",
        descricao_classe="",
        sigla_uf="MA",
        nome_municipio="SAO LUIS",
        descricao_tipo_decisao="acórdão",
        assuntos=["cassação de diploma"],
        partes=["Prefeito"],
        partidos=["MDB"],
        relator="Ministro 1",
        advogados=["Banca Alfa"],
        resultado=["deferido"],
        tema="Cassação de diploma em eleição municipal",
        punchline="Mantida decisão com impacto direto sobre o mandato.",
        texto_decisao="Texto integral 1.",
        noticias=[],
    )
    case_2 = notion.CaseRecord(
        case_id="case-2",
        page_id="page-2",
        source_url="https://www.notion.so/page-2",
        data_decisao="2026-02-28",
        numero_unico="0600002-00.2024.6.00.0000",
        numero_processo="",
        ano_eleicao="2024",
        sigla_classe="PCE",
        descricao_classe="",
        sigla_uf="MA",
        nome_municipio="SAO LUIS",
        descricao_tipo_decisao="acórdão",
        assuntos=["uso irregular do FEFC"],
        partes=["Diretório partidário"],
        partidos=["MDB"],
        relator="Ministro 2",
        advogados=["Banca Alfa"],
        resultado=["parcialmente deferido"],
        tema="Uso irregular do FEFC em prestação de contas",
        punchline="Persistiu risco de devolução ao erário.",
        texto_decisao="Texto integral 2.",
        noticias=[],
    )
    analysis_1 = notion.CaseAnalysis(
        case_id="case-1",
        page_id="page-1",
        title="Cassação de diploma em eleição municipal",
        relevance_score=9,
        display_score=9,
        risk_level="alto",
        includes_public_figure=True,
        includes_party=True,
        public_figures=["Prefeito"],
        parties=["MDB"],
        lawyers_signal="Banca Alfa aparece em contencioso de mandato.",
        what_happened="O Tribunal manteve decisão com impacto direto sobre o mandato.",
        legal_grounds="Abuso de poder político e prova documental.",
        consequence="Risco de vacância e de eleição suplementar.",
        strategic_comment="Sinaliza endurecimento em litígios de mandato nas eleições de 2024.",
        why_relevant="Afeta diretamente a permanência no cargo.",
        source_notes=[],
    )
    analysis_2 = notion.CaseAnalysis(
        case_id="case-2",
        page_id="page-2",
        title="Uso irregular do FEFC em prestação de contas",
        relevance_score=8,
        display_score=8,
        risk_level="alto",
        includes_public_figure=False,
        includes_party=True,
        public_figures=[],
        parties=["MDB"],
        lawyers_signal="Banca Alfa reaparece em tema sensível de contas.",
        what_happened="O Tribunal cobrou lastro documental e manteve glosa relevante.",
        legal_grounds="Uso irregular do FEFC e insuficiência documental.",
        consequence="Pode gerar devolução ao erário e exposição institucional do diretório.",
        strategic_comment="Reforça linha rigorosa em contas partidárias com impacto político imediato.",
        why_relevant="Atinge financiamento e governança partidária.",
        source_notes=[],
    )
    raw_summary = notion.ReportSummary(
        overview_callout="Visão geral.",
        executive_highlights=["Destaque curto."],
        party_alerts=["MDB: citado em 2 processo(s) do período."],
        lawyer_signals=["Banca Alfa: atua em 2 processo(s) no período."],
        watchpoints=["Cassação de diploma em eleição municipal: prioridade 9/10."],
        closing_note="Fechamento.",
    )

    final_summary = notion.finalize_report_summary(
        raw_summary,
        [case_1, case_2],
        [analysis_1, analysis_2],
        party_counter=Counter({"MDB": 2}),
        lawyer_counter=Counter({"Banca Alfa": 2}),
    )

    assert final_summary.party_alerts
    assert "citado em 2 processo(s)" not in final_summary.party_alerts[0]
    assert final_summary.lawyer_signals
    assert "atua em 2 processo(s)" not in final_summary.lawyer_signals[0]
    assert final_summary.watchpoints
    assert "prioridade 9/10" not in final_summary.watchpoints[0]
    assert any("vacância" in item or "eleição suplementar" in item for item in final_summary.watchpoints)


def test_finalize_report_summary_trims_incomplete_summary_tails_from_saved_openai_output():
    raw_summary = notion.ReportSummary(
        overview_callout="Visão geral.",
        executive_highlights=[
            (
                "O TSE manteve indeferimentos de registros de prefeito por improbidade com dano ao erário e "
                "enriquecimento ilícito e por condenação colegiada em AIJE, elevando o risco de candidaturas "
                "sub judice sem suspensão eficaz e"
            ),
            (
                "Nas contas do Cidadania, o TSE manteve irregularidades e devolução ao erário por prova insuficiente "
                "de gastos do Fundo Partidário, reforçando padrão estrito de comprovação documental em rubricas "
                "sensíveis como"
            ),
        ],
        party_alerts=[
            (
                "MDB/TO aparece em precedente de AIJE por fraude à cota de gênero que superou tese de decadência, "
                "aumentando o risco contencioso de legendas e chapas em casos análogos e exigindo prevenção "
                "probatória e revisão de"
            )
        ],
        lawyer_signals=[],
        watchpoints=[
            (
                "O conjunto dos julgados sugere maior valor estratégico de prevenção documental e processual "
                "pré-registro do que de litigância corretiva após indeferimento, sobretudo em majoritárias "
                "municipais e contas partidárias e"
            )
        ],
        closing_note="Fechamento.",
    )

    final_summary = notion.finalize_report_summary(
        raw_summary,
        [_make_case()],
        [_make_analysis()],
        party_counter=Counter({"MDB": 1}),
        lawyer_counter=Counter(),
    )

    assert final_summary.executive_highlights[0].endswith("sem suspensão eficaz")
    assert final_summary.executive_highlights[1].endswith("rubricas sensíveis")
    assert final_summary.party_alerts[0].endswith("prevenção probatória e revisão")
    assert final_summary.watchpoints[0].endswith("contas partidárias")
    assert not final_summary.executive_highlights[0].endswith(" e")
    assert not final_summary.executive_highlights[1].endswith(" como")


def test_report_summary_has_incomplete_items_flags_broken_saved_summary():
    summary = notion.ReportSummary(
        overview_callout="Visão geral.",
        executive_highlights=["Alerta material com fechamento inadequado e"],
        party_alerts=[],
        lawyer_signals=[],
        watchpoints=[],
        closing_note="Fechamento.",
    )

    assert notion.report_summary_has_incomplete_items(summary) is True


def test_build_strategic_alert_section_items_limits_combined_output():
    summary = notion.ReportSummary(
        overview_callout="Visão geral.",
        executive_highlights=[],
        party_alerts=[
            "MDB reaparece em 2 casos reportáveis, com exposição institucional relevante.",
            "PSD concentra litígios com risco de cassação municipal.",
        ],
        lawyer_signals=[
            "Banca Alfa atua em 2 casos reportáveis, ambos com impacto sobre mandato.",
            "Banca Beta reaparece em contas com risco de devolução ao erário.",
        ],
        watchpoints=[
            "Cassação de diploma em eleição municipal: risco de vacância e de eleição suplementar.",
            "Uso irregular do FEFC em prestação de contas: risco de devolução ao erário.",
            "Abuso de poder com prefeito no polo passivo: exposição elevada para o mandato.",
        ],
        closing_note="Fechamento.",
    )

    items = notion.build_strategic_alert_section_items(summary)

    assert len(items) == notion.MAX_STRATEGIC_ALERT_SECTION_ITEMS
    assert items[:3] == summary.watchpoints[:3]


def test_build_publishable_case_pairs_keeps_national_party_and_filters_local_noise():
    local_case = replace(
        _make_case(),
        case_id="local-case",
        page_id="local-page",
        source_url="https://www.notion.so/local-page",
        sigla_uf="SP",
        nome_municipio="CIDADE LOCAL",
        partes=["Prefeito", "Vereador"],
        partidos=[],
        ano_eleicao="2024",
        tema="Cassação municipal por abuso de poder",
        punchline="Disputa local sem partido ou tese federal explícita.",
    )
    local_analysis = replace(
        _make_analysis(),
        case_id="local-case",
        page_id="local-page",
        relevance_score=9,
        display_score=9,
        risk_level="alto",
        includes_public_figure=True,
        includes_party=False,
        public_figures=["Prefeito"],
        parties=[],
        legal_grounds="AIJE municipal por abuso de poder.",
        consequence="Impacto restrito ao mandato local.",
        strategic_comment="Caso municipal sem repercussão institucional para partido ou alto cargo.",
        why_relevant="Relevante apenas para a disputa local.",
    )
    national_case = replace(
        _make_case(),
        case_id="national-case",
        page_id="national-page",
        source_url="https://www.notion.so/national-page",
        sigla_uf="DF",
        nome_municipio="BRASÍLIA",
        partes=["PARTIDO LIBERAL (PL) - NACIONAL"],
        partidos=["PL/Nacional"],
        ano_eleicao="2022",
        tema="Prestação de contas anual do diretório nacional",
        punchline="Discussão sobre Fundo Partidário e consequência para direção nacional.",
    )
    national_analysis = replace(
        _make_analysis(),
        case_id="national-case",
        page_id="national-page",
        relevance_score=9,
        display_score=9,
        risk_level="alto",
        includes_public_figure=False,
        includes_party=True,
        public_figures=[],
        parties=["PL/Nacional"],
        legal_grounds="Prestação de contas anual, Fundo Partidário e direção nacional.",
        consequence="Mantida sanção financeira com impacto sobre cotas futuras.",
        strategic_comment="Interessa diretamente a partido nacional e à governança de recursos públicos.",
        why_relevant="Afeta diretório nacional e financiamento partidário.",
    )

    pairs = notion.build_publishable_case_pairs(
        [local_case, national_case],
        [local_analysis, national_analysis],
    )

    assert pairs == [(national_case, national_analysis)]
    assert notion.is_federal_strategic_interest(national_case, national_analysis)
    assert not notion.is_federal_strategic_interest(local_case, local_analysis)


def test_build_publishable_case_pairs_limits_focused_cases():
    cases = []
    analyses = []
    for idx in range(notion.MAX_PUBLISHED_CASES + 3):
        case = replace(
            _make_case(),
            case_id=f"case-{idx}",
            page_id=f"page-{idx}",
            source_url=f"https://www.notion.so/page-{idx}",
            sigla_uf="DF",
            nome_municipio="BRASÍLIA",
            partes=[f"PARTIDO TESTE {idx} - NACIONAL"],
            partidos=[f"PT{idx}/Nacional"],
            tema="Prestação de contas anual de partido nacional",
        )
        analysis = replace(
            _make_analysis(),
            case_id=f"case-{idx}",
            page_id=f"page-{idx}",
            relevance_score=9,
            display_score=9,
            risk_level="alto",
            includes_public_figure=False,
            includes_party=True,
            public_figures=[],
            parties=[f"PT{idx}/Nacional"],
            legal_grounds="Fundo Partidário e contas anuais de direção nacional.",
            strategic_comment="Tema material para governança partidária nacional.",
            why_relevant="Afeta partido nacional.",
        )
        cases.append(case)
        analyses.append(analysis)

    pairs = notion.build_publishable_case_pairs(cases, analyses)

    assert len(pairs) == notion.MAX_PUBLISHED_CASES
