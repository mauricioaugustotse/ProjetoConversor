from collections import Counter

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


def test_summarize_report_preserves_full_overview_callout_from_openai(monkeypatch):
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
    assert summary.overview_callout == notion._normalize_ws(full_text)
    assert summary.overview_callout.endswith("afastar devolução ao erário.")


def test_build_callout_block_preserves_full_long_text_across_rich_text_chunks():
    full_text = " ".join(f"trecho-{idx:04d}" for idx in range(600))

    block = notion.build_callout_block(full_text)

    assert notion._plain_rich_text(block["callout"]["rich_text"]) == full_text
