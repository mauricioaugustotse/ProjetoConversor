import asyncio
import csv
import json
from pathlib import Path

import pytest

import Artefatos.legado.SJUR_csv_to_csv_NOTIONfriendly as sjur
import SJUR_csv_to_csv_APIenriching as api_enrich
import SJUR_csv_to_csv_NOTIONfriendly_v2 as sjur_v2


RAW_FIELDS = [
    "numeroUnico",
    "dataDecisao",
    "textoDecisao",
    "textoEmenta",
    "descricaoClasse",
    "nomeTipoProcesso",
    "siglaTribunalJE",
    "origemDecisao",
    "siglaUF",
    "nomeMunicipio",
    "partes",
    "assuntos",
    "relator",
    "descricaoTipoDecisao",
    "advogados",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _raw_row(case_id: int) -> dict[str, str]:
    return {
        "numeroUnico": f"060000{case_id:02d}-00.2024.6.26.0000",
        "dataDecisao": "16/03/2026",
        "textoDecisao": f"Texto decisao {case_id}",
        "textoEmenta": f"Ementa {case_id}",
        "descricaoClasse": "AIJE",
        "nomeTipoProcesso": "Acao de Investigacao Judicial Eleitoral",
        "siglaTribunalJE": "TRE-SP",
        "origemDecisao": "SP",
        "siglaUF": "SP",
        "nomeMunicipio": "SAO PAULO",
        "partes": f"Parte {case_id}",
        "assuntos": "Abuso de poder",
        "relator": "Ministro Teste",
        "descricaoTipoDecisao": "acordao",
        "advogados": "",
    }


def _news_row(case_id: int) -> dict[str, str]:
    return {
        "numeroUnico": f"060000{case_id:02d}-00.2024.6.26.0000",
        "assuntos": "Abuso de poder",
        "partes": f"Parte {case_id}",
        "relator": "Ministro Teste",
        "tema": f"Tema {case_id}",
        "punchline": f"Punchline {case_id}",
        "noticia_TSE": "",
        "noticia_TRE": "",
        "noticia_geral_1": "",
        "noticia_geral_2": "",
    }


def _lookup_payload(case_id: int) -> dict[str, str]:
    return {
        "numero_unico": f"060000{case_id:02d}-00.2024.6.26.0000",
        "data_decisao": "03/16/2026",
        "assuntos": "Abuso de poder",
        "partes": f"Parte {case_id}",
        "advogados": "",
        "relator": "Ministro Teste",
        "descricao_classe": "AIJE",
        "nome_tipo_processo": "Acao de Investigacao Judicial Eleitoral",
        "texto_decisao": f"Texto decisao {case_id}",
        "texto_ementa": f"Ementa {case_id}",
        "sigla_uf": "SP",
        "nome_municipio": "SAO PAULO",
        "tribunal": "TRE-SP SP",
        "origem": "SP SAO PAULO",
        "tema": f"Tema {case_id}",
        "punchline": f"Punchline {case_id}",
    }


def _noop_logger(_message: str) -> None:
    return


@pytest.fixture(autouse=True)
def _redirect_artifacts_root(monkeypatch, tmp_path):
    artifacts_root = tmp_path / "Artefatos"
    monkeypatch.setattr(sjur, "ARTIFACTS_ROOT", artifacts_root)
    monkeypatch.setattr(api_enrich.sjur, "ARTIFACTS_ROOT", artifacts_root)
    monkeypatch.setattr(sjur_v2, "ARTIFACTS_ROOT", artifacts_root)


def _disabled_tema_config() -> sjur.TemaPunchlineConfig:
    return sjur.TemaPunchlineConfig(enabled=False, api_key="")


def _disabled_assuntos_config() -> sjur.AssuntosEnrichmentConfig:
    return sjur.AssuntosEnrichmentConfig(enabled=False, api_key="")


def _disabled_web_config() -> sjur.WebLookupConfig:
    return sjur.WebLookupConfig(enabled=False, api_key="")


def _enabled_web_config() -> sjur.WebLookupConfig:
    return sjur.WebLookupConfig(
        enabled=True,
        api_key="test-key",
        model="sonar",
        timeout_seconds=1,
        max_workers=2,
        batch_size=10,
        delay_between_batches=0,
    )


def test_classify_resultado_prioritizes_monocratica_over_origin_unanimity():
    texto = (
        "DECISÃO Eleições 2024. Negado seguimento ao agravo em recurso especial. "
        "Na origem, o Tribunal Regional Eleitoral, por unanimidade, manteve a sentença."
    )

    assert sjur.classify_resultado(texto, "Decisão monocrática") == "monocrática"


def test_classify_resultado_prioritizes_monocratica_over_origin_majority():
    texto = (
        "DECISÃO Eleições 2024. Indeferido o pedido liminar. "
        "Na origem, a Corte Regional, por maioria de votos, negou provimento ao recurso."
    )

    assert sjur.classify_resultado(texto, "Decisão monocrática") == "monocrática"


def test_classify_resultado_keeps_nao_conhecido_in_monocratica():
    texto = (
        "DECISÃO Eleições 2024. Agravo em recurso especial não conhecido. "
        "Na origem, o Tribunal Regional Eleitoral, por unanimidade, manteve a sentença."
    )

    assert sjur.classify_resultado(texto, "Decisão monocrática") == "monocrática,não conhecido"


def test_clean_referencias_legislativas_extracts_article_law_pairs():
    texto = (
        "Eleições 2024. Art. 73, § 10, da Lei nº 9.504/1997. "
        "Abuso de poder (art. 22, XIV, da LC n. 64/1990). "
        "Litigância de má-fé nos termos do art. 80, IV, c.c. art. 81, ambos do CPC."
    )

    assert sjur.clean_referencias_legislativas(texto) == (
        "art. 73 § 10 Lei n° 9.504/97, "
        "art. 22 XIV LC n° 64/90, "
        "art. 80 IV Lei n° 13.105/15, "
        "art. 81 Lei n° 13.105/15"
    )


def test_clean_referencias_legislativas_uses_aliases_and_skips_redundant_law_only_labels():
    texto = (
        "Representação por captação ilícita de sufrágio. "
        "Incidência do art. 41-A da Lei das Eleições e do art. 14, § 10, da Constituição Federal."
    )

    assert sjur.clean_referencias_legislativas(texto) == (
        "art. 41-A Lei n° 9.504/97, art. 14 § 10 CF/88"
    )


def test_clean_referencias_legislativas_normalizes_roman_numerals_and_ordinals():
    texto = (
        "Conforme o art. 373, i, do CPC. "
        "Ainda, o art. 121, § 4°, i, da Constituição Federal."
    )

    assert sjur.clean_referencias_legislativas(texto) == (
        "art. 373 I Lei n° 13.105/15, art. 121 § 4º I CF/88"
    )


def test_clean_referencias_legislativas_splits_semicolon_compound_labels():
    texto = "CF/88; Lei n° 13.105/15; Lei n° 9.504/97; Res. TSE n° 23.709/22"

    assert sjur.clean_referencias_legislativas(texto) == (
        "CF/88, Lei n° 13.105/15, Lei n° 9.504/97, Res. TSE n° 23.709/22"
    )


def test_news_lookup_uses_one_call_per_pending_row(tmp_path, monkeypatch):
    calls: list[str] = []

    async def fake_lookup(_manager, page_data, **_kwargs):
        calls.append(str(page_data.get("numero_unico", "")))
        case_id = len(calls)
        return {
            "status": "ok",
            "tse": f"https://www.tse.jus.br/comunicacao/noticias/2026/03/caso-tse-{case_id}-2026",
            "tre": f"https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/caso-tre-{case_id}-2026",
            "gerais": [f"https://g1.globo.com/politica/noticia/2026/03/geral-{case_id}.ghtml"],
        }

    monkeypatch.setattr(sjur, "buscar_todas_noticias_async", fake_lookup)

    rows = [_news_row(1), _news_row(2), _news_row(3)]
    payloads = [_lookup_payload(1), _lookup_payload(2), _lookup_payload(3)]
    metrics = asyncio.run(
        sjur.enriquecer_rows_com_urls_async(
            rows,
            _noop_logger,
            _enabled_web_config(),
            lookup_payloads=payloads,
            cache_path=tmp_path / "cache.json",
        )
    )

    assert metrics["perplexity_api_calls"] == 3
    assert len(calls) == 3
    assert rows[0]["noticia_TSE"].startswith("https://www.tse.jus.br/comunicacao/noticias/")
    assert rows[1]["noticia_TRE"].startswith("https://www.tre-sp.jus.br/comunicacao/noticias/")
    assert rows[2]["noticia_geral_1"].startswith("https://g1.globo.com/")


def test_build_news_search_queries_uses_short_priority_plan():
    queries = sjur.build_news_search_queries(
        _lookup_payload(1),
        precisa_tse=True,
        precisa_tre=True,
        precisa_gerais=True,
    )

    assert [label for label, _query in queries] == [
        "shared_exact",
        "tse_exact",
        "tre_exact",
        "general_context",
        "shared_context",
        "tse_context",
        "tre_context",
        "general_editorial_mainstream",
        "general_editorial_legal",
    ]


def test_build_news_search_queries_general_only_uses_economic_editorial_plan():
    queries = sjur.build_news_search_queries(
        _lookup_payload(1),
        precisa_tse=False,
        precisa_tre=False,
        precisa_gerais=True,
    )

    assert [label for label, _query in queries] == [
        "general_editorial_mainstream",
        "general_editorial_legal",
        "general_context",
    ]


def test_target_general_lookup_count_prefers_single_link_on_general_only_pass():
    assert (
        sjur._target_general_lookup_count(  # pylint: disable=protected-access
            precisa_tse=False,
            precisa_tre=False,
            missing_general_count=2,
        )
        == 1
    )
    assert (
        sjur._target_general_lookup_count(  # pylint: disable=protected-access
            precisa_tse=True,
            precisa_tre=False,
            missing_general_count=2,
        )
        == 2
    )


def test_general_media_url_accepts_expanded_trusted_domains():
    assert sjur._is_general_media_url(  # pylint: disable=protected-access
        "https://oglobo.globo.com/politica/noticia/2026/03/caso-eleitoral.ghtml"
    )
    assert sjur._is_general_media_url(  # pylint: disable=protected-access
        "https://www.correiobraziliense.com.br/politica/2026/03/caso-eleitoral.shtml"
    )
    assert not sjur._is_general_media_url(  # pylint: disable=protected-access
        "https://www.migalhas.com.br/amanhecidas/131710/migalhas-n--2-616-"
    )


def test_general_search_preferences_follow_query_label():
    assert sjur._search_domain_order_for_query_label(  # pylint: disable=protected-access
        "general_editorial_mainstream"
    ) == sjur.GENERAL_MEDIA_MAINSTREAM_DOMAINS
    assert sjur._search_domain_order_for_query_label(  # pylint: disable=protected-access
        "general_editorial_legal"
    ) == sjur.GENERAL_MEDIA_LEGAL_DOMAINS
    assert sjur._search_domain_order_for_query_label(  # pylint: disable=protected-access
        "general_context"
    ) == sjur.GENERAL_MEDIA_DOMAIN_ORDER
    assert (
        sjur._search_max_tokens_per_page_for_query_label(  # pylint: disable=protected-access
            "general_context"
        )
        == 128
    )
    assert (
        sjur._search_max_tokens_per_page_for_query_label(  # pylint: disable=protected-access
            "shared_exact"
        )
        == sjur.PERPLEXITY_SEARCH_MAX_TOKENS_PER_PAGE
    )


def test_news_cache_skips_filled_and_no_match_entries(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    row_filled = _news_row(1)
    row_no_match = _news_row(2)
    payload_filled = _lookup_payload(1)
    payload_no_match = _lookup_payload(2)

    key_filled = sjur.build_news_request_key(
        sjur.build_news_lookup_request(
            payload_filled,
            model="sonar",
            precisa_tse=True,
            precisa_tre=True,
            needed_general_count=2,
        )
    )
    key_no_match = sjur.build_news_request_key(
        sjur.build_news_lookup_request(
            payload_no_match,
            model="sonar",
            precisa_tse=True,
            precisa_tre=True,
            needed_general_count=2,
        )
    )
    sjur.write_news_cache(
        cache_path,
        {
            key_filled: {
                "status": "filled",
                "tse": "https://www.tse.jus.br/comunicacao/noticias/2026/03/cache-filled",
                "tre": "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/cache-filled",
                "gerais": [
                    "https://g1.globo.com/politica/noticia/2026/03/cache-filled.ghtml",
                    "https://www.cnnbrasil.com.br/politica/cache-filled/",
                ],
                "updated_at": "2026-03-16T12:00:00Z",
            },
            key_no_match: {
                "status": "no_match",
                "tse": "",
                "tre": "",
                "gerais": [],
                "updated_at": "2026-03-16T12:00:00Z",
            },
        },
    )

    async def should_not_call(*_args, **_kwargs):
        raise AssertionError("Perplexity nao deveria ser chamada quando o cache terminal existe.")

    monkeypatch.setattr(sjur, "buscar_todas_noticias_async", should_not_call)

    metrics = asyncio.run(
        sjur.enriquecer_rows_com_urls_async(
            [row_filled, row_no_match],
            _noop_logger,
            _enabled_web_config(),
            lookup_payloads=[payload_filled, payload_no_match],
            cache_path=cache_path,
        )
    )

    assert metrics["perplexity_api_calls"] == 0
    assert metrics["perplexity_cache_hits"] == 2
    assert metrics["perplexity_no_match"] == 1
    assert row_filled["noticia_TSE"] == "https://www.tse.jus.br/comunicacao/noticias/2026/03/cache-filled"
    assert row_filled["noticia_TRE"] == "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/cache-filled"
    assert row_filled["noticia_geral_1"] == "https://g1.globo.com/politica/noticia/2026/03/cache-filled.ghtml"
    assert row_no_match["noticia_TSE"] == ""


def test_invalid_filled_cache_entry_is_ignored_and_requeried(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    row = _news_row(1)
    payload = _lookup_payload(1)
    request_key = sjur.build_news_request_key(
        sjur.build_news_lookup_request(
            payload,
            model="sonar",
            precisa_tse=True,
            precisa_tre=True,
            needed_general_count=2,
        )
    )
    sjur.write_news_cache(
        cache_path,
        {
            request_key: {
                "status": "filled",
                "tse": "https://www.tse.jus.br",
                "tre": "",
                "gerais": [],
                "updated_at": "2026-03-16T12:00:00Z",
            }
        },
    )

    calls: list[str] = []

    async def fake_lookup(_manager, page_data, **_kwargs):
        calls.append(str(page_data.get("numero_unico", "")))
        return {
            "status": "ok",
            "tse": "https://www.tse.jus.br/comunicacao/noticias/2026/03/caso-requeried-2026",
            "tre": "",
            "gerais": [],
        }

    monkeypatch.setattr(sjur, "buscar_todas_noticias_async", fake_lookup)

    metrics = asyncio.run(
        sjur.enriquecer_rows_com_urls_async(
            [row],
            _noop_logger,
            _enabled_web_config(),
            lookup_payloads=[payload],
            cache_path=cache_path,
        )
    )

    assert metrics["perplexity_api_calls"] == 1
    assert metrics["perplexity_cache_hits"] == 0
    assert calls == [payload["numero_unico"]]
    assert row["noticia_TSE"] == "https://www.tse.jus.br/comunicacao/noticias/2026/03/caso-requeried-2026"


def test_cache_with_only_official_link_does_not_skip_general_second_pass(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    row = _news_row(1)
    row["noticia_TSE"] = "https://www.tse.jus.br/comunicacao/noticias/2026/03/tse-ja-preenchida"
    payload = _lookup_payload(1)
    request_key = sjur.build_news_request_key(
        sjur.build_news_lookup_request(
            payload,
            model="sonar",
            precisa_tse=False,
            precisa_tre=True,
            needed_general_count=2,
        )
    )
    sjur.write_news_cache(
        cache_path,
        {
            request_key: {
                "status": "filled",
                "tse": "https://www.tse.jus.br/comunicacao/noticias/2026/03/cache-filled",
                "tre": "",
                "gerais": [],
                "updated_at": "2026-03-16T12:00:00Z",
            }
        },
    )

    calls: list[str] = []

    async def fake_lookup(_manager, page_data, **_kwargs):
        calls.append(str(page_data.get("numero_unico", "")))
        return {
            "status": "ok",
            "tse": None,
            "tre": "",
            "gerais": ["https://g1.globo.com/politica/noticia/2026/03/geral-cache-bypass.ghtml"],
        }

    monkeypatch.setattr(sjur, "buscar_todas_noticias_async", fake_lookup)

    metrics = asyncio.run(
        sjur.enriquecer_rows_com_urls_async(
            [row],
            _noop_logger,
            _enabled_web_config(),
            lookup_payloads=[payload],
            cache_path=cache_path,
        )
    )

    assert metrics["perplexity_api_calls"] == 1
    assert metrics["perplexity_cache_hits"] == 0
    assert calls == [payload["numero_unico"]]
    assert row["noticia_geral_1"] == "https://g1.globo.com/politica/noticia/2026/03/geral-cache-bypass.ghtml"


def test_api_enriching_treats_crude_official_links_as_pending(tmp_path, monkeypatch):
    input_path = tmp_path / "entrada_notion.csv"
    output_path = tmp_path / "saida.csv"
    row = _raw_row(1)
    row.update(
        {
            "tema": "Tema 1",
            "punchline": "Punchline 1",
            "noticia_TSE": "https://www.tse.jus.br",
            "noticia_TRE": "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/tre-valida",
            "noticia_geral_1": "https://g1.globo.com/politica/noticia/2026/03/geral-1.ghtml",
            "noticia_geral_2": "https://www.cnnbrasil.com.br/politica/geral-1/",
        }
    )
    _write_csv(input_path, list(row.keys()), [row])

    calls: list[int] = []

    async def fake_url_enrichment(rows, _logger, _config, **_kwargs):
        calls.append(len(rows))
        rows[0]["noticia_TSE"] = "https://www.tse.jus.br/comunicacao/noticias/2026/03/tse-util-2026"
        return {
            "perplexity_api_calls": 1,
            "perplexity_cache_hits": 0,
            "perplexity_no_match": 0,
            "perplexity_retryable_errors": 0,
            "perplexity_skipped_existing": 0,
        }

    monkeypatch.setattr(api_enrich.sjur, "enriquecer_rows_com_urls_async", fake_url_enrichment)

    api_enrich.enrich_one_file(
        input_path=input_path,
        output_path=output_path,
        use_tema_punchline=False,
        use_assuntos=False,
        use_urls=True,
        tema_config=None,
        assuntos_config=None,
        web_config=_enabled_web_config(),
        backup_before_write=False,
        logger=_noop_logger,
    )

    output_rows = _read_csv_rows(output_path)
    assert calls == [1]
    assert output_rows[0]["noticia_TSE"] == "https://www.tse.jus.br/comunicacao/noticias/2026/03/tse-util-2026"
    assert output_rows[0]["noticia_TRE"] == "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/tre-valida"


def test_process_one_csv_reuses_existing_notion_output_without_checkpoint(tmp_path):
    input_path = tmp_path / "entrada.csv"
    out_dir = tmp_path / "saida"
    _write_csv(input_path, RAW_FIELDS, [_raw_row(1)])

    output_path = out_dir / "entrada_notion.csv"
    _write_csv(
        output_path,
        [
            "numeroUnico",
            "tema",
            "punchline",
            "noticia_TSE",
            "noticia_TRE",
            "noticia_geral_1",
            "noticia_geral_2",
        ],
        [
            {
                "numeroUnico": _raw_row(1)["numeroUnico"],
                "tema": "Tema preservado",
                "punchline": "Punchline preservada",
                "noticia_TSE": "https://www.tse.jus.br/comunicacao/noticias/2026/03/tse-preservado",
                "noticia_TRE": "",
                "noticia_geral_1": "https://g1.globo.com/politica/noticia/2026/03/preservada.ghtml",
                "noticia_geral_2": "",
            }
        ],
    )

    summary = sjur.process_one_csv(
        input_path=input_path,
        out_dir=out_dir,
        max_texto_chars=0,
        replace_newlines=True,
        web_lookup_config=_disabled_web_config(),
        tema_punchline_config=_disabled_tema_config(),
        assuntos_enrichment_config=_disabled_assuntos_config(),
        metadata_extraction_config=sjur.MetadataExtractionConfig(
            include_institutional_entities=True,
            header_max_chars=sjur.DEFAULT_METADATA_HEADER_MAX_CHARS,
        ),
        logger=_noop_logger,
    )

    rows = _read_csv_rows(summary.output_path)
    assert rows[0]["tema"] == "Tema preservado"
    assert rows[0]["punchline"] == "Punchline preservada"
    assert rows[0]["noticia_TSE"].endswith("tse-preservado")
    assert rows[0]["noticia_geral_1"].endswith("preservada.ghtml")

    report_path = sjur.resolve_report_artifact_path(summary.output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["restored_from_output_csv"] >= 1
    assert report["status"] == "completed"


def test_process_one_csv_rebuilds_referencias_legislativas_from_decision_text(tmp_path):
    input_path = tmp_path / "entrada.csv"
    out_dir = tmp_path / "saida"
    fieldnames = [*RAW_FIELDS, "referenciasLegislativas"]
    row = _raw_row(1)
    row["textoDecisao"] = (
        "Decisão. Art. 73, § 10, da Lei nº 9.504/1997. "
        "Também incide o art. 22, XIV, da LC n. 64/1990."
    )
    row["textoEmenta"] = "Aplicação subsidiária do art. 1.022 do Código de Processo Civil."
    row["referenciasLegislativas"] = ""
    _write_csv(input_path, fieldnames, [row])

    summary = sjur.process_one_csv(
        input_path=input_path,
        out_dir=out_dir,
        max_texto_chars=0,
        replace_newlines=True,
        web_lookup_config=_disabled_web_config(),
        tema_punchline_config=_disabled_tema_config(),
        assuntos_enrichment_config=_disabled_assuntos_config(),
        metadata_extraction_config=sjur.MetadataExtractionConfig(
            include_institutional_entities=True,
            header_max_chars=sjur.DEFAULT_METADATA_HEADER_MAX_CHARS,
        ),
        logger=_noop_logger,
    )

    rows = _read_csv_rows(summary.output_path)
    assert rows[0]["referenciasLegislativas"] == (
        "art. 73 § 10 Lei n° 9.504/97,"
        "art. 22 XIV LC n° 64/90,"
        "art. 1.022 Lei n° 13.105/15"
    )


def test_run_batch_respects_out_dir(tmp_path):
    input_path = tmp_path / "entrada.csv"
    chosen_out_dir = tmp_path / "destino-custom"
    _write_csv(input_path, RAW_FIELDS, [_raw_row(1)])

    summaries, combined_path, compiled_rows = sjur.run_batch(
        files=[str(input_path)],
        out_dir=str(chosen_out_dir),
        max_texto_chars=0,
        combined_name="",
        replace_newlines=True,
        web_lookup_config=_disabled_web_config(),
        tema_punchline_config=_disabled_tema_config(),
        assuntos_enrichment_config=_disabled_assuntos_config(),
        metadata_extraction_config=sjur.MetadataExtractionConfig(
            include_institutional_entities=True,
            header_max_chars=sjur.DEFAULT_METADATA_HEADER_MAX_CHARS,
        ),
        logger=_noop_logger,
    )

    assert len(summaries) == 1
    assert summaries[0].output_path.parent == sjur.resolve_intermediate_csv_dir()
    assert summaries[0].output_path.exists()
    assert combined_path.parent == chosen_out_dir
    assert combined_path.exists()
    assert compiled_rows == 1
    assert sjur.resolve_checkpoint_artifact_path(summaries[0].output_path).exists()
    assert sjur.resolve_report_artifact_path(summaries[0].output_path).exists()


def test_checkpoint_resume_only_queries_pending_rows(tmp_path, monkeypatch):
    input_path = tmp_path / "entrada.csv"
    out_dir = tmp_path / "saida"
    rows = [_raw_row(1), _raw_row(2), _raw_row(3)]
    _write_csv(input_path, RAW_FIELDS, rows)

    sjur.process_one_csv(
        input_path=input_path,
        out_dir=out_dir,
        max_texto_chars=0,
        replace_newlines=True,
        web_lookup_config=_disabled_web_config(),
        tema_punchline_config=_disabled_tema_config(),
        assuntos_enrichment_config=_disabled_assuntos_config(),
        metadata_extraction_config=sjur.MetadataExtractionConfig(
            include_institutional_entities=True,
            header_max_chars=sjur.DEFAULT_METADATA_HEADER_MAX_CHARS,
        ),
        logger=_noop_logger,
    )

    checkpoint_path = sjur.resolve_checkpoint_artifact_path(out_dir / "entrada_notion.csv")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["status"] = "running"
    checkpoint["stage"] = "perplexity_batch_1_2_of_3"
    checkpoint["perplexity_enabled"] = True
    for index, suffix in ((0, "1"), (1, "2")):
        checkpoint["processed_rows"][index]["noticia_TSE"] = (
            f"https://www.tse.jus.br/comunicacao/noticias/2026/03/preenchido-{suffix}"
        )
        checkpoint["processed_rows"][index]["noticia_TRE"] = (
            f"https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/preenchido-{suffix}"
        )
        checkpoint["processed_rows"][index]["noticia_geral_1"] = (
            f"https://g1.globo.com/politica/noticia/2026/03/preenchido-{suffix}.ghtml"
        )
        checkpoint["processed_rows"][index]["noticia_geral_2"] = (
            f"https://www.cnnbrasil.com.br/politica/preenchido-{suffix}/"
        )
    checkpoint["processed_rows"][2]["noticia_TSE"] = ""
    checkpoint["processed_rows"][2]["noticia_TRE"] = ""
    checkpoint["processed_rows"][2]["noticia_geral_1"] = ""
    checkpoint["processed_rows"][2]["noticia_geral_2"] = ""
    checkpoint["row_progress"] = {
        "0": {"stage": "perplexity_batch_1_2_of_3", "request_key": "k1", "news_status": "filled"},
        "1": {"stage": "perplexity_batch_1_2_of_3", "request_key": "k2", "news_status": "filled"},
        "2": {"stage": "perplexity_batch_1_2_of_3", "request_key": "k3", "news_status": "pending"},
    }
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

    calls: list[str] = []

    async def fake_lookup(_manager, page_data, **_kwargs):
        calls.append(str(page_data.get("numero_unico", "")))
        return {
            "status": "ok",
            "tse": "https://www.tse.jus.br/comunicacao/noticias/2026/03/preenchido-3",
            "tre": "",
            "gerais": [],
        }

    monkeypatch.setattr(sjur, "buscar_todas_noticias_async", fake_lookup)

    sjur.process_one_csv(
        input_path=input_path,
        out_dir=out_dir,
        max_texto_chars=0,
        replace_newlines=True,
        web_lookup_config=_enabled_web_config(),
        tema_punchline_config=_disabled_tema_config(),
        assuntos_enrichment_config=_disabled_assuntos_config(),
        metadata_extraction_config=sjur.MetadataExtractionConfig(
            include_institutional_entities=True,
            header_max_chars=sjur.DEFAULT_METADATA_HEADER_MAX_CHARS,
        ),
        logger=_noop_logger,
    )

    assert len(calls) == 1

    final_rows = _read_csv_rows(out_dir / "entrada_notion.csv")
    assert final_rows[0]["noticia_TSE"].endswith("preenchido-1")
    assert final_rows[1]["noticia_TSE"].endswith("preenchido-2")
    assert final_rows[2]["noticia_TSE"].endswith("preenchido-3")

    final_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    final_report = json.loads(
        sjur.resolve_report_artifact_path(out_dir / "entrada_notion.csv").read_text(encoding="utf-8")
    )
    assert final_checkpoint["status"] == "completed"
    assert final_checkpoint["stage"] == "final"
    assert "row_progress" in final_checkpoint
    assert final_checkpoint["cache_file"].endswith(sjur.PERPLEXITY_CACHE_FILENAME)
    assert final_report["perplexity_api_calls"] == 1


def test_news_lookup_logs_progress_within_batch(tmp_path, monkeypatch):
    logs: list[str] = []

    async def fake_lookup(_manager, page_data, **_kwargs):
        await asyncio.sleep(0.01 if str(page_data.get("numero_unico", "")).endswith("01-00.2024.6.26.0000") else 0.02)
        return {
            "status": "ok",
            "tse": f"https://www.tse.jus.br/comunicacao/noticias/2026/03/{page_data['numero_unico']}",
            "tre": "",
            "gerais": [],
        }

    monkeypatch.setattr(sjur, "buscar_todas_noticias_async", fake_lookup)

    rows = [_news_row(1), _news_row(2)]
    payloads = [_lookup_payload(1), _lookup_payload(2)]
    asyncio.run(
        sjur.enriquecer_rows_com_urls_async(
            rows,
            logs.append,
            _enabled_web_config(),
            lookup_payloads=payloads,
            cache_path=tmp_path / "cache.json",
        )
    )

    assert any("progresso | consultas=1/2" in message for message in logs)
    assert any("Lote 1-2 concluido" in message for message in logs)


def test_generic_tse_tre_links_are_rejected():
    tse, tre, gerais = sjur._normalize_news_lookup_response(  # pylint: disable=protected-access
        {
            "tse": {
                "url": "https://www.tse.jus.br/institucional",
                "confidence": "high",
                "matched_fields": ["numero_unico", "tema"],
            },
            "tre": {
                "url": "https://www.tre-sp.jus.br/",
                "confidence": "high",
                "matched_fields": ["numero_unico", "tema"],
            },
            "gerais": [
                {
                    "url": "https://g1.globo.com/politica/noticia/2026/03/julgamento-eleitoral-repercussao.ghtml",
                    "confidence": "high",
                    "matched_fields": ["numero_unico", "tema"],
                }
            ],
        },
        precisa_tse=True,
        precisa_tre=True,
        precisa_gerais=True,
    )

    assert tse is None
    assert tre is None
    assert gerais == ["https://g1.globo.com/politica/noticia/2026/03/julgamento-eleitoral-repercussao.ghtml"]


def test_low_confidence_news_candidates_are_rejected():
    tse, tre, gerais = sjur._normalize_news_lookup_response(  # pylint: disable=protected-access
        {
            "tse": {
                "url": "https://www.tse.jus.br/comunicacao/noticias/2026/03/noticia-oficial",
                "confidence": "low",
                "matched_fields": ["numero_unico", "tema"],
            },
            "tre": {
                "url": "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/noticia-oficial",
                "confidence": "low",
                "matched_fields": ["numero_unico", "tema"],
            },
            "gerais": [
                {
                    "url": "https://g1.globo.com/politica/noticia/2026/03/noticia-oficial.ghtml",
                    "confidence": "low",
                    "matched_fields": ["numero_unico", "tema"],
                }
            ],
        },
        precisa_tse=True,
        precisa_tre=True,
        precisa_gerais=True,
    )

    assert tse is None
    assert tre is None
    assert gerais is None


def test_medium_confidence_official_news_can_pass_with_strong_match():
    tse, tre, gerais = sjur._normalize_news_lookup_response(  # pylint: disable=protected-access
        {
            "tse": {
                "url": "https://www.tse.jus.br/comunicacao/noticias/2026/03/noticia-oficial",
                "confidence": "medium",
                "matched_fields": ["numero_unico", "tema"],
            },
            "tre": {
                "url": "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/noticia-oficial",
                "confidence": "medium",
                "matched_fields": ["numero_unico", "tema"],
            },
            "gerais": [],
        },
        precisa_tse=True,
        precisa_tre=True,
        precisa_gerais=False,
    )

    assert tse == "https://www.tse.jus.br/comunicacao/noticias/2026/03/noticia-oficial"
    assert tre == "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/noticia-oficial"
    assert gerais is None


def test_medium_confidence_general_news_can_pass_with_consistent_context():
    tse, tre, gerais = sjur._normalize_news_lookup_response(  # pylint: disable=protected-access
        {
            "tse": {"url": "", "confidence": "none", "matched_fields": []},
            "tre": {"url": "", "confidence": "none", "matched_fields": []},
            "gerais": [
                {
                    "url": "https://g1.globo.com/politica/noticia/2026/03/noticia-oficial.ghtml",
                    "confidence": "medium",
                    "matched_fields": ["tribunal", "tema", "relator"],
                }
            ],
        },
        precisa_tse=False,
        precisa_tre=False,
        precisa_gerais=True,
    )

    assert tse is None
    assert tre is None
    assert gerais == ["https://g1.globo.com/politica/noticia/2026/03/noticia-oficial.ghtml"]


def test_apply_urls_no_row_skips_duplicate_general_links():
    row = _news_row(1)
    row["noticia_geral_1"] = "https://www.migalhas.com.br/quentes/417625/caso-eleitoral"

    sjur._aplicar_urls_no_row(  # pylint: disable=protected-access
        row,
        None,
        None,
        [
            "https://www.migalhas.com.br/quentes/417625/caso-eleitoral",
            "https://g1.globo.com/politica/noticia/2026/03/caso-eleitoral.ghtml",
        ],
    )

    assert row["noticia_geral_1"] == "https://www.migalhas.com.br/quentes/417625/caso-eleitoral"
    assert row["noticia_geral_2"] == "https://g1.globo.com/politica/noticia/2026/03/caso-eleitoral.ghtml"


def test_news_prompt_includes_official_first_query_hints():
    hints = sjur.build_news_query_hints(_lookup_payload(1))
    prompt = sjur.gerar_prompt_noticias(_lookup_payload(1), max_general_urls=2)

    assert hints["tre_domain"] == "tre-sp.jus.br"
    assert 'site:tse.jus.br/comunicacao/noticias "06000001-00.2024.6.26.0000"' in prompt
    assert 'site:tre-sp.jus.br/comunicacao/noticias "06000001-00.2024.6.26.0000"' in prompt
    assert 'Ignore totalmente resultados sobre regras gerais das eleicoes' in prompt


def test_local_page_validation_rejects_generic_roundup_pages():
    summary = {
        "status_code": 200,
        "title": "Confira a pauta de julgamentos do TSE desta quinta-feira (21)",
        "text": (
            "Confira os processos da pauta 06000001-00.2024.6.26.0000 "
            "06000002-00.2024.6.26.0000 06000003-00.2024.6.26.0000"
        ),
    }

    assert not sjur._is_valid_news_page_summary(summary)  # pylint: disable=protected-access


def test_local_page_validation_rejects_generic_rules_and_calendar_titles():
    for title in (
        "Saiba o que diz a lei sobre propaganda eleitoral antecipada",
        "Calendário eleitoral de 2026: veja datas e regras das eleições",
        "TSE abre Ano Judiciário de 2026 nesta segunda (2)",
    ):
        summary = {
            "status_code": 200,
            "title": title,
            "text": "Conteúdo extenso com detalhes institucionais e sem foco em um caso específico.",
        }
        assert not sjur._is_valid_news_page_summary(summary)  # pylint: disable=protected-access


def test_search_based_lookup_uses_editorial_second_pass_for_general_media():
    class FakeManager:
        async def search_news_results(self, **kwargs):
            query = str(kwargs.get("query", ""))
            if "site:g1.globo.com" not in query:
                return []
            return [
                {
                    "url": "https://g1.globo.com/politica/noticia/2026/03/caso-geral-editorial.ghtml",
                    "title": "Caso eleitoral em São Paulo envolve Parte 1",
                    "snippet": (
                        "Tema 1 Parte 1 São Paulo SP AIJE "
                        "Acao de Investigacao Judicial Eleitoral Origem SP Sao Paulo."
                    ),
                    "date": "2026-03-18",
                }
            ]

        async def fetch_page_summary(self, url, timeout=sjur.PERPLEXITY_PAGE_FETCH_TIMEOUT):
            return {
                "status_code": 200,
                "url": url,
                "title": "Caso eleitoral em São Paulo envolve Parte 1",
                "text": (
                    "Tema 1 Parte 1 São Paulo SP AIJE Acao de Investigacao Judicial Eleitoral "
                    "Origem SP Sao Paulo repercute em reportagem jornalística sobre o mesmo caso eleitoral."
                ),
                "content_type": "text/html",
            }

    result = asyncio.run(
        sjur.buscar_todas_noticias_async(
            FakeManager(),
            _lookup_payload(1),
            precisa_tse=False,
            precisa_tre=False,
            precisa_gerais=True,
            needed_general_count=1,
            timeout_seconds=1,
            max_tokens=120,
        )
    )

    assert result["status"] == "ok"
    assert result["gerais"] == ["https://g1.globo.com/politica/noticia/2026/03/caso-geral-editorial.ghtml"]


def test_search_based_lookup_rejects_general_rules_article_without_case_specific_detail():
    payload = _lookup_payload(1)
    payload.update(
        {
            "tema": "",
            "punchline": "",
            "assuntos": "",
            "descricao_classe": "PROPAGANDA PARTIDÁRIA",
            "nome_tipo_processo": "",
            "sigla_uf": "DF",
            "nome_municipio": "BRASÍLIA",
            "origem": "DF BRASÍLIA",
            "tribunal": "TSE DF",
            "partes": "PARTIDO SOCIALISMO E LIBERDADE (PSOL) - NACIONAL",
            "relator": "Min. Nunes Marques",
        }
    )

    class FakeManager:
        async def search_news_results(self, **kwargs):
            query = str(kwargs.get("query", ""))
            if "site:g1.globo.com" not in query:
                return []
            return [
                {
                    "url": "https://g1.globo.com/politica/noticia/2022/02/14/tse-define-regras-para-a-retomada-da-propaganda-partidaria-gratuita.ghtml",
                    "title": "TSE define regras para a retomada da propaganda partidária gratuita | G1",
                    "snippet": (
                        "Matéria geral sobre propaganda partidária em Brasília com menção ao PSOL, "
                        "sem tratar do mesmo processo."
                    ),
                    "date": "2022-02-14",
                }
            ]

        async def fetch_page_summary(self, url, timeout=sjur.PERPLEXITY_PAGE_FETCH_TIMEOUT):
            return {
                "status_code": 200,
                "url": url,
                "title": "TSE define regras para a retomada da propaganda partidária gratuita | G1",
                "text": (
                    "Matéria geral sobre propaganda partidária gratuita. "
                    "Brasília DF e PSOL aparecem no texto, mas sem origem específica, "
                    "sem número do processo e sem detalhe individualizado do caso."
                ),
                "content_type": "text/html",
            }

    result = asyncio.run(
        sjur.buscar_todas_noticias_async(
            FakeManager(),
            payload,
            precisa_tse=False,
            precisa_tre=False,
            precisa_gerais=True,
            needed_general_count=1,
            timeout_seconds=1,
            max_tokens=120,
        )
    )

    assert result["status"] == "ok"
    assert result["gerais"] is None


def test_search_based_lookup_rejects_general_thematic_article_without_geo_anchor():
    payload = _lookup_payload(1)
    payload.update(
        {
            "tema": "Fraude à cota de gênero nas eleições proporcionais",
            "punchline": "Discussão sobre candidatura fictícia e seus efeitos nos diplomas.",
            "assuntos": "Cota gênero",
            "descricao_classe": "AIME",
            "nome_tipo_processo": "Acao de Impugnacao de Mandato Eletivo",
            "sigla_uf": "BA",
            "nome_municipio": "MORRO DO CHAPEU",
            "origem": "BA MORRO DO CHAPEU",
            "tribunal": "TRE-BA BA",
            "partes": "CIDADANIA",
            "relator": "Min. Nunes Marques",
        }
    )

    class FakeManager:
        async def search_news_results(self, **kwargs):
            query = str(kwargs.get("query", ""))
            if "site:poder360.com.br" not in query:
                return []
            return [
                {
                    "url": "https://www.poder360.com.br/poder-eleicoes/tse-confirma-que-cidadania-cometeu-fraude-a-cota-de-genero-em-2020/",
                    "title": "TSE confirma que Cidadania cometeu fraude à cota de gênero em 2020",
                    "snippet": (
                        "Reportagem sobre fraude à cota de gênero envolvendo o Cidadania "
                        "em caso diverso, sem mencionar o município do processo analisado."
                    ),
                    "date": "2026-03-18",
                }
            ]

        async def fetch_page_summary(self, url, timeout=sjur.PERPLEXITY_PAGE_FETCH_TIMEOUT):
            return {
                "status_code": 200,
                "url": url,
                "title": "TSE confirma que Cidadania cometeu fraude à cota de gênero em 2020",
                "text": (
                    "Matéria sobre fraude à cota de gênero envolvendo o Cidadania. "
                    "O texto menciona apenas BA de forma isolada, sem citar Morro do Chapéu "
                    "nem a origem específica do caso atual."
                ),
                "content_type": "text/html",
            }

    result = asyncio.run(
        sjur.buscar_todas_noticias_async(
            FakeManager(),
            payload,
            precisa_tse=False,
            precisa_tre=False,
            precisa_gerais=True,
            needed_general_count=1,
            timeout_seconds=1,
            max_tokens=120,
        )
    )

    assert result["status"] == "ok"
    assert result["gerais"] is None


def test_search_based_lookup_rejects_generic_official_listing_pages():
    class FakeManager:
        async def search_news_results(self, **kwargs):
            query = str(kwargs.get("query", ""))
            if "site:tse.jus.br/comunicacao/noticias" in query:
                return [
                    {
                        "url": "https://www.tse.jus.br/comunicacao/noticias/2024/Marco/confira-a-pauta-de-julgamentos-do-tse-desta-quinta-feira-21",
                        "title": "Confira a pauta de julgamentos do TSE desta quinta-feira (21)",
                        "snippet": "O processo 06000001-00.2024.6.26.0000 sera julgado nesta sessao.",
                        "date": "2024-03-21",
                    }
                ]
            return []

        async def fetch_page_summary(self, url, timeout=sjur.PERPLEXITY_PAGE_FETCH_TIMEOUT):
            return {
                "status_code": 200,
                "url": url,
                "title": "Confira a pauta de julgamentos do TSE desta quinta-feira (21)",
                "text": (
                    "06000001-00.2024.6.26.0000 06000002-00.2024.6.26.0000 "
                    "06000003-00.2024.6.26.0000"
                ),
                "content_type": "text/html",
            }

    result = asyncio.run(
        sjur.buscar_todas_noticias_async(
            FakeManager(),
            _lookup_payload(1),
            precisa_tse=True,
            precisa_tre=False,
            precisa_gerais=False,
            timeout_seconds=1,
            max_tokens=120,
        )
    )

    assert result["status"] == "ok"
    assert result["tse"] is None


def test_search_based_lookup_rejects_same_people_but_wrong_case_article():
    class FakeManager:
        async def search_news_results(self, **kwargs):
            query = str(kwargs.get("query", ""))
            if "site:tse.jus.br/comunicacao/noticias" in query:
                return [
                    {
                        "url": "https://www.tse.jus.br/comunicacao/noticias/2018/Setembro/plenario-nega-representacoes-de-coligacoes-e-de-candidatos-a-presidente-da-republica",
                        "title": "Plenário nega representações de coligações e de candidatos a presidente da República",
                        "snippet": "Cármen Lúcia relatou pedidos envolvendo Guilherme Boulos, sem relação com São Paulo ou com o recurso especial de 2024.",
                        "date": "2018-09-25",
                    }
                ]
            return []

        async def fetch_page_summary(self, url, timeout=sjur.PERPLEXITY_PAGE_FETCH_TIMEOUT):
            return {
                "status_code": 200,
                "url": url,
                "title": "Plenário nega representações de coligações e de candidatos a presidente da República",
                "text": (
                    "Cármen Lúcia relatou pedidos envolvendo Guilherme Boulos e coligações nacionais "
                    "sem mencionar município, UF de origem ou a classe recurso especial eleitoral."
                ),
                "content_type": "text/html",
            }

    result = asyncio.run(
        sjur.buscar_todas_noticias_async(
            FakeManager(),
            _lookup_payload(1),
            precisa_tse=True,
            precisa_tre=False,
            precisa_gerais=False,
            timeout_seconds=1,
            max_tokens=120,
        )
    )

    assert result["status"] == "ok"
    assert result["tse"] is None


def test_search_based_lookup_rejects_official_institutional_page_without_case_specific_signals():
    class FakeManager:
        async def search_news_results(self, **kwargs):
            query = str(kwargs.get("query", ""))
            if "site:tse.jus.br/comunicacao/noticias" in query:
                return [
                    {
                        "url": "https://www.tse.jus.br/comunicacao/noticias/2025/Agosto/tse-realiza-solenidade-pela-posse-dos-ministros-floriano-de-azevedo-marques-e-estela-aranha",
                        "title": "TSE realiza solenidade pela posse dos ministros Floriano de Azevedo Marques e Estela Aranha",
                        "snippet": "Brasília, Partido Renovação Democrática e Podemos são citados no texto institucional.",
                        "date": "2025-08-20",
                    }
                ]
            return []

        async def fetch_page_summary(self, url, timeout=sjur.PERPLEXITY_PAGE_FETCH_TIMEOUT):
            return {
                "status_code": 200,
                "url": url,
                "title": "TSE realiza solenidade pela posse dos ministros Floriano de Azevedo Marques e Estela Aranha",
                "text": (
                    "Brasília recebe solenidade institucional do Tribunal Superior Eleitoral. "
                    "Floriano de Azevedo Marques participa do evento. "
                    "Partido Renovação Democrática e Podemos aparecem apenas em menções laterais, "
                    "sem número do processo, sem origem do caso e sem detalhes do julgamento."
                ),
                "content_type": "text/html",
            }

    result = asyncio.run(
        sjur.buscar_todas_noticias_async(
            FakeManager(),
            _lookup_payload(1),
            precisa_tse=True,
            precisa_tre=False,
            precisa_gerais=False,
            timeout_seconds=1,
            max_tokens=120,
        )
    )

    assert result["status"] == "ok"
    assert result["tse"] is None


def test_search_based_lookup_stops_after_first_valid_query(monkeypatch):
    queries_seen: list[str] = []

    monkeypatch.setattr(
        sjur,
        "build_news_search_queries",
        lambda *_args, **_kwargs: [
            ("shared_exact", "q1"),
            ("tse_context", "q2"),
            ("tre_context", "q3"),
        ],
    )

    class FakeManager:
        async def search_news_results(self, **kwargs):
            query = str(kwargs.get("query", ""))
            queries_seen.append(query)
            if query != "q1":
                return []
            return [
                {
                    "url": "https://www.tse.jus.br/comunicacao/noticias/2026/03/caso-tse-valido",
                    "title": "TSE analisa AIJE em São Paulo com Parte 1",
                    "snippet": "06000001-00.2024.6.26.0000 Parte 1 São Paulo AIJE",
                    "date": "2026-03-18",
                },
                {
                    "url": "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/caso-tre-valido",
                    "title": "TRE-SP julga AIJE em São Paulo com Parte 1",
                    "snippet": "06000001-00.2024.6.26.0000 Parte 1 São Paulo AIJE",
                    "date": "2026-03-18",
                },
                {
                    "url": "https://g1.globo.com/politica/noticia/2026/03/caso-geral-valido.ghtml",
                    "title": "Caso eleitoral em São Paulo envolve Parte 1",
                    "snippet": "06000001-00.2024.6.26.0000 Parte 1 São Paulo AIJE",
                    "date": "2026-03-18",
                },
                {
                    "url": "https://www.cnnbrasil.com.br/politica/caso-geral-valido-2/",
                    "title": "Segunda cobertura do caso eleitoral em São Paulo",
                    "snippet": "06000001-00.2024.6.26.0000 Parte 1 São Paulo AIJE",
                    "date": "2026-03-18",
                },
            ]

        async def fetch_page_summary(self, url, timeout=sjur.PERPLEXITY_PAGE_FETCH_TIMEOUT):
            return {
                "status_code": 200,
                "url": url,
                "title": "Caso eleitoral válido em São Paulo",
                "text": (
                    "06000001-00.2024.6.26.0000 Parte 1 São Paulo AIJE "
                    "Ministro Teste Tema 1 Punchline 1 abuso de poder "
                    "Origem SP São Paulo acao de investigacao judicial eleitoral "
                    "caso noticiado com detalhes do julgamento e do processo."
                ),
                "content_type": "text/html",
            }

    result = asyncio.run(
        sjur.buscar_todas_noticias_async(
            FakeManager(),
            _lookup_payload(1),
            precisa_tse=True,
            precisa_tre=True,
            precisa_gerais=True,
            timeout_seconds=1,
            max_tokens=120,
        )
    )

    assert queries_seen == ["q1"]
    assert result["status"] == "ok"
    assert result["tse"] == "https://www.tse.jus.br/comunicacao/noticias/2026/03/caso-tse-valido"
    assert result["tre"] == "https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/caso-tre-valido"
    assert result["gerais"] == [
        "https://g1.globo.com/politica/noticia/2026/03/caso-geral-valido.ghtml",
        "https://www.cnnbrasil.com.br/politica/caso-geral-valido-2/",
    ]


def test_relator_match_variants_do_not_fall_back_to_single_surname():
    variants = sjur._relator_match_variants("Min. Estela Aranha")  # pylint: disable=protected-access

    assert "estela aranha" in variants
    assert "aranha" not in variants


def _v2_metadata_config() -> sjur_v2.MetadataExtractionConfig:
    return sjur_v2.MetadataExtractionConfig(
        include_institutional_entities=True,
        header_max_chars=sjur_v2.DEFAULT_METADATA_HEADER_MAX_CHARS,
    )


def test_v2_extracts_attorney_name_without_leaving_parenthesis():
    text = (
        "TRIBUNAL SUPERIOR ELEITORAL RECORRENTE: PARTIDO SOCIALISTA BRASILEIRO (PSB) - ESTADUAL "
        "ADVOGADA: LUZIA SANTOS GOIS (OAB/SE 3.136-A) DECISÃO"
    )

    metadata = sjur_v2.extract_header_metadata(text, config=_v2_metadata_config())

    assert metadata["partes"] == ["PARTIDO SOCIALISTA BRASILEIRO (PSB) - ESTADUAL"]
    assert metadata["advogados"] == ["LUZIA SANTOS GOIS"]


def test_v2_extracts_attorney_with_oab_colon_and_ignores_e_outro():
    text = (
        "Relator: Ministro Floriano de Azevedo Marques Recorrentes: Francisco André Régis Júnior e outro "
        "Advogados: Felipe Augusto Cortez Meira de Medeiros - OAB: 3640/RN e outro "
        "Recorrida: Coligação Vontade do Povo DECISÃO"
    )

    metadata = sjur_v2.extract_header_metadata(text, config=_v2_metadata_config())

    assert metadata["advogados"] == ["Felipe Augusto Cortez Meira de Medeiros"]


def test_v2_extracts_multiple_attorneys_from_representantes_do_a():
    text = (
        "RELATOR: MINISTRO ANDRÉ MENDONÇA AGRAVANTE: VALDECIR PARNOFF DOS SANTOS, "
        "SOLIDARIEDADE (SOLIDARIEDADE) - MUNICIPAL "
        "Representantes do(a) AGRAVANTE: JOAO FELIPE CASCO MIRANDA - PR96163, "
        "DHIOGO RAPHAEL ANOIZ - PR58623, WELINGTON EDUARDO LUDKE - PR36906 DECISÃO"
    )

    metadata = sjur_v2.extract_header_metadata(text, config=_v2_metadata_config())

    assert metadata["advogados"] == [
        "JOAO FELIPE CASCO MIRANDA",
        "DHIOGO RAPHAEL ANOIZ",
        "WELINGTON EDUARDO LUDKE",
    ]


def test_v2_advogado_block_stops_at_next_party_label():
    text = (
        "Relator: Ministro Antonio Carlos Ferreira Reclamante: Luís Carlos Domiciano "
        "Advogado: Ademar Francisco Martins Neto "
        "Reclamado: Tribunal Regional Eleitoral de São Paulo DECISÃO"
    )

    metadata = sjur_v2.extract_header_metadata(text, config=_v2_metadata_config())

    assert metadata["advogados"] == ["Ademar Francisco Martins Neto"]
    assert metadata["partes"] == [
        "Luís Carlos Domiciano",
        "Tribunal Regional Eleitoral de São Paulo",
    ]


def test_v2_representantes_da_does_not_contaminate_partes():
    text = (
        "RELATOR: MINISTRO ANDRÉ MENDONÇA AGRAVANTE: MARIA CORDEIRO DE PAULA "
        "Representantes da AGRAVANTE: DAIANE ALVES STOPA - RO7832, "
        "CARLOS EDUARDO ROCHA ALMEIDA - RO3593-A DECISÃO"
    )

    metadata = sjur_v2.extract_header_metadata(text, config=_v2_metadata_config())

    assert metadata["partes"] == ["MARIA CORDEIRO DE PAULA"]
    assert metadata["advogados"] == [
        "DAIANE ALVES STOPA",
        "CARLOS EDUARDO ROCHA ALMEIDA",
    ]


@pytest.mark.parametrize(
    ("text", "expected_partes"),
    [
        (
            "RECORRENTE: PARTIDO SOCIALISMO E LIBERDADE (PSOL) - NACIONAL DECISÃO",
            ["PARTIDO SOCIALISMO E LIBERDADE (PSOL) - NACIONAL"],
        ),
        (
            "RECORRENTE: COLIGAÇÃO PELA UNIÃO E RECONSTRUÇÃO DE RERIUTABA DECISÃO",
            ["COLIGAÇÃO PELA UNIÃO E RECONSTRUÇÃO DE RERIUTABA"],
        ),
        (
            "RECORRENTE: A F SOARES MARKETING E PESQUISAS DE MERCADO LTDA - EPP DECISÃO",
            ["A F SOARES MARKETING E PESQUISAS DE MERCADO LTDA - EPP"],
        ),
    ],
)
def test_v2_does_not_split_entities_that_use_e_in_name(text, expected_partes):
    metadata = sjur_v2.extract_header_metadata(text, config=_v2_metadata_config())
    assert metadata["partes"] == expected_partes


def test_v2_recognizes_reclamante_reclamado_and_paciente_roles():
    text = (
        "RECLAMANTE: LUÍS CARLOS DOMICIANO RECLAMADO: TRIBUNAL REGIONAL ELEITORAL DE SÃO PAULO "
        "PACIENTE: ROBERTO PINTO DOS SANTOS DECISÃO"
    )

    metadata = sjur_v2.extract_header_metadata(text, config=_v2_metadata_config())

    assert metadata["partes"] == [
        "LUÍS CARLOS DOMICIANO",
        "TRIBUNAL REGIONAL ELEITORAL DE SÃO PAULO",
        "ROBERTO PINTO DOS SANTOS",
    ]


def test_v2_process_one_csv_merges_header_partes_and_attorneys(tmp_path):
    input_path = tmp_path / "entrada_v2.csv"
    out_dir = tmp_path / "saida_v2"
    row = _raw_row(91)
    row["partes"] = "MINISTÉRIO PÚBLICO ELEITORAL"
    row["textoDecisao"] = (
        "TRIBUNAL SUPERIOR ELEITORAL REQUERENTE: MINISTÉRIO PÚBLICO ELEITORAL "
        "REQUERIDOS: ALAN AQUINO GUEDES DE MENDONÇA E OUTRA "
        "ADVOGADOS: OSCAR HENRIQUE PERES DE SOUZA KRUGER (OAB/MS 14.369) E OUTRO DECISÃO"
    )
    _write_csv(input_path, RAW_FIELDS, [row])

    summary = sjur_v2.process_one_csv(
        input_path=input_path,
        out_dir=out_dir,
        max_texto_chars=0,
        replace_newlines=True,
        web_lookup_config=sjur_v2.WebLookupConfig(enabled=False, api_key=""),
        tema_punchline_config=sjur_v2.TemaPunchlineConfig(enabled=False, api_key=""),
        assuntos_enrichment_config=sjur_v2.AssuntosEnrichmentConfig(enabled=False, api_key=""),
        metadata_extraction_config=_v2_metadata_config(),
        logger=_noop_logger,
    )

    rows = _read_csv_rows(summary.output_path)

    assert rows[0]["partes"] == "MINISTÉRIO PÚBLICO ELEITORAL,ALAN AQUINO GUEDES DE MENDONÇA"
    assert rows[0]["advogados"] == "OSCAR HENRIQUE PERES DE SOUZA KRUGER"
    assert "(" not in rows[0]["advogados"]
