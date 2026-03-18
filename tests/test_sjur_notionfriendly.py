import asyncio
import csv
import json
from pathlib import Path

import pytest

import SJUR_csv_to_csv_NOTIONfriendly as sjur


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
    monkeypatch.setattr(sjur, "ARTIFACTS_ROOT", tmp_path / "Artefatos")


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


def test_news_lookup_uses_one_call_per_pending_row(tmp_path, monkeypatch):
    calls: list[str] = []

    async def fake_lookup(_manager, page_data, **_kwargs):
        calls.append(str(page_data.get("numero_unico", "")))
        case_id = len(calls)
        return {
            "status": "ok",
            "tse": f"https://www.tse.jus.br/comunicacao/noticias/2026/03/tse-{case_id}",
            "tre": f"https://www.tre-sp.jus.br/comunicacao/noticias/2026/03/tre-{case_id}",
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


def test_news_cache_skips_filled_and_no_match_entries(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    row_filled = _news_row(1)
    row_no_match = _news_row(2)
    payload_filled = _lookup_payload(1)
    payload_no_match = _lookup_payload(2)

    key_filled = sjur.build_news_request_key(sjur.build_news_lookup_request(payload_filled, model="sonar"))
    key_no_match = sjur.build_news_request_key(sjur.build_news_lookup_request(payload_no_match, model="sonar"))
    sjur.write_news_cache(
        cache_path,
        {
            key_filled: {
                "status": "filled",
                "tse": "https://www.tse.jus.br/comunicacao/noticias/2026/03/cache-filled",
                "tre": "",
                "gerais": [],
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
    assert row_no_match["noticia_TSE"] == ""


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


def test_news_prompt_includes_official_first_query_hints():
    hints = sjur.build_news_query_hints(_lookup_payload(1))
    prompt = sjur.gerar_prompt_noticias(_lookup_payload(1), max_general_urls=2)

    assert hints["tre_domain"] == "tre-sp.jus.br"
    assert 'site:tse.jus.br/comunicacao/noticias "06000001-00.2024.6.26.0000"' in prompt
    assert 'site:tre-sp.jus.br/comunicacao/noticias "06000001-00.2024.6.26.0000"' in prompt
    assert 'Ignore totalmente resultados sobre regras gerais das eleicoes' in prompt
