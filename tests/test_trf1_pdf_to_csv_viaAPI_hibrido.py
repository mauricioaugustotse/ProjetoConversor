#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import TRF1_pdf_to_csv_viaAPI_hibrido as trf1


def _logger() -> logging.Logger:
    lg = logging.getLogger("test_trf1_hibrido")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.DEBUG)
    return lg


class TestTRF1HibridoHelpers(unittest.TestCase):
    def test_schema_without_duplicates(self) -> None:
        self.assertNotIn("numero", trf1.CSV_COLUMNS)
        self.assertNotIn("Tipo_de_processo", trf1.CSV_COLUMNS)
        self.assertNotIn("ementa", trf1.CSV_COLUMNS)
        self.assertNotIn("resultado", trf1.CSV_COLUMNS)
        self.assertIn("classe", trf1.CSV_COLUMNS)
        self.assertIn("relator(a)", trf1.CSV_COLUMNS)
        self.assertIn("texto_do_boletim", trf1.CSV_COLUMNS)
        self.assertIn("data_publicacao", trf1.CSV_COLUMNS)

    def test_data_julgamento_date_simple(self) -> None:
        self.assertEqual(trf1.parse_date_fragment_to_mdy("21/01/2020"), "1/21/2020")

    def test_data_julgamento_virtual_period(self) -> None:
        frag = "sessão virtual realizada no período de 26/01 a 02/02/2026"
        self.assertEqual(trf1.parse_date_fragment_to_mdy(frag), "2/2/2026")

    def test_data_julgamento_compact_period(self) -> None:
        frag = "sessão virtual realizada no período de 26 a 30/01/2026"
        self.assertEqual(trf1.parse_date_fragment_to_mdy(frag), "1/30/2026")

    def test_data_julgamento_double_slash(self) -> None:
        self.assertEqual(trf1.parse_date_fragment_to_mdy("10/11//2020"), "11/10/2020")

    def test_data_publicacao_normalized_mdy(self) -> None:
        self.assertEqual(trf1.normalize_date_to_mdy("14/02/2020", source_hint="dmy"), "2/14/2020")

    def test_citation_regex_with_convocada(self) -> None:
        text = (
            "(ApReeNec 0011465-59.2018.4.01.9199, rel. juíza federal "
            "Leão Aparecido Alves (convocada), em 22/01/2020.)"
        )
        m = trf1.CITATION_RE.search(text)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(trf1.normalize_cnj_number(m.group("numero")), "0011465-59.2018.4.01.9199")
        self.assertEqual(trf1.parse_date_fragment_to_mdy(m.group("datefrag")), "1/22/2020")

    def test_normalize_cnj_number_fuzzy_variants(self) -> None:
        got_1, _ = trf1.normalize_cnj_number_fuzzy("(ReeNec 009513.82.2013.4.01.3100, rel. des. federal Fulano, em 15/06/2020.)")
        got_2, _ = trf1.normalize_cnj_number_fuzzy("(RSE 0006200.83-2017.4.01.3000, rel. des. federal Fulano, em 29/06/2020.)")
        got_3, _ = trf1.normalize_cnj_number_fuzzy("(Ap 0008644-82 2015.4.01.3801, rel. des. federal Fulano, em 05/08/2020.)")
        self.assertEqual(got_1, "0009513-82.2013.4.01.3100")
        self.assertEqual(got_2, "0006200-83.2017.4.01.3000")
        self.assertEqual(got_3, "0008644-82.2015.4.01.3801")

    def test_find_citation_segments_fuzzy_without_rel_and_em(self) -> None:
        text = (
            "Texto do caso. Unânime. "
            "(Ap 000198133.2018.4.01.3601, des. federal Ney Bello, 29/06/2021.) "
            "Outro texto."
        )
        segs = trf1.find_citation_segments(text)
        self.assertTrue(segs)
        self.assertEqual(segs[0]["numero"], "0001981-33.2018.4.01.3601")
        self.assertEqual(trf1.parse_date_fragment_to_mdy(segs[0]["datefrag"]), "6/29/2021")

    def test_normalize_relator_desembargador(self) -> None:
        self.assertEqual(trf1.normalize_relator("des. federal João da Silva"), "Des. João da Silva")

    def test_normalize_relator_convocado(self) -> None:
        self.assertEqual(
            trf1.normalize_relator("juiz federal Marlon Sousa (convocado)"),
            "Marlon Sousa (convocado)",
        )

    def test_normalize_relator_convocada(self) -> None:
        self.assertEqual(
            trf1.normalize_relator("juíza federal Maria Souza (convocada)"),
            "Maria Souza (convocada)",
        )

    def test_normalize_bullet_points_three(self) -> None:
        raw = ["Ponto, um", "Segundo ponto", "Terceiro, ponto", "Quarto ponto"]
        got = trf1.normalize_bullet_points_three(raw)
        self.assertEqual(got, "• Ponto - um,\n• Segundo ponto,\n• Terceiro - ponto,")
        self.assertNotIn(",,", got)
        self.assertTrue(got.startswith("• "))
        self.assertEqual(len(got.splitlines()), 3)
        self.assertTrue(all(line.endswith(",") for line in got.splitlines()))

    def test_normalize_bullet_points_three_empty(self) -> None:
        self.assertEqual(trf1.normalize_bullet_points_three(""), "")

    def test_normalize_bullet_points_three_from_single_line_bullets(self) -> None:
        raw = "• Item um, • Item dois, • Item três,"
        got = trf1.normalize_bullet_points_three(raw)
        self.assertEqual(got, "• Item um,\n• Item dois,\n• Item três,")
        self.assertEqual(len(got.splitlines()), 3)

    def test_split_multi_value_newlines(self) -> None:
        got = trf1.split_multi_value("• Um,\n• Dois,\n• Três,")
        self.assertEqual(got, ["• Um", "• Dois", "• Três"])

    def test_normalize_subramo_two(self) -> None:
        raw = "Subramo A, Subramo B, Subramo C"
        got = trf1.normalize_subramo_two(raw)
        self.assertEqual(got, "Subramo A, Subramo B")

    def test_is_noise_line_ocr_header(self) -> None:
        self.assertTrue(
            trf1.is_noise_line(
                "oletim nformativo de urisprudência EssE informativo contém notícias não oficiais"
            )
        )

    def test_openai_done_requires_critical_fields(self) -> None:
        row = trf1.base_empty_row()
        row["contexto"] = "ctx"
        row["tese"] = "tese"
        row["tema"] = "tema"
        row["ramo_do_direito"] = "Direito Administrativo"
        row["subramo_do_direito"] = "Licitações"
        row["punchline"] = "Frase curta"
        row["bullet_points"] = "Um, Dois"
        self.assertFalse(trf1.has_valid_openai_critical_fields(row))
        row["tema"] = "Antecipação de colação de grau para candidato aprovado em concurso público"
        row["subramo_do_direito"] = "Licitações, Contratos"
        row["bullet_points"] = "• Um,\n• Dois,\n• Três,"
        self.assertTrue(trf1.has_valid_openai_critical_fields(row))

    def test_normalize_legislacao(self) -> None:
        raw = ['CF/88 art. 61, § 1º, II, "a"', "CF/88 art. 61 § 1º II a", "CTB art. 165-A"]
        got = trf1.normalize_legislacao(raw)
        # Sem aspas/vírgulas internas por tag e deduplicado
        self.assertNotIn('"', got)
        self.assertIn("CF/88 art. 61 § 1º II a", got)
        self.assertIn("CTB art. 165-A", got)

    def test_extract_texto_do_boletim_excludes_citation(self) -> None:
        base = "Concurso público. Texto do caso. Unânime."
        citation = "(MS 1020604-23.2025.4.01.0000 – PJe, rel. des. federal Pedro Braga Filho, em 18/12/2025.)"
        block = f"{base} {citation}"
        got = trf1.extract_texto_do_boletim_from_block(block, citation)
        self.assertEqual(got, base)
        self.assertNotIn("(MS 1020604-23.2025.4.01.0000", got)
        self.assertTrue(got.endswith("Unânime."))

    def test_extract_texto_do_boletim_normalizes_linebreaks(self) -> None:
        base = "Linha 1 do caso.\nLinha 2 do caso.\nUnânime."
        citation = "(Ap 0003573-37.2012.4.01.3500, rel. des. federal Fulano, em 22/01/2020.)"
        block = f"{base}\n{citation}"
        got = trf1.extract_texto_do_boletim_from_block(block, citation)
        self.assertEqual(got, "Linha 1 do caso. Linha 2 do caso. Unânime.")
        self.assertNotIn("\n", got)

    def test_tema_fallback_never_empty(self) -> None:
        row = trf1.base_empty_row()
        row["texto_do_boletim"] = (
            "Anistia. Readmissão de empregado regido pela CLT. "
            "Transposição para o RJU. Impossibilidade."
        )
        row["tema"] = ""
        trf1.ensure_output_rows_format([row])
        self.assertTrue(row["tema"])

    def test_parse_decisao_detects_unanimidade(self) -> None:
        txt = "Reconhecida a necessidade de adequação do ato administrativo. Unânime."
        self.assertEqual(trf1.parse_decisao(txt), "unanimidade")

    def test_parse_decisao_detects_por_maioria(self) -> None:
        txt = "Nulidade processual arguida. Por maioria."
        self.assertEqual(trf1.parse_decisao(txt), "maioria")

    def test_parse_decisao_detects_maioria_tail_sentence(self) -> None:
        txt = (
            "Em decorrência de mera organização administrativa da vara de origem, "
            "não se verificando prejuízo à defesa. Maioria."
        )
        self.assertEqual(trf1.parse_decisao(txt), "maioria")

    def test_parse_decisao_empty_when_not_explicit(self) -> None:
        txt = "Anistia. Transposição de regime jurídico. Precedente do STF."
        self.assertEqual(trf1.parse_decisao(txt), "")

    def test_ensure_output_rows_sets_decisao_when_missing(self) -> None:
        row = trf1.base_empty_row()
        row["texto_do_boletim"] = "Apelação desprovida. Unânime."
        row["decisao"] = ""
        trf1.ensure_output_rows_format([row])
        self.assertEqual(row["decisao"], "unanimidade")

    def test_openai_prompt_contains_texto_do_boletim(self) -> None:
        row = trf1.base_empty_row()
        row["texto_do_boletim"] = "Texto literal do boletim para análise."
        row["classe"] = "Ap"
        row["numero_processo"] = "0000001-11.2020.4.01.0000"
        prompt = trf1.build_openai_prompt(row)
        self.assertIn("TEXTO PRINCIPAL para análise", prompt)
        self.assertIn("classe: Ap", prompt)
        self.assertIn("texto_do_boletim: Texto literal do boletim para análise.", prompt)
        self.assertIn("tema: entre 8 e 16 palavras", prompt)
        self.assertNotIn("Tipo_de_processo", prompt)
        self.assertNotIn("ementa:", prompt)

    def test_is_tema_detailed_bounds(self) -> None:
        self.assertFalse(trf1.is_tema_detailed("Ensino superior"))
        self.assertTrue(
            trf1.is_tema_detailed(
                "Antecipação de colação de grau para candidato aprovado em concurso público"
            )
        )

    def test_perplexity_prompt_contains_texto_do_boletim(self) -> None:
        row = trf1.base_empty_row()
        row["texto_do_boletim"] = "Texto base para notícia."
        row["classe"] = "Ap"
        prompt = trf1.build_perplexity_prompt(row)
        self.assertIn("Use \"texto_do_boletim\" como fonte principal de contexto.", prompt)
        self.assertIn("REGRAS ESTRITAS", prompt)
        self.assertIn("- classe: Ap", prompt)
        self.assertIn("- texto_do_boletim: Texto base para notícia.", prompt)

    def test_parse_perplexity_content_payload(self) -> None:
        got = trf1.parse_perplexity_content(
            json.dumps(
                {
                    "noticia": "https://g1.globo.com/politica/noticia/2026/02/18/exemplo.ghtml",
                    "evidencia": "Processo 0000001-11.2020.4.01.0000 julgado pelo TRF1.",
                    "confianca": "alta",
                }
            )
        )
        self.assertEqual(got["confianca"], "alta")
        self.assertIn("https://g1.globo.com", got["noticia"])

    def test_parse_perplexity_content_candidates_schema(self) -> None:
        got = trf1.parse_perplexity_content(
            json.dumps(
                {
                    "candidatos": [
                        {
                            "url": "https://g1.globo.com/politica/noticia/2026/02/18/exemplo.ghtml",
                            "evidencia": "Processo 0000001-11.2020.4.01.0000 no TRF1.",
                            "confianca": "alta",
                        }
                    ]
                }
            )
        )
        self.assertEqual(len(got["candidatos"]), 1)
        self.assertEqual(got["candidatos"][0]["confianca"], "alta")
        self.assertIn("https://g1.globo.com", got["candidatos"][0]["url"])

    def test_strict_noticia_candidate_rejects_non_news_url(self) -> None:
        row = trf1.base_empty_row()
        row["numero_processo"] = "0000001-11.2020.4.01.0000"
        row["relator(a)"] = "Des. João da Silva"
        row["tema"] = "Concurso público"
        row["contexto"] = "Anulação de psicotécnico"
        ok = trf1.is_strict_noticia_candidate(
            row=row,
            url="https://consultadocumento.tst.jus.br/consultaDocumento/acordao.do?x=1",
            evidence="Processo 0000001-11.2020.4.01.0000 julgado pelo TRF1.",
            confianca="alta",
        )
        self.assertFalse(ok)

    def test_news_url_requires_mainstream_domain(self) -> None:
        self.assertFalse(trf1.is_probably_news_url("https://meublogjuridico.com/noticia/caso-trf1"))

    def test_news_url_accepts_official_domain(self) -> None:
        self.assertTrue(
            trf1.is_probably_news_url(
                "https://www.trf1.jus.br/trf1/noticias/caso-exemplo-julgado-pela-corte"
            )
        )
        self.assertTrue(
            trf1.is_probably_news_url(
                "https://www.gov.br/agu/pt-br/comunicacao/noticias/agu-obtem-decisao-no-trf1"
            )
        )

    def test_news_url_rejects_blog_or_opinion_section(self) -> None:
        self.assertFalse(trf1.is_probably_news_url("https://g1.globo.com/blog/analise-do-caso.ghtml"))
        self.assertFalse(trf1.is_probably_news_url("https://www.estadao.com.br/opiniao/artigo-exemplo/"))

    def test_strict_noticia_candidate_accepts_clear_match(self) -> None:
        row = trf1.base_empty_row()
        row["numero_processo"] = "0000001-11.2020.4.01.0000"
        row["relator(a)"] = "Des. João da Silva"
        row["tema"] = "Concurso público"
        row["contexto"] = "Anulação de psicotécnico"
        ok = trf1.is_strict_noticia_candidate(
            row=row,
            url="https://g1.globo.com/politica/noticia/2026/02/18/caso-trf1.ghtml",
            evidence="Processo 0000001-11.2020.4.01.0000 tratou de concurso público no TRF1.",
            confianca="alta",
        )
        self.assertTrue(ok)

    def test_select_best_candidate_prioritizes_mainstream(self) -> None:
        row = trf1.base_empty_row()
        row["numero_processo"] = "0000001-11.2020.4.01.0000"
        row["relator(a)"] = "Des. João da Silva"
        row["tema"] = "Concurso público"
        cands = [
            {
                "url": "https://www.trf1.jus.br/trf1/noticias/decisao-sobre-concurso-publico",
                "evidencia": "TRF1 julgou processo 0000001-11.2020.4.01.0000.",
                "confianca": "alta",
            },
            {
                "url": "https://g1.globo.com/politica/noticia/2026/02/18/caso-trf1.ghtml",
                "evidencia": "Processo 0000001-11.2020.4.01.0000 tratado no TRF1.",
                "confianca": "alta",
            },
        ]
        url, reason = trf1.select_best_noticia_candidate(
            row,
            cands,
            min_score_mainstream=2,
            min_score_official=1,
            domain_policy="consagrados_oficiais",
            stage=1,
        )
        self.assertIn("g1.globo.com", url)
        self.assertEqual(reason, "accepted_stage1_mainstream")

    def test_select_best_candidate_falls_back_to_official(self) -> None:
        row = trf1.base_empty_row()
        row["numero_processo"] = "0000001-11.2020.4.01.0000"
        row["relator(a)"] = "Des. João da Silva"
        row["tema"] = "Concurso público"
        cands = [
            {
                "url": "https://g1.globo.com/politica/noticia/2026/02/18/caso-trf1.ghtml",
                "evidencia": "Texto genérico sem aderência.",
                "confianca": "alta",
            },
            {
                "url": "https://www.trf1.jus.br/trf1/noticias/decisao-sobre-concurso-publico",
                "evidencia": "TRF1 julgou o processo 0000001-11.2020.4.01.0000 sobre concurso público.",
                "confianca": "alta",
            },
        ]
        url, reason = trf1.select_best_noticia_candidate(
            row,
            cands,
            min_score_mainstream=3,
            min_score_official=1,
            domain_policy="consagrados_oficiais",
            stage=1,
        )
        self.assertIn("trf1.jus.br", url)
        self.assertEqual(reason, "accepted_stage1_official")

    def test_select_best_candidate_stage2_relaxes_vs_stage1(self) -> None:
        row = trf1.base_empty_row()
        row["numero_processo"] = "0000001-11.2020.4.01.0000"
        row["relator(a)"] = "Des. João da Silva"
        row["tema"] = "Psicotécnico aptidão candidato concurso"
        row["contexto"] = "Anulação em concurso público"
        cands = [
            {
                "url": "https://g1.globo.com/politica/noticia/2026/02/18/trf1-decide-caso.ghtml",
                "evidencia": "Julgamento sobre psicotécnico e aptidão de candidato em concurso.",
                "confianca": "alta",
            }
        ]
        url_stage1, reason_stage1 = trf1.select_best_noticia_candidate(
            row,
            cands,
            min_score_mainstream=2,
            min_score_official=1,
            domain_policy="consagrados_oficiais",
            stage=1,
        )
        self.assertEqual(url_stage1, "")
        self.assertEqual(reason_stage1, "score_below_threshold")

        url_stage2, reason_stage2 = trf1.select_best_noticia_candidate(
            row,
            cands,
            min_score_mainstream=2,
            min_score_official=1,
            domain_policy="consagrados_oficiais",
            stage=2,
        )
        self.assertIn("g1.globo.com", url_stage2)
        self.assertEqual(reason_stage2, "accepted_stage2_mainstream")

    def test_strict_noticia_candidate_accepts_official_source_with_strong_evidence(self) -> None:
        row = trf1.base_empty_row()
        row["numero_processo"] = "0000001-11.2020.4.01.0000"
        row["relator(a)"] = "Des. João da Silva"
        row["tema"] = "Concurso público"
        row["contexto"] = "Anulação de psicotécnico"
        ok = trf1.is_strict_noticia_candidate(
            row=row,
            url="https://www.trf1.jus.br/trf1/noticias/decisao-sobre-concurso-publico",
            evidence="TRF1 julgou o processo 0000001-11.2020.4.01.0000 sobre concurso público.",
            confianca="baixa",
        )
        self.assertTrue(ok)

    def test_strict_noticia_candidate_rejects_official_without_url_anchor(self) -> None:
        row = trf1.base_empty_row()
        row["numero_processo"] = "0000001-11.2020.4.01.0000"
        row["relator(a)"] = "Des. João da Silva"
        row["tema"] = "Concurso público"
        row["contexto"] = "Anulação de psicotécnico"
        ok = trf1.is_strict_noticia_candidate(
            row=row,
            url="https://www.cmd.mg.gov.br/portal/noticias/0/3/33185/boletim-epidemiologico-entenda-como-o-informe-diario-e-construido",
            evidence="TRF1 julgou o processo 0000001-11.2020.4.01.0000 sobre concurso público.",
            confianca="alta",
        )
        self.assertFalse(ok)

    def test_quality_report_includes_new_perplexity_metrics(self) -> None:
        rows = []
        for reason in ("accepted_stage1_mainstream", "accepted_stage2_official", "score_below_threshold"):
            row = trf1.base_empty_row()
            row["_row_id"] = trf1.generate_row_id(row)
            row["_perplexity_reason"] = reason
            if reason.startswith("accepted_"):
                row["noticia"] = "https://g1.globo.com/politica/noticia/2026/02/18/exemplo.ghtml"
                row["_perplexity_done"] = "1"
            rows.append(row)
        report = trf1.build_quality_report(rows)
        metrics = report["metrics"]
        self.assertIn("noticia_fill_rate_pct", metrics)
        self.assertIn("perplexity_stage1_accept_count", metrics)
        self.assertIn("perplexity_stage2_accept_count", metrics)
        self.assertIn("perplexity_reject_by_reason", metrics)
        self.assertEqual(metrics["perplexity_stage1_accept_count"], 1)
        self.assertEqual(metrics["perplexity_stage2_accept_count"], 1)
        self.assertEqual(metrics["perplexity_reject_by_reason"].get("score_below_threshold"), 1)

    def test_cli_without_resume_flags(self) -> None:
        parser = trf1.build_arg_parser()
        help_text = parser.format_help()
        self.assertNotIn("--resume", help_text)
        self.assertNotIn("--no-resume", help_text)
        self.assertIn("--perplexity-fallback-enabled", help_text)
        self.assertIn("--perplexity-fallback-max-fraction", help_text)
        self.assertIn("--perplexity-min-score-mainstream", help_text)
        self.assertIn("--perplexity-min-score-official", help_text)
        self.assertIn("--perplexity-domain-policy", help_text)

    def test_checkpoint_payload_version_current(self) -> None:
        row = trf1.base_empty_row()
        row["texto_do_boletim"] = "Texto"
        row["_row_id"] = trf1.generate_row_id(row)
        payload = trf1.checkpoint_payload(
            manifest=[{"path": "x.pdf", "sha1": "abc"}],
            rows=[row],
            perplexity_state={"workers": 1, "delay": 0.4},
        )
        self.assertEqual(payload["version"], trf1.CHECKPOINT_VERSION)
        self.assertIn("texto_do_boletim", payload["rows"][0])


class TestTRF1HibridoIntegration(unittest.TestCase):
    def test_openai_perplexity_batches_with_mocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / trf1.OUTPUT_NAME
            cp = Path(tmp) / trf1.CHECKPOINT_NAME
            manifest = [{"path": "2020/Bij_507.pdf", "sha1": "abc"}]
            rows = []
            for i in range(2):
                row = trf1.base_empty_row()
                row["classe"] = "Ap"
                row["numero_processo"] = f"000000{i}-11.2020.4.01.0000"
                row["relator(a)"] = "Des. Fulano"
                row["orgao_julgador"] = "Primeira Turma"
                row["data_julgamento"] = "1/21/2020"
                row["data_publicacao"] = "2/14/2020"
                row["texto_do_boletim"] = "Texto literal do boletim. Unânime."
                row["informativo"] = "507"
                row["_row_id"] = trf1.generate_row_id(row)
                row["_openai_done"] = "0"
                row["_perplexity_done"] = "0"
                rows.append(row)

            openai_payload = {
                "contexto": "Contexto x",
                "tese": "Tese y",
                "tema": "Antecipação de colação de grau para candidato aprovado em concurso público",
                "ramo_do_direito": "Direito Administrativo",
                "subramo_do_direito": "Processo Administrativo, Licitações",
                "punchline": "Frase curta de até vinte palavras.",
                "bullet_points": "Ponto 1, Ponto 2, Ponto 3",
                "jurisprudência": "Tema 123",
                "legislacao": "CF/88 art. 37",
            }

            with patch.object(trf1, "OpenAI"), patch.object(
                trf1,
                "openai_call_single",
                return_value=(True, openai_payload, ""),
            ):
                trf1.run_openai_enrichment(
                    rows=rows,
                    config=trf1.OpenAIConfig(
                        enabled=True,
                        api_key="test",
                        model="gpt-5-mini",
                        batch_size=30,
                        max_workers=2,
                        delay=0.0,
                        retries=1,
                        timeout=10,
                    ),
                    output_csv=out_csv,
                    checkpoint_path=cp,
                    manifest=manifest,
                    perplexity_state={"workers": 2, "delay": 0.4},
                    logger=_logger(),
                )

            for row in rows:
                self.assertEqual(row["_openai_done"], "1")
                self.assertTrue(row["contexto"])
                self.assertTrue(row["legislacao"])

            with patch.object(
                trf1,
                "perplexity_call_single",
                return_value=(True, "https://example.com/noticia", False, "", "ok"),
            ):
                state = trf1.run_perplexity_enrichment(
                    rows=rows,
                    config=trf1.PerplexityConfig(
                        enabled=True,
                        api_key="test",
                        model="sonar",
                        batch_size=20,
                        max_workers=2,
                        delay=0.0,
                        retries=1,
                        timeout=10,
                        max_workers_cap=4,
                        fallback_enabled=True,
                        fallback_max_fraction=0.60,
                        min_score_mainstream=2,
                        min_score_official=1,
                        domain_policy="consagrados_oficiais",
                    ),
                    output_csv=out_csv,
                    checkpoint_path=cp,
                    manifest=manifest,
                    perplexity_state={"workers": 2, "delay": 0.4},
                    logger=_logger(),
                )
                self.assertIn("workers", state)
                self.assertIn("delay", state)

            self.assertTrue(out_csv.exists())
            self.assertTrue(cp.exists())
            saved = json.loads(cp.read_text(encoding="utf-8"))
            self.assertIn("rows", saved)
            self.assertEqual(len(saved["rows"]), 2)
            for row in rows:
                self.assertEqual(row["_perplexity_done"], "1")
                self.assertEqual(row["noticia"], "https://example.com/noticia")
                self.assertEqual(row["_perplexity_reason"], "ok")

    def test_perplexity_fallback_stage2_respects_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / trf1.OUTPUT_NAME
            cp = Path(tmp) / trf1.CHECKPOINT_NAME
            manifest = [{"path": "2020/Bij_508.pdf", "sha1": "abc"}]
            rows = []
            for i in range(5):
                row = trf1.base_empty_row()
                row["classe"] = "Ap"
                row["numero_processo"] = f"000000{i}-11.2020.4.01.0000"
                row["relator(a)"] = "Des. Fulano"
                row["texto_do_boletim"] = "Texto literal do boletim. Unânime."
                row["_row_id"] = trf1.generate_row_id(row)
                row["_perplexity_done"] = "0"
                row["_perplexity_reason"] = ""
                rows.append(row)

            def _fake_perplexity_call_single(*args, **kwargs):
                row = args[2]
                stage = kwargs.get("stage")
                if stage == 1:
                    return True, "", False, "", "no_candidates_stage1"
                return True, f"https://g1.globo.com/noticia/{row['_row_id']}", False, "", "accepted_stage2_mainstream"

            with patch.object(trf1, "perplexity_call_single", side_effect=_fake_perplexity_call_single):
                trf1.run_perplexity_enrichment(
                    rows=rows,
                    config=trf1.PerplexityConfig(
                        enabled=True,
                        api_key="test",
                        model="sonar",
                        batch_size=20,
                        max_workers=2,
                        delay=0.0,
                        retries=1,
                        timeout=10,
                        max_workers_cap=4,
                        fallback_enabled=True,
                        fallback_max_fraction=0.40,  # ceil(5*0.4)=2
                        min_score_mainstream=2,
                        min_score_official=1,
                        domain_policy="consagrados_oficiais",
                    ),
                    output_csv=out_csv,
                    checkpoint_path=cp,
                    manifest=manifest,
                    perplexity_state={"workers": 2, "delay": 0.0},
                    logger=_logger(),
                )
            filled = [r for r in rows if r.get("_perplexity_done") == "1"]
            stage2_filled = [r for r in rows if r.get("_perplexity_reason") == "accepted_stage2_mainstream"]
            self.assertEqual(len(filled), 2)
            self.assertEqual(len(stage2_filled), 2)
            self.assertEqual(sum(1 for r in rows if r.get("_perplexity_reason") == "no_candidates_stage1"), 3)

    def test_extract_rows_on_known_pdf_without_header_noise(self) -> None:
        pdf = Path("2020/Bij_510.pdf")
        if not pdf.exists():
            self.skipTest("Arquivo 2020/Bij_510.pdf não disponível no ambiente de teste.")
        rows = trf1.extract_rows_from_pdf(pdf, _logger())
        self.assertTrue(rows)
        self.assertFalse(rows[0]["texto_do_boletim"].lower().startswith("oletim nformativo"))
        self.assertRegex(rows[0]["data_julgamento"], r"^\d{1,2}/\d{1,2}/\d{4}$")
        if rows[0]["data_publicacao"]:
            self.assertRegex(rows[0]["data_publicacao"], r"^\d{1,2}/\d{1,2}/\d{4}$")

    def test_main_clean_run_ignores_legacy_csv_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf = tmp_dir / "dummy.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%dummy\n")

            out_csv = tmp_dir / trf1.OUTPUT_NAME
            checkpoint = tmp_dir / trf1.CHECKPOINT_NAME

            legacy_row = trf1.base_empty_row()
            legacy_row["classe"] = "Ap"
            legacy_row["numero_processo"] = "9999999-99.1999.4.01.0000"
            legacy_row["relator(a)"] = "Des. Legado"
            legacy_row["data_julgamento"] = "1/1/1999"
            legacy_row["data_publicacao"] = "1/2/1999"
            legacy_row["texto_do_boletim"] = "Texto legado."
            legacy_row["contexto"] = "LEGADO"
            trf1.write_csv(out_csv, [legacy_row])

            checkpoint.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "legacy_marker": "yes",
                        "manifest": [{"path": "old.pdf", "sha1": "old"}],
                        "rows": [legacy_row],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            new_row = trf1.base_empty_row()
            new_row["classe"] = "MS"
            new_row["numero_processo"] = "0000001-11.2020.4.01.0000"
            new_row["relator(a)"] = "Des. Novo"
            new_row["orgao_julgador"] = "Primeira Turma"
            new_row["data_julgamento"] = "1/21/2020"
            new_row["data_publicacao"] = "2/14/2020"
            new_row["texto_do_boletim"] = "Texto novo. Unânime."
            new_row["tribunal"] = "TRF1"
            new_row["informativo"] = "507"
            new_row["_source_pdf"] = str(pdf)
            new_row["_row_id"] = trf1.generate_row_id(new_row)

            argv = [
                "TRF1_pdf_to_csv_viaAPI_hibrido.py",
                "--no-gui",
                "--input-files",
                str(pdf),
                "--output-dir",
                str(tmp_dir),
                "--disable-openai",
                "--disable-perplexity",
                "--quiet",
            ]
            with patch.object(trf1, "extract_rows_from_pdf", return_value=[new_row]), patch.object(sys, "argv", argv):
                trf1.main()

            with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["numero_processo"], "0000001-11.2020.4.01.0000")
            self.assertEqual(rows[0]["classe"], "MS")
            self.assertEqual(rows[0]["contexto"], "")
            self.assertNotEqual(rows[0]["numero_processo"], legacy_row["numero_processo"])

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertNotIn("legacy_marker", saved)
            self.assertEqual(saved["version"], trf1.CHECKPOINT_VERSION)
            self.assertEqual(len(saved.get("rows", [])), 1)
            self.assertEqual(saved["rows"][0]["numero_processo"], "0000001-11.2020.4.01.0000")

    def test_main_resumes_from_compatible_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf = tmp_dir / "dummy.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%dummy\n")
            pdf_resolved = pdf.resolve()

            out_csv = tmp_dir / trf1.OUTPUT_NAME
            checkpoint = tmp_dir / trf1.CHECKPOINT_NAME

            row = trf1.base_empty_row()
            row["classe"] = "MS"
            row["numero_processo"] = "0000001-11.2020.4.01.0000"
            row["relator(a)"] = "Des. Novo"
            row["orgao_julgador"] = "Primeira Turma"
            row["data_julgamento"] = "1/21/2020"
            row["data_publicacao"] = "2/14/2020"
            row["texto_do_boletim"] = "Texto novo. Unânime."
            row["tribunal"] = "TRF1"
            row["informativo"] = "507"
            row["contexto"] = "Contexto já salvo"
            row["_source_pdf"] = str(pdf_resolved)
            row["_row_id"] = trf1.generate_row_id(row)
            row["_openai_done"] = "1"
            row["_perplexity_done"] = "0"
            row["_perplexity_reason"] = ""

            manifest = trf1.compute_manifest([pdf_resolved])
            payload = trf1.checkpoint_payload(
                manifest=manifest,
                rows=[row],
                perplexity_state={"workers": 3, "delay": 0.7},
            )
            checkpoint.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            argv = [
                "TRF1_pdf_to_csv_viaAPI_hibrido.py",
                "--no-gui",
                "--input-files",
                str(pdf),
                "--output-dir",
                str(tmp_dir),
                "--disable-openai",
                "--disable-perplexity",
                "--quiet",
            ]
            with patch.object(trf1, "extract_rows_from_pdf") as mocked_extract, patch.object(sys, "argv", argv):
                trf1.main()
                mocked_extract.assert_not_called()

            with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["numero_processo"], row["numero_processo"])
            self.assertEqual(rows[0]["contexto"], "Contexto já salvo")

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(saved["version"], trf1.CHECKPOINT_VERSION)
            self.assertEqual(len(saved.get("rows", [])), 1)
            self.assertEqual(saved["rows"][0]["numero_processo"], row["numero_processo"])


if __name__ == "__main__":
    unittest.main()
