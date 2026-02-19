#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import TRF1_pdf_to_csv_viaAPI as trf1


def _new_logger() -> tuple[logging.Logger, io.StringIO]:
    buf = io.StringIO()
    logger = logging.Logger(f"test_trf1_openai_{id(buf)}", level=logging.DEBUG)
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger, buf


def _sample_row(row_id: str = "row-1") -> dict[str, str]:
    row = trf1.base_empty_row()
    digits = "".join(ch for ch in row_id if ch.isdigit()) or "1"
    seq = digits[-7:].zfill(7)
    row["classe"] = "Apelação Cível"
    row["numero_processo"] = f"{seq}-11.2020.4.01.0000"
    row["relator(a)"] = "Des. Fulano"
    row["orgao_julgador"] = "Primeira Turma"
    row["data_julgamento"] = "1/22/2020"
    row["informativo"] = "999"
    row["texto_do_boletim"] = "Discussão sobre direito à saúde e fornecimento de medicamento pelo SUS."
    row["_row_id"] = row_id
    row["_openai_done"] = "0"
    row["_perplexity_done"] = "0"
    row["_perplexity_reason"] = ""
    return row


def _sample_openai_payload() -> dict[str, object]:
    return {
        "contexto": "Contexto resumido do julgamento.",
        "tese": "Tese principal do acórdão.",
        "tema": "Fornecimento de medicamento de alto custo para tratamento de saúde pública",
        "ramo_do_direito": "Direito Administrativo",
        "subramo_do_direito": ["Saúde Pública", "Responsabilidade do Estado"],
        "punchline": "Estado deve garantir tratamento essencial quando comprovada necessidade médica.",
        "bullet_points": [
            "Direito à saúde possui eficácia imediata",
            "Necessidade clínica comprovada por laudo",
            "Dever estatal prevalece sobre restrição orçamentária",
        ],
        "jurisprudência": ["REsp 123"],
        "legislacao": ["CF/88 art. 196"],
    }


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.refusal = None


class _FakeChoice:
    def __init__(self, finish_reason: str, content: str) -> None:
        self.finish_reason = finish_reason
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, finish_reason: str, content: str) -> None:
        self.choices = [_FakeChoice(finish_reason, content)]


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("No fake responses configured")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(responses)


class TestOpenAIFixes(unittest.TestCase):
    def test_classify_openai_error_maps_length_and_auth(self) -> None:
        self.assertEqual(trf1.classify_openai_error("empty_json_due_length"), "length")
        self.assertEqual(trf1.classify_openai_error("Authentication failed"), "auth")
        self.assertEqual(trf1.classify_openai_error("429 rate limit"), "rate_limit")
        self.assertEqual(trf1.classify_openai_error("read timed out"), "timeout")

    def test_openai_call_single_auto_downgrade_on_length(self) -> None:
        payload = _sample_openai_payload()
        fake = _FakeClient(
            [
                _FakeResponse("length", ""),
                _FakeResponse("stop", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        logger, _ = _new_logger()

        ok, out, err = trf1.openai_call_single(
            client=fake,  # type: ignore[arg-type]
            model="gpt-5-mini",
            row=_sample_row(),
            timeout=20,
            retries=2,
            max_completion_tokens=700,
            max_completion_tokens_cap=4200,
            reasoning_effort="medium",
            verbosity="medium",
            length_fallback_policy="auto_downgrade",
            text_max_chars=0,
            logger=logger,
        )

        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertIn("contexto", out)
        self.assertEqual(fake.chat.completions.calls[0]["reasoning_effort"], "medium")
        self.assertEqual(fake.chat.completions.calls[1]["reasoning_effort"], "minimal")
        self.assertEqual(fake.chat.completions.calls[0]["max_completion_tokens"], 700)
        self.assertGreater(int(fake.chat.completions.calls[1]["max_completion_tokens"]), 700)

    def test_run_openai_length_keeps_workers_and_enables_global_downgrade(self) -> None:
        rows = [_sample_row(f"r{i}") for i in range(4)]
        payload = trf1.normalize_openai_payload(_sample_openai_payload())
        logger, log_buf = _new_logger()
        call_efforts: list[str] = []
        call_idx = {"n": 0}

        def _fake_openai_call_single(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_efforts.append(str(args[7]))
            n = call_idx["n"]
            call_idx["n"] += 1
            if n < 2:
                return False, {}, "empty_json_due_length"
            return True, payload, ""

        cfg = trf1.OpenAIConfig(
            enabled=True,
            api_key="sk-test",
            model="gpt-5-mini",
            batch_size=2,
            max_workers=3,
            delay=0.0,
            retries=1,
            timeout=10,
            max_workers_cap=5,
            max_completion_tokens=700,
            max_completion_tokens_cap=4200,
            reasoning_effort="medium",
            verbosity="medium",
            length_fallback_policy="auto_downgrade",
            length_error_threshold=0.50,
            text_max_chars=0,
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.csv"
            chk = Path(td) / "cp.json"
            with (
                patch.object(trf1, "OpenAI", return_value=object()) as openai_ctor,
                patch.object(trf1, "openai_call_single", side_effect=_fake_openai_call_single),
            ):
                trf1.run_openai_enrichment(
                    rows=rows,
                    config=cfg,
                    output_csv=out,
                    checkpoint_path=chk,
                    manifest=[],
                    perplexity_state={"workers": 1, "delay": 0.0},
                    logger=logger,
                )
                self.assertTrue(openai_ctor.called)
                self.assertEqual(openai_ctor.call_args.kwargs.get("max_retries"), 0)

        self.assertEqual(call_efforts[:2], ["medium", "medium"])
        self.assertEqual(call_efforts[2:], ["minimal", "minimal"])
        log_text = log_buf.getvalue()
        self.assertIn("OpenAI fallback: reasoning_effort -> minimal por excesso de length", log_text)
        self.assertNotIn("OpenAI autoajuste: reduzindo workers=2", log_text)

    def test_run_openai_rate_limit_reduces_workers(self) -> None:
        rows = [_sample_row(f"rr{i}") for i in range(2)]
        logger, log_buf = _new_logger()

        def _fake_openai_call_single(*args, **kwargs):  # type: ignore[no-untyped-def]
            return False, {}, "429 rate limit"

        cfg = trf1.OpenAIConfig(
            enabled=True,
            api_key="sk-test",
            model="gpt-5-mini",
            batch_size=2,
            max_workers=3,
            delay=0.0,
            retries=1,
            timeout=10,
            max_workers_cap=5,
            max_completion_tokens=700,
            max_completion_tokens_cap=4200,
            reasoning_effort="medium",
            verbosity="medium",
            length_fallback_policy="keep_deep",
            length_error_threshold=0.20,
            text_max_chars=0,
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.csv"
            chk = Path(td) / "cp.json"
            with (
                patch.object(trf1, "OpenAI", return_value=object()),
                patch.object(trf1, "openai_call_single", side_effect=_fake_openai_call_single),
            ):
                trf1.run_openai_enrichment(
                    rows=rows,
                    config=cfg,
                    output_csv=out,
                    checkpoint_path=chk,
                    manifest=[],
                    perplexity_state={"workers": 1, "delay": 0.0},
                    logger=logger,
                )

        self.assertIn("OpenAI autoajuste: reduzindo workers=2", log_buf.getvalue())

    def test_run_openai_fail_fast_raises_on_length(self) -> None:
        rows = [_sample_row(f"ff{i}") for i in range(2)]
        logger, _ = _new_logger()

        def _fake_openai_call_single(*args, **kwargs):  # type: ignore[no-untyped-def]
            return False, {}, "empty_json_due_length"

        cfg = trf1.OpenAIConfig(
            enabled=True,
            api_key="sk-test",
            model="gpt-5-mini",
            batch_size=2,
            max_workers=3,
            delay=0.0,
            retries=1,
            timeout=10,
            max_workers_cap=5,
            max_completion_tokens=700,
            max_completion_tokens_cap=4200,
            reasoning_effort="medium",
            verbosity="medium",
            length_fallback_policy="fail_fast",
            length_error_threshold=0.20,
            text_max_chars=0,
        )

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.csv"
            chk = Path(td) / "cp.json"
            with (
                patch.object(trf1, "OpenAI", return_value=object()),
                patch.object(trf1, "openai_call_single", side_effect=_fake_openai_call_single),
            ):
                with self.assertRaisesRegex(RuntimeError, "OpenAI fail_fast"):
                    trf1.run_openai_enrichment(
                        rows=rows,
                        config=cfg,
                        output_csv=out,
                        checkpoint_path=chk,
                        manifest=[],
                        perplexity_state={"workers": 1, "delay": 0.0},
                        logger=logger,
                    )


if __name__ == "__main__":
    unittest.main()
