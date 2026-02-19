#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

import TRF1_pdf_to_csv_viaAPI as trf1


def _new_logger() -> tuple[trf1.logging.Logger, io.StringIO]:
    buf = io.StringIO()
    logger = trf1.logging.Logger(f"test_trf1_perplexity_{id(buf)}", level=trf1.logging.DEBUG)
    handler = trf1.logging.StreamHandler(buf)
    handler.setFormatter(trf1.logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger, buf


def _sample_row(idx: int) -> dict[str, str]:
    row = trf1.base_empty_row()
    row["classe"] = "Ap"
    row["numero_processo"] = f"{idx:07d}-11.2020.4.01.0000"
    row["relator(a)"] = "Des. Fulano"
    row["tema"] = f"Tema específico de teste número {idx} para variar fingerprint do pedido"
    row["texto_do_boletim"] = f"Texto literal do boletim {idx}. Unânime."
    row["_row_id"] = f"row-{idx}"
    row["_source_pdf"] = str(idx)
    row["_openai_done"] = "1"
    row["_perplexity_done"] = "0"
    row["_perplexity_reason"] = ""
    return row


def _sample_rows(n: int) -> list[dict[str, str]]:
    return [_sample_row(i + 1) for i in range(n)]


def _base_cfg(**overrides) -> trf1.PerplexityConfig:
    cfg = trf1.PerplexityConfig(
        enabled=True,
        api_key="pplx-test",
        model="sonar",
        batch_size=20,
        max_workers=3,
        delay=0.75,
        retries=1,
        timeout=10,
        max_workers_cap=8,
        fallback_enabled=False,
        fallback_max_fraction=0.60,
        min_score_mainstream=2,
        min_score_official=1,
        domain_policy="consagrados_oficiais",
        scaling_mode="fixed",
        target_rpm=0,
        resume_rate_state=False,
        max_tokens=128,
        text_max_chars=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class _TimeoutSession:
    def post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.Timeout("read timed out")


class _Response429:
    status_code = 429

    def raise_for_status(self) -> None:
        return None


class _RateLimitSession:
    def post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _Response429()


class TestPerplexityStabilization(unittest.TestCase):
    def test_perplexity_call_single_timeout_not_rate_limited(self) -> None:
        logger, _ = _new_logger()
        ok, url, rate_limited, err, reason = trf1.perplexity_call_single(
            session=_TimeoutSession(),  # type: ignore[arg-type]
            model="sonar",
            row=_sample_row(1),
            stage=1,
            min_score_mainstream=2,
            min_score_official=1,
            domain_policy="consagrados_oficiais",
            timeout=3,
            retries=1,
            max_tokens=64,
            text_max_chars=0,
            pacer=None,
            logger=logger,
        )
        self.assertFalse(ok)
        self.assertEqual(url, "")
        self.assertFalse(rate_limited)
        self.assertEqual(reason, "timeout")
        self.assertIn("timed out", err.lower())

    def test_perplexity_call_single_429_sets_rate_limit(self) -> None:
        logger, _ = _new_logger()
        ok, url, rate_limited, err, reason = trf1.perplexity_call_single(
            session=_RateLimitSession(),  # type: ignore[arg-type]
            model="sonar",
            row=_sample_row(2),
            stage=1,
            min_score_mainstream=2,
            min_score_official=1,
            domain_policy="consagrados_oficiais",
            timeout=3,
            retries=1,
            max_tokens=64,
            text_max_chars=0,
            pacer=None,
            logger=logger,
        )
        self.assertFalse(ok)
        self.assertEqual(url, "")
        self.assertTrue(rate_limited)
        self.assertEqual(reason, "rate_limit")
        self.assertIn("429", err)

    def test_fixed_mode_keeps_workers_constant_under_429(self) -> None:
        rows = _sample_rows(40)
        cfg = _base_cfg(scaling_mode="fixed", max_workers=3, delay=0.0)
        logger, log_buf = _new_logger()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.csv"
            cp = Path(td) / "cp.json"
            with patch.object(
                trf1,
                "perplexity_call_single",
                return_value=(False, "", True, "429 rate limit", "rate_limit"),
            ):
                state = trf1.run_perplexity_enrichment(
                    rows=rows,
                    config=cfg,
                    output_csv=out,
                    checkpoint_path=cp,
                    manifest=[],
                    perplexity_state={"workers": 8, "delay": 5.0},
                    logger=logger,
                )
        self.assertEqual(state["workers"], 3)
        self.assertEqual(state["delay"], 0.0)
        self.assertNotIn("autoajuste: reduzindo workers", log_buf.getvalue())

    def test_fixed_mode_applies_extra_cooldown_on_429(self) -> None:
        rows = _sample_rows(21)
        cfg = _base_cfg(scaling_mode="fixed", max_workers=3, delay=0.0)
        logger, _ = _new_logger()
        sleeps: list[float] = []
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.csv"
            cp = Path(td) / "cp.json"
            with (
                patch.object(
                    trf1,
                    "perplexity_call_single",
                    return_value=(False, "", True, "429 rate limit", "rate_limit"),
                ),
                patch.object(trf1.time, "sleep", side_effect=lambda s: sleeps.append(float(s))),
            ):
                trf1.run_perplexity_enrichment(
                    rows=rows,
                    config=cfg,
                    output_csv=out,
                    checkpoint_path=cp,
                    manifest=[],
                    perplexity_state={"workers": 3, "delay": 0.0},
                    logger=logger,
                )
        self.assertTrue(any(s >= 10.9 for s in sleeps))

    def test_adaptive_mode_does_not_reduce_workers_for_timeout_or_http_error(self) -> None:
        rows = _sample_rows(20)
        cfg = _base_cfg(scaling_mode="adaptive", max_workers=3, delay=0.5)
        logger, log_buf = _new_logger()

        def _fake_perplexity_call_single(*args, **kwargs):  # type: ignore[no-untyped-def]
            row = args[2]
            idx = int(row["_source_pdf"])
            if idx % 2 == 0:
                return False, "", False, "read timed out", "timeout"
            return False, "", False, "503 service unavailable", "http_error"

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.csv"
            cp = Path(td) / "cp.json"
            with patch.object(trf1, "perplexity_call_single", side_effect=_fake_perplexity_call_single):
                state = trf1.run_perplexity_enrichment(
                    rows=rows,
                    config=cfg,
                    output_csv=out,
                    checkpoint_path=cp,
                    manifest=[],
                    perplexity_state={"workers": 3, "delay": 0.5},
                    logger=logger,
                )

        self.assertEqual(state["workers"], 3)
        self.assertEqual(state["delay"], 0.5)
        self.assertNotIn("autoajuste: reduzindo workers", log_buf.getvalue())

    def test_main_ignores_checkpoint_rate_state_by_default(self) -> None:
        captured: dict[str, float | int] = {}
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf = tmp / "dummy.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%dummy\n")
            manifest = trf1.compute_manifest([pdf.resolve()])
            row = _sample_row(1)
            payload = trf1.checkpoint_payload(
                manifest=manifest,
                rows=[row],
                perplexity_state={"workers": 1, "delay": 5.0},
            )
            (tmp / trf1.CHECKPOINT_NAME).write_text(
                trf1.json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            def _fake_run_perplexity_enrichment(**kwargs):  # type: ignore[no-untyped-def]
                state = kwargs["perplexity_state"]
                captured["workers"] = int(state["workers"])
                captured["delay"] = float(state["delay"])
                return state

            argv = [
                "TRF1_pdf_to_csv_viaAPI.py",
                "--no-gui",
                "--input-files",
                str(pdf),
                "--output-dir",
                str(tmp),
                "--disable-openai",
                "--perplexity-api-key",
                "pplx-test",
                "--perplexity-max-workers",
                "3",
                "--perplexity-delay",
                "0.75",
                "--quiet",
            ]
            with (
                patch.object(trf1, "run_openai_enrichment", return_value=None),
                patch.object(trf1, "run_perplexity_enrichment", side_effect=_fake_run_perplexity_enrichment),
                patch.object(trf1.sys, "argv", argv),
            ):
                trf1.main()

        self.assertEqual(captured["workers"], 3)
        self.assertEqual(captured["delay"], 0.75)

    def test_main_applies_checkpoint_rate_state_when_flag_enabled(self) -> None:
        captured: dict[str, float | int] = {}
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pdf = tmp / "dummy.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%dummy\n")
            manifest = trf1.compute_manifest([pdf.resolve()])
            row = _sample_row(1)
            payload = trf1.checkpoint_payload(
                manifest=manifest,
                rows=[row],
                perplexity_state={"workers": 1, "delay": 5.0},
            )
            (tmp / trf1.CHECKPOINT_NAME).write_text(
                trf1.json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            def _fake_run_perplexity_enrichment(**kwargs):  # type: ignore[no-untyped-def]
                state = kwargs["perplexity_state"]
                captured["workers"] = int(state["workers"])
                captured["delay"] = float(state["delay"])
                return state

            argv = [
                "TRF1_pdf_to_csv_viaAPI.py",
                "--no-gui",
                "--input-files",
                str(pdf),
                "--output-dir",
                str(tmp),
                "--disable-openai",
                "--perplexity-api-key",
                "pplx-test",
                "--perplexity-max-workers",
                "3",
                "--perplexity-delay",
                "0.75",
                "--perplexity-resume-rate-state",
                "--quiet",
            ]
            with (
                patch.object(trf1, "run_openai_enrichment", return_value=None),
                patch.object(trf1, "run_perplexity_enrichment", side_effect=_fake_run_perplexity_enrichment),
                patch.object(trf1.sys, "argv", argv),
            ):
                trf1.main()

        self.assertEqual(captured["workers"], 1)
        self.assertEqual(captured["delay"], 5.0)

    def test_perplexity_batch_logs_include_error_breakdown(self) -> None:
        rows = _sample_rows(4)
        cfg = _base_cfg(scaling_mode="fixed", max_workers=3, delay=0.0)
        logger, log_buf = _new_logger()

        def _fake_perplexity_call_single(*args, **kwargs):  # type: ignore[no-untyped-def]
            row = args[2]
            idx = int(row["_source_pdf"])
            if idx == 1:
                return False, "", True, "429 rate limit", "rate_limit"
            if idx == 2:
                return False, "", False, "read timed out", "timeout"
            if idx == 3:
                return False, "", False, "503 bad gateway", "http_error"
            return False, "", False, "unexpected", "error"

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.csv"
            cp = Path(td) / "cp.json"
            with patch.object(trf1, "perplexity_call_single", side_effect=_fake_perplexity_call_single):
                trf1.run_perplexity_enrichment(
                    rows=rows,
                    config=cfg,
                    output_csv=out,
                    checkpoint_path=cp,
                    manifest=[],
                    perplexity_state={"workers": 3, "delay": 0.0},
                    logger=logger,
                )

        log_text = log_buf.getvalue()
        self.assertIn("errors_total=4", log_text)
        self.assertIn("rate_limit_429=1", log_text)
        self.assertIn("timeout=1", log_text)
        self.assertIn("http_error=1", log_text)
        self.assertIn("other_error=1", log_text)


if __name__ == "__main__":
    unittest.main()
