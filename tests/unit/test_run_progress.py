"""Tests for run.progress — log lines are part of the operator-visible
contract; the reporter must emit at periodic ticks, document completion,
and on every failure."""

from __future__ import annotations

import time

import pytest

from ocr_mini_bench.benchmark.run.progress import (
    ProgressEvent,
    create_progress_reporter,
)


@pytest.mark.unit
def test_emits_on_first_and_last_tick(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = create_progress_reporter(
        total_tasks=2,
        expected_runs_by_document={"doc-a": 2},
        started_at_ms=int(time.time() * 1000),
    )
    reporter(ProgressEvent(document_id="doc-a", model_label="m", run_number=1, ok=True))
    reporter(ProgressEvent(document_id="doc-a", model_label="m", run_number=2, ok=True))
    out = capsys.readouterr().out.strip().splitlines()
    # Two ticks: first completion (always logs) + final completion.
    assert len(out) == 2
    assert "1/2" in out[0]
    assert "2/2" in out[1]
    assert "document complete: doc-a (2/2)" in out[1]


@pytest.mark.unit
def test_emits_on_failure(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = create_progress_reporter(
        total_tasks=100,
        expected_runs_by_document={"doc": 100},
        started_at_ms=int(time.time() * 1000),
    )
    # First call always emits (it's the periodic tick at completed==1).
    reporter(ProgressEvent(document_id="doc", model_label="m", run_number=1, ok=True))
    capsys.readouterr()  # drain
    # Second is not on log_every boundary and not document-done — but failure
    # must still emit.
    reporter(
        ProgressEvent(
            document_id="doc", model_label="m", run_number=2, ok=False, error="boom"
        )
    )
    out = capsys.readouterr().out
    assert "fail:1" in out
    assert "last error: boom" in out


@pytest.mark.unit
def test_unknown_error_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = create_progress_reporter(
        total_tasks=1,
        expected_runs_by_document={"doc": 1},
        started_at_ms=int(time.time() * 1000),
    )
    reporter(
        ProgressEvent(document_id="doc", model_label="m", run_number=1, ok=False)
    )
    out = capsys.readouterr().out
    assert "last error: unknown error" in out


@pytest.mark.unit
def test_zero_total_tasks_uses_100_percent_label(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reporter = create_progress_reporter(
        total_tasks=0,
        expected_runs_by_document={},
        started_at_ms=int(time.time() * 1000),
    )
    # Failure path always emits.
    reporter(
        ProgressEvent(
            document_id="x", model_label="m", run_number=1, ok=False, error="e"
        )
    )
    out = capsys.readouterr().out
    assert "(100.0%)" in out
