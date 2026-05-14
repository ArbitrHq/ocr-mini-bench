"""End-to-end orchestrator replay test.

Drives `run_ocr_leaderboard_benchmark` against the real dataset and prompt
files but with `run_ocr_model` monkeypatched to return a synthetic
response. Validates the contract glue: task fan-out, scoring application,
debug-run capture, aggregation, snapshot shape, and the on_task_complete
callback ordering used by the CLI for checkpointing.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ocr_mini_bench.benchmark import orchestrator
from ocr_mini_bench.benchmark.orchestrator import (
    BenchmarkRunExecutionControls,
    BenchmarkTaskCompletionEvent,
    run_ocr_leaderboard_benchmark,
)
from ocr_mini_bench.benchmark.types import BenchmarkRunOptions
from ocr_mini_bench.ocr.types import OCRModelRunResult


def _make_fake_result_factory():
    async def fake_run_ocr_model(request, *, client=None):  # type: ignore[no-untyped-def]
        # Echo all ground-truth keys back so scoring has something to mark
        # as "found" without binding to a specific document.
        keys: list[str] = []
        for line in request.user_prompt.splitlines():
            if line.startswith("- "):
                key = line[2:].split(" (")[0].strip()
                if key:
                    keys.append(key)
        payload = {key: "" for key in keys}
        return OCRModelRunResult(
            text=json.dumps(payload),
            input_tokens=10,
            output_tokens=20,
            latency_ms=42,
            cached_input_tokens=0,
            cache_hit=False,
            cache_write_tokens=0,
            total_cost_usd=0.0001,
        )

    return fake_run_ocr_model


@pytest.mark.replay
def test_orchestrator_runs_and_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "run_ocr_model", _make_fake_result_factory())

    events: list[BenchmarkTaskCompletionEvent] = []

    async def on_complete(event: BenchmarkTaskCompletionEvent) -> None:
        events.append(event)

    options = BenchmarkRunOptions(
        runs_per_model=1,
        max_parallel_requests=2,
        max_documents_per_domain=1,
        provider_parallel=False,
        models=["gemini-3.1-flash-lite-preview"],
        domains=["invoices"],
    )
    controls = BenchmarkRunExecutionControls(on_task_complete=on_complete)
    snapshot = asyncio.run(run_ocr_leaderboard_benchmark(options, controls))

    assert snapshot.run_count == 1
    assert len(events) == 1
    assert events[0].metrics.input_tokens == 10
    assert events[0].metrics.latency_ms == 42
    assert events[0].debug.raw_output  # populated
    assert snapshot.options.runs_per_model == 1
    assert snapshot.options.provider_parallel is False
    assert len(snapshot.leaderboard) == 1
    assert snapshot.leaderboard[0].runs_completed == 1
    assert snapshot.debug is not None
    assert len(snapshot.debug.runs) == 1


@pytest.mark.replay
def test_orchestrator_skip_task_keys_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "run_ocr_model", _make_fake_result_factory())

    options = BenchmarkRunOptions(
        runs_per_model=2,
        max_parallel_requests=1,
        max_documents_per_domain=1,
        models=["gemini-3.1-flash-lite-preview"],
        domains=["invoices"],
    )
    # First, get the set of task keys.
    dry = asyncio.run(run_ocr_leaderboard_benchmark(options))
    assert dry.run_count == 2
    all_task_keys = {r.task_key for r in dry.debug.runs} if dry.debug else set()
    assert len(all_task_keys) == 2

    first_key = next(iter(all_task_keys))
    controls = BenchmarkRunExecutionControls(skip_task_keys={first_key})
    skipped = asyncio.run(run_ocr_leaderboard_benchmark(options, controls))
    # The skipped task is excluded from this invocation's tasks; only one runs.
    new_task_keys = {r.task_key for r in skipped.debug.runs} if skipped.debug else set()
    assert first_key not in new_task_keys
    assert skipped.run_count == 1


@pytest.mark.replay
def test_orchestrator_provider_parallel_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-check that provider_parallel=True doesn't blow up and still
    aggregates results — this exercises the run_by_provider_lanes branch."""
    monkeypatch.setattr(orchestrator, "run_ocr_model", _make_fake_result_factory())

    options = BenchmarkRunOptions(
        runs_per_model=1,
        max_documents_per_domain=1,
        provider_parallel=True,
        models=["gemini-3.1-flash-lite-preview"],
        domains=["invoices"],
    )
    snapshot = asyncio.run(run_ocr_leaderboard_benchmark(options))
    assert snapshot.options.provider_parallel is True
    assert snapshot.run_count == 1
