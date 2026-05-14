"""Tests for the `ocr-bench` CLI internals: fingerprint, option
normalization, checkpoint state writer, and replay-mode resume bookkeeping.

These cover the pure-logic parts of the CLI without needing real provider
calls; the orchestrator/runtime path is exercised by replay and smoke tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ocr_mini_bench.benchmark.types import BenchmarkRunOptions
from ocr_mini_bench.cli.run_benchmark import (
    _compute_benchmark_fingerprint,
    _load_checkpoint_state,
    _load_latest_checkpoint_records,
    _normalize_options_for_compare,
    _round6,
    _split_csv,
    _stable_dumps,
    _summarize_checkpoint,
    _write_checkpoint_state,
)
from ocr_mini_bench.postprocess.types import (
    LegacyCheckpointDebug,
    LegacyCheckpointMetrics,
    LegacyCheckpointRecord,
)


@pytest.mark.unit
def test_split_csv_handles_blanks_and_whitespace() -> None:
    assert _split_csv(None) is None
    assert _split_csv("") is None
    assert _split_csv("  ,  ") is None
    assert _split_csv("a, b , c") == ["a", "b", "c"]


@pytest.mark.unit
def test_stable_dumps_matches_json_stringify_format() -> None:
    # JS JSON.stringify default: no spaces between items/keys, no trailing newline.
    assert _stable_dumps({"b": 2, "a": 1}) == '{"b":2,"a":1}'
    assert _stable_dumps([1, 2, 3]) == "[1,2,3]"


@pytest.mark.unit
def test_normalize_options_sorts_filter_lists() -> None:
    opts = BenchmarkRunOptions(
        runs_per_model=3,
        domains=["receipts", "invoices"],
        models=["m2", "m1"],
    )
    normalized = _normalize_options_for_compare(opts)
    assert normalized["domains"] == ["invoices", "receipts"]
    assert normalized["models"] == ["m1", "m2"]
    # Round-trips the same regardless of original ordering.
    opts2 = BenchmarkRunOptions(
        runs_per_model=3,
        domains=["invoices", "receipts"],
        models=["m1", "m2"],
    )
    assert _stable_dumps(_normalize_options_for_compare(opts)) == _stable_dumps(
        _normalize_options_for_compare(opts2)
    )


@pytest.mark.unit
def test_normalize_options_drops_none_to_match_json_stringify() -> None:
    # JS JSON.stringify drops `undefined`; we must drop `None` so the hash
    # input (and thus the fingerprint) matches TS when --domains is unset.
    opts = BenchmarkRunOptions(
        runs_per_model=2,
        max_parallel_requests=1,
        max_documents_per_domain=3,
        provider_parallel=True,
        models=["claude-haiku-4-5", "gemini-3.1-flash-lite-preview"],
    )
    normalized = _normalize_options_for_compare(opts)
    assert "domains" not in normalized
    # Order must match the TS `normalizeOptionsForCompare` object literal.
    assert list(normalized.keys()) == [
        "runs_per_model",
        "max_parallel_requests",
        "max_documents_per_domain",
        "provider_parallel",
        "models",
    ]


@pytest.mark.unit
def test_fingerprint_changes_when_prompt_changes(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "models.public.json"
    manifest_path = tmp_path / "dataset" / "manifest.json"
    system_path = tmp_path / "prompts" / "system.txt"
    user_path = tmp_path / "prompts" / "user.txt"
    for p in (config_path, manifest_path, system_path, user_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")
    system_path.write_text("hello", encoding="utf-8")
    user_path.write_text("world", encoding="utf-8")

    class _Paths:
        class config:
            models = config_path

        class dataset:
            manifest = manifest_path

        class prompts:
            system = system_path
            user = user_path

    options = BenchmarkRunOptions(runs_per_model=1)
    with patch("ocr_mini_bench.cli.run_benchmark.PATHS", _Paths):
        fp_a = _compute_benchmark_fingerprint(options)
        system_path.write_text("hello modified", encoding="utf-8")
        fp_b = _compute_benchmark_fingerprint(options)
    assert fp_a != fp_b


@pytest.mark.unit
def test_fingerprint_changes_when_options_change(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    system_path = tmp_path / "sys.txt"
    user_path = tmp_path / "user.txt"
    config_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")
    system_path.write_text("s", encoding="utf-8")
    user_path.write_text("u", encoding="utf-8")

    class _Paths:
        class config:
            models = config_path

        class dataset:
            manifest = manifest_path

        class prompts:
            system = system_path
            user = user_path

    with patch("ocr_mini_bench.cli.run_benchmark.PATHS", _Paths):
        fp1 = _compute_benchmark_fingerprint(BenchmarkRunOptions(runs_per_model=2))
        fp2 = _compute_benchmark_fingerprint(BenchmarkRunOptions(runs_per_model=3))
    assert fp1 != fp2


@pytest.mark.unit
def test_round6_truncates_to_six_decimals() -> None:
    assert _round6(0.123456789) == 0.123457
    assert _round6(0.0) == 0.0


def _make_record(task_key: str, *, error: str | None, cost: float) -> LegacyCheckpointRecord:
    common: dict[str, Any] = {
        "task_key": task_key,
        "model_key": "google:m",
        "provider": "google",
        "model_id": "m",
        "model_label": "M",
        "tier": "budget",
        "domain": "invoices",
        "document_id": "doc",
        "run_number": 1,
        "success": error is None,
        "field_total": 1,
        "field_correct": 1 if error is None else 0,
        "critical_total": 1,
        "critical_correct": 1 if error is None else 0,
        "field_accuracy_pct": 100.0 if error is None else 0.0,
        "critical_accuracy_pct": 100.0 if error is None else 0.0,
        "found_key_count": 1,
        "requested_key_count": 1,
        "latency_ms": 100,
        "total_cost_usd": cost,
        "cache_hit": False,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "error": error,
    }
    metrics = LegacyCheckpointMetrics.model_validate(
        {**common, "input_tokens": 1, "output_tokens": 1}
    )
    debug = LegacyCheckpointDebug.model_validate(
        {
            **common,
            "input_tokens": 1,
            "output_tokens": 1,
            "system_prompt_used": "",
            "user_prompt_used": "",
            "raw_output": "",
            "parsed_output": None,
            "extracted_pairs": [],
            "key_comparisons": [],
        }
    )
    return LegacyCheckpointRecord(
        task_key=task_key, completed_at="2026-05-14T07:05:59.242Z", metrics=metrics, debug=debug
    )


@pytest.mark.unit
def test_summarize_checkpoint_counts_failures_and_cost() -> None:
    records = {
        "a": _make_record("a", error=None, cost=0.1),
        "b": _make_record("b", error="boom", cost=0.0),
        "c": _make_record("c", error=None, cost=0.2),
    }
    summary = _summarize_checkpoint(records)
    assert int(summary["total"]) == 3
    assert int(summary["failed"]) == 1
    assert summary["cost_usd"] == pytest.approx(0.3, abs=1e-9)


@pytest.mark.unit
def test_write_and_load_checkpoint_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    options = BenchmarkRunOptions(runs_per_model=2)
    records = {"a": _make_record("a", error=None, cost=0.1)}
    _write_checkpoint_state(
        state_path=state_path,
        mode="fresh",
        options=options,
        records=records,
        current_run_new_records=1,
        current_run_cost_usd=0.1,
        current_run_target_tasks=10,
        benchmark_fingerprint="abc",
        final=False,
    )
    loaded = _load_checkpoint_state(state_path)
    assert loaded is not None
    assert loaded["mode"] == "fresh"
    assert loaded["records_total"] == 1
    assert loaded["records_failed"] == 0
    assert loaded["records_successful"] == 1
    assert loaded["current_run_remaining_tasks"] == 9
    assert loaded["benchmark_fingerprint"] == "abc"
    assert loaded["current_run_avg_cost_usd"] == pytest.approx(0.1)
    assert loaded["current_run_estimated_final_cost_usd"] == pytest.approx(1.0)
    # Format: 2-space indent + trailing newline (matches TS).
    raw = state_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert '  "mode"' in raw


@pytest.mark.unit
def test_load_latest_checkpoint_returns_last_row_per_task(tmp_path: Path) -> None:
    """If a task_key appears multiple times in runs.jsonl, the last wins."""
    runs_log = tmp_path / "runs.jsonl"
    early = _make_record("a", error="boom", cost=0.0)
    late = _make_record("a", error=None, cost=0.05)
    lines = [
        json.dumps(early.model_dump(mode="json")),
        json.dumps(late.model_dump(mode="json")),
    ]
    runs_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loaded = _load_latest_checkpoint_records(runs_log)
    assert "a" in loaded
    assert loaded["a"].metrics.error is None
    assert loaded["a"].metrics.total_cost_usd == 0.05


@pytest.mark.unit
def test_load_latest_checkpoint_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _load_latest_checkpoint_records(tmp_path / "nonexistent.jsonl") == {}


@pytest.mark.unit
def test_load_latest_checkpoint_skips_malformed_lines(tmp_path: Path) -> None:
    runs_log = tmp_path / "runs.jsonl"
    valid = _make_record("good", error=None, cost=0.0)
    lines = [
        "not json",
        "{}",  # missing required keys
        json.dumps({"task_key": "bad", "metrics": "wrong-shape"}),
        json.dumps(valid.model_dump(mode="json")),
    ]
    runs_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = _load_latest_checkpoint_records(runs_log)
    assert list(loaded.keys()) == ["good"]


@pytest.mark.unit
def test_load_checkpoint_state_empty_returns_none(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("", encoding="utf-8")
    assert _load_checkpoint_state(state_path) is None
    state_path.write_text("not json", encoding="utf-8")
    assert _load_checkpoint_state(state_path) is None
