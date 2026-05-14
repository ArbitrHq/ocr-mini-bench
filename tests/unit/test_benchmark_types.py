"""Schema round-trip tests for benchmark.types pydantic models.

These guard the on-disk contract: every field we emit must round-trip
through `model_validate` → `model_dump(mode="json")` byte-equal."""

from __future__ import annotations

import pytest

from ocr_mini_bench.benchmark.types import (
    BenchmarkSnapshot,
    LeaderboardRow,
    SingleRunMetrics,
)

_SAMPLE_METRICS: dict = {
    "model_key": "google:gemini-3.1-flash-lite-preview",
    "provider": "google",
    "model_id": "gemini-3.1-flash-lite-preview",
    "model_label": "Gemini 3.1 Flash-Lite",
    "tier": "balanced",
    "domain": "invoices",
    "document_id": "invoices-auto-repair-shop-en",
    "run_number": 1,
    "success": True,
    "field_total": 41,
    "field_correct": 41,
    "critical_total": 14,
    "critical_correct": 14,
    "field_accuracy_pct": 100.0,
    "critical_accuracy_pct": 100.0,
    "found_key_count": 39,
    "requested_key_count": 41,
    "latency_ms": 13125,
    "input_tokens": 1340,
    "output_tokens": 1153,
    "total_cost_usd": 0.0020645,
    "cache_hit": False,
    "cached_input_tokens": 0,
    "cache_write_tokens": 0,
    "error": None,
}


_SAMPLE_SNAPSHOT: dict = {
    "schema_version": "1.0",
    "generated_at": "2026-05-14T07:05:59.242Z",
    "benchmark_id": "ocr-benchmark-2026-05-14",
    "benchmark_description": "Sample",
    "options": {
        "runs_per_model": 2,
        "max_parallel_requests": 1,
        "max_documents_per_domain": 3,
        "provider_parallel": True,
        "selected_domains": [],
        "models": ["gemini-3.1-flash-lite-preview"],
    },
    "dataset": {
        "total_documents": 9,
        "documents_per_domain": {"invoices": 3, "receipts": 3, "logistics": 3},
        "total_keys": 100,
        "labeled_keys": 100,
        "critical_keys": 40,
        "labeled_critical_keys": 40,
    },
    "leaderboard": [
        {
            "rank": 1,
            "model_key": "google:gemini-3.1-flash-lite-preview",
            "provider": "google",
            "model_id": "gemini-3.1-flash-lite-preview",
            "model_label": "Gemini 3.1 Flash-Lite",
            "tier": "balanced",
            "runs_requested": 18,
            "runs_completed": 18,
            "successful_runs": 14,
            "failed_runs": 0,
            "success_rate_pct": 77.78,
            "pass_at_2_pct": 77.78,
            "pass_at_3_pct": None,
            "pass_at_5_pct": None,
            "pass_at_10_pct": None,
            "avg_field_accuracy_pct": 96.78,
            "avg_critical_accuracy_pct": 96.82,
            "field_accuracy_variance_pct": 4.09,
            "avg_latency_ms": 9544.3,
            "p95_latency_ms": 13125.0,
            "total_cost_usd": 0.033925,
            "avg_cost_per_doc_usd": 0.001885,
            "avg_cost_per_run_usd": 0.001885,
            "cost_per_success_usd": 0.002423,
            "p95_cost_usd": 0.0027433,
            "p05_field_accuracy_pct": 88.0,
        }
    ],
    "by_domain": [],
    "run_count": 18,
    "markdown_table": "| Rank | Model | …",
    "warnings": [],
    "cache_summary": [],
}


@pytest.mark.unit
class TestSingleRunMetricsRoundTrip:
    def test_round_trips_known_metrics_shape(self) -> None:
        metrics = SingleRunMetrics.model_validate(_SAMPLE_METRICS)
        dumped = metrics.model_dump(mode="json")
        for key, value in _SAMPLE_METRICS.items():
            assert key in dumped, f"field {key!r} lost on round-trip"
            assert dumped[key] == value, f"field {key!r} changed: {dumped[key]!r} != {value!r}"


@pytest.mark.unit
class TestBenchmarkSnapshotRoundTrip:
    def test_parses_canonical_snapshot(self) -> None:
        snapshot = BenchmarkSnapshot.model_validate(_SAMPLE_SNAPSHOT)
        assert snapshot.run_count == 18
        assert isinstance(snapshot.leaderboard, list)
        assert isinstance(snapshot.leaderboard[0], LeaderboardRow)
        assert snapshot.leaderboard[0].model_key == "google:gemini-3.1-flash-lite-preview"


@pytest.mark.unit
class TestExtraFieldsPreserved:
    """If a future field shows up in incoming JSON, `extra='allow'` round-trips it."""

    def test_unknown_field_round_trips(self) -> None:
        base = {**_SAMPLE_METRICS, "future_field": "preserve me"}
        metrics = SingleRunMetrics.model_validate(base)
        dumped = metrics.model_dump(mode="json")
        assert dumped.get("future_field") == "preserve me"
