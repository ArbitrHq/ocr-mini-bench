"""Pydantic models for postprocess artifacts. Mirrors
`src/postprocess/types.ts`.

These describe the on-disk contract for the postprocess outputs
(`raw.jsonl`, `comparison.jsonl`, `metrics.snapshot.json`,
`leaderboard.aggregation.json`) and the upstream checkpoint shape
(`runs.jsonl` rows). `extra='allow'` for forward compatibility.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..benchmark.types import (
    BenchmarkTier,
    ExtractedPair,
    KeyComparison,
    KeyMatchMode,
)
from ..config.model_catalog import ModelProvider


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---- Raw / Comparison records --------------------------------------------


class RawModelInfo(_Model):
    model_key: str
    provider: ModelProvider
    model_id: str
    model_label: str
    tier: BenchmarkTier


class RawDocumentInfo(_Model):
    domain: str
    document_id: str
    run_number: int


class RawRuntimeInfo(_Model):
    latency_ms: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    cache_hit: bool
    cached_input_tokens: int
    cache_write_tokens: int
    error: str | None


class RawPayload(_Model):
    system_prompt_used: str
    user_prompt_used: str
    raw_output: str
    parsed_output: Any
    extracted_pairs: list[ExtractedPair]


class RawLegacyMetrics(_Model):
    success: bool
    field_total: int
    field_correct: int
    critical_total: int
    critical_correct: int
    field_accuracy_pct: float
    critical_accuracy_pct: float
    found_key_count: int
    requested_key_count: int


class RawNormalizedRecord(_Model):
    schema_version: Literal["1.0"]
    task_key: str
    completed_at: str | None
    model: RawModelInfo
    document: RawDocumentInfo
    runtime: RawRuntimeInfo
    payload: RawPayload
    legacy_metrics: RawLegacyMetrics


class ComparisonBlock(_Model):
    field_total: int
    field_correct: int
    field_pass_pct: float
    critical_total: int
    critical_correct: int
    critical_pass_pct: float
    found_key_count: int
    requested_key_count: int
    keys_found_pct: float
    success: bool
    key_comparisons: list[KeyComparison]


class ComparisonRecord(_Model):
    schema_version: Literal["1.0"]
    task_key: str
    completed_at: str | None
    model: RawModelInfo
    document: RawDocumentInfo
    runtime: RawRuntimeInfo
    legacy_metrics: RawLegacyMetrics
    comparison: ComparisonBlock | None


# ---- Aggregated metric rows ----------------------------------------------


class MetricRangeStats(_Model):
    count: int
    min: float | None
    p05: float | None
    p50: float | None
    p95: float | None
    max: float | None
    mean: float | None
    stddev: float | None


class MetricRanges(_Model):
    cost_per_doc_usd: MetricRangeStats
    cost_per_success_usd: MetricRangeStats
    success_pct_runs: MetricRangeStats
    success_pct_docs: MetricRangeStats
    pass_at_3_strict_pct_docs: MetricRangeStats
    pass_at_5_strict_pct_docs: MetricRangeStats
    critical_fields_pct: MetricRangeStats
    all_fields_pct: MetricRangeStats
    latency_ms: MetricRangeStats


class AggregatedMetricRow(_Model):
    rank: int
    model_key: str
    provider: ModelProvider
    model_id: str
    model_label: str
    tier: BenchmarkTier
    runs_requested: int
    runs_completed: int
    successful_runs: int
    failed_runs: int
    success_rate_pct: float
    pass_at_2_pct: float | None
    pass_at_3_pct: float | None
    pass_at_5_pct: float | None
    pass_at_10_pct: float | None
    pass_at_2_strict_pct: float | None
    pass_at_3_strict_pct: float | None
    pass_at_5_strict_pct: float | None
    pass_at_10_strict_pct: float | None
    avg_total_field_pass_pct: float
    avg_field_accuracy_pct: float
    avg_critical_accuracy_pct: float
    avg_keys_found_pct: float
    field_pass_stddev_pct: float
    field_accuracy_variance_pct: float
    avg_latency_ms: float
    p95_latency_ms: float
    latency_stddev_ms: float
    latency_cv_pct: float
    total_cost_usd: float
    avg_cost_per_doc_usd: float | None
    avg_cost_per_run_usd: float
    cost_per_success_usd: float | None
    p95_cost_usd: float
    metric_ranges: MetricRanges | None = None


class DomainRows(_Model):
    domain: str
    rows: list[AggregatedMetricRow]


class MetricsSnapshotSource(_Model):
    comparison_jsonl: str


class MetricsSnapshot(_Model):
    schema_version: Literal["1.0"]
    generated_at: str
    source: MetricsSnapshotSource
    run_count: int
    model_rows: list[AggregatedMetricRow]
    by_domain: list[DomainRows]


class LeaderboardAggregationSource(_Model):
    raw_jsonl: str
    comparison_jsonl: str
    metrics_json: str


class LeaderboardDataset(_Model):
    total_documents: int
    documents_per_domain: dict[str, int]
    total_keys: int
    labeled_keys: int
    critical_keys: int
    labeled_critical_keys: int


class LeaderboardAggregationSnapshot(_Model):
    schema_version: Literal["1.0"]
    generated_at: str
    source: LeaderboardAggregationSource
    dataset: LeaderboardDataset
    leaderboard: list[AggregatedMetricRow]
    by_domain: list[DomainRows]
    run_count: int
    warnings: list[str]


# ---- Legacy checkpoint records -------------------------------------------


class LegacyCheckpointDebug(_Model):
    task_key: str | None = None
    model_key: str
    provider: ModelProvider
    model_id: str
    model_label: str
    tier: BenchmarkTier
    domain: str
    document_id: str
    run_number: int
    success: bool
    field_total: int
    field_correct: int
    critical_total: int
    critical_correct: int
    field_accuracy_pct: float
    critical_accuracy_pct: float
    found_key_count: int
    requested_key_count: int
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost_usd: float
    cache_hit: bool
    cached_input_tokens: int
    cache_write_tokens: int
    error: str | None
    system_prompt_used: str | None = None
    user_prompt_used: str | None = None
    raw_output: str
    parsed_output: Any
    extracted_pairs: list[ExtractedPair] | None = None
    key_comparisons: list[KeyComparison] | None = None


class LegacyCheckpointMetrics(_Model):
    task_key: str | None = None
    model_key: str
    provider: ModelProvider
    model_id: str
    model_label: str
    tier: BenchmarkTier
    domain: str
    document_id: str
    run_number: int
    success: bool
    field_total: int
    field_correct: int
    critical_total: int
    critical_correct: int
    field_accuracy_pct: float
    critical_accuracy_pct: float
    found_key_count: int
    requested_key_count: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    cache_hit: bool
    cached_input_tokens: int
    cache_write_tokens: int
    error: str | None


class LegacyCheckpointRecord(_Model):
    task_key: str
    completed_at: str | None = None
    metrics: LegacyCheckpointMetrics
    debug: LegacyCheckpointDebug


# Type alias used by raw-contract helpers — `match_mode` etc. kept here for
# convenience even though they're re-exports from benchmark.types.
__all__ = [
    "AggregatedMetricRow",
    "ComparisonBlock",
    "ComparisonRecord",
    "DomainRows",
    "KeyMatchMode",
    "LeaderboardAggregationSnapshot",
    "LeaderboardAggregationSource",
    "LeaderboardDataset",
    "LegacyCheckpointDebug",
    "LegacyCheckpointMetrics",
    "LegacyCheckpointRecord",
    "MetricRangeStats",
    "MetricRanges",
    "MetricsSnapshot",
    "MetricsSnapshotSource",
    "RawDocumentInfo",
    "RawLegacyMetrics",
    "RawModelInfo",
    "RawNormalizedRecord",
    "RawPayload",
    "RawRuntimeInfo",
]
