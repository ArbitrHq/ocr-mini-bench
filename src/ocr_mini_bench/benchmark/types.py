"""Pydantic v2 models for the benchmark. Mirrors `src/benchmark/types.ts`.

These are the canonical on-disk schemas for `state.json`, `runs.jsonl`,
`raw.jsonl`, `metrics.snapshot.json`, `latest.json`, and `latest.debug.json`.
Field names are snake_case to match the existing TS artifacts byte-for-byte.

Serialization note: TS's `JSON.stringify` omits `undefined` properties. When
emitting these models to disk, use `model_dump(exclude_none=True, mode="json")`
on artifact rows where an optional was absent in the TS output. The schema
itself stays permissive (`extra="allow"`) so we round-trip unknown fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..config.model_catalog import ModelProvider

BenchmarkTier = Literal["budget", "balanced", "sota"]
KeyMatchMode = Literal["exact", "normalized_text", "contains", "numeric"]


class _Model(BaseModel):
    """Base class — permissive on unknown fields so future TS additions
    round-trip through the Python pipeline without data loss."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class BenchmarkModelConfig(_Model):
    provider: ModelProvider
    model_id: str
    model_label: str
    tier: BenchmarkTier


class BenchmarkConfig(_Model):
    description: str
    default_runs_per_model: int
    max_parallel_requests: int
    models: list[BenchmarkModelConfig]


class GroundTruthKey(_Model):
    name: str
    critical: bool
    expected: str | list[str] | None
    data_type: str | None = None
    match: KeyMatchMode | None = None
    notes: str | None = None


class GroundTruthDocument(_Model):
    schema_version: str
    document_id: str
    domain: str
    source_pdf: str
    notes: str | None = None
    keys: list[GroundTruthKey]


class ManifestDocument(_Model):
    document_id: str
    domain: str
    source_pdf: str
    ground_truth: str


class ManifestDomain(_Model):
    id: str
    source_directory: str
    document_count: int
    documents: list[ManifestDocument]


class BenchmarkManifest(_Model):
    schema_version: str
    generated_at: str
    domains: list[ManifestDomain]


class PreparedBenchmarkDocument(_Model):
    document_id: str
    domain: str
    source_pdf: str
    source_pdf_abs: str
    ground_truth_abs: str
    ground_truth_raw: Any
    ground_truth: GroundTruthDocument


class SingleRunMetrics(_Model):
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


class LeaderboardRow(_Model):
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
    avg_field_accuracy_pct: float
    avg_critical_accuracy_pct: float
    field_accuracy_variance_pct: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    avg_cost_per_doc_usd: float | None
    avg_cost_per_run_usd: float
    cost_per_success_usd: float | None
    p95_cost_usd: float
    p05_field_accuracy_pct: float


class DomainLeaderboard(_Model):
    domain: str
    rows: list[LeaderboardRow]


class DatasetSummary(_Model):
    total_documents: int
    documents_per_domain: dict[str, int]
    total_keys: int
    labeled_keys: int
    critical_keys: int
    labeled_critical_keys: int


class BenchmarkRunOptions(_Model):
    runs_per_model: int | None = None
    max_parallel_requests: int | None = None
    domains: list[str] | None = None
    max_documents_per_domain: int | None = None
    provider_parallel: bool | None = None
    models: list[str] | None = None


class ExtractedPair(_Model):
    key: str
    value: str
    found: bool


class KeyComparison(_Model):
    key: str
    critical: bool
    scored: bool
    expected_values: list[str]
    extracted_value: str
    matched: bool
    match_mode: KeyMatchMode


class BenchmarkDebugRun(_Model):
    task_key: str | None = None
    model_key: str
    provider: ModelProvider
    model_id: str
    model_label: str
    tier: BenchmarkTier
    domain: str
    document_id: str
    run_number: int
    latency_ms: int
    total_cost_usd: float
    success: bool
    error: str | None
    field_total: int
    field_correct: int
    critical_total: int
    critical_correct: int
    found_key_count: int
    requested_key_count: int
    system_prompt_used: str
    user_prompt_used: str
    raw_output: str
    parsed_output: Any
    extracted_pairs: list[ExtractedPair]
    key_comparisons: list[KeyComparison]


class BenchmarkDebugDocument(_Model):
    document_id: str
    domain: str
    source_pdf: str
    ground_truth_path: str
    ground_truth: Any


class BenchmarkDebugSnapshot(_Model):
    documents: list[BenchmarkDebugDocument]
    runs: list[BenchmarkDebugRun]


class BenchmarkSnapshotOptions(_Model):
    runs_per_model: int
    max_parallel_requests: int
    provider_parallel: bool
    selected_domains: list[str]
    max_documents_per_domain: int | None


class CacheSummaryEntry(_Model):
    model_key: str
    model_label: str
    provider: ModelProvider
    runs: int
    cache_hits: int
    cache_hit_rate_pct: float
    cached_input_tokens_total: int
    cached_input_tokens_avg: float
    cache_write_tokens_total: int
    cache_write_tokens_avg: float


class BenchmarkSnapshot(_Model):
    generated_at: str
    benchmark_id: str
    benchmark_description: str
    options: BenchmarkSnapshotOptions
    dataset: DatasetSummary
    leaderboard: list[LeaderboardRow]
    by_domain: list[DomainLeaderboard]
    run_count: int
    markdown_table: str
    warnings: list[str]
    cache_summary: list[CacheSummaryEntry] | None = None
    debug: BenchmarkDebugSnapshot | None = None
