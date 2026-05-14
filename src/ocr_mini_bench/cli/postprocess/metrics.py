"""`ocr-bench-metrics`: build leaderboard snapshots from comparison output.

Mirrors `src/cli/postprocess/metrics.ts`. Reads `comparison.jsonl`, aggregates
into `metrics.snapshot.json`, `leaderboard.aggregation.json`, and
`leaderboard.frontend.json`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from ...benchmark.dataset import load_prepared_documents, summarize_dataset
from ...config.paths import PATHS
from ...postprocess.aggregate import aggregate_metric_rows
from ...postprocess.io import (
    read_json_lines_file,
    timestamp_for_filename,
    write_json_file,
)
from ...postprocess.types import (
    AggregatedMetricRow,
    ComparisonRecord,
    DomainRows,
    LeaderboardAggregationSnapshot,
    LeaderboardAggregationSource,
    LeaderboardDataset,
    MetricRanges,
    MetricRangeStats,
    MetricsSnapshot,
    MetricsSnapshotSource,
)

app = typer.Typer(
    help="Build leaderboard snapshots from comparison output.",
    add_completion=False,
    no_args_is_help=False,
)


def _resolve_cwd(value: str) -> Path:
    return (Path.cwd() / value).resolve()


def _round(value: float, decimals: int) -> float:
    precision: int = 10**decimals
    scaled = value * precision
    rounded = int(math.floor(scaled + 0.5) if scaled >= 0 else -math.floor(-scaled + 0.5))
    return rounded / precision


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std_dev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    variance = max(0.0, sum((v - avg) ** 2 for v in values) / len(values))
    return math.sqrt(variance)


def _percentile_nearest_rank(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.ceil((p / 100) * len(sorted_values)) - 1))
    return sorted_values[idx]


def _combination(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    kk = min(k, n - k)
    result = 1.0
    for i in range(1, kk + 1):
        result = (result * (n - kk + i)) / i
    return result


@dataclass
class _DocRangeAccumulator:
    cost_total_usd: float = 0.0
    run_count: int = 0
    success_count: int = 0


@dataclass
class _BucketRangeAccumulator:
    success_pct_runs: list[float] = field(default_factory=list)
    critical_fields_pct: list[float] = field(default_factory=list)
    all_fields_pct: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    docs: dict[str, _DocRangeAccumulator] = field(default_factory=dict)


def _bucket_key(domain: str, model_key: str) -> str:
    return f"{domain}::{model_key}"


def _empty_range_stats() -> MetricRangeStats:
    return MetricRangeStats(
        count=0, min=None, p05=None, p50=None, p95=None, max=None, mean=None, stddev=None
    )


def _summarize_range(values: Sequence[float], decimals: int) -> MetricRangeStats:
    if not values:
        return _empty_range_stats()
    return MetricRangeStats(
        count=len(values),
        min=_round(min(values), decimals),
        p05=_round(_percentile_nearest_rank(values, 5), decimals),
        p50=_round(_percentile_nearest_rank(values, 50), decimals),
        p95=_round(_percentile_nearest_rank(values, 95), decimals),
        max=_round(max(values), decimals),
        mean=_round(_mean(values), decimals),
        stddev=_round(_std_dev(values), decimals),
    )


def _empty_metric_ranges() -> MetricRanges:
    return MetricRanges(
        cost_per_doc_usd=_empty_range_stats(),
        cost_per_success_usd=_empty_range_stats(),
        success_pct_runs=_empty_range_stats(),
        success_pct_docs=_empty_range_stats(),
        pass_at_3_strict_pct_docs=_empty_range_stats(),
        pass_at_5_strict_pct_docs=_empty_range_stats(),
        critical_fields_pct=_empty_range_stats(),
        all_fields_pct=_empty_range_stats(),
        latency_ms=_empty_range_stats(),
    )


def _add_to_bucket(
    bucket: _BucketRangeAccumulator,
    doc_key: str,
    cost_usd: float,
    latency_ms: float,
    critical_fields_pct: float,
    all_fields_pct: float,
    success: bool,
) -> None:
    bucket.success_pct_runs.append(100.0 if success else 0.0)
    bucket.critical_fields_pct.append(critical_fields_pct)
    bucket.all_fields_pct.append(all_fields_pct)
    bucket.latency_ms.append(latency_ms)
    doc = bucket.docs.setdefault(doc_key, _DocRangeAccumulator())
    doc.cost_total_usd += cost_usd
    doc.run_count += 1
    if success:
        doc.success_count += 1


def _finalize_bucket(bucket: _BucketRangeAccumulator) -> MetricRanges:
    cost_per_doc: list[float] = []
    cost_per_success: list[float] = []
    success_pct_docs: list[float] = []
    pass3_strict: list[float] = []
    pass5_strict: list[float] = []

    for doc in bucket.docs.values():
        if doc.run_count > 0:
            cost_per_doc.append(doc.cost_total_usd / doc.run_count)
            success_pct_docs.append((doc.success_count / doc.run_count) * 100)
        if doc.success_count > 0:
            cost_per_success.append(doc.cost_total_usd / doc.success_count)
        if doc.run_count >= 3:
            denom = _combination(doc.run_count, 3)
            num = _combination(doc.success_count, 3)
            pass3_strict.append((num / denom) * 100 if denom > 0 else 0.0)
        if doc.run_count >= 5:
            denom = _combination(doc.run_count, 5)
            num = _combination(doc.success_count, 5)
            pass5_strict.append((num / denom) * 100 if denom > 0 else 0.0)

    return MetricRanges(
        cost_per_doc_usd=_summarize_range(cost_per_doc, 6),
        cost_per_success_usd=_summarize_range(cost_per_success, 6),
        success_pct_runs=_summarize_range(bucket.success_pct_runs, 3),
        success_pct_docs=_summarize_range(success_pct_docs, 3),
        pass_at_3_strict_pct_docs=_summarize_range(pass3_strict, 3),
        pass_at_5_strict_pct_docs=_summarize_range(pass5_strict, 3),
        critical_fields_pct=_summarize_range(bucket.critical_fields_pct, 3),
        all_fields_pct=_summarize_range(bucket.all_fields_pct, 3),
        latency_ms=_summarize_range(bucket.latency_ms, 3),
    )


def _compute_ranges_by_bucket(records: Sequence[ComparisonRecord]) -> dict[str, MetricRanges]:
    accumulators: dict[str, _BucketRangeAccumulator] = {}

    for record in records:
        if record.runtime.error is not None:
            continue

        domain = record.document.domain.lower()
        model_key = record.model.model_key
        doc_key = f"{domain}::{record.document.document_id}"

        success: bool = (
            record.comparison.success if record.comparison is not None
            else record.legacy_metrics.success
        )
        critical_fields_pct = (
            record.comparison.critical_pass_pct if record.comparison is not None
            else record.legacy_metrics.critical_accuracy_pct
        )
        all_fields_pct = (
            record.comparison.field_pass_pct if record.comparison is not None
            else record.legacy_metrics.field_accuracy_pct
        )

        domain_bucket = accumulators.setdefault(
            _bucket_key(domain, model_key), _BucketRangeAccumulator()
        )
        _add_to_bucket(
            domain_bucket,
            doc_key,
            record.runtime.total_cost_usd,
            float(record.runtime.latency_ms),
            critical_fields_pct,
            all_fields_pct,
            success,
        )

        overall_bucket = accumulators.setdefault(
            _bucket_key("overall", model_key), _BucketRangeAccumulator()
        )
        _add_to_bucket(
            overall_bucket,
            doc_key,
            record.runtime.total_cost_usd,
            float(record.runtime.latency_ms),
            critical_fields_pct,
            all_fields_pct,
            success,
        )

    return {key: _finalize_bucket(bucket) for key, bucket in accumulators.items()}


def _attach_metric_ranges(
    rows: list[AggregatedMetricRow],
    domain: str,
    ranges_by_bucket: dict[str, MetricRanges],
) -> list[AggregatedMetricRow]:
    out: list[AggregatedMetricRow] = []
    for row in rows:
        copy = row.model_copy()
        copy.metric_ranges = ranges_by_bucket.get(
            _bucket_key(domain, row.model_key), _empty_metric_ranges()
        )
        out.append(copy)
    return out


def _build_markdown_table(rows: Sequence[AggregatedMetricRow]) -> str:
    head = [
        "| Rank | Model | Provider | Tier | Success % | pass^2 (>=2) % | pass^3 (>=3) % | "
        "pass^5 (>=5) % | pass^10 (>=10) % | pass^2 strict % | pass^3 strict % | "
        "pass^5 strict % | pass^10 strict % | Total Field Pass % | Critical Field % | "
        "Keys Found % | Field Variance (CV) % | Cost / Doc (USD) | Cost / Success (USD) | "
        "Avg Latency (ms) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body: list[str] = []
    for row in rows:
        def fmt(value: float | None) -> str:
            return "" if value is None else f"{value:.2f}"

        pass2 = fmt(row.pass_at_2_pct)
        pass3 = fmt(row.pass_at_3_pct)
        pass5 = fmt(row.pass_at_5_pct)
        pass10 = fmt(row.pass_at_10_pct)
        pass2s = fmt(row.pass_at_2_strict_pct)
        pass3s = fmt(row.pass_at_3_strict_pct)
        pass5s = fmt(row.pass_at_5_strict_pct)
        pass10s = fmt(row.pass_at_10_strict_pct)
        cost_per_doc = (
            "n/a" if row.avg_cost_per_doc_usd is None else f"{row.avg_cost_per_doc_usd:.4f}"
        )
        cost_per_success = (
            "n/a" if row.cost_per_success_usd is None else f"{row.cost_per_success_usd:.4f}"
        )
        body.append(
            f"| {row.rank} | {row.model_label} | {row.provider} | {row.tier} | "
            f"{row.success_rate_pct:.2f} | {pass2} | {pass3} | {pass5} | {pass10} | "
            f"{pass2s} | {pass3s} | {pass5s} | {pass10s} | "
            f"{row.avg_total_field_pass_pct:.2f} | {row.avg_critical_accuracy_pct:.2f} | "
            f"{row.avg_keys_found_pct:.2f} | {row.field_accuracy_variance_pct:.2f} | "
            f"{cost_per_doc} | {cost_per_success} | {row.avg_latency_ms:.1f} |"
        )
    return "\n".join([*head, *body])


def _infer_runs_per_model(records: Sequence[ComparisonRecord]) -> int:
    by_model_doc: dict[str, int] = {}
    for record in records:
        key = f"{record.model.model_key}::{record.document.document_id}"
        by_model_doc[key] = by_model_doc.get(key, 0) + 1
    counts = list(by_model_doc.values())
    return max(counts) if counts else 0


def _build_cache_summary(records: Sequence[ComparisonRecord]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.runtime.error is not None:
            continue
        current = by_model.setdefault(
            record.model.model_key,
            {
                "model_label": record.model.model_label,
                "provider": record.model.provider,
                "runs": 0,
                "cache_hits": 0,
                "cached_input_tokens_total": 0,
                "cache_write_tokens_total": 0,
            },
        )
        current["runs"] += 1
        if record.runtime.cache_hit:
            current["cache_hits"] += 1
        current["cached_input_tokens_total"] += record.runtime.cached_input_tokens
        current["cache_write_tokens_total"] += record.runtime.cache_write_tokens

    entries = []
    for model_key, value in by_model.items():
        runs = value["runs"]
        hit_rate = round(((value["cache_hits"] / runs) * 100), 2) if runs > 0 else 0.0
        entries.append(
            {
                "model_key": model_key,
                "model_label": value["model_label"],
                "provider": value["provider"],
                "runs": runs,
                "cache_hits": value["cache_hits"],
                "cache_hit_rate_pct": hit_rate,
                "cached_input_tokens_total": round(value["cached_input_tokens_total"]),
                "cached_input_tokens_avg": (
                    round(value["cached_input_tokens_total"] / runs, 1) if runs > 0 else 0.0
                ),
                "cache_write_tokens_total": round(value["cache_write_tokens_total"]),
                "cache_write_tokens_avg": (
                    round(value["cache_write_tokens_total"] / runs, 1) if runs > 0 else 0.0
                ),
            }
        )

    entries.sort(key=lambda e: (-e["cache_hit_rate_pct"], e["model_label"]))
    return entries


@app.callback(invoke_without_command=True)
def main(
    comparison_jsonl: Annotated[
        str | None,
        typer.Option("--comparison-jsonl", help="Comparison JSONL input."),
    ] = None,
    raw_jsonl: Annotated[
        str | None,
        typer.Option("--raw-jsonl", help="Raw JSONL (metadata only)."),
    ] = None,
    output_metrics_json: Annotated[
        str | None,
        typer.Option("--output-metrics-json", help="Metrics snapshot output."),
    ] = None,
    output_aggregation_json: Annotated[
        str | None,
        typer.Option("--output-aggregation-json", help="Leaderboard aggregation output."),
    ] = None,
    output_frontend_json: Annotated[
        str | None,
        typer.Option("--output-frontend-json", help="Frontend snapshot output."),
    ] = None,
) -> None:
    comp_path = (
        _resolve_cwd(comparison_jsonl) if comparison_jsonl
        else PATHS.postprocess.comparison_jsonl
    )
    raw_path: Path | None
    if raw_jsonl is None:
        raw_path = PATHS.postprocess.raw_jsonl
    else:
        raw_path = _resolve_cwd(raw_jsonl) if raw_jsonl else None
    out_metrics = (
        _resolve_cwd(output_metrics_json) if output_metrics_json
        else PATHS.postprocess.metrics_snapshot
    )
    out_aggregation = (
        _resolve_cwd(output_aggregation_json) if output_aggregation_json
        else PATHS.postprocess.leaderboard_aggregation
    )
    out_frontend = (
        _resolve_cwd(output_frontend_json) if output_frontend_json
        else PATHS.postprocess.leaderboard_frontend
    )

    rows = read_json_lines_file(comp_path)
    records = [ComparisonRecord.model_validate(row) for row in rows]
    if not records:
        raise typer.Exit(code=1)

    overall_rows = aggregate_metric_rows(records)
    ranges_by_bucket = _compute_ranges_by_bucket(records)

    selected_domains = sorted({r.document.domain.lower() for r in records})

    by_domain_rows = [
        (
            d,
            aggregate_metric_rows(
                [r for r in records if r.document.domain.lower() == d]
            ),
        )
        for d in selected_domains
    ]

    overall_with_ranges = _attach_metric_ranges(overall_rows, "overall", ranges_by_bucket)
    by_domain = [
        DomainRows(domain=d, rows=_attach_metric_ranges(rows, d, ranges_by_bucket))
        for d, rows in by_domain_rows
    ]

    prepared = load_prepared_documents(domains=selected_domains)
    selected_doc_ids = {
        f"{r.document.domain.lower()}::{r.document.document_id}" for r in records
    }
    selected_docs = [
        doc for doc in prepared
        if f"{doc.domain.lower()}::{doc.document_id}" in selected_doc_ids
    ]
    dataset_summary = summarize_dataset(selected_docs)
    dataset = LeaderboardDataset.model_validate(dataset_summary.model_dump())

    warnings: list[str] = []
    if dataset.labeled_keys < dataset.total_keys:
        warnings.append(
            f"Only {dataset.labeled_keys}/{dataset.total_keys} keys are currently labeled."
        )
    for domain, count in dataset.documents_per_domain.items():
        if count < 10:
            warnings.append(
                f'Domain "{domain}" has {count} documents; target is >= 10 for launch quality.'
            )

    if any(row.pass_at_10_pct is None for row in overall_with_ranges):
        warnings.append(
            "pass^10 is empty for one or more models because at least one document has fewer "
            "than 10 completed runs."
        )

    now = datetime.now(UTC)
    generated_at = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    metrics_snapshot = MetricsSnapshot(
        schema_version="1.0",
        generated_at=generated_at,
        source=MetricsSnapshotSource(comparison_jsonl=str(comp_path)),
        run_count=len(records),
        model_rows=overall_with_ranges,
        by_domain=by_domain,
    )

    aggregation = LeaderboardAggregationSnapshot(
        schema_version="1.0",
        generated_at=generated_at,
        source=LeaderboardAggregationSource(
            raw_jsonl=str(raw_path) if raw_path else "",
            comparison_jsonl=str(comp_path),
            metrics_json=str(out_metrics),
        ),
        dataset=dataset,
        leaderboard=overall_with_ranges,
        by_domain=by_domain,
        run_count=len(records),
        warnings=warnings,
    )

    frontend = {
        "generated_at": generated_at,
        "benchmark_id": f"ocr-benchmark-postprocess-{timestamp_for_filename(now)}",
        "benchmark_description": "Postprocessed OCR mini-bench (raw -> comparison -> metrics).",
        "options": {
            "runs_per_model": _infer_runs_per_model(records),
            "max_parallel_requests": 1,
            "provider_parallel": False,
            "selected_domains": selected_domains,
            "max_documents_per_domain": None,
        },
        "dataset": dataset.model_dump(mode="json"),
        "leaderboard": [r.model_dump(mode="json") for r in overall_with_ranges],
        "by_domain": [d.model_dump(mode="json") for d in by_domain],
        "run_count": len(records),
        "markdown_table": _build_markdown_table(overall_with_ranges),
        "warnings": warnings,
        "cache_summary": _build_cache_summary(records),
    }

    write_json_file(out_metrics, metrics_snapshot.model_dump(mode="json"))
    write_json_file(out_aggregation, aggregation.model_dump(mode="json"))
    write_json_file(out_frontend, frontend)

    typer.echo(f"Metrics rows: {len(overall_with_ranges)}")
    typer.echo(f"Run count: {len(records)}")
    typer.echo(f"Metrics JSON: {out_metrics}")
    typer.echo(f"Aggregation JSON: {out_aggregation}")
    typer.echo(f"Frontend JSON: {out_frontend}")
