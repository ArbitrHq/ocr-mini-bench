"""Aggregate per-run comparison records into model-level rows.

Mirrors `src/postprocess/aggregate.ts`. Note the `pass@N` math: a "threshold"
variant counts docs with >= N successes; a "strict" variant uses the
combinatoric estimator from the HumanEval pass@k paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import inf

from ..benchmark.run.math import (
    mean,
    pct,
    percentile,
    std_dev,
)
from ..benchmark.run.math import (
    round_half_away_from_zero as _round,
)
from .types import AggregatedMetricRow, ComparisonRecord


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


def _document_run_stats(runs: Sequence[ComparisonRecord]) -> list[tuple[int, int]]:
    """Returns a list of (trials, successes) per unique document_id."""
    by_document: dict[str, list[int]] = {}
    for run in runs:
        bucket = by_document.setdefault(run.document.document_id, [0, 0])
        bucket[0] += 1
        succeeded = run.runtime.error is None and run.comparison is not None and run.comparison.success
        if succeeded:
            bucket[1] += 1
    return [(b[0], b[1]) for b in by_document.values()]


def _pass_threshold_at_n(runs: Sequence[ComparisonRecord], n: int) -> float | None:
    stats = _document_run_stats(runs)
    if not stats:
        return None
    if any(trials < n for trials, _ in stats):
        return None
    passing = sum(1 for _, successes in stats if successes >= n)
    return _round(pct(passing, len(stats)), 2)


def _pass_strict_at_n(runs: Sequence[ComparisonRecord], n: int) -> float | None:
    stats = _document_run_stats(runs)
    if not stats:
        return None
    if any(trials < n for trials, _ in stats):
        return None
    total = 0.0
    for trials, successes in stats:
        denominator = _combination(trials, n)
        numerator = _combination(successes, n)
        total += (numerator / denominator) if denominator > 0 else 0.0
    return _round((total / len(stats)) * 100, 2)


def aggregate_metric_rows(records: Sequence[ComparisonRecord]) -> list[AggregatedMetricRow]:
    by_model: dict[str, list[ComparisonRecord]] = {}
    for record in records:
        by_model.setdefault(record.model.model_key, []).append(record)

    rows: list[AggregatedMetricRow] = []
    for model_key, runs in by_model.items():
        sample = runs[0]
        runs_completed = len(runs)
        failed_runs = sum(1 for r in runs if r.runtime.error is not None)
        successful_runs = sum(
            1
            for r in runs
            if r.runtime.error is None and r.comparison is not None and r.comparison.success
        )

        scored_runs = [r for r in runs if r.runtime.error is None and r.comparison is not None]

        field_passes = [r.comparison.field_pass_pct for r in scored_runs if r.comparison]
        critical_passes = [r.comparison.critical_pass_pct for r in scored_runs if r.comparison]
        keys_found_pct = [r.comparison.keys_found_pct for r in scored_runs if r.comparison]

        latencies = [r.runtime.latency_ms for r in scored_runs]
        costs = [r.runtime.total_cost_usd for r in scored_runs]

        total_cost = sum(costs)
        success_rate = pct(successful_runs, runs_completed)

        avg_field_pass = mean(field_passes)
        field_std = std_dev(field_passes)
        field_cv = (field_std / avg_field_pass) * 100 if avg_field_pass > 0 else 100

        avg_latency_ms = mean(latencies)
        latency_std_ms = std_dev(latencies)
        latency_cv = (latency_std_ms / avg_latency_ms) * 100 if avg_latency_ms > 0 else 100

        avg_cost_per_run = mean(costs)

        rows.append(
            AggregatedMetricRow(
                rank=0,
                model_key=model_key,
                provider=sample.model.provider,
                model_id=sample.model.model_id,
                model_label=sample.model.model_label,
                tier=sample.model.tier,
                runs_requested=runs_completed,
                runs_completed=runs_completed,
                successful_runs=successful_runs,
                failed_runs=failed_runs,
                success_rate_pct=_round(success_rate, 2),
                pass_at_2_pct=_pass_threshold_at_n(runs, 2),
                pass_at_3_pct=_pass_threshold_at_n(runs, 3),
                pass_at_5_pct=_pass_threshold_at_n(runs, 5),
                pass_at_10_pct=_pass_threshold_at_n(runs, 10),
                pass_at_2_strict_pct=_pass_strict_at_n(runs, 2),
                pass_at_3_strict_pct=_pass_strict_at_n(runs, 3),
                pass_at_5_strict_pct=_pass_strict_at_n(runs, 5),
                pass_at_10_strict_pct=_pass_strict_at_n(runs, 10),
                avg_total_field_pass_pct=_round(avg_field_pass, 2),
                avg_field_accuracy_pct=_round(avg_field_pass, 2),
                avg_critical_accuracy_pct=_round(mean(critical_passes), 2),
                avg_keys_found_pct=_round(mean(keys_found_pct), 2),
                field_pass_stddev_pct=_round(field_std, 2),
                field_accuracy_variance_pct=_round(field_cv, 2),
                avg_latency_ms=_round(avg_latency_ms, 1),
                p95_latency_ms=_round(percentile(latencies, 95), 1),
                latency_stddev_ms=_round(latency_std_ms, 2),
                latency_cv_pct=_round(latency_cv, 2),
                total_cost_usd=_round(total_cost, 6),
                avg_cost_per_doc_usd=_round(avg_cost_per_run, 6) if costs else None,
                avg_cost_per_run_usd=_round(avg_cost_per_run, 6),
                cost_per_success_usd=(
                    _round(total_cost / successful_runs, 6) if successful_runs > 0 else None
                ),
                p95_cost_usd=_round(percentile(costs, 95), 6),
            )
        )

    def _sort_key(row: AggregatedMetricRow) -> tuple[float, float, float, float]:
        cost = row.cost_per_success_usd if row.cost_per_success_usd is not None else inf
        # TS sorts descending on success_rate, then field_pass, ascending on cost, then latency.
        # Returning negative values for descending keys works with tuple comparison.
        return (-row.success_rate_pct, -row.avg_total_field_pass_pct, cost, row.avg_latency_ms)

    rows.sort(key=_sort_key)
    for index, row in enumerate(rows):
        row.rank = index + 1
    return rows
