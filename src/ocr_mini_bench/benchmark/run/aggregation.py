"""In-memory aggregation of per-run metrics into leaderboard rows.

Mirrors `src/benchmark/run/aggregation.ts`. This produces the lighter
`LeaderboardRow` shape used by `latest.json` / `leaderboard.frontend.json`,
distinct from the postprocess `AggregatedMetricRow` which is computed from
the on-disk `comparison.jsonl`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf

from ...config.model_catalog import ModelProvider
from ..types import CacheSummaryEntry, LeaderboardRow, SingleRunMetrics
from .math import mean, pct, percentile, std_dev
from .math import round_half_away_from_zero as _round


def _pass_at_n(success_rate_unit: float, runs_completed: int, n: int) -> float | None:
    """Geometric pass@N estimator: `success_rate ** n * 100`.

    Different from postprocess `pass_threshold_at_n` which uses observed
    per-document success counts. The in-memory variant is a quick estimate
    based on overall success rate; the postprocess variant is the
    contract-grade number.
    """
    if runs_completed < n:
        return None
    return _round((success_rate_unit**n) * 100, 2)


def aggregate_rows(
    run_results: Sequence[SingleRunMetrics],
    requested_by_model: dict[str, int],
) -> list[LeaderboardRow]:
    by_model: dict[str, list[SingleRunMetrics]] = {}
    for run in run_results:
        by_model.setdefault(run.model_key, []).append(run)

    rows: list[LeaderboardRow] = []
    for model_key, runs in by_model.items():
        sample = runs[0]
        runs_completed = len(runs)
        failed_runs = sum(1 for r in runs if r.error)
        successful_runs = sum(1 for r in runs if r.success)

        field_accuracy = [
            (r.field_correct / r.field_total) * 100
            for r in runs
            if not r.error and r.field_total > 0
        ]
        critical_accuracy = [
            (r.critical_correct / r.critical_total) * 100
            for r in runs
            if not r.error and r.critical_total > 0
        ]
        latencies = [float(r.latency_ms) for r in runs if not r.error]
        costs = [r.total_cost_usd for r in runs if not r.error]

        total_cost = sum(costs)
        success_rate = pct(successful_runs, runs_completed)
        success_rate_unit = success_rate / 100
        avg_field_accuracy = mean(field_accuracy)
        field_std = std_dev(field_accuracy)
        field_variance_pct = (
            (field_std / avg_field_accuracy) * 100 if avg_field_accuracy > 0 else 100
        )
        avg_cost_per_run = mean(costs)
        avg_cost_per_doc = avg_cost_per_run if costs else None

        rows.append(
            LeaderboardRow(
                rank=0,
                model_key=model_key,
                provider=sample.provider,
                model_id=sample.model_id,
                model_label=sample.model_label,
                tier=sample.tier,
                runs_requested=requested_by_model.get(model_key, runs_completed),
                runs_completed=runs_completed,
                successful_runs=successful_runs,
                failed_runs=failed_runs,
                success_rate_pct=_round(success_rate, 2),
                pass_at_2_pct=_pass_at_n(success_rate_unit, runs_completed, 2),
                pass_at_3_pct=_pass_at_n(success_rate_unit, runs_completed, 3),
                pass_at_5_pct=_pass_at_n(success_rate_unit, runs_completed, 5),
                pass_at_10_pct=_pass_at_n(success_rate_unit, runs_completed, 10),
                avg_field_accuracy_pct=_round(avg_field_accuracy, 2),
                avg_critical_accuracy_pct=_round(mean(critical_accuracy), 2),
                field_accuracy_variance_pct=_round(field_variance_pct, 2),
                avg_latency_ms=_round(mean(latencies), 1),
                p95_latency_ms=_round(percentile(latencies, 95), 1),
                total_cost_usd=_round(total_cost, 6),
                avg_cost_per_doc_usd=_round(avg_cost_per_doc, 6) if avg_cost_per_doc is not None else None,
                avg_cost_per_run_usd=_round(avg_cost_per_run, 6),
                cost_per_success_usd=(
                    _round(total_cost / successful_runs, 6) if successful_runs > 0 else None
                ),
                p95_cost_usd=_round(percentile(costs, 95), 6),
                p05_field_accuracy_pct=_round(percentile(field_accuracy, 5), 2),
            )
        )

    def _sort_key(row: LeaderboardRow) -> tuple[float, float, float, float]:
        cost = row.cost_per_success_usd if row.cost_per_success_usd is not None else inf
        return (
            -row.success_rate_pct,
            -row.avg_field_accuracy_pct,
            cost,
            row.avg_latency_ms,
        )

    rows.sort(key=_sort_key)
    for index, row in enumerate(rows):
        row.rank = index + 1
    return rows


@dataclass
class _CacheBucket:
    model_label: str
    provider: ModelProvider
    runs: int = 0
    cache_hits: int = 0
    cached_input_tokens_total: int = 0
    cache_write_tokens_total: int = 0


def build_cache_summary(run_results: Sequence[SingleRunMetrics]) -> list[CacheSummaryEntry]:
    by_model: dict[str, _CacheBucket] = {}
    for run in run_results:
        if run.error:
            continue
        bucket = by_model.setdefault(
            run.model_key,
            _CacheBucket(model_label=run.model_label, provider=run.provider),
        )
        bucket.runs += 1
        if run.cache_hit:
            bucket.cache_hits += 1
        bucket.cached_input_tokens_total += run.cached_input_tokens
        bucket.cache_write_tokens_total += run.cache_write_tokens

    entries: list[CacheSummaryEntry] = []
    for model_key, bucket in by_model.items():
        runs = bucket.runs
        entries.append(
            CacheSummaryEntry(
                model_key=model_key,
                model_label=bucket.model_label,
                provider=bucket.provider,
                runs=runs,
                cache_hits=bucket.cache_hits,
                cache_hit_rate_pct=_round(pct(bucket.cache_hits, runs), 2),
                cached_input_tokens_total=round(bucket.cached_input_tokens_total),
                cached_input_tokens_avg=_round(bucket.cached_input_tokens_total / max(1, runs), 1),
                cache_write_tokens_total=round(bucket.cache_write_tokens_total),
                cache_write_tokens_avg=_round(bucket.cache_write_tokens_total / max(1, runs), 1),
            )
        )

    entries.sort(key=lambda e: (-e.cache_hit_rate_pct, e.model_label))
    return entries


def build_markdown_table(rows: Sequence[LeaderboardRow]) -> str:
    head = [
        "| Rank | Model | Provider | Tier | Success % | pass^2 % | pass^3 % | "
        "pass^5 % | pass^10 % | Avg Field % | Critical Field % | Variance % | "
        "Cost / Doc (USD) | Cost / Success (USD) | Avg Latency (ms) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |",
    ]
    body: list[str] = []
    for row in rows:
        pass2 = "" if row.pass_at_2_pct is None else f"{row.pass_at_2_pct:.2f}"
        pass3 = "" if row.pass_at_3_pct is None else f"{row.pass_at_3_pct:.2f}"
        pass5 = "" if row.pass_at_5_pct is None else f"{row.pass_at_5_pct:.2f}"
        pass10 = "" if row.pass_at_10_pct is None else f"{row.pass_at_10_pct:.2f}"
        cost_per_doc = "n/a" if row.avg_cost_per_doc_usd is None else f"{row.avg_cost_per_doc_usd:.4f}"
        cost_per_success = (
            "n/a" if row.cost_per_success_usd is None else f"{row.cost_per_success_usd:.4f}"
        )
        body.append(
            f"| {row.rank} | {row.model_label} | {row.provider} | {row.tier} | "
            f"{row.success_rate_pct:.2f} | {pass2} | {pass3} | {pass5} | {pass10} | "
            f"{row.avg_field_accuracy_pct:.2f} | {row.avg_critical_accuracy_pct:.2f} | "
            f"{row.field_accuracy_variance_pct:.2f} | {cost_per_doc} | {cost_per_success} | "
            f"{row.avg_latency_ms:.1f} |"
        )
    return "\n".join([*head, *body])
