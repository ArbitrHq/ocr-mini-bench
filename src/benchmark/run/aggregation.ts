import type { BenchmarkModelConfig, LeaderboardRow, SingleRunMetrics } from '../types';
import { mean, pct, percentile, round, stdDev } from './math';

function passAtN(successRateUnit: number, runsCompleted: number, n: number): number | null {
  if (runsCompleted < n) return null;
  return round(Math.pow(successRateUnit, n) * 100, 2);
}

export function aggregateRows(
  runResults: SingleRunMetrics[],
  requestedByModel: Map<string, number>
): LeaderboardRow[] {
  const byModel = new Map<string, SingleRunMetrics[]>();

  for (const run of runResults) {
    const list = byModel.get(run.model_key) ?? [];
    list.push(run);
    byModel.set(run.model_key, list);
  }

  const rows: LeaderboardRow[] = [];

  for (const [modelKey, runs] of byModel.entries()) {
    const sample = runs[0];
    const runsCompleted = runs.length;
    const failedRuns = runs.filter((run) => Boolean(run.error)).length;
    const successfulRuns = runs.filter((run) => run.success).length;

    const fieldAccuracy = runs
      .filter((run) => !run.error && run.field_total > 0)
      .map((run) => (run.field_correct / run.field_total) * 100);

    const criticalAccuracy = runs
      .filter((run) => !run.error && run.critical_total > 0)
      .map((run) => (run.critical_correct / run.critical_total) * 100);

    const latencies = runs.filter((run) => !run.error).map((run) => run.latency_ms);
    const costs = runs.filter((run) => !run.error).map((run) => run.total_cost_usd);

    const totalCost = costs.reduce((sum, value) => sum + value, 0);
    const successRate = pct(successfulRuns, runsCompleted);
    const successRateUnit = successRate / 100;
    const avgFieldAccuracy = mean(fieldAccuracy);
    const fieldStdDev = stdDev(fieldAccuracy);
    const fieldAccuracyVariancePct = avgFieldAccuracy > 0 ? (fieldStdDev / avgFieldAccuracy) * 100 : 100;
    const avgCostPerRun = mean(costs);
    const avgCostPerDoc = costs.length > 0 ? avgCostPerRun : null;

    rows.push({
      rank: 0,
      model_key: modelKey,
      provider: sample.provider,
      model_id: sample.model_id,
      model_label: sample.model_label,
      tier: sample.tier,
      runs_requested: requestedByModel.get(modelKey) ?? runsCompleted,
      runs_completed: runsCompleted,
      successful_runs: successfulRuns,
      failed_runs: failedRuns,
      success_rate_pct: round(successRate, 2),
      pass_at_2_pct: passAtN(successRateUnit, runsCompleted, 2),
      pass_at_3_pct: passAtN(successRateUnit, runsCompleted, 3),
      pass_at_5_pct: passAtN(successRateUnit, runsCompleted, 5),
      pass_at_10_pct: passAtN(successRateUnit, runsCompleted, 10),
      avg_field_accuracy_pct: round(avgFieldAccuracy, 2),
      avg_critical_accuracy_pct: round(mean(criticalAccuracy), 2),
      field_accuracy_variance_pct: round(fieldAccuracyVariancePct, 2),
      avg_latency_ms: round(mean(latencies), 1),
      p95_latency_ms: round(percentile(latencies, 95), 1),
      total_cost_usd: round(totalCost, 6),
      avg_cost_per_doc_usd: avgCostPerDoc === null ? null : round(avgCostPerDoc, 6),
      avg_cost_per_run_usd: round(avgCostPerRun, 6),
      cost_per_success_usd: successfulRuns > 0 ? round(totalCost / successfulRuns, 6) : null,
      p95_cost_usd: round(percentile(costs, 95), 6),
      p05_field_accuracy_pct: round(percentile(fieldAccuracy, 5), 2),
    });
  }

  rows.sort((a, b) => {
    if (b.success_rate_pct !== a.success_rate_pct) return b.success_rate_pct - a.success_rate_pct;
    if (b.avg_field_accuracy_pct !== a.avg_field_accuracy_pct) {
      return b.avg_field_accuracy_pct - a.avg_field_accuracy_pct;
    }

    const aCost = a.cost_per_success_usd ?? Number.POSITIVE_INFINITY;
    const bCost = b.cost_per_success_usd ?? Number.POSITIVE_INFINITY;
    if (aCost !== bCost) return aCost - bCost;

    return a.avg_latency_ms - b.avg_latency_ms;
  });

  return rows.map((row, index) => ({ ...row, rank: index + 1 }));
}

export function buildCacheSummary(runResults: SingleRunMetrics[]): Array<{
  model_key: string;
  model_label: string;
  provider: BenchmarkModelConfig['provider'];
  runs: number;
  cache_hits: number;
  cache_hit_rate_pct: number;
  cached_input_tokens_total: number;
  cached_input_tokens_avg: number;
  cache_write_tokens_total: number;
  cache_write_tokens_avg: number;
}> {
  const byModel = new Map<
    string,
    {
      model_label: string;
      provider: BenchmarkModelConfig['provider'];
      runs: number;
      cache_hits: number;
      cached_input_tokens_total: number;
      cache_write_tokens_total: number;
    }
  >();

  for (const run of runResults) {
    if (run.error) continue;
    const current = byModel.get(run.model_key) ?? {
      model_label: run.model_label,
      provider: run.provider,
      runs: 0,
      cache_hits: 0,
      cached_input_tokens_total: 0,
      cache_write_tokens_total: 0,
    };
    current.runs += 1;
    if (run.cache_hit) current.cache_hits += 1;
    current.cached_input_tokens_total += run.cached_input_tokens;
    current.cache_write_tokens_total += run.cache_write_tokens;
    byModel.set(run.model_key, current);
  }

  return Array.from(byModel.entries())
    .map(([model_key, value]) => ({
      model_key,
      model_label: value.model_label,
      provider: value.provider,
      runs: value.runs,
      cache_hits: value.cache_hits,
      cache_hit_rate_pct: round(pct(value.cache_hits, value.runs), 2),
      cached_input_tokens_total: Math.round(value.cached_input_tokens_total),
      cached_input_tokens_avg: round(value.cached_input_tokens_total / Math.max(1, value.runs), 1),
      cache_write_tokens_total: Math.round(value.cache_write_tokens_total),
      cache_write_tokens_avg: round(value.cache_write_tokens_total / Math.max(1, value.runs), 1),
    }))
    .sort((a, b) => b.cache_hit_rate_pct - a.cache_hit_rate_pct || a.model_label.localeCompare(b.model_label));
}

export function buildMarkdownTable(rows: LeaderboardRow[]): string {
  const head = [
    '| Rank | Model | Provider | Tier | Success % | pass^2 % | pass^3 % | pass^5 % | pass^10 % | Avg Field % | Critical Field % | Variance % | Cost / Doc (USD) | Cost / Success (USD) | Avg Latency (ms) |',
    '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  ];

  const body = rows.map((row) => {
    const pass2 = row.pass_at_2_pct === null ? '' : row.pass_at_2_pct.toFixed(2);
    const pass3 = row.pass_at_3_pct === null ? '' : row.pass_at_3_pct.toFixed(2);
    const pass5 = row.pass_at_5_pct === null ? '' : row.pass_at_5_pct.toFixed(2);
    const pass10 = row.pass_at_10_pct === null ? '' : row.pass_at_10_pct.toFixed(2);
    const costPerDoc = row.avg_cost_per_doc_usd === null ? 'n/a' : row.avg_cost_per_doc_usd.toFixed(4);
    const costPerSuccess = row.cost_per_success_usd === null ? 'n/a' : row.cost_per_success_usd.toFixed(4);
    return `| ${row.rank} | ${row.model_label} | ${row.provider} | ${row.tier} | ${row.success_rate_pct.toFixed(2)} | ${pass2} | ${pass3} | ${pass5} | ${pass10} | ${row.avg_field_accuracy_pct.toFixed(2)} | ${row.avg_critical_accuracy_pct.toFixed(2)} | ${row.field_accuracy_variance_pct.toFixed(2)} | ${costPerDoc} | ${costPerSuccess} | ${row.avg_latency_ms.toFixed(1)} |`;
  });

  return [...head, ...body].join('\n');
}
