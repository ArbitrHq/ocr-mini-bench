import type { AggregatedMetricRow, ComparisonRecord } from './types';

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdDev(values: number[]): number {
  if (values.length <= 1) return 0;
  const avg = mean(values);
  const variance = Math.max(0, values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length);
  return Math.sqrt(variance);
}

function pct(part: number, total: number): number {
  if (total <= 0) return 0;
  return (part / total) * 100;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[rank];
}

function round(value: number, decimals = 4): number {
  const precision = 10 ** decimals;
  return Math.round(value * precision) / precision;
}

function combination(n: number, k: number): number {
  if (!Number.isFinite(n) || !Number.isFinite(k)) return 0;
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  const kk = Math.min(k, n - k);
  let result = 1;
  for (let i = 1; i <= kk; i += 1) {
    result = (result * (n - kk + i)) / i;
  }
  return result;
}

function documentRunStats(runs: ComparisonRecord[]): Array<{ trials: number; successes: number }> {
  const byDocument = new Map<string, { trials: number; successes: number }>();

  for (const run of runs) {
    const current = byDocument.get(run.document.document_id) ?? { trials: 0, successes: 0 };
    current.trials += 1;
    const succeeded = run.runtime.error === null && run.comparison?.success === true;
    if (succeeded) current.successes += 1;
    byDocument.set(run.document.document_id, current);
  }

  return Array.from(byDocument.values());
}

function passThresholdAtN(runs: ComparisonRecord[], n: number): number | null {
  const stats = documentRunStats(runs);
  if (stats.length === 0) return null;
  if (stats.some((doc) => doc.trials < n)) return null;
  const passingDocs = stats.filter((doc) => doc.successes >= n).length;
  return round(pct(passingDocs, stats.length), 2);
}

function passStrictAtN(runs: ComparisonRecord[], n: number): number | null {
  const stats = documentRunStats(runs);
  if (stats.length === 0) return null;
  if (stats.some((doc) => doc.trials < n)) return null;

  let sum = 0;
  for (const doc of stats) {
    const denominator = combination(doc.trials, n);
    const numerator = combination(doc.successes, n);
    sum += denominator > 0 ? numerator / denominator : 0;
  }

  return round((sum / stats.length) * 100, 2);
}

export function aggregateMetricRows(records: ComparisonRecord[]): AggregatedMetricRow[] {
  const byModel = new Map<string, ComparisonRecord[]>();

  for (const record of records) {
    const key = record.model.model_key;
    const list = byModel.get(key) ?? [];
    list.push(record);
    byModel.set(key, list);
  }

  const rows: AggregatedMetricRow[] = [];

  for (const [modelKey, runs] of byModel.entries()) {
    const sample = runs[0];
    const runsCompleted = runs.length;
    const failedRuns = runs.filter((run) => run.runtime.error !== null).length;
    const successfulRuns = runs.filter((run) => run.runtime.error === null && run.comparison?.success === true).length;

    const scoredRuns = runs.filter((run) => run.runtime.error === null && run.comparison !== null);

    const fieldPasses = scoredRuns.map((run) => run.comparison!.field_pass_pct);
    const criticalPasses = scoredRuns.map((run) => run.comparison!.critical_pass_pct);
    const keysFoundPct = scoredRuns.map((run) => run.comparison!.keys_found_pct);

    const latencies = scoredRuns.map((run) => run.runtime.latency_ms);
    const costs = scoredRuns.map((run) => run.runtime.total_cost_usd);

    const totalCost = costs.reduce((sum, value) => sum + value, 0);
    const successRate = pct(successfulRuns, runsCompleted);

    const avgFieldPass = mean(fieldPasses);
    const fieldStdDev = stdDev(fieldPasses);
    const fieldCv = avgFieldPass > 0 ? (fieldStdDev / avgFieldPass) * 100 : 100;

    const avgLatencyMs = mean(latencies);
    const latencyStdDevMs = stdDev(latencies);
    const latencyCv = avgLatencyMs > 0 ? (latencyStdDevMs / avgLatencyMs) * 100 : 100;

    const avgCostPerRun = mean(costs);

    rows.push({
      rank: 0,
      model_key: modelKey,
      provider: sample.model.provider,
      model_id: sample.model.model_id,
      model_label: sample.model.model_label,
      tier: sample.model.tier,
      runs_requested: runsCompleted,
      runs_completed: runsCompleted,
      successful_runs: successfulRuns,
      failed_runs: failedRuns,
      success_rate_pct: round(successRate, 2),
      pass_at_2_pct: passThresholdAtN(runs, 2),
      pass_at_3_pct: passThresholdAtN(runs, 3),
      pass_at_5_pct: passThresholdAtN(runs, 5),
      pass_at_10_pct: passThresholdAtN(runs, 10),
      pass_at_2_strict_pct: passStrictAtN(runs, 2),
      pass_at_3_strict_pct: passStrictAtN(runs, 3),
      pass_at_5_strict_pct: passStrictAtN(runs, 5),
      pass_at_10_strict_pct: passStrictAtN(runs, 10),
      avg_total_field_pass_pct: round(avgFieldPass, 2),
      avg_field_accuracy_pct: round(avgFieldPass, 2),
      avg_critical_accuracy_pct: round(mean(criticalPasses), 2),
      avg_keys_found_pct: round(mean(keysFoundPct), 2),
      field_pass_stddev_pct: round(fieldStdDev, 2),
      field_accuracy_variance_pct: round(fieldCv, 2),
      avg_latency_ms: round(avgLatencyMs, 1),
      p95_latency_ms: round(percentile(latencies, 95), 1),
      latency_stddev_ms: round(latencyStdDevMs, 2),
      latency_cv_pct: round(latencyCv, 2),
      total_cost_usd: round(totalCost, 6),
      avg_cost_per_doc_usd: costs.length > 0 ? round(avgCostPerRun, 6) : null,
      avg_cost_per_run_usd: round(avgCostPerRun, 6),
      cost_per_success_usd: successfulRuns > 0 ? round(totalCost / successfulRuns, 6) : null,
      p95_cost_usd: round(percentile(costs, 95), 6),
    });
  }

  rows.sort((a, b) => {
    if (b.success_rate_pct !== a.success_rate_pct) return b.success_rate_pct - a.success_rate_pct;
    if (b.avg_total_field_pass_pct !== a.avg_total_field_pass_pct) {
      return b.avg_total_field_pass_pct - a.avg_total_field_pass_pct;
    }

    const aCost = a.cost_per_success_usd ?? Number.POSITIVE_INFINITY;
    const bCost = b.cost_per_success_usd ?? Number.POSITIVE_INFINITY;
    if (aCost !== bCost) return aCost - bCost;

    return a.avg_latency_ms - b.avg_latency_ms;
  });

  return rows.map((row, index) => ({ ...row, rank: index + 1 }));
}
