import path from 'node:path';
import { loadPreparedDocuments, summarizeDataset } from '../../benchmark/dataset';
import { aggregateMetricRows } from '../../postprocess/aggregate';
import type {
  AggregatedMetricRow,
  ComparisonRecord,
  LeaderboardAggregationSnapshot,
  MetricRangeStats,
  MetricRanges,
  MetricsSnapshot,
} from '../../postprocess/types';
import { readJsonLinesFile, timestampForFilename, writeJsonFile } from '../../postprocess/io';

type CliArgs = {
  comparisonJsonl: string;
  rawJsonl: string | null;
  outputMetricsJson: string;
  outputAggregationJson: string;
  outputFrontendJson: string;
};

function printHelp(): void {
  console.log(`Usage:
  npm run postprocess:metrics -- [options]

Options:
  --comparison-jsonl=<path>       Comparison JSONL input (default: artifacts/postprocess/comparison.jsonl)
  --raw-jsonl=<path>              Raw JSONL path for metadata only (default: artifacts/postprocess/raw.jsonl)
  --output-metrics-json=<path>    Metrics snapshot output (default: artifacts/postprocess/metrics.snapshot.json)
  --output-aggregation-json=<path> Leaderboard aggregation output (default: artifacts/postprocess/leaderboard.aggregation.json)
  --output-frontend-json=<path>   Frontend snapshot output (default: artifacts/postprocess/leaderboard.frontend.json)
  -h, --help                      Show this help

Examples:
  npm run postprocess:metrics
  npm run postprocess:metrics -- --comparison-jsonl=artifacts/smoke/postprocess/comparison.jsonl
`);
}

function wantsHelp(argv: string[]): boolean {
  return argv.includes('--help') || argv.includes('-h');
}

function parseArgs(argv: string[]): CliArgs {
  const defaultDir = path.resolve(process.cwd(), 'artifacts/postprocess');

  const out: CliArgs = {
    comparisonJsonl: path.resolve(defaultDir, 'comparison.jsonl'),
    rawJsonl: path.resolve(defaultDir, 'raw.jsonl'),
    outputMetricsJson: path.resolve(defaultDir, 'metrics.snapshot.json'),
    outputAggregationJson: path.resolve(defaultDir, 'leaderboard.aggregation.json'),
    outputFrontendJson: path.resolve(defaultDir, 'leaderboard.frontend.json'),
  };

  for (const arg of argv) {
    if (arg.startsWith('--comparison-jsonl=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.comparisonJsonl = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg.startsWith('--raw-jsonl=')) {
      const value = arg.split('=')[1] ?? '';
      out.rawJsonl = value.trim() ? path.resolve(process.cwd(), value.trim()) : null;
      continue;
    }
    if (arg.startsWith('--output-metrics-json=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.outputMetricsJson = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg.startsWith('--output-aggregation-json=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.outputAggregationJson = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg.startsWith('--output-frontend-json=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.outputFrontendJson = path.resolve(process.cwd(), value.trim());
    }
  }

  return out;
}

type RangeDomainKey = 'overall' | string;

type DocRangeAccumulator = {
  cost_total_usd: number;
  run_count: number;
  success_count: number;
};

type BucketRangeAccumulator = {
  success_pct_runs: number[];
  critical_fields_pct: number[];
  all_fields_pct: number[];
  latency_ms: number[];
  docs: Map<string, DocRangeAccumulator>;
};

function round(value: number, decimals: number): number {
  const precision = 10 ** decimals;
  return Math.round(value * precision) / precision;
}

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

function percentileNearestRank(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
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

function bucketKey(domain: RangeDomainKey, modelKey: string): string {
  return `${domain}::${modelKey}`;
}

function emptyMetricRangeStats(): MetricRangeStats {
  return {
    count: 0,
    min: null,
    p05: null,
    p50: null,
    p95: null,
    max: null,
    mean: null,
    stddev: null,
  };
}

function summarizeRange(values: number[], decimals: number): MetricRangeStats {
  if (values.length === 0) {
    return emptyMetricRangeStats();
  }

  return {
    count: values.length,
    min: round(Math.min(...values), decimals),
    p05: round(percentileNearestRank(values, 5), decimals),
    p50: round(percentileNearestRank(values, 50), decimals),
    p95: round(percentileNearestRank(values, 95), decimals),
    max: round(Math.max(...values), decimals),
    mean: round(mean(values), decimals),
    stddev: round(stdDev(values), decimals),
  };
}

function emptyMetricRanges(): MetricRanges {
  return {
    cost_per_doc_usd: emptyMetricRangeStats(),
    cost_per_success_usd: emptyMetricRangeStats(),
    success_pct_runs: emptyMetricRangeStats(),
    success_pct_docs: emptyMetricRangeStats(),
    pass_at_3_strict_pct_docs: emptyMetricRangeStats(),
    pass_at_5_strict_pct_docs: emptyMetricRangeStats(),
    critical_fields_pct: emptyMetricRangeStats(),
    all_fields_pct: emptyMetricRangeStats(),
    latency_ms: emptyMetricRangeStats(),
  };
}

function ensureBucket(
  map: Map<string, BucketRangeAccumulator>,
  domain: RangeDomainKey,
  modelKey: string
): BucketRangeAccumulator {
  const key = bucketKey(domain, modelKey);
  const existing = map.get(key);
  if (existing) return existing;

  const created: BucketRangeAccumulator = {
    success_pct_runs: [],
    critical_fields_pct: [],
    all_fields_pct: [],
    latency_ms: [],
    docs: new Map<string, DocRangeAccumulator>(),
  };
  map.set(key, created);
  return created;
}

function addRecordToBucket(
  bucket: BucketRangeAccumulator,
  docKey: string,
  costUsd: number,
  latencyMs: number,
  criticalFieldsPct: number,
  allFieldsPct: number,
  success: boolean
): void {
  bucket.success_pct_runs.push(success ? 100 : 0);
  bucket.critical_fields_pct.push(criticalFieldsPct);
  bucket.all_fields_pct.push(allFieldsPct);
  bucket.latency_ms.push(latencyMs);

  const doc = bucket.docs.get(docKey) ?? { cost_total_usd: 0, run_count: 0, success_count: 0 };
  doc.cost_total_usd += costUsd;
  doc.run_count += 1;
  if (success) doc.success_count += 1;
  bucket.docs.set(docKey, doc);
}

function finalizeBucketRanges(bucket: BucketRangeAccumulator): MetricRanges {
  const costPerDoc: number[] = [];
  const costPerSuccess: number[] = [];
  const successPctDocs: number[] = [];
  const passAt3StrictPctDocs: number[] = [];
  const passAt5StrictPctDocs: number[] = [];

  for (const doc of bucket.docs.values()) {
    if (doc.run_count > 0) {
      costPerDoc.push(doc.cost_total_usd / doc.run_count);
      successPctDocs.push((doc.success_count / doc.run_count) * 100);
    }
    if (doc.success_count > 0) {
      costPerSuccess.push(doc.cost_total_usd / doc.success_count);
    }

    if (doc.run_count >= 3) {
      const denominator = combination(doc.run_count, 3);
      const numerator = combination(doc.success_count, 3);
      passAt3StrictPctDocs.push(denominator > 0 ? (numerator / denominator) * 100 : 0);
    }
    if (doc.run_count >= 5) {
      const denominator = combination(doc.run_count, 5);
      const numerator = combination(doc.success_count, 5);
      passAt5StrictPctDocs.push(denominator > 0 ? (numerator / denominator) * 100 : 0);
    }
  }

  return {
    cost_per_doc_usd: summarizeRange(costPerDoc, 6),
    cost_per_success_usd: summarizeRange(costPerSuccess, 6),
    success_pct_runs: summarizeRange(bucket.success_pct_runs, 3),
    success_pct_docs: summarizeRange(successPctDocs, 3),
    pass_at_3_strict_pct_docs: summarizeRange(passAt3StrictPctDocs, 3),
    pass_at_5_strict_pct_docs: summarizeRange(passAt5StrictPctDocs, 3),
    critical_fields_pct: summarizeRange(bucket.critical_fields_pct, 3),
    all_fields_pct: summarizeRange(bucket.all_fields_pct, 3),
    latency_ms: summarizeRange(bucket.latency_ms, 3),
  };
}

function computeMetricRangesByBucket(records: ComparisonRecord[]): Map<string, MetricRanges> {
  const accumulators = new Map<string, BucketRangeAccumulator>();

  for (const record of records) {
    if (record.runtime.error !== null) continue;

    const domain = record.document.domain.toLowerCase();
    const modelKey = record.model.model_key;
    const docKey = `${domain}::${record.document.document_id}`;

    const success = record.comparison?.success ?? record.legacy_metrics.success;
    const criticalFieldsPct =
      typeof record.comparison?.critical_pass_pct === 'number'
        ? record.comparison.critical_pass_pct
        : record.legacy_metrics.critical_accuracy_pct;
    const allFieldsPct =
      typeof record.comparison?.field_pass_pct === 'number'
        ? record.comparison.field_pass_pct
        : record.legacy_metrics.field_accuracy_pct;

    const domainBucket = ensureBucket(accumulators, domain, modelKey);
    addRecordToBucket(
      domainBucket,
      docKey,
      record.runtime.total_cost_usd,
      record.runtime.latency_ms,
      criticalFieldsPct,
      allFieldsPct,
      success
    );

    const overallBucket = ensureBucket(accumulators, 'overall', modelKey);
    addRecordToBucket(
      overallBucket,
      docKey,
      record.runtime.total_cost_usd,
      record.runtime.latency_ms,
      criticalFieldsPct,
      allFieldsPct,
      success
    );
  }

  const out = new Map<string, MetricRanges>();
  for (const [key, bucket] of accumulators.entries()) {
    out.set(key, finalizeBucketRanges(bucket));
  }
  return out;
}

function attachMetricRanges(
  rows: AggregatedMetricRow[],
  domain: RangeDomainKey,
  metricRangesByBucket: Map<string, MetricRanges>
): AggregatedMetricRow[] {
  return rows.map((row) => ({
    ...row,
    metric_ranges: metricRangesByBucket.get(bucketKey(domain, row.model_key)) ?? emptyMetricRanges(),
  }));
}

function buildMarkdownTable(rows: AggregatedMetricRow[]): string {
  const head = [
    '| Rank | Model | Provider | Tier | Success % | pass^2 (>=2) % | pass^3 (>=3) % | pass^5 (>=5) % | pass^10 (>=10) % | pass^2 strict % | pass^3 strict % | pass^5 strict % | pass^10 strict % | Total Field Pass % | Critical Field % | Keys Found % | Field Variance (CV) % | Cost / Doc (USD) | Cost / Success (USD) | Avg Latency (ms) |',
    '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  ];

  const body = rows.map((row) => {
    const pass2 = row.pass_at_2_pct === null ? '' : row.pass_at_2_pct.toFixed(2);
    const pass3 = row.pass_at_3_pct === null ? '' : row.pass_at_3_pct.toFixed(2);
    const pass5 = row.pass_at_5_pct === null ? '' : row.pass_at_5_pct.toFixed(2);
    const pass10 = row.pass_at_10_pct === null ? '' : row.pass_at_10_pct.toFixed(2);

    const pass2Strict = row.pass_at_2_strict_pct === null ? '' : row.pass_at_2_strict_pct.toFixed(2);
    const pass3Strict = row.pass_at_3_strict_pct === null ? '' : row.pass_at_3_strict_pct.toFixed(2);
    const pass5Strict = row.pass_at_5_strict_pct === null ? '' : row.pass_at_5_strict_pct.toFixed(2);
    const pass10Strict = row.pass_at_10_strict_pct === null ? '' : row.pass_at_10_strict_pct.toFixed(2);

    const costPerDoc = row.avg_cost_per_doc_usd === null ? 'n/a' : row.avg_cost_per_doc_usd.toFixed(4);
    const costPerSuccess = row.cost_per_success_usd === null ? 'n/a' : row.cost_per_success_usd.toFixed(4);

    return `| ${row.rank} | ${row.model_label} | ${row.provider} | ${row.tier} | ${row.success_rate_pct.toFixed(2)} | ${pass2} | ${pass3} | ${pass5} | ${pass10} | ${pass2Strict} | ${pass3Strict} | ${pass5Strict} | ${pass10Strict} | ${row.avg_total_field_pass_pct.toFixed(2)} | ${row.avg_critical_accuracy_pct.toFixed(2)} | ${row.avg_keys_found_pct.toFixed(2)} | ${row.field_accuracy_variance_pct.toFixed(2)} | ${costPerDoc} | ${costPerSuccess} | ${row.avg_latency_ms.toFixed(1)} |`;
  });

  return [...head, ...body].join('\n');
}

function inferRunsPerModel(records: ComparisonRecord[]): number {
  const byModelDoc = new Map<string, number>();

  for (const record of records) {
    const key = `${record.model.model_key}::${record.document.document_id}`;
    byModelDoc.set(key, (byModelDoc.get(key) ?? 0) + 1);
  }

  const counts = Array.from(byModelDoc.values());
  if (counts.length === 0) return 0;
  return Math.max(...counts);
}

function buildCacheSummary(records: ComparisonRecord[]): Array<{
  model_key: string;
  model_label: string;
  provider: string;
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
      provider: string;
      runs: number;
      cache_hits: number;
      cached_input_tokens_total: number;
      cache_write_tokens_total: number;
    }
  >();

  for (const record of records) {
    if (record.runtime.error !== null) continue;

    const current = byModel.get(record.model.model_key) ?? {
      model_label: record.model.model_label,
      provider: record.model.provider,
      runs: 0,
      cache_hits: 0,
      cached_input_tokens_total: 0,
      cache_write_tokens_total: 0,
    };

    current.runs += 1;
    if (record.runtime.cache_hit) current.cache_hits += 1;
    current.cached_input_tokens_total += record.runtime.cached_input_tokens;
    current.cache_write_tokens_total += record.runtime.cache_write_tokens;

    byModel.set(record.model.model_key, current);
  }

  return Array.from(byModel.entries())
    .map(([model_key, value]) => ({
      model_key,
      model_label: value.model_label,
      provider: value.provider,
      runs: value.runs,
      cache_hits: value.cache_hits,
      cache_hit_rate_pct: value.runs > 0 ? Number(((value.cache_hits / value.runs) * 100).toFixed(2)) : 0,
      cached_input_tokens_total: Math.round(value.cached_input_tokens_total),
      cached_input_tokens_avg:
        value.runs > 0 ? Number((value.cached_input_tokens_total / value.runs).toFixed(1)) : 0,
      cache_write_tokens_total: Math.round(value.cache_write_tokens_total),
      cache_write_tokens_avg:
        value.runs > 0 ? Number((value.cache_write_tokens_total / value.runs).toFixed(1)) : 0,
    }))
    .sort((a, b) => b.cache_hit_rate_pct - a.cache_hit_rate_pct || a.model_label.localeCompare(b.model_label));
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (wantsHelp(argv)) {
    printHelp();
    return;
  }
  const args = parseArgs(argv);
  const comparisonRecords = await readJsonLinesFile<ComparisonRecord>(args.comparisonJsonl);

  if (comparisonRecords.length === 0) {
    throw new Error(`No comparison records found in ${args.comparisonJsonl}`);
  }

  const overallRows = aggregateMetricRows(comparisonRecords);
  const metricRangesByBucket = computeMetricRangesByBucket(comparisonRecords);

  const selectedDomains = Array.from(
    new Set(comparisonRecords.map((record) => record.document.domain.toLowerCase()))
  ).sort((a, b) => a.localeCompare(b));

  const byDomainRows = selectedDomains.map((domain) => {
    const rows = aggregateMetricRows(
      comparisonRecords.filter((record) => record.document.domain.toLowerCase() === domain)
    );
    return { domain, rows };
  });

  const overallRowsWithRanges = attachMetricRanges(overallRows, 'overall', metricRangesByBucket);
  const byDomain = byDomainRows.map(({ domain, rows }) => ({
    domain,
    rows: attachMetricRanges(rows, domain, metricRangesByBucket),
  }));

  const preparedDocuments = await loadPreparedDocuments({ domains: selectedDomains });
  const selectedDocIds = new Set(
    comparisonRecords.map((record) => `${record.document.domain.toLowerCase()}::${record.document.document_id}`)
  );
  const selectedDocs = preparedDocuments.filter((doc) =>
    selectedDocIds.has(`${doc.domain.toLowerCase()}::${doc.document_id}`)
  );
  const dataset = summarizeDataset(selectedDocs);

  const warnings: string[] = [];
  if (dataset.labeled_keys < dataset.total_keys) {
    warnings.push(`Only ${dataset.labeled_keys}/${dataset.total_keys} keys are currently labeled.`);
  }
  for (const [domain, count] of Object.entries(dataset.documents_per_domain)) {
    if (count < 10) {
      warnings.push(`Domain "${domain}" has ${count} documents; target is >= 10 for launch quality.`);
    }
  }

  const hasIncompletePass10 = overallRowsWithRanges.some((row) => row.pass_at_10_pct === null);
  if (hasIncompletePass10) {
    warnings.push('pass^10 is empty for one or more models because at least one document has fewer than 10 completed runs.');
  }

  const metricsSnapshot: MetricsSnapshot = {
    schema_version: '1.0',
    generated_at: new Date().toISOString(),
    source: {
      comparison_jsonl: args.comparisonJsonl,
    },
    run_count: comparisonRecords.length,
    model_rows: overallRowsWithRanges,
    by_domain: byDomain,
  };

  const aggregation: LeaderboardAggregationSnapshot = {
    schema_version: '1.0',
    generated_at: metricsSnapshot.generated_at,
    source: {
      raw_jsonl: args.rawJsonl ?? '',
      comparison_jsonl: args.comparisonJsonl,
      metrics_json: args.outputMetricsJson,
    },
    dataset,
    leaderboard: overallRowsWithRanges,
    by_domain: byDomain,
    run_count: comparisonRecords.length,
    warnings,
  };

  const frontendSnapshot = {
    generated_at: metricsSnapshot.generated_at,
    benchmark_id: `ocr-benchmark-postprocess-${timestampForFilename(new Date(metricsSnapshot.generated_at))}`,
    benchmark_description: 'Postprocessed OCR mini-bench (raw -> comparison -> metrics).',
    options: {
      runs_per_model: inferRunsPerModel(comparisonRecords),
      max_parallel_requests: 1,
      provider_parallel: false,
      selected_domains: selectedDomains,
      max_documents_per_domain: null,
    },
    dataset,
    leaderboard: overallRowsWithRanges,
    by_domain: byDomain,
    run_count: comparisonRecords.length,
    markdown_table: buildMarkdownTable(overallRowsWithRanges),
    warnings,
    cache_summary: buildCacheSummary(comparisonRecords),
  };

  await Promise.all([
    writeJsonFile(args.outputMetricsJson, metricsSnapshot),
    writeJsonFile(args.outputAggregationJson, aggregation),
    writeJsonFile(args.outputFrontendJson, frontendSnapshot),
  ]);

  console.log(`Metrics rows: ${overallRowsWithRanges.length}`);
  console.log(`Run count: ${comparisonRecords.length}`);
  console.log(`Metrics JSON: ${args.outputMetricsJson}`);
  console.log(`Aggregation JSON: ${args.outputAggregationJson}`);
  console.log(`Frontend JSON: ${args.outputFrontendJson}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
