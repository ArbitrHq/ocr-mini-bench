import path from 'node:path';
import { loadPreparedDocuments, summarizeDataset } from '../../benchmark/dataset';
import { aggregateMetricRows } from '../../postprocess/aggregate';
import type {
  AggregatedMetricRow,
  ComparisonRecord,
  LeaderboardAggregationSnapshot,
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

function parseArgs(argv: string[]): CliArgs {
  const defaultDir = path.resolve(process.cwd(), 'artifacts/postprocess');

  const out: CliArgs = {
    comparisonJsonl: path.resolve(defaultDir, 'comparison.jsonl'),
    rawJsonl: path.resolve(defaultDir, 'raw.normalized.jsonl'),
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
  const args = parseArgs(process.argv.slice(2));
  const comparisonRecords = await readJsonLinesFile<ComparisonRecord>(args.comparisonJsonl);

  if (comparisonRecords.length === 0) {
    throw new Error(`No comparison records found in ${args.comparisonJsonl}`);
  }

  const overallRows = aggregateMetricRows(comparisonRecords);

  const selectedDomains = Array.from(
    new Set(comparisonRecords.map((record) => record.document.domain.toLowerCase()))
  ).sort((a, b) => a.localeCompare(b));

  const byDomain = selectedDomains.map((domain) => {
    const rows = aggregateMetricRows(
      comparisonRecords.filter((record) => record.document.domain.toLowerCase() === domain)
    );
    return { domain, rows };
  });

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

  const hasIncompletePass10 = overallRows.some((row) => row.pass_at_10_pct === null);
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
    model_rows: overallRows,
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
    leaderboard: overallRows,
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
    leaderboard: overallRows,
    by_domain: byDomain,
    run_count: comparisonRecords.length,
    markdown_table: buildMarkdownTable(overallRows),
    warnings,
    cache_summary: buildCacheSummary(comparisonRecords),
  };

  await Promise.all([
    writeJsonFile(args.outputMetricsJson, metricsSnapshot),
    writeJsonFile(args.outputAggregationJson, aggregation),
    writeJsonFile(args.outputFrontendJson, frontendSnapshot),
  ]);

  console.log(`Metrics rows: ${overallRows.length}`);
  console.log(`Run count: ${comparisonRecords.length}`);
  console.log(`Metrics JSON: ${args.outputMetricsJson}`);
  console.log(`Aggregation JSON: ${args.outputAggregationJson}`);
  console.log(`Frontend JSON: ${args.outputFrontendJson}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
