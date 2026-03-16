import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..');

function parseArgs(argv) {
  let checkpointSet = false;
  let outputSet = false;
  const out = {
    repoRoot: REPO_ROOT,
    checkpointDir: path.resolve(REPO_ROOT, 'artifacts/checkpoints'),
    outputDir: path.resolve(REPO_ROOT, 'artifacts'),
  };

  for (const arg of argv) {
    if (arg.startsWith('--checkpoint-dir=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) {
        out.checkpointDir = path.resolve(process.cwd(), value.trim());
        checkpointSet = true;
      }
      continue;
    }
    if (arg.startsWith('--repo-root=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.repoRoot = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg.startsWith('--output-dir=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) {
        out.outputDir = path.resolve(process.cwd(), value.trim());
        outputSet = true;
      }
      continue;
    }
  }

  if (!checkpointSet) {
    out.checkpointDir = path.resolve(out.repoRoot, 'artifacts/checkpoints');
  }
  if (!outputSet) {
    out.outputDir = path.resolve(out.repoRoot, 'artifacts');
  }

  return out;
}

function normalizeDomain(value) {
  return String(value ?? '').trim().toLowerCase();
}

function mean(values) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdDev(values) {
  if (values.length <= 1) return 0;
  const avg = mean(values);
  const variance = Math.max(0, values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length);
  return Math.sqrt(variance);
}

function pct(part, total) {
  if (total <= 0) return 0;
  return (part / total) * 100;
}

function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[rank];
}

function round(value, decimals = 4) {
  const precision = 10 ** decimals;
  return Math.round(value * precision) / precision;
}

function idForModel(provider, modelId) {
  return `${provider}:${modelId}`;
}

function combination(n, k) {
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

function documentRunStats(runs) {
  const byDocument = new Map();
  for (const run of runs) {
    const current = byDocument.get(run.document_id) ?? { trials: 0, successes: 0 };
    current.trials += 1;
    if (run.success) current.successes += 1;
    byDocument.set(run.document_id, current);
  }
  return Array.from(byDocument.values());
}

function atLeastPassAtN(runs, n) {
  const stats = documentRunStats(runs);
  if (stats.length === 0) return null;
  if (stats.some((doc) => doc.trials < n)) return null;
  const passingDocs = stats.filter((doc) => doc.successes >= n).length;
  return round(pct(passingDocs, stats.length), 2);
}

function strictPassAtN(runs, n) {
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

function aggregateRows(runResults, requestedByModel) {
  const byModel = new Map();

  for (const run of runResults) {
    const list = byModel.get(run.model_key) ?? [];
    list.push(run);
    byModel.set(run.model_key, list);
  }

  const rows = [];

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
    const keysFoundPct = runs
      .filter((run) => !run.error && run.requested_key_count > 0)
      .map((run) => (run.found_key_count / run.requested_key_count) * 100);

    const totalCost = costs.reduce((sum, value) => sum + value, 0);
    const successRate = pct(successfulRuns, runsCompleted);
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
      pass_at_2_pct: atLeastPassAtN(runs, 2),
      pass_at_3_pct: atLeastPassAtN(runs, 3),
      pass_at_5_pct: atLeastPassAtN(runs, 5),
      pass_at_10_pct: atLeastPassAtN(runs, 10),
      pass_at_2_strict_pct: strictPassAtN(runs, 2),
      pass_at_3_strict_pct: strictPassAtN(runs, 3),
      pass_at_5_strict_pct: strictPassAtN(runs, 5),
      pass_at_10_strict_pct: strictPassAtN(runs, 10),
      avg_keys_found_pct: round(mean(keysFoundPct), 2),
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
    if (b.avg_field_accuracy_pct !== a.avg_field_accuracy_pct) return b.avg_field_accuracy_pct - a.avg_field_accuracy_pct;
    const aCost = a.cost_per_success_usd ?? Number.POSITIVE_INFINITY;
    const bCost = b.cost_per_success_usd ?? Number.POSITIVE_INFINITY;
    if (aCost !== bCost) return aCost - bCost;
    return a.avg_latency_ms - b.avg_latency_ms;
  });

  return rows.map((row, index) => ({ ...row, rank: index + 1 }));
}

function buildCacheSummary(runResults) {
  const byModel = new Map();

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

function buildMarkdownTable(rows) {
  const head = [
    '| Rank | Model | Provider | Tier | Success % | pass^2 (>=2) % | pass^3 (>=3) % | pass^5 (>=5) % | pass^10 (>=10) % | pass^2 strict % | pass^3 strict % | pass^5 strict % | pass^10 strict % | Keys Found % | Avg Field % | Critical Field % | Variance % | Cost / Doc (USD) | Cost / Success (USD) | Avg Latency (ms) |',
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

    return `| ${row.rank} | ${row.model_label} | ${row.provider} | ${row.tier} | ${row.success_rate_pct.toFixed(2)} | ${pass2} | ${pass3} | ${pass5} | ${pass10} | ${pass2Strict} | ${pass3Strict} | ${pass5Strict} | ${pass10Strict} | ${row.avg_keys_found_pct.toFixed(2)} | ${row.avg_field_accuracy_pct.toFixed(2)} | ${row.avg_critical_accuracy_pct.toFixed(2)} | ${row.field_accuracy_variance_pct.toFixed(2)} | ${costPerDoc} | ${costPerSuccess} | ${row.avg_latency_ms.toFixed(1)} |`;
  });

  return [...head, ...body].join('\n');
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function hasValueKey(record) {
  return isRecord(record) && Object.prototype.hasOwnProperty.call(record, 'value');
}

function collectValueNodes(input, output = []) {
  if (!isRecord(input)) return output;
  for (const value of Object.values(input)) {
    if (hasValueKey(value)) {
      output.push({
        value: value.value,
        critical: Boolean(value.critical),
      });
      continue;
    }

    if (isRecord(value)) {
      collectValueNodes(value, output);
      continue;
    }

    if (Array.isArray(value)) {
      for (const item of value) {
        if (isRecord(item)) collectValueNodes(item, output);
      }
    }
  }
  return output;
}

function isLabeledValue(value) {
  if (typeof value === 'string') return value.trim().length > 0;
  if (Array.isArray(value)) return value.some((item) => String(item ?? '').trim().length > 0);
  if (typeof value === 'number' || typeof value === 'boolean') return true;
  return false;
}

function summarizeDataset(documents) {
  const documentsPerDomain = {};
  let totalKeys = 0;
  let labeledKeys = 0;
  let criticalKeys = 0;
  let labeledCriticalKeys = 0;

  for (const document of documents) {
    documentsPerDomain[document.domain] = (documentsPerDomain[document.domain] ?? 0) + 1;

    const valueNodes = collectValueNodes(document.ground_truth_raw, []);
    for (const node of valueNodes) {
      totalKeys += 1;
      if (node.critical) criticalKeys += 1;
      if (isLabeledValue(node.value)) {
        labeledKeys += 1;
        if (node.critical) labeledCriticalKeys += 1;
      }
    }
  }

  return {
    total_documents: documents.length,
    documents_per_domain: documentsPerDomain,
    total_keys: totalKeys,
    labeled_keys: labeledKeys,
    critical_keys: criticalKeys,
    labeled_critical_keys: labeledCriticalKeys,
  };
}

function timestampForFilename(date = new Date()) {
  return date.toISOString().replace(/[:.]/g, '-');
}

async function fileExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const runsPath = path.resolve(args.checkpointDir, 'runs.jsonl');
  const statePath = path.resolve(args.checkpointDir, 'state.json');
  const configPath = path.resolve(args.repoRoot, 'config/models.public.json');
  const manifestPath = path.resolve(args.repoRoot, 'dataset/manifest.json');

  if (!(await fileExists(runsPath))) {
    throw new Error(`Checkpoint runs log not found: ${runsPath}`);
  }

  const [runsRaw, stateRaw, configRaw, manifestRaw] = await Promise.all([
    fs.readFile(runsPath, 'utf8'),
    fileExists(statePath) ? fs.readFile(statePath, 'utf8') : Promise.resolve('{}'),
    fs.readFile(configPath, 'utf8'),
    fs.readFile(manifestPath, 'utf8'),
  ]);

  const state = stateRaw.trim() ? JSON.parse(stateRaw) : {};
  const options = state.options ?? {};
  const config = JSON.parse(configRaw);
  const manifest = JSON.parse(manifestRaw);

  const latestByTask = new Map();
  const lines = runsRaw.split(/\r?\n/).filter((line) => line.trim().length > 0);
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed?.task_key && parsed.metrics && parsed.debug) {
        latestByTask.set(parsed.task_key, parsed);
      }
    } catch {
      // Ignore malformed/partial lines.
    }
  }

  const runRecords = Array.from(latestByTask.values());
  if (runRecords.length === 0) {
    throw new Error('No valid records found in checkpoint log.');
  }

  const selectedDomains = Array.isArray(options.domains)
    ? options.domains.map((value) => normalizeDomain(value)).filter(Boolean)
    : [];
  const domainFilter = new Set(selectedDomains);
  const maxDocumentsPerDomain =
    Number.isFinite(options.max_documents_per_domain) && Number(options.max_documents_per_domain) > 0
      ? Number(options.max_documents_per_domain)
      : null;

  const selectedDocuments = [];
  for (const domain of manifest.domains ?? []) {
    const normalizedDomain = normalizeDomain(domain.id);
    if (domainFilter.size > 0 && !domainFilter.has(normalizedDomain)) continue;

    const docs = Array.isArray(domain.documents) ? domain.documents : [];
    const picked = maxDocumentsPerDomain === null ? docs : docs.slice(0, maxDocumentsPerDomain);

    for (const doc of picked) {
      const gtAbs = path.resolve(args.repoRoot, doc.ground_truth);
      const gtRaw = JSON.parse(await fs.readFile(gtAbs, 'utf8'));
      selectedDocuments.push({
        document_id: doc.document_id,
        domain: doc.domain,
        source_pdf: doc.source_pdf,
        ground_truth_path: doc.ground_truth,
        ground_truth_raw: gtRaw,
      });
    }
  }

  const dataset = summarizeDataset(selectedDocuments);
  const warnings = [];
  if (dataset.labeled_keys === 0) {
    warnings.push('Ground truth is not labeled yet. Fill expected values before trusting rankings.');
  }
  if (dataset.labeled_keys < dataset.total_keys) {
    warnings.push(`Only ${dataset.labeled_keys}/${dataset.total_keys} keys are currently labeled.`);
  }
  for (const [domain, count] of Object.entries(dataset.documents_per_domain)) {
    if (count < 10) warnings.push(`Domain "${domain}" has ${count} documents; target is >= 10 for launch quality.`);
  }

  const runsPerModel = Math.max(1, Number(options.runs_per_model || config.default_runs_per_model || 1));
  const requestedByModel = new Map();
  const requestedByDomainModel = new Map();

  for (const model of config.models ?? []) {
    const modelKey = idForModel(model.provider, model.model_id);
    requestedByModel.set(modelKey, (selectedDocuments.length || 0) * runsPerModel);
    for (const document of selectedDocuments) {
      const key = `${document.domain}::${modelKey}`;
      requestedByDomainModel.set(key, (requestedByDomainModel.get(key) ?? 0) + runsPerModel);
    }
  }

  const runResults = runRecords.map((record) => record.metrics);
  const debugRuns = runRecords.map((record) => record.debug);

  const leaderboard = aggregateRows(runResults, requestedByModel);
  const byDomain = Object.keys(dataset.documents_per_domain)
    .sort((a, b) => a.localeCompare(b))
    .map((domain) => {
      const domainRuns = runResults.filter((run) => run.domain === domain);
      const domainRequested = new Map();
      for (const [key, value] of requestedByDomainModel.entries()) {
        const [candidateDomain, modelKey] = key.split('::');
        if (candidateDomain === domain) {
          domainRequested.set(modelKey, value);
        }
      }
      return {
        domain,
        rows: aggregateRows(domainRuns, domainRequested),
      };
    });

  const cacheSummary = buildCacheSummary(runResults);
  const generatedAt = new Date().toISOString();

  const snapshot = {
    generated_at: generatedAt,
    benchmark_id: `ocr-benchmark-rebuild-${timestampForFilename(new Date(generatedAt))}`,
    benchmark_description: config.description,
    options: {
      runs_per_model: runsPerModel,
      max_parallel_requests:
        Number.isFinite(options.max_parallel_requests) && Number(options.max_parallel_requests) > 0
          ? Number(options.max_parallel_requests)
          : Number(config.max_parallel_requests || 1),
      provider_parallel: options.provider_parallel === true,
      selected_domains: selectedDomains,
      max_documents_per_domain: maxDocumentsPerDomain,
    },
    dataset,
    leaderboard,
    by_domain: byDomain,
    run_count: runResults.length,
    markdown_table: buildMarkdownTable(leaderboard),
    warnings,
    cache_summary: cacheSummary,
    debug: {
      documents: selectedDocuments.map((document) => ({
        document_id: document.document_id,
        domain: document.domain,
        source_pdf: document.source_pdf,
        ground_truth_path: document.ground_truth_path,
        ground_truth: document.ground_truth_raw,
      })),
      runs: debugRuns,
    },
  };

  const timestamp = timestampForFilename();
  await fs.mkdir(args.outputDir, { recursive: true });

  const snapshotPath = path.resolve(args.outputDir, `snapshot-${timestamp}.json`);
  const snapshotDebugPath = path.resolve(args.outputDir, `snapshot-${timestamp}.debug.json`);
  const latestJsonPath = path.resolve(args.outputDir, 'latest.json');
  const latestDebugPath = path.resolve(args.outputDir, 'latest.debug.json');
  const latestMarkdownPath = path.resolve(args.outputDir, 'latest.md');

  const debugPayload = snapshot.debug ?? { documents: [], runs: [] };
  const publicPayload = { ...snapshot };
  delete publicPayload.debug;

  await Promise.all([
    fs.writeFile(snapshotPath, `${JSON.stringify(publicPayload, null, 2)}\n`, 'utf8'),
    fs.writeFile(snapshotDebugPath, `${JSON.stringify(debugPayload, null, 2)}\n`, 'utf8'),
    fs.writeFile(latestJsonPath, `${JSON.stringify(publicPayload, null, 2)}\n`, 'utf8'),
    fs.writeFile(latestDebugPath, `${JSON.stringify(debugPayload, null, 2)}\n`, 'utf8'),
    fs.writeFile(latestMarkdownPath, `${String(publicPayload.markdown_table || '')}\n`, 'utf8'),
  ]);

  console.log(`Rebuilt snapshot from checkpoint records: ${runRecords.length}`);
  console.log(`Checkpoint dir: ${args.checkpointDir}`);
  console.log(`Output dir: ${args.outputDir}`);
  console.log(`Latest artifact: ${latestJsonPath}`);
  console.log(`Latest debug artifact: ${latestDebugPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
