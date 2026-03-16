import path from 'node:path';
import type { LegacyCheckpointRecord, RawNormalizedRecord } from '../../postprocess/types';
import { readJsonFile, readJsonLinesFile, timestampForFilename, writeJsonFile, writeJsonLinesFile } from '../../postprocess/io';

type CliArgs = {
  checkpointDir?: string;
  debugFile?: string;
  outputJsonl: string;
  outputSummary: string;
};

function parseArgs(argv: string[]): CliArgs {
  const defaultOutputDir = path.resolve(process.cwd(), 'artifacts/postprocess');

  const out: CliArgs = {
    outputJsonl: path.resolve(defaultOutputDir, 'raw.normalized.jsonl'),
    outputSummary: path.resolve(defaultOutputDir, 'raw.normalized.summary.json'),
  };

  for (const arg of argv) {
    if (arg.startsWith('--checkpoint-dir=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.checkpointDir = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg.startsWith('--debug-file=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.debugFile = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg.startsWith('--output-jsonl=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.outputJsonl = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg.startsWith('--output-summary=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.outputSummary = path.resolve(process.cwd(), value.trim());
    }
  }

  if (!out.checkpointDir && !out.debugFile) {
    out.checkpointDir = path.resolve(process.cwd(), 'artifacts/checkpoints');
  }

  return out;
}

function fromCheckpointRecord(record: LegacyCheckpointRecord): RawNormalizedRecord {
  const metrics = record.metrics;
  const debug = record.debug;

  return {
    schema_version: '1.0',
    task_key: record.task_key,
    completed_at: typeof record.completed_at === 'string' ? record.completed_at : null,
    model: {
      model_key: metrics.model_key,
      provider: metrics.provider,
      model_id: metrics.model_id,
      model_label: metrics.model_label,
      tier: metrics.tier,
    },
    document: {
      domain: metrics.domain,
      document_id: metrics.document_id,
      run_number: metrics.run_number,
    },
    runtime: {
      latency_ms: metrics.latency_ms,
      input_tokens: metrics.input_tokens,
      output_tokens: metrics.output_tokens,
      total_cost_usd: metrics.total_cost_usd,
      cache_hit: metrics.cache_hit,
      cached_input_tokens: metrics.cached_input_tokens,
      cache_write_tokens: metrics.cache_write_tokens,
      error: metrics.error,
    },
    payload: {
      system_prompt_used: debug.system_prompt_used ?? '',
      user_prompt_used: debug.user_prompt_used ?? '',
      raw_output: debug.raw_output ?? '',
      parsed_output: debug.parsed_output ?? null,
      extracted_pairs: Array.isArray(debug.extracted_pairs) ? debug.extracted_pairs : [],
    },
    legacy_metrics: {
      success: metrics.success,
      field_total: metrics.field_total,
      field_correct: metrics.field_correct,
      critical_total: metrics.critical_total,
      critical_correct: metrics.critical_correct,
      field_accuracy_pct: metrics.field_accuracy_pct,
      critical_accuracy_pct: metrics.critical_accuracy_pct,
      found_key_count: metrics.found_key_count,
      requested_key_count: metrics.requested_key_count,
    },
  };
}

function fromDebugRun(run: LegacyCheckpointRecord['debug']): RawNormalizedRecord {
  const taskKey =
    run.task_key ??
    `${run.model_key}::${run.domain}::${run.document_id}::run${run.run_number}`;

  return {
    schema_version: '1.0',
    task_key: taskKey,
    completed_at: null,
    model: {
      model_key: run.model_key,
      provider: run.provider,
      model_id: run.model_id,
      model_label: run.model_label,
      tier: run.tier,
    },
    document: {
      domain: run.domain,
      document_id: run.document_id,
      run_number: run.run_number,
    },
    runtime: {
      latency_ms: run.latency_ms,
      input_tokens: typeof run.input_tokens === 'number' ? run.input_tokens : 0,
      output_tokens: typeof run.output_tokens === 'number' ? run.output_tokens : 0,
      total_cost_usd: run.total_cost_usd,
      cache_hit: run.cache_hit,
      cached_input_tokens: run.cached_input_tokens,
      cache_write_tokens: run.cache_write_tokens,
      error: run.error,
    },
    payload: {
      system_prompt_used: run.system_prompt_used ?? '',
      user_prompt_used: run.user_prompt_used ?? '',
      raw_output: run.raw_output ?? '',
      parsed_output: run.parsed_output ?? null,
      extracted_pairs: Array.isArray(run.extracted_pairs) ? run.extracted_pairs : [],
    },
    legacy_metrics: {
      success: run.success,
      field_total: run.field_total,
      field_correct: run.field_correct,
      critical_total: run.critical_total,
      critical_correct: run.critical_correct,
      field_accuracy_pct: run.field_accuracy_pct,
      critical_accuracy_pct: run.critical_accuracy_pct,
      found_key_count: run.found_key_count,
      requested_key_count: run.requested_key_count,
    },
  };
}

async function loadFromCheckpoint(checkpointDir: string): Promise<RawNormalizedRecord[]> {
  const runsPath = path.resolve(checkpointDir, 'runs.jsonl');
  const lines = await readJsonLinesFile<LegacyCheckpointRecord>(runsPath);
  const latestByTask = new Map<string, LegacyCheckpointRecord>();

  for (const record of lines) {
    if (record?.task_key && record.metrics && record.debug) {
      latestByTask.set(record.task_key, record);
    }
  }

  return Array.from(latestByTask.values()).map(fromCheckpointRecord);
}

async function loadFromDebug(debugFile: string): Promise<RawNormalizedRecord[]> {
  const debug = await readJsonFile<{ runs?: LegacyCheckpointRecord['debug'][] }>(debugFile);
  const runs = Array.isArray(debug.runs) ? debug.runs : [];
  const latestByTask = new Map<string, LegacyCheckpointRecord['debug']>();

  for (const run of runs) {
    const taskKey =
      run.task_key ??
      `${run.model_key}::${run.domain}::${run.document_id}::run${run.run_number}`;
    latestByTask.set(taskKey, run);
  }

  return Array.from(latestByTask.values()).map(fromDebugRun);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  const rawRecords = args.debugFile
    ? await loadFromDebug(args.debugFile)
    : await loadFromCheckpoint(args.checkpointDir!);

  rawRecords.sort((a, b) => a.task_key.localeCompare(b.task_key));

  await writeJsonLinesFile(args.outputJsonl, rawRecords);

  const byModel = new Map<string, number>();
  const byDomain = new Map<string, number>();
  let errored = 0;

  for (const record of rawRecords) {
    byModel.set(record.model.model_label, (byModel.get(record.model.model_label) ?? 0) + 1);
    byDomain.set(record.document.domain, (byDomain.get(record.document.domain) ?? 0) + 1);
    if (record.runtime.error !== null) errored += 1;
  }

  const summary = {
    generated_at: new Date().toISOString(),
    source: args.debugFile ? { debug_file: args.debugFile } : { checkpoint_dir: args.checkpointDir },
    output_jsonl: args.outputJsonl,
    schema_version: '1.0',
    records: rawRecords.length,
    errored_records: errored,
    models: Array.from(byModel.entries())
      .map(([model_label, count]) => ({ model_label, count }))
      .sort((a, b) => a.model_label.localeCompare(b.model_label)),
    domains: Array.from(byDomain.entries())
      .map(([domain, count]) => ({ domain, count }))
      .sort((a, b) => a.domain.localeCompare(b.domain)),
    build_id: `raw-normalized-${timestampForFilename()}`,
  };

  await writeJsonFile(args.outputSummary, summary);

  console.log(`Raw normalized records: ${rawRecords.length}`);
  console.log(`Errors: ${errored}`);
  console.log(`JSONL: ${args.outputJsonl}`);
  console.log(`Summary: ${args.outputSummary}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
