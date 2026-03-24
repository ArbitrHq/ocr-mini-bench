import 'dotenv/config';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { runOCRLeaderboardBenchmark } from '../benchmark/run';
import { loadBenchmarkConfig, loadPreparedDocuments } from '../benchmark/dataset';
import type { BenchmarkDebugRun, BenchmarkRunOptions, SingleRunMetrics } from '../benchmark/types';
import type { RawNormalizedRecord } from '../postprocess/types';
import { fromCheckpointRecord } from '../postprocess/raw-contract';
import { writeJsonLinesFile } from '../postprocess/io';
import { PATHS } from '../config/paths';

type CliArgs = {
  runsPerModel?: number;
  maxParallelRequests?: number;
  maxDocumentsPerDomain?: number;
  domains?: string[];
  models?: string[];
  providerParallel?: boolean;
  outputDir: string;
  checkpointDir: string;
  resume: boolean;
  retryFailed: boolean;
};

function printHelp(): void {
  console.log(`Usage:
  npm run benchmark:run -- [options]

Options:
  --runs=<n>                Runs per model/document
  --parallel=<n>            Max parallel requests per provider lane
  --provider-parallel       Enable provider-lane parallel mode
  --no-provider-parallel    Disable provider-lane parallel mode
  --docs-per-domain=<n>     Limit documents per selected domain
  --domains=<csv>           Domain filter (e.g. invoices,receipts,logistics)
  --models=<csv>            Model filter by model_id or model_label
  --output-dir=<path>       Snapshot + postprocess output directory (default: artifacts)
  --checkpoint-dir=<path>   Checkpoint directory (default: artifacts/checkpoints)
  --resume                  Resume unfinished tasks from checkpoint
  --retry-failed            Run only failed tasks from checkpoint
  -h, --help                Show this help

Examples:
  npm run benchmark:run -- --runs=2 --docs-per-domain=3 --models="gemini-3.1-flash-lite-preview"
  npm run benchmark:run -- --resume --checkpoint-dir=artifacts/checkpoints/full-2026-03-14
  npm run benchmark:run -- --retry-failed --checkpoint-dir=artifacts/checkpoints/full-2026-03-14

Notes:
  --models matches model_id or model_label from config/models.public.json.
  Do not prefix provider in --models (use gemini-3.1-flash-lite-preview, not google:...).
`);
}

function wantsHelp(argv: string[]): boolean {
  return argv.includes('--help') || argv.includes('-h');
}

const KNOWN_FLAGS = new Set([
  '--runs',
  '--parallel',
  '--provider-parallel',
  '--no-provider-parallel',
  '--docs-per-domain',
  '--domains',
  '--models',
  '--output-dir',
  '--checkpoint-dir',
  '--resume',
  '--retry-failed',
  '--help',
  '-h',
]);

function isKnownFlag(arg: string): boolean {
  if (KNOWN_FLAGS.has(arg)) return true;
  const flagName = arg.split('=')[0];
  return KNOWN_FLAGS.has(flagName);
}

function parseArgs(argv: string[]): CliArgs {
  const out: CliArgs = {
    outputDir: PATHS.artifacts.root,
    checkpointDir: PATHS.artifacts.checkpoints,
    resume: false,
    retryFailed: false,
  };

  for (const arg of argv) {
    if (arg.startsWith('--runs=')) {
      const value = Number(arg.split('=')[1]);
      if (Number.isFinite(value) && value > 0) out.runsPerModel = value;
      continue;
    }
    if (arg.startsWith('--parallel=')) {
      const value = Number(arg.split('=')[1]);
      if (Number.isFinite(value) && value > 0) out.maxParallelRequests = value;
      continue;
    }
    if (arg === '--provider-parallel') {
      out.providerParallel = true;
      continue;
    }
    if (arg === '--no-provider-parallel') {
      out.providerParallel = false;
      continue;
    }
    if (arg.startsWith('--docs-per-domain=')) {
      const value = Number(arg.split('=')[1]);
      if (Number.isFinite(value) && value > 0) out.maxDocumentsPerDomain = value;
      continue;
    }
    if (arg.startsWith('--domains=')) {
      const value = arg.split('=')[1] || '';
      const domains = value
        .split(',')
        .map((item) => item.trim().toLowerCase())
        .filter(Boolean);
      if (domains.length > 0) out.domains = domains;
      continue;
    }
    if (arg.startsWith('--models=')) {
      const value = arg.split('=')[1] || '';
      const models = value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
      if (models.length > 0) out.models = models;
      continue;
    }
    if (arg.startsWith('--output-dir=')) {
      const value = arg.split('=')[1] || '';
      if (value.trim()) out.outputDir = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg.startsWith('--checkpoint-dir=')) {
      const value = arg.split('=')[1] || '';
      if (value.trim()) out.checkpointDir = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg === '--resume') {
      out.resume = true;
      continue;
    }
    if (arg === '--retry-failed') {
      out.retryFailed = true;
      continue;
    }
    // Warn about unknown flags
    if (arg.startsWith('-') && !isKnownFlag(arg)) {
      console.warn(`Warning: Unknown flag "${arg}". Run --help for available options.`);
    }
  }

  return out;
}

function timestampForFilename(date = new Date()): string {
  return date.toISOString().replace(/[:.]/g, '-');
}

type CheckpointLine = {
  task_key: string;
  completed_at: string;
  metrics: SingleRunMetrics;
  debug: BenchmarkDebugRun;
};

type CheckpointState = {
  updated_at: string;
  mode: 'fresh' | 'resume' | 'retry-failed';
  options: BenchmarkRunOptions;
  records_total: number;
  records_failed: number;
  records_successful: number;
  current_run_new_records: number;
  final: boolean;
  benchmark_fingerprint?: string;
};

async function fileExists(targetPath: string): Promise<boolean> {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function loadLatestCheckpointRecords(runsLogPath: string): Promise<Map<string, CheckpointLine>> {
  const latestByTask = new Map<string, CheckpointLine>();
  if (!(await fileExists(runsLogPath))) {
    return latestByTask;
  }
  const raw = await fs.readFile(runsLogPath, 'utf8');
  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line) as CheckpointLine;
      if (parsed?.task_key && parsed.metrics && parsed.debug) {
        latestByTask.set(parsed.task_key, parsed);
      }
    } catch {
      // Ignore malformed line and continue.
    }
  }
  return latestByTask;
}

async function loadCheckpointState(statePath: string): Promise<CheckpointState | null> {
  if (!(await fileExists(statePath))) return null;
  try {
    const raw = await fs.readFile(statePath, 'utf8');
    if (!raw.trim()) return null;
    return JSON.parse(raw) as CheckpointState;
  } catch {
    return null;
  }
}

function normalizeOptionsForCompare(options: BenchmarkRunOptions): BenchmarkRunOptions {
  return {
    runs_per_model: options.runs_per_model,
    max_parallel_requests: options.max_parallel_requests,
    max_documents_per_domain: options.max_documents_per_domain,
    provider_parallel: options.provider_parallel,
    domains: options.domains ? [...options.domains].sort((a, b) => a.localeCompare(b)) : undefined,
    models: options.models ? [...options.models].sort((a, b) => a.localeCompare(b)) : undefined,
  };
}

async function computeBenchmarkFingerprint(options: BenchmarkRunOptions): Promise<string> {
  const hash = createHash('sha256');
  hash.update(JSON.stringify(normalizeOptionsForCompare(options)));

  const files = [
    PATHS.config.models,
    PATHS.dataset.manifest,
    PATHS.prompts.system,
    PATHS.prompts.user,
  ];

  for (const filePath of files) {
    const content = await fs.readFile(filePath, 'utf8');
    hash.update('\n@@');
    hash.update(path.basename(filePath));
    hash.update('@@\n');
    hash.update(content);
  }

  return hash.digest('hex');
}

function round(value: number, decimals = 6): number {
  const precision = 10 ** decimals;
  return Math.round(value * precision) / precision;
}

async function estimatePlannedTaskCount(options: BenchmarkRunOptions): Promise<number | null> {
  try {
    const config = await loadBenchmarkConfig();
    const runsPerModel = Math.max(1, options.runs_per_model ?? config.default_runs_per_model);
    const selectedDomains = options.domains?.map((value) => value.trim().toLowerCase()).filter(Boolean) ?? [];
    const maxDocumentsPerDomain =
      typeof options.max_documents_per_domain === 'number' && options.max_documents_per_domain > 0
        ? options.max_documents_per_domain
        : undefined;

    const documents = await loadPreparedDocuments({
      domains: selectedDomains,
      maxDocumentsPerDomain,
    });

    const selectedModelFilters = options.models?.map((value) => value.trim().toLowerCase()).filter(Boolean) ?? [];
    const selectedModels =
      selectedModelFilters.length === 0
        ? config.models
        : config.models.filter((model) => {
            const modelId = model.model_id.trim().toLowerCase();
            const modelLabel = model.model_label.trim().toLowerCase();
            return selectedModelFilters.some((needle) => needle === modelId || needle === modelLabel);
          });

    return selectedModels.length * documents.length * runsPerModel;
  } catch {
    return null;
  }
}

function summarizeCheckpoint(records: Map<string, CheckpointLine>): {
  total: number;
  failed: number;
  costUsd: number;
} {
  let failed = 0;
  let costUsd = 0;
  for (const record of records.values()) {
    if (record.metrics.error) failed += 1;
    const runCost = Number(record.metrics.total_cost_usd);
    if (Number.isFinite(runCost) && runCost > 0) {
      costUsd += runCost;
    }
  }
  return { total: records.size, failed, costUsd: round(costUsd, 6) };
}

async function writeCheckpointState(params: {
  statePath: string;
  mode: 'fresh' | 'resume' | 'retry-failed';
  options: BenchmarkRunOptions;
  records: Map<string, CheckpointLine>;
  currentRunNewRecords: number;
  currentRunCostUsd: number;
  currentRunTargetTasks: number | null;
  benchmarkFingerprint: string;
  final?: boolean;
}): Promise<void> {
  const summary = summarizeCheckpoint(params.records);
  const completedThisRun = params.currentRunNewRecords;
  const targetThisRun = params.currentRunTargetTasks;
  const avgCostThisRun = completedThisRun > 0 ? params.currentRunCostUsd / completedThisRun : 0;
  const estimatedFinalCostThisRun =
    typeof targetThisRun === 'number' && targetThisRun > 0 && completedThisRun > 0
      ? avgCostThisRun * targetThisRun
      : null;

  const statePayload = {
    updated_at: new Date().toISOString(),
    mode: params.mode,
    options: params.options,
    records_total: summary.total,
    records_failed: summary.failed,
    records_successful: summary.total - summary.failed,
    records_total_cost_usd: summary.costUsd,
    current_run_new_records: params.currentRunNewRecords,
    current_run_target_tasks: targetThisRun,
    current_run_remaining_tasks:
      typeof targetThisRun === 'number' ? Math.max(0, targetThisRun - completedThisRun) : null,
    current_run_cost_usd: round(params.currentRunCostUsd, 6),
    current_run_avg_cost_usd: round(avgCostThisRun, 6),
    current_run_estimated_final_cost_usd:
      estimatedFinalCostThisRun === null ? null : round(estimatedFinalCostThisRun, 6),
    final: Boolean(params.final),
    benchmark_fingerprint: params.benchmarkFingerprint,
  };
  await fs.writeFile(params.statePath, `${JSON.stringify(statePayload, null, 2)}\n`, 'utf8');
}

async function run(): Promise<void> {
  const argv = process.argv.slice(2);
  if (wantsHelp(argv)) {
    printHelp();
    return;
  }
  const args = parseArgs(argv);
  const mode: 'fresh' | 'resume' | 'retry-failed' = args.retryFailed
    ? 'retry-failed'
    : args.resume
      ? 'resume'
      : 'fresh';
  const options: BenchmarkRunOptions = {
    runs_per_model: args.runsPerModel,
    max_parallel_requests: args.maxParallelRequests,
    max_documents_per_domain: args.maxDocumentsPerDomain,
    domains: args.domains,
    models: args.models,
    provider_parallel: args.providerParallel,
  };
  const plannedTotalTasks = await estimatePlannedTaskCount(options);
  const benchmarkFingerprint = await computeBenchmarkFingerprint(options);

  await fs.mkdir(args.outputDir, { recursive: true });
  await fs.mkdir(args.checkpointDir, { recursive: true });
  const runsLogPath = path.resolve(args.checkpointDir, 'runs.jsonl');
  const rawCheckpointLogPath = path.resolve(args.checkpointDir, 'raw.runs.jsonl');
  const rawCheckpointLatestPath = path.resolve(args.checkpointDir, 'raw.jsonl');
  const postprocessDir = path.resolve(args.outputDir, 'postprocess');
  const rawOutputPath = path.resolve(postprocessDir, 'raw.jsonl');
  const statePath = path.resolve(args.checkpointDir, 'state.json');

  if (mode === 'fresh') {
    await Promise.all([
      fs.writeFile(runsLogPath, '', 'utf8'),
      fs.writeFile(rawCheckpointLogPath, '', 'utf8'),
      fs.writeFile(statePath, '', 'utf8'),
    ]);
  } else if (!(await fileExists(runsLogPath))) {
    throw new Error(`Checkpoint log not found at ${runsLogPath}. Run once without --resume/--retry-failed first.`);
  }

  const latestRecords = mode === 'fresh' ? new Map<string, CheckpointLine>() : await loadLatestCheckpointRecords(runsLogPath);
  const existingState = mode === 'fresh' ? null : await loadCheckpointState(statePath);
  if (mode !== 'fresh' && existingState) {
    const priorOptions = normalizeOptionsForCompare(existingState.options ?? {});
    const currentOptions = normalizeOptionsForCompare(options);
    if (JSON.stringify(priorOptions) !== JSON.stringify(currentOptions)) {
      throw new Error(
        `Checkpoint options mismatch. Previous options differ from current CLI options. Use matching options or a new --checkpoint-dir.`
      );
    }
    if (existingState.benchmark_fingerprint && existingState.benchmark_fingerprint !== benchmarkFingerprint) {
      throw new Error(
        `Checkpoint fingerprint mismatch (models/manifest/prompts changed). Use a new --checkpoint-dir for this run definition.`
      );
    }
  }
  const checkpointSummary = summarizeCheckpoint(latestRecords);
  if (mode !== 'fresh') {
    console.log(
      `Loaded checkpoint: ${checkpointSummary.total} records (${checkpointSummary.failed} failed) from ${runsLogPath}`
    );
  }

  const failedTaskKeys = new Set<string>();
  for (const [taskKey, record] of latestRecords.entries()) {
    if (record.metrics.error) failedTaskKeys.add(taskKey);
  }

  let initialMetrics: SingleRunMetrics[] = [];
  let initialDebugRuns: BenchmarkDebugRun[] = [];
  let skipTaskKeys: Set<string> | undefined;
  let onlyTaskKeys: Set<string> | undefined;

  if (mode === 'resume') {
    initialMetrics = Array.from(latestRecords.values()).map((record) => record.metrics);
    initialDebugRuns = Array.from(latestRecords.values()).map((record) => record.debug);
    skipTaskKeys = new Set(latestRecords.keys());
  } else if (mode === 'retry-failed') {
    initialMetrics = Array.from(latestRecords.values())
      .filter((record) => !failedTaskKeys.has(record.task_key))
      .map((record) => record.metrics);
    initialDebugRuns = Array.from(latestRecords.values())
      .filter((record) => !failedTaskKeys.has(record.task_key))
      .map((record) => record.debug);
    onlyTaskKeys = failedTaskKeys;
    console.log(`Retry mode: scheduling ${failedTaskKeys.size} previously failed tasks.`);
  }

  let currentRunTargetTasks: number | null = null;
  if (mode === 'fresh') {
    currentRunTargetTasks = plannedTotalTasks;
  } else if (mode === 'resume') {
    currentRunTargetTasks =
      typeof plannedTotalTasks === 'number' ? Math.max(0, plannedTotalTasks - latestRecords.size) : null;
  } else if (mode === 'retry-failed') {
    currentRunTargetTasks = failedTaskKeys.size;
  }

  let newRecordsThisRun = 0;
  let currentRunCostUsd = 0;
  let checkpointWriteChain = Promise.resolve();
  const queueCheckpointWrite = (line: CheckpointLine) => {
    checkpointWriteChain = checkpointWriteChain.then(async () => {
      const rawLine = fromCheckpointRecord(line);
      await fs.appendFile(runsLogPath, `${JSON.stringify(line)}\n`, 'utf8');
      await fs.appendFile(rawCheckpointLogPath, `${JSON.stringify(rawLine)}\n`, 'utf8');
      latestRecords.set(line.task_key, line);
      newRecordsThisRun += 1;
      const runCost = Number(line.metrics.total_cost_usd);
      if (Number.isFinite(runCost) && runCost > 0) {
        currentRunCostUsd += runCost;
      }
      await writeCheckpointState({
        statePath,
        mode,
        options,
        records: latestRecords,
        currentRunNewRecords: newRecordsThisRun,
        currentRunCostUsd,
        currentRunTargetTasks,
        benchmarkFingerprint,
      });
    });
    return checkpointWriteChain;
  };

  await writeCheckpointState({
    statePath,
    mode,
    options,
    records: latestRecords,
    currentRunNewRecords: 0,
    currentRunCostUsd: 0,
    currentRunTargetTasks,
    benchmarkFingerprint,
  });

  console.log('Running OCR benchmark (standalone repository)...');
  if (options.provider_parallel) {
    const laneInfo =
      typeof options.max_parallel_requests === 'number'
        ? `provider lanes capped at ${options.max_parallel_requests}`
        : 'one lane per provider';
    console.log(`Provider-parallel mode enabled (${laneInfo}).`);
  }
  const snapshot = await runOCRLeaderboardBenchmark(options, {
    initial_runs: initialMetrics,
    initial_debug_runs: initialDebugRuns,
    skip_task_keys: skipTaskKeys,
    only_task_keys: onlyTaskKeys,
    on_task_complete: async (event) => {
      await queueCheckpointWrite({
        task_key: event.task_key,
        completed_at: new Date().toISOString(),
        metrics: event.metrics,
        debug: event.debug,
      });
    },
  });
  await checkpointWriteChain;

  const finalRawRecords: RawNormalizedRecord[] = Array.from(latestRecords.values())
    .map((record) => fromCheckpointRecord(record))
    .sort((a, b) => a.task_key.localeCompare(b.task_key));
  await Promise.all([
    writeJsonLinesFile(rawCheckpointLatestPath, finalRawRecords),
    writeJsonLinesFile(rawOutputPath, finalRawRecords),
  ]);

  const timestamp = timestampForFilename();
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

  console.log(`Snapshot written: ${snapshotPath}`);
  console.log(`Debug snapshot written: ${snapshotDebugPath}`);
  console.log(`Latest artifact written: ${latestJsonPath}`);
  console.log(`Latest debug artifact written: ${latestDebugPath}`);
  console.log(`Markdown table written: ${latestMarkdownPath}`);
  console.log(`Rows: ${Array.isArray(publicPayload.leaderboard) ? publicPayload.leaderboard.length : 0}`);
  console.log(`Runs: ${snapshot.run_count}`);
  console.log(`Checkpoint log: ${runsLogPath}`);
  console.log(`Checkpoint raw log: ${rawCheckpointLogPath}`);
  console.log(`Checkpoint canonical raw: ${rawCheckpointLatestPath}`);
  console.log(`Canonical raw output: ${rawOutputPath}`);
  console.log(`Checkpoint state: ${statePath}`);
  console.log(`New checkpoint records this run: ${newRecordsThisRun}`);
  if (Array.isArray(snapshot.cache_summary) && snapshot.cache_summary.length > 0) {
    console.log('Cache summary by model:');
    for (const row of snapshot.cache_summary) {
      console.log(
        `- ${row.model_label}: hit ${row.cache_hits}/${row.runs} (${row.cache_hit_rate_pct.toFixed(1)}%) | cached_in_avg=${row.cached_input_tokens_avg.toFixed(1)} | cache_write_avg=${row.cache_write_tokens_avg.toFixed(1)}`
      );
    }
  }

  await writeCheckpointState({
    statePath,
    mode,
    options,
    records: latestRecords,
    currentRunNewRecords: newRecordsThisRun,
    currentRunCostUsd,
    currentRunTargetTasks,
    benchmarkFingerprint,
    final: true,
  });
}

run().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
