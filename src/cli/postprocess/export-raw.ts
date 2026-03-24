import path from 'node:path';
import type { LegacyCheckpointRecord, RawNormalizedRecord } from '../../postprocess/types';
import { fromCheckpointRecord, fromDebugRun, toTaskKeyFromDebug } from '../../postprocess/raw-contract';
import { readJsonFile, readJsonLinesFile, timestampForFilename, writeJsonFile, writeJsonLinesFile } from '../../postprocess/io';
import { PATHS } from '../../config/paths';

type CliArgs = {
  checkpointDir?: string;
  debugFile?: string;
  inputJsonl?: string;
  outputJsonl: string;
  outputSummary: string;
};

function printHelp(): void {
  console.log(`Usage:
  npm run postprocess:raw -- [options]

Options:
  --checkpoint-dir=<path>   Checkpoint directory (uses runs.jsonl or raw.jsonl)
  --debug-file=<path>       Build raw records from debug snapshot file
  --input-jsonl=<path>      Re-read canonical raw JSONL and de-duplicate by task_key
  --output-jsonl=<path>     Output raw JSONL (default: artifacts/postprocess/raw.jsonl)
  --output-summary=<path>   Output summary JSON (default: artifacts/postprocess/raw.summary.json)
  -h, --help                Show this help

Examples:
  npm run postprocess:raw
  npm run postprocess:raw -- --checkpoint-dir=artifacts/checkpoints/full-2026-03-14
  npm run postprocess:raw -- --debug-file=artifacts/latest.debug.json
`);
}

function wantsHelp(argv: string[]): boolean {
  return argv.includes('--help') || argv.includes('-h');
}

function parseArgs(argv: string[]): CliArgs {
  const out: CliArgs = {
    outputJsonl: PATHS.postprocess.rawJsonl,
    outputSummary: PATHS.postprocess.rawSummary,
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
    if (arg.startsWith('--input-jsonl=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.inputJsonl = path.resolve(process.cwd(), value.trim());
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
    out.checkpointDir = PATHS.checkpoint.root;
  }

  return out;
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
    const taskKey = toTaskKeyFromDebug(run);
    latestByTask.set(taskKey, run);
  }

  return Array.from(latestByTask.values()).map(fromDebugRun);
}

async function loadFromCanonicalRaw(inputJsonl: string): Promise<RawNormalizedRecord[]> {
  const lines = await readJsonLinesFile<RawNormalizedRecord>(inputJsonl);
  const latestByTask = new Map<string, RawNormalizedRecord>();
  for (const line of lines) {
    if (line?.task_key && line.model && line.document && line.runtime && line.payload && line.legacy_metrics) {
      latestByTask.set(line.task_key, line);
    }
  }
  return Array.from(latestByTask.values());
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (wantsHelp(argv)) {
    printHelp();
    return;
  }
  const args = parseArgs(argv);

  const checkpointRawJsonl = args.checkpointDir ? path.resolve(args.checkpointDir, 'raw.jsonl') : undefined;

  const rawRecords = args.inputJsonl
    ? await loadFromCanonicalRaw(args.inputJsonl)
    : args.debugFile
      ? await loadFromDebug(args.debugFile)
      : checkpointRawJsonl
        ? await loadFromCanonicalRaw(checkpointRawJsonl).catch(async () => loadFromCheckpoint(args.checkpointDir!))
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
    source: args.inputJsonl
      ? { input_jsonl: args.inputJsonl }
      : args.debugFile
        ? { debug_file: args.debugFile }
        : checkpointRawJsonl
          ? { checkpoint_raw_jsonl: checkpointRawJsonl, checkpoint_dir: args.checkpointDir }
          : { checkpoint_dir: args.checkpointDir },
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
    build_id: `raw-${timestampForFilename()}`,
  };

  await writeJsonFile(args.outputSummary, summary);

  console.log(`Raw records: ${rawRecords.length}`);
  console.log(`Errors: ${errored}`);
  console.log(`JSONL: ${args.outputJsonl}`);
  console.log(`Summary: ${args.outputSummary}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
