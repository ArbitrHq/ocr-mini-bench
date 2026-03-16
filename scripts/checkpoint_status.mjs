import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..');

function parseArgs(argv) {
  const out = {
    checkpointDir: path.resolve(REPO_ROOT, 'artifacts/checkpoints'),
    json: false,
  };

  for (const arg of argv) {
    if (arg.startsWith('--checkpoint-dir=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.checkpointDir = path.resolve(process.cwd(), value.trim());
      continue;
    }
    if (arg === '--json') {
      out.json = true;
    }
  }

  return out;
}

function toModelKey(metrics) {
  const modelKey = metrics?.model_key;
  if (typeof modelKey === 'string' && modelKey.trim()) return modelKey;
  const provider = metrics?.provider ?? 'unknown';
  const modelId = metrics?.model_id ?? 'unknown';
  return `${provider}:${modelId}`;
}

function incCounter(map, key) {
  map.set(key, (map.get(key) ?? 0) + 1);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const runsPath = path.resolve(args.checkpointDir, 'runs.jsonl');
  const statePath = path.resolve(args.checkpointDir, 'state.json');

  try {
    await fs.access(runsPath);
  } catch {
    if (args.json) {
      console.log(
        JSON.stringify(
          {
            checkpoint_dir: args.checkpointDir,
            runs_log: runsPath,
            state_file: statePath,
            exists: false,
            message: 'No checkpoint log found. Run the benchmark once first.',
          },
          null,
          2
        )
      );
      return;
    }
    console.log(`Checkpoint dir: ${args.checkpointDir}`);
    console.log('No checkpoint log found. Run the benchmark once first.');
    return;
  }

  const runsRaw = await fs.readFile(runsPath, 'utf8');
  const lines = runsRaw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const latestByTask = new Map();

  for (const line of lines) {
    let parsed;
    try {
      parsed = JSON.parse(line);
    } catch {
      continue;
    }

    const taskKey = parsed?.task_key;
    const metrics = parsed?.metrics;
    if (!taskKey || !metrics) continue;
    latestByTask.set(taskKey, parsed);
  }

  let state = null;
  try {
    const raw = await fs.readFile(statePath, 'utf8');
    if (raw.trim()) {
      state = JSON.parse(raw);
    }
  } catch {
    // optional
  }

  let failed = 0;
  let successful = 0;
  const failedByModel = new Map();
  const successByModel = new Map();
  const failedErrorTypes = new Map();

  for (const record of latestByTask.values()) {
    const metrics = record.metrics;
    const modelKey = toModelKey(metrics);
    if (metrics?.error) {
      failed += 1;
      incCounter(failedByModel, modelKey);
      incCounter(failedErrorTypes, String(metrics.error));
    } else {
      successful += 1;
      incCounter(successByModel, modelKey);
    }
  }

  const summary = {
    checkpoint_dir: args.checkpointDir,
    runs_log: runsPath,
    state_file: statePath,
    raw_lines: lines.length,
    latest_task_records: latestByTask.size,
    successful_records: successful,
    failed_records: failed,
    failure_rate_pct: latestByTask.size > 0 ? (failed / latestByTask.size) * 100 : 0,
    state,
    by_model: Array.from(new Set([...successByModel.keys(), ...failedByModel.keys()]))
      .sort((a, b) => a.localeCompare(b))
      .map((model) => ({
        model_key: model,
        successful: successByModel.get(model) ?? 0,
        failed: failedByModel.get(model) ?? 0,
      })),
    top_errors: Array.from(failedErrorTypes.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([error, count]) => ({ count, error })),
  };

  if (args.json) {
    console.log(JSON.stringify(summary, null, 2));
    return;
  }

  console.log(`Checkpoint dir: ${summary.checkpoint_dir}`);
  console.log(`Records (latest by task): ${summary.latest_task_records}`);
  console.log(`Successful: ${summary.successful_records}`);
  console.log(`Failed: ${summary.failed_records} (${summary.failure_rate_pct.toFixed(2)}%)`);

  if (summary.state) {
    console.log(`Mode: ${summary.state.mode ?? 'unknown'} | Final: ${summary.state.final ?? false}`);
    console.log(`Updated at: ${summary.state.updated_at ?? 'n/a'}`);
  }

  console.log('\nBy model:');
  for (const row of summary.by_model) {
    console.log(`- ${row.model_key}: ok=${row.successful}, fail=${row.failed}`);
  }

  if (summary.top_errors.length > 0) {
    console.log('\nTop errors:');
    for (const row of summary.top_errors) {
      console.log(`- (${row.count}) ${row.error}`);
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
