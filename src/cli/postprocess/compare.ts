import path from 'node:path';
import { loadPreparedDocuments } from '../../benchmark/dataset';
import { scoreModelOutputDetailed } from '../../benchmark/scoring';
import type { ComparisonRecord, RawNormalizedRecord } from '../../postprocess/types';
import { fileExists, readJsonLinesFile, writeJsonFile, writeJsonLinesFile } from '../../postprocess/io';

type CliArgs = {
  rawJsonl: string;
  outputJsonl: string;
  outputSummary: string;
};

function printHelp(): void {
  console.log(`Usage:
  npm run postprocess:compare -- [options]

Options:
  --raw-jsonl=<path>        Canonical raw JSONL input (default: artifacts/postprocess/raw.jsonl)
  --output-jsonl=<path>     Comparison JSONL output (default: artifacts/postprocess/comparison.jsonl)
  --output-summary=<path>   Comparison summary output (default: artifacts/postprocess/comparison.summary.json)
  -h, --help                Show this help

Examples:
  npm run postprocess:compare
  npm run postprocess:compare -- --raw-jsonl=artifacts/smoke/postprocess/raw.jsonl
`);
}

function wantsHelp(argv: string[]): boolean {
  return argv.includes('--help') || argv.includes('-h');
}

function parseArgs(argv: string[]): CliArgs {
  const defaultDir = path.resolve(process.cwd(), 'artifacts/postprocess');
  const out: CliArgs = {
    rawJsonl: path.resolve(defaultDir, 'raw.jsonl'),
    outputJsonl: path.resolve(defaultDir, 'comparison.jsonl'),
    outputSummary: path.resolve(defaultDir, 'comparison.summary.json'),
  };

  for (const arg of argv) {
    if (arg.startsWith('--raw-jsonl=')) {
      const value = arg.split('=')[1] ?? '';
      if (value.trim()) out.rawJsonl = path.resolve(process.cwd(), value.trim());
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

  return out;
}

function pct(part: number, total: number): number {
  if (total <= 0) return 0;
  return (part / total) * 100;
}

function round(value: number, decimals = 4): number {
  const precision = 10 ** decimals;
  return Math.round(value * precision) / precision;
}

function toDocKey(domain: string, documentId: string): string {
  return `${domain.toLowerCase()}::${documentId}`;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (wantsHelp(argv)) {
    printHelp();
    return;
  }
  const args = parseArgs(argv);
  if (!(await fileExists(args.rawJsonl))) {
    const legacyRawJsonl = path.resolve(path.dirname(args.rawJsonl), 'raw.normalized.jsonl');
    if (await fileExists(legacyRawJsonl)) {
      args.rawJsonl = legacyRawJsonl;
    }
  }

  const rawRecords = await readJsonLinesFile<RawNormalizedRecord>(args.rawJsonl);
  if (rawRecords.length === 0) {
    throw new Error(`No raw records found in ${args.rawJsonl}`);
  }

  const selectedDomains = Array.from(new Set(rawRecords.map((record) => record.document.domain.toLowerCase())));
  const preparedDocuments = await loadPreparedDocuments({ domains: selectedDomains });
  const docByKey = new Map(preparedDocuments.map((doc) => [toDocKey(doc.domain, doc.document_id), doc]));

  const comparisons: ComparisonRecord[] = [];
  let missingGroundTruth = 0;
  let scoredRuns = 0;
  let runErrors = 0;
  let successChanged = 0;

  for (const record of rawRecords) {
    const docKey = toDocKey(record.document.domain, record.document.document_id);
    const prepared = docByKey.get(docKey);

    if (!prepared) {
      missingGroundTruth += 1;
      comparisons.push({
        schema_version: '1.0',
        task_key: record.task_key,
        completed_at: record.completed_at,
        model: record.model,
        document: record.document,
        runtime: record.runtime,
        legacy_metrics: record.legacy_metrics,
        comparison: null,
      });
      continue;
    }

    if (record.runtime.error !== null) {
      runErrors += 1;
      comparisons.push({
        schema_version: '1.0',
        task_key: record.task_key,
        completed_at: record.completed_at,
        model: record.model,
        document: record.document,
        runtime: record.runtime,
        legacy_metrics: record.legacy_metrics,
        comparison: null,
      });
      continue;
    }

    const score = scoreModelOutputDetailed(record.payload.raw_output, prepared.ground_truth);
    const fieldPassPct = score.fieldTotal > 0 ? pct(score.fieldCorrect, score.fieldTotal) : 0;
    const criticalPassPct = score.criticalTotal > 0 ? pct(score.criticalCorrect, score.criticalTotal) : 0;
    const keyFoundPct = score.requestedKeyCount > 0 ? pct(score.foundKeyCount, score.requestedKeyCount) : 0;
    const success = score.criticalTotal > 0 && score.criticalCorrect === score.criticalTotal;

    if (success !== record.legacy_metrics.success) {
      successChanged += 1;
    }

    scoredRuns += 1;

    comparisons.push({
      schema_version: '1.0',
      task_key: record.task_key,
      completed_at: record.completed_at,
      model: record.model,
      document: record.document,
      runtime: record.runtime,
      legacy_metrics: record.legacy_metrics,
      comparison: {
        field_total: score.fieldTotal,
        field_correct: score.fieldCorrect,
        field_pass_pct: round(fieldPassPct, 2),
        critical_total: score.criticalTotal,
        critical_correct: score.criticalCorrect,
        critical_pass_pct: round(criticalPassPct, 2),
        found_key_count: score.foundKeyCount,
        requested_key_count: score.requestedKeyCount,
        keys_found_pct: round(keyFoundPct, 2),
        success,
        key_comparisons: score.keyComparisons.map((row) => ({
          key: row.key,
          critical: row.critical,
          scored: row.scored,
          expected_values: row.expectedValues,
          extracted_value: row.extractedValue,
          matched: row.matched,
          match_mode: row.matchMode,
        })),
      },
    });
  }

  comparisons.sort((a, b) => a.task_key.localeCompare(b.task_key));

  await writeJsonLinesFile(args.outputJsonl, comparisons);

  const summary = {
    generated_at: new Date().toISOString(),
    input_raw_jsonl: args.rawJsonl,
    output_comparison_jsonl: args.outputJsonl,
    records_total: comparisons.length,
    records_scored: scoredRuns,
    records_with_runtime_error: runErrors,
    records_missing_ground_truth: missingGroundTruth,
    success_changed_vs_legacy: successChanged,
  };

  await writeJsonFile(args.outputSummary, summary);

  console.log(`Comparison records: ${comparisons.length}`);
  console.log(`Scored: ${scoredRuns}`);
  console.log(`Runtime errors: ${runErrors}`);
  console.log(`Missing GT: ${missingGroundTruth}`);
  console.log(`Success changed vs legacy: ${successChanged}`);
  console.log(`JSONL: ${args.outputJsonl}`);
  console.log(`Summary: ${args.outputSummary}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
