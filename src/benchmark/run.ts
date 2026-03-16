import { promises as fs } from 'node:fs';
import path from 'node:path';
import { estimateCostUsd, runOCRModel } from '../ocr/runner';
import { loadBenchmarkConfig, loadPreparedDocuments, summarizeDataset } from './dataset';
import { buildBenchmarkSystemPrompt, buildBenchmarkUserPrompt, scoreModelOutputDetailed } from './scoring';
import type {
  BenchmarkDebugRun,
  BenchmarkModelConfig,
  BenchmarkRunOptions,
  BenchmarkSnapshot,
  PreparedBenchmarkDocument,
  SingleRunMetrics,
} from './types';
import { aggregateRows, buildCacheSummary, buildMarkdownTable } from './run/aggregation';
import {
  buildBenchmarkId,
  buildBenchmarkRunTaskKey,
  idForModel,
  toErrorMessage,
  toRepoRelativePath,
} from './run/identity';
import { pct, round } from './run/math';
import { runInPool } from './run/pool';
import { runByProviderLanes } from './run/provider-lanes';
import { createProgressReporter } from './run/progress';

const BENCHMARK_SYSTEM_PROMPT_PATH = path.resolve(
  process.cwd(),
  'prompts/ocr/benchmark/extract_system.txt'
);
const BENCHMARK_USER_PROMPT_PATH = path.resolve(
  process.cwd(),
  'prompts/ocr/benchmark/extract_user.txt'
);
const REPO_ROOT = process.cwd();

export { buildBenchmarkRunTaskKey };

export interface BenchmarkTaskCompletionEvent {
  task_key: string;
  metrics: SingleRunMetrics;
  debug: BenchmarkDebugRun;
}

export interface BenchmarkRunExecutionControls {
  initial_runs?: SingleRunMetrics[];
  initial_debug_runs?: BenchmarkDebugRun[];
  skip_task_keys?: Set<string>;
  only_task_keys?: Set<string>;
  on_task_complete?: (event: BenchmarkTaskCompletionEvent) => void | Promise<void>;
}

type RunTask = {
  provider: BenchmarkModelConfig['provider'];
  model: BenchmarkModelConfig;
  document: PreparedBenchmarkDocument;
  runNumber: number;
  modelKey: string;
  taskKey: string;
};

function makeMissingPdfResult(task: RunTask, systemPrompt: string, userPrompt: string): {
  metrics: SingleRunMetrics;
  debug: BenchmarkDebugRun;
} {
  const baseRunResult: SingleRunMetrics = {
    task_key: task.taskKey,
    model_key: task.modelKey,
    provider: task.model.provider,
    model_id: task.model.model_id,
    model_label: task.model.model_label,
    tier: task.model.tier,
    domain: task.document.domain,
    document_id: task.document.document_id,
    run_number: task.runNumber,
    success: false,
    field_total: 0,
    field_correct: 0,
    critical_total: 0,
    critical_correct: 0,
    field_accuracy_pct: 0,
    critical_accuracy_pct: 0,
    found_key_count: 0,
    requested_key_count: task.document.ground_truth.keys.length,
    latency_ms: 0,
    input_tokens: 0,
    output_tokens: 0,
    total_cost_usd: 0,
    cache_hit: false,
    cached_input_tokens: 0,
    cache_write_tokens: 0,
    error: 'PDF base64 payload missing.',
  };

  const debugRun: BenchmarkDebugRun = {
    ...baseRunResult,
    task_key: task.taskKey,
    system_prompt_used: systemPrompt,
    user_prompt_used: userPrompt,
    raw_output: '',
    parsed_output: null,
    extracted_pairs: [],
    key_comparisons: [],
  };

  return { metrics: baseRunResult, debug: debugRun };
}

async function executeTask(task: RunTask, params: {
  benchmarkSystemPromptTemplate: string;
  benchmarkUserPromptTemplate: string;
  pdfByDocumentId: Map<string, string>;
  controls: BenchmarkRunExecutionControls;
  runResults: SingleRunMetrics[];
  debugRuns: BenchmarkDebugRun[];
  reportProgress: ReturnType<typeof createProgressReporter>;
}) {
  const {
    benchmarkSystemPromptTemplate,
    benchmarkUserPromptTemplate,
    pdfByDocumentId,
    controls,
    runResults,
    debugRuns,
    reportProgress,
  } = params;

  const systemPrompt = buildBenchmarkSystemPrompt(benchmarkSystemPromptTemplate);
  const userPrompt = buildBenchmarkUserPrompt(benchmarkUserPromptTemplate, task.document.ground_truth);
  const pdfBase64 = pdfByDocumentId.get(task.document.document_id);

  if (!pdfBase64) {
    const { metrics, debug } = makeMissingPdfResult(task, systemPrompt, userPrompt);
    runResults.push(metrics);
    debugRuns.push(debug);
    if (controls.on_task_complete) {
      await controls.on_task_complete({ task_key: task.taskKey, metrics, debug });
    }
    reportProgress({
      documentId: task.document.document_id,
      modelLabel: task.model.model_label,
      runNumber: task.runNumber,
      ok: false,
      error: 'PDF base64 payload missing.',
    });
    return;
  }

  try {
    const result = await runOCRModel({
      provider: task.model.provider,
      modelId: task.model.model_id,
      systemPrompt,
      userPrompt,
      pdfBase64,
      filename: path.basename(task.document.source_pdf_abs),
      maxOutputTokens: 4000,
    });

    const score = scoreModelOutputDetailed(result.text, task.document.ground_truth);
    const fieldAccuracyPct = score.fieldTotal > 0 ? pct(score.fieldCorrect, score.fieldTotal) : 0;
    const criticalAccuracyPct = score.criticalTotal > 0 ? pct(score.criticalCorrect, score.criticalTotal) : 0;
    const hasScorableFields = score.fieldTotal > 0;
    const hasScorableCriticalFields = score.criticalTotal > 0;
    const success = hasScorableCriticalFields
      ? score.criticalCorrect === score.criticalTotal
      : hasScorableFields && score.fieldCorrect === score.fieldTotal;

    const totalCostUsd =
      typeof result.totalCostUsd === 'number'
        ? result.totalCostUsd
        : estimateCostUsd(task.model.model_id, result.inputTokens, result.outputTokens, {
            cachedInputTokens: result.cachedInputTokens,
          });

    const metrics: SingleRunMetrics = {
      task_key: task.taskKey,
      model_key: task.modelKey,
      provider: task.model.provider,
      model_id: task.model.model_id,
      model_label: task.model.model_label,
      tier: task.model.tier,
      domain: task.document.domain,
      document_id: task.document.document_id,
      run_number: task.runNumber,
      success,
      field_total: score.fieldTotal,
      field_correct: score.fieldCorrect,
      critical_total: score.criticalTotal,
      critical_correct: score.criticalCorrect,
      field_accuracy_pct: round(fieldAccuracyPct, 2),
      critical_accuracy_pct: round(criticalAccuracyPct, 2),
      found_key_count: score.foundKeyCount,
      requested_key_count: score.requestedKeyCount,
      latency_ms: result.latencyMs,
      input_tokens: result.inputTokens,
      output_tokens: result.outputTokens,
      total_cost_usd: totalCostUsd,
      cache_hit: result.cacheHit,
      cached_input_tokens: result.cachedInputTokens,
      cache_write_tokens: result.cacheWriteTokens,
      error: null,
    };

    const debug: BenchmarkDebugRun = {
      ...metrics,
      task_key: task.taskKey,
      system_prompt_used: systemPrompt,
      user_prompt_used: userPrompt,
      raw_output: result.text,
      parsed_output: score.parsedOutput,
      extracted_pairs: score.extractedPairs,
      key_comparisons: score.keyComparisons.map((comparison) => ({
        key: comparison.key,
        critical: comparison.critical,
        scored: comparison.scored,
        expected_values: comparison.expectedValues,
        extracted_value: comparison.extractedValue,
        matched: comparison.matched,
        match_mode: comparison.matchMode,
      })),
    };

    runResults.push(metrics);
    debugRuns.push(debug);
    if (controls.on_task_complete) {
      await controls.on_task_complete({ task_key: task.taskKey, metrics, debug });
    }

    reportProgress({
      documentId: task.document.document_id,
      modelLabel: task.model.model_label,
      runNumber: task.runNumber,
      ok: true,
    });
  } catch (error: unknown) {
    const errorMessage = toErrorMessage(error);
    const metrics: SingleRunMetrics = {
      task_key: task.taskKey,
      model_key: task.modelKey,
      provider: task.model.provider,
      model_id: task.model.model_id,
      model_label: task.model.model_label,
      tier: task.model.tier,
      domain: task.document.domain,
      document_id: task.document.document_id,
      run_number: task.runNumber,
      success: false,
      field_total: 0,
      field_correct: 0,
      critical_total: 0,
      critical_correct: 0,
      field_accuracy_pct: 0,
      critical_accuracy_pct: 0,
      found_key_count: 0,
      requested_key_count: task.document.ground_truth.keys.length,
      latency_ms: 0,
      input_tokens: 0,
      output_tokens: 0,
      total_cost_usd: 0,
      cache_hit: false,
      cached_input_tokens: 0,
      cache_write_tokens: 0,
      error: errorMessage,
    };

    const debug: BenchmarkDebugRun = {
      ...metrics,
      task_key: task.taskKey,
      system_prompt_used: systemPrompt,
      user_prompt_used: userPrompt,
      raw_output: '',
      parsed_output: null,
      extracted_pairs: [],
      key_comparisons: [],
    };

    runResults.push(metrics);
    debugRuns.push(debug);
    if (controls.on_task_complete) {
      await controls.on_task_complete({ task_key: task.taskKey, metrics, debug });
    }

    reportProgress({
      documentId: task.document.document_id,
      modelLabel: task.model.model_label,
      runNumber: task.runNumber,
      ok: false,
      error: errorMessage,
    });
  }
}

export async function runOCRLeaderboardBenchmark(
  options: BenchmarkRunOptions = {},
  controls: BenchmarkRunExecutionControls = {}
): Promise<BenchmarkSnapshot> {
  const [config, benchmarkSystemPromptTemplate, benchmarkUserPromptTemplate] = await Promise.all([
    loadBenchmarkConfig(),
    fs.readFile(BENCHMARK_SYSTEM_PROMPT_PATH, 'utf8'),
    fs.readFile(BENCHMARK_USER_PROMPT_PATH, 'utf8'),
  ]);

  const runsPerModel = Math.max(1, options.runs_per_model ?? config.default_runs_per_model);
  const providerParallel = options.provider_parallel === true;
  const selectedDomains = options.domains?.map((value) => value.trim().toLowerCase()).filter(Boolean) ?? [];
  const selectedModelFilters = options.models?.map((value) => value.trim().toLowerCase()).filter(Boolean) ?? [];
  const maxDocumentsPerDomain =
    typeof options.max_documents_per_domain === 'number' && options.max_documents_per_domain > 0
      ? options.max_documents_per_domain
      : null;

  const documents = await loadPreparedDocuments({
    domains: selectedDomains,
    maxDocumentsPerDomain: maxDocumentsPerDomain ?? undefined,
  });

  if (documents.length === 0) {
    throw new Error('No benchmark documents selected. Check manifest and domain filters.');
  }

  const selectedModels =
    selectedModelFilters.length === 0
      ? config.models
      : config.models.filter((model) => {
          const modelId = model.model_id.trim().toLowerCase();
          const modelLabel = model.model_label.trim().toLowerCase();
          return selectedModelFilters.some((needle) => needle === modelId || needle === modelLabel);
        });

  if (selectedModels.length === 0) {
    throw new Error(
      `No benchmark models selected. Check --models filters against config/models.public.json model_id or model_label.`
    );
  }

  const dataset = summarizeDataset(documents);
  const warnings: string[] = [];

  if (dataset.labeled_keys === 0) {
    warnings.push('Ground truth is not labeled yet. Fill expected values before trusting rankings.');
  }
  if (dataset.labeled_keys < dataset.total_keys) {
    warnings.push(`Only ${dataset.labeled_keys}/${dataset.total_keys} keys are currently labeled.`);
  }

  for (const [domain, count] of Object.entries(dataset.documents_per_domain)) {
    if (count < 10) {
      warnings.push(`Domain "${domain}" has ${count} documents; target is >= 10 for launch quality.`);
    }
  }

  const pdfByDocumentId = new Map<string, string>();
  await Promise.all(
    documents.map(async (document) => {
      const buffer = await fs.readFile(document.source_pdf_abs);
      pdfByDocumentId.set(document.document_id, buffer.toString('base64'));
    })
  );

  const runTasks: RunTask[] = [];
  const requestedByModel = new Map<string, number>();
  const requestedByDomainModel = new Map<string, number>();

  for (const model of selectedModels) {
    const modelKey = idForModel(model.provider, model.model_id);

    for (const document of documents) {
      for (let runNumber = 1; runNumber <= runsPerModel; runNumber += 1) {
        const taskKey = buildBenchmarkRunTaskKey({
          modelKey,
          domain: document.domain,
          documentId: document.document_id,
          runNumber,
        });
        const inOnlySet = !controls.only_task_keys || controls.only_task_keys.has(taskKey);
        const notSkipped = !controls.skip_task_keys || !controls.skip_task_keys.has(taskKey);
        if (inOnlySet && notSkipped) {
          runTasks.push({ provider: model.provider, model, document, runNumber, modelKey, taskKey });
        }
      }

      requestedByDomainModel.set(
        `${document.domain}::${modelKey}`,
        (requestedByDomainModel.get(`${document.domain}::${modelKey}`) ?? 0) + runsPerModel
      );
    }

    requestedByModel.set(modelKey, (requestedByModel.get(modelKey) ?? 0) + documents.length * runsPerModel);
  }

  const providerCount = new Set(runTasks.map((task) => task.provider)).size;
  const providerDefaultParallel = providerCount > 0 ? providerCount : 1;
  const maxParallelRequests = providerParallel
    ? Math.max(1, Math.min(providerDefaultParallel, options.max_parallel_requests ?? providerDefaultParallel))
    : Math.max(1, options.max_parallel_requests ?? config.max_parallel_requests);

  const runResults: SingleRunMetrics[] = [...(controls.initial_runs ?? [])];
  const debugRuns: BenchmarkDebugRun[] = [...(controls.initial_debug_runs ?? [])];

  const benchmarkStartedAtMs = Date.now();
  const expectedRunsByDocument = new Map<string, number>();
  for (const task of runTasks) {
    expectedRunsByDocument.set(task.document.document_id, (expectedRunsByDocument.get(task.document.document_id) ?? 0) + 1);
  }

  const reportProgress = createProgressReporter({
    totalTasks: runTasks.length,
    expectedRunsByDocument,
    startedAtMs: benchmarkStartedAtMs,
  });

  if (providerParallel) {
    await runByProviderLanes(runTasks, maxParallelRequests, async (task) => {
      await executeTask(task, {
        benchmarkSystemPromptTemplate,
        benchmarkUserPromptTemplate,
        pdfByDocumentId,
        controls,
        runResults,
        debugRuns,
        reportProgress,
      });
    });
  } else {
    await runInPool(runTasks, maxParallelRequests, async (task) => {
      await executeTask(task, {
        benchmarkSystemPromptTemplate,
        benchmarkUserPromptTemplate,
        pdfByDocumentId,
        controls,
        runResults,
        debugRuns,
        reportProgress,
      });
    });
  }

  const leaderboard = aggregateRows(runResults, requestedByModel);
  const cacheSummary = buildCacheSummary(runResults);
  const byDomain = Object.keys(dataset.documents_per_domain)
    .sort((a, b) => a.localeCompare(b))
    .map((domain) => {
      const domainRuns = runResults.filter((run) => run.domain === domain);
      const domainRequested = new Map<string, number>();

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

  const snapshot: BenchmarkSnapshot = {
    generated_at: new Date().toISOString(),
    benchmark_id: buildBenchmarkId(),
    benchmark_description: config.description,
    options: {
      runs_per_model: runsPerModel,
      max_parallel_requests: maxParallelRequests,
      provider_parallel: providerParallel,
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
      documents: documents.map((document) => ({
        document_id: document.document_id,
        domain: document.domain,
        source_pdf: document.source_pdf,
        ground_truth_path: toRepoRelativePath(document.ground_truth_abs, REPO_ROOT),
        ground_truth: document.ground_truth_raw,
      })),
      runs: debugRuns,
    },
  };

  return snapshot;
}
