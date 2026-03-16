import type { LegacyCheckpointRecord, RawNormalizedRecord } from './types';
import type { SingleRunMetrics } from '../benchmark/types';

type CheckpointDebugLike = {
  system_prompt_used?: string;
  user_prompt_used?: string;
  raw_output?: string;
  parsed_output?: unknown;
  extracted_pairs?: Array<{ key: string; value: string; found: boolean }>;
};

export type CheckpointLineLike = {
  task_key: string;
  completed_at?: string;
  metrics: SingleRunMetrics;
  debug: CheckpointDebugLike;
};

export function toTaskKeyFromDebug(run: LegacyCheckpointRecord['debug']): string {
  return run.task_key ?? `${run.model_key}::${run.domain}::${run.document_id}::run${run.run_number}`;
}

export function fromCheckpointRecord(record: CheckpointLineLike): RawNormalizedRecord {
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

export function fromDebugRun(run: LegacyCheckpointRecord['debug']): RawNormalizedRecord {
  const taskKey = toTaskKeyFromDebug(run);

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
