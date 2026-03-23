import type { BenchmarkTier, KeyMatchMode } from '../benchmark/types';
import type { ModelProvider } from '../config/model-catalog';

export interface LegacyCheckpointRecord {
  task_key: string;
  completed_at?: string;
  metrics: {
    task_key?: string;
    model_key: string;
    provider: ModelProvider;
    model_id: string;
    model_label: string;
    tier: BenchmarkTier;
    domain: string;
    document_id: string;
    run_number: number;
    success: boolean;
    field_total: number;
    field_correct: number;
    critical_total: number;
    critical_correct: number;
    field_accuracy_pct: number;
    critical_accuracy_pct: number;
    found_key_count: number;
    requested_key_count: number;
    latency_ms: number;
    input_tokens: number;
    output_tokens: number;
    total_cost_usd: number;
    cache_hit: boolean;
    cached_input_tokens: number;
    cache_write_tokens: number;
    error: string | null;
  };
  debug: {
    task_key?: string;
    model_key: string;
    provider: ModelProvider;
    model_id: string;
    model_label: string;
    tier: BenchmarkTier;
    domain: string;
    document_id: string;
    run_number: number;
    success: boolean;
    field_total: number;
    field_correct: number;
    critical_total: number;
    critical_correct: number;
    field_accuracy_pct: number;
    critical_accuracy_pct: number;
    found_key_count: number;
    requested_key_count: number;
    latency_ms: number;
    input_tokens?: number;
    output_tokens?: number;
    total_cost_usd: number;
    cache_hit: boolean;
    cached_input_tokens: number;
    cache_write_tokens: number;
    error: string | null;
    system_prompt_used?: string;
    user_prompt_used?: string;
    raw_output: string;
    parsed_output: unknown;
    extracted_pairs?: Array<{ key: string; value: string; found: boolean }>;
    key_comparisons?: Array<{
      key: string;
      critical: boolean;
      scored: boolean;
      expected_values: string[];
      extracted_value: string;
      matched: boolean;
      match_mode: KeyMatchMode;
    }>;
  };
}

export interface RawNormalizedRecord {
  schema_version: '1.0';
  task_key: string;
  completed_at: string | null;
  model: {
    model_key: string;
    provider: ModelProvider;
    model_id: string;
    model_label: string;
    tier: BenchmarkTier;
  };
  document: {
    domain: string;
    document_id: string;
    run_number: number;
  };
  runtime: {
    latency_ms: number;
    input_tokens: number;
    output_tokens: number;
    total_cost_usd: number;
    cache_hit: boolean;
    cached_input_tokens: number;
    cache_write_tokens: number;
    error: string | null;
  };
  payload: {
    system_prompt_used: string;
    user_prompt_used: string;
    raw_output: string;
    parsed_output: unknown;
    extracted_pairs: Array<{ key: string; value: string; found: boolean }>;
  };
  legacy_metrics: {
    success: boolean;
    field_total: number;
    field_correct: number;
    critical_total: number;
    critical_correct: number;
    field_accuracy_pct: number;
    critical_accuracy_pct: number;
    found_key_count: number;
    requested_key_count: number;
  };
}

export interface ComparisonRecord {
  schema_version: '1.0';
  task_key: string;
  completed_at: string | null;
  model: RawNormalizedRecord['model'];
  document: RawNormalizedRecord['document'];
  runtime: RawNormalizedRecord['runtime'];
  legacy_metrics: RawNormalizedRecord['legacy_metrics'];
  comparison:
    | {
        field_total: number;
        field_correct: number;
        field_pass_pct: number;
        critical_total: number;
        critical_correct: number;
        critical_pass_pct: number;
        found_key_count: number;
        requested_key_count: number;
        keys_found_pct: number;
        success: boolean;
        key_comparisons: Array<{
          key: string;
          critical: boolean;
          scored: boolean;
          expected_values: string[];
          extracted_value: string;
          matched: boolean;
          match_mode: KeyMatchMode;
        }>;
      }
    | null;
}

export interface MetricRangeStats {
  count: number;
  min: number | null;
  p05: number | null;
  p50: number | null;
  p95: number | null;
  max: number | null;
  mean: number | null;
  stddev: number | null;
}

export interface MetricRanges {
  cost_per_doc_usd: MetricRangeStats;
  cost_per_success_usd: MetricRangeStats;
  success_pct_runs: MetricRangeStats;
  success_pct_docs: MetricRangeStats;
  pass_at_3_strict_pct_docs: MetricRangeStats;
  pass_at_5_strict_pct_docs: MetricRangeStats;
  critical_fields_pct: MetricRangeStats;
  all_fields_pct: MetricRangeStats;
  latency_ms: MetricRangeStats;
}

export interface AggregatedMetricRow {
  rank: number;
  model_key: string;
  provider: ModelProvider;
  model_id: string;
  model_label: string;
  tier: BenchmarkTier;
  runs_requested: number;
  runs_completed: number;
  successful_runs: number;
  failed_runs: number;
  success_rate_pct: number;
  pass_at_2_pct: number | null;
  pass_at_3_pct: number | null;
  pass_at_5_pct: number | null;
  pass_at_10_pct: number | null;
  pass_at_2_strict_pct: number | null;
  pass_at_3_strict_pct: number | null;
  pass_at_5_strict_pct: number | null;
  pass_at_10_strict_pct: number | null;
  avg_total_field_pass_pct: number;
  avg_field_accuracy_pct: number;
  avg_critical_accuracy_pct: number;
  avg_keys_found_pct: number;
  field_pass_stddev_pct: number;
  field_accuracy_variance_pct: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  latency_stddev_ms: number;
  latency_cv_pct: number;
  total_cost_usd: number;
  avg_cost_per_doc_usd: number | null;
  avg_cost_per_run_usd: number;
  cost_per_success_usd: number | null;
  p95_cost_usd: number;
  metric_ranges?: MetricRanges;
}

export interface MetricsSnapshot {
  schema_version: '1.0';
  generated_at: string;
  source: {
    comparison_jsonl: string;
  };
  run_count: number;
  model_rows: AggregatedMetricRow[];
  by_domain: Array<{ domain: string; rows: AggregatedMetricRow[] }>;
}

export interface LeaderboardAggregationSnapshot {
  schema_version: '1.0';
  generated_at: string;
  source: {
    raw_jsonl: string;
    comparison_jsonl: string;
    metrics_json: string;
  };
  dataset: {
    total_documents: number;
    documents_per_domain: Record<string, number>;
    total_keys: number;
    labeled_keys: number;
    critical_keys: number;
    labeled_critical_keys: number;
  };
  leaderboard: AggregatedMetricRow[];
  by_domain: Array<{ domain: string; rows: AggregatedMetricRow[] }>;
  run_count: number;
  warnings: string[];
}
