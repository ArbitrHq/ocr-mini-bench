import type { ModelProvider } from '../config/model-catalog';

export type BenchmarkTier = 'budget' | 'balanced' | 'sota';

export interface BenchmarkModelConfig {
  provider: ModelProvider;
  model_id: string;
  model_label: string;
  tier: BenchmarkTier;
}

export interface BenchmarkConfig {
  description: string;
  default_runs_per_model: number;
  max_parallel_requests: number;
  models: BenchmarkModelConfig[];
}

export type KeyMatchMode = 'exact' | 'normalized_text' | 'contains' | 'numeric';

export interface GroundTruthKey {
  name: string;
  critical: boolean;
  expected: string | string[] | null;
  data_type?: string;
  match?: KeyMatchMode;
  notes?: string;
}

export interface GroundTruthDocument {
  schema_version: string;
  document_id: string;
  domain: string;
  source_pdf: string;
  notes?: string;
  keys: GroundTruthKey[];
}

export interface ManifestDocument {
  document_id: string;
  domain: string;
  source_pdf: string;
  ground_truth: string;
}

export interface ManifestDomain {
  id: string;
  source_directory: string;
  document_count: number;
  documents: ManifestDocument[];
}

export interface BenchmarkManifest {
  schema_version: string;
  generated_at: string;
  domains: ManifestDomain[];
}

export interface PreparedBenchmarkDocument {
  document_id: string;
  domain: string;
  source_pdf: string;
  source_pdf_abs: string;
  ground_truth_abs: string;
  ground_truth_raw: unknown;
  ground_truth: GroundTruthDocument;
}

export interface SingleRunMetrics {
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
}

export interface LeaderboardRow {
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
  avg_field_accuracy_pct: number;
  avg_critical_accuracy_pct: number;
  field_accuracy_variance_pct: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  total_cost_usd: number;
  avg_cost_per_doc_usd: number | null;
  avg_cost_per_run_usd: number;
  cost_per_success_usd: number | null;
  p95_cost_usd: number;
  p05_field_accuracy_pct: number;
}

export interface DomainLeaderboard {
  domain: string;
  rows: LeaderboardRow[];
}

export interface DatasetSummary {
  total_documents: number;
  documents_per_domain: Record<string, number>;
  total_keys: number;
  labeled_keys: number;
  critical_keys: number;
  labeled_critical_keys: number;
}

export interface BenchmarkRunOptions {
  runs_per_model?: number;
  max_parallel_requests?: number;
  domains?: string[];
  max_documents_per_domain?: number;
  provider_parallel?: boolean;
  models?: string[];
}

export interface BenchmarkDebugRun {
  task_key?: string;
  model_key: string;
  provider: ModelProvider;
  model_id: string;
  model_label: string;
  tier: BenchmarkTier;
  domain: string;
  document_id: string;
  run_number: number;
  latency_ms: number;
  total_cost_usd: number;
  success: boolean;
  error: string | null;
  field_total: number;
  field_correct: number;
  critical_total: number;
  critical_correct: number;
  found_key_count: number;
  requested_key_count: number;
  system_prompt_used: string;
  user_prompt_used: string;
  raw_output: string;
  parsed_output: unknown;
  extracted_pairs: Array<{ key: string; value: string; found: boolean }>;
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

export interface BenchmarkDebugDocument {
  document_id: string;
  domain: string;
  source_pdf: string;
  ground_truth_path: string;
  ground_truth: unknown;
}

export interface BenchmarkDebugSnapshot {
  documents: BenchmarkDebugDocument[];
  runs: BenchmarkDebugRun[];
}

export interface BenchmarkSnapshot {
  generated_at: string;
  benchmark_id: string;
  benchmark_description: string;
  options: {
    runs_per_model: number;
    max_parallel_requests: number;
    provider_parallel: boolean;
    selected_domains: string[];
    max_documents_per_domain: number | null;
  };
  dataset: DatasetSummary;
  leaderboard: LeaderboardRow[];
  by_domain: DomainLeaderboard[];
  run_count: number;
  markdown_table: string;
  warnings: string[];
  cache_summary?: Array<{
    model_key: string;
    model_label: string;
    provider: ModelProvider;
    runs: number;
    cache_hits: number;
    cache_hit_rate_pct: number;
    cached_input_tokens_total: number;
    cached_input_tokens_avg: number;
    cache_write_tokens_total: number;
    cache_write_tokens_avg: number;
  }>;
  debug?: BenchmarkDebugSnapshot;
}
