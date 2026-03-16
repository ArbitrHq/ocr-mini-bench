import type { ModelProvider } from '../config/model-catalog';

export interface OCRModelRunRequest {
  provider: ModelProvider;
  modelId: string;
  systemPrompt: string;
  userPrompt: string;
  pdfBase64: string;
  filename: string;
  maxOutputTokens?: number;
}

export interface OCRModelRunResult {
  text: string;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
  cachedInputTokens: number;
  cacheHit: boolean;
  cacheWriteTokens: number;
  totalCostUsd?: number;
  noCacheCostUsd?: number;
}

export interface ApprovedKey {
  name: string;
  critical: boolean;
}
