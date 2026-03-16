import { DEFAULT_MAX_OUTPUT_TOKENS } from './constants';
import { flattenCatalogModels } from './catalog';
import { estimateCostUsd } from './cost';
import { parseFirstJsonObject, normalizeKeyName, fallbackExtractKeys } from './parsing';
import { getProviderApiKey } from './provider-utils';
import { runAnthropicOCR } from './providers/anthropic';
import { runOpenAIOCR } from './providers/openai';
import { runGeminiOCR } from './providers/gemini';
import { runMistralOCR } from './providers/mistral';
import type { OCRModelRunRequest, OCRModelRunResult, ApprovedKey } from './types';

export type { OCRModelRunRequest, OCRModelRunResult, ApprovedKey };

export async function runOCRModel(request: OCRModelRunRequest): Promise<OCRModelRunResult> {
  const apiKey = getProviderApiKey(request.provider);
  if (!apiKey) {
    throw new Error(`Missing API key for ${request.provider}.`);
  }

  const startedAt = Date.now();
  const maxOutputTokens = request.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS;

  if (request.provider === 'anthropic') {
    return runAnthropicOCR({ request, apiKey, startedAt, maxOutputTokens });
  }

  if (request.provider === 'openai') {
    return runOpenAIOCR({ request, apiKey, startedAt, maxOutputTokens });
  }

  if (request.provider === 'mistral') {
    return runMistralOCR({ request, apiKey, startedAt, maxOutputTokens });
  }

  return runGeminiOCR({ request, apiKey, startedAt, maxOutputTokens });
}

export { flattenCatalogModels, estimateCostUsd, parseFirstJsonObject, normalizeKeyName, fallbackExtractKeys };
