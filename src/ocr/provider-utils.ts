import { createHash } from 'node:crypto';
import type { ModelProvider } from '../config/model-catalog';
import { parseFirstJsonObject } from './parsing';
import { RETRY_MAX_OUTPUT_TOKENS, RETRY_OUTPUT_MULTIPLIER } from './constants';

export function getProviderApiKey(provider: ModelProvider): string | undefined {
  if (provider === 'anthropic') return process.env.ANTHROPIC_API_KEY;
  if (provider === 'openai') return process.env.OPENAI_API_KEY;
  if (provider === 'mistral') return process.env.MISTRAL_API_KEY;
  return process.env.GOOGLE_API_KEY;
}

export function buildPromptCacheKey(
  modelId: string,
  systemPrompt: string,
  userPrompt: string,
  pdfBase64: string
): string {
  return createHash('sha256')
    .update(modelId)
    .update('\n')
    .update(systemPrompt)
    .update('\n')
    .update(userPrompt)
    .update('\n')
    .update(pdfBase64)
    .digest('hex');
}

export type OpenAIReasoningEffort =
  | 'none'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh';

export function getOpenAIReasoningEffort(modelId: string): OpenAIReasoningEffort | null {
  const normalized = modelId.toLowerCase();

  // GPT-5.4 family uses a newer effort enum that does not accept "minimal".
  if (normalized.startsWith('gpt-5.4')) return 'low';

  if (normalized.startsWith('gpt-5') || normalized.startsWith('o')) return 'minimal';

  return null;
}

export function isMistralOcrModel(modelId: string): boolean {
  const normalized = modelId.toLowerCase();
  return normalized === 'mistral-ocr-latest' || normalized.startsWith('mistral-ocr');
}

export function isLikelyTruncatedText(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) return false;
  if (parseFirstJsonObject(trimmed)) return false;

  if (trimmed.endsWith(',') || trimmed.endsWith(':')) return true;
  const openCurly = (trimmed.match(/\{/g) || []).length;
  const closeCurly = (trimmed.match(/\}/g) || []).length;
  const openSquare = (trimmed.match(/\[/g) || []).length;
  const closeSquare = (trimmed.match(/\]/g) || []).length;
  if (openCurly > closeCurly || openSquare > closeSquare) return true;

  return false;
}

export function getRetryMaxOutputTokens(current: number): number {
  return Math.min(RETRY_MAX_OUTPUT_TOKENS, Math.max(current + 400, current * RETRY_OUTPUT_MULTIPLIER));
}
