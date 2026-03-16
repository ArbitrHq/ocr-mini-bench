export const DEFAULT_MAX_OUTPUT_TOKENS = 1200;
export const RETRY_MAX_OUTPUT_TOKENS = 8000;
export const RETRY_OUTPUT_MULTIPLIER = 2;

export const MISTRAL_OCR_COST_PER_PAGE_USD = 2 / 1000;
export const MISTRAL_OCR_ANNOTATED_COST_PER_PAGE_USD = 3 / 1000;

export const GEMINI_EXPLICIT_CACHE_MODELS = new Set([
  'gemini-2.5-flash-lite',
  'gemini-2.5-flash',
  'gemini-3-flash-preview',
  'gemini-3.1-flash-lite-preview',
  'gemini-3.1-pro-preview',
  'gemini-3-pro-preview',
]);

export const GEMINI_CACHE_TTL_SECONDS = 600;

export interface GeminiCacheEntry {
  name: string;
  expireAtMs: number;
}
