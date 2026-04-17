import type { ModelProvider } from './model-catalog';

export interface ModelPricing {
  input: number;
  output: number;
  cache_input?: number;
}

const PRICING_REGISTRY: Record<string, ModelPricing> = {
  // Anthropic
  'claude-opus-4-7': { input: 5.0, output: 25.0 },
  'claude-opus-4-6': { input: 5.0, output: 25.0 },
  'claude-opus-4-5': { input: 5.0, output: 25.0 },
  'claude-opus-4-1': { input: 15.0, output: 75.0 },
  'claude-sonnet-4-6': { input: 3.0, output: 15.0 },
  'claude-sonnet-4-5': { input: 3.0, output: 15.0 },
  'claude-sonnet-4': { input: 3.0, output: 15.0 },
  'claude-haiku-4-5': { input: 1.0, output: 5.0 },
  'claude-haiku-3': { input: 0.25, output: 1.25 },
  // OpenAI
  'gpt-5-2': { input: 1.75, output: 14.0 },
  'gpt-5': { input: 1.25, output: 10.0 },
  'gpt-5.4': { input: 2.5, output: 15.0 },
  'gpt-5-mini': { input: 0.25, output: 2.0 },
  'gpt-5-nano': { input: 0.05, output: 0.4 },
  'gpt-5.4-mini': { input: 0.75, output: 4.5 },
  'gpt-5.4-nano': { input: 0.2, output: 1.25 },
  'gpt-5-pro': { input: 15.0, output: 120.0 },
  'gpt-4.1': { input: 2.0, output: 8.0 },
  'gpt-4.1-mini': { input: 0.4, output: 1.6 },
  'gpt-4.1-nano': { input: 0.1, output: 0.4 },
  'gpt-4o': { input: 2.5, output: 10.0 },
  'gpt-4o-mini': { input: 0.15, output: 0.6 },
  o3: { input: 2.0, output: 8.0 },
  'o4-mini': { input: 1.1, output: 4.4 },
  // Gemini
  'gemini-3.1-pro-preview': { input: 2.0, output: 12.0 },
  'gemini-2.5-pro': { input: 1.25, output: 10.0 },
  'gemini-2.5-flash': { input: 0.3, output: 2.5 },
  'gemini-2.5-flash-lite': { input: 0.1, output: 0.4 },
  'gemini-3.1-flash-lite-preview': { input: 0.25, output: 1.5, cache_input: 0.025 },
  'gemini-3-pro-preview': { input: 2.0, output: 12.0 },
  'gemini-3-flash-preview': { input: 0.5, output: 3.0 },
  // Mistral LLMs
  'mistral-small-latest': { input: 0.1, output: 0.3 },
  'mistral-medium-latest': { input: 0.4, output: 2.0 },
  'mistral-large-latest': { input: 0.5, output: 1.5 },
  // Fallback
  default: { input: 0.0, output: 0.0 },
};

export function loadPricingRegistryFromBackendConfig(): Record<string, ModelPricing> {
  return PRICING_REGISTRY;
}

export function inferProviderFromModelId(modelId: string): ModelProvider {
  const normalized = modelId.toLowerCase();
  if (normalized.startsWith('claude')) return 'anthropic';
  if (normalized.startsWith('gemini')) return 'google';
  if (
    normalized.startsWith('mistral') ||
    normalized.startsWith('ministral') ||
    normalized.startsWith('pixtral') ||
    normalized.startsWith('codestral') ||
    normalized.startsWith('devstral') ||
    normalized.startsWith('voxtral')
  ) {
    return 'mistral';
  }
  return 'openai';
}
