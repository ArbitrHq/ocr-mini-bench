export type ModelProvider = 'anthropic' | 'openai' | 'google' | 'mistral';

export interface ModelOption {
  id: string;
  label: string;
}

// Shared model catalog used for provider inference and label fallback.
export const MODEL_CATALOG: Record<ModelProvider, ModelOption[]> = {
  anthropic: [
    { id: 'claude-opus-4-1-20250805', label: 'Claude Opus 4.1' },
    { id: 'claude-opus-4-20250514', label: 'Claude Opus 4' },
    { id: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4' },
    { id: 'claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5' },
    { id: 'claude-3-7-sonnet-20250219', label: 'Claude Sonnet 3.7' },
    { id: 'claude-3-5-haiku-20241022', label: 'Claude Haiku 3.5' },
  ],
  openai: [
    { id: 'gpt-5', label: 'GPT-5' },
    { id: 'gpt-5.4', label: 'GPT-5.4' },
    { id: 'gpt-5.5', label: 'GPT-5.5' },
    { id: 'gpt-5-mini', label: 'GPT-5 mini' },
    { id: 'gpt-5-nano', label: 'GPT-5 nano' },
    { id: 'gpt-5.4-mini', label: 'GPT-5.4 mini' },
    { id: 'gpt-5.4-nano', label: 'GPT-5.4 nano' },
    { id: 'gpt-5-pro', label: 'GPT-5 pro' },
    { id: 'gpt-4.1', label: 'GPT-4.1' },
    { id: 'gpt-4.1-mini', label: 'GPT-4.1 mini' },
    { id: 'gpt-4.1-nano', label: 'GPT-4.1 nano' },
    { id: 'gpt-4o', label: 'GPT-4o' },
    { id: 'gpt-4o-mini', label: 'GPT-4o mini' },
    { id: 'o3', label: 'o3' },
    { id: 'o4-mini', label: 'o4-mini' },
  ],
  google: [
    { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
    { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
    { id: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash-Lite' },
    { id: 'gemini-3.1-flash-lite-preview', label: 'Gemini 3.1 Flash-Lite' },
    { id: 'gemini-3-pro-preview', label: 'Gemini 3 Pro' },
    { id: 'gemini-3-flash-preview', label: 'Gemini 3 Flash' },
  ],
  mistral: [
    { id: 'mistral-ocr-latest', label: 'Mistral OCR (Latest)' },
    { id: 'mistral-large-latest', label: 'Mistral Large (Latest)' },
    { id: 'mistral-medium-latest', label: 'Mistral Medium (Latest)' },
    { id: 'mistral-small-latest', label: 'Mistral Small (Latest)' },
  ],
};

export function defaultModelForProvider(provider: ModelProvider): string {
  return MODEL_CATALOG[provider][0]?.id ?? '';
}
