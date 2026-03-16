import { loadPricingRegistryFromBackendConfig } from '../config/backend-config';

export function estimateCostUsd(
  modelId: string,
  inputTokens: number,
  outputTokens: number,
  options?: { cachedInputTokens?: number }
): number {
  const pricingRegistry = loadPricingRegistryFromBackendConfig();
  const pricing = pricingRegistry[modelId];
  if (!pricing) {
    return 0;
  }
  const cachedInputTokens = Math.max(0, Number(options?.cachedInputTokens || 0));
  const nonCachedInputTokens = Math.max(0, inputTokens - cachedInputTokens);
  const cacheInputRate = pricing.cache_input ?? pricing.input;
  return (
    (nonCachedInputTokens / 1_000_000) * pricing.input +
    (cachedInputTokens / 1_000_000) * cacheInputRate +
    (outputTokens / 1_000_000) * pricing.output
  );
}
