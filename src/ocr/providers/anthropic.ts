import Anthropic from '@anthropic-ai/sdk';
import type { OCRModelRunRequest, OCRModelRunResult } from '../types';
import { buildPromptCacheKey } from '../provider-utils';
import { readTextFromAnthropicContent } from '../text-readers';
import type {
  CacheableTextBlockParam,
  CacheableMessageParam,
  AnthropicUsageWithCache,
} from './anthropic.types';

export async function runAnthropicOCR(params: {
  request: OCRModelRunRequest;
  apiKey: string;
  startedAt: number;
  maxOutputTokens: number;
}): Promise<OCRModelRunResult> {
  const { request, apiKey, startedAt, maxOutputTokens } = params;
  const client = new Anthropic({ apiKey });
  const promptCacheKey = buildPromptCacheKey(
    request.modelId,
    request.systemPrompt,
    request.userPrompt,
    request.pdfBase64
  );

  const systemBlocks: CacheableTextBlockParam[] = [
    {
      type: 'text',
      text: request.systemPrompt,
      cache_control: { type: 'ephemeral' },
    },
  ];

  const userMessage: CacheableMessageParam = {
    role: 'user',
    content: [
      {
        type: 'document',
        source: {
          type: 'base64',
          media_type: 'application/pdf',
          data: request.pdfBase64,
        },
        cache_control: { type: 'ephemeral' },
      },
      {
        type: 'text',
        text: request.userPrompt,
        cache_control: { type: 'ephemeral' },
      },
    ],
  };

  const response = await client.messages.create({
    model: request.modelId,
    temperature: 0,
    max_tokens: maxOutputTokens,
    // Type assertions needed because SDK types don't include cache_control yet
    system: systemBlocks as Anthropic.Messages.TextBlockParam[],
    messages: [userMessage as Anthropic.Messages.MessageParam],
    metadata: {
      user_id: `ocr-${promptCacheKey.slice(0, 24)}`,
    },
  });

  const usage = response.usage as AnthropicUsageWithCache;
  const cacheReadTokens = usage.cache_read_input_tokens ?? 0;
  const cacheCreationTokens = usage.cache_creation_input_tokens ?? 0;

  return {
    text: readTextFromAnthropicContent(response.content),
    inputTokens: response.usage?.input_tokens ?? 0,
    outputTokens: response.usage?.output_tokens ?? 0,
    latencyMs: Date.now() - startedAt,
    cachedInputTokens: cacheReadTokens,
    cacheHit: cacheReadTokens > 0,
    cacheWriteTokens: cacheCreationTokens,
  };
}
