import type Anthropic from '@anthropic-ai/sdk';

/**
 * Extended text block with cache control support.
 * The official SDK types don't yet include cache_control for text blocks.
 */
export interface CacheableTextBlockParam extends Anthropic.Messages.TextBlockParam {
  cache_control?: { type: 'ephemeral' };
}

/**
 * Document block with cache control support for PDF inputs.
 */
export interface CacheableDocumentBlockParam {
  type: 'document';
  source: {
    type: 'base64';
    media_type: 'application/pdf';
    data: string;
  };
  cache_control?: { type: 'ephemeral' };
}

/**
 * Extended message param that supports cacheable document blocks.
 */
export interface CacheableMessageParam {
  role: 'user' | 'assistant';
  content: Array<CacheableTextBlockParam | CacheableDocumentBlockParam>;
}

/**
 * Extended usage object that includes cache token metrics.
 */
export interface AnthropicUsageWithCache {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}
