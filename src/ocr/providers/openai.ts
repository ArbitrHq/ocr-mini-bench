import type { OCRModelRunRequest, OCRModelRunResult } from '../types';
import {
  buildPromptCacheKey,
  getOpenAIReasoningEffort,
  getRetryMaxOutputTokens,
  isLikelyTruncatedText,
} from '../provider-utils';
import { readTextFromOpenAIResponse } from '../text-readers';

export async function runOpenAIOCR(params: {
  request: OCRModelRunRequest;
  apiKey: string;
  startedAt: number;
  maxOutputTokens: number;
}): Promise<OCRModelRunResult> {
  const { request, apiKey, startedAt, maxOutputTokens } = params;
  const promptCacheKey = buildPromptCacheKey(
    request.modelId,
    request.systemPrompt,
    request.userPrompt,
    request.pdfBase64
  );

  const runOpenAI = async (maxTokensForCall: number) => {
    const payload: Record<string, unknown> = {
      model: request.modelId,
      max_output_tokens: maxTokensForCall,
      input: [
        {
          role: 'system',
          content: [{ type: 'input_text', text: request.systemPrompt }],
        },
        {
          role: 'user',
          content: [
            {
              type: 'input_file',
              filename: request.filename || 'document.pdf',
              file_data: `data:application/pdf;base64,${request.pdfBase64}`,
            },
            {
              type: 'input_text',
              text: request.userPrompt,
            },
          ],
        },
      ],
      prompt_cache_key: promptCacheKey,
    };

    const reasoningEffort = getOpenAIReasoningEffort(request.modelId);
    if (reasoningEffort) {
      payload.reasoning = { effort: reasoningEffort };
    }

    const response = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = (await response.json()) as Record<string, unknown>;
    if (!response.ok) {
      const error =
        data.error && typeof data.error === 'object'
          ? (data.error as Record<string, unknown>).message
          : null;
      throw new Error(typeof error === 'string' ? error : 'OpenAI request failed.');
    }
    return data;
  };

  let data = await runOpenAI(maxOutputTokens);
  let text = readTextFromOpenAIResponse(data);
  const status = typeof data.status === 'string' ? data.status : '';
  const incompleteDetails =
    data.incomplete_details && typeof data.incomplete_details === 'object'
      ? (data.incomplete_details as Record<string, unknown>)
      : null;
  const incompleteReason =
    incompleteDetails && typeof incompleteDetails.reason === 'string' ? incompleteDetails.reason : '';
  const truncatedByTokens = status === 'incomplete' && incompleteReason === 'max_output_tokens';

  if (truncatedByTokens || isLikelyTruncatedText(text)) {
    const retryMax = getRetryMaxOutputTokens(maxOutputTokens);
    if (retryMax > maxOutputTokens) {
      data = await runOpenAI(retryMax);
      text = readTextFromOpenAIResponse(data);
    }
  }

  const usageObject =
    data.usage && typeof data.usage === 'object' ? (data.usage as Record<string, unknown>) : null;
  const promptTokensDetails =
    usageObject?.prompt_tokens_details && typeof usageObject.prompt_tokens_details === 'object'
      ? (usageObject.prompt_tokens_details as Record<string, unknown>)
      : null;
  const cachedInputTokens = promptTokensDetails ? Number(promptTokensDetails.cached_tokens || 0) : 0;

  return {
    text,
    inputTokens: usageObject ? Number(usageObject.input_tokens || 0) : 0,
    outputTokens: usageObject ? Number(usageObject.output_tokens || 0) : 0,
    latencyMs: Date.now() - startedAt,
    cachedInputTokens,
    cacheHit: cachedInputTokens > 0,
    cacheWriteTokens: 0,
  };
}
