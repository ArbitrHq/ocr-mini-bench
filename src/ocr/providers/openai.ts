import type { OCRModelRunRequest, OCRModelRunResult } from '../types';
import {
  buildPromptCacheKey,
  getOpenAIReasoningEffort,
  getRetryMaxOutputTokens,
  isLikelyTruncatedText,
} from '../provider-utils';
import { readTextFromOpenAIResponse } from '../text-readers';
import {
  isRecord,
  getRecordProperty,
  getStringProperty,
  getNumberProperty,
} from '../../lib/type-guards';

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

    const data: unknown = await response.json();
    if (!isRecord(data)) {
      throw new Error('OpenAI request returned invalid response.');
    }
    if (!response.ok) {
      const errorObj = getRecordProperty(data, 'error');
      const errorMessage = errorObj ? getStringProperty(errorObj, 'message') : undefined;
      throw new Error(errorMessage ?? 'OpenAI request failed.');
    }
    return data;
  };

  let data = await runOpenAI(maxOutputTokens);
  let text = readTextFromOpenAIResponse(data);
  const status = getStringProperty(data, 'status') ?? '';
  const incompleteDetails = getRecordProperty(data, 'incomplete_details');
  const incompleteReason = incompleteDetails ? getStringProperty(incompleteDetails, 'reason') ?? '' : '';
  const truncatedByTokens = status === 'incomplete' && incompleteReason === 'max_output_tokens';

  if (truncatedByTokens || isLikelyTruncatedText(text)) {
    const retryMax = getRetryMaxOutputTokens(maxOutputTokens);
    if (retryMax > maxOutputTokens) {
      data = await runOpenAI(retryMax);
      text = readTextFromOpenAIResponse(data);
    }
  }

  const usageObject = getRecordProperty(data, 'usage');
  const promptTokensDetails = usageObject ? getRecordProperty(usageObject, 'prompt_tokens_details') : undefined;
  const cachedInputTokens = promptTokensDetails ? getNumberProperty(promptTokensDetails, 'cached_tokens') ?? 0 : 0;

  return {
    text,
    inputTokens: usageObject ? getNumberProperty(usageObject, 'input_tokens') ?? 0 : 0,
    outputTokens: usageObject ? getNumberProperty(usageObject, 'output_tokens') ?? 0 : 0,
    latencyMs: Date.now() - startedAt,
    cachedInputTokens,
    cacheHit: cachedInputTokens > 0,
    cacheWriteTokens: 0,
  };
}
