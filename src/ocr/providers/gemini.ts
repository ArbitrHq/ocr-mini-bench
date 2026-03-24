import type { OCRModelRunRequest, OCRModelRunResult } from '../types';
import { buildGeminiGeneratePayload } from '../gemini-cache';
import { getRetryMaxOutputTokens, isLikelyTruncatedText } from '../provider-utils';
import { readTextFromGeminiResponse } from '../text-readers';
import {
  isRecord,
  getRecordProperty,
  getStringProperty,
  getNumberProperty,
  getArrayProperty,
} from '../../lib/type-guards';

export async function runGeminiOCR(params: {
  request: OCRModelRunRequest;
  apiKey: string;
  startedAt: number;
  maxOutputTokens: number;
}): Promise<OCRModelRunResult> {
  const { request, apiKey, startedAt, maxOutputTokens } = params;

  const runGemini = async (maxTokensForCall: number) => {
    const geminiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(request.modelId)}:generateContent?key=${encodeURIComponent(apiKey)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          await buildGeminiGeneratePayload({
            apiKey,
            modelId: request.modelId,
            systemPrompt: request.systemPrompt,
            userPrompt: request.userPrompt,
            pdfBase64: request.pdfBase64,
            maxOutputTokens: maxTokensForCall,
          })
        ),
      }
    );

    const geminiData: unknown = await geminiResponse.json();
    if (!isRecord(geminiData)) {
      throw new Error('Gemini request returned invalid response.');
    }
    if (!geminiResponse.ok) {
      const errorObj = getRecordProperty(geminiData, 'error');
      const errorMessage = errorObj ? getStringProperty(errorObj, 'message') : undefined;
      throw new Error(errorMessage ?? 'Gemini request failed.');
    }
    return geminiData;
  };

  let geminiData = await runGemini(maxOutputTokens);
  let geminiText = readTextFromGeminiResponse(geminiData);
  const geminiCandidates = getArrayProperty(geminiData, 'candidates') ?? [];
  const firstGeminiCandidate = geminiCandidates.length > 0 && isRecord(geminiCandidates[0])
    ? geminiCandidates[0]
    : undefined;
  const finishReason = firstGeminiCandidate
    ? getStringProperty(firstGeminiCandidate, 'finishReason') ?? ''
    : '';

  if (finishReason === 'MAX_TOKENS' || isLikelyTruncatedText(geminiText)) {
    const retryMax = getRetryMaxOutputTokens(maxOutputTokens);
    if (retryMax > maxOutputTokens) {
      geminiData = await runGemini(retryMax);
      geminiText = readTextFromGeminiResponse(geminiData);
    }
  }

  const usageMetadata = getRecordProperty(geminiData, 'usageMetadata');
  const cachedInputTokens = usageMetadata ? getNumberProperty(usageMetadata, 'cachedContentTokenCount') ?? 0 : 0;

  return {
    text: geminiText,
    inputTokens: usageMetadata ? getNumberProperty(usageMetadata, 'promptTokenCount') ?? 0 : 0,
    outputTokens: usageMetadata ? getNumberProperty(usageMetadata, 'candidatesTokenCount') ?? 0 : 0,
    latencyMs: Date.now() - startedAt,
    cachedInputTokens,
    cacheHit: cachedInputTokens > 0,
    cacheWriteTokens: 0,
  };
}
