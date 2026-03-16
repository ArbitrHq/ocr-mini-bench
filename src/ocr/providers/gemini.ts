import type { OCRModelRunRequest, OCRModelRunResult } from '../types';
import { buildGeminiGeneratePayload } from '../gemini-cache';
import { getRetryMaxOutputTokens, isLikelyTruncatedText } from '../provider-utils';
import { readTextFromGeminiResponse } from '../text-readers';

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

    const geminiData = (await geminiResponse.json()) as Record<string, unknown>;
    if (!geminiResponse.ok) {
      const error =
        geminiData.error && typeof geminiData.error === 'object'
          ? (geminiData.error as Record<string, unknown>).message
          : null;
      throw new Error(typeof error === 'string' ? error : 'Gemini request failed.');
    }
    return geminiData;
  };

  let geminiData = await runGemini(maxOutputTokens);
  let geminiText = readTextFromGeminiResponse(geminiData);
  const geminiCandidates = Array.isArray(geminiData.candidates) ? geminiData.candidates : [];
  const firstGeminiCandidate =
    geminiCandidates.length > 0 && typeof geminiCandidates[0] === 'object' && geminiCandidates[0]
      ? (geminiCandidates[0] as Record<string, unknown>)
      : null;
  const finishReason =
    firstGeminiCandidate && typeof firstGeminiCandidate.finishReason === 'string'
      ? firstGeminiCandidate.finishReason
      : '';

  if (finishReason === 'MAX_TOKENS' || isLikelyTruncatedText(geminiText)) {
    const retryMax = getRetryMaxOutputTokens(maxOutputTokens);
    if (retryMax > maxOutputTokens) {
      geminiData = await runGemini(retryMax);
      geminiText = readTextFromGeminiResponse(geminiData);
    }
  }

  const usageMetadata =
    geminiData.usageMetadata && typeof geminiData.usageMetadata === 'object'
      ? (geminiData.usageMetadata as Record<string, unknown>)
      : null;
  const cachedInputTokens = usageMetadata ? Number(usageMetadata.cachedContentTokenCount || 0) : 0;

  return {
    text: geminiText,
    inputTokens: usageMetadata ? Number(usageMetadata.promptTokenCount || 0) : 0,
    outputTokens: usageMetadata ? Number(usageMetadata.candidatesTokenCount || 0) : 0,
    latencyMs: Date.now() - startedAt,
    cachedInputTokens,
    cacheHit: cachedInputTokens > 0,
    cacheWriteTokens: 0,
  };
}
