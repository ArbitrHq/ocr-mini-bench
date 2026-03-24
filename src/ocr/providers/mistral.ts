import type { OCRModelRunRequest, OCRModelRunResult } from '../types';
import {
  MISTRAL_OCR_ANNOTATED_COST_PER_PAGE_USD,
  MISTRAL_OCR_COST_PER_PAGE_USD,
} from '../constants';
import { isMistralOcrModel } from '../provider-utils';
import {
  readMistralOcrAnnotationAsText,
  readMistralOcrMarkdown,
  readTextFromMistralChatResponse,
} from '../text-readers';
import {
  isRecord,
  getRecordProperty,
  getStringProperty,
  getNumberProperty,
  getArrayProperty,
} from '../../lib/type-guards';

export async function runMistralOCR(params: {
  request: OCRModelRunRequest;
  apiKey: string;
  startedAt: number;
  maxOutputTokens: number;
}): Promise<OCRModelRunResult> {
  const { request, apiKey, startedAt, maxOutputTokens } = params;

  if (isMistralOcrModel(request.modelId)) {
    const annotationPrompt = `${request.systemPrompt}\n\n${request.userPrompt}`.trim();
    const ocrResponse = await fetch('https://api.mistral.ai/v1/ocr', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: request.modelId,
        document: {
          type: 'document_url',
          document_url: `data:application/pdf;base64,${request.pdfBase64}`,
        },
        document_annotation_format: {
          type: 'json_schema',
          json_schema: {
            name: 'ocr_key_extraction',
            schema: {
              type: 'object',
              properties: {
                pairs: {
                  type: 'array',
                  items: {
                    type: 'object',
                    properties: {
                      key: { type: 'string' },
                      value: { type: 'string' },
                      found: { type: 'boolean' },
                    },
                    required: ['key', 'value', 'found'],
                    additionalProperties: false,
                  },
                },
                missing_keys: {
                  type: 'array',
                  items: { type: 'string' },
                },
                notes: { type: 'string' },
              },
              required: ['pairs', 'missing_keys'],
              additionalProperties: false,
            },
          },
        },
        document_annotation_prompt: annotationPrompt,
      }),
    });

    const ocrData: unknown = await ocrResponse.json();
    if (!isRecord(ocrData)) {
      throw new Error('Mistral OCR request returned invalid response.');
    }
    if (!ocrResponse.ok) {
      const errorObj = getRecordProperty(ocrData, 'error');
      const errorMessage = errorObj ? getStringProperty(errorObj, 'message') : undefined;
      throw new Error(errorMessage ?? 'Mistral OCR request failed.');
    }

    const annotationText = readMistralOcrAnnotationAsText(ocrData);
    const markdownText = readMistralOcrMarkdown(ocrData);
    const responseText =
      annotationText ||
      JSON.stringify({
        pairs: [],
        missing_keys: [],
        notes: markdownText ? 'No document annotation returned; markdown fallback included.' : 'No OCR output.',
        markdown: markdownText,
      });

    const usageInfo = getRecordProperty(ocrData, 'usage_info');
    const pagesProcessed = usageInfo ? getNumberProperty(usageInfo, 'pages_processed') ?? 0 : 0;
    const hasAnnotation = Boolean(annotationText);
    const perPageCost = hasAnnotation ? MISTRAL_OCR_ANNOTATED_COST_PER_PAGE_USD : MISTRAL_OCR_COST_PER_PAGE_USD;
    const totalCostUsd = Math.max(0, pagesProcessed) * perPageCost;

    return {
      text: responseText,
      inputTokens: 0,
      outputTokens: 0,
      latencyMs: Date.now() - startedAt,
      cachedInputTokens: 0,
      cacheHit: false,
      cacheWriteTokens: 0,
      totalCostUsd,
      noCacheCostUsd: totalCostUsd,
    };
  }

  const ocrResponse = await fetch('https://api.mistral.ai/v1/ocr', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'mistral-ocr-latest',
      document: {
        type: 'document_url',
        document_url: `data:application/pdf;base64,${request.pdfBase64}`,
      },
    }),
  });

  const ocrData: unknown = await ocrResponse.json();
  if (!isRecord(ocrData)) {
    throw new Error('Mistral OCR request returned invalid response.');
  }
  if (!ocrResponse.ok) {
    const errorObj = getRecordProperty(ocrData, 'error');
    const errorMessage = errorObj ? getStringProperty(errorObj, 'message') : undefined;
    throw new Error(errorMessage ?? 'Mistral OCR request failed.');
  }

  const pages = getArrayProperty(ocrData, 'pages') ?? [];
  const ocrMarkdown = pages
    .map((page) => {
      if (!isRecord(page)) return '';
      return getStringProperty(page, 'markdown') ?? '';
    })
    .filter(Boolean)
    .join('\n\n');

  const chatResponse = await fetch('https://api.mistral.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: request.modelId,
      temperature: 0,
      max_tokens: maxOutputTokens,
      messages: [
        {
          role: 'system',
          content: request.systemPrompt,
        },
        {
          role: 'user',
          content: `${request.userPrompt}\n\nDocument OCR markdown:\n${ocrMarkdown}`,
        },
      ],
    }),
  });

  const chatData: unknown = await chatResponse.json();
  if (!isRecord(chatData)) {
    throw new Error('Mistral chat request returned invalid response.');
  }
  if (!chatResponse.ok) {
    const errorObj = getRecordProperty(chatData, 'error');
    const errorMessage = errorObj ? getStringProperty(errorObj, 'message') : undefined;
    throw new Error(errorMessage ?? 'Mistral chat request failed.');
  }

  const usage = getRecordProperty(chatData, 'usage');

  return {
    text: readTextFromMistralChatResponse(chatData),
    inputTokens: usage ? getNumberProperty(usage, 'prompt_tokens') ?? 0 : 0,
    outputTokens: usage ? getNumberProperty(usage, 'completion_tokens') ?? 0 : 0,
    latencyMs: Date.now() - startedAt,
    cachedInputTokens: 0,
    cacheHit: false,
    cacheWriteTokens: 0,
  };
}
