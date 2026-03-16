import { createHash } from 'node:crypto';
import {
  GEMINI_CACHE_TTL_SECONDS,
  GEMINI_EXPLICIT_CACHE_MODELS,
  type GeminiCacheEntry,
} from './constants';

const geminiCacheByKey = new Map<string, GeminiCacheEntry>();
const geminiCacheCreateInFlight = new Map<string, Promise<string | null>>();

export function shouldUseGeminiExplicitCache(modelId: string): boolean {
  return GEMINI_EXPLICIT_CACHE_MODELS.has(modelId);
}

export function buildGeminiThinkingConfig(modelId: string): Record<string, number> | undefined {
  const normalized = modelId.toLowerCase();
  if (normalized.includes('flash')) {
    return { thinkingBudget: 0 };
  }
  return undefined;
}

function buildGeminiCacheKey(modelId: string, systemPrompt: string, pdfBase64: string): string {
  const hash = createHash('sha256').update(systemPrompt).update('\n').update(pdfBase64).digest('hex');
  return `${modelId}:${hash}`;
}

async function getOrCreateGeminiCacheName(params: {
  apiKey: string;
  modelId: string;
  systemPrompt: string;
  pdfBase64: string;
}): Promise<string | null> {
  const cacheKey = buildGeminiCacheKey(params.modelId, params.systemPrompt, params.pdfBase64);
  const now = Date.now();
  const existing = geminiCacheByKey.get(cacheKey);
  if (existing && existing.expireAtMs > now + 5000) {
    return existing.name;
  }

  const inFlight = geminiCacheCreateInFlight.get(cacheKey);
  if (inFlight) {
    return inFlight;
  }

  const createPromise = (async () => {
    const createResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/cachedContents?key=${encodeURIComponent(params.apiKey)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: `models/${params.modelId}`,
          displayName: `ocr-${cacheKey.slice(0, 16)}`,
          systemInstruction: {
            parts: [{ text: params.systemPrompt }],
          },
          contents: [
            {
              role: 'user',
              parts: [
                {
                  inlineData: {
                    mimeType: 'application/pdf',
                    data: params.pdfBase64,
                  },
                },
              ],
            },
          ],
          ttl: `${GEMINI_CACHE_TTL_SECONDS}s`,
        }),
      }
    );

    const createData = (await createResponse.json()) as Record<string, unknown>;
    if (!createResponse.ok) {
      return null;
    }

    const name = typeof createData.name === 'string' ? createData.name : null;
    if (!name) {
      return null;
    }

    const expireTime = typeof createData.expireTime === 'string' ? Date.parse(createData.expireTime) : NaN;
    const expireAtMs = Number.isNaN(expireTime) ? Date.now() + GEMINI_CACHE_TTL_SECONDS * 1000 : expireTime;
    geminiCacheByKey.set(cacheKey, { name, expireAtMs });
    return name;
  })();

  geminiCacheCreateInFlight.set(cacheKey, createPromise);
  try {
    return await createPromise;
  } finally {
    geminiCacheCreateInFlight.delete(cacheKey);
  }
}

export async function buildGeminiGeneratePayload(params: {
  apiKey: string;
  modelId: string;
  systemPrompt: string;
  userPrompt: string;
  pdfBase64: string;
  maxOutputTokens: number;
}): Promise<Record<string, unknown>> {
  const thinkingConfig = buildGeminiThinkingConfig(params.modelId);

  if (shouldUseGeminiExplicitCache(params.modelId)) {
    const cacheName = await getOrCreateGeminiCacheName({
      apiKey: params.apiKey,
      modelId: params.modelId,
      systemPrompt: params.systemPrompt,
      pdfBase64: params.pdfBase64,
    });

    if (cacheName) {
      return {
        cachedContent: cacheName,
        contents: [
          {
            role: 'user',
            parts: [{ text: params.userPrompt }],
          },
        ],
        generationConfig: {
          maxOutputTokens: params.maxOutputTokens,
          temperature: 0,
          ...(thinkingConfig ? { thinkingConfig } : {}),
        },
      };
    }
  }

  return {
    systemInstruction: {
      parts: [{ text: params.systemPrompt }],
    },
    contents: [
      {
        role: 'user',
        parts: [
          {
            inlineData: {
              mimeType: 'application/pdf',
              data: params.pdfBase64,
            },
          },
          { text: params.userPrompt },
        ],
      },
    ],
    generationConfig: {
      maxOutputTokens: params.maxOutputTokens,
      temperature: 0,
      ...(thinkingConfig ? { thinkingConfig } : {}),
    },
  };
}
