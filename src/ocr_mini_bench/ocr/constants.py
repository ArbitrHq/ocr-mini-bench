"""Provider-runner constants. Mirrors `src/ocr/constants.ts`."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_OUTPUT_TOKENS = 1200
RETRY_MAX_OUTPUT_TOKENS = 8000
RETRY_OUTPUT_MULTIPLIER = 2

MISTRAL_OCR_COST_PER_PAGE_USD = 2 / 1000
MISTRAL_OCR_ANNOTATED_COST_PER_PAGE_USD = 3 / 1000

GEMINI_EXPLICIT_CACHE_MODELS: frozenset[str] = frozenset(
    {
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.5-flash",
    }
)

GEMINI_CACHE_TTL_SECONDS = 600


@dataclass
class GeminiCacheEntry:
    name: str
    expire_at_ms: int
