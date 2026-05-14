"""Provider dispatcher. Mirrors `src/ocr/runner.ts`.

`run_ocr_model` is the single entry point used by the benchmark orchestration:
it resolves the API key from env, picks the provider-specific implementation,
and returns a uniform `OCRModelRunResult`.
"""

from __future__ import annotations

import time

import httpx

from .constants import DEFAULT_MAX_OUTPUT_TOKENS
from .provider_utils import get_provider_api_key
from .providers.anthropic import run_anthropic_ocr
from .providers.gemini import run_gemini_ocr
from .providers.mistral import run_mistral_ocr
from .providers.openai import run_openai_ocr
from .types import OCRModelRunRequest, OCRModelRunResult


async def run_ocr_model(
    request: OCRModelRunRequest,
    *,
    client: httpx.AsyncClient | None = None,
) -> OCRModelRunResult:
    api_key = get_provider_api_key(request.provider)
    if not api_key:
        raise RuntimeError(f"Missing API key for {request.provider}.")

    started_at = time.time()
    max_output_tokens = request.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS

    if request.provider == "anthropic":
        return await run_anthropic_ocr(
            request=request,
            api_key=api_key,
            started_at=started_at,
            max_output_tokens=max_output_tokens,
        )

    owns_client = client is None
    http_client = client if client is not None else httpx.AsyncClient(timeout=300.0)
    try:
        if request.provider == "openai":
            return await run_openai_ocr(
                request=request,
                api_key=api_key,
                started_at=started_at,
                max_output_tokens=max_output_tokens,
                client=http_client,
            )
        if request.provider == "mistral":
            return await run_mistral_ocr(
                request=request,
                api_key=api_key,
                started_at=started_at,
                max_output_tokens=max_output_tokens,
                client=http_client,
            )
        return await run_gemini_ocr(
            request=request,
            api_key=api_key,
            started_at=started_at,
            max_output_tokens=max_output_tokens,
            client=http_client,
        )
    finally:
        if owns_client:
            await http_client.aclose()
