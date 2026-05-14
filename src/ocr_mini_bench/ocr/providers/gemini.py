"""Gemini OCR provider. Mirrors `src/ocr/providers/gemini.ts`."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ...lib.type_guards import (
    get_array_property,
    get_number_property,
    get_record_property,
    get_string_property,
    is_record,
)
from ..gemini_cache import build_gemini_generate_payload
from ..provider_utils import get_retry_max_output_tokens, is_likely_truncated_text
from ..text_readers import read_text_from_gemini_response
from ..types import OCRModelRunRequest, OCRModelRunResult


async def run_gemini_ocr(
    *,
    request: OCRModelRunRequest,
    api_key: str,
    started_at: float,
    max_output_tokens: int,
    client: httpx.AsyncClient,
) -> OCRModelRunResult:
    """Run a single OCR call against Gemini's `generateContent` API."""

    async def _call(max_tokens_for_call: int) -> dict[str, Any]:
        payload = await build_gemini_generate_payload(
            client=client,
            api_key=api_key,
            model_id=request.model_id,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            pdf_base64=request.pdf_base64,
            max_output_tokens=max_tokens_for_call,
        )
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{request.model_id}:generateContent",
            params={"key": api_key},
            json=payload,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Gemini request returned invalid response.") from exc
        if not is_record(data):
            raise RuntimeError("Gemini request returned invalid response.")
        if response.status_code >= 400:
            error_obj = get_record_property(data, "error")
            message = get_string_property(error_obj, "message") if error_obj else None
            raise RuntimeError(message or "Gemini request failed.")
        return data

    data = await _call(max_output_tokens)
    text = read_text_from_gemini_response(data)

    candidates = get_array_property(data, "candidates") or []
    first_candidate = candidates[0] if candidates and is_record(candidates[0]) else None
    finish_reason = get_string_property(first_candidate, "finishReason") or "" if first_candidate else ""

    if finish_reason == "MAX_TOKENS" or is_likely_truncated_text(text):
        retry_max = get_retry_max_output_tokens(max_output_tokens)
        if retry_max > max_output_tokens:
            data = await _call(retry_max)
            text = read_text_from_gemini_response(data)

    usage = get_record_property(data, "usageMetadata")
    cached_input_tokens = int(get_number_property(usage, "cachedContentTokenCount") or 0) if usage else 0
    input_tokens = int(get_number_property(usage, "promptTokenCount") or 0) if usage else 0
    output_tokens = int(get_number_property(usage, "candidatesTokenCount") or 0) if usage else 0

    return OCRModelRunResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int((time.time() - started_at) * 1000),
        cached_input_tokens=cached_input_tokens,
        cache_hit=cached_input_tokens > 0,
        cache_write_tokens=0,
    )
