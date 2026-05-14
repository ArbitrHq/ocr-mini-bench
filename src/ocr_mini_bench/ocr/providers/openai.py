"""OpenAI OCR provider. Mirrors `src/ocr/providers/openai.ts`.

Calls the `/v1/responses` endpoint with the PDF passed inline as a base64
`input_file`. Reasoning effort is selected per model id; truncation is retried
once with a larger `max_output_tokens` ceiling.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ...lib.type_guards import (
    get_number_property,
    get_record_property,
    get_string_property,
    is_record,
)
from ..provider_utils import (
    build_prompt_cache_key,
    get_openai_reasoning_effort,
    get_retry_max_output_tokens,
    is_likely_truncated_text,
)
from ..text_readers import read_text_from_openai_response
from ..types import OCRModelRunRequest, OCRModelRunResult


async def run_openai_ocr(
    *,
    request: OCRModelRunRequest,
    api_key: str,
    started_at: float,
    max_output_tokens: int,
    client: httpx.AsyncClient,
) -> OCRModelRunResult:
    prompt_cache_key = build_prompt_cache_key(
        request.model_id, request.system_prompt, request.user_prompt, request.pdf_base64
    )
    filename = request.filename or "document.pdf"

    async def _call(max_tokens_for_call: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "max_output_tokens": max_tokens_for_call,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": request.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": filename,
                            "file_data": f"data:application/pdf;base64,{request.pdf_base64}",
                        },
                        {"type": "input_text", "text": request.user_prompt},
                    ],
                },
            ],
            "prompt_cache_key": prompt_cache_key,
        }

        reasoning_effort = get_openai_reasoning_effort(request.model_id)
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}

        response = await client.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("OpenAI request returned invalid response.") from exc
        if not is_record(data):
            raise RuntimeError("OpenAI request returned invalid response.")
        if response.status_code >= 400:
            error_obj = get_record_property(data, "error")
            message = get_string_property(error_obj, "message") if error_obj else None
            raise RuntimeError(message or "OpenAI request failed.")
        return data

    data = await _call(max_output_tokens)
    text = read_text_from_openai_response(data)
    status = get_string_property(data, "status") or ""
    incomplete_details = get_record_property(data, "incomplete_details")
    incomplete_reason = (
        get_string_property(incomplete_details, "reason") or "" if incomplete_details else ""
    )
    truncated_by_tokens = status == "incomplete" and incomplete_reason == "max_output_tokens"

    if truncated_by_tokens or is_likely_truncated_text(text):
        retry_max = get_retry_max_output_tokens(max_output_tokens)
        if retry_max > max_output_tokens:
            data = await _call(retry_max)
            text = read_text_from_openai_response(data)

    usage = get_record_property(data, "usage")
    prompt_tokens_details = (
        get_record_property(usage, "prompt_tokens_details") if usage else None
    )
    cached_input_tokens = (
        int(get_number_property(prompt_tokens_details, "cached_tokens") or 0)
        if prompt_tokens_details
        else 0
    )
    input_tokens = int(get_number_property(usage, "input_tokens") or 0) if usage else 0
    output_tokens = int(get_number_property(usage, "output_tokens") or 0) if usage else 0

    return OCRModelRunResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int((time.time() - started_at) * 1000),
        cached_input_tokens=cached_input_tokens,
        cache_hit=cached_input_tokens > 0,
        cache_write_tokens=0,
    )
