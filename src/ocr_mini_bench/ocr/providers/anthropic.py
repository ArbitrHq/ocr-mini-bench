"""Anthropic OCR provider. Mirrors `src/ocr/providers/anthropic.ts`.

Uses the official `anthropic` Python SDK (decision per
[[feedback-python-port-decisions]]). All cache-control blocks are passed
explicitly so cache_read/cache_creation token metrics flow through to
`OCRModelRunResult`. Temperature is retried without it on the deprecation error
the TS port also catches.
"""

from __future__ import annotations

import time
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message

from ..provider_utils import build_prompt_cache_key
from ..text_readers import read_text_from_anthropic_content
from ..types import OCRModelRunRequest, OCRModelRunResult


def _to_error_message(error: object) -> str:
    if isinstance(error, BaseException):
        return str(error)
    if isinstance(error, str):
        return error
    message = getattr(error, "message", None)
    if isinstance(message, str):
        return message
    return ""


def _is_temperature_deprecated_error(error: object) -> bool:
    message = _to_error_message(error).lower()
    return "temperature" in message and "deprecated" in message


async def run_anthropic_ocr(
    *,
    request: OCRModelRunRequest,
    api_key: str,
    started_at: float,
    max_output_tokens: int,
    client: AsyncAnthropic | None = None,
) -> OCRModelRunResult:
    sdk_client = client if client is not None else AsyncAnthropic(api_key=api_key)
    prompt_cache_key = build_prompt_cache_key(
        request.model_id, request.system_prompt, request.user_prompt, request.pdf_base64
    )

    system_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": request.system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user_content: list[dict[str, Any]] = [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": request.pdf_base64,
            },
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": request.user_prompt,
            "cache_control": {"type": "ephemeral"},
        },
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

    base_payload: dict[str, Any] = {
        "model": request.model_id,
        "max_tokens": max_output_tokens,
        "system": system_blocks,
        "messages": messages,
        "metadata": {"user_id": f"ocr-{prompt_cache_key[:24]}"},
    }

    try:
        response: Message = await sdk_client.messages.create(temperature=0, **base_payload)
    except Exception as error:
        if not _is_temperature_deprecated_error(error):
            raise
        response = await sdk_client.messages.create(**base_payload)

    usage_obj = response.usage
    cache_read_tokens = cast(int, getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
    cache_creation_tokens = cast(int, getattr(usage_obj, "cache_creation_input_tokens", 0) or 0)
    input_tokens = cast(int, getattr(usage_obj, "input_tokens", 0) or 0)
    output_tokens = cast(int, getattr(usage_obj, "output_tokens", 0) or 0)

    return OCRModelRunResult(
        text=read_text_from_anthropic_content(list(response.content)),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int((time.time() - started_at) * 1000),
        cached_input_tokens=cache_read_tokens,
        cache_hit=cache_read_tokens > 0,
        cache_write_tokens=cache_creation_tokens,
    )
