"""Offline replay test for the Anthropic provider.

The Anthropic provider uses the official `anthropic` SDK rather than raw
httpx (per the locked-in port decision). We inject a custom
`httpx.AsyncClient` with a `MockTransport` into `AsyncAnthropic` so the SDK's
request/response cycle is exercised end-to-end without network I/O.
"""

from __future__ import annotations

import json

import httpx
import pytest
from anthropic import AsyncAnthropic

from ocr_mini_bench.ocr.providers.anthropic import run_anthropic_ocr
from ocr_mini_bench.ocr.types import OCRModelRunRequest


def _mock_anthropic_client(handler: object) -> AsyncAnthropic:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    http_client = httpx.AsyncClient(transport=transport, base_url="https://api.anthropic.com")
    return AsyncAnthropic(api_key="fake", http_client=http_client)


@pytest.mark.replay
async def test_anthropic_provider_sends_cache_control_blocks_and_parses_text() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": '{"pairs":[]}'}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 8,
                    "cache_read_input_tokens": 40,
                    "cache_creation_input_tokens": 50,
                },
            },
        )

    sdk_client = _mock_anthropic_client(handler)
    request = OCRModelRunRequest(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        system_prompt="sys",
        user_prompt="user",
        pdf_base64="ZmFrZQ==",
        filename="x.pdf",
    )
    try:
        result = await run_anthropic_ocr(
            request=request,
            api_key="fake",
            started_at=0.0,
            max_output_tokens=1200,
            client=sdk_client,
        )
    finally:
        await sdk_client.close()

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 1200
    # All three text/document blocks carry cache_control.
    system_blocks = body["system"]
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    user_content = body["messages"][0]["content"]
    types_ = [block["type"] for block in user_content]
    assert types_ == ["document", "text"]
    assert user_content[0]["cache_control"] == {"type": "ephemeral"}
    assert user_content[0]["source"]["media_type"] == "application/pdf"
    assert user_content[1]["cache_control"] == {"type": "ephemeral"}
    assert body["metadata"]["user_id"].startswith("ocr-")

    assert result.text == '{"pairs":[]}'
    assert result.input_tokens == 100
    assert result.output_tokens == 8
    assert result.cached_input_tokens == 40
    assert result.cache_hit is True
    assert result.cache_write_tokens == 50


@pytest.mark.replay
async def test_anthropic_provider_retries_without_temperature_on_deprecation() -> None:
    seen_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_payloads.append(payload)
        if "temperature" in payload:
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "The `temperature` parameter is deprecated for this model.",
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-1-20250805",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    sdk_client = _mock_anthropic_client(handler)
    request = OCRModelRunRequest(
        provider="anthropic",
        model_id="claude-opus-4-1-20250805",
        system_prompt="s",
        user_prompt="u",
        pdf_base64="ZmFrZQ==",
        filename="x.pdf",
    )
    try:
        result = await run_anthropic_ocr(
            request=request,
            api_key="fake",
            started_at=0.0,
            max_output_tokens=1200,
            client=sdk_client,
        )
    finally:
        await sdk_client.close()

    # First call had temperature; the SDK also auto-retries on 4xx (max_retries=2),
    # so we may see more than 2 total. What matters is that *some* call without
    # temperature succeeded and produced the final result.
    assert any("temperature" not in p for p in seen_payloads)
    assert result.text == "ok"
