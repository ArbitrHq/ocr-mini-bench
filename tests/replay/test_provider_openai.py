"""Offline replay test for the OpenAI provider.

Drives `run_openai_ocr` through an `httpx.MockTransport`. Verifies:
- the request body has the expected payload shape (input file, prompt cache key, reasoning effort)
- the response is parsed via the responses-API output array path
- truncation by `incomplete_details.reason == "max_output_tokens"` triggers retry
"""

from __future__ import annotations

import json

import httpx
import pytest

from ocr_mini_bench.ocr.provider_utils import build_prompt_cache_key
from ocr_mini_bench.ocr.providers.openai import run_openai_ocr
from ocr_mini_bench.ocr.types import OCRModelRunRequest


@pytest.mark.replay
async def test_openai_provider_sends_expected_payload_and_parses_output() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": '{"pairs":[]}'},
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 1234,
                    "output_tokens": 56,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            },
        )

    request = OCRModelRunRequest(
        provider="openai",
        model_id="gpt-5-mini",
        system_prompt="sys",
        user_prompt="user",
        pdf_base64="ZmFrZQ==",
        filename="doc.pdf",
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_openai_ocr(
            request=request,
            api_key="fake",
            started_at=0.0,
            max_output_tokens=1200,
            client=client,
        )

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["auth"] == "Bearer fake"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-5-mini"
    assert body["max_output_tokens"] == 1200
    assert body["prompt_cache_key"] == build_prompt_cache_key(
        "gpt-5-mini", "sys", "user", "ZmFrZQ=="
    )
    assert body["reasoning"] == {"effort": "minimal"}
    user_content = body["input"][1]["content"]
    assert user_content[0]["type"] == "input_file"
    assert user_content[0]["filename"] == "doc.pdf"
    assert user_content[0]["file_data"].startswith("data:application/pdf;base64,")

    assert result.text == '{"pairs":[]}'
    assert result.input_tokens == 1234
    assert result.output_tokens == 56
    assert result.cached_input_tokens == 100
    assert result.cache_hit is True
    assert result.cache_write_tokens == 0


@pytest.mark.replay
async def test_openai_provider_retries_on_incomplete_max_output_tokens() -> None:
    seen_max: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_max.append(int(payload["max_output_tokens"]))
        if len(seen_max) == 1:
            return httpx.Response(
                200,
                json={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output_text": '{"a":1,',
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": '{"a":1}',
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )

    request = OCRModelRunRequest(
        provider="openai",
        model_id="gpt-4o",
        system_prompt="sys",
        user_prompt="u",
        pdf_base64="ZmFrZQ==",
        filename="x.pdf",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_openai_ocr(
            request=request,
            api_key="fake",
            started_at=0.0,
            max_output_tokens=1200,
            client=client,
        )
    assert seen_max == [1200, 2400]
    assert result.text == '{"a":1}'


@pytest.mark.replay
async def test_openai_provider_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    request = OCRModelRunRequest(
        provider="openai",
        model_id="gpt-4o",
        system_prompt="s",
        user_prompt="u",
        pdf_base64="ZmFrZQ==",
        filename="x.pdf",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="bad request"):
            await run_openai_ocr(
                request=request,
                api_key="fake",
                started_at=0.0,
                max_output_tokens=1200,
                client=client,
            )
