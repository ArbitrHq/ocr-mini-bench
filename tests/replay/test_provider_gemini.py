"""Offline replay test for the Gemini provider.

We don't have raw upstream HTTP bodies in the fixture set (the recorded
JSONs are post-parse), so we wrap each recorded `raw_output` into the
Gemini wire shape and drive the real provider code through an
`httpx.MockTransport`. This exercises the full
build-payload → POST → parse → metrics path without hitting the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ocr_mini_bench.ocr.providers.gemini import run_gemini_ocr
from ocr_mini_bench.ocr.types import OCRModelRunRequest

FIXTURES_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "responses" / "ts-2026-05-14"
)


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text())


def _build_gemini_body(fixture: dict[str, Any]) -> dict[str, Any]:
    runtime = fixture["runtime"]
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": fixture["raw_output"]}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": runtime["input_tokens"],
            "candidatesTokenCount": runtime["output_tokens"],
            "cachedContentTokenCount": runtime["cached_input_tokens"],
        },
    }


def _make_request(fixture: dict[str, Any]) -> OCRModelRunRequest:
    model = fixture["model"]
    return OCRModelRunRequest(
        provider=model["provider"] if model["provider"] in {"anthropic", "openai", "google", "mistral"} else "google",  # type: ignore[arg-type]
        model_id=model["model_id"],
        system_prompt="system",
        user_prompt="user",
        pdf_base64="ZmFrZQ==",  # "fake"
        filename=f"{fixture['document']['document_id']}.pdf",
    )


def _mock_transport(body: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # The Gemini provider uses cached payloads for some models; ignore
        # cachedContents creation calls and respond OK with no name so the
        # builder falls back to the inline path.
        if "cachedContents" in str(request.url):
            return httpx.Response(200, json={})
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.mark.replay
async def test_gemini_provider_parses_recorded_output() -> None:
    fixture = _load_fixture(
        "google_gemini-3.1-flash-lite-preview_invoices_invoices-auto-repair-shop-en_1.json"
    )
    body = _build_gemini_body(fixture)
    transport = _mock_transport(body)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await run_gemini_ocr(
            request=_make_request(fixture),
            api_key="fake-key",
            started_at=0.0,
            max_output_tokens=1200,
            client=client,
        )

    runtime = fixture["runtime"]
    assert result.text == fixture["raw_output"].strip()
    assert result.input_tokens == runtime["input_tokens"]
    assert result.output_tokens == runtime["output_tokens"]
    assert result.cached_input_tokens == runtime["cached_input_tokens"]
    assert result.cache_hit is (runtime["cached_input_tokens"] > 0)
    assert result.cache_write_tokens == 0


@pytest.mark.replay
async def test_gemini_provider_retries_on_max_tokens() -> None:
    """When `finishReason=MAX_TOKENS`, the provider retries with a larger ceiling."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "cachedContents" in str(request.url):
            return httpx.Response(200, json={})
        payload = json.loads(request.content)
        calls.append(int(payload["generationConfig"]["maxOutputTokens"]))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": '{"a": 1,'}]},  # truncated
                            "finishReason": "MAX_TOKENS",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "cachedContentTokenCount": 0,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": '{"a": 1}'}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 6,
                    "cachedContentTokenCount": 0,
                },
            },
        )

    request = OCRModelRunRequest(
        provider="google",
        model_id="gemini-3.1-flash-lite-preview",
        system_prompt="s",
        user_prompt="u",
        pdf_base64="ZmFrZQ==",
        filename="x.pdf",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_gemini_ocr(
            request=request,
            api_key="fake",
            started_at=0.0,
            max_output_tokens=1200,
            client=client,
        )
    assert len(calls) == 2
    assert calls[0] == 1200
    assert calls[1] == 2400  # get_retry_max_output_tokens(1200)
    assert result.text == '{"a": 1}'
    assert result.output_tokens == 6
