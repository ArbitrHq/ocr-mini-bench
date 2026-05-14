"""Offline replay tests for the Mistral provider.

Covers both branches:
- `mistral-ocr-*` model → single `/v1/ocr` call with annotation schema, cost computed from `pages_processed`.
- Any other Mistral model → `/v1/ocr` for markdown, then `/v1/chat/completions` for extraction.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ocr_mini_bench.ocr.providers.mistral import run_mistral_ocr
from ocr_mini_bench.ocr.types import OCRModelRunRequest


@pytest.mark.replay
async def test_mistral_ocr_model_uses_annotation_and_computes_cost() -> None:
    body_captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body_captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "document_annotation": '{"pairs": [], "missing_keys": [],  "notes": ""}',
                "pages": [{"markdown": "page 1"}, {"markdown": "page 2"}],
                "usage_info": {"pages_processed": 2},
            },
        )

    request = OCRModelRunRequest(
        provider="mistral",
        model_id="mistral-ocr-latest",
        system_prompt="sys",
        user_prompt="user",
        pdf_base64="ZmFrZQ==",
        filename="x.pdf",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_mistral_ocr(
            request=request,
            api_key="fake",
            started_at=0.0,
            max_output_tokens=1200,
            client=client,
        )

    assert len(body_captured) == 1
    sent = body_captured[0]
    assert sent["model"] == "mistral-ocr-latest"
    assert sent["document_annotation_prompt"] == "sys\n\nuser"
    schema = sent["document_annotation_format"]["json_schema"]["schema"]  # type: ignore[index]
    assert "pairs" in schema["properties"]

    # Reparsed annotation is compact JSON.
    assert result.text == '{"pairs":[],"missing_keys":[],"notes":""}'
    # 2 pages * annotated rate (3/1000).
    assert result.total_cost_usd == pytest.approx(2 * 3 / 1000)
    assert result.no_cache_cost_usd == result.total_cost_usd
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.replay
async def test_mistral_non_ocr_model_does_ocr_then_chat() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), body))
        if str(request.url).endswith("/v1/ocr"):
            return httpx.Response(
                200,
                json={
                    "pages": [
                        {"markdown": "header"},
                        {"markdown": "body"},
                    ],
                    "usage_info": {"pages_processed": 2},
                },
            )
        # chat
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"pairs":[]}'}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 4},
            },
        )

    request = OCRModelRunRequest(
        provider="mistral",
        model_id="mistral-large-latest",
        system_prompt="sys",
        user_prompt="user",
        pdf_base64="ZmFrZQ==",
        filename="x.pdf",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_mistral_ocr(
            request=request,
            api_key="fake",
            started_at=0.0,
            max_output_tokens=1200,
            client=client,
        )

    assert len(calls) == 2
    ocr_url, ocr_body = calls[0]
    chat_url, chat_body = calls[1]
    assert ocr_url.endswith("/v1/ocr")
    assert ocr_body["model"] == "mistral-ocr-latest"  # always use latest for OCR step
    assert chat_url.endswith("/v1/chat/completions")
    assert chat_body["model"] == "mistral-large-latest"
    assert chat_body["max_tokens"] == 1200
    user_msg = chat_body["messages"][1]["content"]  # type: ignore[index]
    assert "Document OCR markdown:\nheader\n\nbody" in user_msg

    assert result.text == '{"pairs":[]}'
    assert result.input_tokens == 80
    assert result.output_tokens == 4


@pytest.mark.replay
async def test_mistral_ocr_model_falls_back_to_markdown_when_no_annotation() -> None:
    """If the annotation is missing, output a synthesized JSON object that includes the markdown."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                # no document_annotation
                "pages": [{"markdown": "fallback md"}],
                "usage_info": {"pages_processed": 1},
            },
        )

    request = OCRModelRunRequest(
        provider="mistral",
        model_id="mistral-ocr-latest",
        system_prompt="s",
        user_prompt="u",
        pdf_base64="ZmFrZQ==",
        filename="x.pdf",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_mistral_ocr(
            request=request,
            api_key="fake",
            started_at=0.0,
            max_output_tokens=1200,
            client=client,
        )

    parsed = json.loads(result.text)
    assert parsed["pairs"] == []
    assert parsed["missing_keys"] == []
    assert parsed["markdown"] == "fallback md"
    assert "markdown fallback" in parsed["notes"]
    # Non-annotated rate applies: 1 page * 2/1000.
    assert result.total_cost_usd == pytest.approx(2 / 1000)
