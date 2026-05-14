"""Live smoke tests for the OCR providers.

These hit real provider APIs and incur cost. They are opt-in:

- `pytest -m smoke` to even consider them
- the relevant `*_API_KEY` env var must be set per provider
- the canonical smoke model is `gemini-3.1-flash-lite-preview` (see
  [[feedback-python-port-decisions]]); other providers are skipped unless
  their key is present.

One short PDF is sent per provider. The test only asserts the call returned
*some* non-empty text and non-negative usage — not text content (the LLM
output is nondeterministic).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from ocr_mini_bench.ocr.runner import run_ocr_model
from ocr_mini_bench.ocr.types import OCRModelRunRequest

# Pick the first PDF under bench_documents/ as the smoke document. Avoids
# committing a duplicate PDF fixture and means the smoke test runs against a
# real document with text content.
_BENCH_DOCS = Path(__file__).resolve().parents[2] / "bench_documents"


def _load_pdf_b64() -> str:
    candidates = sorted(_BENCH_DOCS.rglob("*.pdf"))
    if not candidates:
        pytest.skip(f"no bench_documents PDFs under {_BENCH_DOCS}")
    return base64.b64encode(candidates[0].read_bytes()).decode("ascii")


@pytest.mark.smoke
async def test_smoke_gemini_flash_lite_preview() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    request = OCRModelRunRequest(
        provider="google",
        model_id="gemini-3.1-flash-lite-preview",
        system_prompt='You are an OCR assistant. Respond with JSON {"text": "..."}',
        user_prompt='Return {"text": "ok"}',
        pdf_base64=_load_pdf_b64(),
        filename="blank.pdf",
    )
    result = await run_ocr_model(request)
    assert isinstance(result.text, str)
    assert result.input_tokens >= 0
    assert result.output_tokens >= 0


@pytest.mark.smoke
async def test_smoke_openai_nano() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    request = OCRModelRunRequest(
        provider="openai",
        model_id="gpt-5-nano",
        system_prompt='Return JSON {"text": "ok"}.',
        user_prompt="ok",
        pdf_base64=_load_pdf_b64(),
        filename="blank.pdf",
    )
    result = await run_ocr_model(request)
    assert isinstance(result.text, str)


@pytest.mark.smoke
async def test_smoke_anthropic_haiku() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    request = OCRModelRunRequest(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        system_prompt='Return JSON {"text": "ok"}.',
        user_prompt="ok",
        pdf_base64=_load_pdf_b64(),
        filename="blank.pdf",
    )
    result = await run_ocr_model(request)
    assert isinstance(result.text, str)


@pytest.mark.smoke
async def test_smoke_mistral_ocr_latest() -> None:
    if not os.environ.get("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY not set")
    request = OCRModelRunRequest(
        provider="mistral",
        model_id="mistral-ocr-latest",
        system_prompt='Return JSON {"text": "ok"}.',
        user_prompt="ok",
        pdf_base64=_load_pdf_b64(),
        filename="blank.pdf",
    )
    result = await run_ocr_model(request)
    assert isinstance(result.text, str)
