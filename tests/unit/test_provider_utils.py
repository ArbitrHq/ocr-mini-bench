"""Unit tests for `ocr.provider_utils`. Mirrors implicit TS expectations in
`src/ocr/provider-utils.ts` and `src/ocr/providers/*.test` style assertions.
"""

from __future__ import annotations

import os

import pytest

from ocr_mini_bench.ocr.provider_utils import (
    build_prompt_cache_key,
    get_openai_reasoning_effort,
    get_provider_api_key,
    get_retry_max_output_tokens,
    is_likely_truncated_text,
    is_mistral_ocr_model,
)


@pytest.mark.unit
def test_build_prompt_cache_key_is_stable_and_distinguishes_inputs() -> None:
    a = build_prompt_cache_key("m", "sys", "user", "pdf")
    b = build_prompt_cache_key("m", "sys", "user", "pdf")
    c = build_prompt_cache_key("m", "sys", "user", "pdf2")
    assert a == b
    assert a != c
    # sha256 hex
    assert len(a) == 64


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("gpt-5", "minimal"),
        ("gpt-5-pro", "minimal"),
        ("gpt-5.4", "low"),
        ("gpt-5.4-mini", "low"),
        ("gpt-5.5", "low"),
        ("o3", "minimal"),
        ("o4-mini", "minimal"),
        ("gpt-4.1", None),
        ("gpt-4o", None),
        ("claude-opus-4-1-20250805", None),
    ],
)
def test_get_openai_reasoning_effort(model_id: str, expected: str | None) -> None:
    assert get_openai_reasoning_effort(model_id) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("mistral-ocr-latest", True),
        ("mistral-ocr-2509", True),
        ("mistral-large-latest", False),
        ("mistral-small-latest", False),
        ("MISTRAL-OCR-LATEST", True),
    ],
)
def test_is_mistral_ocr_model(model_id: str, expected: bool) -> None:
    assert is_mistral_ocr_model(model_id) is expected


@pytest.mark.unit
def test_is_likely_truncated_text_handles_complete_json() -> None:
    assert is_likely_truncated_text('{"a":1}') is False
    assert is_likely_truncated_text("") is False


@pytest.mark.unit
def test_is_likely_truncated_text_detects_dangling_punctuation() -> None:
    assert is_likely_truncated_text('{"a": 1,') is True
    assert is_likely_truncated_text('{"a":') is True


@pytest.mark.unit
def test_is_likely_truncated_text_detects_unbalanced_brackets() -> None:
    assert is_likely_truncated_text('{"a": [1, 2') is True
    assert is_likely_truncated_text('{"a": 1') is True
    # parse_first_json_object can still recover this via best-effort substring
    # extraction (start '{', end '}'), so it is NOT flagged. That's deliberate.
    assert is_likely_truncated_text('garbage {"a": 1} garbage') is False


@pytest.mark.unit
def test_get_retry_max_output_tokens_doubles_until_ceiling() -> None:
    assert get_retry_max_output_tokens(1200) == 2400
    assert get_retry_max_output_tokens(100) == 500  # 100+400 dominates 100*2
    # Ceiling is RETRY_MAX_OUTPUT_TOKENS = 8000.
    assert get_retry_max_output_tokens(5000) == 8000
    assert get_retry_max_output_tokens(9000) == 8000


@pytest.mark.unit
def test_get_provider_api_key_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert get_provider_api_key("anthropic") == "a"
    assert get_provider_api_key("openai") == "o"
    assert get_provider_api_key("mistral") == "m"
    assert get_provider_api_key("google") == "g"


@pytest.mark.unit
def test_get_provider_api_key_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert get_provider_api_key("anthropic") is None
    assert get_provider_api_key("google") is None
    # Don't leak env values into other tests.
    assert "ANTHROPIC_API_KEY" not in os.environ
