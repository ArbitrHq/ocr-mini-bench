"""Tests for lib.errors — no TS counterpart, but the contract is exercised
by raw.jsonl rows whenever a provider call raises."""

from __future__ import annotations

import pytest

from ocr_mini_bench.lib.errors import to_error_message


@pytest.mark.unit
def test_returns_message_for_exception() -> None:
    assert to_error_message(ValueError("boom")) == "boom"


@pytest.mark.unit
def test_returns_fallback_for_non_exception() -> None:
    assert to_error_message("just a string") == "Model run failed."
    assert to_error_message(None) == "Model run failed."
    assert to_error_message({"shape": "dict"}) == "Model run failed."


@pytest.mark.unit
def test_returns_fallback_for_empty_exception() -> None:
    assert to_error_message(RuntimeError("")) == "Model run failed."
