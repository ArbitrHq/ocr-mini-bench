"""Tests for ocr.cost. The TS reference has no `.test.ts`, but cost values
flow into every artifact, so guard the math directly."""

from __future__ import annotations

import pytest

from ocr_mini_bench.ocr.cost import estimate_cost_usd


@pytest.mark.unit
def test_zero_for_unknown_model() -> None:
    assert estimate_cost_usd("nonexistent-model", 1000, 1000) == 0.0


@pytest.mark.unit
def test_basic_pricing_math() -> None:
    # gpt-5: 1.25 / 10.0 per million
    cost = estimate_cost_usd("gpt-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(1.25 + 10.0)


@pytest.mark.unit
def test_cached_tokens_use_cache_rate_when_available() -> None:
    # gemini-3.1-flash-lite-preview: input 0.25, cache 0.025, output 1.5
    # 1M cached input, 0 non-cached, 1M output
    cost = estimate_cost_usd("gemini-3.1-flash-lite-preview", 1_000_000, 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.025 + 1.5)


@pytest.mark.unit
def test_cached_falls_back_to_input_rate_when_no_cache_pricing() -> None:
    # gpt-5: no cache_input → cached tokens billed at regular input rate
    cost = estimate_cost_usd("gpt-5", 1_000_000, 0, 500_000)
    # 500k non-cached @ 1.25/M + 500k cached @ 1.25/M = 1.25
    assert cost == pytest.approx(1.25)


@pytest.mark.unit
def test_cached_clamped_to_total_input() -> None:
    # cached > input should clamp non_cached to 0
    cost = estimate_cost_usd("gpt-5", 100, 0, 1_000_000)
    # all "cached" though only 100 reported; non_cached clamped to 0
    assert cost == pytest.approx((1_000_000 / 1_000_000) * 1.25)
