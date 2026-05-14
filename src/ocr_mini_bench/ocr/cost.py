"""Cost estimation. Mirrors `src/ocr/cost.ts`."""

from __future__ import annotations

from ..config.backend_config import load_pricing_registry_from_backend_config


def estimate_cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    pricing_registry = load_pricing_registry_from_backend_config()
    pricing = pricing_registry.get(model_id)
    if pricing is None:
        return 0.0
    cached = max(0, cached_input_tokens)
    non_cached = max(0, input_tokens - cached)
    cache_input_rate = pricing.cache_input if pricing.cache_input is not None else pricing.input
    return (
        (non_cached / 1_000_000) * pricing.input
        + (cached / 1_000_000) * cache_input_rate
        + (output_tokens / 1_000_000) * pricing.output
    )
