"""Pricing registry and provider inference. Mirrors `src/config/backend-config.ts`.

Prices are USD per million tokens (input/output/cache_input).
"""

from __future__ import annotations

from dataclasses import dataclass

from .model_catalog import ModelProvider


@dataclass(frozen=True)
class ModelPricing:
    input: float
    output: float
    cache_input: float | None = None


PRICING_REGISTRY: dict[str, ModelPricing] = {
    # Anthropic
    "claude-opus-4-7": ModelPricing(input=5.0, output=25.0),
    "claude-opus-4-6": ModelPricing(input=5.0, output=25.0),
    "claude-opus-4-5": ModelPricing(input=5.0, output=25.0),
    "claude-opus-4-1": ModelPricing(input=15.0, output=75.0),
    "claude-sonnet-4-6": ModelPricing(input=3.0, output=15.0),
    "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0),
    "claude-sonnet-4": ModelPricing(input=3.0, output=15.0),
    "claude-haiku-4-5": ModelPricing(input=1.0, output=5.0),
    "claude-haiku-3": ModelPricing(input=0.25, output=1.25),
    # OpenAI
    "gpt-5-2": ModelPricing(input=1.75, output=14.0),
    "gpt-5": ModelPricing(input=1.25, output=10.0),
    "gpt-5.4": ModelPricing(input=2.5, output=15.0),
    "gpt-5-mini": ModelPricing(input=0.25, output=2.0),
    "gpt-5-nano": ModelPricing(input=0.05, output=0.4),
    "gpt-5.4-mini": ModelPricing(input=0.75, output=4.5),
    "gpt-5.4-nano": ModelPricing(input=0.2, output=1.25),
    "gpt-5.5": ModelPricing(input=5.0, output=30.0),
    "gpt-5-pro": ModelPricing(input=15.0, output=120.0),
    "gpt-4.1": ModelPricing(input=2.0, output=8.0),
    "gpt-4.1-mini": ModelPricing(input=0.4, output=1.6),
    "gpt-4.1-nano": ModelPricing(input=0.1, output=0.4),
    "gpt-4o": ModelPricing(input=2.5, output=10.0),
    "gpt-4o-mini": ModelPricing(input=0.15, output=0.6),
    "o3": ModelPricing(input=2.0, output=8.0),
    "o4-mini": ModelPricing(input=1.1, output=4.4),
    # Gemini
    "gemini-3.1-pro-preview": ModelPricing(input=2.0, output=12.0),
    "gemini-2.5-pro": ModelPricing(input=1.25, output=10.0),
    "gemini-2.5-flash": ModelPricing(input=0.3, output=2.5),
    "gemini-2.5-flash-lite": ModelPricing(input=0.1, output=0.4),
    "gemini-3.1-flash-lite-preview": ModelPricing(input=0.25, output=1.5, cache_input=0.025),
    "gemini-3-pro-preview": ModelPricing(input=2.0, output=12.0),
    "gemini-3-flash-preview": ModelPricing(input=0.5, output=3.0),
    # Mistral LLMs
    "mistral-small-latest": ModelPricing(input=0.1, output=0.3),
    "mistral-medium-latest": ModelPricing(input=0.4, output=2.0),
    "mistral-large-latest": ModelPricing(input=0.5, output=1.5),
    # Fallback
    "default": ModelPricing(input=0.0, output=0.0),
}


def load_pricing_registry_from_backend_config() -> dict[str, ModelPricing]:
    return PRICING_REGISTRY


_MISTRAL_PREFIXES = ("mistral", "ministral", "pixtral", "codestral", "devstral", "voxtral")


def infer_provider_from_model_id(model_id: str) -> ModelProvider:
    normalized = model_id.lower()
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith("gemini"):
        return "google"
    if normalized.startswith(_MISTRAL_PREFIXES):
        return "mistral"
    return "openai"
