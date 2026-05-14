"""Shared provider helpers. Mirrors `src/ocr/provider-utils.ts`."""

from __future__ import annotations

import hashlib
import os
from typing import Literal

from ..config.model_catalog import ModelProvider
from .constants import RETRY_MAX_OUTPUT_TOKENS, RETRY_OUTPUT_MULTIPLIER
from .parsing import parse_first_json_object

OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


def get_provider_api_key(provider: ModelProvider) -> str | None:
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if provider == "mistral":
        return os.environ.get("MISTRAL_API_KEY")
    return os.environ.get("GOOGLE_API_KEY")


def build_prompt_cache_key(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    pdf_base64: str,
) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\n")
    h.update(system_prompt.encode("utf-8"))
    h.update(b"\n")
    h.update(user_prompt.encode("utf-8"))
    h.update(b"\n")
    h.update(pdf_base64.encode("utf-8"))
    return h.hexdigest()


def get_openai_reasoning_effort(model_id: str) -> OpenAIReasoningEffort | None:
    normalized = model_id.lower()

    # GPT-5.4 and GPT-5.5 families use a newer effort enum that does not accept "minimal".
    if normalized.startswith("gpt-5.4") or normalized.startswith("gpt-5.5"):
        return "low"

    if normalized.startswith("gpt-5") or normalized.startswith("o"):
        return "minimal"

    return None


def is_mistral_ocr_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return normalized == "mistral-ocr-latest" or normalized.startswith("mistral-ocr")


def is_likely_truncated_text(text: str) -> bool:
    trimmed = text.strip()
    if not trimmed:
        return False
    if parse_first_json_object(trimmed):
        return False

    if trimmed.endswith(",") or trimmed.endswith(":"):
        return True
    open_curly = trimmed.count("{")
    close_curly = trimmed.count("}")
    open_square = trimmed.count("[")
    close_square = trimmed.count("]")
    return open_curly > close_curly or open_square > close_square


def get_retry_max_output_tokens(current: int) -> int:
    return min(RETRY_MAX_OUTPUT_TOKENS, max(current + 400, current * RETRY_OUTPUT_MULTIPLIER))
