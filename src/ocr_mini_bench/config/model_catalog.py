"""Provider/model catalog. Mirrors `src/config/model-catalog.ts`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelProvider = Literal["anthropic", "openai", "google", "mistral"]


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str


MODEL_CATALOG: dict[ModelProvider, list[ModelOption]] = {
    "anthropic": [
        ModelOption(id="claude-opus-4-1-20250805", label="Claude Opus 4.1"),
        ModelOption(id="claude-opus-4-20250514", label="Claude Opus 4"),
        ModelOption(id="claude-sonnet-4-20250514", label="Claude Sonnet 4"),
        ModelOption(id="claude-haiku-4-5-20251001", label="Claude Haiku 4.5"),
        ModelOption(id="claude-3-7-sonnet-20250219", label="Claude Sonnet 3.7"),
        ModelOption(id="claude-3-5-haiku-20241022", label="Claude Haiku 3.5"),
    ],
    "openai": [
        ModelOption(id="gpt-5", label="GPT-5"),
        ModelOption(id="gpt-5.4", label="GPT-5.4"),
        ModelOption(id="gpt-5.5", label="GPT-5.5"),
        ModelOption(id="gpt-5-mini", label="GPT-5 mini"),
        ModelOption(id="gpt-5-nano", label="GPT-5 nano"),
        ModelOption(id="gpt-5.4-mini", label="GPT-5.4 mini"),
        ModelOption(id="gpt-5.4-nano", label="GPT-5.4 nano"),
        ModelOption(id="gpt-5-pro", label="GPT-5 pro"),
        ModelOption(id="gpt-4.1", label="GPT-4.1"),
        ModelOption(id="gpt-4.1-mini", label="GPT-4.1 mini"),
        ModelOption(id="gpt-4.1-nano", label="GPT-4.1 nano"),
        ModelOption(id="gpt-4o", label="GPT-4o"),
        ModelOption(id="gpt-4o-mini", label="GPT-4o mini"),
        ModelOption(id="o3", label="o3"),
        ModelOption(id="o4-mini", label="o4-mini"),
    ],
    "google": [
        ModelOption(id="gemini-2.5-pro", label="Gemini 2.5 Pro"),
        ModelOption(id="gemini-2.5-flash", label="Gemini 2.5 Flash"),
        ModelOption(id="gemini-2.5-flash-lite", label="Gemini 2.5 Flash-Lite"),
        ModelOption(id="gemini-3.1-flash-lite-preview", label="Gemini 3.1 Flash-Lite"),
        ModelOption(id="gemini-3-pro-preview", label="Gemini 3 Pro"),
        ModelOption(id="gemini-3-flash-preview", label="Gemini 3 Flash"),
    ],
    "mistral": [
        ModelOption(id="mistral-ocr-latest", label="Mistral OCR (Latest)"),
        ModelOption(id="mistral-large-latest", label="Mistral Large (Latest)"),
        ModelOption(id="mistral-medium-latest", label="Mistral Medium (Latest)"),
        ModelOption(id="mistral-small-latest", label="Mistral Small (Latest)"),
    ],
}


def default_model_for_provider(provider: ModelProvider) -> str:
    options = MODEL_CATALOG[provider]
    return options[0].id if options else ""
