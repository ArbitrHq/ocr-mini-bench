"""Runtime request/result types for OCR provider calls.

Mirrors `src/ocr/types.ts`. Kept as plain dataclasses (not pydantic) — these
are in-process call/result shapes, not serialized artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.model_catalog import ModelProvider


@dataclass(frozen=True)
class OCRModelRunRequest:
    provider: ModelProvider
    model_id: str
    system_prompt: str
    user_prompt: str
    pdf_base64: str
    filename: str
    max_output_tokens: int | None = None


@dataclass
class OCRModelRunResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cached_input_tokens: int
    cache_hit: bool
    cache_write_tokens: int
    total_cost_usd: float | None = None
    no_cache_cost_usd: float | None = None


@dataclass(frozen=True)
class ApprovedKey:
    name: str
    critical: bool
