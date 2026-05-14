"""Scoring-internal types. Mirrors `src/benchmark/scoring/types.ts`.

These are lightweight pydantic models for in-memory state during scoring.
The public on-disk `KeyComparison` / `ExtractedPair` types live in
`benchmark/types.py`; this file holds the scorer's working representations,
which match the TS interfaces but use camelCase-converted-to-snake_case names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import GroundTruthDocument, KeyMatchMode


@dataclass
class ExtractedPair:
    """Parsed `{key, value, found?}` from a provider response."""

    key: str
    value: str
    found: bool | None = None


@dataclass
class ExtractedValue:
    """Per-key extracted value used during scoring lookups."""

    value: str
    found: bool


@dataclass
class ScoreResult:
    field_total: int
    field_correct: int
    critical_total: int
    critical_correct: int
    found_key_count: int
    requested_key_count: int


@dataclass
class ScoreKeyComparison:
    key: str
    critical: bool
    scored: bool
    expected_values: list[str]
    extracted_value: str
    matched: bool
    match_mode: KeyMatchMode


@dataclass
class ScoreResultDetailed:
    field_total: int
    field_correct: int
    critical_total: int
    critical_correct: int
    found_key_count: int
    requested_key_count: int
    parsed_output: Any
    extracted_pairs: list[dict[str, Any]] = field(default_factory=list)
    key_comparisons: list[ScoreKeyComparison] = field(default_factory=list)


@dataclass
class ScoringContext:
    document: GroundTruthDocument
    value_by_key: dict[str, ExtractedValue]
    found_key_count: int
