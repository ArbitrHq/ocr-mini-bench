"""Score a raw model response against a ground-truth document.

Mirrors `src/benchmark/scoring/score.ts`. Produces both the lightweight
`ScoreResult` (used by the running benchmark) and the detailed variant
used by the debug snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...ocr.parsing import normalize_key_name, parse_first_json_object
from ..types import GroundTruthDocument
from .extraction import (
    build_value_by_key,
    count_found_keys,
    extract_pairs_from_parsed_json,
    extract_pairs_from_raw_text,
)
from .matching import is_match, matches_empty_expectation, pick_expected_values, should_score_key
from .types import (
    ExtractedPair,
    ExtractedValue,
    ScoreKeyComparison,
    ScoreResult,
    ScoreResultDetailed,
)


@dataclass
class _ExtractAndIndex:
    parsed: Any
    extracted_pairs_raw: list[ExtractedPair]
    value_by_key: dict[str, ExtractedValue]
    found_key_count: int


def _extract_and_index_values(raw_output: str, document: GroundTruthDocument) -> _ExtractAndIndex:
    parsed = parse_first_json_object(raw_output)
    pairs_from_json = extract_pairs_from_parsed_json(parsed, document.keys)
    extracted_pairs_raw = (
        pairs_from_json
        if pairs_from_json
        else extract_pairs_from_raw_text(raw_output, document.keys)
    )
    value_by_key = build_value_by_key(extracted_pairs_raw)
    found_key_count = count_found_keys(value_by_key, document.keys)
    return _ExtractAndIndex(
        parsed=parsed,
        extracted_pairs_raw=extracted_pairs_raw,
        value_by_key=value_by_key,
        found_key_count=found_key_count,
    )


def score_model_output(raw_output: str, document: GroundTruthDocument) -> ScoreResult:
    state = _extract_and_index_values(raw_output, document)

    field_total = 0
    field_correct = 0
    critical_total = 0
    critical_correct = 0

    for key in document.keys:
        normalized_key = normalize_key_name(key.name)
        extracted_record = state.value_by_key.get(normalized_key)
        extracted = extracted_record.value if extracted_record else ""
        mode = key.match if key.match is not None else "normalized_text"
        candidates = pick_expected_values(key)
        scored = should_score_key(key) or key.critical
        if not scored:
            continue
        if candidates:
            matched = any(is_match(extracted, expected, mode, key) for expected in candidates)
        else:
            matched = matches_empty_expectation(extracted_record)

        field_total += 1
        if matched:
            field_correct += 1
        if key.critical:
            critical_total += 1
            if matched:
                critical_correct += 1

    return ScoreResult(
        field_total=field_total,
        field_correct=field_correct,
        critical_total=critical_total,
        critical_correct=critical_correct,
        found_key_count=state.found_key_count,
        requested_key_count=len(document.keys),
    )


def score_model_output_detailed(
    raw_output: str, document: GroundTruthDocument
) -> ScoreResultDetailed:
    state = _extract_and_index_values(raw_output, document)

    field_total = 0
    field_correct = 0
    critical_total = 0
    critical_correct = 0
    key_comparisons: list[ScoreKeyComparison] = []

    for key in document.keys:
        normalized_key = normalize_key_name(key.name)
        extracted_record = state.value_by_key.get(normalized_key)
        extracted = extracted_record.value if extracted_record else ""
        mode = key.match if key.match is not None else "normalized_text"
        candidates = pick_expected_values(key)
        scored = should_score_key(key) or key.critical
        if not scored:
            matched = False
        elif candidates:
            matched = any(is_match(extracted, expected, mode, key) for expected in candidates)
        else:
            matched = matches_empty_expectation(extracted_record)

        key_comparisons.append(
            ScoreKeyComparison(
                key=key.name,
                critical=key.critical,
                scored=scored,
                expected_values=candidates,
                extracted_value=extracted,
                matched=matched,
                match_mode=mode,
            )
        )

        if not scored:
            continue

        field_total += 1
        if matched:
            field_correct += 1
        if key.critical:
            critical_total += 1
            if matched:
                critical_correct += 1

    extracted_pairs_out: list[dict[str, Any]] = [
        {"key": pair.key, "value": pair.value, "found": pair.found is not False}
        for pair in state.extracted_pairs_raw
    ]

    return ScoreResultDetailed(
        field_total=field_total,
        field_correct=field_correct,
        critical_total=critical_total,
        critical_correct=critical_correct,
        found_key_count=state.found_key_count,
        requested_key_count=len(document.keys),
        parsed_output=state.parsed,
        extracted_pairs=extracted_pairs_out,
        key_comparisons=key_comparisons,
    )
