"""Extract `{key, value, found}` triples from a parsed/raw model response.

Mirrors `src/benchmark/scoring/extraction.ts`.
"""

from __future__ import annotations

import re
from typing import Any

from ...ocr.parsing import normalize_key_name
from .text_normalization import safe_trim
from .types import ExtractedPair, ExtractedValue


def _js_string(value: Any) -> str:
    """Mirror JS `String(value)` for the value-coercion path.

    Most importantly, `String(20.0)` returns `"20"` because JS unifies the
    Number type; Python's `str(20.0)` returns `"20.0"`. Lossless decimals
    are stringified without the trailing `.0` so scoring output (and the
    downstream comparison.jsonl) matches the TS reference byte-for-byte.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

_PAIR_OBJECT_RE = re.compile(
    r'\{[^{}]*"key"\s*:\s*"([^"]+)"[^{}]*"value"\s*:\s*"([^"]*)"[^{}]*'
    r'(?:"found"\s*:\s*(true|false))?[^{}]*\}'
)
_LINE_KV_RE = re.compile(r"""^\s*["']?([^:"']{2,120})["']?\s*:\s*(.+)\s*$""")
_QUOTE_STRIP_RE = re.compile(r"""^["']|["']$""")


def _key_name(k: Any) -> str:
    """Read the canonical `name` from a `GroundTruthKey` model or a raw dict."""
    if hasattr(k, "name"):
        return str(k.name)
    return str(k["name"])


def _extract_pairs_from_object_map(
    obj: dict[str, Any], requested_keys: list[Any]
) -> list[ExtractedPair]:
    requested = {normalize_key_name(_key_name(k)) for k in requested_keys}
    out: list[ExtractedPair] = []
    for key, value in obj.items():
        if normalize_key_name(key) not in requested:
            continue
        out.append(ExtractedPair(key=key, value=_js_string(value)))
    return out


def extract_pairs_from_parsed_json(
    parsed: Any, requested_keys: list[Any]
) -> list[ExtractedPair]:
    if not isinstance(parsed, dict):
        return []
    out: list[ExtractedPair] = []
    pairs = parsed.get("pairs")
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            key_candidate = pair.get("key") or pair.get("name") or pair.get("field") or ""
            value_candidate = pair.get("value")
            if value_candidate is None:
                value_candidate = pair.get("extracted_value", "")
            found_candidate = pair.get("found")
            key = safe_trim(key_candidate)
            if not key:
                continue
            out.append(
                ExtractedPair(
                    key=key,
                    value=_js_string(value_candidate),
                    found=found_candidate if isinstance(found_candidate, bool) else None,
                )
            )

    values_candidate = parsed.get("values")
    if isinstance(values_candidate, dict):
        out.extend(_extract_pairs_from_object_map(values_candidate, requested_keys))

    if not out:
        out.extend(_extract_pairs_from_object_map(parsed, requested_keys))

    return out


def extract_pairs_from_raw_text(raw: str, requested_keys: list[Any]) -> list[ExtractedPair]:
    requested_map: dict[str, str] = {}
    for key in requested_keys:
        canonical_name = _key_name(key)
        requested_map[normalize_key_name(canonical_name)] = canonical_name

    collected: dict[str, ExtractedPair] = {}

    for match in _PAIR_OBJECT_RE.finditer(raw):
        raw_key = match.group(1) or ""
        raw_value = match.group(2) or ""
        raw_found = match.group(3)
        normalized = normalize_key_name(raw_key)
        canonical = requested_map.get(normalized)
        if canonical is None:
            continue
        collected[normalized] = ExtractedPair(
            key=canonical,
            value=raw_value,
            found=(raw_found == "true") if raw_found else None,
        )

    for line in re.split(r"\r?\n", raw):
        line_match = _LINE_KV_RE.match(line)
        if not line_match:
            continue
        key = normalize_key_name(line_match.group(1) or "")
        canonical = requested_map.get(key)
        if canonical is None:
            continue
        raw_value = safe_trim(line_match.group(2))
        value = _QUOTE_STRIP_RE.sub("", raw_value)
        if not value:
            continue
        collected[key] = ExtractedPair(key=canonical, value=value)

    return list(collected.values())


def build_value_by_key(extracted_pairs: list[ExtractedPair]) -> dict[str, ExtractedValue]:
    value_by_key: dict[str, ExtractedValue] = {}
    for pair in extracted_pairs:
        key = normalize_key_name(pair.key)
        current = value_by_key.get(key)
        found_flag = pair.found is not False
        value = safe_trim(pair.value)
        if current is None or (value and not current.value):
            value_by_key[key] = ExtractedValue(value=value, found=found_flag)
    return value_by_key


def count_found_keys(
    value_by_key: dict[str, ExtractedValue], requested_keys: list[Any]
) -> int:
    requested_set = {normalize_key_name(_key_name(k)) for k in requested_keys}
    return sum(
        1
        for key, value in value_by_key.items()
        if key in requested_set and value.found and value.value
    )
