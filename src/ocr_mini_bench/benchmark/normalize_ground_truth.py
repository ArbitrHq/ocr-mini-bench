"""Convert raw ground-truth JSON files into `GroundTruthDocument` shape.

Mirrors `src/benchmark/normalize-ground-truth.ts`. The raw files in
`bench_documents/<domain>/ground_truth/<slug>.json` follow a nested
`{value, critical, type, match?, notes?}` schema; this module walks that
tree and produces the flattened `keys` list that the scorer consumes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .types import GroundTruthDocument, GroundTruthKey, KeyMatchMode

_VALID_MATCH_MODES: frozenset[str] = frozenset({"exact", "normalized_text", "contains", "numeric"})
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class _RawLeaf:
    leaf_name: str
    full_path: str
    critical: bool
    expected: str | list[str] | None
    data_type: str | None
    match: KeyMatchMode
    notes: str | None


def _is_value_node(value: Any) -> bool:
    return isinstance(value, dict) and "value" in value


def _infer_primitive_type(value: Any) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        # JS: `Number.isInteger(value)` — Python floats with no fractional
        # part are still floats here, matching TS behavior on `1.0`.
        return "float"
    if isinstance(value, str):
        trimmed = value.strip()
        if _ISO_DATE_RE.match(trimmed):
            return "date"
        return "string"
    return None


def _normalize_expected(value: Any) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        normalized = [str(entry if entry is not None else "").strip() for entry in value]
        filtered = [entry for entry in normalized if entry]
        return filtered if filtered else None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        # Mirror JS `String(value)`: ints don't get `.0`, floats use repr that
        # matches TS for the values we see in fixtures.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _infer_match_mode(type_hint: Any) -> KeyMatchMode:
    normalized = type_hint.strip().lower() if isinstance(type_hint, str) else ""
    if normalized in {"float", "integer", "number"}:
        return "numeric"
    return "normalized_text"


def _normalize_match_mode(mode: Any, fallback: KeyMatchMode) -> KeyMatchMode:
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized in _VALID_MATCH_MODES:
            return normalized  # type: ignore[return-value]
    return fallback


def _push_from_value_node(
    output: list[_RawLeaf], node: dict[str, Any], leaf_name: str, full_path_segments: list[str]
) -> None:
    fallback_match = _infer_match_mode(node.get("type"))
    data_type = node.get("type") if isinstance(node.get("type"), str) else None
    notes = node.get("notes") if isinstance(node.get("notes"), str) else None
    output.append(
        _RawLeaf(
            leaf_name=leaf_name,
            full_path="_".join(full_path_segments),
            critical=bool(node.get("critical")),
            expected=_normalize_expected(node.get("value")),
            data_type=data_type,
            match=_normalize_match_mode(node.get("match"), fallback_match),
            notes=notes,
        )
    )


def _push_primitive_leaf(
    output: list[_RawLeaf], value: Any, leaf_name: str, full_path_segments: list[str]
) -> None:
    primitive_type = _infer_primitive_type(value)
    fallback_match = _infer_match_mode(primitive_type)
    output.append(
        _RawLeaf(
            leaf_name=leaf_name,
            full_path="_".join(full_path_segments),
            critical=False,
            expected=_normalize_expected(value),
            data_type=primitive_type,
            match=fallback_match,
            notes=None,
        )
    )


def _collect_leaves(
    obj: dict[str, Any], path_segments: list[str], output: list[_RawLeaf]
) -> None:
    for key, value in obj.items():
        if _is_value_node(value):
            _push_from_value_node(output, value, key, [*path_segments, key])
            continue

        if isinstance(value, dict):
            _collect_leaves(value, [*path_segments, key], output)
            continue

        if isinstance(value, list):
            for index, entry in enumerate(value):
                index_path = [*path_segments, key, str(index + 1)]

                if _is_value_node(entry):
                    _push_from_value_node(output, entry, key, index_path)
                    continue

                if isinstance(entry, dict):
                    for row_key, row_value in entry.items():
                        row_path = [*index_path, row_key]

                        if _is_value_node(row_value):
                            _push_from_value_node(output, row_value, row_key, row_path)
                            continue
                        if isinstance(row_value, dict):
                            _collect_leaves(row_value, row_path, output)
                            continue
                        if isinstance(row_value, list):
                            for nested_index, nested_value in enumerate(row_value):
                                _push_primitive_leaf(
                                    output,
                                    nested_value,
                                    row_key,
                                    [*row_path, str(nested_index + 1)],
                                )
                            continue
                        _push_primitive_leaf(output, row_value, row_key, row_path)
                    continue

                _push_primitive_leaf(output, entry, key, index_path)


def _deduplicate_leaf_names(candidates: list[_RawLeaf]) -> list[GroundTruthKey]:
    leaf_counts: dict[str, int] = {}
    for candidate in candidates:
        leaf_counts[candidate.leaf_name] = leaf_counts.get(candidate.leaf_name, 0) + 1

    seen_names: set[str] = set()
    keys: list[GroundTruthKey] = []
    for candidate in candidates:
        preferred = (
            candidate.full_path if leaf_counts.get(candidate.leaf_name, 0) > 1 else candidate.leaf_name
        )
        final_name = preferred
        suffix = 2
        while final_name in seen_names:
            final_name = f"{preferred}_{suffix}"
            suffix += 1
        seen_names.add(final_name)
        if not final_name.strip():
            continue
        keys.append(
            GroundTruthKey(
                name=final_name,
                critical=candidate.critical,
                expected=candidate.expected,
                data_type=candidate.data_type,
                match=candidate.match,
                notes=candidate.notes,
            )
        )
    return keys


@dataclass
class GroundTruthFallback:
    document_id: str
    domain: str
    source_pdf: str


def normalize_ground_truth_document(
    raw_input: Any, fallback: GroundTruthFallback
) -> GroundTruthDocument:
    raw = raw_input if isinstance(raw_input, dict) else {}
    if isinstance(raw.get("keys"), list):
        raise ValueError(
            f"Unsupported legacy ground-truth schema in {fallback.document_id}. "
            "Use field objects with {{ value, critical, type }}."
        )

    leaves: list[_RawLeaf] = []
    _collect_leaves(raw, [], leaves)
    keys = _deduplicate_leaf_names(leaves)
    if not keys:
        raise ValueError(
            f"No comparable fields found in {fallback.document_id}. "
            "Expected objects containing {{ value, critical, type }}."
        )

    return GroundTruthDocument(
        schema_version="2.0",
        document_id=fallback.document_id,
        domain=fallback.domain,
        source_pdf=fallback.source_pdf,
        notes=None,
        keys=keys,
    )
