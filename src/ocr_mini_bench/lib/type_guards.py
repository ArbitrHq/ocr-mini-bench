"""Type-guard utilities for safely narrowing values parsed from JSON.

Mirrors `src/lib/type-guards.ts`. Used when interpreting provider responses
or other untrusted payloads where pydantic models aren't a fit.
"""

from __future__ import annotations

import math
from typing import Any, TypeGuard


def is_record(value: object) -> TypeGuard[dict[str, Any]]:
    """True for plain dicts. Lists, None, and primitives are not records."""
    return isinstance(value, dict)


def is_string(value: object) -> TypeGuard[str]:
    return isinstance(value, str)


def is_number(value: object) -> TypeGuard[float]:
    """True for finite numbers. Matches TS `Number.isFinite` semantics — rejects
    NaN, ±Infinity, and bool (Python's quirk where `True` is also `int`)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def is_array(value: object) -> TypeGuard[list[Any]]:
    return isinstance(value, list)


def has_property(obj: object, key: str) -> bool:
    return is_record(obj) and key in obj


def get_string_property(obj: dict[str, Any], key: str) -> str | None:
    value = obj.get(key)
    return value if is_string(value) else None


def get_number_property(obj: dict[str, Any], key: str) -> float | None:
    value = obj.get(key)
    return float(value) if is_number(value) else None


def get_record_property(obj: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = obj.get(key)
    return value if is_record(value) else None


def get_array_property(obj: dict[str, Any], key: str) -> list[Any] | None:
    value = obj.get(key)
    return value if is_array(value) else None
