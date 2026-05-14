"""Parsing helpers for OCR responses. Mirrors `src/ocr/parsing.ts`."""

from __future__ import annotations

import json
import re
from typing import Any

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_FENCED_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_FALLBACK_KEY_RE = re.compile(r"^[-*]?\s*[\"']?([A-Za-z0-9_\-\s]{2,60})[\"']?\s*[:\-]")


def parse_first_json_object(raw: str) -> Any:
    """Best-effort JSON extraction from a provider response.

    Tries direct parse, fenced code block (```json ... ```), then the
    substring between the first `{` and last `}`. Returns None when nothing
    parses cleanly. Matches the TS reference's permissive contract.
    """
    trimmed = raw.strip()
    if not trimmed:
        return None
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass

    fenced = _FENCED_RE.search(trimmed)
    if fenced and fenced.group(1):
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start != -1 and end > start:
        candidate = trimmed[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def normalize_key_name(value: str) -> str:
    # NOTE: TS does not strip at the end — only at the start. Trailing
    # punctuation produces a trailing space. Preserve that quirk so the
    # normalized-key Map keys match across implementations.
    lowered = value.strip().lower()
    spaced = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", spaced)


def fallback_extract_keys(text: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _FALLBACK_KEY_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip() if match.group(1) else ""
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys[:50]
