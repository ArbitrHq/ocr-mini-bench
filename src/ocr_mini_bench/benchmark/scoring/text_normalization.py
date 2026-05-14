"""Text normalization helpers used by the scoring layer.

Mirrors `src/benchmark/scoring/text-normalization.ts`. Logic and output must
match the TS reference exactly — these functions normalize ground-truth and
model-extracted strings before comparison, so any divergence directly
changes scoring.
"""

from __future__ import annotations

import re
import unicodedata

VISUAL_CONFUSABLE_TO_ASCII: dict[str, str] = {
    # Greek
    "Α": "A",  # Α
    "Β": "B",  # Β
    "Ε": "E",  # Ε
    "Ζ": "Z",  # Ζ
    "Η": "H",  # Η
    "Ι": "I",  # Ι
    "Κ": "K",  # Κ
    "Μ": "M",  # Μ
    "Ν": "N",  # Ν
    "Ο": "O",  # Ο
    "Ρ": "P",  # Ρ
    "Τ": "T",  # Τ
    "Υ": "Y",  # Υ
    "Χ": "X",  # Χ
    "α": "a",  # α
    "β": "b",  # β
    "ε": "e",  # ε
    "ι": "i",  # ι
    "κ": "k",  # κ
    "μ": "m",  # μ
    "ν": "n",  # ν
    "ο": "o",  # ο
    "ρ": "p",  # ρ
    "τ": "t",  # τ
    "υ": "y",  # υ
    "χ": "x",  # χ
    # Cyrillic
    "А": "A",  # А
    "В": "B",  # В
    "Е": "E",  # Е
    "К": "K",  # К
    "М": "M",  # М
    "Н": "H",  # Н
    "О": "O",  # О
    "Р": "P",  # Р
    "С": "C",  # С
    "Т": "T",  # Т
    "У": "Y",  # У
    "Х": "X",  # Х
    "а": "a",  # а
    "е": "e",  # е
    "к": "k",  # к
    "м": "m",  # м
    "о": "o",  # о
    "р": "p",  # р
    "с": "c",  # с
    "т": "t",  # т
    "у": "y",  # у
    "х": "x",  # х
}

_COMBINING_RE = re.compile(r"[̀-ͯ]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_STRIP_RE = re.compile(r"[^0-9.\-]")
_WHITESPACE_ALL_RE = re.compile(r"\s")


def normalize_visual_confusables(value: str) -> str:
    return "".join(VISUAL_CONFUSABLE_TO_ASCII.get(ch, ch) for ch in value)


def normalize_text(value: str) -> str:
    source = normalize_visual_confusables(value)
    lowered = source.strip().lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = _COMBINING_RE.sub("", decomposed)
    spaced = _NON_ALNUM_RE.sub(" ", stripped)
    collapsed = _WHITESPACE_RE.sub(" ", spaced)
    return collapsed.strip()


def normalize_compact_alpha_numeric(value: str) -> str:
    source = normalize_visual_confusables(value)
    lowered = source.strip().lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = _COMBINING_RE.sub("", decomposed)
    return _NON_ALNUM_RE.sub("", stripped)


def normalize_numeric(value: str) -> float | None:
    """Parse a possibly-formatted numeric string.

    Matches TS `Number(...)` semantics:
      - "" → null (here: None)
      - multiple dots after comma→dot translation → not finite → None
      - trailing/leading garbage stripped to digits/dot/minus only
    """
    no_ws = _WHITESPACE_ALL_RE.sub("", value)
    comma_to_dot = no_ws.replace(",", ".")
    normalized = _NUMERIC_STRIP_RE.sub("", comma_to_dot)
    if not normalized:
        return None
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):  # NaN/inf guard
        return None
    return parsed


def safe_trim(value: object) -> str:
    """TS `safeTrim`: stringify-then-trim, with null/undefined coerced to ''."""
    if value is None:
        return ""
    return str(value).strip()
