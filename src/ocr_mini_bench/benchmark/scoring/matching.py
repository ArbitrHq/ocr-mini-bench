"""Per-key value matching. Mirrors `src/benchmark/scoring/matching.ts`.

Compares an extracted value against the ground truth for a single key, with
type-aware fallbacks (date, time, currency, company name, freight terms,
package type, weight unit, transport mode, numeric, compact alphanumeric).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from ...ocr.parsing import normalize_key_name
from ..types import GroundTruthKey, KeyMatchMode
from .text_normalization import (
    normalize_compact_alpha_numeric,
    normalize_numeric,
    normalize_text,
)
from .types import ExtractedValue

_YMD_RE = re.compile(r"(\d{4})[\/.\-](\d{1,2})[\/.\-](\d{1,2})")
_DMY_OR_MDY_RE = re.compile(r"(\d{1,2})[\/.\-](\d{1,2})[\/.\-](\d{2,4})")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap]\.?m\.?)?", re.IGNORECASE)
_PUNCT_RE = re.compile(r"\.")
_HAS_DIGIT_RE = re.compile(r"\d")


def _to_iso_date(year: int, month: int, day: int) -> str | None:
    if year < 100:
        year += 2000
    if month < 1 or month > 12:
        return None
    if day < 1 or day > 31:
        return None
    try:
        datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _parse_iso_datetime(value: str) -> str | None:
    """TS `Date.parse(value)` fallback. The TS code accepts anything Node's
    `Date.parse` recognizes; we restrict to ISO 8601 forms — that covers the
    cases the test suite exercises and avoids the locale-dependent fuzzy
    parsing that diverges between JS and Python."""
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    utc = parsed.astimezone(UTC)
    return f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}"


def _collect_date_candidates(raw_value: str) -> set[str]:
    value = raw_value.strip()
    out: set[str] = set()
    if not value:
        return out

    ymd = _YMD_RE.search(value)
    if ymd:
        iso = _to_iso_date(int(ymd.group(1)), int(ymd.group(2)), int(ymd.group(3)))
        if iso:
            out.add(iso)

    dmy_or_mdy = _DMY_OR_MDY_RE.search(value)
    if dmy_or_mdy:
        a = int(dmy_or_mdy.group(1))
        b = int(dmy_or_mdy.group(2))
        year = int(dmy_or_mdy.group(3))
        dmy = _to_iso_date(year, b, a)
        if dmy:
            out.add(dmy)
        mdy = _to_iso_date(year, a, b)
        if mdy:
            out.add(mdy)

    fallback = _parse_iso_datetime(value)
    if fallback:
        out.add(fallback)

    return out


@dataclass
class _ParsedTime:
    hours: int
    minutes: int
    seconds: int
    has_seconds: bool


def _parse_time(raw_value: str) -> _ParsedTime | None:
    value = raw_value.strip()
    if not value:
        return None
    match = _TIME_RE.search(value)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3)) if match.group(3) is not None else 0
    has_seconds = match.group(3) is not None
    ampm_raw = (match.group(4) or "").lower()
    ampm = _PUNCT_RE.sub("", ampm_raw)
    if minutes < 0 or minutes > 59 or seconds < 0 or seconds > 59:
        return None
    if ampm == "pm" and hours < 12:
        hours += 12
    elif ampm == "am" and hours == 12:
        hours = 0
    if hours < 0 or hours > 23:
        return None
    return _ParsedTime(hours=hours, minutes=minutes, seconds=seconds, has_seconds=has_seconds)


def _data_type(key: GroundTruthKey) -> str:
    return (key.data_type or "").lower()


def _is_date_like_key(key: GroundTruthKey) -> bool:
    type_ = _data_type(key)
    if "date" in type_ or "datetime" in type_ or "timestamp" in type_:
        return True
    return "date" in normalize_key_name(key.name)


def _is_time_like_key(key: GroundTruthKey) -> bool:
    type_ = _data_type(key)
    if "time" in type_ or "datetime" in type_ or "timestamp" in type_:
        return True
    return "time" in normalize_key_name(key.name)


def _is_currency_like_key(key: GroundTruthKey) -> bool:
    if "currency" in _data_type(key):
        return True
    return "currency" in normalize_key_name(key.name)


def _is_freight_terms_key(key: GroundTruthKey) -> bool:
    name = normalize_key_name(key.name)
    return name == "freight terms" or "freight terms" in name


def _is_package_type_key(key: GroundTruthKey) -> bool:
    return "package type" in normalize_key_name(key.name)


def _is_weight_unit_key(key: GroundTruthKey) -> bool:
    name = normalize_key_name(key.name)
    type_ = _data_type(key)
    return "weight unit" in name or ("weight" in name and "unit" in type_)


def _is_name_like_key(key: GroundTruthKey) -> bool:
    return "name" in normalize_key_name(key.name)


def _is_transport_mode_key(key: GroundTruthKey) -> bool:
    name = normalize_key_name(key.name)
    return name == "transport mode" or "transport mode" in name


_COMPANY_SUFFIX_TOKENS: frozenset[str] = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "llc",
        "ltd",
        "limited",
        "plc",
        "bv",
        "nv",
        "sa",
        "sarl",
        "gmbh",
        "ag",
        "aps",
        "as",
        "oy",
        "ab",
        "pte",
    }
)

_CARE_OF_RE = re.compile(r"\bcare\s+of\b", re.IGNORECASE)
_C_O_RE = re.compile(r"\bc\s*/?\s*o\b", re.IGNORECASE)
_A_S_RE = re.compile(r"\ba\s*[/.]?\s*s\b", re.IGNORECASE)


def _normalize_company_name(value: str) -> str:
    rewritten = _CARE_OF_RE.sub(" ", value)
    rewritten = _C_O_RE.sub(" ", rewritten)
    rewritten = _A_S_RE.sub("as", rewritten)
    tokens = [tok for tok in normalize_text(rewritten).split(" ") if tok]
    while tokens and tokens[-1] in _COMPANY_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


_FREIGHT_IGNORED: frozenset[str] = frozenset({"freight", "term", "terms"})


def _normalize_freight_terms(value: str) -> str:
    return " ".join(
        tok for tok in normalize_text(value).split(" ") if tok and tok not in _FREIGHT_IGNORED
    )


def _normalize_package_type_value(value: str) -> str:
    return " ".join(tok for tok in normalize_text(value).split(" ") if tok and tok != "stc")


def _normalize_weight_unit(value: str) -> str | None:
    compact = normalize_compact_alpha_numeric(value)
    if not compact:
        return None
    if compact in {"kg", "kgs", "kilogram", "kilograms"}:
        return "kg"
    if compact in {"g", "gram", "grams"}:
        return "g"
    if compact in {"lb", "lbs", "pound", "pounds"}:
        return "lb"
    if compact in {"ton", "tons", "tonne", "tonnes", "mt"}:
        return "tonne"
    return compact


_MISSING_VALUE_MARKERS: frozenset[str] = frozenset(
    {
        "na",
        "none",
        "null",
        "nil",
        "nill",
        "empty",
        "blank",
        "unknown",
        "nvd",
        "notavailable",
        "notapplicable",
        "notprovided",
        "notstated",
        "notspecified",
    }
)


def is_missing_like(value: str) -> bool:
    if not value.strip():
        return True
    compact = normalize_compact_alpha_numeric(value)
    if not compact:
        return True
    return compact in _MISSING_VALUE_MARKERS


# Order matters: longer prefixes first so "US$" wins over "$".
_CURRENCY_SYMBOL_TO_CODE: list[tuple[str, str]] = [
    ("US$", "USD"),
    ("USD$", "USD"),
    ("C$", "CAD"),
    ("CA$", "CAD"),
    ("A$", "AUD"),
    ("AU$", "AUD"),
    ("HK$", "HKD"),
    ("SG$", "SGD"),
    ("MX$", "MXN"),
    ("NZ$", "NZD"),
    ("R$", "BRL"),
    ("CHF", "CHF"),
    ("NOK", "NOK"),
    ("DKK", "DKK"),
    ("$", "USD"),
    ("€", "EUR"),
    ("£", "GBP"),
    ("¥", "JPY"),
    ("₹", "INR"),
    ("₩", "KRW"),
    ("₺", "TRY"),
    ("₽", "RUB"),
    ("kr", "SEK"),
    ("zł", "PLN"),
]

_KNOWN_CURRENCY_CODES: frozenset[str] = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CAD",
        "AUD",
        "CHF",
        "INR",
        "HKD",
        "SGD",
        "BRL",
        "KRW",
        "TRY",
        "RUB",
        "MXN",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "PLN",
    }
)

_THREE_LETTERS_RE = re.compile(r"^[A-Z]{3}$")


def _normalize_currency_value(value: str) -> str | None:
    trimmed = value.strip()
    if not trimmed:
        return None
    upper = trimmed.upper()
    if _THREE_LETTERS_RE.match(upper) and upper in _KNOWN_CURRENCY_CODES:
        return upper
    for symbol, code in _CURRENCY_SYMBOL_TO_CODE:
        if trimmed == symbol or upper == symbol.upper():
            return code
    for code in _KNOWN_CURRENCY_CODES:
        if code in upper:
            return code
    for symbol, code in _CURRENCY_SYMBOL_TO_CODE:
        if symbol in trimmed:
            return code
    return None


def _has_date_like_value(value: str) -> bool:
    return len(_collect_date_candidates(value)) > 0


def _has_time_like_value(value: str) -> bool:
    return _parse_time(value) is not None


def _date_equivalent(actual: str, expected: str) -> bool:
    actual_dates = _collect_date_candidates(actual)
    expected_dates = _collect_date_candidates(expected)
    if not actual_dates or not expected_dates:
        return False
    return bool(actual_dates & expected_dates)


def _time_equivalent(actual: str, expected: str) -> bool:
    actual_time = _parse_time(actual)
    expected_time = _parse_time(expected)
    if actual_time is None or expected_time is None:
        return False
    if actual_time.hours != expected_time.hours or actual_time.minutes != expected_time.minutes:
        return False
    if actual_time.has_seconds and expected_time.has_seconds:
        return actual_time.seconds == expected_time.seconds
    return True


def is_match(actual: str, expected: str, mode: KeyMatchMode, key: GroundTruthKey) -> bool:
    if not actual.strip() and expected.strip():
        return is_missing_like(expected)

    if mode == "exact":
        return actual.strip() == expected.strip()

    key_date_hint = _is_date_like_key(key)
    key_time_hint = _is_time_like_key(key)
    value_date_hint = _has_date_like_value(actual) or _has_date_like_value(expected)
    value_time_hint = _has_time_like_value(actual) or _has_time_like_value(expected)

    date_match = _date_equivalent(actual, expected)
    time_match = _time_equivalent(actual, expected)

    if (key_date_hint or value_date_hint) and date_match:
        actual_time_parsed = _parse_time(actual)
        expected_time_parsed = _parse_time(expected)
        if (
            actual_time_parsed is not None
            and expected_time_parsed is not None
            and (key_time_hint or value_time_hint)
        ):
            return time_match
        return True

    if (key_time_hint or value_time_hint) and time_match:
        return True

    if _is_currency_like_key(key):
        actual_currency = _normalize_currency_value(actual)
        expected_currency = _normalize_currency_value(expected)
        if actual_currency and expected_currency and actual_currency == expected_currency:
            return True

    if _is_freight_terms_key(key):
        norm_actual = _normalize_freight_terms(actual)
        norm_expected = _normalize_freight_terms(expected)
        if (
            norm_actual
            and norm_expected
            and (
                norm_actual == norm_expected
                or norm_expected in norm_actual
                or norm_actual in norm_expected
            )
        ):
            return True

    if _is_package_type_key(key):
        norm_actual = _normalize_package_type_value(actual)
        norm_expected = _normalize_package_type_value(expected)
        if norm_actual and norm_expected and norm_actual == norm_expected:
            return True
        compact_actual = normalize_compact_alpha_numeric(norm_actual)
        compact_expected = normalize_compact_alpha_numeric(norm_expected)
        if compact_actual and compact_expected and compact_actual == compact_expected:
            return True

    if _is_weight_unit_key(key):
        actual_unit = _normalize_weight_unit(actual)
        expected_unit = _normalize_weight_unit(expected)
        if actual_unit and expected_unit and actual_unit == expected_unit:
            return True

    if _is_name_like_key(key):
        name_actual = _normalize_company_name(actual)
        name_expected = _normalize_company_name(expected)
        if name_actual and name_expected and name_actual == name_expected:
            return True

    if _is_transport_mode_key(key):
        norm_actual = " ".join(
            tok for tok in normalize_text(actual).split(" ") if tok != "freight"
        )
        norm_expected = " ".join(
            tok for tok in normalize_text(expected).split(" ") if tok != "freight"
        )
        if (
            norm_actual
            and norm_expected
            and (
                norm_actual == norm_expected
                or norm_expected in norm_actual
                or norm_actual in norm_expected
            )
        ):
            return True

    compact_actual = normalize_compact_alpha_numeric(actual)
    compact_expected = normalize_compact_alpha_numeric(expected)
    if (
        compact_actual
        and compact_expected
        and compact_actual == compact_expected
        and _HAS_DIGIT_RE.search(compact_actual)
        and _HAS_DIGIT_RE.search(compact_expected)
    ):
        return True

    if mode == "contains":
        norm_actual = normalize_text(actual)
        norm_expected = normalize_text(expected)
        return norm_expected in norm_actual

    if mode == "numeric":
        actual_number = normalize_numeric(actual)
        expected_number = normalize_numeric(expected)
        if actual_number is None or expected_number is None:
            return False
        return abs(actual_number - expected_number) < 1e-6

    return normalize_text(actual) == normalize_text(expected)


def should_score_key(key: GroundTruthKey) -> bool:
    expected = key.expected
    if isinstance(expected, str):
        return len(expected.strip()) > 0
    if isinstance(expected, list):
        return any(value.strip() for value in expected)
    return False


def pick_expected_values(key: GroundTruthKey) -> list[str]:
    if isinstance(key.expected, str):
        return [key.expected]
    if isinstance(key.expected, list):
        return list(key.expected)
    return []


def matches_empty_expectation(extracted: ExtractedValue | None) -> bool:
    if extracted is None:
        return True
    if extracted.found is False:
        return True
    return is_missing_like(extracted.value)
