"""Port of src/benchmark/scoring/matching.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from ocr_mini_bench.benchmark.scoring.matching import (
    is_match,
    is_missing_like,
    matches_empty_expectation,
    pick_expected_values,
    should_score_key,
)
from ocr_mini_bench.benchmark.scoring.types import ExtractedValue
from ocr_mini_bench.benchmark.types import GroundTruthKey


def make_key(**overrides: Any) -> GroundTruthKey:
    base = {
        "name": "test_key",
        "critical": False,
        "expected": "expected value",
        "match": "normalized_text",
    }
    base.update(overrides)
    return GroundTruthKey.model_validate(base)


@pytest.mark.unit
class TestIsMatchDate:
    def test_matches_various_date_formats_to_iso(self) -> None:
        key = make_key(name="invoice_date", data_type="date")
        assert is_match("2024-01-15", "2024-01-15", "normalized_text", key)
        assert is_match("01/15/2024", "2024-01-15", "normalized_text", key)
        assert is_match("15/01/2024", "2024-01-15", "normalized_text", key)
        assert is_match("15.01.2024", "2024-01-15", "normalized_text", key)

    def test_infers_date_type_from_key_name(self) -> None:
        key = make_key(name="due_date")  # no data_type, but name contains "date"
        assert is_match("2024-03-01", "2024-03-01", "normalized_text", key)


@pytest.mark.unit
class TestIsMatchCurrency:
    def test_matches_codes_and_normalizes_symbols(self) -> None:
        key = make_key(name="currency", data_type="currency")
        assert is_match("USD", "USD", "normalized_text", key)
        assert is_match("usd", "USD", "normalized_text", key)
        assert is_match("$", "USD", "normalized_text", key)
        assert is_match("€", "EUR", "normalized_text", key)
        assert is_match("USD 100.00", "USD", "normalized_text", key)


@pytest.mark.unit
class TestIsMatchCompanyName:
    def test_normalizes_suffixes_and_c_o(self) -> None:
        key = make_key(name="vendor_name")
        assert is_match("Acme Inc", "Acme Inc.", "normalized_text", key)
        assert is_match("Acme Corporation", "Acme Corp", "normalized_text", key)
        assert is_match("Acme LLC", "Acme", "normalized_text", key)
        assert is_match("c/o John Smith", "John Smith", "normalized_text", key)


@pytest.mark.unit
class TestIsMatchNumeric:
    def test_handles_decimal_separators_and_currency(self) -> None:
        key = make_key(data_type="float")
        assert is_match("100.5", "100.50", "numeric", key)
        assert is_match("100,50", "100.50", "numeric", key)
        assert is_match("$100.50", "100.50", "numeric", key)
        assert is_match("1 000.00", "1000", "numeric", key)


@pytest.mark.unit
class TestIsMatchModes:
    def test_exact_mode(self) -> None:
        key = make_key()
        assert is_match("test", "test", "exact", key)
        assert is_match(" test ", "test", "exact", key)  # trimmed
        assert not is_match("TEST", "test", "exact", key)

    def test_contains_mode(self) -> None:
        key = make_key()
        assert is_match("Invoice Number: INV-001", "INV-001", "contains", key)
        assert is_match("INVOICE NUMBER", "invoice number", "contains", key)

    def test_normalized_text_mode(self) -> None:
        key = make_key()
        assert is_match("Invoice Number", "invoice number", "normalized_text", key)
        assert is_match("INV-001", "INV001", "normalized_text", key)


@pytest.mark.unit
class TestIsMatchMissingValue:
    def test_empty_actual_against_missing_like_expected(self) -> None:
        key = make_key()
        assert is_match("", "N/A", "normalized_text", key)
        assert is_match("", "None", "normalized_text", key)
        assert not is_match("", "real value", "normalized_text", key)


@pytest.mark.unit
class TestIsMissingLike:
    def test_empty_and_markers(self) -> None:
        assert is_missing_like("")
        assert is_missing_like("   ")
        assert is_missing_like("N/A")
        assert is_missing_like("None")
        assert is_missing_like("null")
        assert is_missing_like("not available")

    def test_false_for_real_values(self) -> None:
        assert not is_missing_like("John Doe")
        assert not is_missing_like("100.00")


@pytest.mark.unit
class TestShouldScoreKey:
    def test_true_for_non_empty_expected(self) -> None:
        assert should_score_key(make_key(expected="value"))
        assert should_score_key(make_key(expected=["a", "b"]))

    def test_false_for_empty_or_null(self) -> None:
        assert not should_score_key(make_key(expected=""))
        assert not should_score_key(make_key(expected=[]))
        assert not should_score_key(make_key(expected=None))


@pytest.mark.unit
class TestPickExpectedValues:
    def test_normalizes_to_array(self) -> None:
        assert pick_expected_values(make_key(expected="value")) == ["value"]
        assert pick_expected_values(make_key(expected=["a", "b"])) == ["a", "b"]
        assert pick_expected_values(make_key(expected=None)) == []


@pytest.mark.unit
class TestMatchesEmptyExpectation:
    def test_true_for_missing_or_not_found(self) -> None:
        assert matches_empty_expectation(None)
        assert matches_empty_expectation(ExtractedValue(value="x", found=False))
        assert matches_empty_expectation(ExtractedValue(value="N/A", found=True))

    def test_false_for_real_extracted(self) -> None:
        assert not matches_empty_expectation(ExtractedValue(value="real", found=True))
