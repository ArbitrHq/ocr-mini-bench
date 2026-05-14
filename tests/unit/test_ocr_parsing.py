"""Port of src/ocr/parsing.test.ts."""

from __future__ import annotations

import pytest

from ocr_mini_bench.ocr.parsing import (
    fallback_extract_keys,
    normalize_key_name,
    parse_first_json_object,
)


@pytest.mark.unit
class TestParseFirstJsonObject:
    def test_parses_valid_json_directly(self) -> None:
        assert parse_first_json_object('{"key": "value"}') == {"key": "value"}

    def test_handles_whitespace_and_nesting(self) -> None:
        assert parse_first_json_object('  {"outer": {"inner": [1,2,3]}}  ') == {
            "outer": {"inner": [1, 2, 3]}
        }

    def test_extracts_json_from_fenced_code_blocks(self) -> None:
        assert parse_first_json_object('```json\n{"key": "value"}\n```') == {"key": "value"}
        assert parse_first_json_object('```\n{"key": "value"}\n```') == {"key": "value"}

    def test_extracts_json_embedded_in_text(self) -> None:
        assert parse_first_json_object('Here is the result: {"key": "value"} done') == {
            "key": "value"
        }

    def test_returns_none_for_invalid_or_missing(self) -> None:
        assert parse_first_json_object("") is None
        assert parse_first_json_object("not json") is None
        assert parse_first_json_object("{ invalid }") is None

    def test_parses_arrays(self) -> None:
        assert parse_first_json_object("[1, 2, 3]") == [1, 2, 3]


@pytest.mark.unit
class TestNormalizeKeyName:
    def test_lowercases_trims_normalizes_separators(self) -> None:
        assert normalize_key_name("  Invoice Number  ") == "invoice number"
        assert normalize_key_name("TOTAL_AMOUNT") == "total amount"
        assert normalize_key_name("invoice-number") == "invoice number"
        assert normalize_key_name("date/time") == "date time"

    def test_collapses_spaces_preserves_numbers(self) -> None:
        assert normalize_key_name("invoice   number") == "invoice number"
        assert normalize_key_name("PO-12345") == "po 12345"

    def test_handles_empty_input(self) -> None:
        assert normalize_key_name("") == ""
        assert normalize_key_name("   ") == ""


@pytest.mark.unit
class TestFallbackExtractKeys:
    def test_extracts_from_various_list_formats(self) -> None:
        markdown = "- Invoice Number: INV-001\n- Total Amount: $100.00"
        asterisk = "* Vendor Name: Acme Corp"
        plain = "Company: Test Inc"
        assert "Invoice Number" in fallback_extract_keys(markdown)
        assert "Total Amount" in fallback_extract_keys(markdown)
        assert "Vendor Name" in fallback_extract_keys(asterisk)
        assert "Company" in fallback_extract_keys(plain)

    def test_deduplicates_and_limits(self) -> None:
        duplicate = "- Name: John\n- Name: Jane"
        assert len([k for k in fallback_extract_keys(duplicate) if k == "Name"]) == 1
        many = "\n".join(f"Key{i}: value" for i in range(100))
        assert len(fallback_extract_keys(many)) <= 50

    def test_returns_empty_without_patterns(self) -> None:
        assert fallback_extract_keys("No colons here at all") == []
