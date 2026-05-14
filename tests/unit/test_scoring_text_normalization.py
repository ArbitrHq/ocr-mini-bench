"""Port of src/benchmark/scoring/text-normalization.test.ts."""

from __future__ import annotations

import pytest

from ocr_mini_bench.benchmark.scoring.text_normalization import (
    normalize_compact_alpha_numeric,
    normalize_numeric,
    normalize_text,
    normalize_visual_confusables,
    safe_trim,
)


@pytest.mark.unit
class TestNormalizeVisualConfusables:
    def test_converts_greek_and_cyrillic_lookalikes_to_ascii(self) -> None:
        assert normalize_visual_confusables("Ηello") == "Hello"  # Greek Η
        assert normalize_visual_confusables("ΤEST") == "TEST"  # Greek Τ
        # Cyrillic А В Е К М Н О Р С Т У Х
        assert normalize_visual_confusables("АВЕКМНОРСТУХ") == "ABEKMHOPCTYX"

    def test_preserves_regular_ascii(self) -> None:
        assert normalize_visual_confusables("Hello World 123") == "Hello World 123"


@pytest.mark.unit
class TestNormalizeText:
    def test_lowercases_trims_and_normalizes_separators(self) -> None:
        assert normalize_text("  HELLO WORLD  ") == "hello world"
        assert normalize_text("hello-world") == "hello world"
        assert normalize_text("hello_world") == "hello world"
        assert normalize_text("hello    world") == "hello world"

    def test_removes_diacritics_and_normalizes_confusables(self) -> None:
        assert normalize_text("café") == "cafe"
        assert normalize_text("résumé") == "resume"
        assert normalize_text("ΤEST") == "test"  # Greek Τ

    def test_preserves_numbers(self) -> None:
        assert normalize_text("Invoice #123") == "invoice 123"
        assert normalize_text("PO-2024-001") == "po 2024 001"


@pytest.mark.unit
class TestNormalizeCompactAlphaNumeric:
    def test_removes_all_non_alphanumeric_and_lowercases(self) -> None:
        assert normalize_compact_alpha_numeric("INV-001") == "inv001"
        assert normalize_compact_alpha_numeric("PO #12345") == "po12345"
        assert normalize_compact_alpha_numeric("café123") == "cafe123"

    def test_handles_empty_or_punctuation_only_input(self) -> None:
        assert normalize_compact_alpha_numeric("") == ""
        assert normalize_compact_alpha_numeric("---") == ""


@pytest.mark.unit
class TestNormalizeNumeric:
    def test_parses_integers_and_decimals(self) -> None:
        assert normalize_numeric("123") == 123
        assert normalize_numeric("123.45") == 123.45
        assert normalize_numeric("-456") == -456

    def test_handles_comma_as_decimal_separator(self) -> None:
        assert normalize_numeric("123,45") == 123.45

    def test_strips_whitespace_and_currency_symbols(self) -> None:
        assert normalize_numeric("1 000") == 1000
        assert normalize_numeric("$100") == 100
        assert normalize_numeric("€99.99") == 99.99

    def test_returns_none_for_invalid_input(self) -> None:
        assert normalize_numeric("") is None
        assert normalize_numeric("abc") is None
        assert normalize_numeric("12.34.56") is None


@pytest.mark.unit
class TestSafeTrim:
    def test_trims_strings_and_converts_other_types(self) -> None:
        assert safe_trim("  hello  ") == "hello"
        assert safe_trim(123) == "123"
        assert safe_trim(None) == ""
