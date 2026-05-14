"""Port of src/lib/type-guards.test.ts."""

from __future__ import annotations

import math

import pytest

from ocr_mini_bench.lib.type_guards import (
    get_array_property,
    get_number_property,
    get_record_property,
    get_string_property,
    has_property,
    is_array,
    is_number,
    is_record,
    is_string,
)


@pytest.mark.unit
class TestIsRecord:
    def test_returns_true_for_plain_objects_false_otherwise(self) -> None:
        assert is_record({"a": 1}) is True
        assert is_record([]) is False
        assert is_record(None) is False
        assert is_record("string") is False


@pytest.mark.unit
class TestPrimitiveGuards:
    def test_is_string(self) -> None:
        assert is_string("hello") is True
        assert is_string(123) is False

    def test_is_number(self) -> None:
        assert is_number(42) is True
        assert is_number(math.nan) is False
        assert is_number(math.inf) is False
        # TS Number.isFinite rejects non-numbers; Python booleans are also
        # ints, so we explicitly reject them to match TS behavior on `true`.
        assert is_number(True) is False

    def test_is_array(self) -> None:
        assert is_array([1, 2]) is True
        assert is_array({}) is False


@pytest.mark.unit
class TestHasProperty:
    def test_checks_existence(self) -> None:
        assert has_property({"value": 1}, "value") is True
        # key exists even if value is None
        assert has_property({"a": None}, "a") is True
        assert has_property({}, "missing") is False
        assert has_property(None, "value") is False


@pytest.mark.unit
class TestGetStringProperty:
    def test_extracts_strings(self) -> None:
        assert get_string_property({"name": "test"}, "name") == "test"
        assert get_string_property({"num": 123}, "num") is None
        assert get_string_property({}, "missing") is None


@pytest.mark.unit
class TestGetNumberProperty:
    def test_extracts_finite_numbers(self) -> None:
        assert get_number_property({"count": 42}, "count") == 42
        assert get_number_property({"zero": 0}, "zero") == 0
        assert get_number_property({"bad": math.nan}, "bad") is None
        assert get_number_property({"str": "123"}, "str") is None


@pytest.mark.unit
class TestRecordAndArrayProperty:
    def test_extracts_records_rejects_arrays(self) -> None:
        nested = {"inner": 1}
        assert get_record_property({"nested": nested}, "nested") is nested
        assert get_record_property({"arr": []}, "arr") is None

    def test_extracts_arrays_rejects_records(self) -> None:
        items = [1, 2, 3]
        assert get_array_property({"items": items}, "items") is items
        assert get_array_property({"obj": {}}, "obj") is None
