"""Smoke test that the package imports — keeps `pytest -m unit` green on an empty Phase 1 tree."""

import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    import ocr_mini_bench

    assert ocr_mini_bench.__version__ == "0.1.0"
