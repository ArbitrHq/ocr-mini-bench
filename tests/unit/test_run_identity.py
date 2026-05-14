"""Tests for run.identity — task/model key string format is part of the
contract: it appears in checkpoint and raw.jsonl rows."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ocr_mini_bench.benchmark.run.identity import (
    build_benchmark_id,
    build_benchmark_run_task_key,
    id_for_model,
    to_repo_relative_path,
)


@pytest.mark.unit
def test_id_for_model() -> None:
    assert id_for_model("google", "gemini-3.1-flash-lite-preview") == (
        "google:gemini-3.1-flash-lite-preview"
    )


@pytest.mark.unit
def test_task_key_format() -> None:
    # Golden run uses this exact format
    expected = "google:gemini-3.1-flash-lite-preview::invoices::invoices-auto-repair-shop-en::1"
    assert (
        build_benchmark_run_task_key(
            model_key="google:gemini-3.1-flash-lite-preview",
            domain="invoices",
            document_id="invoices-auto-repair-shop-en",
            run_number=1,
        )
        == expected
    )


@pytest.mark.unit
def test_benchmark_id_replaces_colons_and_dots() -> None:
    moment = datetime(2026, 5, 14, 7, 5, 59, 242_000, tzinfo=UTC)
    assert build_benchmark_id(moment) == "ocr-benchmark-2026-05-14T07-05-59-242Z"


@pytest.mark.unit
def test_to_repo_relative_path(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    assert to_repo_relative_path(nested, tmp_path) == "a/b/c.txt"
