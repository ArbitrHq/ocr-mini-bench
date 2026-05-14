"""Tests for `scripts/checkpoint_status.py`. Spawn the script as a
subprocess so we exercise the actual CLI path users invoke."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "checkpoint_status.py").resolve()


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.unit
def test_missing_checkpoint_dir(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-dir"
    result = _run(f"--checkpoint-dir={missing}")
    assert "No checkpoint log found" in result.stdout
    assert str(missing) in result.stdout


@pytest.mark.unit
def test_missing_checkpoint_dir_json(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-dir"
    result = _run(f"--checkpoint-dir={missing}", "--json")
    payload = json.loads(result.stdout)
    assert payload["exists"] is False
    assert payload["checkpoint_dir"] == str(missing)
    assert payload["message"].startswith("No checkpoint log")


@pytest.mark.unit
def test_summary_with_mixed_success_and_failure(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    rows = [
        # task A: model gemini, success
        {
            "task_key": "google:gemini-3.1-flash-lite-preview::invoices::doc-1::1",
            "metrics": {
                "model_key": "google:gemini-3.1-flash-lite-preview",
                "error": None,
            },
        },
        # task B: model claude, failure
        {
            "task_key": "anthropic:claude-haiku-4-5::receipts::doc-2::1",
            "metrics": {
                "model_key": "anthropic:claude-haiku-4-5",
                "error": "RateLimitError",
            },
        },
        # task A again, latest wins (still success)
        {
            "task_key": "google:gemini-3.1-flash-lite-preview::invoices::doc-1::1",
            "metrics": {
                "model_key": "google:gemini-3.1-flash-lite-preview",
                "error": None,
            },
        },
    ]
    (ckpt / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (ckpt / "state.json").write_text(
        json.dumps({"mode": "fresh", "final": True, "updated_at": "2026-05-14T00:00:00Z"})
    )

    result = _run(f"--checkpoint-dir={ckpt}", "--json")
    payload = json.loads(result.stdout)
    assert payload["raw_lines"] == 3
    assert payload["latest_task_records"] == 2  # task A deduped
    assert payload["successful_records"] == 1
    assert payload["failed_records"] == 1
    assert payload["failure_rate_pct"] == 50.0
    by_model = {row["model_key"]: row for row in payload["by_model"]}
    assert by_model["google:gemini-3.1-flash-lite-preview"] == {
        "model_key": "google:gemini-3.1-flash-lite-preview",
        "successful": 1,
        "failed": 0,
    }
    assert by_model["anthropic:claude-haiku-4-5"]["failed"] == 1
    assert payload["top_errors"] == [{"count": 1, "error": "RateLimitError"}]


@pytest.mark.unit
def test_text_output_uses_lowercase_boolean(tmp_path: Path) -> None:
    """Matches TS verbatim: `Final: true` (not Python's `True`)."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "runs.jsonl").write_text(
        json.dumps(
            {
                "task_key": "google:m::d::doc::1",
                "metrics": {"model_key": "google:m", "error": None},
            }
        )
        + "\n"
    )
    (ckpt / "state.json").write_text(json.dumps({"mode": "fresh", "final": True}))

    result = _run(f"--checkpoint-dir={ckpt}")
    assert "Final: true" in result.stdout
    assert "Final: True" not in result.stdout


@pytest.mark.unit
def test_zero_failure_rate_serializes_as_int_not_float(tmp_path: Path) -> None:
    """Matches JS JSON.stringify: a zero rate is `0`, not `0.0`."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "runs.jsonl").write_text(
        json.dumps(
            {
                "task_key": "google:m::d::doc::1",
                "metrics": {"model_key": "google:m", "error": None},
            }
        )
        + "\n"
    )
    result = _run(f"--checkpoint-dir={ckpt}", "--json")
    # Look at the raw text so we catch float vs int serialization.
    assert '"failure_rate_pct": 0,' in result.stdout
    assert '"failure_rate_pct": 0.0' not in result.stdout
