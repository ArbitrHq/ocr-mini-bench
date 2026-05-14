"""Smoke test for `scripts/summarize_labels.py`. Runs the script against the
real dataset/manifest.json and asserts the output shape — one line per domain
plus the header."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_labels.py"


@pytest.mark.unit
def test_summarize_runs_and_emits_one_line_per_domain() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines[0] == "Label completeness summary"
    # One line per domain: invoices / receipts / logistics
    domain_lines = [ln for ln in lines[1:] if ":" in ln]
    assert {ln.split(":", 1)[0] for ln in domain_lines} == {"invoices", "receipts", "logistics"}
    # Each domain line contains its labeled/total fraction and a "critical" sub-tally.
    for ln in domain_lines:
        assert "keys labeled" in ln
        assert "critical" in ln
        assert "invalid files" in ln
