"""Smoke test for `scripts/validate_dataset.py`. The real dataset is the
canonical fixture — validation must succeed against it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_dataset.py"


@pytest.mark.unit
def test_validate_passes_on_real_dataset() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"validate exited {result.returncode}: {result.stdout}\n{result.stderr}"
    assert "Dataset validation passed." in result.stdout
    assert "Manifest:" in result.stdout
    assert "Documents:" in result.stdout
    assert "Comparable keys:" in result.stdout
