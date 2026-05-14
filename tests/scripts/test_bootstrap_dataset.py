"""Tests for `scripts/bootstrap_dataset.py`.

The bootstrap script reads the real bench_documents/ tree and writes a manifest
JSON. We verify structural parity against the existing canonical
dataset/manifest.json (which was produced by the TS reference)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "bootstrap_dataset.py"
REAL_MANIFEST = REPO_ROOT / "dataset" / "manifest.json"


def _without_timestamp(manifest: dict) -> dict:
    return {k: v for k, v in manifest.items() if k != "generated_at"}


@pytest.mark.unit
def test_python_bootstrap_matches_existing_manifest(tmp_path: Path) -> None:
    """The Python bootstrap, run against the real bench_documents/, should
    produce a manifest structurally identical to the one already on disk."""
    out = tmp_path / "manifest.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), f"--output={out}"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=True,
    )
    assert "Manifest written" in result.stdout
    py_manifest = json.loads(out.read_text())
    real_manifest = json.loads(REAL_MANIFEST.read_text())
    assert _without_timestamp(py_manifest) == _without_timestamp(real_manifest)


@pytest.mark.unit
def test_generated_at_uses_millisecond_precision(tmp_path: Path) -> None:
    """Match JS `new Date().toISOString()`: three-digit milliseconds, not
    Python's default six-digit microseconds."""
    out = tmp_path / "manifest.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), f"--output={out}"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=True,
    )
    manifest = json.loads(out.read_text())
    ts = manifest["generated_at"]
    # YYYY-MM-DDTHH:MM:SS.mmmZ
    assert len(ts) == 24
    assert ts.endswith("Z")
    assert ts[-5] == "."
    assert ts[-4:-1].isdigit()
