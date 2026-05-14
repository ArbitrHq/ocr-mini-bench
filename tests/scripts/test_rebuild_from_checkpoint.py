"""Tests for `scripts/rebuild_from_checkpoint.py`.

Self-contained: builds a minimal checkpoint fixture in tmp_path, runs the
rebuild, verifies the output files and top-level shape."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "rebuild_from_checkpoint.py"


def _build_fixture_checkpoint(tmp_path: Path) -> Path:
    """Hand-built minimal checkpoint with 2 models x 2 docs x 2 runs = 8 rows."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    rows: list[dict] = []
    for model_idx, (provider, model_id, model_label, tier) in enumerate(
        [
            ("google", "gemini-test", "Gemini Test", "balanced"),
            ("anthropic", "claude-test", "Claude Test", "budget"),
        ]
    ):
        model_key = f"{provider}:{model_id}"
        for doc_id in ["invoices-doc-1", "invoices-doc-2"]:
            for run_idx in (1, 2):
                rows.append(
                    {
                        "task_key": f"{model_key}::invoices::{doc_id}::{run_idx}",
                        "metrics": {
                            "model_key": model_key,
                            "provider": provider,
                            "model_id": model_id,
                            "model_label": model_label,
                            "tier": tier,
                            "domain": "invoices",
                            "document_id": doc_id,
                            "field_total": 10,
                            "field_correct": 9 if run_idx == 1 else 10,
                            "critical_total": 5,
                            "critical_correct": 5,
                            "found_key_count": 10,
                            "requested_key_count": 10,
                            "latency_ms": 1000 + 100 * run_idx,
                            "input_tokens": 500,
                            "output_tokens": 300,
                            "total_cost_usd": 0.001 * (model_idx + 1),
                            "cache_hit": False,
                            "cached_input_tokens": 0,
                            "cache_write_tokens": 0,
                            "error": None,
                            "success": True,
                        },
                        "debug": {"raw_output": f"<mock for {doc_id} run {run_idx}>"},
                    }
                )

    (ckpt / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (ckpt / "state.json").write_text(
        json.dumps(
            {
                "mode": "fresh",
                "final": True,
                "options": {
                    "runs_per_model": 2,
                    "max_parallel_requests": 1,
                    "max_documents_per_domain": 2,
                    "provider_parallel": True,
                    "models": ["gemini-test", "claude-test"],
                },
            }
        )
    )
    return ckpt


@pytest.mark.unit
def test_rebuild_on_fixture_checkpoint_emits_expected_files(tmp_path: Path) -> None:
    ckpt = _build_fixture_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"--checkpoint-dir={ckpt}",
            f"--output-dir={out}",
            f"--repo-root={REPO_ROOT}",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Rebuilt snapshot from checkpoint records: 8" in result.stdout

    assert (out / "latest.json").exists()
    assert (out / "latest.debug.json").exists()
    assert (out / "latest.md").exists()
    snapshots = sorted(out.glob("snapshot-*.json"))
    assert any(not p.name.endswith(".debug.json") for p in snapshots)
    assert any(p.name.endswith(".debug.json") for p in snapshots)

    payload = json.loads((out / "latest.json").read_text())
    assert payload["run_count"] == 8
    assert {row["model_key"] for row in payload["leaderboard"]} == {
        "google:gemini-test",
        "anthropic:claude-test",
    }
    assert payload["markdown_table"].startswith("| Rank | Model | Provider | Tier |")
    assert all(row["pass_at_2_pct"] is not None for row in payload["leaderboard"])
    assert all(row["pass_at_3_pct"] is None for row in payload["leaderboard"])
