"""End-to-end CLI test: invoke the typer app with a mocked provider
runner and assert the on-disk checkpoint files match the documented
contract (runs.jsonl, raw.runs.jsonl, raw.jsonl, state.json, snapshot
files, latest.* files).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ocr_mini_bench.benchmark import orchestrator
from ocr_mini_bench.cli.run_benchmark import app
from ocr_mini_bench.ocr.types import OCRModelRunResult


async def _fake_run_ocr_model(request, *, client=None):  # type: ignore[no-untyped-def]
    keys: list[str] = []
    for line in request.user_prompt.splitlines():
        if line.startswith("- "):
            key = line[2:].split(" (")[0].strip()
            if key:
                keys.append(key)
    payload = {key: "" for key in keys}
    return OCRModelRunResult(
        text=json.dumps(payload),
        input_tokens=5,
        output_tokens=7,
        latency_ms=11,
        cached_input_tokens=0,
        cache_hit=False,
        cache_write_tokens=0,
        total_cost_usd=0.0001,
    )


@pytest.mark.replay
def test_cli_fresh_run_writes_checkpoint_and_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "run_ocr_model", _fake_run_ocr_model)
    output_dir = tmp_path / "artifacts"
    checkpoint_dir = tmp_path / "checkpoints"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--runs",
            "1",
            "--parallel",
            "1",
            "--docs-per-domain",
            "1",
            "--domains",
            "invoices",
            "--models",
            "gemini-3.1-flash-lite-preview",
            "--output-dir",
            str(output_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    runs_log = checkpoint_dir / "runs.jsonl"
    raw_runs_log = checkpoint_dir / "raw.runs.jsonl"
    raw_canonical = checkpoint_dir / "raw.jsonl"
    state_path = checkpoint_dir / "state.json"

    assert runs_log.exists() and runs_log.read_text(encoding="utf-8").strip()
    assert raw_runs_log.exists() and raw_runs_log.read_text(encoding="utf-8").strip()
    assert raw_canonical.exists()
    assert state_path.exists()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["mode"] == "fresh"
    assert state["records_total"] == 1
    assert state["records_failed"] == 0
    assert state["final"] is True
    assert isinstance(state["benchmark_fingerprint"], str)
    assert len(state["benchmark_fingerprint"]) == 64

    # latest.json + latest.md emitted
    latest_json = output_dir / "latest.json"
    latest_md = output_dir / "latest.md"
    assert latest_json.exists()
    assert latest_md.exists()
    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    assert payload["run_count"] == 1
    assert "debug" not in payload
    assert payload["options"]["runs_per_model"] == 1


@pytest.mark.replay
def test_cli_resume_skips_completed_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "run_ocr_model", _fake_run_ocr_model)
    output_dir = tmp_path / "artifacts"
    checkpoint_dir = tmp_path / "checkpoints"

    runner = CliRunner()
    common = [
        "--runs",
        "1",
        "--docs-per-domain",
        "1",
        "--domains",
        "invoices",
        "--models",
        "gemini-3.1-flash-lite-preview",
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    first = runner.invoke(app, common)
    assert first.exit_code == 0, first.output

    # Now resume — the orchestrator should see all task_keys in skip_task_keys
    # and run zero new tasks.
    call_count = {"n": 0}

    async def counting_run(request, *, client=None):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        return await _fake_run_ocr_model(request, client=client)

    monkeypatch.setattr(orchestrator, "run_ocr_model", counting_run)
    second = runner.invoke(app, [*common, "--resume"])
    assert second.exit_code == 0, second.output
    assert call_count["n"] == 0

    state = json.loads((checkpoint_dir / "state.json").read_text(encoding="utf-8"))
    assert state["mode"] == "resume"
    assert state["current_run_new_records"] == 0


@pytest.mark.replay
def test_cli_rejects_options_change_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "run_ocr_model", _fake_run_ocr_model)
    output_dir = tmp_path / "artifacts"
    checkpoint_dir = tmp_path / "checkpoints"

    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "--runs",
            "1",
            "--docs-per-domain",
            "1",
            "--domains",
            "invoices",
            "--models",
            "gemini-3.1-flash-lite-preview",
            "--output-dir",
            str(output_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )
    assert first.exit_code == 0

    second = runner.invoke(
        app,
        [
            "--runs",
            "2",  # changed
            "--docs-per-domain",
            "1",
            "--domains",
            "invoices",
            "--models",
            "gemini-3.1-flash-lite-preview",
            "--output-dir",
            str(output_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--resume",
        ],
    )
    assert second.exit_code != 0
    assert "Checkpoint options mismatch" in second.output


@pytest.mark.replay
def test_cli_resume_combined_with_retry_failed_is_rejected(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--resume",
            "--retry-failed",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
        ],
    )
    assert result.exit_code == 2
    assert "Cannot combine" in result.output
