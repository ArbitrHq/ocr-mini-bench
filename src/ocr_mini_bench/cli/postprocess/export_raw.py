"""`ocr-bench-export-raw`: rebuild `raw.jsonl` from checkpoint records.

Mirrors `src/cli/postprocess/export-raw.ts`. Sources (in priority order):
  --input-jsonl  : re-read canonical raw JSONL, de-duplicate by task_key
  --debug-file   : build from a `latest.debug.json`-style snapshot
  --checkpoint-dir: default — prefer `raw.jsonl` if present, fall back to `runs.jsonl`
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from ...config.paths import PATHS
from ...postprocess.io import (
    read_json_file,
    read_json_lines_file,
    timestamp_for_filename,
    write_json_file,
    write_json_lines_file,
)
from ...postprocess.raw_contract import (
    from_checkpoint_record,
    from_debug_run,
    to_task_key_from_debug,
)
from ...postprocess.types import (
    LegacyCheckpointDebug,
    LegacyCheckpointRecord,
    RawNormalizedRecord,
)

app = typer.Typer(
    help="Rebuild raw.jsonl from checkpoint records.",
    add_completion=False,
    no_args_is_help=False,
)


def _resolve_cwd(value: str) -> Path:
    return (Path.cwd() / value).resolve()


def _load_from_checkpoint(checkpoint_dir: Path) -> list[RawNormalizedRecord]:
    runs_path = checkpoint_dir / "runs.jsonl"
    rows = read_json_lines_file(runs_path)
    latest_by_task: dict[str, LegacyCheckpointRecord] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("task_key") and row.get("metrics") and row.get("debug"):
            record = LegacyCheckpointRecord.model_validate(row)
            latest_by_task[record.task_key] = record
    return [from_checkpoint_record(r) for r in latest_by_task.values()]


def _load_from_debug(debug_file: Path) -> list[RawNormalizedRecord]:
    payload = read_json_file(debug_file)
    runs_field = payload.get("runs") if isinstance(payload, dict) else None
    runs = runs_field if isinstance(runs_field, list) else []
    latest_by_task: dict[str, LegacyCheckpointDebug] = {}
    for raw in runs:
        run = LegacyCheckpointDebug.model_validate(raw)
        key = to_task_key_from_debug(run)
        latest_by_task[key] = run
    return [from_debug_run(r) for r in latest_by_task.values()]


def _load_from_canonical_raw(input_jsonl: Path) -> list[RawNormalizedRecord]:
    rows = read_json_lines_file(input_jsonl)
    latest_by_task: dict[str, RawNormalizedRecord] = {}
    for row in rows:
        if (
            isinstance(row, dict)
            and row.get("task_key")
            and row.get("model")
            and row.get("document")
            and row.get("runtime")
            and row.get("payload")
            and row.get("legacy_metrics")
        ):
            record = RawNormalizedRecord.model_validate(row)
            latest_by_task[record.task_key] = record
    return list(latest_by_task.values())


@app.callback(invoke_without_command=True)
def main(
    checkpoint_dir: Annotated[
        str | None,
        typer.Option("--checkpoint-dir", help="Checkpoint directory."),
    ] = None,
    debug_file: Annotated[
        str | None,
        typer.Option("--debug-file", help="Build from a debug snapshot file."),
    ] = None,
    input_jsonl: Annotated[
        str | None,
        typer.Option("--input-jsonl", help="Re-read canonical raw JSONL."),
    ] = None,
    output_jsonl: Annotated[
        str | None,
        typer.Option("--output-jsonl", help="Output raw JSONL."),
    ] = None,
    output_summary: Annotated[
        str | None,
        typer.Option("--output-summary", help="Output summary JSON."),
    ] = None,
) -> None:
    cp_dir: Path | None = _resolve_cwd(checkpoint_dir) if checkpoint_dir else None
    dbg_file: Path | None = _resolve_cwd(debug_file) if debug_file else None
    in_jsonl: Path | None = _resolve_cwd(input_jsonl) if input_jsonl else None
    out_jsonl: Path = _resolve_cwd(output_jsonl) if output_jsonl else PATHS.postprocess.raw_jsonl
    out_summary: Path = (
        _resolve_cwd(output_summary) if output_summary else PATHS.postprocess.raw_summary
    )

    if cp_dir is None and dbg_file is None and in_jsonl is None:
        cp_dir = PATHS.checkpoint.root

    checkpoint_raw_jsonl: Path | None = (cp_dir / "raw.jsonl") if cp_dir is not None else None

    records: list[RawNormalizedRecord]
    if in_jsonl is not None:
        records = _load_from_canonical_raw(in_jsonl)
    elif dbg_file is not None:
        records = _load_from_debug(dbg_file)
    elif checkpoint_raw_jsonl is not None and checkpoint_raw_jsonl.exists():
        try:
            records = _load_from_canonical_raw(checkpoint_raw_jsonl)
        except Exception:
            assert cp_dir is not None
            records = _load_from_checkpoint(cp_dir)
    else:
        assert cp_dir is not None
        records = _load_from_checkpoint(cp_dir)

    records.sort(key=lambda r: r.task_key)
    write_json_lines_file(out_jsonl, [r.model_dump(mode="json") for r in records])

    by_model: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    errored = 0
    for record in records:
        by_model[record.model.model_label] = by_model.get(record.model.model_label, 0) + 1
        by_domain[record.document.domain] = by_domain.get(record.document.domain, 0) + 1
        if record.runtime.error is not None:
            errored += 1

    now = datetime.now(UTC)
    generated_at = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    source: dict[str, Any]
    if in_jsonl is not None:
        source = {"input_jsonl": str(in_jsonl)}
    elif dbg_file is not None:
        source = {"debug_file": str(dbg_file)}
    elif checkpoint_raw_jsonl is not None and checkpoint_raw_jsonl.exists():
        source = {
            "checkpoint_raw_jsonl": str(checkpoint_raw_jsonl),
            "checkpoint_dir": str(cp_dir) if cp_dir else None,
        }
    else:
        source = {"checkpoint_dir": str(cp_dir) if cp_dir else None}

    summary = {
        "generated_at": generated_at,
        "source": source,
        "output_jsonl": str(out_jsonl),
        "schema_version": "1.0",
        "records": len(records),
        "errored_records": errored,
        "models": [
            {"model_label": label, "count": count}
            for label, count in sorted(by_model.items(), key=lambda kv: kv[0])
        ],
        "domains": [
            {"domain": d, "count": c}
            for d, c in sorted(by_domain.items(), key=lambda kv: kv[0])
        ],
        "build_id": f"raw-{timestamp_for_filename(now)}",
    }
    write_json_file(out_summary, summary)

    typer.echo(f"Raw records: {len(records)}")
    typer.echo(f"Errors: {errored}")
    typer.echo(f"JSONL: {out_jsonl}")
    typer.echo(f"Summary: {out_summary}")
