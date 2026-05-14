"""`ocr-bench` — end-to-end benchmark runner with checkpointing.

Mirrors `src/cli/run-benchmark.ts`. Same flag names and same on-disk
contract for `runs.jsonl`, `raw.runs.jsonl`, `state.json`, plus the
snapshot artifacts (`snapshot-<ts>.json`, `latest.json`, `latest.md`).

CLI uses typer but flags are exposed as their TS spellings (`--runs`,
`--parallel`, `--docs-per-domain`, `--provider-parallel`,
`--checkpoint-dir`, …) to preserve muscle memory across the port.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from dotenv import load_dotenv

from ..benchmark.dataset import load_benchmark_config, load_prepared_documents
from ..benchmark.orchestrator import (
    BenchmarkRunExecutionControls,
    BenchmarkTaskCompletionEvent,
    run_ocr_leaderboard_benchmark,
)
from ..benchmark.types import (
    BenchmarkDebugRun,
    BenchmarkRunOptions,
    SingleRunMetrics,
)
from ..config.paths import PATHS
from ..postprocess.io import timestamp_for_filename
from ..postprocess.raw_contract import from_checkpoint_record
from ..postprocess.types import (
    LegacyCheckpointDebug,
    LegacyCheckpointMetrics,
    LegacyCheckpointRecord,
)

CheckpointMode = Literal["fresh", "resume", "retry-failed"]

app = typer.Typer(
    help="Run the OCR benchmark (Python port).",
    add_completion=False,
    no_args_is_help=False,
)


def _now_iso_millis() -> str:
    moment = datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _round6(value: float) -> float:
    return round(value * 1_000_000) / 1_000_000


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    filtered = [item for item in parts if item]
    return filtered or None


def _normalize_options_for_compare(options: BenchmarkRunOptions) -> dict[str, Any]:
    # Key order and presence-rules must mirror the TS `normalizeOptionsForCompare`
    # exactly so the SHA-256 fingerprint matches. TS uses `JSON.stringify`, which
    # drops keys whose value is `undefined` (notably `domains` when not set);
    # we drop `None` values here to match that behavior.
    out: dict[str, Any] = {
        "runs_per_model": options.runs_per_model,
        "max_parallel_requests": options.max_parallel_requests,
        "max_documents_per_domain": options.max_documents_per_domain,
        "provider_parallel": options.provider_parallel,
        "domains": sorted(options.domains) if options.domains else None,
        "models": sorted(options.models) if options.models else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def _stable_dumps(value: Any) -> str:
    """Match JS JSON.stringify default formatting (no spaces, no trailing
    newline). Insertion order is preserved by both runtimes for plain dicts.
    """
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _compute_benchmark_fingerprint(options: BenchmarkRunOptions) -> str:
    hasher = hashlib.sha256()
    hasher.update(_stable_dumps(_normalize_options_for_compare(options)).encode("utf-8"))
    files = [
        PATHS.config.models,
        PATHS.dataset.manifest,
        PATHS.prompts.system,
        PATHS.prompts.user,
    ]
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        hasher.update(b"\n@@")
        hasher.update(file_path.name.encode("utf-8"))
        hasher.update(b"@@\n")
        hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def _estimate_planned_task_count(options: BenchmarkRunOptions) -> int | None:
    try:
        config = load_benchmark_config()
        runs_per_model = max(
            1,
            options.runs_per_model
            if options.runs_per_model is not None
            else config.default_runs_per_model,
        )
        selected_domains = [
            v.strip().lower() for v in (options.domains or []) if v and v.strip()
        ]
        cap = (
            options.max_documents_per_domain
            if isinstance(options.max_documents_per_domain, int)
            and options.max_documents_per_domain > 0
            else None
        )
        documents = load_prepared_documents(
            domains=selected_domains, max_documents_per_domain=cap
        )
        selected_filters = [
            v.strip().lower() for v in (options.models or []) if v and v.strip()
        ]
        if not selected_filters:
            selected_models = list(config.models)
        else:
            selected_models = [
                m
                for m in config.models
                if any(
                    needle == m.model_id.strip().lower()
                    or needle == m.model_label.strip().lower()
                    for needle in selected_filters
                )
            ]
        return len(selected_models) * len(documents) * runs_per_model
    except Exception:
        return None


def _load_latest_checkpoint_records(
    runs_log_path: Path,
) -> dict[str, LegacyCheckpointRecord]:
    latest: dict[str, LegacyCheckpointRecord] = {}
    if not runs_log_path.exists():
        return latest
    raw = runs_log_path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(parsed, dict)
            and parsed.get("task_key")
            and parsed.get("metrics")
            and parsed.get("debug")
        ):
            try:
                record = LegacyCheckpointRecord.model_validate(parsed)
            except Exception:
                continue
            latest[record.task_key] = record
    return latest


def _load_checkpoint_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    text = state_path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _summarize_checkpoint(records: dict[str, LegacyCheckpointRecord]) -> dict[str, float]:
    failed = 0
    cost_usd = 0.0
    for record in records.values():
        if record.metrics.error:
            failed += 1
        run_cost = record.metrics.total_cost_usd
        if run_cost is not None and run_cost > 0:
            cost_usd += run_cost
    return {"total": len(records), "failed": failed, "cost_usd": _round6(cost_usd)}


def _write_checkpoint_state(
    *,
    state_path: Path,
    mode: CheckpointMode,
    options: BenchmarkRunOptions,
    records: dict[str, LegacyCheckpointRecord],
    current_run_new_records: int,
    current_run_cost_usd: float,
    current_run_target_tasks: int | None,
    benchmark_fingerprint: str,
    final: bool = False,
) -> None:
    summary = _summarize_checkpoint(records)
    completed_this_run = current_run_new_records
    target = current_run_target_tasks
    avg_cost = current_run_cost_usd / completed_this_run if completed_this_run > 0 else 0.0
    estimated_final_cost = (
        avg_cost * target if isinstance(target, int) and target > 0 and completed_this_run > 0 else None
    )

    payload: dict[str, Any] = {
        "updated_at": _now_iso_millis(),
        "mode": mode,
        "options": options.model_dump(exclude_none=False, mode="json"),
        "records_total": int(summary["total"]),
        "records_failed": int(summary["failed"]),
        "records_successful": int(summary["total"] - summary["failed"]),
        "records_total_cost_usd": summary["cost_usd"],
        "current_run_new_records": current_run_new_records,
        "current_run_target_tasks": target,
        "current_run_remaining_tasks": (
            max(0, target - completed_this_run) if isinstance(target, int) else None
        ),
        "current_run_cost_usd": _round6(current_run_cost_usd),
        "current_run_avg_cost_usd": _round6(avg_cost),
        "current_run_estimated_final_cost_usd": (
            None if estimated_final_cost is None else _round6(estimated_final_cost)
        ),
        "final": bool(final),
        "benchmark_fingerprint": benchmark_fingerprint,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _metrics_to_legacy(metrics: SingleRunMetrics) -> LegacyCheckpointMetrics:
    return LegacyCheckpointMetrics.model_validate(metrics.model_dump(mode="json"))


def _debug_to_legacy(debug: BenchmarkDebugRun) -> LegacyCheckpointDebug:
    return LegacyCheckpointDebug.model_validate(debug.model_dump(mode="json"))


def _checkpoint_line(event: BenchmarkTaskCompletionEvent) -> LegacyCheckpointRecord:
    return LegacyCheckpointRecord(
        task_key=event.task_key,
        completed_at=_now_iso_millis(),
        metrics=_metrics_to_legacy(event.metrics),
        debug=_debug_to_legacy(event.debug),
    )


async def _run_benchmark(
    options: BenchmarkRunOptions,
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    mode: CheckpointMode,
) -> None:
    planned_total_tasks = _estimate_planned_task_count(options)
    benchmark_fingerprint = _compute_benchmark_fingerprint(options)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs_log_path = checkpoint_dir / "runs.jsonl"
    raw_checkpoint_log_path = checkpoint_dir / "raw.runs.jsonl"
    raw_checkpoint_latest_path = checkpoint_dir / "raw.jsonl"
    postprocess_dir = output_dir / "postprocess"
    raw_output_path = postprocess_dir / "raw.jsonl"
    state_path = checkpoint_dir / "state.json"

    if mode == "fresh":
        runs_log_path.write_text("", encoding="utf-8")
        raw_checkpoint_log_path.write_text("", encoding="utf-8")
        state_path.write_text("", encoding="utf-8")
    elif not runs_log_path.exists():
        raise RuntimeError(
            f"Checkpoint log not found at {runs_log_path}. "
            "Run once without --resume/--retry-failed first."
        )

    latest_records: dict[str, LegacyCheckpointRecord] = (
        {} if mode == "fresh" else _load_latest_checkpoint_records(runs_log_path)
    )
    existing_state = None if mode == "fresh" else _load_checkpoint_state(state_path)
    if mode != "fresh" and existing_state is not None:
        prior_options_raw = existing_state.get("options") or {}
        try:
            prior_options = BenchmarkRunOptions.model_validate(prior_options_raw)
        except Exception:
            prior_options = BenchmarkRunOptions()
        if _stable_dumps(_normalize_options_for_compare(prior_options)) != _stable_dumps(
            _normalize_options_for_compare(options)
        ):
            raise RuntimeError(
                "Checkpoint options mismatch. Previous options differ from current CLI "
                "options. Use matching options or a new --checkpoint-dir."
            )
        prior_fp = existing_state.get("benchmark_fingerprint")
        if prior_fp and prior_fp != benchmark_fingerprint:
            raise RuntimeError(
                "Checkpoint fingerprint mismatch (models/manifest/prompts changed). "
                "Use a new --checkpoint-dir for this run definition."
            )

    checkpoint_summary = _summarize_checkpoint(latest_records)
    if mode != "fresh":
        typer.echo(
            f"Loaded checkpoint: {int(checkpoint_summary['total'])} records "
            f"({int(checkpoint_summary['failed'])} failed) from {runs_log_path}"
        )

    failed_task_keys: set[str] = {
        task_key for task_key, record in latest_records.items() if record.metrics.error
    }

    initial_metrics: list[SingleRunMetrics] = []
    initial_debug_runs: list[BenchmarkDebugRun] = []
    skip_task_keys: set[str] | None = None
    only_task_keys: set[str] | None = None

    if mode == "resume":
        initial_metrics = [
            SingleRunMetrics.model_validate(r.metrics.model_dump(mode="json"))
            for r in latest_records.values()
        ]
        initial_debug_runs = [
            BenchmarkDebugRun.model_validate(r.debug.model_dump(mode="json"))
            for r in latest_records.values()
        ]
        skip_task_keys = set(latest_records.keys())
    elif mode == "retry-failed":
        initial_metrics = [
            SingleRunMetrics.model_validate(r.metrics.model_dump(mode="json"))
            for r in latest_records.values()
            if r.task_key not in failed_task_keys
        ]
        initial_debug_runs = [
            BenchmarkDebugRun.model_validate(r.debug.model_dump(mode="json"))
            for r in latest_records.values()
            if r.task_key not in failed_task_keys
        ]
        only_task_keys = set(failed_task_keys)
        typer.echo(f"Retry mode: scheduling {len(failed_task_keys)} previously failed tasks.")

    if mode == "fresh":
        current_run_target_tasks = planned_total_tasks
    elif mode == "resume":
        current_run_target_tasks = (
            max(0, planned_total_tasks - len(latest_records))
            if planned_total_tasks is not None
            else None
        )
    else:
        current_run_target_tasks = len(failed_task_keys)

    new_records_this_run = 0
    current_run_cost_usd = 0.0
    checkpoint_lock = asyncio.Lock()

    async def queue_checkpoint_write(event: BenchmarkTaskCompletionEvent) -> None:
        nonlocal new_records_this_run, current_run_cost_usd
        async with checkpoint_lock:
            line = _checkpoint_line(event)
            raw_line = from_checkpoint_record(line)
            with runs_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line.model_dump(mode="json")) + "\n")
            with raw_checkpoint_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(raw_line.model_dump(mode="json")) + "\n")
            latest_records[line.task_key] = line
            new_records_this_run += 1
            run_cost = line.metrics.total_cost_usd
            if run_cost is not None and run_cost > 0:
                current_run_cost_usd += run_cost
            _write_checkpoint_state(
                state_path=state_path,
                mode=mode,
                options=options,
                records=latest_records,
                current_run_new_records=new_records_this_run,
                current_run_cost_usd=current_run_cost_usd,
                current_run_target_tasks=current_run_target_tasks,
                benchmark_fingerprint=benchmark_fingerprint,
            )

    _write_checkpoint_state(
        state_path=state_path,
        mode=mode,
        options=options,
        records=latest_records,
        current_run_new_records=0,
        current_run_cost_usd=0.0,
        current_run_target_tasks=current_run_target_tasks,
        benchmark_fingerprint=benchmark_fingerprint,
    )

    typer.echo("Running OCR benchmark (standalone repository)...")
    if options.provider_parallel:
        lane_info = (
            f"provider lanes capped at {options.max_parallel_requests}"
            if options.max_parallel_requests is not None
            else "one lane per provider"
        )
        typer.echo(f"Provider-parallel mode enabled ({lane_info}).")

    controls = BenchmarkRunExecutionControls(
        initial_runs=initial_metrics,
        initial_debug_runs=initial_debug_runs,
        skip_task_keys=skip_task_keys,
        only_task_keys=only_task_keys,
        on_task_complete=queue_checkpoint_write,
    )
    snapshot = await run_ocr_leaderboard_benchmark(options, controls)

    final_raw_records = sorted(
        (from_checkpoint_record(r) for r in latest_records.values()),
        key=lambda r: r.task_key,
    )

    raw_checkpoint_latest_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_lines = [
        json.dumps(r.model_dump(mode="json"), ensure_ascii=False) for r in final_raw_records
    ]
    body = "\n".join(payload_lines)
    if body:
        body += "\n"
    raw_checkpoint_latest_path.write_text(body, encoding="utf-8")
    raw_output_path.write_text(body, encoding="utf-8")

    timestamp = timestamp_for_filename()
    snapshot_path = output_dir / f"snapshot-{timestamp}.json"
    snapshot_debug_path = output_dir / f"snapshot-{timestamp}.debug.json"
    latest_json_path = output_dir / "latest.json"
    latest_debug_path = output_dir / "latest.debug.json"
    latest_markdown_path = output_dir / "latest.md"

    debug_payload = (
        snapshot.debug.model_dump(mode="json")
        if snapshot.debug is not None
        else {"documents": [], "runs": []}
    )
    public_payload = snapshot.model_dump(mode="json")
    public_payload.pop("debug", None)

    snapshot_path.write_text(
        json.dumps(public_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    snapshot_debug_path.write_text(
        json.dumps(debug_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    latest_json_path.write_text(
        json.dumps(public_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    latest_debug_path.write_text(
        json.dumps(debug_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    latest_markdown_path.write_text(
        f"{public_payload.get('markdown_table') or ''}\n", encoding="utf-8"
    )

    typer.echo(f"Snapshot written: {snapshot_path}")
    typer.echo(f"Debug snapshot written: {snapshot_debug_path}")
    typer.echo(f"Latest artifact written: {latest_json_path}")
    typer.echo(f"Latest debug artifact written: {latest_debug_path}")
    typer.echo(f"Markdown table written: {latest_markdown_path}")
    leaderboard = public_payload.get("leaderboard")
    rows = len(leaderboard) if isinstance(leaderboard, list) else 0
    typer.echo(f"Rows: {rows}")
    typer.echo(f"Runs: {snapshot.run_count}")
    typer.echo(f"Checkpoint log: {runs_log_path}")
    typer.echo(f"Checkpoint raw log: {raw_checkpoint_log_path}")
    typer.echo(f"Checkpoint canonical raw: {raw_checkpoint_latest_path}")
    typer.echo(f"Canonical raw output: {raw_output_path}")
    typer.echo(f"Checkpoint state: {state_path}")
    typer.echo(f"New checkpoint records this run: {new_records_this_run}")
    if snapshot.cache_summary:
        typer.echo("Cache summary by model:")
        for row in snapshot.cache_summary:
            typer.echo(
                f"- {row.model_label}: hit {row.cache_hits}/{row.runs} "
                f"({row.cache_hit_rate_pct:.1f}%) | "
                f"cached_in_avg={row.cached_input_tokens_avg:.1f} | "
                f"cache_write_avg={row.cache_write_tokens_avg:.1f}"
            )

    _write_checkpoint_state(
        state_path=state_path,
        mode=mode,
        options=options,
        records=latest_records,
        current_run_new_records=new_records_this_run,
        current_run_cost_usd=current_run_cost_usd,
        current_run_target_tasks=current_run_target_tasks,
        benchmark_fingerprint=benchmark_fingerprint,
        final=True,
    )


@app.callback(invoke_without_command=True)
def main(
    runs: Annotated[
        int | None,
        typer.Option("--runs", help="Runs per model/document.", min=1),
    ] = None,
    parallel: Annotated[
        int | None,
        typer.Option("--parallel", help="Max parallel requests per provider lane.", min=1),
    ] = None,
    provider_parallel: Annotated[
        bool | None,
        typer.Option(
            "--provider-parallel/--no-provider-parallel",
            help="Enable provider-lane parallel mode.",
        ),
    ] = None,
    docs_per_domain: Annotated[
        int | None,
        typer.Option("--docs-per-domain", help="Limit documents per selected domain.", min=1),
    ] = None,
    domains: Annotated[
        str | None,
        typer.Option("--domains", help="Domain filter (csv)."),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option("--models", help="Model filter by model_id or model_label (csv)."),
    ] = None,
    output_dir: Annotated[
        str | None,
        typer.Option("--output-dir", help="Snapshot + postprocess output directory."),
    ] = None,
    checkpoint_dir: Annotated[
        str | None,
        typer.Option("--checkpoint-dir", help="Checkpoint directory."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume unfinished tasks from checkpoint."),
    ] = False,
    retry_failed: Annotated[
        bool,
        typer.Option("--retry-failed", help="Run only failed tasks from checkpoint."),
    ] = False,
) -> None:
    load_dotenv()

    if resume and retry_failed:
        typer.echo("Cannot combine --resume and --retry-failed.", err=True)
        raise typer.Exit(code=2)

    mode: CheckpointMode = "retry-failed" if retry_failed else ("resume" if resume else "fresh")

    options = BenchmarkRunOptions(
        runs_per_model=runs,
        max_parallel_requests=parallel,
        max_documents_per_domain=docs_per_domain,
        domains=_split_csv(domains.lower() if domains else None),
        models=_split_csv(models),
        provider_parallel=provider_parallel,
    )

    resolved_output_dir = (
        Path(output_dir).expanduser().resolve() if output_dir else PATHS.artifacts.root
    )
    resolved_checkpoint_dir = (
        Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else PATHS.artifacts.checkpoints
    )

    try:
        asyncio.run(
            _run_benchmark(
                options,
                output_dir=resolved_output_dir,
                checkpoint_dir=resolved_checkpoint_dir,
                mode=mode,
            )
        )
    except Exception as error:
        typer.echo(str(error), err=True)
        sys.exit(1)
