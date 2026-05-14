"""`ocr-bench-compare`: score raw OCR runs against ground truth.

Mirrors `src/cli/postprocess/compare.ts`. Reads canonical `raw.jsonl`,
re-scores each record against the live ground-truth (via `loadPreparedDocuments`),
and writes `comparison.jsonl` + `comparison.summary.json`.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from ...benchmark.dataset import load_prepared_documents
from ...benchmark.scoring.score import score_model_output_detailed
from ...benchmark.types import KeyComparison
from ...config.paths import PATHS
from ...postprocess.io import (
    file_exists,
    read_json_lines_file,
    write_json_file,
    write_json_lines_file,
)
from ...postprocess.types import (
    ComparisonBlock,
    ComparisonRecord,
    RawNormalizedRecord,
)

app = typer.Typer(
    help="Score raw OCR runs against ground truth.",
    add_completion=False,
    no_args_is_help=False,
)


def _resolve_cwd(value: str) -> Path:
    return (Path.cwd() / value).resolve()


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (part / total) * 100


def _round(value: float, decimals: int = 4) -> float:
    """JS `Math.round` semantics — round-half-away-from-zero. Reuses the
    same rule as `benchmark.run.math.round_half_away_from_zero` but kept
    inline to match the TS file's private helper."""
    precision: int = 10**decimals
    scaled = value * precision
    rounded = int(math.floor(scaled + 0.5) if scaled >= 0 else -math.floor(-scaled + 0.5))
    return rounded / precision


def _doc_key(domain: str, document_id: str) -> str:
    return f"{domain.lower()}::{document_id}"


@app.callback(invoke_without_command=True)
def main(
    raw_jsonl: Annotated[
        str | None,
        typer.Option("--raw-jsonl", help="Canonical raw JSONL input."),
    ] = None,
    output_jsonl: Annotated[
        str | None,
        typer.Option("--output-jsonl", help="Comparison JSONL output."),
    ] = None,
    output_summary: Annotated[
        str | None,
        typer.Option("--output-summary", help="Comparison summary JSON output."),
    ] = None,
) -> None:
    raw_path = _resolve_cwd(raw_jsonl) if raw_jsonl else PATHS.postprocess.raw_jsonl
    out_jsonl = _resolve_cwd(output_jsonl) if output_jsonl else PATHS.postprocess.comparison_jsonl
    out_summary = (
        _resolve_cwd(output_summary)
        if output_summary
        else PATHS.postprocess.comparison_summary
    )

    if not file_exists(raw_path):
        legacy = raw_path.parent / "raw.normalized.jsonl"
        if file_exists(legacy):
            raw_path = legacy

    rows = read_json_lines_file(raw_path)
    raw_records = [RawNormalizedRecord.model_validate(row) for row in rows]
    if not raw_records:
        raise typer.Exit(code=1)

    selected_domains = sorted({r.document.domain.lower() for r in raw_records})
    prepared = load_prepared_documents(domains=selected_domains)
    doc_by_key = {
        _doc_key(doc.domain, doc.document_id): doc for doc in prepared
    }

    comparisons: list[ComparisonRecord] = []
    missing_ground_truth = 0
    scored_runs = 0
    run_errors = 0
    success_changed = 0

    for record in raw_records:
        key = _doc_key(record.document.domain, record.document.document_id)
        prepared_doc = doc_by_key.get(key)

        if prepared_doc is None:
            missing_ground_truth += 1
            comparisons.append(
                ComparisonRecord(
                    schema_version="1.0",
                    task_key=record.task_key,
                    completed_at=record.completed_at,
                    model=record.model,
                    document=record.document,
                    runtime=record.runtime,
                    legacy_metrics=record.legacy_metrics,
                    comparison=None,
                )
            )
            continue

        if record.runtime.error is not None:
            run_errors += 1
            comparisons.append(
                ComparisonRecord(
                    schema_version="1.0",
                    task_key=record.task_key,
                    completed_at=record.completed_at,
                    model=record.model,
                    document=record.document,
                    runtime=record.runtime,
                    legacy_metrics=record.legacy_metrics,
                    comparison=None,
                )
            )
            continue

        score = score_model_output_detailed(record.payload.raw_output, prepared_doc.ground_truth)
        field_pass_pct = (
            _pct(score.field_correct, score.field_total) if score.field_total > 0 else 0.0
        )
        critical_pass_pct = (
            _pct(score.critical_correct, score.critical_total) if score.critical_total > 0 else 0.0
        )
        keys_found_pct = (
            _pct(score.found_key_count, score.requested_key_count)
            if score.requested_key_count > 0
            else 0.0
        )
        success = score.critical_total > 0 and score.critical_correct == score.critical_total

        if success != record.legacy_metrics.success:
            success_changed += 1

        scored_runs += 1

        comparisons.append(
            ComparisonRecord(
                schema_version="1.0",
                task_key=record.task_key,
                completed_at=record.completed_at,
                model=record.model,
                document=record.document,
                runtime=record.runtime,
                legacy_metrics=record.legacy_metrics,
                comparison=ComparisonBlock(
                    field_total=score.field_total,
                    field_correct=score.field_correct,
                    field_pass_pct=_round(field_pass_pct, 2),
                    critical_total=score.critical_total,
                    critical_correct=score.critical_correct,
                    critical_pass_pct=_round(critical_pass_pct, 2),
                    found_key_count=score.found_key_count,
                    requested_key_count=score.requested_key_count,
                    keys_found_pct=_round(keys_found_pct, 2),
                    success=success,
                    key_comparisons=[
                        KeyComparison(
                            key=row.key,
                            critical=row.critical,
                            scored=row.scored,
                            expected_values=row.expected_values,
                            extracted_value=row.extracted_value,
                            matched=row.matched,
                            match_mode=row.match_mode,
                        )
                        for row in score.key_comparisons
                    ],
                ),
            )
        )

    comparisons.sort(key=lambda c: c.task_key)

    serialized_rows = [c.model_dump(mode="json") for c in comparisons]
    write_json_lines_file(out_jsonl, serialized_rows)

    now = datetime.now(UTC)
    generated_at = (
        now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    )
    summary = {
        "generated_at": generated_at,
        "input_raw_jsonl": str(raw_path),
        "output_comparison_jsonl": str(out_jsonl),
        "records_total": len(comparisons),
        "records_scored": scored_runs,
        "records_with_runtime_error": run_errors,
        "records_missing_ground_truth": missing_ground_truth,
        "success_changed_vs_legacy": success_changed,
    }
    write_json_file(out_summary, summary)

    typer.echo(f"Comparison records: {len(comparisons)}")
    typer.echo(f"Scored: {scored_runs}")
    typer.echo(f"Runtime errors: {run_errors}")
    typer.echo(f"Missing GT: {missing_ground_truth}")
    typer.echo(f"Success changed vs legacy: {success_changed}")
    typer.echo(f"JSONL: {out_jsonl}")
    typer.echo(f"Summary: {out_summary}")
