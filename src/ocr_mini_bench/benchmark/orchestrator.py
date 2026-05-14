"""Top-level benchmark orchestration. Mirrors `src/benchmark/run.ts`.

Wires together dataset loading, prompt building, the provider runner, the
async scheduling primitives (pool / provider-lanes), the scoring layer,
the in-memory leaderboard aggregation, and the snapshot assembly.

This file is the integration point: everything below it is provider- or
postprocess-specific; everything above it (`cli/run_benchmark.py`) just
handles flags, checkpoints, and output paths.
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from ..config.paths import PATHS, REPO_ROOT
from ..lib.errors import to_error_message
from ..ocr.cost import estimate_cost_usd
from ..ocr.runner import run_ocr_model
from ..ocr.types import OCRModelRunRequest
from .dataset import (
    load_benchmark_config,
    load_prepared_documents,
    summarize_dataset,
)
from .run.aggregation import aggregate_rows, build_cache_summary, build_markdown_table
from .run.identity import (
    build_benchmark_id,
    build_benchmark_run_task_key,
    id_for_model,
    to_repo_relative_path,
)
from .run.math import pct
from .run.math import round_half_away_from_zero as _round
from .run.pool import run_in_pool
from .run.progress import ProgressEvent, create_progress_reporter
from .run.provider_lanes import run_by_provider_lanes
from .scoring import score
from .scoring.prompt_builders import (
    build_benchmark_system_prompt,
    build_benchmark_user_prompt,
)
from .types import (
    BenchmarkDebugDocument,
    BenchmarkDebugRun,
    BenchmarkDebugSnapshot,
    BenchmarkModelConfig,
    BenchmarkRunOptions,
    BenchmarkSnapshot,
    BenchmarkSnapshotOptions,
    DomainLeaderboard,
    PreparedBenchmarkDocument,
    SingleRunMetrics,
)

BENCHMARK_MAX_OUTPUT_TOKENS = 4000


@dataclass
class BenchmarkTaskCompletionEvent:
    task_key: str
    metrics: SingleRunMetrics
    debug: BenchmarkDebugRun


@dataclass
class BenchmarkRunExecutionControls:
    initial_runs: list[SingleRunMetrics] = field(default_factory=list)
    initial_debug_runs: list[BenchmarkDebugRun] = field(default_factory=list)
    skip_task_keys: set[str] | None = None
    only_task_keys: set[str] | None = None
    on_task_complete: (
        Callable[[BenchmarkTaskCompletionEvent], Awaitable[None]] | None
    ) = None


@dataclass
class _RunTask:
    provider: str
    model: BenchmarkModelConfig
    document: PreparedBenchmarkDocument
    run_number: int
    model_key: str
    task_key: str


def _now_iso_millis() -> str:
    """Match Node's `new Date().toISOString()`: ms precision, 'Z' suffix."""
    moment = datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _make_failed_metrics(
    task: _RunTask, error_message: str
) -> SingleRunMetrics:
    return SingleRunMetrics(
        task_key=task.task_key,
        model_key=task.model_key,
        provider=task.model.provider,
        model_id=task.model.model_id,
        model_label=task.model.model_label,
        tier=task.model.tier,
        domain=task.document.domain,
        document_id=task.document.document_id,
        run_number=task.run_number,
        success=False,
        field_total=0,
        field_correct=0,
        critical_total=0,
        critical_correct=0,
        field_accuracy_pct=0,
        critical_accuracy_pct=0,
        found_key_count=0,
        requested_key_count=len(task.document.ground_truth.keys),
        latency_ms=0,
        input_tokens=0,
        output_tokens=0,
        total_cost_usd=0,
        cache_hit=False,
        cached_input_tokens=0,
        cache_write_tokens=0,
        error=error_message,
    )


def _build_debug_run(
    *,
    task: _RunTask,
    metrics: SingleRunMetrics,
    system_prompt: str,
    user_prompt: str,
    raw_output: str,
    parsed_output: object,
    extracted_pairs: list[dict[str, object]],
    key_comparisons: list[dict[str, object]],
) -> BenchmarkDebugRun:
    """Match TS `{...metrics, task_key, system_prompt_used, ...}` shape.

    Pydantic's `extra='allow'` round-trips the merged metric fields, so
    the on-disk debug record carries the full `SingleRunMetrics` surface
    plus the debug-only fields (matches the TS spread).
    """
    payload: dict[str, object] = metrics.model_dump(mode="json")
    payload.update(
        task_key=task.task_key,
        system_prompt_used=system_prompt,
        user_prompt_used=user_prompt,
        raw_output=raw_output,
        parsed_output=parsed_output,
        extracted_pairs=extracted_pairs,
        key_comparisons=key_comparisons,
    )
    return BenchmarkDebugRun.model_validate(payload)


@dataclass
class _ExecuteParams:
    system_prompt_template: str
    user_prompt_template: str
    pdf_by_document_id: dict[str, str]
    controls: BenchmarkRunExecutionControls
    run_results: list[SingleRunMetrics]
    debug_runs: list[BenchmarkDebugRun]
    report_progress: Callable[[ProgressEvent], None]
    http_client: httpx.AsyncClient


async def _execute_task(task: _RunTask, params: _ExecuteParams) -> None:
    system_prompt = build_benchmark_system_prompt(params.system_prompt_template)
    user_prompt = build_benchmark_user_prompt(
        params.user_prompt_template, task.document.ground_truth
    )
    pdf_base64 = params.pdf_by_document_id.get(task.document.document_id)

    if not pdf_base64:
        metrics = _make_failed_metrics(task, "PDF base64 payload missing.")
        debug = _build_debug_run(
            task=task,
            metrics=metrics,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output="",
            parsed_output=None,
            extracted_pairs=[],
            key_comparisons=[],
        )
        params.run_results.append(metrics)
        params.debug_runs.append(debug)
        if params.controls.on_task_complete:
            await params.controls.on_task_complete(
                BenchmarkTaskCompletionEvent(
                    task_key=task.task_key, metrics=metrics, debug=debug
                )
            )
        params.report_progress(
            ProgressEvent(
                document_id=task.document.document_id,
                model_label=task.model.model_label,
                run_number=task.run_number,
                ok=False,
                error="PDF base64 payload missing.",
            )
        )
        return

    try:
        result = await run_ocr_model(
            OCRModelRunRequest(
                provider=task.model.provider,
                model_id=task.model.model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                pdf_base64=pdf_base64,
                filename=os.path.basename(task.document.source_pdf_abs),
                max_output_tokens=BENCHMARK_MAX_OUTPUT_TOKENS,
            ),
            client=params.http_client,
        )

        detailed = score.score_model_output_detailed(
            result.text, task.document.ground_truth
        )
        field_accuracy_pct = (
            pct(detailed.field_correct, detailed.field_total)
            if detailed.field_total > 0
            else 0.0
        )
        critical_accuracy_pct = (
            pct(detailed.critical_correct, detailed.critical_total)
            if detailed.critical_total > 0
            else 0.0
        )
        has_scorable_fields = detailed.field_total > 0
        has_scorable_critical = detailed.critical_total > 0
        if has_scorable_critical:
            success = detailed.critical_correct == detailed.critical_total
        else:
            success = has_scorable_fields and detailed.field_correct == detailed.field_total

        if result.total_cost_usd is not None:
            total_cost_usd = result.total_cost_usd
        else:
            total_cost_usd = estimate_cost_usd(
                task.model.model_id,
                result.input_tokens,
                result.output_tokens,
                cached_input_tokens=result.cached_input_tokens,
            )

        metrics = SingleRunMetrics(
            task_key=task.task_key,
            model_key=task.model_key,
            provider=task.model.provider,
            model_id=task.model.model_id,
            model_label=task.model.model_label,
            tier=task.model.tier,
            domain=task.document.domain,
            document_id=task.document.document_id,
            run_number=task.run_number,
            success=success,
            field_total=detailed.field_total,
            field_correct=detailed.field_correct,
            critical_total=detailed.critical_total,
            critical_correct=detailed.critical_correct,
            field_accuracy_pct=_round(field_accuracy_pct, 2),
            critical_accuracy_pct=_round(critical_accuracy_pct, 2),
            found_key_count=detailed.found_key_count,
            requested_key_count=detailed.requested_key_count,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_cost_usd=total_cost_usd,
            cache_hit=result.cache_hit,
            cached_input_tokens=result.cached_input_tokens,
            cache_write_tokens=result.cache_write_tokens,
            error=None,
        )

        debug = _build_debug_run(
            task=task,
            metrics=metrics,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=result.text,
            parsed_output=detailed.parsed_output,
            extracted_pairs=[
                {"key": p["key"], "value": p["value"], "found": p["found"]}
                for p in detailed.extracted_pairs
            ],
            key_comparisons=[
                {
                    "key": c.key,
                    "critical": c.critical,
                    "scored": c.scored,
                    "expected_values": c.expected_values,
                    "extracted_value": c.extracted_value,
                    "matched": c.matched,
                    "match_mode": c.match_mode,
                }
                for c in detailed.key_comparisons
            ],
        )

        params.run_results.append(metrics)
        params.debug_runs.append(debug)
        if params.controls.on_task_complete:
            await params.controls.on_task_complete(
                BenchmarkTaskCompletionEvent(
                    task_key=task.task_key, metrics=metrics, debug=debug
                )
            )
        params.report_progress(
            ProgressEvent(
                document_id=task.document.document_id,
                model_label=task.model.model_label,
                run_number=task.run_number,
                ok=True,
            )
        )
    except Exception as error:
        error_message = to_error_message(error)
        metrics = _make_failed_metrics(task, error_message)
        debug = _build_debug_run(
            task=task,
            metrics=metrics,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output="",
            parsed_output=None,
            extracted_pairs=[],
            key_comparisons=[],
        )
        params.run_results.append(metrics)
        params.debug_runs.append(debug)
        if params.controls.on_task_complete:
            await params.controls.on_task_complete(
                BenchmarkTaskCompletionEvent(
                    task_key=task.task_key, metrics=metrics, debug=debug
                )
            )
        params.report_progress(
            ProgressEvent(
                document_id=task.document.document_id,
                model_label=task.model.model_label,
                run_number=task.run_number,
                ok=False,
                error=error_message,
            )
        )


async def run_ocr_leaderboard_benchmark(
    options: BenchmarkRunOptions | None = None,
    controls: BenchmarkRunExecutionControls | None = None,
) -> BenchmarkSnapshot:
    opts = options if options is not None else BenchmarkRunOptions()
    ctrl = controls if controls is not None else BenchmarkRunExecutionControls()

    config = load_benchmark_config()
    system_prompt_template = PATHS.prompts.system.read_text(encoding="utf-8")
    user_prompt_template = PATHS.prompts.user.read_text(encoding="utf-8")

    runs_per_model = max(
        1, opts.runs_per_model if opts.runs_per_model is not None else config.default_runs_per_model
    )
    provider_parallel = opts.provider_parallel is True
    selected_domains = [
        v.strip().lower() for v in (opts.domains or []) if v and v.strip()
    ]
    selected_model_filters = [
        v.strip().lower() for v in (opts.models or []) if v and v.strip()
    ]
    max_documents_per_domain = (
        opts.max_documents_per_domain
        if isinstance(opts.max_documents_per_domain, int) and opts.max_documents_per_domain > 0
        else None
    )

    documents = load_prepared_documents(
        domains=selected_domains,
        max_documents_per_domain=max_documents_per_domain,
    )
    if not documents:
        raise RuntimeError(
            "No benchmark documents selected. Check manifest and domain filters."
        )

    if not selected_model_filters:
        selected_models = list(config.models)
    else:
        selected_models = [
            model
            for model in config.models
            if any(
                needle == model.model_id.strip().lower()
                or needle == model.model_label.strip().lower()
                for needle in selected_model_filters
            )
        ]
    if not selected_models:
        raise RuntimeError(
            "No benchmark models selected. Check --models filters against "
            "config/models.public.json model_id or model_label."
        )

    dataset = summarize_dataset(documents)
    warnings: list[str] = []
    if dataset.labeled_keys == 0:
        warnings.append(
            "Ground truth is not labeled yet. Fill expected values before trusting rankings."
        )
    if dataset.labeled_keys < dataset.total_keys:
        warnings.append(
            f"Only {dataset.labeled_keys}/{dataset.total_keys} keys are currently labeled."
        )
    for domain_name, count in dataset.documents_per_domain.items():
        if count < 10:
            warnings.append(
                f'Domain "{domain_name}" has {count} documents; target is >= 10 for launch quality.'
            )

    pdf_by_document_id: dict[str, str] = {}
    for document in documents:
        raw_bytes = open(document.source_pdf_abs, "rb").read()  # noqa: SIM115
        pdf_by_document_id[document.document_id] = base64.b64encode(raw_bytes).decode("ascii")

    run_tasks: list[_RunTask] = []
    requested_by_model: dict[str, int] = {}
    requested_by_domain_model: dict[str, int] = {}

    for model in selected_models:
        model_key = id_for_model(model.provider, model.model_id)
        for document in documents:
            for run_number in range(1, runs_per_model + 1):
                task_key = build_benchmark_run_task_key(
                    model_key=model_key,
                    domain=document.domain,
                    document_id=document.document_id,
                    run_number=run_number,
                )
                in_only = ctrl.only_task_keys is None or task_key in ctrl.only_task_keys
                not_skipped = ctrl.skip_task_keys is None or task_key not in ctrl.skip_task_keys
                if in_only and not_skipped:
                    run_tasks.append(
                        _RunTask(
                            provider=model.provider,
                            model=model,
                            document=document,
                            run_number=run_number,
                            model_key=model_key,
                            task_key=task_key,
                        )
                    )
            key = f"{document.domain}::{model_key}"
            requested_by_domain_model[key] = (
                requested_by_domain_model.get(key, 0) + runs_per_model
            )
        requested_by_model[model_key] = (
            requested_by_model.get(model_key, 0) + len(documents) * runs_per_model
        )

    provider_count = len({task.provider for task in run_tasks})
    provider_default_parallel = provider_count if provider_count > 0 else 1
    if provider_parallel:
        cap = opts.max_parallel_requests if opts.max_parallel_requests is not None else provider_default_parallel
        max_parallel_requests = max(1, min(provider_default_parallel, cap))
    else:
        max_parallel_requests = max(
            1,
            opts.max_parallel_requests
            if opts.max_parallel_requests is not None
            else config.max_parallel_requests,
        )

    run_results: list[SingleRunMetrics] = list(ctrl.initial_runs)
    debug_runs: list[BenchmarkDebugRun] = list(ctrl.initial_debug_runs)

    started_at_ms = int(time.time() * 1000)
    expected_runs_by_document: dict[str, int] = {}
    for task in run_tasks:
        expected_runs_by_document[task.document.document_id] = (
            expected_runs_by_document.get(task.document.document_id, 0) + 1
        )

    report_progress = create_progress_reporter(
        total_tasks=len(run_tasks),
        expected_runs_by_document=expected_runs_by_document,
        started_at_ms=started_at_ms,
    )

    async with httpx.AsyncClient(timeout=300.0) as http_client:
        exec_params = _ExecuteParams(
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            pdf_by_document_id=pdf_by_document_id,
            controls=ctrl,
            run_results=run_results,
            debug_runs=debug_runs,
            report_progress=report_progress,
            http_client=http_client,
        )

        async def runner(task: _RunTask) -> None:
            await _execute_task(task, exec_params)

        if provider_parallel:
            await run_by_provider_lanes(run_tasks, max_parallel_requests, runner)
        else:
            await run_in_pool(run_tasks, max_parallel_requests, runner)

    leaderboard = aggregate_rows(run_results, requested_by_model)
    cache_summary = build_cache_summary(run_results)

    by_domain: list[DomainLeaderboard] = []
    for domain in sorted(dataset.documents_per_domain.keys()):
        domain_runs = [r for r in run_results if r.domain == domain]
        domain_requested: dict[str, int] = {}
        for key, value in requested_by_domain_model.items():
            candidate_domain, model_key = key.split("::", 1)
            if candidate_domain == domain:
                domain_requested[model_key] = value
        by_domain.append(
            DomainLeaderboard(domain=domain, rows=aggregate_rows(domain_runs, domain_requested))
        )

    snapshot = BenchmarkSnapshot(
        generated_at=_now_iso_millis(),
        benchmark_id=build_benchmark_id(),
        benchmark_description=config.description,
        options=BenchmarkSnapshotOptions(
            runs_per_model=runs_per_model,
            max_parallel_requests=max_parallel_requests,
            provider_parallel=provider_parallel,
            selected_domains=selected_domains,
            max_documents_per_domain=max_documents_per_domain,
        ),
        dataset=dataset,
        leaderboard=leaderboard,
        by_domain=by_domain,
        run_count=len(run_results),
        markdown_table=build_markdown_table(leaderboard),
        warnings=warnings,
        cache_summary=cache_summary,
        debug=BenchmarkDebugSnapshot(
            documents=[
                BenchmarkDebugDocument(
                    document_id=d.document_id,
                    domain=d.domain,
                    source_pdf=d.source_pdf,
                    ground_truth_path=to_repo_relative_path(d.ground_truth_abs, REPO_ROOT),
                    ground_truth=d.ground_truth_raw,
                )
                for d in documents
            ],
            runs=debug_runs,
        ),
    )

    return snapshot
