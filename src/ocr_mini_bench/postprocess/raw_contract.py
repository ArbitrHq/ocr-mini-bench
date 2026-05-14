"""Convert checkpoint records / debug runs into the canonical raw.jsonl
shape. Mirrors `src/postprocess/raw-contract.ts`.
"""

from __future__ import annotations

from .types import (
    LegacyCheckpointDebug,
    LegacyCheckpointRecord,
    RawDocumentInfo,
    RawLegacyMetrics,
    RawModelInfo,
    RawNormalizedRecord,
    RawPayload,
    RawRuntimeInfo,
)


def to_task_key_from_debug(run: LegacyCheckpointDebug) -> str:
    if run.task_key:
        return run.task_key
    return f"{run.model_key}::{run.domain}::{run.document_id}::run{run.run_number}"


def from_checkpoint_record(record: LegacyCheckpointRecord) -> RawNormalizedRecord:
    metrics = record.metrics
    debug = record.debug
    return RawNormalizedRecord(
        schema_version="1.0",
        task_key=record.task_key,
        completed_at=record.completed_at if isinstance(record.completed_at, str) else None,
        model=RawModelInfo(
            model_key=metrics.model_key,
            provider=metrics.provider,
            model_id=metrics.model_id,
            model_label=metrics.model_label,
            tier=metrics.tier,
        ),
        document=RawDocumentInfo(
            domain=metrics.domain,
            document_id=metrics.document_id,
            run_number=metrics.run_number,
        ),
        runtime=RawRuntimeInfo(
            latency_ms=metrics.latency_ms,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_cost_usd=metrics.total_cost_usd,
            cache_hit=metrics.cache_hit,
            cached_input_tokens=metrics.cached_input_tokens,
            cache_write_tokens=metrics.cache_write_tokens,
            error=metrics.error,
        ),
        payload=RawPayload(
            system_prompt_used=debug.system_prompt_used or "",
            user_prompt_used=debug.user_prompt_used or "",
            raw_output=debug.raw_output or "",
            parsed_output=debug.parsed_output,
            extracted_pairs=debug.extracted_pairs or [],
        ),
        legacy_metrics=RawLegacyMetrics(
            success=metrics.success,
            field_total=metrics.field_total,
            field_correct=metrics.field_correct,
            critical_total=metrics.critical_total,
            critical_correct=metrics.critical_correct,
            field_accuracy_pct=metrics.field_accuracy_pct,
            critical_accuracy_pct=metrics.critical_accuracy_pct,
            found_key_count=metrics.found_key_count,
            requested_key_count=metrics.requested_key_count,
        ),
    )


def from_debug_run(run: LegacyCheckpointDebug) -> RawNormalizedRecord:
    task_key = to_task_key_from_debug(run)
    return RawNormalizedRecord(
        schema_version="1.0",
        task_key=task_key,
        completed_at=None,
        model=RawModelInfo(
            model_key=run.model_key,
            provider=run.provider,
            model_id=run.model_id,
            model_label=run.model_label,
            tier=run.tier,
        ),
        document=RawDocumentInfo(
            domain=run.domain,
            document_id=run.document_id,
            run_number=run.run_number,
        ),
        runtime=RawRuntimeInfo(
            latency_ms=run.latency_ms,
            input_tokens=run.input_tokens if run.input_tokens is not None else 0,
            output_tokens=run.output_tokens if run.output_tokens is not None else 0,
            total_cost_usd=run.total_cost_usd,
            cache_hit=run.cache_hit,
            cached_input_tokens=run.cached_input_tokens,
            cache_write_tokens=run.cache_write_tokens,
            error=run.error,
        ),
        payload=RawPayload(
            system_prompt_used=run.system_prompt_used or "",
            user_prompt_used=run.user_prompt_used or "",
            raw_output=run.raw_output or "",
            parsed_output=run.parsed_output,
            extracted_pairs=run.extracted_pairs or [],
        ),
        legacy_metrics=RawLegacyMetrics(
            success=run.success,
            field_total=run.field_total,
            field_correct=run.field_correct,
            critical_total=run.critical_total,
            critical_correct=run.critical_correct,
            field_accuracy_pct=run.field_accuracy_pct,
            critical_accuracy_pct=run.critical_accuracy_pct,
            found_key_count=run.found_key_count,
            requested_key_count=run.requested_key_count,
        ),
    )
