"""Identity / key construction. Mirrors `src/benchmark/run/identity.ts`.

Used to build the canonical `model_key` and `task_key` strings that flow
through every artifact, plus repo-relative path conversion.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def id_for_model(provider: str, model_id: str) -> str:
    return f"{provider}:{model_id}"


def build_benchmark_run_task_key(
    *, model_key: str, domain: str, document_id: str, run_number: int
) -> str:
    return f"{model_key}::{domain}::{document_id}::{run_number}"


def build_benchmark_id(now: datetime | None = None) -> str:
    """`ocr-benchmark-<iso>` with `:`/`.` replaced by `-`. Matches the TS form
    `new Date().toISOString().replace(/[:.]/g, '-')`."""
    moment = now if now is not None else datetime.now(UTC)
    # JS Date.toISOString() always renders milliseconds: "2026-05-14T07:05:59.242Z"
    iso = moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{moment.microsecond // 1000:03d}Z"
    )
    return f"ocr-benchmark-{iso.replace(':', '-').replace('.', '-')}"


def to_repo_relative_path(abs_path: str | Path, repo_root: str | Path) -> str:
    return Path(abs_path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
