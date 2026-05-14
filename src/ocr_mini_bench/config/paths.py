"""Canonical filesystem paths shared by both TS and Python implementations.

Mirrors `src/config/paths.ts`. The TS code anchors on `process.cwd()` which
assumes the user runs commands from the repo root; we anchor on the package
location so `ocr-bench` works from anywhere.

Override the anchor with `OCR_MINI_BENCH_REPO_ROOT` if you need to point at a
different dataset tree (used by replay tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_repo_root() -> Path:
    env = os.environ.get("OCR_MINI_BENCH_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # src/ocr_mini_bench/config/paths.py -> repo root is 3 levels up
    # (paths.py -> config -> ocr_mini_bench -> src -> repo root)
    return Path(__file__).resolve().parents[3]


REPO_ROOT: Path = _default_repo_root()


@dataclass(frozen=True)
class ConfigPaths:
    models: Path


@dataclass(frozen=True)
class DatasetPaths:
    manifest: Path


@dataclass(frozen=True)
class PromptPaths:
    system: Path
    user: Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    checkpoints: Path
    postprocess: Path
    latest_json: Path
    latest_debug: Path
    latest_markdown: Path


@dataclass(frozen=True)
class PostprocessPaths:
    root: Path
    raw_jsonl: Path
    raw_summary: Path
    comparison_jsonl: Path
    comparison_summary: Path
    metrics_snapshot: Path
    leaderboard_aggregation: Path
    leaderboard_frontend: Path


@dataclass(frozen=True)
class CheckpointPaths:
    root: Path
    runs_jsonl: Path
    raw_runs_jsonl: Path
    raw_jsonl: Path
    state: Path


@dataclass(frozen=True)
class Paths:
    config: ConfigPaths
    dataset: DatasetPaths
    prompts: PromptPaths
    artifacts: ArtifactPaths
    postprocess: PostprocessPaths
    checkpoint: CheckpointPaths


def build_paths(repo_root: Path | None = None) -> Paths:
    root = repo_root if repo_root is not None else REPO_ROOT
    artifacts = root / "artifacts"
    checkpoints = artifacts / "checkpoints"
    postprocess = artifacts / "postprocess"
    return Paths(
        config=ConfigPaths(models=root / "config" / "models.public.json"),
        dataset=DatasetPaths(manifest=root / "dataset" / "manifest.json"),
        prompts=PromptPaths(
            system=root / "prompts" / "ocr" / "benchmark" / "extract_system.txt",
            user=root / "prompts" / "ocr" / "benchmark" / "extract_user.txt",
        ),
        artifacts=ArtifactPaths(
            root=artifacts,
            checkpoints=checkpoints,
            postprocess=postprocess,
            latest_json=artifacts / "latest.json",
            latest_debug=artifacts / "latest.debug.json",
            latest_markdown=artifacts / "latest.md",
        ),
        postprocess=PostprocessPaths(
            root=postprocess,
            raw_jsonl=postprocess / "raw.jsonl",
            raw_summary=postprocess / "raw.summary.json",
            comparison_jsonl=postprocess / "comparison.jsonl",
            comparison_summary=postprocess / "comparison.summary.json",
            metrics_snapshot=postprocess / "metrics.snapshot.json",
            leaderboard_aggregation=postprocess / "leaderboard.aggregation.json",
            leaderboard_frontend=postprocess / "leaderboard.frontend.json",
        ),
        checkpoint=CheckpointPaths(
            root=checkpoints,
            runs_jsonl=checkpoints / "runs.jsonl",
            raw_runs_jsonl=checkpoints / "raw.runs.jsonl",
            raw_jsonl=checkpoints / "raw.jsonl",
            state=checkpoints / "state.json",
        ),
    )


PATHS: Paths = build_paths()
