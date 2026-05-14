"""Summarize the state of a checkpoint dir. Python port of `scripts/checkpoint_status.mjs`.

Usage:
    uv run python scripts/checkpoint_status.py [--checkpoint-dir=PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ocr_mini_bench.config.paths import PATHS


def _to_model_key(metrics: dict[str, Any]) -> str:
    model_key = metrics.get("model_key")
    if isinstance(model_key, str) and model_key.strip():
        return model_key
    provider = metrics.get("provider", "unknown")
    model_id = metrics.get("model_id", "unknown")
    return f"{provider}:{model_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PATHS.checkpoint.root,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    # Use absolute() not resolve() so we don't follow symlinks — matches
    # Node's path.resolve() behaviour (e.g. /tmp stays /tmp on macOS).
    checkpoint_dir: Path = args.checkpoint_dir.absolute()
    runs_path = checkpoint_dir / "runs.jsonl"
    state_path = checkpoint_dir / "state.json"

    if not runs_path.exists():
        if args.json:
            print(
                json.dumps(
                    {
                        "checkpoint_dir": str(checkpoint_dir),
                        "runs_log": str(runs_path),
                        "state_file": str(state_path),
                        "exists": False,
                        "message": "No checkpoint log found. Run the benchmark once first.",
                    },
                    indent=2,
                )
            )
            return 0
        print(f"Checkpoint dir: {checkpoint_dir}")
        print("No checkpoint log found. Run the benchmark once first.")
        return 0

    lines = [line for line in runs_path.read_text().splitlines() if line.strip()]
    latest_by_task: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_key = parsed.get("task_key")
        metrics = parsed.get("metrics")
        if not task_key or not metrics:
            continue
        latest_by_task[task_key] = parsed

    state: dict[str, Any] | None = None
    if state_path.exists():
        raw = state_path.read_text().strip()
        if raw:
            try:
                state = json.loads(raw)
            except json.JSONDecodeError:
                state = None

    failed = 0
    successful = 0
    failed_by_model: Counter[str] = Counter()
    success_by_model: Counter[str] = Counter()
    failed_error_types: Counter[str] = Counter()

    for record in latest_by_task.values():
        metrics = record["metrics"]
        model_key = _to_model_key(metrics)
        if metrics.get("error"):
            failed += 1
            failed_by_model[model_key] += 1
            failed_error_types[str(metrics["error"])] += 1
        else:
            successful += 1
            success_by_model[model_key] += 1

    by_model = [
        {
            "model_key": model,
            "successful": success_by_model.get(model, 0),
            "failed": failed_by_model.get(model, 0),
        }
        for model in sorted(set(success_by_model) | set(failed_by_model))
    ]
    top_errors = [
        {"count": count, "error": err}
        for err, count in failed_error_types.most_common(10)
    ]
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "runs_log": str(runs_path),
        "state_file": str(state_path),
        "raw_lines": len(lines),
        "latest_task_records": len(latest_by_task),
        "successful_records": successful,
        "failed_records": failed,
        # Match TS JSON output: when the rate is exactly zero, JS serializes
        # it as `0` (int-shaped) rather than `0.0`. Mirror that here.
        "failure_rate_pct": (
            (failed / len(latest_by_task) * 100) if latest_by_task and failed else 0
        ),
        "state": state,
        "by_model": by_model,
        "top_errors": top_errors,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Checkpoint dir: {summary['checkpoint_dir']}")
    print(f"Records (latest by task): {summary['latest_task_records']}")
    print(f"Successful: {summary['successful_records']}")
    print(
        f"Failed: {summary['failed_records']} "
        f"({summary['failure_rate_pct']:.2f}%)"
    )

    if state:
        # JS-style lowercase booleans, to match TS output verbatim.
        final_str = str(state.get("final", False)).lower()
        print(f"Mode: {state.get('mode', 'unknown')} | Final: {final_str}")
        print(f"Updated at: {state.get('updated_at', 'n/a')}")

    print("\nBy model:")
    for row in by_model:
        print(f"- {row['model_key']}: ok={row['successful']}, fail={row['failed']}")

    if top_errors:
        print("\nTop errors:")
        for row in top_errors:
            print(f"- ({row['count']}) {row['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
