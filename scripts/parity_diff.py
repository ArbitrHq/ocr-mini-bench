"""Structural parity diff between two `artifacts/` directories.

Phase 6 verification tool: compare benchmark artifacts produced by the TS and
Python implementations for structural equivalence. Numeric fields are compared
with a tolerance because LLM output is nondeterministic and per-row scores
drift run-to-run. Volatile fields (timestamps, raw LLM text, prompts, schema
version banners, file paths) are ignored by name.

Usage:
    uv run python scripts/parity_diff.py <ts_artifacts_dir> <py_artifacts_dir>

Exits 0 when no structural mismatches are found (numeric drift only warns).
Exits 1 when structural mismatches are present.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VOLATILE_KEYS: set[str] = {
    "completed_at",
    "generated_at",
    "updated_at",
    "benchmark_id",
    "benchmark_fingerprint",
    "input_raw_jsonl",
    "output_comparison_jsonl",
    "source",
    "markdown_table",
    "system_prompt_used",
    "user_prompt_used",
    "raw_output",
    "parsed_payload",
    # Per-row API timing/token/cost vary every call; aggregates (avg/p95/total
    # in metrics.snapshot.json + leaderboard.aggregation.json) still get
    # compared with numeric tolerance.
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "total_cost_usd",
}

# Containers whose contents are partially LLM-emitted. When we descend into
# one of these, we still compare stable keys (e.g. `key`, `critical`, `scored`,
# `match_mode`, `expected_values` — derived from config/ground truth) but skip
# keys in LLM_VALUE_KEYS (the model's OCR output).
LLM_OUTPUT_KEYS: set[str] = {
    "extracted_pairs",
    "parsed_output",
    "key_comparisons",
}

# Leaf keys whose values come from the LLM and drift run-to-run. Only treated
# as volatile when nested inside an LLM_OUTPUT_KEYS container, so a top-level
# `value` (e.g. in summary stats) is still compared normally.
LLM_VALUE_KEYS: set[str] = {
    "value",
    "found",
    "extracted_value",
    "matched",
    "notes",
    "missing_keys",
}

NUMERIC_TOLERANCE_PCT = 25.0
NUMERIC_TOLERANCE_ABS = 1.0


@dataclass
class DiffReport:
    structural: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    ignored: int = 0

    def fatal(self, msg: str) -> None:
        self.structural.append(msg)

    def warn(self, msg: str) -> None:
        self.numeric.append(msg)

    def merge(self, other: DiffReport) -> None:
        self.structural.extend(other.structural)
        self.numeric.extend(other.numeric)
        self.ignored += other.ignored


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _numeric_drift_ok(a: float, b: float) -> bool:
    if a == b:
        return True
    diff = abs(a - b)
    if diff <= NUMERIC_TOLERANCE_ABS:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and (diff / denom) * 100 <= NUMERIC_TOLERANCE_PCT


def _compare_value(
    a: Any,
    b: Any,
    path: str,
    report: DiffReport,
    *,
    key_name: str | None = None,
    in_llm_output: bool = False,
) -> None:
    if key_name in VOLATILE_KEYS:
        report.ignored += 1
        return

    # Inside an LLM-output container, the model's emitted fields are volatile.
    if in_llm_output and key_name in LLM_VALUE_KEYS:
        report.ignored += 1
        return

    if type(a) is not type(b):
        if _is_number(a) and _is_number(b):
            pass
        else:
            report.fatal(f"{path}: type mismatch {type(a).__name__} vs {type(b).__name__}")
            return

    if isinstance(a, dict):
        assert isinstance(b, dict)
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        for k in sorted(only_a):
            if k in VOLATILE_KEYS or (in_llm_output and k in LLM_VALUE_KEYS):
                report.ignored += 1
                continue
            report.fatal(f"{path}: key only in A: {k!r}")
        for k in sorted(only_b):
            if k in VOLATILE_KEYS or (in_llm_output and k in LLM_VALUE_KEYS):
                report.ignored += 1
                continue
            report.fatal(f"{path}: key only in B: {k!r}")
        for k in sorted(keys_a & keys_b):
            child_in_llm = in_llm_output or k in LLM_OUTPUT_KEYS
            _compare_value(
                a[k], b[k], f"{path}.{k}", report, key_name=k, in_llm_output=child_in_llm
            )
        return

    if isinstance(a, list):
        assert isinstance(b, list)
        if len(a) != len(b):
            report.fatal(f"{path}: list length mismatch {len(a)} vs {len(b)}")
            return
        for i, (ea, eb) in enumerate(zip(a, b, strict=True)):
            _compare_value(ea, eb, f"{path}[{i}]", report, in_llm_output=in_llm_output)
        return

    if _is_number(a) and _is_number(b):
        if not _numeric_drift_ok(float(a), float(b)):
            report.warn(f"{path}: numeric drift {a!r} vs {b!r}")
        return

    if a != b:
        # Strings: structural mismatch only matters if they look like enum-ish
        # values (provider/tier/match_mode). Free-form text drift is expected.
        if isinstance(a, str) and isinstance(b, str):
            if " " not in a and " " not in b and len(a) < 40 and len(b) < 40:
                report.fatal(f"{path}: enum-like string mismatch {a!r} vs {b!r}")
            else:
                report.ignored += 1
            return
        report.fatal(f"{path}: value mismatch {a!r} vs {b!r}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sort_jsonl_by_task_key(rows: list[Any]) -> list[Any]:
    def key(r: Any) -> str:
        return str(r.get("task_key", "")) if isinstance(r, dict) else ""

    return sorted(rows, key=key)


def _diff_jsonl(a_path: Path, b_path: Path, report: DiffReport) -> None:
    a_rows = _sort_jsonl_by_task_key(_load_jsonl(a_path))
    b_rows = _sort_jsonl_by_task_key(_load_jsonl(b_path))

    if len(a_rows) != len(b_rows):
        report.fatal(
            f"{a_path.name}: row count mismatch {len(a_rows)} vs {len(b_rows)}"
        )
        # still try to align by task_key for additional info
        a_by_key = {r.get("task_key"): r for r in a_rows if isinstance(r, dict)}
        b_by_key = {r.get("task_key"): r for r in b_rows if isinstance(r, dict)}
        only_a = set(a_by_key) - set(b_by_key)
        only_b = set(b_by_key) - set(a_by_key)
        for k in sorted(only_a):
            if k is not None:
                report.fatal(f"{a_path.name}: task_key only in A: {k!r}")
        for k in sorted(only_b):
            if k is not None:
                report.fatal(f"{a_path.name}: task_key only in B: {k!r}")
        return

    for i, (ra, rb) in enumerate(zip(a_rows, b_rows, strict=True)):
        _compare_value(ra, rb, f"{a_path.name}[{i}]", report)


def _diff_json(a_path: Path, b_path: Path, report: DiffReport) -> None:
    a = _load_json(a_path)
    b = _load_json(b_path)
    _compare_value(a, b, a_path.name, report)


FILES_JSONL = ["raw.jsonl", "comparison.jsonl"]
FILES_JSON = [
    "comparison.summary.json",
    "metrics.snapshot.json",
    "leaderboard.aggregation.json",
    "leaderboard.frontend.json",
]


def diff_postprocess_dirs(ts_dir: Path, py_dir: Path) -> DiffReport:
    overall = DiffReport()

    for name in FILES_JSONL:
        a = ts_dir / name
        b = py_dir / name
        if not a.exists() or not b.exists():
            overall.fatal(f"{name}: missing (A exists={a.exists()} B exists={b.exists()})")
            continue
        per = DiffReport()
        _diff_jsonl(a, b, per)
        overall.merge(per)
        print(
            f"  {name}: structural={len(per.structural)} numeric_warns={len(per.numeric)} ignored={per.ignored}"
        )

    for name in FILES_JSON:
        a = ts_dir / name
        b = py_dir / name
        if not a.exists() or not b.exists():
            overall.fatal(f"{name}: missing (A exists={a.exists()} B exists={b.exists()})")
            continue
        per = DiffReport()
        _diff_json(a, b, per)
        overall.merge(per)
        print(
            f"  {name}: structural={len(per.structural)} numeric_warns={len(per.numeric)} ignored={per.ignored}"
        )

    return overall


def _print_findings(label: str, findings: Iterable[str], *, limit: int = 50) -> None:
    findings = list(findings)
    if not findings:
        return
    print(f"\n{label} ({len(findings)}):")
    for line in findings[:limit]:
        print(f"  - {line}")
    if len(findings) > limit:
        print(f"  ... ({len(findings) - limit} more)")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ts_dir = Path(sys.argv[1]).resolve()
    py_dir = Path(sys.argv[2]).resolve()

    if not ts_dir.is_dir() or not py_dir.is_dir():
        print(f"error: both args must be directories. got {ts_dir} / {py_dir}", file=sys.stderr)
        return 2

    # Each arg should be a postprocess dir or a parent containing one
    def normalize(p: Path) -> Path:
        if (p / "postprocess").is_dir():
            return p / "postprocess"
        return p

    ts_pp = normalize(ts_dir)
    py_pp = normalize(py_dir)

    print("Comparing postprocess artifacts:")
    print(f"  A (TS):     {ts_pp}")
    print(f"  B (Python): {py_pp}")
    print()
    print("Per-file results:")
    report = diff_postprocess_dirs(ts_pp, py_pp)

    _print_findings("STRUCTURAL MISMATCHES (fatal)", report.structural)
    _print_findings("NUMERIC DRIFT (within tolerance ignored)", report.numeric)

    print()
    print(f"Summary: structural={len(report.structural)} numeric_warns={len(report.numeric)} ignored={report.ignored}")

    return 0 if not report.structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
