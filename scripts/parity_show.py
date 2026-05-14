"""Human-readable side-by-side view of the two leaderboard files.

Walks `leaderboard.aggregation.json` and `leaderboard.frontend.json` from two
`artifacts/` (or `artifacts/postprocess/`) directories and prints every numeric
field side-by-side with drift percent. Volatile keys (timestamps, banners) are
skipped.

Usage:
    uv run python scripts/parity_show.py <ts_dir> <py_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SKIP = {
    "generated_at",
    "benchmark_id",
    "source",
    "markdown_table",
    "schema_version",
    "benchmark_description",
}


def walk(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    out: list[tuple[str, Any, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP:
                continue
            sub = f"{path}.{k}" if path else k
            if k not in a:
                out.append((sub, "<missing>", b[k]))
                continue
            if k not in b:
                out.append((sub, a[k], "<missing>"))
                continue
            out.extend(walk(a[k], b[k], sub))
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            ai = a[i] if i < len(a) else "<missing>"
            bi = b[i] if i < len(b) else "<missing>"
            out.extend(walk(ai, bi, f"{path}[{i}]"))
    else:
        if a != b:
            out.append((path, a, b))
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)[:18]


def _drift(a: Any, b: Any) -> str:
    if isinstance(a, bool) or isinstance(b, bool):
        return ""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return ""
    if a == 0 and b == 0:
        return "0%"
    if a == 0:
        return "+inf"
    return f"{(b - a) / abs(a) * 100:+.1f}%"


def report(ts_dir: Path, py_dir: Path, name: str) -> None:
    print(f"\n{'=' * 100}")
    print(f"{name}   (TS: {ts_dir}   PY: {py_dir})")
    print("=" * 100)
    a = json.loads((ts_dir / name).read_text())
    b = json.loads((py_dir / name).read_text())
    diffs = walk(a, b)
    if not diffs:
        print("  (no differences)")
        return
    print(f"  {'PATH':<70} {'TS':>18} {'PY':>18}  DRIFT")
    print(f"  {'-' * 70} {'-' * 18} {'-' * 18}  -----")
    for path, ts, py in diffs:
        print(f"  {path:<70} {_fmt(ts):>18} {_fmt(py):>18}  {_drift(ts, py)}")
    print(f"\n  {len(diffs)} differing fields")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ts = Path(sys.argv[1]).resolve()
    py = Path(sys.argv[2]).resolve()
    if (ts / "postprocess").is_dir():
        ts = ts / "postprocess"
    if (py / "postprocess").is_dir():
        py = py / "postprocess"
    for name in ("leaderboard.aggregation.json", "leaderboard.frontend.json"):
        report(ts, py, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
