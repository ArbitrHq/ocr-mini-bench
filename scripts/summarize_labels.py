"""Summarize ground-truth label completeness per domain. Python port of
`scripts/summarize_labels.mjs`.

Usage:
    uv run python scripts/summarize_labels.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ocr_mini_bench.config.paths import PATHS

REPO_ROOT = PATHS.dataset.manifest.parent.parent
MANIFEST_PATH = PATHS.dataset.manifest


def _has_label(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    return False


def _collect_value_nodes(node: Any) -> list[dict[str, Any]]:
    """Mirror the TS `collectValueNodes`: walk dict-of-dicts, stop at any
    dict that has a `value` key (those are leaf field descriptors)."""
    out: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return out
    for key, value in node.items():
        if not isinstance(value, dict):
            continue
        if "value" in value:
            out.append(
                {
                    "name": key,
                    "critical": bool(value.get("critical")),
                    "expected": value["value"],
                }
            )
            continue
        out.extend(_collect_value_nodes(value))
    return out


def _read_comparable_keys(ground_truth: Any) -> list[dict[str, Any]]:
    if isinstance(ground_truth, dict) and isinstance(ground_truth.get("keys"), list):
        raise ValueError(
            "Legacy keys[] schema is not supported. Use field objects with { value, critical, type }."
        )
    keys = [
        {"critical": bool(k["critical"]), "expected": k["expected"]}
        for k in _collect_value_nodes(ground_truth)
    ]
    if not keys:
        raise ValueError("No comparable fields found.")
    return keys


def _pct(num: int, denom: int) -> float:
    if not denom:
        return 0
    return num / denom * 100


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    lines: list[str] = ["Label completeness summary", ""]

    for domain in manifest.get("domains", []) or []:
        total_keys = 0
        labeled_keys = 0
        critical_keys = 0
        labeled_critical_keys = 0
        invalid_files = 0

        for document in domain.get("documents", []) or []:
            gt_path = (REPO_ROOT / document["ground_truth"]).resolve()
            try:
                ground_truth = json.loads(gt_path.read_text())
            except (OSError, json.JSONDecodeError):
                invalid_files += 1
                continue

            try:
                keys = _read_comparable_keys(ground_truth)
            except ValueError:
                invalid_files += 1
                continue

            total_keys += len(keys)
            labeled_keys += sum(1 for k in keys if _has_label(k["expected"]))
            critical_keys += sum(1 for k in keys if k["critical"])
            labeled_critical_keys += sum(
                1 for k in keys if k["critical"] and _has_label(k["expected"])
            )

        lines.append(
            f"{domain['id']}: {labeled_keys}/{total_keys} keys labeled "
            f"({_pct(labeled_keys, total_keys):.1f}%), "
            f"critical {labeled_critical_keys}/{critical_keys} "
            f"({_pct(labeled_critical_keys, critical_keys):.1f}%), "
            f"invalid files {invalid_files}"
        )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
