"""Rebuild snapshot artifacts from a checkpoint dir. Python port of
`scripts/rebuild_from_checkpoint.mjs`.

Reads `runs.jsonl` (the legacy per-run log with metrics + debug payload),
re-aggregates a leaderboard with per-document `pass@N` and `pass@N strict`
columns, and writes `snapshot-<ts>.json`, `snapshot-<ts>.debug.json`,
`latest.json`, `latest.debug.json`, `latest.md` to `--output-dir` (default
`artifacts/`).

Note: the aggregation math here is *separate* from `benchmark/run/aggregation.py`
because the rebuild flow uses per-document trial counts (not probabilistic
estimates from success rate). Markdown table is also wider — includes
`pass^N strict` columns and `Keys Found %`.

Usage:
    uv run python scripts/rebuild_from_checkpoint.py [--checkpoint-dir=PATH] [--output-dir=PATH] [--repo-root=PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocr_mini_bench.config.paths import PATHS

REPO_ROOT_DEFAULT = PATHS.dataset.manifest.parent.parent


def _normalize_domain(value: Any) -> str:
    return str(value or "").strip().lower()


def _mean(values: list[float]) -> float:
    if not values:
        return 0
    return sum(values) / len(values)


def _std_dev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0
    avg = _mean(values)
    variance = max(0.0, sum((v - avg) ** 2 for v in values) / len(values))
    return math.sqrt(variance)


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0
    return (part / total) * 100


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0
    sorted_vals = sorted(values)
    rank = min(len(sorted_vals) - 1, max(0, math.ceil((p / 100) * len(sorted_vals)) - 1))
    return sorted_vals[rank]


def _round(value: float, decimals: int = 4) -> float:
    """Match JS Math.round (round-half-away-from-zero for positives)."""
    precision = 10**decimals
    scaled = value * precision
    rounded = int(math.floor(scaled + 0.5) if scaled >= 0 else -math.floor(-scaled + 0.5))
    return rounded / precision


def _id_for_model(provider: str, model_id: str) -> str:
    return f"{provider}:{model_id}"


def _combination(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    kk = min(k, n - k)
    result: float = 1
    for i in range(1, kk + 1):
        result = (result * (n - kk + i)) / i
    return result


def _document_run_stats(runs: list[dict[str, Any]]) -> list[dict[str, int]]:
    by_doc: dict[str, dict[str, int]] = {}
    for run in runs:
        cur = by_doc.setdefault(run["document_id"], {"trials": 0, "successes": 0})
        cur["trials"] += 1
        if run.get("success"):
            cur["successes"] += 1
    return list(by_doc.values())


def _at_least_pass_at_n(runs: list[dict[str, Any]], n: int) -> float | None:
    stats = _document_run_stats(runs)
    if not stats:
        return None
    if any(doc["trials"] < n for doc in stats):
        return None
    passing_docs = sum(1 for doc in stats if doc["successes"] >= n)
    return _round(_pct(passing_docs, len(stats)), 2)


def _strict_pass_at_n(runs: list[dict[str, Any]], n: int) -> float | None:
    stats = _document_run_stats(runs)
    if not stats:
        return None
    if any(doc["trials"] < n for doc in stats):
        return None
    total = 0.0
    for doc in stats:
        denom = _combination(doc["trials"], n)
        num = _combination(doc["successes"], n)
        total += num / denom if denom > 0 else 0
    return _round((total / len(stats)) * 100, 2)


def _aggregate_rows(
    run_results: list[dict[str, Any]],
    requested_by_model: dict[str, int],
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in run_results:
        by_model[run["model_key"]].append(run)

    rows: list[dict[str, Any]] = []
    for model_key, runs in by_model.items():
        sample = runs[0]
        runs_completed = len(runs)
        failed_runs = sum(1 for r in runs if r.get("error"))
        successful_runs = sum(1 for r in runs if r.get("success"))

        non_error = [r for r in runs if not r.get("error")]
        field_accuracy = [
            (r["field_correct"] / r["field_total"]) * 100
            for r in non_error
            if r.get("field_total", 0) > 0
        ]
        critical_accuracy = [
            (r["critical_correct"] / r["critical_total"]) * 100
            for r in non_error
            if r.get("critical_total", 0) > 0
        ]
        latencies = [r["latency_ms"] for r in non_error]
        costs = [r["total_cost_usd"] for r in non_error]
        keys_found_pct = [
            (r["found_key_count"] / r["requested_key_count"]) * 100
            for r in non_error
            if r.get("requested_key_count", 0) > 0
        ]

        total_cost = sum(costs)
        success_rate = _pct(successful_runs, runs_completed)
        avg_field_accuracy = _mean(field_accuracy)
        field_std = _std_dev(field_accuracy)
        field_variance_pct = (field_std / avg_field_accuracy) * 100 if avg_field_accuracy > 0 else 100
        avg_cost_per_run = _mean(costs)
        avg_cost_per_doc = avg_cost_per_run if costs else None

        rows.append(
            {
                "rank": 0,
                "model_key": model_key,
                "provider": sample["provider"],
                "model_id": sample["model_id"],
                "model_label": sample["model_label"],
                "tier": sample["tier"],
                "runs_requested": requested_by_model.get(model_key, runs_completed),
                "runs_completed": runs_completed,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate_pct": _round(success_rate, 2),
                "pass_at_2_pct": _at_least_pass_at_n(runs, 2),
                "pass_at_3_pct": _at_least_pass_at_n(runs, 3),
                "pass_at_5_pct": _at_least_pass_at_n(runs, 5),
                "pass_at_10_pct": _at_least_pass_at_n(runs, 10),
                "pass_at_2_strict_pct": _strict_pass_at_n(runs, 2),
                "pass_at_3_strict_pct": _strict_pass_at_n(runs, 3),
                "pass_at_5_strict_pct": _strict_pass_at_n(runs, 5),
                "pass_at_10_strict_pct": _strict_pass_at_n(runs, 10),
                "avg_keys_found_pct": _round(_mean(keys_found_pct), 2),
                "avg_field_accuracy_pct": _round(avg_field_accuracy, 2),
                "avg_critical_accuracy_pct": _round(_mean(critical_accuracy), 2),
                "field_accuracy_variance_pct": _round(field_variance_pct, 2),
                "avg_latency_ms": _round(_mean(latencies), 1),
                "p95_latency_ms": _round(_percentile(latencies, 95), 1),
                "total_cost_usd": _round(total_cost, 6),
                "avg_cost_per_doc_usd": _round(avg_cost_per_doc, 6) if avg_cost_per_doc is not None else None,
                "avg_cost_per_run_usd": _round(avg_cost_per_run, 6),
                "cost_per_success_usd": (
                    _round(total_cost / successful_runs, 6) if successful_runs > 0 else None
                ),
                "p95_cost_usd": _round(_percentile(costs, 95), 6),
                "p05_field_accuracy_pct": _round(_percentile(field_accuracy, 5), 2),
            }
        )

    def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        cost = row["cost_per_success_usd"] if row["cost_per_success_usd"] is not None else math.inf
        return (-row["success_rate_pct"], -row["avg_field_accuracy_pct"], cost, row["avg_latency_ms"])

    rows.sort(key=_sort_key)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def _build_cache_summary(run_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for run in run_results:
        if run.get("error"):
            continue
        cur = by_model.setdefault(
            run["model_key"],
            {
                "model_label": run["model_label"],
                "provider": run["provider"],
                "runs": 0,
                "cache_hits": 0,
                "cached_input_tokens_total": 0,
                "cache_write_tokens_total": 0,
            },
        )
        cur["runs"] += 1
        if run.get("cache_hit"):
            cur["cache_hits"] += 1
        cur["cached_input_tokens_total"] += run.get("cached_input_tokens", 0)
        cur["cache_write_tokens_total"] += run.get("cache_write_tokens", 0)

    entries = [
        {
            "model_key": model_key,
            "model_label": v["model_label"],
            "provider": v["provider"],
            "runs": v["runs"],
            "cache_hits": v["cache_hits"],
            "cache_hit_rate_pct": _round(_pct(v["cache_hits"], v["runs"]), 2),
            "cached_input_tokens_total": round(v["cached_input_tokens_total"]),
            "cached_input_tokens_avg": _round(
                v["cached_input_tokens_total"] / max(1, v["runs"]), 1
            ),
            "cache_write_tokens_total": round(v["cache_write_tokens_total"]),
            "cache_write_tokens_avg": _round(
                v["cache_write_tokens_total"] / max(1, v["runs"]), 1
            ),
        }
        for model_key, v in by_model.items()
    ]
    entries.sort(key=lambda e: (-e["cache_hit_rate_pct"], e["model_label"]))
    return entries


def _build_markdown_table(rows: list[dict[str, Any]]) -> str:
    head = [
        "| Rank | Model | Provider | Tier | Success % | pass^2 (>=2) % | pass^3 (>=3) % | pass^5 (>=5) % | pass^10 (>=10) % | pass^2 strict % | pass^3 strict % | pass^5 strict % | pass^10 strict % | Keys Found % | Avg Field % | Critical Field % | Variance % | Cost / Doc (USD) | Cost / Success (USD) | Avg Latency (ms) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body: list[str] = []
    for row in rows:
        def _fmt_pct(v: float | None) -> str:
            return "" if v is None else f"{v:.2f}"

        def _fmt_cost(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.4f}"

        body.append(
            f"| {row['rank']} | {row['model_label']} | {row['provider']} | {row['tier']} | "
            f"{row['success_rate_pct']:.2f} | "
            f"{_fmt_pct(row['pass_at_2_pct'])} | "
            f"{_fmt_pct(row['pass_at_3_pct'])} | "
            f"{_fmt_pct(row['pass_at_5_pct'])} | "
            f"{_fmt_pct(row['pass_at_10_pct'])} | "
            f"{_fmt_pct(row['pass_at_2_strict_pct'])} | "
            f"{_fmt_pct(row['pass_at_3_strict_pct'])} | "
            f"{_fmt_pct(row['pass_at_5_strict_pct'])} | "
            f"{_fmt_pct(row['pass_at_10_strict_pct'])} | "
            f"{row['avg_keys_found_pct']:.2f} | "
            f"{row['avg_field_accuracy_pct']:.2f} | "
            f"{row['avg_critical_accuracy_pct']:.2f} | "
            f"{row['field_accuracy_variance_pct']:.2f} | "
            f"{_fmt_cost(row['avg_cost_per_doc_usd'])} | "
            f"{_fmt_cost(row['cost_per_success_usd'])} | "
            f"{row['avg_latency_ms']:.1f} |"
        )
    return "\n".join([*head, *body])


def _collect_value_nodes(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(node, dict):
        return out
    for value in node.values():
        if isinstance(value, dict) and "value" in value:
            out.append({"value": value["value"], "critical": bool(value.get("critical"))})
            continue
        if isinstance(value, dict):
            out.extend(_collect_value_nodes(value))
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    out.extend(_collect_value_nodes(item))
    return out


def _is_labeled_value(value: Any) -> bool:
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, list):
        return any(len(str(item or "").strip()) > 0 for item in value)
    if isinstance(value, bool):
        return True
    return isinstance(value, (int, float))


def _summarize_dataset(documents: list[dict[str, Any]]) -> dict[str, Any]:
    documents_per_domain: dict[str, int] = {}
    total_keys = 0
    labeled_keys = 0
    critical_keys = 0
    labeled_critical_keys = 0

    for doc in documents:
        documents_per_domain[doc["domain"]] = documents_per_domain.get(doc["domain"], 0) + 1
        for node in _collect_value_nodes(doc["ground_truth_raw"]):
            total_keys += 1
            if node["critical"]:
                critical_keys += 1
            if _is_labeled_value(node["value"]):
                labeled_keys += 1
                if node["critical"]:
                    labeled_critical_keys += 1

    return {
        "total_documents": len(documents),
        "documents_per_domain": documents_per_domain,
        "total_keys": total_keys,
        "labeled_keys": labeled_keys,
        "critical_keys": critical_keys,
        "labeled_critical_keys": labeled_critical_keys,
    }


def _iso_now() -> str:
    """Match JS `new Date().toISOString()`: millisecond precision (3 digits)."""
    now = datetime.now(UTC)
    millis = now.microsecond // 1000
    return now.strftime("%Y-%m-%dT%H:%M:%S") + f".{millis:03d}Z"


def _timestamp_for_filename(iso: str | None = None) -> str:
    return (iso or _iso_now()).replace(":", "-").replace(".", "-")


def _json_dumps(value: Any) -> str:
    """Match JS JSON.stringify(value, null, 2): 2-space indent, no sort, no
    trailing newline (caller adds one)."""
    return json.dumps(value, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root: Path = args.repo_root.absolute()
    checkpoint_dir: Path = (
        args.checkpoint_dir.absolute() if args.checkpoint_dir else repo_root / "artifacts" / "checkpoints"
    )
    output_dir: Path = args.output_dir.absolute() if args.output_dir else repo_root / "artifacts"

    runs_path = checkpoint_dir / "runs.jsonl"
    state_path = checkpoint_dir / "state.json"
    config_path = repo_root / "config" / "models.public.json"
    manifest_path = repo_root / "dataset" / "manifest.json"

    if not runs_path.exists():
        raise SystemExit(f"Checkpoint runs log not found: {runs_path}")

    state = json.loads(state_path.read_text()) if state_path.exists() and state_path.read_text().strip() else {}
    options = state.get("options") or {}
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    latest_by_task: dict[str, dict[str, Any]] = {}
    for line in runs_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if parsed.get("task_key") and parsed.get("metrics") and parsed.get("debug"):
            latest_by_task[parsed["task_key"]] = parsed

    run_records = list(latest_by_task.values())
    if not run_records:
        raise SystemExit("No valid records found in checkpoint log.")

    raw_domains = options.get("domains")
    selected_domains: list[str] = (
        [d for d in (_normalize_domain(v) for v in raw_domains) if d]
        if isinstance(raw_domains, list)
        else []
    )
    domain_filter = set(selected_domains)
    raw_max = options.get("max_documents_per_domain")
    max_documents_per_domain: int | None = (
        int(raw_max) if isinstance(raw_max, (int, float)) and float(raw_max) > 0 else None
    )

    selected_documents: list[dict[str, Any]] = []
    for domain in manifest.get("domains") or []:
        normalized_domain = _normalize_domain(domain.get("id"))
        if domain_filter and normalized_domain not in domain_filter:
            continue
        docs = domain.get("documents") if isinstance(domain.get("documents"), list) else []
        picked = docs if max_documents_per_domain is None else docs[:max_documents_per_domain]
        for doc in picked:
            gt_abs = (repo_root / doc["ground_truth"]).resolve()
            gt_raw = json.loads(gt_abs.read_text())
            selected_documents.append(
                {
                    "document_id": doc["document_id"],
                    "domain": doc["domain"],
                    "source_pdf": doc["source_pdf"],
                    "ground_truth_path": doc["ground_truth"],
                    "ground_truth_raw": gt_raw,
                }
            )

    dataset = _summarize_dataset(selected_documents)
    warnings: list[str] = []
    if dataset["labeled_keys"] == 0:
        warnings.append("Ground truth is not labeled yet. Fill expected values before trusting rankings.")
    if dataset["labeled_keys"] < dataset["total_keys"]:
        warnings.append(
            f"Only {dataset['labeled_keys']}/{dataset['total_keys']} keys are currently labeled."
        )
    for domain_name, count in dataset["documents_per_domain"].items():
        if count < 10:
            warnings.append(
                f'Domain "{domain_name}" has {count} documents; target is >= 10 for launch quality.'
            )

    runs_per_model = max(
        1, int(options.get("runs_per_model") or config.get("default_runs_per_model") or 1)
    )
    requested_by_model: dict[str, int] = {}
    requested_by_domain_model: dict[str, int] = {}
    for model in config.get("models") or []:
        model_key = _id_for_model(model["provider"], model["model_id"])
        requested_by_model[model_key] = len(selected_documents) * runs_per_model
        for document in selected_documents:
            key = f"{document['domain']}::{model_key}"
            requested_by_domain_model[key] = requested_by_domain_model.get(key, 0) + runs_per_model

    run_results = [r["metrics"] for r in run_records]
    debug_runs = [r["debug"] for r in run_records]

    leaderboard = _aggregate_rows(run_results, requested_by_model)
    by_domain = []
    for domain_name in sorted(dataset["documents_per_domain"].keys()):
        domain_runs = [r for r in run_results if r.get("domain") == domain_name]
        domain_requested: dict[str, int] = {}
        for key, value in requested_by_domain_model.items():
            candidate_domain, model_key = key.split("::", 1)
            if candidate_domain == domain_name:
                domain_requested[model_key] = value
        by_domain.append(
            {
                "domain": domain_name,
                "rows": _aggregate_rows(domain_runs, domain_requested),
            }
        )

    cache_summary = _build_cache_summary(run_results)
    generated_at = _iso_now()

    max_parallel_raw = options.get("max_parallel_requests")
    max_parallel_requests = (
        int(max_parallel_raw)
        if isinstance(max_parallel_raw, (int, float)) and float(max_parallel_raw) > 0
        else int(config.get("max_parallel_requests") or 1)
    )

    snapshot: dict[str, Any] = {
        "generated_at": generated_at,
        "benchmark_id": f"ocr-benchmark-rebuild-{_timestamp_for_filename(generated_at)}",
        "benchmark_description": config.get("description"),
        "options": {
            "runs_per_model": runs_per_model,
            "max_parallel_requests": max_parallel_requests,
            "provider_parallel": options.get("provider_parallel") is True,
            "selected_domains": selected_domains,
            "max_documents_per_domain": max_documents_per_domain,
        },
        "dataset": dataset,
        "leaderboard": leaderboard,
        "by_domain": by_domain,
        "run_count": len(run_results),
        "markdown_table": _build_markdown_table(leaderboard),
        "warnings": warnings,
        "cache_summary": cache_summary,
        "debug": {
            "documents": [
                {
                    "document_id": d["document_id"],
                    "domain": d["domain"],
                    "source_pdf": d["source_pdf"],
                    "ground_truth_path": d["ground_truth_path"],
                    "ground_truth": d["ground_truth_raw"],
                }
                for d in selected_documents
            ],
            "runs": debug_runs,
        },
    }

    timestamp = _timestamp_for_filename()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / f"snapshot-{timestamp}.json"
    snapshot_debug_path = output_dir / f"snapshot-{timestamp}.debug.json"
    latest_json_path = output_dir / "latest.json"
    latest_debug_path = output_dir / "latest.debug.json"
    latest_markdown_path = output_dir / "latest.md"

    debug_payload = snapshot.get("debug") or {"documents": [], "runs": []}
    public_payload = {k: v for k, v in snapshot.items() if k != "debug"}

    snapshot_path.write_text(_json_dumps(public_payload) + "\n")
    snapshot_debug_path.write_text(_json_dumps(debug_payload) + "\n")
    latest_json_path.write_text(_json_dumps(public_payload) + "\n")
    latest_debug_path.write_text(_json_dumps(debug_payload) + "\n")
    latest_markdown_path.write_text(str(public_payload.get("markdown_table") or "") + "\n")

    print(f"Rebuilt snapshot from checkpoint records: {len(run_records)}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Latest artifact: {latest_json_path}")
    print(f"Latest debug artifact: {latest_debug_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
