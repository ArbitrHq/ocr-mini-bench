"""Validate the bench_documents/ tree and dataset/manifest.json. Python port
of `scripts/validate_dataset.mjs`.

Exits 1 on any error (missing PDFs/GT, duplicate IDs, legacy schema, etc.).
Warnings are printed but non-fatal.

Usage:
    uv run python scripts/validate_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ocr_mini_bench.config.paths import PATHS

REPO_ROOT = PATHS.dataset.manifest.parent.parent
MANIFEST_PATH = PATHS.dataset.manifest
BENCH_DOCUMENTS_ROOT = REPO_ROOT / "bench_documents"


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _collect_value_nodes(node: Any) -> list[dict[str, Any]]:
    """Walk the GT tree and return all leaf descriptors that look like
    `{value, critical?, ...}`. Mirrors `collectValueNodes` in the TS file:
    descends into nested dicts and into lists of dicts."""
    out: list[dict[str, Any]] = []
    if not _is_record(node):
        return out
    for key, value in node.items():
        if _is_record(value) and "value" in value:
            out.append({"key": key, "value": value["value"], "critical": bool(value.get("critical"))})
            continue
        if _is_record(value):
            out.extend(_collect_value_nodes(value))
            continue
        if isinstance(value, list):
            for item in value:
                if _is_record(item):
                    out.extend(_collect_value_nodes(item))
    return out


def _rel(value: Path) -> str:
    """Repo-relative path with forward slashes for cross-platform parity."""
    try:
        return value.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(value)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if BENCH_DOCUMENTS_ROOT.is_dir():
        for entry in BENCH_DOCUMENTS_ROOT.iterdir():
            if not entry.is_dir():
                continue
            for child in entry.iterdir():
                if not child.is_dir():
                    continue
                if child.name in ("ground_truth", "reduced_size", "full_size", "full size"):
                    continue
                files = [p.name for p in child.iterdir() if p.is_file()]
                if any(f.lower().endswith(".pdf") for f in files):
                    warnings.append(
                        f"Nested PDF directory detected: {_rel(child)}. "
                        f"Keep canonical PDFs at the domain root and use standard subfolders "
                        f"(reduced_size/full_size) only."
                    )

    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"Failed to read/parse manifest: {MANIFEST_PATH}", file=sys.stderr)
        raise SystemExit(1) from e

    if not isinstance(manifest, dict) or not isinstance(manifest.get("domains"), list):
        raise SystemExit("Manifest is missing a valid domains array.")

    seen_document_ids: set[str] = set()
    seen_pdf_paths: set[str] = set()
    document_count = 0
    total_comparable_keys = 0

    for domain in manifest["domains"]:
        if not domain.get("id") or not isinstance(domain.get("documents"), list):
            errors.append(f"Invalid domain entry in manifest: {json.dumps(domain)}")
            continue

        doc_count_attr = domain.get("document_count")
        if isinstance(doc_count_attr, int) and doc_count_attr != len(domain["documents"]):
            warnings.append(
                f"Domain {domain['id']} has document_count={doc_count_attr}, "
                f"but documents.length={len(domain['documents'])}."
            )

        for document in domain["documents"]:
            document_count += 1
            doc_id = document.get("document_id", "<missing-id>") if isinstance(document, dict) else "<missing-id>"
            context = f"{domain['id']}:{doc_id}"

            if not isinstance(document, dict) or not document.get("document_id") \
                    or not document.get("source_pdf") or not document.get("ground_truth"):
                errors.append(f"Missing required fields in manifest document entry ({context}).")
                continue

            if document["document_id"] in seen_document_ids:
                errors.append(f"Duplicate document_id: {document['document_id']}")
            seen_document_ids.add(document["document_id"])

            if document["source_pdf"] in seen_pdf_paths:
                warnings.append(f"source_pdf reused by multiple entries: {document['source_pdf']}")
            seen_pdf_paths.add(document["source_pdf"])

            pdf_abs = (REPO_ROOT / document["source_pdf"]).resolve()
            gt_abs = (REPO_ROOT / document["ground_truth"]).resolve()

            if not pdf_abs.exists():
                errors.append(f"Missing source PDF for {context}: {_rel(pdf_abs)}")

            try:
                gt_raw = gt_abs.read_text()
            except OSError:
                errors.append(f"Missing ground-truth JSON for {context}: {_rel(gt_abs)}")
                continue

            try:
                gt = json.loads(gt_raw)
            except json.JSONDecodeError:
                errors.append(f"Invalid JSON in ground-truth for {context}: {_rel(gt_abs)}")
                continue

            if isinstance(gt, dict) and isinstance(gt.get("keys"), list):
                errors.append(f"Legacy keys[] schema detected for {context}: {_rel(gt_abs)}")
                continue

            value_nodes = _collect_value_nodes(gt)
            if not value_nodes:
                errors.append(f"No comparable value nodes found for {context}: {_rel(gt_abs)}")
                continue

            total_comparable_keys += len(value_nodes)

    print(f"Manifest: {_rel(MANIFEST_PATH)}")
    print(f"Domains: {len(manifest['domains'])}")
    print(f"Documents: {document_count}")
    print(f"Comparable keys: {total_comparable_keys}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("\nDataset validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
