"""Generate dataset/manifest.json from bench_documents/. Python port of
`scripts/bootstrap_dataset.mjs`.

Usage:
    uv run python scripts/bootstrap_dataset.py [--create-placeholders] [--output=PATH]

By default rewrites the canonical dataset/manifest.json. Pass --output to
target a different path (used in tests).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocr_mini_bench.config.paths import PATHS

REPO_ROOT = PATHS.dataset.manifest.parent.parent
BENCH_DOCUMENTS_ROOT = REPO_ROOT / "bench_documents"
MANIFEST_PATH = PATHS.dataset.manifest

DOMAIN_DIRS: dict[str, str] = {
    "Invoices": "invoices",
    "Receipts": "receipts",
    "Logistics": "logistics",
}

KEY_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "invoices": [
        {"name": "invoice_number", "critical": True, "type": "string"},
        {"name": "invoice_date", "critical": True, "type": "date"},
        {"name": "due_date", "critical": False, "type": "date"},
        {"name": "supplier_name", "critical": True, "type": "string"},
        {"name": "customer_name", "critical": False, "type": "string"},
        {"name": "subtotal_amount", "critical": False, "type": "float"},
        {"name": "vat_amount", "critical": False, "type": "float"},
        {"name": "total_amount", "critical": True, "type": "float"},
        {"name": "currency", "critical": True, "type": "string"},
        {"name": "payment_reference", "critical": False, "type": "string"},
    ],
    "receipts": [
        {"name": "vendor_name", "critical": True, "type": "string"},
        {"name": "receipt_date", "critical": True, "type": "date"},
        {"name": "receipt_time", "critical": False, "type": "string"},
        {"name": "total_amount", "critical": True, "type": "float"},
        {"name": "total_tax", "critical": False, "type": "float"},
        {"name": "tax_rate", "critical": False, "type": "float"},
        {"name": "currency", "critical": False, "type": "string"},
        {"name": "transaction_number", "critical": True, "type": "string"},
        {"name": "payment_method", "critical": False, "type": "string"},
        {"name": "store_name", "critical": False, "type": "string"},
    ],
    "logistics": [
        {"name": "order_number", "critical": True, "type": "string"},
        {"name": "bill_of_lading_number", "critical": True, "type": "string"},
        {"name": "shipper_name", "critical": True, "type": "string"},
        {"name": "consignee_name", "critical": True, "type": "string"},
        {"name": "origin_address", "critical": False, "type": "string"},
        {"name": "destination_address", "critical": False, "type": "string"},
        {"name": "pickup_date", "critical": False, "type": "date"},
        {"name": "delivery_date", "critical": False, "type": "date"},
        {"name": "container_number", "critical": True, "type": "string"},
        {"name": "total_weight", "critical": False, "type": "float"},
    ],
}

_EXT_RE = re.compile(r"\.[^.]+$")
_NON_ALNUM_DASH = re.compile(r"[^a-z0-9]+")
_NON_ALNUM_UNDER = re.compile(r"[^a-z0-9]+")
_MULTIPLE_UNDERSCORES = re.compile(r"_+")
_LEADING_TRAILING_DASH = re.compile(r"(^-|-$)")
_LEADING_TRAILING_UNDER = re.compile(r"^_+|_+$")


def _slugify(value: str) -> str:
    out = value.lower()
    out = _EXT_RE.sub("", out)
    out = _NON_ALNUM_DASH.sub("-", out)
    out = _LEADING_TRAILING_DASH.sub("", out)
    return out[:80]


def _normalize_for_match(value: str) -> str:
    out = value.lower()
    out = _EXT_RE.sub("", out)
    out = _NON_ALNUM_DASH.sub("", out)
    return out


def _to_safe_json_basename(value: str) -> str:
    out = value.lower()
    out = _EXT_RE.sub("", out)
    out = _NON_ALNUM_UNDER.sub("_", out)
    out = _LEADING_TRAILING_UNDER.sub("", out)
    out = _MULTIPLE_UNDERSCORES.sub("_", out)
    return out[:120]


def _list_files_by_ext(dir_path: Path, ext_re: re.Pattern[str]) -> list[str]:
    if not dir_path.is_dir():
        return []
    names = [p.name for p in dir_path.iterdir() if p.is_file() and ext_re.search(p.name)]
    return sorted(names)


def _find_matching_ground_truth(
    pdf_filename: str,
    json_files: list[str],
    json_by_normalized_base: dict[str, list[str]],
) -> str | None:
    pdf_base = _EXT_RE.sub("", pdf_filename)
    exact = f"{pdf_base}.json"
    if exact in json_files:
        return exact

    normalized = _normalize_for_match(pdf_base)
    candidates = json_by_normalized_base.get(normalized, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        preferred = f"{_to_safe_json_basename(pdf_base)}.json"
        if preferred in candidates:
            return preferred
    return None


def _ensure_placeholder_ground_truth(absolute_gt_path: Path, domain: str) -> bool:
    if absolute_gt_path.exists():
        return False
    template = {
        key["name"]: {"value": None, "critical": key["critical"], "type": key["type"]}
        for key in KEY_TEMPLATES.get(domain, [])
    }
    absolute_gt_path.write_text(json.dumps(template, indent=2) + "\n")
    return True


def _now_iso() -> str:
    """Match JS `new Date().toISOString()`: `YYYY-MM-DDTHH:MM:SS.mmmZ` with
    three-digit milliseconds (not Python's six-digit microseconds)."""
    now = datetime.now(UTC)
    millis = now.microsecond // 1000
    return now.strftime("%Y-%m-%dT%H:%M:%S") + f".{millis:03d}Z"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--create-placeholders", action="store_true")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)

    domains: list[dict[str, Any]] = []
    created_gt_files = 0
    reused_gt_files = 0
    missing_gt_files: list[str] = []

    for source_dir, domain in DOMAIN_DIRS.items():
        source_path = BENCH_DOCUMENTS_ROOT / source_dir
        ground_truth_dir = source_path / "ground_truth"

        pdf_files = _list_files_by_ext(source_path, re.compile(r"\.pdf$", re.IGNORECASE))
        json_files = _list_files_by_ext(ground_truth_dir, re.compile(r"\.json$", re.IGNORECASE))

        json_by_normalized_base: dict[str, list[str]] = {}
        for jf in json_files:
            json_by_normalized_base.setdefault(_normalize_for_match(jf), []).append(jf)

        ground_truth_dir.mkdir(parents=True, exist_ok=True)

        documents: list[dict[str, Any]] = []
        for filename in pdf_files:
            document_id = f"{domain}-{_slugify(filename)}"
            relative_pdf_path = f"bench_documents/{source_dir}/{filename}"

            matched_gt = _find_matching_ground_truth(filename, json_files, json_by_normalized_base)
            gt_filename = matched_gt or f"{_to_safe_json_basename(filename)}.json"
            absolute_gt_path = ground_truth_dir / gt_filename
            relative_gt_path = f"bench_documents/{source_dir}/ground_truth/{gt_filename}"

            if matched_gt:
                reused_gt_files += 1
            elif args.create_placeholders:
                if _ensure_placeholder_ground_truth(absolute_gt_path, domain):
                    created_gt_files += 1
            else:
                missing_gt_files.append(relative_gt_path)

            documents.append(
                {
                    "document_id": document_id,
                    "domain": domain,
                    "source_pdf": relative_pdf_path,
                    "ground_truth": relative_gt_path,
                }
            )

        domains.append(
            {
                "id": domain,
                "source_directory": f"bench_documents/{source_dir}",
                "document_count": len(documents),
                "documents": documents,
            }
        )

    if missing_gt_files:
        print(
            f"Missing {len(missing_gt_files)} ground-truth file(s). "
            f"Add labels or run with --create-placeholders:",
            file=sys.stderr,
        )
        for f in missing_gt_files[:25]:
            print(f"- {f}", file=sys.stderr)
        if len(missing_gt_files) > 25:
            print(f"- ...and {len(missing_gt_files) - 25} more", file=sys.stderr)
        return 1

    manifest = {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "domains": domains,
    }

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")

    total_documents = sum(d["document_count"] for d in domains)
    print(f"Manifest written: {out_path}")
    print(f"Documents indexed: {total_documents}")
    print(f"Ground-truth files reused: {reused_gt_files}")
    print(f"Ground-truth placeholders created: {created_gt_files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
