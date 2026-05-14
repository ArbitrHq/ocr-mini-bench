"""Load the benchmark dataset from disk. Mirrors `src/benchmark/dataset.ts`.

Reads `config/models.public.json` and `dataset/manifest.json` from the repo
root, walks the manifest to load each document's ground-truth JSON, and
normalizes it into the `PreparedBenchmarkDocument` shape that the scorer
and CLIs consume.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config.paths import PATHS, REPO_ROOT
from .normalize_ground_truth import (
    GroundTruthFallback,
    normalize_ground_truth_document,
)
from .types import (
    BenchmarkConfig,
    BenchmarkManifest,
    DatasetSummary,
    PreparedBenchmarkDocument,
)


def _read_json_file(target: Path) -> object:
    return json.loads(target.read_text(encoding="utf-8"))


def _normalize_domain(value: str) -> str:
    return value.strip().lower()


def load_benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(_read_json_file(PATHS.config.models))


def load_benchmark_manifest() -> BenchmarkManifest:
    return BenchmarkManifest.model_validate(_read_json_file(PATHS.dataset.manifest))


def load_prepared_documents(
    *,
    domains: list[str] | None = None,
    max_documents_per_domain: int | None = None,
) -> list[PreparedBenchmarkDocument]:
    manifest = load_benchmark_manifest()
    domain_filter = {_normalize_domain(d) for d in (domains or [])}
    cap = max_documents_per_domain if (
        isinstance(max_documents_per_domain, int) and max_documents_per_domain > 0
    ) else None

    output: list[PreparedBenchmarkDocument] = []

    for domain in manifest.domains:
        normalized_domain = _normalize_domain(domain.id)
        if domain_filter and normalized_domain not in domain_filter:
            continue

        selected = domain.documents if cap is None else domain.documents[:cap]

        for document in selected:
            source_abs = (REPO_ROOT / document.source_pdf).resolve()
            gt_abs = (REPO_ROOT / document.ground_truth).resolve()
            raw_ground_truth = _read_json_file(gt_abs)
            ground_truth = normalize_ground_truth_document(
                raw_ground_truth,
                GroundTruthFallback(
                    document_id=document.document_id,
                    domain=document.domain,
                    source_pdf=document.source_pdf,
                ),
            )
            output.append(
                PreparedBenchmarkDocument(
                    document_id=document.document_id,
                    domain=document.domain,
                    source_pdf=document.source_pdf,
                    source_pdf_abs=str(source_abs),
                    ground_truth_abs=str(gt_abs),
                    ground_truth_raw=raw_ground_truth,
                    ground_truth=ground_truth,
                )
            )

    return output


def summarize_dataset(documents: list[PreparedBenchmarkDocument]) -> DatasetSummary:
    documents_per_domain: dict[str, int] = {}
    total_keys = 0
    labeled_keys = 0
    critical_keys = 0
    labeled_critical_keys = 0

    for document in documents:
        documents_per_domain[document.domain] = (
            documents_per_domain.get(document.domain, 0) + 1
        )
        for key in document.ground_truth.keys:
            total_keys += 1
            if key.critical:
                critical_keys += 1

            expected = key.expected
            has_label = (
                (isinstance(expected, str) and bool(expected.strip()))
                or (
                    isinstance(expected, list)
                    and any(isinstance(v, str) and v.strip() for v in expected)
                )
            )

            if has_label:
                labeled_keys += 1
                if key.critical:
                    labeled_critical_keys += 1

    return DatasetSummary(
        total_documents=len(documents),
        documents_per_domain=documents_per_domain,
        total_keys=total_keys,
        labeled_keys=labeled_keys,
        critical_keys=critical_keys,
        labeled_critical_keys=labeled_critical_keys,
    )
