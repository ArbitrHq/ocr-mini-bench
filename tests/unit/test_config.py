"""Tests for config/paths, model_catalog, backend_config.

The TS config modules have no `.test.ts` counterparts; these tests guard the
parity-critical bits (provider inference rules, pricing keys, path layout)
that downstream artifact generation depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_mini_bench.config.backend_config import (
    PRICING_REGISTRY,
    infer_provider_from_model_id,
    load_pricing_registry_from_backend_config,
)
from ocr_mini_bench.config.model_catalog import (
    MODEL_CATALOG,
    default_model_for_provider,
)
from ocr_mini_bench.config.paths import PATHS, REPO_ROOT, build_paths


@pytest.mark.unit
class TestInferProviderFromModelId:
    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("claude-opus-4-1-20250805", "anthropic"),
            ("Claude-Sonnet-4", "anthropic"),
            ("gemini-3.1-flash-lite-preview", "google"),
            ("mistral-large-latest", "mistral"),
            ("ministral-3b", "mistral"),
            ("pixtral-12b", "mistral"),
            ("codestral-latest", "mistral"),
            ("devstral-small", "mistral"),
            ("voxtral-mini", "mistral"),
            ("gpt-5", "openai"),
            ("o3", "openai"),
            ("some-random-id", "openai"),
        ],
    )
    def test_known_prefixes(self, model_id: str, expected: str) -> None:
        assert infer_provider_from_model_id(model_id) == expected


@pytest.mark.unit
class TestPricingRegistry:
    def test_load_returns_same_dict(self) -> None:
        assert load_pricing_registry_from_backend_config() is PRICING_REGISTRY

    def test_has_default_fallback(self) -> None:
        assert "default" in PRICING_REGISTRY
        default = PRICING_REGISTRY["default"]
        assert default.input == 0.0
        assert default.output == 0.0

    def test_gemini_31_flash_lite_has_cache_pricing(self) -> None:
        # Cache instrumentation depends on this entry having cache_input set —
        # it's the canonical smoke-test model.
        entry = PRICING_REGISTRY["gemini-3.1-flash-lite-preview"]
        assert entry.cache_input == 0.025

    def test_pricing_keys_unique(self) -> None:
        # dict literal already enforces uniqueness, but assert as a sanity check
        # that the source file wasn't accidentally edited to duplicate a key.
        assert len(PRICING_REGISTRY) == len(set(PRICING_REGISTRY.keys()))


@pytest.mark.unit
class TestModelCatalog:
    def test_all_providers_present(self) -> None:
        assert set(MODEL_CATALOG.keys()) == {"anthropic", "openai", "google", "mistral"}

    def test_default_model_is_first(self) -> None:
        for provider, options in MODEL_CATALOG.items():
            assert default_model_for_provider(provider) == options[0].id


@pytest.mark.unit
class TestPaths:
    def test_repo_root_resolves_to_expected_layout(self) -> None:
        # Repo root should contain the canonical top-level directories
        # the benchmark reads at startup.
        assert (REPO_ROOT / "src").is_dir()
        assert (REPO_ROOT / "config").is_dir()
        assert (REPO_ROOT / "prompts").is_dir()
        assert (REPO_ROOT / "dataset").is_dir()

    def test_artifact_subpaths_under_root(self) -> None:
        assert PATHS.artifacts.checkpoints == PATHS.artifacts.root / "checkpoints"
        assert PATHS.artifacts.postprocess == PATHS.artifacts.root / "postprocess"
        assert PATHS.postprocess.raw_jsonl.parent == PATHS.postprocess.root

    def test_build_paths_accepts_override(self, tmp_path: Path) -> None:
        paths = build_paths(repo_root=tmp_path)
        assert paths.artifacts.root == tmp_path / "artifacts"
        assert paths.dataset.manifest == tmp_path / "dataset" / "manifest.json"
