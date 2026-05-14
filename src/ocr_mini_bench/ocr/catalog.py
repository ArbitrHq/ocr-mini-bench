"""Flatten the provider/model catalog. Mirrors `src/ocr/catalog.ts`."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.model_catalog import MODEL_CATALOG, ModelProvider


@dataclass(frozen=True)
class FlatCatalogModel:
    provider: ModelProvider
    model_id: str
    label: str


def flatten_catalog_models() -> list[FlatCatalogModel]:
    out: list[FlatCatalogModel] = []
    for provider, models in MODEL_CATALOG.items():
        for option in models:
            out.append(FlatCatalogModel(provider=provider, model_id=option.id, label=option.label))
    return out
