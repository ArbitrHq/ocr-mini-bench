import { MODEL_CATALOG, type ModelProvider } from '../config/model-catalog';

export function flattenCatalogModels(): Array<{ provider: ModelProvider; modelId: string; label: string }> {
  const models: Array<{ provider: ModelProvider; modelId: string; label: string }> = [];
  (Object.keys(MODEL_CATALOG) as ModelProvider[]).forEach((provider) => {
    MODEL_CATALOG[provider].forEach((model) => {
      models.push({ provider, modelId: model.id, label: model.label });
    });
  });
  return models;
}
