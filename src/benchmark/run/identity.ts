export function idForModel(provider: string, modelId: string): string {
  return `${provider}:${modelId}`;
}

export function buildBenchmarkRunTaskKey(params: {
  modelKey: string;
  domain: string;
  documentId: string;
  runNumber: number;
}): string {
  return `${params.modelKey}::${params.domain}::${params.documentId}::${params.runNumber}`;
}

export function buildBenchmarkId(): string {
  return `ocr-benchmark-${new Date().toISOString().replace(/[:.]/g, '-')}`;
}

export function toRepoRelativePath(absPath: string, repoRoot: string): string {
  return path.relative(repoRoot, absPath).split(path.sep).join('/');
}
import path from 'node:path';
