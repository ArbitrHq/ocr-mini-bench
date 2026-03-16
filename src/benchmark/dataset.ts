import { promises as fs } from 'node:fs';
import path from 'node:path';
import type {
  BenchmarkConfig,
  BenchmarkManifest,
  DatasetSummary,
  PreparedBenchmarkDocument,
} from './types';
import { normalizeGroundTruthDocument } from './normalize-ground-truth';

const BENCHMARK_ROOT = process.cwd();

const MANIFEST_PATH = path.resolve(BENCHMARK_ROOT, 'dataset/manifest.json');
const CONFIG_PATH = path.resolve(BENCHMARK_ROOT, 'config/models.public.json');

async function readJsonFile<T>(targetPath: string): Promise<T> {
  const raw = await fs.readFile(targetPath, 'utf8');
  return JSON.parse(raw) as T;
}

function normalizeDomain(value: string): string {
  return value.trim().toLowerCase();
}

export async function loadBenchmarkConfig(): Promise<BenchmarkConfig> {
  return readJsonFile<BenchmarkConfig>(CONFIG_PATH);
}

export async function loadBenchmarkManifest(): Promise<BenchmarkManifest> {
  return readJsonFile<BenchmarkManifest>(MANIFEST_PATH);
}

export async function loadPreparedDocuments(options?: {
  domains?: string[];
  maxDocumentsPerDomain?: number;
}): Promise<PreparedBenchmarkDocument[]> {
  const manifest = await loadBenchmarkManifest();
  const domainFilter = new Set((options?.domains ?? []).map(normalizeDomain));
  const maxDocumentsPerDomain =
    typeof options?.maxDocumentsPerDomain === 'number' && options.maxDocumentsPerDomain > 0
      ? options.maxDocumentsPerDomain
      : null;

  const output: PreparedBenchmarkDocument[] = [];

  for (const domain of manifest.domains) {
    const normalizedDomain = normalizeDomain(domain.id);
    if (domainFilter.size > 0 && !domainFilter.has(normalizedDomain)) {
      continue;
    }

    const selectedDocuments =
      maxDocumentsPerDomain === null
        ? domain.documents
        : domain.documents.slice(0, maxDocumentsPerDomain);

    for (const document of selectedDocuments) {
      const sourceAbs = path.resolve(process.cwd(), document.source_pdf);
      const gtAbs = path.resolve(process.cwd(), document.ground_truth);
      const rawGroundTruth = await readJsonFile<unknown>(gtAbs);
      const groundTruth = normalizeGroundTruthDocument(rawGroundTruth, {
        documentId: document.document_id,
        domain: document.domain,
        sourcePdf: document.source_pdf,
      });

      output.push({
        document_id: document.document_id,
        domain: document.domain,
        source_pdf: document.source_pdf,
        source_pdf_abs: sourceAbs,
        ground_truth_abs: gtAbs,
        ground_truth_raw: rawGroundTruth,
        ground_truth: groundTruth,
      });
    }
  }

  return output;
}

export function summarizeDataset(documents: PreparedBenchmarkDocument[]): DatasetSummary {
  const documentsPerDomain: Record<string, number> = {};

  let totalKeys = 0;
  let labeledKeys = 0;
  let criticalKeys = 0;
  let labeledCriticalKeys = 0;

  for (const document of documents) {
    documentsPerDomain[document.domain] = (documentsPerDomain[document.domain] ?? 0) + 1;

    for (const key of document.ground_truth.keys) {
      totalKeys += 1;
      if (key.critical) {
        criticalKeys += 1;
      }

      const expected = key.expected;
      const hasLabel =
        (typeof expected === 'string' && expected.trim().length > 0) ||
        (Array.isArray(expected) && expected.some((value) => value.trim().length > 0));

      if (hasLabel) {
        labeledKeys += 1;
        if (key.critical) {
          labeledCriticalKeys += 1;
        }
      }
    }
  }

  return {
    total_documents: documents.length,
    documents_per_domain: documentsPerDomain,
    total_keys: totalKeys,
    labeled_keys: labeledKeys,
    critical_keys: criticalKeys,
    labeled_critical_keys: labeledCriticalKeys,
  };
}
