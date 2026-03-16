import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..');
const MANIFEST_PATH = path.resolve(REPO_ROOT, 'dataset/manifest.json');
const BENCH_DOCUMENTS_ROOT = path.resolve(REPO_ROOT, 'bench_documents');

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function collectValueNodes(input, output = []) {
  if (!isRecord(input)) return output;
  for (const [key, value] of Object.entries(input)) {
    if (isRecord(value) && Object.prototype.hasOwnProperty.call(value, 'value')) {
      output.push({ key, value: value.value, critical: Boolean(value.critical) });
      continue;
    }
    if (isRecord(value)) {
      collectValueNodes(value, output);
      continue;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        if (isRecord(item)) collectValueNodes(item, output);
      }
    }
  }
  return output;
}

function rel(value) {
  return path.relative(REPO_ROOT, value).split(path.sep).join('/');
}

async function main() {
  const errors = [];
  const warnings = [];

  try {
    const domainDirs = await fs.readdir(BENCH_DOCUMENTS_ROOT, { withFileTypes: true });
    for (const entry of domainDirs) {
      if (!entry.isDirectory()) continue;
      const domainPath = path.resolve(BENCH_DOCUMENTS_ROOT, entry.name);
      const nested = await fs.readdir(domainPath, { withFileTypes: true });
      for (const child of nested) {
        if (!child.isDirectory() || child.name === 'ground_truth') continue;
        if (child.name === 'reduced_size' || child.name === 'full_size' || child.name === 'full size') continue;
        const childPath = path.resolve(domainPath, child.name);
        const files = await fs.readdir(childPath);
        if (files.some((file) => file.toLowerCase().endsWith('.pdf'))) {
          warnings.push(
            `Nested PDF directory detected: ${rel(childPath)}. Keep canonical PDFs at the domain root and use standard subfolders (reduced_size/full_size) only.`
          );
        }
      }
    }
  } catch {
    // Ignore bench_documents shape checks when directory is unavailable.
  }

  let manifest;
  try {
    const raw = await fs.readFile(MANIFEST_PATH, 'utf8');
    manifest = JSON.parse(raw);
  } catch (error) {
    console.error(`Failed to read/parse manifest: ${MANIFEST_PATH}`);
    throw error;
  }

  if (!Array.isArray(manifest?.domains)) {
    throw new Error('Manifest is missing a valid domains array.');
  }

  const seenDocumentIds = new Set();
  const seenPdfPaths = new Set();

  let documentCount = 0;
  let totalComparableKeys = 0;

  for (const domain of manifest.domains) {
    if (!domain?.id || !Array.isArray(domain?.documents)) {
      errors.push(`Invalid domain entry in manifest: ${JSON.stringify(domain)}`);
      continue;
    }

    if (typeof domain.document_count === 'number' && domain.document_count !== domain.documents.length) {
      warnings.push(
        `Domain ${domain.id} has document_count=${domain.document_count}, but documents.length=${domain.documents.length}.`
      );
    }

    for (const document of domain.documents) {
      documentCount += 1;
      const context = `${domain.id}:${document?.document_id ?? '<missing-id>'}`;

      if (!document?.document_id || !document?.source_pdf || !document?.ground_truth) {
        errors.push(`Missing required fields in manifest document entry (${context}).`);
        continue;
      }

      if (seenDocumentIds.has(document.document_id)) {
        errors.push(`Duplicate document_id: ${document.document_id}`);
      }
      seenDocumentIds.add(document.document_id);

      if (seenPdfPaths.has(document.source_pdf)) {
        warnings.push(`source_pdf reused by multiple entries: ${document.source_pdf}`);
      }
      seenPdfPaths.add(document.source_pdf);

      const pdfAbs = path.resolve(REPO_ROOT, document.source_pdf);
      const gtAbs = path.resolve(REPO_ROOT, document.ground_truth);

      try {
        await fs.access(pdfAbs);
      } catch {
        errors.push(`Missing source PDF for ${context}: ${rel(pdfAbs)}`);
      }

      let gtRaw;
      try {
        gtRaw = await fs.readFile(gtAbs, 'utf8');
      } catch {
        errors.push(`Missing ground-truth JSON for ${context}: ${rel(gtAbs)}`);
        continue;
      }

      let gt;
      try {
        gt = JSON.parse(gtRaw);
      } catch {
        errors.push(`Invalid JSON in ground-truth for ${context}: ${rel(gtAbs)}`);
        continue;
      }

      if (Array.isArray(gt?.keys)) {
        errors.push(`Legacy keys[] schema detected for ${context}: ${rel(gtAbs)}`);
        continue;
      }

      const valueNodes = collectValueNodes(gt);
      if (valueNodes.length === 0) {
        errors.push(`No comparable value nodes found for ${context}: ${rel(gtAbs)}`);
        continue;
      }

      totalComparableKeys += valueNodes.length;
    }
  }

  console.log(`Manifest: ${rel(MANIFEST_PATH)}`);
  console.log(`Domains: ${manifest.domains.length}`);
  console.log(`Documents: ${documentCount}`);
  console.log(`Comparable keys: ${totalComparableKeys}`);

  if (warnings.length > 0) {
    console.log('\nWarnings:');
    for (const warning of warnings) console.log(`- ${warning}`);
  }

  if (errors.length > 0) {
    console.log('\nErrors:');
    for (const error of errors) console.log(`- ${error}`);
    process.exitCode = 1;
    return;
  }

  console.log('\nDataset validation passed.');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
