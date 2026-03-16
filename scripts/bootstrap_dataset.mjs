import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const BENCHMARK_ROOT = path.resolve(SCRIPT_DIR, "..");
const REPO_ROOT = BENCHMARK_ROOT;
const BENCH_DOCUMENTS_ROOT = path.resolve(REPO_ROOT, "bench_documents");
const MANIFEST_PATH = path.resolve(BENCHMARK_ROOT, "dataset/manifest.json");

const DOMAIN_DIRS = {
  Invoices: "invoices",
  Receipts: "receipts",
  Logistics: "logistics",
};

const KEY_TEMPLATES = {
  invoices: [
    { name: "invoice_number", critical: true, type: "string" },
    { name: "invoice_date", critical: true, type: "date" },
    { name: "due_date", critical: false, type: "date" },
    { name: "supplier_name", critical: true, type: "string" },
    { name: "customer_name", critical: false, type: "string" },
    { name: "subtotal_amount", critical: false, type: "float" },
    { name: "vat_amount", critical: false, type: "float" },
    { name: "total_amount", critical: true, type: "float" },
    { name: "currency", critical: true, type: "string" },
    { name: "payment_reference", critical: false, type: "string" },
  ],
  receipts: [
    { name: "vendor_name", critical: true, type: "string" },
    { name: "receipt_date", critical: true, type: "date" },
    { name: "receipt_time", critical: false, type: "string" },
    { name: "total_amount", critical: true, type: "float" },
    { name: "total_tax", critical: false, type: "float" },
    { name: "tax_rate", critical: false, type: "float" },
    { name: "currency", critical: false, type: "string" },
    { name: "transaction_number", critical: true, type: "string" },
    { name: "payment_method", critical: false, type: "string" },
    { name: "store_name", critical: false, type: "string" },
  ],
  logistics: [
    { name: "order_number", critical: true, type: "string" },
    { name: "bill_of_lading_number", critical: true, type: "string" },
    { name: "shipper_name", critical: true, type: "string" },
    { name: "consignee_name", critical: true, type: "string" },
    { name: "origin_address", critical: false, type: "string" },
    { name: "destination_address", critical: false, type: "string" },
    { name: "pickup_date", critical: false, type: "date" },
    { name: "delivery_date", critical: false, type: "date" },
    { name: "container_number", critical: true, type: "string" },
    { name: "total_weight", critical: false, type: "float" },
  ],
};

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "")
    .slice(0, 80);
}

function normalizeForMatch(value) {
  return value
    .toLowerCase()
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9]+/g, "");
}

function toSafeJsonBasename(value) {
  return value
    .toLowerCase()
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_+/g, "_")
    .slice(0, 120);
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function fileExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function listFilesByExtension(dirPath, extensionRegex) {
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isFile() && extensionRegex.test(entry.name))
      .map((entry) => entry.name)
      .sort((a, b) => a.localeCompare(b));
  } catch {
    return [];
  }
}

function findMatchingGroundTruth(pdfFilename, jsonFiles, jsonByNormalizedBase) {
  const pdfBase = pdfFilename.replace(/\.[^.]+$/i, "");
  const exact = `${pdfBase}.json`;
  if (jsonFiles.includes(exact)) {
    return exact;
  }

  const normalized = normalizeForMatch(pdfBase);
  const candidates = jsonByNormalizedBase.get(normalized) ?? [];

  if (candidates.length === 1) {
    return candidates[0];
  }

  if (candidates.length > 1) {
    const preferred = `${toSafeJsonBasename(pdfBase)}.json`;
    if (candidates.includes(preferred)) {
      return preferred;
    }
  }

  return null;
}

async function ensurePlaceholderGroundTruth({
  absoluteGtPath,
  domain,
}) {
  if (await fileExists(absoluteGtPath)) {
    return false;
  }

  const templateFields = Object.fromEntries(
    (KEY_TEMPLATES[domain] ?? []).map((key) => [
      key.name,
      {
        value: null,
        critical: key.critical,
        type: key.type,
      },
    ])
  );
  const groundTruthTemplate = templateFields;

  await fs.writeFile(absoluteGtPath, `${JSON.stringify(groundTruthTemplate, null, 2)}\n`, "utf8");
  return true;
}

async function bootstrap() {
  const createPlaceholders = process.argv.includes('--create-placeholders');
  const domains = [];
  let createdGroundTruthFiles = 0;
  let reusedGroundTruthFiles = 0;
  const missingGroundTruthFiles = [];

  for (const [sourceDir, domain] of Object.entries(DOMAIN_DIRS)) {
    const sourcePath = path.resolve(BENCH_DOCUMENTS_ROOT, sourceDir);
    const groundTruthDir = path.resolve(sourcePath, "ground_truth");

    const pdfFiles = await listFilesByExtension(sourcePath, /\.pdf$/i);
    const jsonFiles = await listFilesByExtension(groundTruthDir, /\.json$/i);

    const jsonByNormalizedBase = new Map();
    for (const jsonFile of jsonFiles) {
      const normalized = normalizeForMatch(jsonFile);
      const current = jsonByNormalizedBase.get(normalized) ?? [];
      current.push(jsonFile);
      jsonByNormalizedBase.set(normalized, current);
    }

    await ensureDir(groundTruthDir);

    const documents = [];

    for (const filename of pdfFiles) {
      const documentId = `${domain}-${slugify(filename)}`;
      const relativePdfPath = path.posix.join("bench_documents", sourceDir, filename);

      const matchedGroundTruthFilename = findMatchingGroundTruth(
        filename,
        jsonFiles,
        jsonByNormalizedBase
      );

      const groundTruthFilename = matchedGroundTruthFilename ?? `${toSafeJsonBasename(filename)}.json`;
      const absoluteGtPath = path.resolve(groundTruthDir, groundTruthFilename);
      const relativeGtPath = path.posix.join(
        "bench_documents",
        sourceDir,
        "ground_truth",
        groundTruthFilename
      );

      if (matchedGroundTruthFilename) {
        reusedGroundTruthFiles += 1;
      } else {
        if (createPlaceholders) {
          const created = await ensurePlaceholderGroundTruth({
            absoluteGtPath,
            domain,
          });
          if (created) {
            createdGroundTruthFiles += 1;
          }
        } else {
          missingGroundTruthFiles.push(relativeGtPath);
        }
      }

      documents.push({
        document_id: documentId,
        domain,
        source_pdf: relativePdfPath,
        ground_truth: relativeGtPath,
      });
    }

    domains.push({
      id: domain,
      source_directory: path.posix.join("bench_documents", sourceDir),
      document_count: documents.length,
      documents,
    });
  }

  if (missingGroundTruthFiles.length > 0) {
    console.error(
      `Missing ${missingGroundTruthFiles.length} ground-truth file(s). Add labels or run with --create-placeholders:`
    );
    for (const file of missingGroundTruthFiles.slice(0, 25)) {
      console.error(`- ${file}`);
    }
    if (missingGroundTruthFiles.length > 25) {
      console.error(`- ...and ${missingGroundTruthFiles.length - 25} more`);
    }
    process.exit(1);
  }

  const manifest = {
    schema_version: "1.0",
    generated_at: new Date().toISOString(),
    domains,
  };

  await ensureDir(path.dirname(MANIFEST_PATH));
  await fs.writeFile(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  const totalDocuments = domains.reduce((sum, domain) => sum + domain.document_count, 0);

  console.log(`Manifest written: ${MANIFEST_PATH}`);
  console.log(`Documents indexed: ${totalDocuments}`);
  console.log(`Ground-truth files reused: ${reusedGroundTruthFiles}`);
  console.log(`Ground-truth placeholders created: ${createdGroundTruthFiles}`);
}

bootstrap().catch((error) => {
  console.error(error);
  process.exit(1);
});
