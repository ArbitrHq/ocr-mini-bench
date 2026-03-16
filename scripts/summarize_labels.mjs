import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const BENCHMARK_ROOT = path.resolve(SCRIPT_DIR, "..");
const REPO_ROOT = BENCHMARK_ROOT;
const MANIFEST_PATH = path.resolve(BENCHMARK_ROOT, "dataset/manifest.json");

function pct(numerator, denominator) {
  if (!denominator) return 0;
  return (numerator / denominator) * 100;
}

function hasLabel(value) {
  if (value === null || typeof value === "undefined") return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (typeof value === "number" || typeof value === "boolean") return true;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return false;
}

function collectValueNodes(input, output = []) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    return output;
  }

  for (const [key, value] of Object.entries(input)) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      continue;
    }

    if (Object.prototype.hasOwnProperty.call(value, "value")) {
      output.push({
        name: key,
        critical: Boolean(value.critical),
        expected: value.value,
      });
      continue;
    }

    collectValueNodes(value, output);
  }

  return output;
}

function readComparableKeys(groundTruth) {
  if (Array.isArray(groundTruth?.keys)) {
    throw new Error("Legacy keys[] schema is not supported. Use field objects with { value, critical, type }.");
  }

  const keys = collectValueNodes(groundTruth).map((key) => ({
    critical: Boolean(key.critical),
    expected: key.expected,
  }));
  if (keys.length === 0) {
    throw new Error("No comparable fields found.");
  }
  return keys;
}

async function summarize() {
  const manifestRaw = await fs.readFile(MANIFEST_PATH, "utf8");
  const manifest = JSON.parse(manifestRaw);

  const lines = [];
  lines.push("Label completeness summary");
  lines.push("");

  for (const domain of manifest.domains || []) {
    let totalKeys = 0;
    let labeledKeys = 0;
    let criticalKeys = 0;
    let labeledCriticalKeys = 0;
    let invalidFiles = 0;

    for (const document of domain.documents || []) {
      const groundTruthPath = path.resolve(REPO_ROOT, document.ground_truth);

      let groundTruth;
      try {
        const groundTruthRaw = await fs.readFile(groundTruthPath, "utf8");
        groundTruth = JSON.parse(groundTruthRaw);
      } catch {
        invalidFiles += 1;
        continue;
      }

      let keys;
      try {
        keys = readComparableKeys(groundTruth);
      } catch {
        invalidFiles += 1;
        continue;
      }

      totalKeys += keys.length;
      labeledKeys += keys.filter((key) => hasLabel(key.expected)).length;
      criticalKeys += keys.filter((key) => key.critical).length;
      labeledCriticalKeys += keys.filter((key) => key.critical && hasLabel(key.expected)).length;
    }

    lines.push(
      `${domain.id}: ${labeledKeys}/${totalKeys} keys labeled (${pct(labeledKeys, totalKeys).toFixed(1)}%), ` +
        `critical ${labeledCriticalKeys}/${criticalKeys} (${pct(labeledCriticalKeys, criticalKeys).toFixed(1)}%), ` +
        `invalid files ${invalidFiles}`
    );
  }

  console.log(lines.join("\n"));
}

summarize().catch((error) => {
  console.error(error);
  process.exit(1);
});
