import type { GroundTruthDocument, GroundTruthKey, KeyMatchMode } from './types';

const VALID_MATCH_MODES = new Set<KeyMatchMode>(['exact', 'normalized_text', 'contains', 'numeric']);

interface ValueNode {
  value?: unknown;
  critical?: unknown;
  type?: unknown;
  match?: unknown;
  notes?: unknown;
}

interface RawLeafCandidate {
  leafName: string;
  fullPath: string;
  critical: boolean;
  expected: string | string[] | null;
  dataType?: string;
  match: KeyMatchMode;
  notes?: string;
}

function inferPrimitiveType(value: unknown): string | undefined {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? 'integer' : 'float';
  }
  if (typeof value === 'boolean') {
    return 'boolean';
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
      return 'date';
    }
    return 'string';
  }
  return undefined;
}

function normalizeExpected(value: unknown): string | string[] | null {
  if (value === null || typeof value === 'undefined') {
    return null;
  }

  if (Array.isArray(value)) {
    const normalized = value
      .map((entry) => String(entry ?? '').trim())
      .filter((entry) => entry.length > 0);
    return normalized.length > 0 ? normalized : null;
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  if (typeof value === 'object') {
    return JSON.stringify(value);
  }

  return String(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isValueNode(value: unknown): value is ValueNode {
  return isRecord(value) && Object.prototype.hasOwnProperty.call(value, 'value');
}

function inferMatchMode(typeHint: unknown): KeyMatchMode {
  const normalizedType = typeof typeHint === 'string' ? typeHint.trim().toLowerCase() : '';
  if (normalizedType === 'float' || normalizedType === 'integer' || normalizedType === 'number') {
    return 'numeric';
  }
  return 'normalized_text';
}

function normalizeMatchMode(mode: unknown, fallback: KeyMatchMode): KeyMatchMode {
  if (typeof mode === 'string') {
    const normalized = mode.trim().toLowerCase() as KeyMatchMode;
    if (VALID_MATCH_MODES.has(normalized)) {
      return normalized;
    }
  }
  return fallback;
}

function collectValueNodeLeaves(
  input: Record<string, unknown>,
  pathSegments: string[] = [],
  output: RawLeafCandidate[] = []
): RawLeafCandidate[] {
  const pushFromValueNode = (valueNode: ValueNode, leafName: string, fullPathSegments: string[]) => {
    const fallbackMatch = inferMatchMode(valueNode.type);
    output.push({
      leafName,
      fullPath: fullPathSegments.join('_'),
      critical: Boolean(valueNode.critical),
      expected: normalizeExpected(valueNode.value),
      dataType: typeof valueNode.type === 'string' ? valueNode.type : undefined,
      match: normalizeMatchMode(valueNode.match, fallbackMatch),
      notes: typeof valueNode.notes === 'string' ? valueNode.notes : undefined,
    });
  };

  const pushPrimitiveLeaf = (value: unknown, leafName: string, fullPathSegments: string[]) => {
    const primitiveType = inferPrimitiveType(value);
    const fallbackMatch = inferMatchMode(primitiveType);
    output.push({
      leafName,
      fullPath: fullPathSegments.join('_'),
      critical: false,
      expected: normalizeExpected(value),
      dataType: primitiveType,
      match: fallbackMatch,
      notes: undefined,
    });
  };

  for (const [key, value] of Object.entries(input)) {
    if (isValueNode(value)) {
      pushFromValueNode(value as ValueNode, key, [...pathSegments, key]);
      continue;
    }

    if (isRecord(value)) {
      collectValueNodeLeaves(value, [...pathSegments, key], output);
      continue;
    }

    if (Array.isArray(value)) {
      value.forEach((entry, index) => {
        const indexPath = [...pathSegments, key, String(index + 1)];

        if (isValueNode(entry)) {
          pushFromValueNode(entry as ValueNode, key, indexPath);
          return;
        }

        if (isRecord(entry)) {
          for (const [rowKey, rowValue] of Object.entries(entry)) {
            const rowPath = [...indexPath, rowKey];

            if (isValueNode(rowValue)) {
              pushFromValueNode(rowValue as ValueNode, rowKey, rowPath);
              continue;
            }

            if (isRecord(rowValue)) {
              collectValueNodeLeaves(rowValue, rowPath, output);
              continue;
            }

            if (Array.isArray(rowValue)) {
              rowValue.forEach((nestedValue, nestedIndex) => {
                pushPrimitiveLeaf(nestedValue, rowKey, [...rowPath, String(nestedIndex + 1)]);
              });
              continue;
            }

            pushPrimitiveLeaf(rowValue, rowKey, rowPath);
          }
          return;
        }

        pushPrimitiveLeaf(entry, key, indexPath);
      });
    }
  }

  return output;
}

function deduplicateLeafNames(candidates: RawLeafCandidate[]): GroundTruthKey[] {
  const leafCounts = new Map<string, number>();

  for (const candidate of candidates) {
    leafCounts.set(candidate.leafName, (leafCounts.get(candidate.leafName) ?? 0) + 1);
  }

  const seenNames = new Set<string>();

  return candidates
    .map((candidate) => {
      const preferredName = (leafCounts.get(candidate.leafName) ?? 0) > 1 ? candidate.fullPath : candidate.leafName;
      let finalName = preferredName;
      let suffix = 2;

      while (seenNames.has(finalName)) {
        finalName = `${preferredName}_${suffix}`;
        suffix += 1;
      }
      seenNames.add(finalName);

      return {
        name: finalName,
        critical: candidate.critical,
        expected: candidate.expected,
        data_type: candidate.dataType,
        match: candidate.match,
        notes: candidate.notes,
      } satisfies GroundTruthKey;
    })
    .filter((key) => key.name.trim().length > 0);
}

export function normalizeGroundTruthDocument(
  rawInput: unknown,
  fallback: {
    documentId: string;
    domain: string;
    sourcePdf: string;
  }
): GroundTruthDocument {
  const raw = isRecord(rawInput) ? rawInput : {};
  if (Array.isArray(raw.keys)) {
    throw new Error(
      `Unsupported legacy ground-truth schema in ${fallback.documentId}. Use field objects with { value, critical, type }.`
    );
  }

  const keys = deduplicateLeafNames(collectValueNodeLeaves(raw));
  if (keys.length === 0) {
    throw new Error(
      `No comparable fields found in ${fallback.documentId}. Expected objects containing { value, critical, type }.`
    );
  }

  return {
    schema_version: '2.0',
    document_id: fallback.documentId,
    domain: fallback.domain,
    source_pdf: fallback.sourcePdf,
    notes: undefined,
    keys,
  };
}
