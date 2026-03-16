import { normalizeKeyName } from '../../ocr/parsing';
import { safeTrim } from './text-normalization';
import type { ExtractedPair, ExtractedValue } from './types';

function extractPairsFromObjectMap(object: Record<string, unknown>, requestedKeys: { name: string }[]): ExtractedPair[] {
  const requested = new Set(requestedKeys.map((key) => normalizeKeyName(key.name)));
  return Object.entries(object)
    .filter(([key]) => requested.has(normalizeKeyName(key)))
    .map(([key, value]) => ({ key, value: String(value ?? '') }));
}

export function extractPairsFromParsedJson(parsed: unknown, requestedKeys: { name: string }[]): ExtractedPair[] {
  if (!parsed || typeof parsed !== 'object') return [];

  const parsedObject = parsed as Record<string, unknown>;
  const out: ExtractedPair[] = [];

  const pairs = parsedObject.pairs;
  if (Array.isArray(pairs)) {
    for (const pair of pairs) {
      if (!pair || typeof pair !== 'object') continue;
      const pairObject = pair as Record<string, unknown>;
      const keyCandidate = pairObject.key ?? pairObject.name ?? pairObject.field ?? '';
      const valueCandidate = pairObject.value ?? pairObject.extracted_value ?? '';
      const foundCandidate = pairObject.found;
      const key = safeTrim(keyCandidate);
      if (!key) continue;
      out.push({
        key,
        value: String(valueCandidate ?? ''),
        found: typeof foundCandidate === 'boolean' ? foundCandidate : undefined,
      });
    }
  }

  const valuesCandidate = parsedObject.values;
  if (valuesCandidate && typeof valuesCandidate === 'object' && !Array.isArray(valuesCandidate)) {
    out.push(...extractPairsFromObjectMap(valuesCandidate as Record<string, unknown>, requestedKeys));
  }

  if (out.length === 0) {
    out.push(...extractPairsFromObjectMap(parsedObject, requestedKeys));
  }

  return out;
}

export function extractPairsFromRawText(raw: string, requestedKeys: { name: string }[]): ExtractedPair[] {
  const requestedMap = new Map<string, string>();
  for (const key of requestedKeys) {
    requestedMap.set(normalizeKeyName(key.name), key.name);
  }

  const collected = new Map<string, ExtractedPair>();
  const pairObjectPattern =
    /\{[^{}]*"key"\s*:\s*"([^"]+)"[^{}]*"value"\s*:\s*"([^"]*)"[^{}]*(?:"found"\s*:\s*(true|false))?[^{}]*\}/g;

  let match = pairObjectPattern.exec(raw);
  while (match) {
    const rawKey = match[1] ?? '';
    const rawValue = match[2] ?? '';
    const rawFound = match[3];
    const normalized = normalizeKeyName(rawKey);
    const canonical = requestedMap.get(normalized);
    if (canonical) {
      collected.set(normalized, {
        key: canonical,
        value: rawValue,
        found: rawFound ? rawFound === 'true' : undefined,
      });
    }
    match = pairObjectPattern.exec(raw);
  }

  for (const line of raw.split(/\r?\n/)) {
    const lineMatch = line.match(/^\s*["']?([^:"']{2,120})["']?\s*:\s*(.+)\s*$/);
    if (!lineMatch) continue;
    const key = normalizeKeyName(lineMatch[1] ?? '');
    const canonical = requestedMap.get(key);
    if (!canonical) continue;
    const value = safeTrim(lineMatch[2]).replace(/^["']|["']$/g, '');
    if (!value) continue;
    collected.set(key, { key: canonical, value });
  }

  return Array.from(collected.values());
}

export function buildValueByKey(extractedPairs: ExtractedPair[]): Map<string, ExtractedValue> {
  const valueByKey = new Map<string, ExtractedValue>();

  for (const pair of extractedPairs) {
    const key = normalizeKeyName(pair.key);
    const current = valueByKey.get(key);
    const foundFlag = pair.found !== false;
    const value = safeTrim(pair.value);
    if (!current || (value && !current.value)) {
      valueByKey.set(key, { value, found: foundFlag });
    }
  }

  return valueByKey;
}

export function countFoundKeys(
  valueByKey: Map<string, ExtractedValue>,
  requestedKeys: { name: string }[]
): number {
  const requestedSet = new Set(requestedKeys.map((key) => normalizeKeyName(key.name)));
  let foundKeyCount = 0;

  for (const [key, value] of valueByKey.entries()) {
    if (requestedSet.has(key) && value.found && value.value) {
      foundKeyCount += 1;
    }
  }

  return foundKeyCount;
}
