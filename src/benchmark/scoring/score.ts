import { normalizeKeyName, parseFirstJsonObject } from '../../ocr/parsing';
import type { GroundTruthDocument } from '../types';
import {
  buildValueByKey,
  countFoundKeys,
  extractPairsFromParsedJson,
  extractPairsFromRawText,
} from './extraction';
import { isMatch, matchesEmptyExpectation, pickExpectedValues, shouldScoreKey } from './matching';
import type { ScoreResult, ScoreResultDetailed } from './types';

function extractAndIndexValues(rawOutput: string, document: GroundTruthDocument) {
  const parsed = parseFirstJsonObject(rawOutput);
  const pairsFromJson = extractPairsFromParsedJson(parsed, document.keys);
  const extractedPairsRaw = pairsFromJson.length > 0 ? pairsFromJson : extractPairsFromRawText(rawOutput, document.keys);
  const valueByKey = buildValueByKey(extractedPairsRaw);
  const foundKeyCount = countFoundKeys(valueByKey, document.keys);

  return {
    parsed,
    extractedPairsRaw,
    valueByKey,
    foundKeyCount,
  };
}

export function scoreModelOutput(rawOutput: string, document: GroundTruthDocument): ScoreResult {
  const { valueByKey, foundKeyCount } = extractAndIndexValues(rawOutput, document);

  let fieldTotal = 0;
  let fieldCorrect = 0;
  let criticalTotal = 0;
  let criticalCorrect = 0;

  for (const key of document.keys) {
    const normalizedKey = normalizeKeyName(key.name);
    const extractedRecord = valueByKey.get(normalizedKey);
    const extracted = extractedRecord?.value ?? '';
    const mode = key.match ?? 'normalized_text';
    const candidates = pickExpectedValues(key);
    const scored = shouldScoreKey(key) || key.critical;
    if (!scored) {
      continue;
    }
    const matched =
      candidates.length > 0
        ? candidates.some((expected) => isMatch(extracted, expected, mode, key))
        : matchesEmptyExpectation(extractedRecord);

    fieldTotal += 1;
    if (matched) {
      fieldCorrect += 1;
    }

    if (key.critical) {
      criticalTotal += 1;
      if (matched) {
        criticalCorrect += 1;
      }
    }
  }

  return {
    fieldTotal,
    fieldCorrect,
    criticalTotal,
    criticalCorrect,
    foundKeyCount,
    requestedKeyCount: document.keys.length,
  };
}

export function scoreModelOutputDetailed(rawOutput: string, document: GroundTruthDocument): ScoreResultDetailed {
  const { parsed, extractedPairsRaw, valueByKey, foundKeyCount } = extractAndIndexValues(rawOutput, document);

  let fieldTotal = 0;
  let fieldCorrect = 0;
  let criticalTotal = 0;
  let criticalCorrect = 0;
  const keyComparisons: ScoreResultDetailed['keyComparisons'] = [];

  for (const key of document.keys) {
    const normalizedKey = normalizeKeyName(key.name);
    const extractedRecord = valueByKey.get(normalizedKey);
    const extracted = extractedRecord?.value ?? '';
    const mode = key.match ?? 'normalized_text';
    const candidates = pickExpectedValues(key);
    const scored = shouldScoreKey(key) || key.critical;
    const matched = !scored
      ? false
      : candidates.length > 0
        ? candidates.some((expected) => isMatch(extracted, expected, mode, key))
        : matchesEmptyExpectation(extractedRecord);

    keyComparisons.push({
      key: key.name,
      critical: key.critical,
      scored,
      expectedValues: candidates,
      extractedValue: extracted,
      matched,
      matchMode: mode,
    });

    if (!scored) {
      continue;
    }

    fieldTotal += 1;
    if (matched) {
      fieldCorrect += 1;
    }

    if (key.critical) {
      criticalTotal += 1;
      if (matched) {
        criticalCorrect += 1;
      }
    }
  }

  return {
    fieldTotal,
    fieldCorrect,
    criticalTotal,
    criticalCorrect,
    foundKeyCount,
    requestedKeyCount: document.keys.length,
    parsedOutput: parsed,
    extractedPairs: extractedPairsRaw.map((pair) => ({
      key: pair.key,
      value: pair.value,
      found: pair.found !== false,
    })),
    keyComparisons,
  };
}
