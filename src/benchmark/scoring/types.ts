import type { GroundTruthDocument, KeyMatchMode } from '../types';

export interface ExtractedPair {
  key: string;
  value: string;
  found?: boolean;
}

export interface ExtractedValue {
  value: string;
  found: boolean;
}

export interface ScoreResult {
  fieldTotal: number;
  fieldCorrect: number;
  criticalTotal: number;
  criticalCorrect: number;
  foundKeyCount: number;
  requestedKeyCount: number;
}

export interface ScoreKeyComparison {
  key: string;
  critical: boolean;
  scored: boolean;
  expectedValues: string[];
  extractedValue: string;
  matched: boolean;
  matchMode: KeyMatchMode;
}

export interface ScoreResultDetailed extends ScoreResult {
  parsedOutput: unknown | null;
  extractedPairs: Array<{ key: string; value: string; found: boolean }>;
  keyComparisons: ScoreKeyComparison[];
}

export interface ScoringContext {
  document: GroundTruthDocument;
  valueByKey: Map<string, ExtractedValue>;
  foundKeyCount: number;
}
