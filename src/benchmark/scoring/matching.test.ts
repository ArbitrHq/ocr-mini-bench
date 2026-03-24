import { describe, expect, it } from 'vitest';
import {
  isMatch,
  isMissingLike,
  shouldScoreKey,
  pickExpectedValues,
  matchesEmptyExpectation,
} from './matching';
import type { GroundTruthKey } from '../types';

function makeKey(overrides: Partial<GroundTruthKey> = {}): GroundTruthKey {
  return {
    name: 'test_key',
    critical: false,
    expected: 'expected value',
    match: 'normalized_text',
    ...overrides,
  };
}

describe('isMatch - date matching', () => {
  it('matches various date formats to ISO', () => {
    const key = makeKey({ name: 'invoice_date', data_type: 'date' });
    expect(isMatch('2024-01-15', '2024-01-15', 'normalized_text', key)).toBe(true);
    expect(isMatch('01/15/2024', '2024-01-15', 'normalized_text', key)).toBe(true);
    expect(isMatch('15/01/2024', '2024-01-15', 'normalized_text', key)).toBe(true);
    expect(isMatch('15.01.2024', '2024-01-15', 'normalized_text', key)).toBe(true);
  });

  it('infers date type from key name', () => {
    const key = makeKey({ name: 'due_date' }); // no data_type, but name contains "date"
    expect(isMatch('2024-03-01', '2024-03-01', 'normalized_text', key)).toBe(true);
  });
});

describe('isMatch - currency matching', () => {
  it('matches currency codes and normalizes symbols', () => {
    const key = makeKey({ name: 'currency', data_type: 'currency' });
    expect(isMatch('USD', 'USD', 'normalized_text', key)).toBe(true);
    expect(isMatch('usd', 'USD', 'normalized_text', key)).toBe(true);
    expect(isMatch('$', 'USD', 'normalized_text', key)).toBe(true);
    expect(isMatch('€', 'EUR', 'normalized_text', key)).toBe(true);
    expect(isMatch('USD 100.00', 'USD', 'normalized_text', key)).toBe(true);
  });
});

describe('isMatch - company name matching', () => {
  it('normalizes company suffixes and c/o variations', () => {
    const key = makeKey({ name: 'vendor_name' });
    expect(isMatch('Acme Inc', 'Acme Inc.', 'normalized_text', key)).toBe(true);
    expect(isMatch('Acme Corporation', 'Acme Corp', 'normalized_text', key)).toBe(true);
    expect(isMatch('Acme LLC', 'Acme', 'normalized_text', key)).toBe(true);
    expect(isMatch('c/o John Smith', 'John Smith', 'normalized_text', key)).toBe(true);
  });
});

describe('isMatch - numeric matching', () => {
  it('handles decimal separators and currency symbols', () => {
    const key = makeKey({ data_type: 'float' });
    expect(isMatch('100.5', '100.50', 'numeric', key)).toBe(true);
    expect(isMatch('100,50', '100.50', 'numeric', key)).toBe(true);
    expect(isMatch('$100.50', '100.50', 'numeric', key)).toBe(true);
    expect(isMatch('1 000.00', '1000', 'numeric', key)).toBe(true);
  });
});

describe('isMatch - match modes', () => {
  it('exact mode compares trimmed strings case-sensitively', () => {
    const key = makeKey();
    expect(isMatch('test', 'test', 'exact', key)).toBe(true);
    expect(isMatch(' test ', 'test', 'exact', key)).toBe(true); // trimmed
    expect(isMatch('TEST', 'test', 'exact', key)).toBe(false);
  });

  it('contains mode checks if actual contains expected (normalized)', () => {
    const key = makeKey();
    expect(isMatch('Invoice Number: INV-001', 'INV-001', 'contains', key)).toBe(true);
    expect(isMatch('INVOICE NUMBER', 'invoice number', 'contains', key)).toBe(true);
  });

  it('normalized_text mode normalizes case, whitespace, and punctuation', () => {
    const key = makeKey();
    expect(isMatch('Invoice Number', 'invoice number', 'normalized_text', key)).toBe(true);
    expect(isMatch('INV-001', 'INV001', 'normalized_text', key)).toBe(true);
  });
});

describe('isMatch - missing value handling', () => {
  it('matches empty actual against missing-like expected values', () => {
    const key = makeKey();
    expect(isMatch('', 'N/A', 'normalized_text', key)).toBe(true);
    expect(isMatch('', 'None', 'normalized_text', key)).toBe(true);
    expect(isMatch('', 'real value', 'normalized_text', key)).toBe(false);
  });
});

describe('isMissingLike', () => {
  it('identifies empty and common missing markers', () => {
    expect(isMissingLike('')).toBe(true);
    expect(isMissingLike('   ')).toBe(true);
    expect(isMissingLike('N/A')).toBe(true);
    expect(isMissingLike('None')).toBe(true);
    expect(isMissingLike('null')).toBe(true);
    expect(isMissingLike('not available')).toBe(true);
  });

  it('returns false for real values', () => {
    expect(isMissingLike('John Doe')).toBe(false);
    expect(isMissingLike('100.00')).toBe(false);
  });
});

describe('shouldScoreKey', () => {
  it('returns true for non-empty expected values', () => {
    expect(shouldScoreKey(makeKey({ expected: 'value' }))).toBe(true);
    expect(shouldScoreKey(makeKey({ expected: ['a', 'b'] }))).toBe(true);
  });

  it('returns false for empty/null expected values', () => {
    expect(shouldScoreKey(makeKey({ expected: '' }))).toBe(false);
    expect(shouldScoreKey(makeKey({ expected: [] }))).toBe(false);
    expect(shouldScoreKey(makeKey({ expected: null }))).toBe(false);
  });
});

describe('pickExpectedValues', () => {
  it('normalizes expected to array format', () => {
    expect(pickExpectedValues(makeKey({ expected: 'value' }))).toEqual(['value']);
    expect(pickExpectedValues(makeKey({ expected: ['a', 'b'] }))).toEqual(['a', 'b']);
    expect(pickExpectedValues(makeKey({ expected: null }))).toEqual([]);
  });
});

describe('matchesEmptyExpectation', () => {
  it('returns true for undefined, not-found, or missing-like values', () => {
    expect(matchesEmptyExpectation(undefined)).toBe(true);
    expect(matchesEmptyExpectation({ value: 'x', found: false })).toBe(true);
    expect(matchesEmptyExpectation({ value: 'N/A', found: true })).toBe(true);
  });

  it('returns false for real extracted values', () => {
    expect(matchesEmptyExpectation({ value: 'real', found: true })).toBe(false);
  });
});
