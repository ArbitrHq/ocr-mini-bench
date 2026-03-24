import { describe, expect, it } from 'vitest';
import {
  normalizeVisualConfusables,
  normalizeText,
  normalizeCompactAlphaNumeric,
  normalizeNumeric,
  safeTrim,
} from './text-normalization';

describe('normalizeVisualConfusables', () => {
  it('converts Greek and Cyrillic lookalikes to ASCII', () => {
    // Greek: Α->A, Η->H, etc.
    expect(normalizeVisualConfusables('Ηello')).toBe('Hello'); // Greek Η
    expect(normalizeVisualConfusables('ΤEST')).toBe('TEST'); // Greek Τ

    // Cyrillic: А->A, С->C, etc.
    expect(normalizeVisualConfusables('АВЕКМНОРСТУХ')).toBe('ABEKMHOPCTYX');
  });

  it('preserves regular ASCII', () => {
    expect(normalizeVisualConfusables('Hello World 123')).toBe('Hello World 123');
  });
});

describe('normalizeText', () => {
  it('lowercases, trims, and normalizes separators', () => {
    expect(normalizeText('  HELLO WORLD  ')).toBe('hello world');
    expect(normalizeText('hello-world')).toBe('hello world');
    expect(normalizeText('hello_world')).toBe('hello world');
    expect(normalizeText('hello    world')).toBe('hello world');
  });

  it('removes diacritics and normalizes confusables', () => {
    expect(normalizeText('café')).toBe('cafe');
    expect(normalizeText('résumé')).toBe('resume');
    expect(normalizeText('ΤEST')).toBe('test'); // Greek Τ
  });

  it('preserves numbers', () => {
    expect(normalizeText('Invoice #123')).toBe('invoice 123');
    expect(normalizeText('PO-2024-001')).toBe('po 2024 001');
  });
});

describe('normalizeCompactAlphaNumeric', () => {
  it('removes all non-alphanumeric and lowercases', () => {
    expect(normalizeCompactAlphaNumeric('INV-001')).toBe('inv001');
    expect(normalizeCompactAlphaNumeric('PO #12345')).toBe('po12345');
    expect(normalizeCompactAlphaNumeric('café123')).toBe('cafe123');
  });

  it('handles empty/punctuation-only input', () => {
    expect(normalizeCompactAlphaNumeric('')).toBe('');
    expect(normalizeCompactAlphaNumeric('---')).toBe('');
  });
});

describe('normalizeNumeric', () => {
  it('parses integers and decimals', () => {
    expect(normalizeNumeric('123')).toBe(123);
    expect(normalizeNumeric('123.45')).toBe(123.45);
    expect(normalizeNumeric('-456')).toBe(-456);
  });

  it('handles comma as decimal separator', () => {
    expect(normalizeNumeric('123,45')).toBe(123.45);
  });

  it('strips whitespace and currency symbols', () => {
    expect(normalizeNumeric('1 000')).toBe(1000);
    expect(normalizeNumeric('$100')).toBe(100);
    expect(normalizeNumeric('€99.99')).toBe(99.99);
  });

  it('returns null for invalid input', () => {
    expect(normalizeNumeric('')).toBeNull();
    expect(normalizeNumeric('abc')).toBeNull();
    expect(normalizeNumeric('12.34.56')).toBeNull();
  });
});

describe('safeTrim', () => {
  it('trims strings and converts other types', () => {
    expect(safeTrim('  hello  ')).toBe('hello');
    expect(safeTrim(123)).toBe('123');
    expect(safeTrim(null)).toBe('');
    expect(safeTrim(undefined)).toBe('');
  });
});
