import { describe, expect, it } from 'vitest';
import { parseFirstJsonObject, normalizeKeyName, fallbackExtractKeys } from './parsing';

describe('parseFirstJsonObject', () => {
  it('parses valid JSON directly', () => {
    expect(parseFirstJsonObject('{"key": "value"}')).toEqual({ key: 'value' });
  });

  it('handles whitespace and nested structures', () => {
    expect(parseFirstJsonObject('  {"outer": {"inner": [1,2,3]}}  ')).toEqual({
      outer: { inner: [1, 2, 3] },
    });
  });

  it('extracts JSON from fenced code blocks', () => {
    expect(parseFirstJsonObject('```json\n{"key": "value"}\n```')).toEqual({ key: 'value' });
    expect(parseFirstJsonObject('```\n{"key": "value"}\n```')).toEqual({ key: 'value' });
  });

  it('extracts JSON embedded in text', () => {
    expect(parseFirstJsonObject('Here is the result: {"key": "value"} done')).toEqual({
      key: 'value',
    });
  });

  it('returns null for invalid or missing JSON', () => {
    expect(parseFirstJsonObject('')).toBeNull();
    expect(parseFirstJsonObject('not json')).toBeNull();
    expect(parseFirstJsonObject('{ invalid }')).toBeNull();
  });

  it('parses arrays too (not just objects)', () => {
    expect(parseFirstJsonObject('[1, 2, 3]')).toEqual([1, 2, 3]);
  });
});

describe('normalizeKeyName', () => {
  it('lowercases, trims, and normalizes separators', () => {
    expect(normalizeKeyName('  Invoice Number  ')).toBe('invoice number');
    expect(normalizeKeyName('TOTAL_AMOUNT')).toBe('total amount');
    expect(normalizeKeyName('invoice-number')).toBe('invoice number');
    expect(normalizeKeyName('date/time')).toBe('date time');
  });

  it('collapses multiple spaces and preserves numbers', () => {
    expect(normalizeKeyName('invoice   number')).toBe('invoice number');
    expect(normalizeKeyName('PO-12345')).toBe('po 12345');
  });

  it('handles empty input', () => {
    expect(normalizeKeyName('')).toBe('');
    expect(normalizeKeyName('   ')).toBe('');
  });
});

describe('fallbackExtractKeys', () => {
  it('extracts keys from various list formats', () => {
    const markdown = '- Invoice Number: INV-001\n- Total Amount: $100.00';
    const asterisk = '* Vendor Name: Acme Corp';
    const plain = 'Company: Test Inc';

    expect(fallbackExtractKeys(markdown)).toContain('Invoice Number');
    expect(fallbackExtractKeys(markdown)).toContain('Total Amount');
    expect(fallbackExtractKeys(asterisk)).toContain('Vendor Name');
    expect(fallbackExtractKeys(plain)).toContain('Company');
  });

  it('deduplicates keys and limits to 50', () => {
    const duplicate = '- Name: John\n- Name: Jane';
    expect(fallbackExtractKeys(duplicate).filter((k) => k === 'Name').length).toBe(1);

    const many = Array.from({ length: 100 }, (_, i) => `Key${i}: value`).join('\n');
    expect(fallbackExtractKeys(many).length).toBeLessThanOrEqual(50);
  });

  it('returns empty for text without key patterns', () => {
    expect(fallbackExtractKeys('No colons here at all')).toEqual([]);
  });
});
