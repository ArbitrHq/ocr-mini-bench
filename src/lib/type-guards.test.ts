import { describe, expect, it } from 'vitest';
import {
  isRecord,
  isString,
  isNumber,
  isArray,
  hasProperty,
  getStringProperty,
  getNumberProperty,
  getRecordProperty,
  getArrayProperty,
} from './type-guards';

describe('isRecord', () => {
  it('returns true for plain objects, false for arrays/null/primitives', () => {
    expect(isRecord({ a: 1 })).toBe(true);
    expect(isRecord([])).toBe(false);
    expect(isRecord(null)).toBe(false);
    expect(isRecord('string')).toBe(false);
  });
});

describe('isString / isNumber / isArray', () => {
  it('correctly identifies types', () => {
    expect(isString('hello')).toBe(true);
    expect(isString(123)).toBe(false);

    expect(isNumber(42)).toBe(true);
    expect(isNumber(NaN)).toBe(false);
    expect(isNumber(Infinity)).toBe(false);

    expect(isArray([1, 2])).toBe(true);
    expect(isArray({})).toBe(false);
  });
});

describe('hasProperty', () => {
  it('checks property existence on objects', () => {
    expect(hasProperty({ value: 1 }, 'value')).toBe(true);
    expect(hasProperty({ a: undefined }, 'a')).toBe(true); // key exists even if undefined
    expect(hasProperty({}, 'missing')).toBe(false);
    expect(hasProperty(null, 'value')).toBe(false);
  });
});

describe('getStringProperty', () => {
  it('extracts string values, returns undefined otherwise', () => {
    expect(getStringProperty({ name: 'test' }, 'name')).toBe('test');
    expect(getStringProperty({ num: 123 }, 'num')).toBeUndefined();
    expect(getStringProperty({}, 'missing')).toBeUndefined();
  });
});

describe('getNumberProperty', () => {
  it('extracts finite numbers, returns undefined for NaN/Infinity/non-numbers', () => {
    expect(getNumberProperty({ count: 42 }, 'count')).toBe(42);
    expect(getNumberProperty({ zero: 0 }, 'zero')).toBe(0);
    expect(getNumberProperty({ bad: NaN }, 'bad')).toBeUndefined();
    expect(getNumberProperty({ str: '123' }, 'str')).toBeUndefined();
  });
});

describe('getRecordProperty / getArrayProperty', () => {
  it('extracts nested objects and arrays', () => {
    const nested = { inner: 1 };
    const items = [1, 2, 3];

    expect(getRecordProperty({ nested }, 'nested')).toBe(nested);
    expect(getRecordProperty({ arr: [] }, 'arr')).toBeUndefined(); // array is not a record

    expect(getArrayProperty({ items }, 'items')).toBe(items);
    expect(getArrayProperty({ obj: {} }, 'obj')).toBeUndefined();
  });
});
