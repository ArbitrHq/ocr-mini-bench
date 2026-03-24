/**
 * Type guard utilities for safely narrowing unknown values.
 */

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

export function isString(value: unknown): value is string {
  return typeof value === 'string';
}

export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function isArray(value: unknown): value is unknown[] {
  return Array.isArray(value);
}

export function hasProperty<K extends string>(obj: unknown, key: K): obj is Record<K, unknown> {
  return isRecord(obj) && Object.prototype.hasOwnProperty.call(obj, key);
}

export function getStringProperty(obj: Record<string, unknown>, key: string): string | undefined {
  const value = obj[key];
  return isString(value) ? value : undefined;
}

export function getNumberProperty(obj: Record<string, unknown>, key: string): number | undefined {
  const value = obj[key];
  return isNumber(value) ? value : undefined;
}

export function getRecordProperty(obj: Record<string, unknown>, key: string): Record<string, unknown> | undefined {
  const value = obj[key];
  return isRecord(value) ? value : undefined;
}

export function getArrayProperty(obj: Record<string, unknown>, key: string): unknown[] | undefined {
  const value = obj[key];
  return isArray(value) ? value : undefined;
}
