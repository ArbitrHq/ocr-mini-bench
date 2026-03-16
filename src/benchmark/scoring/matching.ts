import { normalizeKeyName } from '../../ocr/parsing';
import type { GroundTruthKey, KeyMatchMode } from '../types';
import { normalizeCompactAlphaNumeric, normalizeNumeric, normalizeText } from './text-normalization';
import type { ExtractedValue } from './types';

function toIsoDate(year: number, month: number, day: number): string | null {
  if (year < 100) year += 2000;
  if (month < 1 || month > 12) return null;
  if (day < 1 || day > 31) return null;
  const candidate = new Date(Date.UTC(year, month - 1, day));
  if (
    candidate.getUTCFullYear() !== year ||
    candidate.getUTCMonth() !== month - 1 ||
    candidate.getUTCDate() !== day
  ) {
    return null;
  }
  return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
}

function collectDateCandidates(rawValue: string): Set<string> {
  const value = rawValue.trim();
  const out = new Set<string>();
  if (!value) return out;

  const ymd = value.match(/(\d{4})[\/.\-](\d{1,2})[\/.\-](\d{1,2})/);
  if (ymd) {
    const iso = toIsoDate(Number(ymd[1]), Number(ymd[2]), Number(ymd[3]));
    if (iso) out.add(iso);
  }

  const dmyOrMdy = value.match(/(\d{1,2})[\/.\-](\d{1,2})[\/.\-](\d{2,4})/);
  if (dmyOrMdy) {
    const a = Number(dmyOrMdy[1]);
    const b = Number(dmyOrMdy[2]);
    const year = Number(dmyOrMdy[3]);

    const dmy = toIsoDate(year, b, a);
    if (dmy) out.add(dmy);
    const mdy = toIsoDate(year, a, b);
    if (mdy) out.add(mdy);
  }

  const parsedMs = Date.parse(value);
  if (!Number.isNaN(parsedMs)) {
    const parsed = new Date(parsedMs);
    const iso = `${parsed.getUTCFullYear()}-${String(parsed.getUTCMonth() + 1).padStart(2, '0')}-${String(parsed.getUTCDate()).padStart(2, '0')}`;
    out.add(iso);
  }

  return out;
}

type ParsedTime = {
  hours: number;
  minutes: number;
  seconds: number;
  hasSeconds: boolean;
};

function parseTime(rawValue: string): ParsedTime | null {
  const value = rawValue.trim();
  if (!value) return null;

  const match = value.match(/(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap]\.?m\.?)?/i);
  if (!match) return null;

  let hours = Number(match[1]);
  const minutes = Number(match[2]);
  const seconds = Number(match[3] ?? '0');
  const hasSeconds = typeof match[3] === 'string';
  const ampm = (match[4] ?? '').toLowerCase().replace(/\./g, '');

  if (minutes < 0 || minutes > 59 || seconds < 0 || seconds > 59) return null;

  if (ampm === 'pm' && hours < 12) {
    hours += 12;
  } else if (ampm === 'am' && hours === 12) {
    hours = 0;
  }

  if (hours < 0 || hours > 23) return null;

  return { hours, minutes, seconds, hasSeconds };
}

function isDateLikeKey(key: GroundTruthKey): boolean {
  const type = (key.data_type ?? '').toLowerCase();
  if (type.includes('date') || type.includes('datetime') || type.includes('timestamp')) return true;
  const name = normalizeKeyName(key.name);
  return name.includes('date');
}

function isTimeLikeKey(key: GroundTruthKey): boolean {
  const type = (key.data_type ?? '').toLowerCase();
  if (type.includes('time') || type.includes('datetime') || type.includes('timestamp')) return true;
  const name = normalizeKeyName(key.name);
  return name.includes('time');
}

function isCurrencyLikeKey(key: GroundTruthKey): boolean {
  const type = (key.data_type ?? '').toLowerCase();
  if (type.includes('currency')) return true;
  const name = normalizeKeyName(key.name);
  return name.includes('currency');
}

function isFreightTermsKey(key: GroundTruthKey): boolean {
  const name = normalizeKeyName(key.name);
  return name === 'freight terms' || name.includes('freight terms');
}

function isPackageTypeKey(key: GroundTruthKey): boolean {
  const name = normalizeKeyName(key.name);
  return name.includes('package type');
}

function isWeightUnitKey(key: GroundTruthKey): boolean {
  const name = normalizeKeyName(key.name);
  const type = (key.data_type ?? '').toLowerCase();
  return name.includes('weight unit') || (name.includes('weight') && type.includes('unit'));
}

function isNameLikeKey(key: GroundTruthKey): boolean {
  const name = normalizeKeyName(key.name);
  return name.includes('name');
}

function isTransportModeKey(key: GroundTruthKey): boolean {
  const name = normalizeKeyName(key.name);
  return name === 'transport mode' || name.includes('transport mode');
}

const COMPANY_SUFFIX_TOKENS = new Set<string>([
  'inc',
  'incorporated',
  'corp',
  'corporation',
  'company',
  'co',
  'llc',
  'ltd',
  'limited',
  'plc',
  'bv',
  'nv',
  'sa',
  'sarl',
  'gmbh',
  'ag',
  'aps',
  'as',
  'oy',
  'ab',
  'pte',
]);

function normalizeCompanyName(value: string): string {
  const rewritten = value
    .replace(/\bcare\s+of\b/gi, ' ')
    .replace(/\bc\s*\/?\s*o\b/gi, ' ')
    .replace(/\ba\s*[/.]?\s*s\b/gi, 'as');
  const tokens = normalizeText(rewritten)
    .split(' ')
    .filter((token) => token.length > 0);

  while (tokens.length > 0) {
    const last = tokens[tokens.length - 1];
    if (!COMPANY_SUFFIX_TOKENS.has(last)) break;
    tokens.pop();
  }

  return tokens.join(' ');
}

function normalizeFreightTerms(value: string): string {
  const ignored = new Set(['freight', 'term', 'terms']);
  return normalizeText(value)
    .split(' ')
    .filter((token) => token.length > 0 && !ignored.has(token))
    .join(' ');
}

function normalizePackageTypeValue(value: string): string {
  return normalizeText(value)
    .split(' ')
    .filter((token) => token.length > 0 && token !== 'stc')
    .join(' ');
}

function normalizeWeightUnit(value: string): string | null {
  const compact = normalizeCompactAlphaNumeric(value);
  if (!compact) return null;
  if (compact === 'kg' || compact === 'kgs' || compact === 'kilogram' || compact === 'kilograms') return 'kg';
  if (compact === 'g' || compact === 'gram' || compact === 'grams') return 'g';
  if (compact === 'lb' || compact === 'lbs' || compact === 'pound' || compact === 'pounds') return 'lb';
  if (compact === 'ton' || compact === 'tons' || compact === 'tonne' || compact === 'tonnes' || compact === 'mt') {
    return 'tonne';
  }
  return compact;
}

const MISSING_VALUE_MARKERS = new Set<string>([
  'na',
  'none',
  'null',
  'nil',
  'nill',
  'empty',
  'blank',
  'unknown',
  'nvd',
  'notavailable',
  'notapplicable',
  'notprovided',
  'notstated',
  'notspecified',
]);

export function isMissingLike(value: string): boolean {
  if (!value.trim()) return true;
  const compact = normalizeCompactAlphaNumeric(value);
  if (!compact) return true;
  return MISSING_VALUE_MARKERS.has(compact);
}

const CURRENCY_SYMBOL_TO_CODE = new Map<string, string>([
  ['$', 'USD'],
  ['US$', 'USD'],
  ['USD$', 'USD'],
  ['€', 'EUR'],
  ['£', 'GBP'],
  ['¥', 'JPY'],
  ['C$', 'CAD'],
  ['CA$', 'CAD'],
  ['A$', 'AUD'],
  ['AU$', 'AUD'],
  ['CHF', 'CHF'],
  ['₹', 'INR'],
  ['HK$', 'HKD'],
  ['SG$', 'SGD'],
  ['R$', 'BRL'],
  ['₩', 'KRW'],
  ['₺', 'TRY'],
  ['₽', 'RUB'],
  ['MX$', 'MXN'],
  ['NZ$', 'NZD'],
  ['kr', 'SEK'],
  ['NOK', 'NOK'],
  ['DKK', 'DKK'],
  ['zł', 'PLN'],
]);

const KNOWN_CURRENCY_CODES = new Set<string>([
  'USD',
  'EUR',
  'GBP',
  'JPY',
  'CAD',
  'AUD',
  'CHF',
  'INR',
  'HKD',
  'SGD',
  'BRL',
  'KRW',
  'TRY',
  'RUB',
  'MXN',
  'NZD',
  'SEK',
  'NOK',
  'DKK',
  'PLN',
]);

function normalizeCurrencyValue(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return null;

  const upper = trimmed.toUpperCase();
  if (/^[A-Z]{3}$/.test(upper) && KNOWN_CURRENCY_CODES.has(upper)) {
    return upper;
  }

  for (const [symbol, code] of CURRENCY_SYMBOL_TO_CODE.entries()) {
    if (trimmed === symbol || upper === symbol.toUpperCase()) {
      return code;
    }
  }

  for (const code of KNOWN_CURRENCY_CODES) {
    if (upper.includes(code)) {
      return code;
    }
  }

  for (const [symbol, code] of CURRENCY_SYMBOL_TO_CODE.entries()) {
    if (trimmed.includes(symbol)) {
      return code;
    }
  }

  return null;
}

function hasDateLikeValue(value: string): boolean {
  return collectDateCandidates(value).size > 0;
}

function hasTimeLikeValue(value: string): boolean {
  return parseTime(value) !== null;
}

function dateEquivalent(actual: string, expected: string): boolean {
  const actualDates = collectDateCandidates(actual);
  const expectedDates = collectDateCandidates(expected);
  if (!actualDates.size || !expectedDates.size) return false;
  for (const date of actualDates) {
    if (expectedDates.has(date)) return true;
  }
  return false;
}

function timeEquivalent(actual: string, expected: string): boolean {
  const actualTime = parseTime(actual);
  const expectedTime = parseTime(expected);
  if (!actualTime || !expectedTime) return false;
  if (actualTime.hours !== expectedTime.hours || actualTime.minutes !== expectedTime.minutes) {
    return false;
  }
  if (actualTime.hasSeconds && expectedTime.hasSeconds) {
    return actualTime.seconds === expectedTime.seconds;
  }
  return true;
}

export function isMatch(actual: string, expected: string, mode: KeyMatchMode, key: GroundTruthKey): boolean {
  if (!actual.trim() && expected.trim()) {
    return isMissingLike(expected);
  }

  if (mode === 'exact') {
    return actual.trim() === expected.trim();
  }

  const keyDateHint = isDateLikeKey(key);
  const keyTimeHint = isTimeLikeKey(key);
  const valueDateHint = hasDateLikeValue(actual) || hasDateLikeValue(expected);
  const valueTimeHint = hasTimeLikeValue(actual) || hasTimeLikeValue(expected);

  const dateMatch = dateEquivalent(actual, expected);
  const timeMatch = timeEquivalent(actual, expected);

  if ((keyDateHint || valueDateHint) && dateMatch) {
    const actualTimeParsed = parseTime(actual);
    const expectedTimeParsed = parseTime(expected);
    if (actualTimeParsed && expectedTimeParsed && (keyTimeHint || valueTimeHint)) {
      return timeMatch;
    }
    return true;
  }

  if ((keyTimeHint || valueTimeHint) && timeMatch) {
    return true;
  }

  if (isCurrencyLikeKey(key)) {
    const actualCurrency = normalizeCurrencyValue(actual);
    const expectedCurrency = normalizeCurrencyValue(expected);
    if (actualCurrency && expectedCurrency && actualCurrency === expectedCurrency) {
      return true;
    }
  }

  if (isFreightTermsKey(key)) {
    const normActual = normalizeFreightTerms(actual);
    const normExpected = normalizeFreightTerms(expected);
    if (normActual && normExpected) {
      return normActual === normExpected || normActual.includes(normExpected) || normExpected.includes(normActual);
    }
  }

  if (isPackageTypeKey(key)) {
    const normActual = normalizePackageTypeValue(actual);
    const normExpected = normalizePackageTypeValue(expected);
    if (normActual && normExpected && normActual === normExpected) {
      return true;
    }
    const compactActual = normalizeCompactAlphaNumeric(normActual);
    const compactExpected = normalizeCompactAlphaNumeric(normExpected);
    if (compactActual && compactExpected && compactActual === compactExpected) {
      return true;
    }
  }

  if (isWeightUnitKey(key)) {
    const actualUnit = normalizeWeightUnit(actual);
    const expectedUnit = normalizeWeightUnit(expected);
    if (actualUnit && expectedUnit && actualUnit === expectedUnit) {
      return true;
    }
  }

  if (isNameLikeKey(key)) {
    const nameActual = normalizeCompanyName(actual);
    const nameExpected = normalizeCompanyName(expected);
    if (nameActual && nameExpected && nameActual === nameExpected) {
      return true;
    }
  }

  if (isTransportModeKey(key)) {
    const normActual = normalizeText(actual)
      .split(' ')
      .filter((token) => token !== 'freight')
      .join(' ');
    const normExpected = normalizeText(expected)
      .split(' ')
      .filter((token) => token !== 'freight')
      .join(' ');
    if (normActual && normExpected) {
      return normActual === normExpected || normActual.includes(normExpected) || normExpected.includes(normActual);
    }
  }

  const compactActual = normalizeCompactAlphaNumeric(actual);
  const compactExpected = normalizeCompactAlphaNumeric(expected);
  if (
    compactActual.length > 0 &&
    compactExpected.length > 0 &&
    compactActual === compactExpected &&
    /\d/.test(compactActual) &&
    /\d/.test(compactExpected)
  ) {
    return true;
  }

  if (mode === 'contains') {
    const normActual = normalizeText(actual);
    const normExpected = normalizeText(expected);
    return normActual.includes(normExpected);
  }

  if (mode === 'numeric') {
    const actualNumber = normalizeNumeric(actual);
    const expectedNumber = normalizeNumeric(expected);
    if (actualNumber === null || expectedNumber === null) return false;
    return Math.abs(actualNumber - expectedNumber) < 1e-6;
  }

  return normalizeText(actual) === normalizeText(expected);
}

export function shouldScoreKey(key: GroundTruthKey): boolean {
  const expected = key.expected;
  if (typeof expected === 'string') return expected.trim().length > 0;
  if (Array.isArray(expected)) return expected.some((value) => value.trim().length > 0);
  return false;
}

export function pickExpectedValues(key: GroundTruthKey): string[] {
  if (typeof key.expected === 'string') return [key.expected];
  if (Array.isArray(key.expected)) return key.expected;
  return [];
}

export function matchesEmptyExpectation(extracted: ExtractedValue | undefined): boolean {
  if (!extracted) return true;
  if (extracted.found === false) return true;
  return isMissingLike(extracted.value);
}
