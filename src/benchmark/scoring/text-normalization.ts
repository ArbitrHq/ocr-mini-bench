const VISUAL_CONFUSABLE_TO_ASCII: Record<string, string> = {
  // Greek
  Α: 'A',
  Β: 'B',
  Ε: 'E',
  Ζ: 'Z',
  Η: 'H',
  Ι: 'I',
  Κ: 'K',
  Μ: 'M',
  Ν: 'N',
  Ο: 'O',
  Ρ: 'P',
  Τ: 'T',
  Υ: 'Y',
  Χ: 'X',
  α: 'a',
  β: 'b',
  ε: 'e',
  ι: 'i',
  κ: 'k',
  μ: 'm',
  ν: 'n',
  ο: 'o',
  ρ: 'p',
  τ: 't',
  υ: 'y',
  χ: 'x',
  // Cyrillic
  А: 'A',
  В: 'B',
  Е: 'E',
  К: 'K',
  М: 'M',
  Н: 'H',
  О: 'O',
  Р: 'P',
  С: 'C',
  Т: 'T',
  У: 'Y',
  Х: 'X',
  а: 'a',
  е: 'e',
  к: 'k',
  м: 'm',
  о: 'o',
  р: 'p',
  с: 'c',
  т: 't',
  у: 'y',
  х: 'x',
};

export function normalizeVisualConfusables(value: string): string {
  return Array.from(value)
    .map((ch) => VISUAL_CONFUSABLE_TO_ASCII[ch] ?? ch)
    .join('');
}

export function normalizeText(value: string): string {
  const source = normalizeVisualConfusables(value);
  return source
    .trim()
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function normalizeCompactAlphaNumeric(value: string): string {
  const source = normalizeVisualConfusables(value);
  return source
    .trim()
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '');
}

export function normalizeNumeric(value: string): number | null {
  const normalized = value.replace(/\s/g, '').replace(/,/g, '.').replace(/[^0-9.-]/g, '');
  if (!normalized) return null;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

export function safeTrim(value: unknown): string {
  return String(value ?? '').trim();
}
