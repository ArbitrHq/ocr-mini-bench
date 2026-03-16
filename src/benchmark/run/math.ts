export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function stdDev(values: number[]): number {
  if (values.length <= 1) return 0;
  const avg = mean(values);
  const variance = Math.max(
    0,
    values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length
  );
  return Math.sqrt(variance);
}

export function pct(part: number, total: number): number {
  if (total <= 0) return 0;
  return (part / total) * 100;
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[rank];
}

export function round(value: number, decimals = 4): number {
  const precision = 10 ** decimals;
  return Math.round(value * precision) / precision;
}
