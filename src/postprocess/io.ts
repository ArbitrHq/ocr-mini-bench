import { promises as fs } from 'node:fs';
import path from 'node:path';

export async function fileExists(targetPath: string): Promise<boolean> {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

export async function ensureParentDir(targetPath: string): Promise<void> {
  await fs.mkdir(path.dirname(targetPath), { recursive: true });
}

export async function readJsonFile<T>(targetPath: string): Promise<T> {
  const raw = await fs.readFile(targetPath, 'utf8');
  return JSON.parse(raw) as T;
}

export async function writeJsonFile(targetPath: string, value: unknown): Promise<void> {
  await ensureParentDir(targetPath);
  await fs.writeFile(targetPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

export async function writeJsonLinesFile(targetPath: string, rows: unknown[]): Promise<void> {
  await ensureParentDir(targetPath);
  const payload = rows.map((row) => JSON.stringify(row)).join('\n');
  await fs.writeFile(targetPath, payload.length > 0 ? `${payload}\n` : '', 'utf8');
}

export async function readJsonLinesFile<T>(targetPath: string): Promise<T[]> {
  const raw = await fs.readFile(targetPath, 'utf8');
  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);

  const out: T[] = [];
  for (const line of lines) {
    try {
      out.push(JSON.parse(line) as T);
    } catch {
      // Ignore malformed line.
    }
  }
  return out;
}

export function timestampForFilename(date = new Date()): string {
  return date.toISOString().replace(/[:.]/g, '-');
}
