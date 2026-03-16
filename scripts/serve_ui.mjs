import { createServer } from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..');
const UI_DIR = path.resolve(REPO_ROOT, 'ui');
const ARTIFACTS_DIR = path.resolve(REPO_ROOT, 'artifacts');

function parseArgs(argv) {
  const out = { port: 4173 };
  for (const arg of argv) {
    if (arg.startsWith('--port=')) {
      const n = Number(arg.split('=')[1]);
      if (Number.isFinite(n) && n > 0) out.port = n;
    }
  }
  return out;
}

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function walkFiles(rootDir) {
  const out = [];
  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.resolve(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      else out.push(full);
    }
  }
  if (await exists(rootDir)) await walk(rootDir);
  return out;
}

async function latestByName(filename) {
  const files = (await walkFiles(ARTIFACTS_DIR)).filter((p) => path.basename(p) === filename);
  if (!files.length) return null;
  const stats = await Promise.all(files.map(async (p) => ({ p, s: await fs.stat(p) })));
  stats.sort((a, b) => b.s.mtimeMs - a.s.mtimeMs);
  return stats[0].p;
}

async function discoverArtifactPaths() {
  const leaderboardPath =
    (await latestByName('leaderboard.frontend.json')) ||
    (await latestByName('latest.frontend.json')) ||
    (await latestByName('latest.json'));

  const debugPath = (await latestByName('latest.debug.json')) || (await latestByName('snapshot.debug.json'));

  return { leaderboardPath, debugPath };
}

function json(res, status, payload) {
  const body = `${JSON.stringify(payload, null, 2)}\n`;
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  res.end(body);
}

function text(res, status, body, contentType = 'text/plain; charset=utf-8') {
  res.writeHead(status, { 'content-type': contentType });
  res.end(body);
}

function contentTypeFor(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.html') return 'text/html; charset=utf-8';
  if (ext === '.js') return 'text/javascript; charset=utf-8';
  if (ext === '.css') return 'text/css; charset=utf-8';
  if (ext === '.json') return 'application/json; charset=utf-8';
  if (ext === '.pdf') return 'application/pdf';
  return 'application/octet-stream';
}

async function sendFile(res, filePath) {
  try {
    const data = await fs.readFile(filePath);
    res.writeHead(200, { 'content-type': contentTypeFor(filePath) });
    res.end(data);
  } catch {
    text(res, 404, 'Not found');
  }
}

function safeResolveRepoPath(inputPath) {
  const abs = path.resolve(REPO_ROOT, inputPath);
  if (!abs.startsWith(REPO_ROOT)) return null;
  return abs;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const artifacts = await discoverArtifactPaths();

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url || '/', 'http://localhost');
      const pathname = url.pathname;

      if (pathname === '/api/meta') {
        return json(res, 200, {
          repo_root: REPO_ROOT,
          artifacts_dir: ARTIFACTS_DIR,
          leaderboard_path: artifacts.leaderboardPath,
          debug_path: artifacts.debugPath,
        });
      }

      if (pathname === '/api/leaderboard') {
        if (!artifacts.leaderboardPath) return json(res, 404, { error: 'No leaderboard artifact found.' });
        const raw = await fs.readFile(artifacts.leaderboardPath, 'utf8');
        res.writeHead(200, { 'content-type': 'application/json; charset=utf-8' });
        return res.end(raw);
      }

      if (pathname === '/api/debug') {
        if (!artifacts.debugPath) return json(res, 404, { error: 'No debug artifact found.' });
        const raw = await fs.readFile(artifacts.debugPath, 'utf8');
        res.writeHead(200, { 'content-type': 'application/json; charset=utf-8' });
        return res.end(raw);
      }

      if (pathname === '/api/document') {
        const sourcePdf = url.searchParams.get('source_pdf') || '';
        const target = safeResolveRepoPath(sourcePdf);
        if (!target) return json(res, 400, { error: 'Invalid source_pdf path.' });
        if (!(await exists(target))) return json(res, 404, { error: `Document not found: ${sourcePdf}` });
        return sendFile(res, target);
      }

      if (pathname === '/' || pathname === '/index.html') return sendFile(res, path.resolve(UI_DIR, 'index.html'));
      if (pathname === '/debug' || pathname === '/debug.html') return sendFile(res, path.resolve(UI_DIR, 'debug.html'));

      const uiFile = safeResolveRepoPath(path.relative(REPO_ROOT, path.resolve(UI_DIR, pathname.slice(1))));
      if (uiFile && uiFile.startsWith(UI_DIR) && (await exists(uiFile))) {
        return sendFile(res, uiFile);
      }

      text(res, 404, 'Not found');
    } catch (error) {
      json(res, 500, { error: error instanceof Error ? error.message : 'Internal server error.' });
    }
  });

  server.listen(args.port, '127.0.0.1', () => {
    console.log(`OCR mini-bench UI available at http://127.0.0.1:${args.port}`);
    console.log(`Leaderboard source: ${artifacts.leaderboardPath || 'not found'}`);
    console.log(`Debug source: ${artifacts.debugPath || 'not found'}`);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
