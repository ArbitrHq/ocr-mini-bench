import path from 'node:path';

export const REPO_ROOT = process.cwd();

export const PATHS = {
  config: {
    models: path.resolve(REPO_ROOT, 'config/models.public.json'),
  },
  dataset: {
    manifest: path.resolve(REPO_ROOT, 'dataset/manifest.json'),
  },
  prompts: {
    system: path.resolve(REPO_ROOT, 'prompts/ocr/benchmark/extract_system.txt'),
    user: path.resolve(REPO_ROOT, 'prompts/ocr/benchmark/extract_user.txt'),
  },
  artifacts: {
    root: path.resolve(REPO_ROOT, 'artifacts'),
    checkpoints: path.resolve(REPO_ROOT, 'artifacts/checkpoints'),
    postprocess: path.resolve(REPO_ROOT, 'artifacts/postprocess'),
    latestJson: path.resolve(REPO_ROOT, 'artifacts/latest.json'),
    latestDebug: path.resolve(REPO_ROOT, 'artifacts/latest.debug.json'),
    latestMarkdown: path.resolve(REPO_ROOT, 'artifacts/latest.md'),
  },
  postprocess: {
    root: path.resolve(REPO_ROOT, 'artifacts/postprocess'),
    rawJsonl: path.resolve(REPO_ROOT, 'artifacts/postprocess/raw.jsonl'),
    rawSummary: path.resolve(REPO_ROOT, 'artifacts/postprocess/raw.summary.json'),
    comparisonJsonl: path.resolve(REPO_ROOT, 'artifacts/postprocess/comparison.jsonl'),
    comparisonSummary: path.resolve(REPO_ROOT, 'artifacts/postprocess/comparison.summary.json'),
    metricsSnapshot: path.resolve(REPO_ROOT, 'artifacts/postprocess/metrics.snapshot.json'),
    leaderboardAggregation: path.resolve(REPO_ROOT, 'artifacts/postprocess/leaderboard.aggregation.json'),
    leaderboardFrontend: path.resolve(REPO_ROOT, 'artifacts/postprocess/leaderboard.frontend.json'),
  },
  checkpoint: {
    root: path.resolve(REPO_ROOT, 'artifacts/checkpoints'),
    runsJsonl: path.resolve(REPO_ROOT, 'artifacts/checkpoints/runs.jsonl'),
    rawRunsJsonl: path.resolve(REPO_ROOT, 'artifacts/checkpoints/raw.runs.jsonl'),
    rawJsonl: path.resolve(REPO_ROOT, 'artifacts/checkpoints/raw.jsonl'),
    state: path.resolve(REPO_ROOT, 'artifacts/checkpoints/state.json'),
  },
} as const;
