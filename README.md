# OCR Mini-bench

[![Live Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-OCR%20Mini--bench-0ea5e9?style=plastic)](https://arbitrhq.ai/leaderboards/ocr-mini-bench)
[![Blog](https://img.shields.io/badge/Blog-OCR%20Mini--bench-22c55e?style=plastic)](https://arbitrhq.ai/blog/ocr-mini-bench)
![Node >=20](https://img.shields.io/badge/Node-%3E%3D20-339933?style=plastic&logo=node.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=plastic&logo=typescript&logoColor=white)
![License](https://img.shields.io/github/license/ArbitrHq/ocr-mini-bench?style=plastic)
![Latest release](https://img.shields.io/github/v/release/ArbitrHq/ocr-mini-bench?style=plastic)
![Last commit](https://img.shields.io/github/last-commit/ArbitrHq/ocr-mini-bench?style=plastic)
![Repo size](https://img.shields.io/github/repo-size/ArbitrHq/ocr-mini-bench?style=plastic)
![Open issues](https://img.shields.io/github/issues/ArbitrHq/ocr-mini-bench?style=plastic)



A lightweight, reproducible benchmark to compare OCR extraction quality, reliability, latency, and cost across multiple LLM providers on business documents.

## Quick start (recommended)

```bash
npm install
cp .env.example .env
# add API keys to .env

# optional safety checks
npm run benchmark:preflight

# smoke run: 1 model, 3 docs/domain, 2 runs
npm run benchmark:run -- \
  --models="gemini-3.1-flash-lite-preview" \
  --runs=2 \
  --docs-per-domain=3 \
  --parallel=1 \
  --provider-parallel

# postprocess
npm run postprocess:compare
npm run postprocess:metrics

# local UI
npm run ui:serve
```

## Dataset

This benchmark focuses on information abstraction for standard documents, in particular invoices, receipts, and logistics, with per-document ground-truth JSON labels. 

- Documents: `bench_documents/<Domain>/documents/*.pdf`
- Ground truth: `bench_documents/<Domain>/ground_truth/*.json`
- Dataset utilities and checks: `scripts/bootstrap_dataset.mjs`, `scripts/validate_dataset.mjs`, `scripts/summarize_labels.mjs`

## Getting started

### 1) Clone and install

```bash
git clone <your-repo-url>
cd ocr-mini-bench
npm install
cp .env.example .env
# Fill in your API keys in .env
```

### 2) Build checks (recommended)

```bash
npm run benchmark:preflight
```

This runs dataset validation and TypeScript checks before a benchmark run.

## Choosing models

Model config lives in:

- `config/models.public.json`

You can either:

- Keep all models in that file, or
- Run only a subset via CLI `--models`.

Examples:

```bash
# Run only three models by label
npm run benchmark:run -- --runs=1 --domains=invoices,receipts --models="GPT-5 nano,Claude Haiku 4.5,Gemini 2.5 Flash-Lite"

# Run by model_id values (no provider prefix in --models)
npm run benchmark:run -- --runs=1 --models="gpt-5-nano,claude-haiku-4-5,gemini-3.1-flash-lite-preview"
```

Important:

- `--models` matches `model_id` or `model_label`
- Do not prefix with provider in `--models` (use `gemini-3.1-flash-lite-preview`, not `google:gemini-3.1-flash-lite-preview`)

## Run benchmark (CLI)

```bash
# Typical run
npm run benchmark:run -- --runs=10 --domains=invoices,receipts,logistics --parallel=1 --provider-parallel
```

Useful flags:

- `--runs=<n>`: runs per model/document
- `--domains=<csv>`: e.g. `invoices,receipts,logistics`
- `--docs-per-domain=<n>`: sample small subsets for smoke runs
- `--parallel=<n>`: per-provider task parallelism
- `--provider-parallel`: run providers in parallel while keeping per-provider control
- `--models=<csv>`: optional filter, defaults to all models in config

## Track progress (`state.json`)

During a run, progress is written to:

- `artifacts/checkpoints/state.json`
- `artifacts/checkpoints/runs.jsonl`
- `artifacts/checkpoints/raw.runs.jsonl`

Quick monitor commands:

```bash
npm run benchmark:checkpoint

# optional watch
while true; do clear; npm run benchmark:checkpoint; sleep 10; done
```

`state.json` includes run progress, success/fail counts, and rolling cost estimates.

## Failures and retries

The benchmark is checkpointed and resumable.

```bash
# Rebuild final outputs from checkpoint data
npm run benchmark:rebuild

# Retry failed tasks only
npm run benchmark:run -- --retry-failed

# Resume unfinished tasks
npm run benchmark:run -- --resume
```

This prevents losing completed work during long runs.

## Post-process pipeline

Pipeline:

1. `benchmark:run` writes canonical raw output
2. `postprocess:compare` scores raw output against ground truth
3. `postprocess:metrics` builds leaderboard snapshots

`benchmark:run` writes canonical raw to:

- `artifacts/postprocess/raw.jsonl`
- `artifacts/checkpoints/raw.jsonl` (latest deduplicated checkpoint view)

Run stages 2 and 3:

```bash
npm run postprocess:compare
npm run postprocess:metrics
```

These produce:

- raw run output (`raw.jsonl`)
- ground-truth comparison output
- final leaderboard/debug artifacts for visualization

For legacy checkpoints that only have `runs.jsonl`, you can still build raw with:

```bash
npm run postprocess:raw
```

## Small frontend for results + debugging

Serve the included minimal UI:

```bash
npm run ui:serve
```

If port `4173` is busy:

```bash
npm run ui:serve -- --port=4174
```

Then open the local URL printed in terminal. The UI auto-discovers the latest artifacts under `artifacts/`.

Frontend file to publish/copy:

- `leaderboard.frontend.json` (this is the artifact your external frontend should consume)

The UI provides:

- Leaderboard view (aggregated metrics)
- Debug view (document/key/model inspection)

## Output artifacts

Primary outputs under `artifacts/`:

- Checkpoints: `checkpoints/runs.jsonl`, `checkpoints/raw.runs.jsonl`, `checkpoints/raw.jsonl`, `checkpoints/state.json`
- Raw: `postprocess/raw.jsonl`
- Compare: `postprocess/comparison.jsonl`, `postprocess/comparison.summary.json`
- Metrics: `postprocess/metrics.snapshot.json`
- Leaderboard: `postprocess/leaderboard.aggregation.json`, `postprocess/leaderboard.frontend.json`
- Debug and snapshots from run: `latest.debug.json`, `snapshot-*.debug.json`, `latest.json`, `snapshot-*.json`
