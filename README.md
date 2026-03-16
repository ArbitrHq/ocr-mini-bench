# OCR Mini-bench

[![Live Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-OCR%20Mini--bench-0ea5e9?style=flat-square)](https://arbitrhq.ai/leaderboards/ocr-mini-bench)
[![Blog](https://img.shields.io/badge/Blog-OCR%20Mini--bench-22c55e?style=flat-square)](https://arbitrhq.ai/blog/ocr-mini-bench)
![Node >=20](https://img.shields.io/badge/Node-%3E%3D20-339933?style=flat-square&logo=node.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat-square&logo=typescript&logoColor=white)
![License](https://img.shields.io/github/license/ArbitrHq/ocr-mini-bench?style=flat-square)
![Latest release](https://img.shields.io/github/v/release/ArbitrHq/ocr-mini-bench?style=flat-square)
![Last commit](https://img.shields.io/github/last-commit/ArbitrHq/ocr-mini-bench?style=flat-square)
![Repo size](https://img.shields.io/github/repo-size/ArbitrHq/ocr-mini-bench?style=flat-square)
![Open issues](https://img.shields.io/github/issues/ArbitrHq/ocr-mini-bench?style=flat-square)



**Optimal model for standard OCR tasks benchmark**

A lightweight, reproducible benchmark to compare OCR extraction quality, reliability, latency, and cost across multiple LLM providers using business-oriented metrics. It tests OCR capabilities for (document + expected keys)-pairs. Accompanying blogpost: [ref]

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

# Run by model_id values
npm run benchmark:run -- --runs=1 --models="openai:gpt-5-nano,anthropic:claude-haiku-4.5"
```

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
npm run benchmark:run -- --mode=retry-failed

# Resume unfinished tasks
npm run benchmark:run -- --mode=resume
```

This prevents losing completed work during long runs.

## Post-process pipeline

After a run, generate stable outputs in three stages:

```bash
npm run postprocess:raw
npm run postprocess:compare
npm run postprocess:metrics
```

These produce:

- raw normalized run output
- ground-truth comparison output
- final leaderboard/debug artifacts for visualization

## Small frontend for results + debugging

Serve the included minimal UI:

```bash
npm run ui:serve
```

Then open the local URL printed in terminal. The UI provides:

- Leaderboard view (aggregated metrics)
- Debug view (document/key/model inspection)

## Output artifacts

Primary outputs are written under `artifacts/`, including:

- checkpoint files
- per-stage postprocess files
- final leaderboard/debug JSON files
