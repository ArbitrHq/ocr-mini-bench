# Project Structure

## Runtime Flow

1. `src/cli/run-benchmark.ts`
- Parses CLI args.
- Handles checkpoint/resume/retry wiring.
- Calls benchmark runner.
- Writes `artifacts/latest*.{json,md}` and timestamped snapshots.

2. `src/benchmark/run.ts`
- Loads config + prompts + prepared documents.
- Builds task grid: `model x document x run`.
- Executes tasks with bounded parallelism.
- Scores output and aggregates leaderboard/domain tables.
- Produces public + debug snapshot payload.
  Supporting modules:
  - `src/benchmark/run/aggregation.ts`
  - `src/benchmark/run/identity.ts`
  - `src/benchmark/run/math.ts`
  - `src/benchmark/run/pool.ts`
  - `src/benchmark/run/provider-lanes.ts`
  - `src/benchmark/run/progress.ts`

3. `src/ocr/runner.ts`
- Thin facade that routes OCR calls by provider.
  Supporting modules:
  - `src/ocr/providers/{openai,anthropic,google,mistral}.ts`
  - `src/ocr/gemini-cache.ts`
  - `src/ocr/provider-utils.ts`
  - `src/ocr/text-readers.ts`
  - `src/ocr/parsing.ts`
  - `src/ocr/cost.ts`
  - `src/ocr/catalog.ts`

4. `src/benchmark/scoring.ts`
- Thin export surface for scoring.
  Supporting modules:
  - `src/benchmark/scoring/extraction.ts`
  - `src/benchmark/scoring/matching.ts`
  - `src/benchmark/scoring/text-normalization.ts`
  - `src/benchmark/scoring/prompt-builders.ts`
  - `src/benchmark/scoring/score.ts`

5. `src/benchmark/dataset.ts`
- Reads `dataset/manifest.json`.
- Loads and normalizes ground truth JSONs.
- Builds dataset summary used in snapshot headers.

## Configuration and Data

- `config/models.public.json`
  Public benchmark model lineup and default run parameters.

- `dataset/manifest.json`
  Generated source-of-truth for which documents participate.

- `bench_documents/<Domain>/`
  Canonical benchmark PDFs and `ground_truth/*.json` labels.

- `prompts/ocr/benchmark/*.txt`
  Prompt templates used by runner/scorer.

## Operational Scripts

- `scripts/bootstrap_dataset.mjs`
  Rebuilds manifest from `bench_documents`.

- `scripts/validate_dataset.mjs`
  Checks manifest/GT consistency and key comparability.

- `scripts/summarize_labels.mjs`
  Summarizes labeling coverage.

- `scripts/checkpoint_status.mjs`
  Summarizes checkpoint progress/failures by model.

## Generated Outputs

- `artifacts/latest.json`
  Public snapshot payload for leaderboard rendering.

- `artifacts/latest.debug.json`
  Full run-level debug payload (per-document/per-run details).

- `artifacts/latest.md`
  Markdown leaderboard.

- `artifacts/checkpoints/*`
  Append-only call log + checkpoint state for resume/retry.
