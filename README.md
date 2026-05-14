# OCR Mini-bench

[![Live Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-OCR%20Mini--bench-0ea5e9?style=plastic)](https://arbitrhq.ai/leaderboards/ocr-mini-bench)
[![Blog](https://img.shields.io/badge/Blog-OCR%20Mini--bench-22c55e?style=plastic)](https://arbitrhq.ai/blog/ocr-mini-bench)
![Python >=3.11](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=plastic&logo=python&logoColor=white)
![License](https://img.shields.io/github/license/ArbitrHq/ocr-mini-bench?style=plastic)
![Latest release](https://img.shields.io/github/v/release/ArbitrHq/ocr-mini-bench?style=plastic)
![Last commit](https://img.shields.io/github/last-commit/ArbitrHq/ocr-mini-bench?style=plastic)
![Repo size](https://img.shields.io/github/repo-size/ArbitrHq/ocr-mini-bench?style=plastic)
![Open issues](https://img.shields.io/github/issues/ArbitrHq/ocr-mini-bench?style=plastic)

A lightweight, reproducible benchmark to compare OCR extraction quality, reliability, latency, and cost across multiple LLM providers on business documents.

[Typescript Version](https://github.com/ArbitrHq/ocr-mini-bench-typescript)

## Quick start

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
# add API keys to .env

# smoke run: 1 model, 3 docs/domain, 2 runs
uv run ocr-bench \
  --models=gemini-3.1-flash-lite-preview \
  --runs=2 \
  --docs-per-domain=3 \
  --parallel=1 \
  --provider-parallel

# postprocess
uv run ocr-bench-compare
uv run ocr-bench-metrics

# local UI
uv run python scripts/serve_ui.py
```

## Dataset

This benchmark focuses on information abstraction for standard documents — invoices, receipts, and logistics — with per-document ground-truth JSON labels.

- Documents: `bench_documents/<Domain>/*.pdf`
- Ground truth: `bench_documents/<Domain>/ground_truth/*.json`
- Generated manifest: `dataset/manifest.json` (pairs each PDF with its ground-truth file)
- Dataset utilities: `scripts/bootstrap_dataset.py`, `scripts/validate_dataset.py`, `scripts/summarize_labels.py`

### Bring your own dataset

The framework is dataset-agnostic. To benchmark your own:

1. **Add documents and labels** under `bench_documents/<YourDomain>/`. PDFs at the domain root; one JSON per PDF under `ground_truth/` with the same basename. Each ground-truth leaf is a field descriptor:

   ```json
   {
     "invoice_number": { "value": "INV-1689", "critical": true, "type": "string" },
     "total_amount":   { "value": 1234.56,    "critical": true, "type": "float" },
     "due_date":       { "value": "2026-04-15", "critical": false, "type": "date" }
   }
   ```

   `value` is required (`null` means "known to be absent"). `critical` flags fields that count toward critical accuracy. `type` (`string` / `date` / `integer` / `float`) drives the default match mode; override with `match_mode` (`exact`, `normalized_text`, `contains`, `numeric`). Nesting is allowed — keys are flattened for scoring.

2. **Generate and validate the manifest:**

   ```bash
   uv run python scripts/bootstrap_dataset.py
   uv run python scripts/validate_dataset.py
   uv run python scripts/summarize_labels.py    # check labeling coverage
   ```

3. **Run on your domain:**

   ```bash
   uv run ocr-bench --domains=yourdomain --runs=2 --docs-per-domain=3 --provider-parallel
   ```

The prompts in `prompts/ocr/benchmark/` are domain-agnostic — they ask the model to extract whatever field list the ground truth defines, so no prompt edits needed.

## Getting started

### 1) Clone and install

```bash
git clone <your-repo-url>
cd ocr-mini-bench
uv sync
cp .env.example .env
# Fill in your API keys in .env
```

### 2) Validate the dataset (optional)

```bash
uv run python scripts/validate_dataset.py
```

## Choosing models

Model config lives in:

- `config/models.public.json`

You can either:

- Keep all models in that file, or
- Run only a subset via CLI `--models`.

Examples:

```bash
# Run only three models by label
uv run ocr-bench --runs=1 --domains=invoices,receipts --models="GPT-5 nano,Claude Haiku 4.5,Gemini 2.5 Flash-Lite"

# Run by model_id values (no provider prefix in --models)
uv run ocr-bench --runs=1 --models=gpt-5-nano,claude-haiku-4-5,gemini-3.1-flash-lite-preview
```

Important:

- `--models` matches `model_id` or `model_label`
- Do not prefix with provider in `--models` (use `gemini-3.1-flash-lite-preview`, not `google:gemini-3.1-flash-lite-preview`)

## Run benchmark (CLI)

```bash
# Typical run
uv run ocr-bench --runs=10 --domains=invoices,receipts,logistics --parallel=1 --provider-parallel
```

Useful flags:

- `--runs=<n>`: runs per model/document
- `--domains=<csv>`: e.g. `invoices,receipts,logistics`
- `--docs-per-domain=<n>`: sample small subsets for smoke runs
- `--parallel=<n>`: per-provider task parallelism
- `--provider-parallel`: run providers in parallel while keeping per-provider control
- `--models=<csv>`: optional filter, defaults to all models in config
- `--output-dir=<path>`: snapshot + postprocess output directory (default: `artifacts/`)
- `--checkpoint-dir=<path>`: checkpoint directory (default: `artifacts/checkpoints/`)

## Track progress (`state.json`)

During a run, progress is written to:

- `artifacts/checkpoints/state.json`
- `artifacts/checkpoints/runs.jsonl`
- `artifacts/checkpoints/raw.runs.jsonl`

Quick monitor commands:

```bash
uv run python scripts/checkpoint_status.py

# optional watch
while true; do clear; uv run python scripts/checkpoint_status.py; sleep 10; done
```

`state.json` includes run progress, success/fail counts, and rolling cost estimates.

## Failures and retries

The benchmark is checkpointed and resumable.

```bash
# Rebuild final outputs from checkpoint data
uv run python scripts/rebuild_from_checkpoint.py

# Retry failed tasks only
uv run ocr-bench --retry-failed

# Resume unfinished tasks
uv run ocr-bench --resume
```

This prevents losing completed work during long runs.

## Post-process pipeline

Pipeline:

1. `ocr-bench` writes canonical raw output
2. `ocr-bench-compare` scores raw output against ground truth
3. `ocr-bench-metrics` builds leaderboard snapshots

`ocr-bench` writes canonical raw to:

- `artifacts/postprocess/raw.jsonl`
- `artifacts/checkpoints/raw.jsonl` (latest deduplicated checkpoint view)

Run stages 2 and 3:

```bash
uv run ocr-bench-compare
uv run ocr-bench-metrics
```

These produce:

- raw run output (`raw.jsonl`)
- ground-truth comparison output
- final leaderboard/debug artifacts for visualization (`leaderboard.frontend.json` includes per-row `metric_ranges`)

For legacy checkpoints that only have `runs.jsonl`, you can still build raw with:

```bash
uv run ocr-bench-export-raw
```

## Small frontend for results + debugging

Serve the included minimal UI:

```bash
uv run python scripts/serve_ui.py
```

If port `4173` is busy:

```bash
uv run python scripts/serve_ui.py --port=4174
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

## Development

### Running tests

```bash
uv run pytest                          # all tests
uv run mypy src                        # type check
uv run ruff check src tests scripts    # lint
```

### Adding a new model

1. Add the model to `config/models.public.json`:

```json
{
  "provider": "openai",
  "model_id": "gpt-6-mini",
  "model_label": "GPT-6 mini",
  "tier": "balanced"
}
```

2. Make sure the corresponding API key is in `.env`:

```
OPENAI_API_KEY=sk-...
```

3. Run with the new model:

```bash
uv run ocr-bench --models="GPT-6 mini" --runs=1 --docs-per-domain=1
```

Supported providers: `openai`, `anthropic`, `google`, `mistral`

Tiers: `budget`, `balanced`, `sota` (used for leaderboard grouping)

### Adding a new provider

Provider implementations live in `src/ocr_mini_bench/ocr/providers/`. To add a new provider:

1. Create `src/ocr_mini_bench/ocr/providers/<provider>.py` implementing the OCR interface
2. Wire the provider into `src/ocr_mini_bench/ocr/runner.py`
3. Add the API key to `.env.example`
4. Add models to `config/models.public.json`

### Project structure

```
src/ocr_mini_bench/
├── benchmark/           # Benchmark orchestration and scoring
│   ├── scoring/         # Match logic, text normalization
│   └── run/             # Parallelization, progress, aggregation
├── cli/                 # CLI entry points
│   └── postprocess/     # Post-processing CLIs
├── config/              # Centralized paths and model catalog
├── lib/                 # Shared utilities (type guards, errors)
├── ocr/                 # OCR provider implementations
│   └── providers/       # OpenAI, Anthropic, Google, Mistral
└── postprocess/         # Raw/comparison/metrics transforms

scripts/                 # Operational helpers (status, validate, bootstrap, rebuild, serve_ui)
tests/                   # unit, replay (frozen responses), scripts, smoke (live API)

config/
└── models.public.json   # Model definitions

dataset/
├── manifest.json        # Document registry
└── <domain>/            # PDFs and ground truth per domain

prompts/
└── ocr/benchmark/       # System and user prompts
```
