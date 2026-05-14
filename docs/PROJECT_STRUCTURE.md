# Project Structure

## Runtime Flow

1. `src/ocr_mini_bench/cli/run_benchmark.py`
   - Parses CLI args.
   - Handles checkpoint/resume/retry wiring.
   - Computes the benchmark fingerprint that gates resume/retry.
   - Calls the benchmark orchestrator.
   - Writes `artifacts/latest*.{json,md}` and timestamped snapshots.

2. `src/ocr_mini_bench/benchmark/orchestrator.py`
   - Loads config + prompts + prepared documents.
   - Builds task grid: `model × document × run`.
   - Executes tasks with bounded parallelism using a shared `httpx.AsyncClient`.
   - Scores output and aggregates leaderboard / per-domain tables.
   - Produces public + debug snapshot payload.

   Supporting modules under `src/ocr_mini_bench/benchmark/run/`:
   - `aggregation.py`
   - `identity.py`
   - `math.py`
   - `pool.py`
   - `provider_lanes.py`
   - `progress.py`

3. `src/ocr_mini_bench/ocr/runner.py`
   - Thin facade that routes OCR calls by provider.

   Supporting modules under `src/ocr_mini_bench/ocr/`:
   - `providers/{openai,anthropic,gemini,mistral}.py`
   - `gemini_cache.py`
   - `provider_utils.py`
   - `text_readers.py`
   - `parsing.py`
   - `cost.py`
   - `catalog.py`

4. `src/ocr_mini_bench/benchmark/scoring/` (no top-level facade — import modules directly)
   - `extraction.py`
   - `matching.py`
   - `text_normalization.py`
   - `prompt_builders.py`
   - `score.py`

5. `src/ocr_mini_bench/benchmark/dataset.py`
   - Reads `dataset/manifest.json`.
   - Loads and normalizes ground truth JSONs (`normalize_ground_truth.py`).
   - Builds dataset summary used in snapshot headers.

## Postprocess Pipeline

Driven by the `ocr-bench-{compare,metrics,export-raw}` CLIs in
`src/ocr_mini_bench/cli/postprocess/`. Logic lives under
`src/ocr_mini_bench/postprocess/`:

- `raw_contract.py` — canonical raw-row shape
- `aggregate.py` — leaderboard / per-domain aggregation
- `io.py` — JSONL + JSON readers/writers with TS-compatible formatting
- `types.py` — pydantic models

## Configuration and Data

- `config/models.public.json` — public benchmark model lineup and default run parameters.
- `dataset/manifest.json` — generated source-of-truth for which documents participate.
- `bench_documents/<Domain>/` — canonical benchmark PDFs and `ground_truth/*.json` labels.
- `prompts/ocr/benchmark/*.txt` — system + user prompt templates.

## Operational Scripts

All under `scripts/` and invoked as `uv run python scripts/<name>.py`:

- `bootstrap_dataset.py` — rebuilds `dataset/manifest.json` from `bench_documents/`.
- `validate_dataset.py` — checks manifest / GT consistency and key comparability.
- `summarize_labels.py` — summarizes labeling coverage per domain.
- `checkpoint_status.py` — summarizes checkpoint progress/failures by model.
- `rebuild_from_checkpoint.py` — reconstructs snapshot artifacts from a checkpoint dir
  (note: uses its own per-document `pass@N` math, distinct from the orchestrator's
  probabilistic aggregation).
- `serve_ui.py` — tiny stdlib HTTP server for the `ui/` frontend.
- `parity_diff.py`, `parity_show.py` — historical Phase 6 parity tools; safe to delete.

## Generated Outputs

- `artifacts/latest.json` — public snapshot payload for leaderboard rendering.
- `artifacts/latest.debug.json` — full run-level debug payload (per-document/per-run details).
- `artifacts/latest.md` — markdown leaderboard.
- `artifacts/snapshot-<ts>.{json,debug.json}` — timestamped copies of the latest.
- `artifacts/postprocess/raw.jsonl` — canonical raw run output.
- `artifacts/postprocess/comparison.jsonl` — raw scored against ground truth.
- `artifacts/postprocess/comparison.summary.json` — aggregate summary.
- `artifacts/postprocess/metrics.snapshot.json` — post-processed metrics.
- `artifacts/postprocess/leaderboard.aggregation.json` — leaderboard rows (internal shape).
- `artifacts/postprocess/leaderboard.frontend.json` — UI-facing leaderboard (consumed by `ui/`).
- `artifacts/checkpoints/state.json` — run progress + cost summary.
- `artifacts/checkpoints/runs.jsonl` — append-only per-run log used by `--resume` / `--retry-failed`.
- `artifacts/checkpoints/raw.runs.jsonl` — full per-run raw log (incl. prompts + parsed payload).
- `artifacts/checkpoints/raw.jsonl` — deduplicated canonical raw checkpoint view.

The `artifacts/` directory is gitignored except `artifacts/public-rc-gemini31/`,
which is kept as a canonical example snapshot.

## Tests

- `tests/unit/` — pure-function unit tests (no I/O, no network).
- `tests/replay/` — offline tests replaying frozen HTTP responses
  (`tests/fixtures/responses/`) through the providers + orchestrator.
- `tests/scripts/` — subprocess tests for the helper scripts in `scripts/`.
- `tests/smoke/` — opt-in live-API tests; require provider API keys in `.env`.

Markers: `unit`, `replay`, `smoke` (declared in `pyproject.toml`).
