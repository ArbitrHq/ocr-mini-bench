# Benchmark Documents

This folder contains the canonical benchmark dataset used by the standalone runner.

## Layout

- `Invoices/`
- `Receipts/`
- `Logistics/`

Each domain folder contains:
- source PDFs (one file per document)
- `ground_truth/*.json` labels

## Ground Truth Expectations

- One JSON file per document.
- JSON uses field-object schema (`value`, `critical`, `type`, optional metadata).
- Nested fields are allowed; they are flattened for scoring.

## Maintenance Rules

1. Keep PDF filenames stable once benchmark results are published.
2. Re-run manifest generation after adding/removing docs:
   - `npm run benchmark:bootstrap-dataset`
3. Validate before runs:
   - `npm run benchmark:validate`
4. Keep canonical PDFs at domain root; store alternates in `reduced_size/` or `full_size/`.
