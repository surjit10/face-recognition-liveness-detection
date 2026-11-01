# Reports

This directory stores public-safe or internal aggregate evidence:

- `phase_receipts/`: gate decisions and exact commands/results.
- `environments/`: Python/package/hardware/provider inventories.
- `audits/`: data integrity and manifest summaries.
- `evaluation/`: metrics tied to data/split/model/config hashes.
- `benchmarks/`: target-hardware latency/resource measurements.

Never store raw frames, crops, embeddings, galleries, real names, secrets or restricted dataset samples in reports.

The active reports are limited to the final recognition evaluation, PAD evidence,
runtime readiness, target-hardware benchmarks and privacy-safe phase receipts. Older
machine inventories, preprocessing iterations and the superseded promotion receipt are
archived under `reference/archive/final-submission-excluded/reports/`.
