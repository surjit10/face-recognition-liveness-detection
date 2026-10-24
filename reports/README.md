# Reports

This directory stores public-safe or internal aggregate evidence:

- `phase_receipts/`: gate decisions and exact commands/results.
- `environments/`: Python/package/hardware/provider inventories.
- `audits/`: data integrity and manifest summaries.
- `evaluation/`: metrics tied to data/split/model/config hashes.
- `benchmarks/`: target-hardware latency/resource measurements.

Never store raw frames, crops, embeddings, galleries, real names, secrets or restricted dataset samples in reports.

Current P0 evidence:

- `environments/environment_local_before.json`: system-Python preflight before the isolated environment was installed; it correctly records that the system environment has broken unrelated requirements and is not the project runtime.
- `audits/frad_audit.json`: draft structural audit tied to the current inventory hash; face-model checks remain pending.
