# Documentation

The final prototype documentation is public-safe and contains no biometric data:

- [ARCHITECTURE.md](ARCHITECTURE.md) — unified UI, gallery/history storage,
  camera-to-decision flow, gates, states and runtime configuration.
- [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) — setup, all application screens, supervised
  operation, authorized demo recording, shutdown and troubleshooting.
- [PROTOTYPE_MODEL_CARD.md](PROTOTYPE_MODEL_CARD.md) — models, pinned hashes, data
  scope, metrics, license and mandatory limitations.
- [PRIVACY.md](PRIVACY.md) — local-only processing, zero retention, restricted paths
  and logging rules.
- [EVALUATION.md](EVALUATION.md) — exploratory split, threshold selection, test results,
  benchmark summary and deferred acceptance cases.
- [MLOPS.md](MLOPS.md) — implemented CI/artifact checks and the remaining release and
  experiment-tracking gaps.

Development runbooks for embedding extraction, PAD training and the runtime pipeline
remain in [EMBEDDING_EXTRACTION.md](EMBEDDING_EXTRACTION.md),
[KAGGLE_EMBEDDING_RUNBOOK.md](KAGGLE_EMBEDDING_RUNBOOK.md) and
[RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md).

Planning and progress material remains isolated in the removable `reference/`
directory.
