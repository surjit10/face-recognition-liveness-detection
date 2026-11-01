# Data Boundary

The active `data/` directory contains only public-safe manifests and empty placeholders:

```text
data/
├── manifests/   # pseudonymous split definitions, hashes and summaries
├── raw/         # empty local-only input placeholder
└── interim/     # empty local-only working placeholder
```

Raw faces, crops, embeddings and other biometric data are never included in the
submission. The completed processed dataset was moved to the restricted local archive
at `reference/archive/final-submission-excluded/restricted-data/processed/`; it is not
required by the live application.

The frozen recognition evaluation is represented by aggregate reports and manifests
under `reports/evaluation/`, `reports/audits/` and `data/manifests/`. Do not add
identifiable names or image content to version control.
