# Submission Structure and Packaging

This repository is organized into three clear zones. Do not manually copy the whole
working directory for submission because it also contains local environments, private
biometric artifacts and experiment storage.

## Submission-safe project structure

```text
FaceGuard/
├── README.md                 # project entry point and demo command
├── pyproject.toml            # Python package metadata
├── requirements-*.in|lock   # runtime, development and local MLOps dependencies
├── src/faceguard/            # reusable application and verification logic
├── scripts/                  # command-line entry points and submission builder
├── tests/                    # automated unit, contract, integration and privacy tests
├── configs/                  # frozen runtime, scope and performance configuration
├── docs/                     # architecture, runbook, evaluation, privacy and model card
├── reports/                  # aggregate audits, metrics, benchmarks and receipts
├── data/                     # public manifests plus ignored private data zones
├── models/                   # public schema plus ignored private model zones
├── deployment/               # Docker and Compose packaging
├── .github/workflows/        # public and private CI definitions
├── dvc.yaml / dvc.lock       # local reproducibility pipeline
└── reference/                # removable planning/history and restricted archive
```

## What is intentionally excluded

The submission ZIP never includes:

- `.git/`, `.venv/`, `.venv-mlops/`, editor state or caches;
- `data/raw/`, `data/interim/` or `data/processed/`;
- `models/private/` or `models/releases/`;
- `reports/private/`, local attendance history or exported biometric data;
- `mlflow.db`, `mlruns/`, old `dist/` output or `reference/archive/`;
- internal preflight/runtime inventory and promotion receipts containing machine or
  operator-specific details;
- model weights, galleries, frames, crops, embeddings, credentials or restricted
  dataset content.

The private runtime models, gallery, environments and MLflow storage stay in their
expected local paths because the demonstration depends on them. Completed research,
Kaggle and data-preparation material is isolated under
`reference/archive/final-submission-excluded/`.

## Build the final ZIP

Run the privacy scan, tests and allowlisted builder:

```bash
.venv/bin/python scripts/scan_artifacts.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_submission.py
```

The output is `dist/faceguard-academic-submission.zip`. It contains a generated
`SUBMISSION_MANIFEST.json` with the size and SHA-256 hash of every included file.

Planning/history under `reference/` is excluded. Do not use the working directory as
the submission artifact; submit only the generated ZIP.

## View before and after versions

In VS Code, open **Source Control** (`Ctrl+Shift+G`) and click any changed file. The
left side is the last committed version and the right side is the current version.

From the terminal:

```bash
git status --short
git diff --stat
git diff -- path/to/file
git log --oneline --decorate -10
```

After two versions are committed, compare them with `git diff OLD_COMMIT NEW_COMMIT`.
