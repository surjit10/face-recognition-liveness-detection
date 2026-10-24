# FaceGuard AI v2 — Repository Structure and Ownership

This is the maintainable repository structure initialized on 24 October 2026. Create implementation files only when their phase starts; do not populate the tree with fabricated artifacts, empty model files, or fake reports. Historical v1 and pre-v2 crop tooling are quarantined under `reference/archive/` so no application or test may import them. They remain preserved until the user removes the temporary `reference/` area.

## Target tree

```text
face-antispoofing-recognition-system-main/
├── README.md                         # public project entry; measured claims only
├── pyproject.toml                    # package metadata and developer tooling
├── .gitignore                        # raw biometric/private/model/run exclusions
├── configs/
│   ├── decisions.yaml                # verified/frozen decision register
│   ├── data.yaml                     # dataset roots + manifest hashes
│   ├── recognition.yaml              # model/scoring/threshold policy
│   ├── pad.yaml                      # training + preprocessing contract
│   └── runtime.yaml                  # queue/quality/tracking/session policy
├── data/
│   ├── README.md                     # acquisition and privacy instructions
│   ├── manifests/                    # versionable metadata; no raw images
│   ├── raw/                          # local-only restricted input, Git-ignored
│   ├── interim/                      # local/Kaggle scratch, Git-ignored
│   └── processed/                    # restricted derived data, Git-ignored
├── notebooks/
│   ├── 01_kaggle_pad.ipynb           # thin orchestration only
│   └── 02_kaggle_frad_embeddings.ipynb
├── src/faceguard/
│   ├── __init__.py
│   ├── config.py                     # strict schema loading; reject unknown keys
│   ├── contracts.py                  # observations/results/state dataclasses
│   ├── models.py                     # InsightFace and PAD ONNX adapters
│   ├── preprocessing.py              # deterministic crop/color/normalization
│   ├── gallery.py                    # private embeddings + search/delete
│   ├── tracking.py                   # single-face IoU continuity only
│   ├── pipeline.py                   # fail-closed verification state machine
│   ├── video.py                      # sequential/live sources and bounded queue
│   ├── metrics.py                    # metric definitions and runtime telemetry
│   ├── train_pad.py                  # reusable training implementation
│   ├── evaluate.py                   # recognition/PAD/video evaluation
│   └── export.py                     # ONNX export, parity and artifact hashing
├── scripts/
│   ├── preflight.py
│   ├── audit_data.py
│   ├── prepare_pad.py
│   ├── extract_gallery.py
│   ├── benchmark.py
│   └── run_demo.py
├── tests/
│   ├── unit/                         # pure logic, synthetic inputs
│   ├── contract/                     # schema/model/interface/hash checks
│   ├── integration/                  # authorized tiny files/MP4 or synthetic fixtures
│   ├── privacy/                      # log/public-artifact leakage checks
│   └── fixtures/                     # visibly synthetic or authorized non-public inputs
├── models/
│   ├── manifest.json                 # public-safe metadata and hashes
│   ├── private/                      # weights/gallery, Git-ignored
│   └── releases/                     # local immutable candidates, Git-ignored
├── reports/
│   ├── phase_receipts/
│   ├── environments/
│   ├── audits/
│   ├── evaluation/
│   └── benchmarks/
├── deployment/
│   ├── Dockerfile
│   ├── compose.yaml                  # only if needed for the chosen local demo
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   ├── model_card.md
│   ├── limitations.md
│   └── runbook.md
├── reference/                        # planning/control documents; no biometric data
│   ├── README.md
│   ├── FaceGuard_AI_Implementation_Blueprint.md
│   ├── architecture/
│   ├── archive/                      # legacy/disposable material, isolated from v2
│   ├── governance/
│   ├── tracking/
│   └── templates/
├── requirements-runtime.in
├── requirements-runtime.lock
├── requirements-train.in
├── requirements-train.lock
├── requirements-dev.in
└── requirements-dev.lock
```

## Dependency direction

```text
scripts / notebooks / CLI
            |
            v
pipeline + train/evaluate/export orchestration
            |
            v
models + preprocessing + gallery + tracking + video
            |
            v
contracts + config + pure metrics
```

Lower layers never import scripts or notebooks. Training code may import shared preprocessing/contracts, but CPU runtime must not import torch, MLflow, DVC, notebook code or training-only modules. Notebooks call importable source functions and contain no unique business logic.

## Ownership boundaries

| Area | Owns | Must not own |
|---|---|---|
| `reference/` | Contract, tracker, governance, templates, and Git-ignored temporary archive | Active runtime code, current datasets, embeddings, release model files |
| `configs/` | Strict resolved choices and decision states | Secrets, hard-coded user paths, fabricated thresholds |
| `data/manifests/` | Portable metadata and hashes | Redistributable copies of restricted images |
| `src/faceguard/` | Tested reusable product/training logic | Notebook-only implementations or absolute paths |
| `scripts/` | Thin command entry points | Core algorithms duplicated from `src/` |
| `tests/` | Unit/contract/integration/privacy evidence | Unlicensed biometric fixtures in public Git |
| `models/private/` | Approved weights and protected gallery | Publicly committed biometric vectors/restricted weights |
| `reports/` | Audits, metrics, receipts and benchmarks | Raw face crops, embeddings, secrets |
| `docs/` | Public-safe explanation and measured claims | Unimplemented claims or private evidence |

## Privacy zones

- **Public-safe:** source code, schemas, synthetic fixtures, aggregate reports, documentation and checksums after privacy review.
- **Internal:** environment reports, detailed manifests without names, run metadata and error summaries.
- **Restricted biometric:** raw/cropped faces, video, embeddings, galleries and any artifact permitting identity linkage. Keep outside public Git; encrypt/protect access where feasible.
- **Restricted licensed:** third-party dataset contents and model weights whose terms prohibit redistribution.

Every generated artifact declares one of these classifications in its phase receipt. When classification is unclear, treat it as restricted.

## Maintainability rules

1. One implementation per responsibility; do not duplicate preprocessing between training and runtime.
2. Strict configuration schemas reject unknown/missing keys and save resolved configs with runs.
3. All paths are supplied through configuration/CLI and normalized safely; no developer-specific absolute paths in source.
4. Interfaces stay small and concrete. Add abstractions only when two implemented backends genuinely require them.
5. Release code is fail-closed and hash-verified; experimentation code cannot silently alter release assets.
6. Tests mirror source responsibilities and name the security/privacy behavior they protect.
7. Public documentation distinguishes historical, measured, pending and unsupported claims.
8. Empty directories are not evidence. Create each area with its first real, phase-authorized artifact.

## Interview presentation order

1. `README.md`: problem, demo, verified results and limitations.
2. `docs/architecture.md`: one end-to-end diagram and module boundaries.
3. `docs/evaluation.md`: splits, threshold methodology and error analysis.
4. `docs/model_card.md`: model/data/metrics/limitations/ethical use.
5. `reports/benchmarks/`: target-hardware latency and resource evidence.
6. `reference/tracking/STATUS.md`: transparent implementation status and remaining work.

This order demonstrates product thinking, computer-vision correctness, reproducibility, privacy and engineering discipline without presenting unnecessary infrastructure.
