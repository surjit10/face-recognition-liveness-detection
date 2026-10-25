# FaceGuard AI v2

FaceGuard AI v2 is a consent-based face recognition and RGB presentation-attack detection project. The intended production flow is SCRFD detection and alignment, MobileNetV3-Small PAD, ArcFace embeddings, calibrated open-set gallery matching, and a fail-closed local video runtime.

## Current state

The hash-verified PAD candidate, audited ArcFace embeddings, protected gallery
implementation, PAD-before-recognition state machine, bounded live queue and strict
runtime configuration are present. The recognition gallery/threshold and local
InsightFace model files are still required, so the runtime deliberately refuses to
start instead of guessing. The v2 system is **not yet validated or released**.

See `docs/RUNTIME_PIPELINE.md` for the ordered completion and verification runbook.

Implementation proceeds through these gates:

```text
P0 audit/governance
 -> P1 recognition
 -> P2 anti-spoofing
 -> P3 sequential-video integration
 -> P4 live runtime/benchmarking
 -> P5 reproducible release
```

The temporary `reference/` directory contains the implementation contract, tracker and archived pre-v2 material. It is intentionally isolated from application imports so it can be removed after the project is completed and its permanent public documentation has been promoted to `docs/`.

## Repository areas

| Path | Responsibility |
|---|---|
| `configs/` | Strict decisions and resolved run configuration |
| `data/` | Private inputs/derived data plus versionable manifests |
| `src/faceguard/` | Reusable application and ML implementation |
| `scripts/` | Thin command-line entry points |
| `tests/` | Unit, contract, integration and privacy checks |
| `models/` | Public-safe manifest plus ignored private artifacts |
| `reports/` | Audits, evaluations, benchmarks and phase receipts |
| `deployment/` | Minimal CPU-runtime packaging |
| `docs/` | Final public-safe architecture, evaluation and model documentation |
| `reference/` | Temporary planning, tracking and archived material |

## Privacy boundary

Face images, videos, embeddings and galleries are restricted biometric data. They must not be committed to public Git, logged, copied into reports, or uploaded to third-party services without verified permission. The current legacy recognition crops are stored under the Git-ignored `data/processed/` zone.

## Development status

Do not install arbitrary package versions or download model weights from mirrors. Environment locks, model hashes, dataset manifests, splits and thresholds are established only through the phase gates. Until those gates pass, any generated implementation is unverified.

Current developer entry points:

- `docs/EMBEDDING_EXTRACTION.md` for local audit and candidate embeddings.
- `docs/KAGGLE_EMBEDDING_RUNBOOK.md` for private Kaggle extraction.
- `scripts/preflight.py` before any environment change.
- `scripts/audit_data.py` for the non-destructive image inventory.
- `scripts/extract_gallery.py` for permission-gated private embeddings.
