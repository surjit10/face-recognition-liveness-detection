# Data Zones

All face images, video and derived biometric data are restricted. This directory is excluded from public Git except for this documentation and approved non-sensitive manifests.

```text
data/
├── manifests/   # checksums, pseudonymous IDs, splits and provenance
├── raw/         # authorized original inputs; Git-ignored
├── interim/     # temporary preparation/cache; Git-ignored
└── processed/   # derived crops/features; Git-ignored
```

## Original restricted dataset

- Location: `data/processed/original/`
- Classification: restricted biometric, original recognition photographs
- Discovered files: 2,804
- Discovered candidate folders: 234
- Known structural facts: folders `000` and `0183` contain 10 images each; the other 232 folders contain 12 images each.

These structural facts do not verify consent, identity semantics, session independence, face correctness or dataset suitability. Training/evaluation must consume frozen manifests rather than recursively discovering files.

## Context-preserving v2 derivative

- Dataset: `data/processed/frad_context_crops_v2/`
- Upload archive: `data/processed/frad_context_crops_v2.zip`
- Preprocessing manifest: `data/manifests/frad_context_crops_v2.csv`
- Preprocessing manifest SHA-256: `6e174d676e8fd322db094f9115530f1b660cc8c57bc7f12c585fe30a8673dd6c`
- Preprocessing report: `reports/audits/frad_context_crops_v2.json`
- Structural inventory: `data/manifests/frad_context_inventory.csv`
- Structural audit: `reports/audits/frad_context_audit.json`
- Decode result: 2,804/2,804 succeeded
- Preprocessing outcomes: 2,515 exactly-one-face context crops and 289 full-frame multiple-face fallbacks
- Structural audit: one exact duplicate excluded; all other rows remain `UNRESOLVED`

This remains a draft dataset. The YuNet preprocessing outcome is not a substitute for the frozen SCRFD single-face gate. It is not frozen and does not yet contain approved identities, verified sessions or split assignments.

## Context-preserving preprocessing

When authorized original recognition photographs are available, generate a separate derived dataset without modifying the originals:

```bash
.venv/bin/python scripts/prepare_recognition_context.py \
  --input-root data/processed/original \
  --output-root data/processed/frad_context_crops_v2 \
  --manifest data/manifests/frad_context_crops_v2.csv \
  --report reports/audits/frad_context_crops_v2.json \
  --detector-model reference/archive/legacy_crop_tool/models/face_detection_yunet_2023mar.onnx \
  --workers 1
```

This stage uses YuNet only to propose a generous context crop. It keeps exactly-one-face proposals at 2× the detected face-box size and retains no-face or multi-face inputs as full-frame fallbacks. It never treats this preprocessing result as proof of a valid face: the frozen SCRFD/ArcFace extraction gate still decides whether an embedding is acceptable. The single worker and single OpenCV thread keep laptop CPU use bounded.
