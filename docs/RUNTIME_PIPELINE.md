# Runtime pipeline

## Current status

The maintainable runtime code is implemented and tested, and the PAD candidate is
installed privately. Production execution remains intentionally blocked until:

1. the official `buffalo_m` detector and recognizer files are supplied locally;
2. folder-to-person semantics and an enrollment/validation/test protocol are verified;
3. a gallery is built from the frozen enrollment manifest;
4. the recognition threshold and top-1/top-2 margin are calibrated on validation
   searches and inserted into `configs/runtime.yaml`.

The existing `frad_embeddings.npz` is an audit artifact. It must not be silently
promoted into a production gallery or used to invent a recognition threshold.

## Architecture

```mermaid
flowchart LR
    A[Camera or sequential MP4] --> B[SCRFD: exactly one face]
    B --> C[Quality and IoU continuity]
    C --> D[1.3x RGB PAD crop]
    D --> E[MobileNetV3 ONNX]
    E --> F[Temporal median]
    F -->|below threshold| G[Spoof]
    F -->|passes| H[ArcFace embedding]
    H --> I[Protected gallery search]
    I -->|threshold and margin pass| J[Verified pseudonymous ID]
    I -->|otherwise| K[Unknown]
```

No frame or embedding is included in the structured result event. Live camera mode
uses a bounded latest-frame queue and drops the oldest waiting frame under load.
Sequential video mode processes every frame in order for accuracy evaluation.

## Source layout

| Path | Responsibility |
|---|---|
| `src/faceguard/gallery.py` | Gallery validation, enrollment, deletion and open-set matching |
| `src/faceguard/pad.py` | Exact PAD crop/preprocessing and hash-verified ONNX inference |
| `src/faceguard/pipeline.py` | Temporal PAD-before-recognition state machine |
| `src/faceguard/runtime.py` | Bounded latest-frame queue |
| `src/faceguard/runtime_config.py` | Strict, no-unknown-key runtime configuration |
| `src/faceguard/recognition_metrics.py` | Validation-only threshold selection |
| `scripts/run_demo.py` | Thin video/webcam entry point |

Private weights and galleries stay under `models/private/` and are ignored by Git.

## Reproducible setup sequence

### 1. Promote the trained PAD bundle

```bash
.venv/bin/python scripts/prepare_runtime_artifacts.py \
  --pad-bundle data/processed/pad_model_bundle.zip \
  --output-dir models/private/pad_candidate
```

The command checks the ZIP CRC and the internal ONNX, checkpoint and configuration
hashes before using atomic writes with mode `0600`.

### 2. Install the official local InsightFace pack

Download/retrieve the official `buffalo_m` ZIP under its applicable terms, then run:

```bash
.venv/bin/python scripts/install_insightface_pack.py \
  --pack-zip /private/path/buffalo_m.zip \
  --model-root models/private/insightface \
  --acknowledge-non-commercial-model-terms
```

The installer never downloads weights. It promotes only the detector and recognizer
after matching the hashes frozen in `configs/decisions.yaml`.

### 3. Freeze recognition manifests

Create explicit enrollment, validation-known, validation-impostor, test-known and
test-impostor CSVs. A known probe must not reuse an enrollment file hash. Impostor
identities must not occur in the gallery. Do not infer sessions from filenames.

The enrollment CSV consumed by `build_gallery.py` requires:

```text
relative_path,verified_person_id,source_sha256
```

### 4. Build the private gallery

Validate the five populated manifests first:

```bash
.venv/bin/python scripts/validate_recognition_manifests.py \
  --dataset-root data/processed/frad_legacy_crops \
  --enrollment data/manifests/recognition_enrollment.csv \
  --val-known data/manifests/recognition_val_known.csv \
  --val-impostor data/manifests/recognition_val_impostor.csv \
  --test-known data/manifests/recognition_test_known.csv \
  --test-impostor data/manifests/recognition_test_impostor.csv \
  --allow-unknown-session-metadata \
  --output reports/audits/recognition_split_validation.json
```

Use `--allow-unknown-session-metadata` only after recording that limitation in
`FRAD_SESSION_POLICY`; it is not equivalent to verified cross-session evaluation.

```bash
.venv/bin/python scripts/build_gallery.py \
  --embedding-archive models/private/frad_candidate_embeddings/frad_embeddings.npz \
  --enrollment-manifest data/manifests/recognition_enrollment.csv \
  --output models/private/gallery/gallery.npz
```

Copy the printed `gallery_sha256` into `configs/runtime.yaml` only after reviewing
the enrollment manifest and report.

### 5. Calibrate recognition

Generate one row per validation gallery search:

```bash
.venv/bin/python scripts/evaluate_recognition.py \
  --embedding-archive models/private/frad_candidate_embeddings/frad_embeddings.npz \
  --gallery models/private/gallery/gallery.npz \
  --gallery-report models/private/gallery/gallery.json \
  --known-manifest data/manifests/recognition_val_known.csv \
  --impostor-manifest data/manifests/recognition_val_impostor.csv \
  --output reports/private/recognition_validation_scores.csv
```

Then select the threshold using a declared validation policy:

```bash
.venv/bin/python scripts/calibrate_recognition.py \
  --validation-scores reports/private/recognition_validation_scores.csv \
  --output reports/evaluation/recognition_threshold.json \
  --policy recognition_fmr_bound \
  --max-fmr 0.001 \
  --min-margin <VALIDATION_SELECTED_MARGIN>
```

The FMR target and margin are project decisions, not defaults. Put the resulting
threshold and reviewed margin into `configs/runtime.yaml`. Never tune them on test.

### 6. Run sequential video before webcam

```bash
.venv/bin/python scripts/run_demo.py \
  --config configs/runtime.yaml \
  --video /private/path/evaluation.mp4 \
  --no-display
```

After spoof, unknown, enrolled, no-face, multi-face and interruption MP4 cases pass:

```bash
.venv/bin/python scripts/run_demo.py --config configs/runtime.yaml --camera 0
```

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/python scripts/preflight.py \
  --output reports/environments/environment_local_runtime.json
```

The current PAD candidate has a test APCER of approximately 20.84%. It is suitable
for pipeline integration and further evaluation, not a production security claim.
