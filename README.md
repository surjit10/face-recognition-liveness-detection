# FaceGuard AI

FaceGuard AI is a real-time computer vision system combining presentation attack detection (anti-spoofing) with open-set face recognition. The application features a desktop interface for live camera verification, guided enrollment, user management, and local attendance tracking.

> **Note**: Designed as an academic prototype for supervised attendance verification. Not intended for unsupervised high-security access control.

## Quick start

Python 3.12 and a graphical Linux desktop are the verified local environment.

```bash
# 1. Set up Python virtual environment
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-runtime.lock

# 2. Download pre-trained models (164 MB)
curl -L -O https://github.com/surjit10/face-recognition-liveness-detection/releases/download/v0.1.0/faceguard-models.zip
unzip faceguard-models.zip

# 3. Launch the desktop application
.venv/bin/python scripts/run_app.py
```

The pre-trained model package contains:
| Model Component | Architecture / Purpose | Location |
|---|---|---|
| **Face Detector** | SCRFD 2.5G (`det_2.5g.onnx`, 3.3 MB) | `models/private/insightface/models/buffalo_m/` |
| **Face Recognizer** | ArcFace ResNet-50 (`w600k_r50.onnx`, 174 MB) | `models/private/insightface/models/buffalo_m/` |
| **Anti-Spoofing (PAD)** | MobileNetV3 (`mobilenetv3_pad.onnx`, 6.1 MB) | `models/private/pad_candidate/` |
| **Identity Gallery** | 1,388 quality-gated 512-d unit vectors, 186 identities (`gallery.npz`, 2.7 MB) | `models/private/gallery/` |


The application performs a camera-free integrity check before enabling camera access.
Camera `0` is opened only after an operator chooses to start verification or enrollment.

## Application screens

- **Verify & attendance** — live preview, quality guidance, liveness progress,
  fail-closed identity decision, and automatic attendance after a successful
  verification. The camera stops automatically when the attendance result is saved.
- **Enroll person** — consent acknowledgement, five quality-checked webcam samples,
  duplicate-identity warning, metadata preview, explicit confirmation, and atomic
  gallery persistence. Raw face images are never saved.
- **Enrolled users** — searchable identity list, enrollment metadata inspection,
  confirmed deletion, and active-runtime gallery refresh.
- **Attendance register** — successful attendance grouped by month and day, month
  filtering, CSV export, and confirmed reset. The SQLite database contains identity
  references and event metadata only; each identity is marked at most once per local day.
- **Settings** — clean readiness, privacy, operator-access, gallery totals, presentation
  controls, and a safe demo-session reset. Internal release/model identifiers are not
  shown in the operator interface.

Navigation stops the active camera, clears verification state, and prevents a stale
verified identity from carrying into another screen.

### Optional operator PIN

Enrollment, user management, private history, and settings can be protected by a local
session PIN. Configure the Git-ignored PBKDF2 credential without exposing the PIN on
the command line:

```bash
.venv/bin/python scripts/configure_operator.py
```

Only a salted PBKDF2-SHA256 digest is saved with file mode `0600`. With no private
operator file, the interface clearly reports **Supervised operator
mode** and remains usable for an attended demonstration. With a configured PIN, the
protected workspace starts locked and can be relocked from the sidebar or Settings.

Keyboard shortcuts `Ctrl+1` through `Ctrl+5` switch screens. `F11` enters presentation
mode and `Esc` returns to the windowed layout.

## Privacy boundary

- Verification frames, face crops, and query embeddings exist in memory only.
- Enrollment persists ArcFace vectors in the protected gallery only after recorded
  consent and explicit confirmation; it does not persist webcam frames.
- Attendance history is stored under `reports/private/checkins.sqlite3`, which is ignored
  by Git. Unknown and rejected liveness outcomes cannot create attendance records.
- Private galleries, metadata, model weights, histories, datasets, credentials, and
  exported biometric artifacts remain excluded from the public repository.

See [`docs/PRIVACY.md`](docs/PRIVACY.md) for the complete data-handling contract.

## Verification and tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_private_models.py
.venv/bin/python scripts/scan_artifacts.py
```

The Docker packaging remains available for camera-free readiness and the legacy demo
runner; the submission UI is launched locally with `scripts/run_app.py`.

## System Limitations & Future Work

- **Open-Set Identification**: On the held-out test split, 348 of 351 enrolled-person probes (99.15%) are recognised with no misidentifications, and 0 of 274 stranger probes are accepted (one-sided 95% upper bound 1.09%, from 24 people). The images are same-session (most people were photographed within one minute), so this is not a cross-session result. These numbers follow a data-quality fix: the offline embedding step had accepted small background bystanders as subjects, so the runtime quality gate is now applied offline too. The fix is disclosed with the superseded results in [`docs/EVALUATION.md`](docs/EVALUATION.md).
- **Anti-Spoofing Characteristics**: The MobileNetV3 PAD model accepts 20.8% of attack images at frame level on the CelebA-Spoof test split (validation: 0.74%), and live testing covered a single phone-screen trial. Printed photos, other screens, masks and replays are unevaluated. Unsupervised use would need active challenge-response (blink, head turn) or IR/depth sensing.
- **Liveness/identity binding**: Every frame of a verification track must show the same face (ArcFace cosine ≥ 0.5 with the track's first frame), so liveness evidence from one face cannot be combined with another face's identity.
- **Compute Distribution**: ArcFace accounts for ~68% of raw compute (81.4 ms on CPU) and runs every frame for the identity-continuity check. The pipeline meets its 5 FPS academic-demo target (6.52 FPS, 285 ms p95 latency on an i5-10300H). Next steps would be a smaller detector input, INT8 quantisation, and running the continuity check every second frame.

These limitations are detailed in the
[`Prototype Model Card`](docs/PROTOTYPE_MODEL_CARD.md) and
[`Evaluation Report`](docs/EVALUATION.md).

## Documentation

- [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md) — installation, screens, supervised
  operation, authorized demo recording, and troubleshooting.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — UI/runtime/storage architecture and
  frozen verification flow.
- [`docs/PRIVACY.md`](docs/PRIVACY.md) — consent, retention, deletion, and private paths.
- [`docs/SUBMISSION_GUIDE.md`](docs/SUBMISSION_GUIDE.md) — professional repository map,
  submission-safe file boundaries and final ZIP creation.

## Build a submission-safe archive

Do not submit the complete working directory because it contains local environments,
private models/data and local experiment storage. Build the allowlisted archive instead:

```bash
.venv/bin/python scripts/scan_artifacts.py
.venv/bin/python scripts/build_submission.py
```

The resulting `dist/faceguard-academic-submission.zip` contains source, tests,
configuration, documentation and privacy-safe aggregate evidence plus a SHA-256 file
manifest. Private biometric/model paths, local environments, MLflow storage and
`reference/archive/` are excluded automatically.
