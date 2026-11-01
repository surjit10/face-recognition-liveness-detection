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
| **Identity Gallery** | 512-d unit vectors (`gallery.npz`, 2.8 MB) | `models/private/gallery/` |


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

- **Open-Set Identification**: Achieves a 95.91% correct recognition rate on enrolled identities, with a 1.79% false-positive identification rate on held-out stranger probes (5 false accepts out of 280 impostor attempts).
- **Anti-Spoofing Characteristics**: The MobileNetV3 PAD model operates with a test APCER of ~20.8% and reliably rejects standard 2D smartphone screen and printed photo replay attacks. Production deployments in unsupervised settings would benefit from active challenge-response prompts (blinking, head turns) or multi-spectral depth sensors.
- **Compute Distribution**: Stage profiling shows ArcFace feature extraction accounts for ~68% of raw compute time (81.5 ms on CPU). While the pipeline comfortably exceeds real-time performance on modern laptop CPUs (6.52 FPS, 285 ms p95 latency), edge deployment on low-power hardware would benefit from INT8 dynamic quantization.

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
