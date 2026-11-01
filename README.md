# FaceGuard AI 0.1.0

FaceGuard AI is a local, consent-based academic prototype combining RGB
presentation-attack detection (PAD) with open-set face recognition. The application
supports supervised verification, guided webcam enrollment, enrolled-user management,
and privacy-limited local attendance history.

It is **not** a production-certified biometric security system. Do not use it for
physical access control, surveillance, policing, employment, healthcare, credit, or
other consequential decisions.

## Quick start

Python 3.12 and a graphical Linux desktop are the verified local environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-runtime.lock
.venv/bin/python -m pip check
```

Install the authorized private artifacts at these exact Git-ignored paths:

| Artifact | Required path |
|---|---|
| SCRFD detector | `models/private/insightface/models/buffalo_m/det_2.5g.onnx` |
| ArcFace recognizer | `models/private/insightface/models/buffalo_m/w600k_r50.onnx` |
| PAD candidate | `models/private/pad_candidate/mobilenetv3_pad.onnx` |
| Protected gallery | `models/private/gallery/gallery.npz` |

The official `buffalo_m` archive can be installed with the permission-gated command in
[`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md). The runtime never downloads or
substitutes model files, and startup refuses files whose hashes differ from
[`configs/runtime.yaml`](configs/runtime.yaml).

Launch the complete application with one command:

```bash
.venv/bin/python scripts/run_app.py
```

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

## Current limitations

- Recognition evaluation is exploratory because session and device metadata are
  unavailable. The held-out unknown-person false-positive identification rate was
  5/280 searches (approximately 1.79%).
- The PAD candidate's recorded test APCER is approximately 20.84%. The official
  CelebA-Spoof non-commercial terms and label mapping are documented, but the used
  third-party Kaggle mirror reports an unknown license and the exact split CSVs were
  not exported for independent verification. By explicit academic-scope approval,
  reproducing this third-party online training run is deferred: the existing model is
  an opaque frozen development baseline, and this project makes no training-
  reproducibility, production, redistribution or certification claim.
- One target-camera phone-display rejection and one full-path benchmark with a face in
  view passed; broader attack, device, lighting and hardware validation and model
  retraining remain deferred.
- The supplied InsightFace model terms restrict the present prototype to applicable
  non-commercial research use.

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
