# Demo Runbook

Operator procedure for the local FaceGuard AI prototype. Camera `0` is the only
authorized source. The system is a fail-closed academic demo, not an access-control
product.

## 1. Environment

Use Python 3.12 and the frozen runtime lock:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-runtime.lock
.venv/bin/python -m pip check
```

The face pack is not downloaded automatically. Install the supplied official
`buffalo_m` package after reading its terms:

```bash
.venv/bin/python scripts/install_insightface_pack.py \
  --pack-zip data/processed/buffalo_m.zip \
  --model-root models/private/insightface \
  --acknowledge-non-commercial-model-terms
```

Each operator must acknowledge the supplied InsightFace-model terms. Do not substitute
another pack or disable hash checks. The PAD candidate and gallery must already exist
under `models/private/`. They are Git-ignored and are never recreated by the runtime.

## 2. Readiness check (no camera)

```bash
.venv/bin/python scripts/run_demo.py \
  --config configs/runtime.yaml \
  --check-only \
  --readiness-report reports/environments/runtime_readiness.json
```

Expected for the supplied baseline: exit code `0`, `"event": "runtime_ready"`,
`"camera_opened": false`, every face provider is `CPUExecutionProvider`,
`"gallery_vectors": 1430`, `"gallery_identities": 187`, and the printed face/PAD hashes match
`configs/runtime.yaml`. Any mismatch or hash error exits `2`; do not continue to the
camera until it passes.

## 3. Centralized desktop application

The recommended operator entry point is the single-window local desktop application:

```bash
.venv/bin/python scripts/run_app.py
```

The window automatically performs the camera-free readiness check. **Start camera**
remains disabled until the configured face models, PAD model, gallery, hashes and CPU
providers pass. After the check succeeds, press **Start camera** to open authorized
camera `0` and run the complete pipeline. The same window displays the live preview,
fail-closed result, PAD progress, FPS, latency, frame age and drop counts.

Use the persistent left navigation for the five application screens:

1. **Verify & attendance** runs the unchanged PAD-before-recognition pipeline. After a
   verified result, attendance is marked automatically once for that identity on the
   local calendar day. It is never marked for unknown, retry, or liveness-rejected
   results. After attendance is saved, the camera stops and the completion result
   remains visible until the next person is started.
2. **Enroll person** requires a non-sensitive identity label and explicit consent.
   Start camera `0`, capture five accepted samples while varying pose slightly, review
   any duplicate warning, then confirm. The app stores vectors and private metadata,
   never frames or crops.
3. **Enrolled users** searches canonical IDs and local display aliases. Inspect shows
   vector counts and available metadata. Delete requires confirmation, updates the
   configured gallery hash, and reloads the runtime.
4. **Attendance register** groups verified events by month and day, filters by month,
   exports permitted fields to CSV, and deletes all events only after confirmation.
5. **Settings** shows readiness, privacy/operator state, provider and gallery totals.
   It also provides operator locking and a safe demo reset that clears unfinished
   enrollment and local attendance without deleting enrolled identities.

Changing screens stops the active camera and clears the displayed verification result.
Only one camera loop and one initialized model runtime are used by the application.
Use `Ctrl+1` through `Ctrl+5` for navigation. `F11` enables a clean full-screen
presentation view and `Esc` exits it.

For an unattended or shared-machine demonstration, configure the optional Git-ignored
operator credential:

```bash
.venv/bin/python scripts/configure_operator.py
```

The plaintext PIN is never stored; the private file contains a salted PBKDF2-SHA256
digest and is written with mode `0600`. Without this private file, the application remains
in clearly labelled supervised operator mode.

Optional human-readable names are loaded only from the Git-ignored local file
`configs/private/identity_labels.yaml`. They affect the desktop banner only. Gallery
matching, structured events and reports continue to use the canonical pseudonymous ID.

Press **Stop** or close the window to clear the displayed identity, stop both worker
threads and release the camera. The app does not provide camera recording, screenshots,
single-image verification, threshold editing or model-training controls.

## 4. Legacy CLI camera interface

```bash
.venv/bin/python scripts/run_demo.py --config configs/runtime.yaml --camera 0
```

For a separate dark metrics dashboard, add `--metrics-ui`:

```bash
.venv/bin/python scripts/run_demo.py --config configs/runtime.yaml --camera 0 --metrics-ui
```

The dashboard runs in an isolated local process and shows processed FPS, median and p95
frame age, pipeline median latency, processed/captured and dropped frame counts, PAD
observations, elapsed time and the current fail-closed state. It receives aggregate
metrics only; no frames or embeddings cross the process boundary. Closing the
dashboard stops the session cleanly. Use `--no-display --metrics-ui` to run the
dashboard without the OpenCV preview.

Presentation:

- Sit front-facing, well lit, with one face at least `80 px` wide.
- Hold still while the overlay shows `CHECKING LIVENESS`; eight PAD observations are
  collected and aggregated by median.
- `UNKNOWN` is a valid fail-closed outcome, not a crash. It means PAD passed but the
  gallery did not accept the face. Do not tune thresholds to force a match.
- `POSSIBLE SPOOF / REJECTED` means the PAD median did not pass. This may be an attack or
  model uncertainty; identity matching is not run.
- Keep other people and phones/printed photos out of the frame for the single-user
  demonstration. Any second face produces `MULTIPLE FACES`.

## 5. Interface states

| Overlay | Operator action |
|---|---|
| `LOOKING FOR FACE` | Present one face to the camera. |
| `MULTIPLE FACES` | Numbered boxes point to every detected face; identities remain unchecked until only one face is visible. |
| `MOVE CLOSER` / `IMPROVE LIGHTING` / `HOLD STILL` | Follow the specific quality correction without changing thresholds. |
| `CHECKING LIVENESS` | Hold still; evidence is being collected. |
| `POSSIBLE SPOOF / REJECTED` | Liveness did not pass. This can be an attack or model uncertainty; do not lower the threshold. |
| `UNKNOWN` | Record the fail-closed result; do not claim recognition. |
| `VERIFIED: <ID>` | PAD passed and the pseudonymous label was accepted. |

The identity is shown only after the temporal PAD decision passes.

## 6. Stopping and shutdown

| Input | Behavior |
|---|---|
| `q` or `Esc` in the preview window | Stops capture, releases the camera, closes OpenCV windows and the metrics dashboard, prints `ENDED`, exit code `0`. |
| Close the metrics dashboard window | Same clean shutdown as `q`/`Esc`; exit code `0`. |
| `Ctrl+C` | Same cleanup, prints `interrupted` and `ENDED`, exit code `130`. The dashboard process ignores the terminal signal and is closed by the parent. |
| Camera cannot open | Clear error on stderr, exit code `2`, no window left open. |
| Camera read fails repeatedly during a run | Stops capture, prints `runtime blocked: camera read failed`, exit code `2`. |
| **Stop** or desktop window close | Stops workers, releases camera `0`, clears the preview and identity. |

A clean run ends with `capture_summary` (`camera_released: true`) and one `ENDED`
event. No camera frames are recorded by the application.

## 7. Optional benchmark

```bash
.venv/bin/python scripts/benchmark_runtime.py \
  --config configs/runtime.yaml \
  --camera 0 \
  --seconds 15 \
  --output reports/benchmarks/prototype_camera.json
```

The receipt contains aggregate hardware and latency values only. It is not a general
performance guarantee. Run it while the intended operator is in frame so PAD and
recognition stages are exercised; an empty scene measures capture and detection only.
The existing receipt in `reports/benchmarks/prototype_camera.json` was captured on a
scene with no face and therefore reports only `no_face` retries.

## 8. Recording an authorized submission demo

The application has no recording feature. If a submission requires video, use an
operating-system screen recorder only after all visible participants have explicitly
authorized that recording and its intended audience.

1. Use a non-sensitive local display alias and hide unrelated notifications/windows.
2. Record only the minimum verification/enrollment flow needed for review.
3. Do not display private files, model weights, embeddings, credentials, or another
   person's attendance history.
4. Keep the recording outside Git and public issue trackers. Copy it only to the
   authorized submission location.
5. Delete the local recording after the approved review and retention period.

Recording is external to FaceGuard; the application's no-frame-retention promise does
not apply to the screen recorder's output.

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| `runtime startup blocked` | Readiness check, artifact paths and `configs/runtime.yaml` hashes. |
| `unable to open video source` | Camera `0` is free and authorized; close other applications using it. |
| Persistent `HOLD STILL` | Lighting, face size, blur and camera continuity. |
| `VERIFIED` for the wrong person | Stop; record the failure. Do not tune thresholds on the acceptance cases. |

Phase receipts and the benchmark are aggregate-only. Never attach frames, crops,
embeddings or screen recordings containing faces to reports or issue trackers.
