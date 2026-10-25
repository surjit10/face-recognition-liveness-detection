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

Expected: exit code `0`, `"event": "runtime_ready"`, `"camera_opened": false`, every
face provider is `CPUExecutionProvider`, `"gallery_vectors": 1430`,
`"gallery_identities": 187`, and the printed face/PAD hashes match
`configs/runtime.yaml`. Any mismatch or hash error exits `2`; do not continue to the
camera until it passes.

## 3. Live camera

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
- `SPOOF` means liveness failed; identity matching is not run.
- Keep other people and phones/printed photos out of the frame for the single-user
  demonstration. Any second face produces `MULTIPLE FACES`.

## 4. Overlay states

| Overlay | Operator action |
|---|---|
| `NO FACE` | Present one face to the camera. |
| `MULTIPLE FACES` | Remove additional faces from view. |
| `HOLD STILL` | Steady the face, improve lighting, move closer, avoid motion blur. |
| `CHECKING LIVENESS` | Hold still; evidence is being collected. |
| `SPOOF` | End the session and investigate the presentation; do not repeat until it passes. |
| `UNKNOWN` | Record the fail-closed result; do not claim recognition. |
| `VERIFIED: <ID>` | PAD passed and the pseudonymous label was accepted. |

The identity is shown only after the temporal PAD decision passes.

## 5. Stopping and shutdown

| Input | Behavior |
|---|---|
| `q` or `Esc` in the preview window | Stops capture, releases the camera, closes OpenCV windows and the metrics dashboard, prints `ENDED`, exit code `0`. |
| Close the metrics dashboard window | Same clean shutdown as `q`/`Esc`; exit code `0`. |
| `Ctrl+C` | Same cleanup, prints `interrupted` and `ENDED`, exit code `130`. The dashboard process ignores the terminal signal and is closed by the parent. |
| Camera cannot open | Clear error on stderr, exit code `2`, no window left open. |
| Camera read fails repeatedly during a run | Stops capture, prints `runtime blocked: camera read failed`, exit code `2`. |

A clean run ends with `capture_summary` (`camera_released: true`) and one `ENDED`
event. No camera frames are recorded by the application.

## 6. Optional benchmark

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

## 7. Troubleshooting

| Symptom | Check |
|---|---|
| `runtime startup blocked` | Readiness check, artifact paths and `configs/runtime.yaml` hashes. |
| `unable to open video source` | Camera `0` is free and authorized; close other applications using it. |
| Persistent `HOLD STILL` | Lighting, face size, blur and camera continuity. |
| `VERIFIED` for the wrong person | Stop; record the failure. Do not tune thresholds on the acceptance cases. |

Phase receipts and the benchmark are aggregate-only. Never attach frames, crops,
embeddings or screen recordings containing faces to reports or issue trackers.
