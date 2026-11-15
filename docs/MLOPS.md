# MLOps Architecture & Tooling

This document outlines the Machine Learning Operations (MLOps) architecture, tracking pipelines, security guardrails, and deployment packaging implemented in FaceGuard AI.

## Implemented Controls & Components

| Component | Implementation | Engineering Purpose |
|---|---|---|
| **Runtime Pinning** | `requirements-runtime.lock` | Fully pinned CPU runtime ensuring deterministic dependency resolution across machines. |
| **Tooling Isolation** | `requirements-mlops.lock`, `.venv-mlops/` | DVC and MLflow run in an isolated environment so their dependency trees do not pollute runtime inference. |
| **Automated Testing** | `tests/`, `.github/workflows/ci.yml` | 185 automated unit, contract, integration, and privacy tests run automatically on every push. |
| **Continuous Integration** | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Verifies clean installation, dependency compatibility (`pip check`), test suite execution, and security scans. |
| **Biometric & Secret Scanner** | [`scripts/scan_artifacts.py`](../scripts/scan_artifacts.py) | Scans all tracked and staged files to prevent accidental leakage of raw faces, embeddings, model weights, or API keys. |
| **Reproducible Evaluation (DVC)** | [`dvc.yaml`](../dvc.yaml), `dvc.lock`, [`scripts/verify_recognition_metrics.py`](../scripts/verify_recognition_metrics.py) | Three stages: `readiness` (hash-verified model/gallery load), `evaluate` (re-derives the validation threshold, the 612-point validation curve and validation/test metrics from private per-probe scores and fails on any drift from the committed reports or `configs/runtime.yaml`), and `scan`. `dvc.lock` pins the hash of every input, including the git-ignored score files, without publishing them. |
| **Evaluation Tracking** | [`scripts/log_mlflow.py`](../scripts/log_mlflow.py), local `mlflow.db` | Logs each evaluated release (aggregate latency, FPS, memory, recognition error rates, config hashes and the JSON reports) as an MLflow run. Pretrained models are used, so there are no training runs or model registry. |
| **Containerization** | [`Dockerfile`](../Dockerfile), `.dockerignore` | Minimal CPU runtime container image; pre-trained weights are mounted read-only at runtime rather than baked into layers. |
| **Startup Integrity Checks** | `scripts/run_demo.py --check-only` | Verifies SHA-256 cryptographic fingerprints of models and gallery before the camera is ever engaged. |
| **Runtime Benchmarking** | `scripts/benchmark_runtime.py` | Profiles real-world p95 latency, FPS, memory footprint, and frame queue drops on physical hardware. |
| **Visual Dashboard** | [`reports/mlops_dashboard.html`](../reports/mlops_dashboard.html) | Pipeline, benchmark, recognition outcomes and audit receipts. Opened from disk it shows the recorded benchmark; served by the runtime it shows live metrics. Its threshold simulator reads the real validation curve, and `tests/unit/test_dashboard_curve.py` keeps the curve and the recorded numbers identical to their source reports. |
| **Live Runtime Metrics** | [`src/faceguard/live_metrics.py`](../src/faceguard/live_metrics.py), `--metrics-port` | Opt-in loopback HTTP endpoint (`/metrics.json`, Prometheus `/metrics`) fed by the inference loop; drives the dashboard's live mode. Aggregate counters only. |
| **Gallery Lifecycle Management** | `src/faceguard/gallery_admin.py` | Atomic enrollment, duplicate prevention, verification, sidecar synchronization, and rollback. |
| **Local Event Persistence** | `src/faceguard/checkins.py` | Private SQLite attendance database with daily deduplication, CSV export, and clear separation from raw frames. |

## Running MLOps Workflows Locally

### 1. Automated Testing & Secret Scanning

Run the test suite and verify dependency integrity:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python scripts/scan_artifacts.py
```

### 2. DVC Pipeline Execution

DVC operations run within the isolated `.venv-mlops` environment:

```bash
# Set up isolated MLOps environment
.venv/bin/python -m venv .venv-mlops
.venv-mlops/bin/python -m pip install -r requirements-mlops.lock

# Reproduce pipeline stages
.venv-mlops/bin/dvc repro
.venv-mlops/bin/dvc status
```

`dvc repro` runs the hash-verified readiness check, the recognition evaluation check and the artifact scan. The pipeline tracks model hashes in `models/private` locally without needing a cloud bucket.

The `evaluate` stage reads the private per-probe search scores from
`data/interim/recognition_scores/` (git-ignored; one row per probe with top-1 score,
top-1/top-2 margin and a correctness flag, no embeddings). It then:

1. checks that the score files, threshold report and validation/test reports all reference each other by SHA-256;
2. re-runs threshold selection on validation only and checks the threshold, the full 612-point curve and `configs/runtime.yaml`;
3. re-scores validation and test at the frozen threshold and checks every published count and rate, including the zero-event 95% upper bound;
4. writes the aggregate `reports/evaluation/recognition_metrics.json`.

```bash
.venv-mlops/bin/dvc metrics show          # current verified metrics
.venv-mlops/bin/dvc metrics diff HEAD~1   # compare with an earlier commit
```

The committed reports carry hand-written disclosures (quality-gate exclusions,
superseded results, limitations), so this stage verifies them rather than regenerating
them. Without the private scores (for example in public CI) the stage cannot run; its
recorded result is in `dvc.lock`.

### 3. MLflow Tracking & Dashboard

Launch the local MLflow dashboard to visualize historical runs, metrics, and parameters:

```bash
# Log latest metrics to MLflow
.venv-mlops/bin/python scripts/log_mlflow.py

# Launch interactive UI
.venv-mlops/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open **`http://127.0.0.1:5000`** in your browser. The dashboard records:
- Operating parameters, provider (`CPUExecutionProvider`), model hashes, and decision thresholds.
- Validation and test recognition accuracy and error rates.
- Benchmark throughput (FPS), p95 latency, and memory footprint.
- JSON receipts stored safely under local Git-ignored storage.

### 4. Docker Container Deployment

Build and run the containerized runtime:

```bash
# Build lightweight CPU image
docker build -t faceguard-runtime:prototype .

# Run camera-free integrity check inside container
docker run --rm \
  -v "$PWD/models/private:/app/models/private:ro" \
  faceguard-runtime:prototype --check-only
```

For live webcam access inside Docker, pass the video device and display variables:
```bash
docker run --rm \
  --device /dev/video0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -v "$PWD/models/private:/app/models/private:ro" \
  faceguard-runtime:prototype --camera 0
```

### 5. Live Runtime Metrics

Both `run_demo.py` and the desktop app `run_app.py` accept `--metrics-port PORT`
(1024-65535). The runtime then serves, on `127.0.0.1` only:

| Path | Content |
|---|---|
| `/` | the MLOps dashboard, which polls `/metrics.json` every second |
| `/metrics.json` | rolling-window FPS, p50/p95 model latency and capture-to-decision frame age, processed/captured/dropped frames, queue depth, current and peak RSS memory, frame counts per pipeline state and reason code, camera reconnects, 2-minute frame-age history, and the frozen targets |
| `/metrics` | the same counters in Prometheus text format, ready for a Prometheus scraper and Grafana |
| `/healthz` | liveness probe |

```bash
.venv/bin/python scripts/run_demo.py --camera 0 --metrics-port 8765
# open http://127.0.0.1:8765/  or:  curl -s http://127.0.0.1:8765/metrics
```

Design choices:

- **Same origin, no CORS.** The dashboard is served by the metrics server, so the
  browser never needs an `Access-Control-Allow-Origin` header and other websites cannot
  read the feed.
- **DNS-rebinding guard.** Requests whose `Host` header is not a loopback name get 403.
- **Never blocks inference.** The hub takes a short lock per frame; HTTP handling runs
  on its own daemon threads. If the port is busy, the runtime starts without metrics.
- **Same yardstick as the benchmark.** Live values are judged against
  `configs/performance_objective.yaml`, and frame age is measured at the same point as
  in the benchmark.
- **Privacy.** Only timings, counters, state names and fixed reason codes are exposed.
  Person IDs, similarity and PAD scores, track IDs, operator messages, frames and
  embeddings are never recorded (`tests/unit/test_live_metrics.py`).

Opened directly from disk, the dashboard falls back to the recorded 1 November 2026
benchmark and labels every value as recorded.

## Architecture & Design Rationale

- **Isolated `.venv-mlops`**: Keeps heavy experiment tracking tools (MLflow, DVC, SQLAlchemy) separate from the runtime camera application, ensuring that development dependencies never interfere with live inference.
- **Local DVC Tracking**: Model hashes and dependency checks run locally without uploading biometric weights to remote storage.
- **Privacy-by-Design**: MLflow and DVC record aggregate metrics, hashes, and timings only. Raw frames, crops, face embeddings, and identity databases are never passed to tracking tools.
- **Read-Only Model Mounting**: Mounting model files at container launch keeps proprietary weights out of public image registries and reduces image build times.


## Not Implemented (Scope Limits)

This is a single-machine academic prototype. The following are deliberately out of scope:

- **No fleet monitoring or alerting.** The live feed covers one local runtime and keeps
  only a 2-minute window in memory. A deployment would scrape `/metrics` from every site
  into Prometheus, keep history, and alert when p95 frame age, dropped-frame ratio,
  camera reconnects or the `UNKNOWN` / `PAD_FAILED` rates leave their validation-time
  range. Similarity-score drift would need a privacy-reviewed histogram metric, which is
  deliberately not exported today.
- **No training pipeline or model registry.** Detection and recognition use pretrained
  InsightFace models; the PAD model was trained once (see `reports/audits/pad_training_evidence.json`).
- **No remote DVC or MLflow server.** Both stay local so biometric-derived files never
  leave the machine.
