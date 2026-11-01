# MLOps Architecture & Tooling

This document outlines the Machine Learning Operations (MLOps) architecture, tracking pipelines, security guardrails, and deployment packaging implemented in FaceGuard AI.

## Implemented Controls & Components

| Component | Implementation | Engineering Purpose |
|---|---|---|
| **Runtime Pinning** | `requirements-runtime.lock` | Fully pinned CPU runtime ensuring deterministic dependency resolution across machines. |
| **Tooling Isolation** | `requirements-mlops.lock`, `.venv-mlops/` | DVC and MLflow run in an isolated environment so their dependency trees do not pollute runtime inference. |
| **Automated Testing** | `tests/`, `.github/workflows/ci.yml` | 158 automated unit, contract, integration, and privacy tests run automatically on every push. |
| **Continuous Integration** | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Verifies clean installation, dependency compatibility (`pip check`), test suite execution, and security scans. |
| **Biometric & Secret Scanner** | [`scripts/scan_artifacts.py`](../scripts/scan_artifacts.py) | Scans all tracked and staged files to prevent accidental leakage of raw faces, embeddings, model weights, or API keys. |
| **Pipeline Versioning** | [`dvc.yaml`](../dvc.yaml), `dvc.lock` | Reproducible stages (`readiness` and `scan`) with cryptographic model hashing; avoids pushing large weights to Git. |
| **Experiment & Metric Tracking** | [`scripts/log_mlflow.py`](../scripts/log_mlflow.py), local `mlflow.db` | Logs aggregate latency, FPS, memory, accuracy, and JSON evaluation reports to an interactive MLflow dashboard. |
| **Containerization** | [`Dockerfile`](../Dockerfile), `.dockerignore` | Minimal CPU runtime container image; pre-trained weights are mounted read-only at runtime rather than baked into layers. |
| **Startup Integrity Checks** | `scripts/run_demo.py --check-only` | Verifies SHA-256 cryptographic fingerprints of models and gallery before the camera is ever engaged. |
| **Runtime Benchmarking** | `scripts/benchmark_runtime.py` | Profiles real-world p95 latency, FPS, memory footprint, and frame queue drops on physical hardware. |
| **Visual Dashboard** | [`reports/mlops_dashboard.html`](../reports/mlops_dashboard.html) | Interactive web UI displaying the 6-stage pipeline, performance cockpits, confusion matrices, and audit receipts. |
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

`dvc repro` executes the hash-verified readiness check and artifact scan. The pipeline tracks model hashes in `models/private` locally without needing a cloud bucket.

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

## Architecture & Design Rationale

- **Isolated `.venv-mlops`**: Keeps heavy experiment tracking tools (MLflow, DVC, SQLAlchemy) separate from the runtime camera application, ensuring that development dependencies never interfere with live inference.
- **Local DVC Tracking**: Model hashes and dependency checks run locally without uploading biometric weights to remote storage.
- **Privacy-by-Design**: MLflow and DVC record aggregate metrics, hashes, and timings only. Raw frames, crops, face embeddings, and identity databases are never passed to tracking tools.
- **Read-Only Model Mounting**: Mounting model files at container launch keeps proprietary weights out of public image registries and reduces image build times.

