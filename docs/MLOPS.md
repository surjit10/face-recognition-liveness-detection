# MLOps Status

This is an interview-minimum MLOps layer around a frozen academic prototype. It is
deliberately **not** a production platform: release promotion is paused because PAD
provenance is unresolved.

## Implemented

| Control | Location | Purpose |
|---|---|---|
| Frozen runtime | `requirements-runtime.lock` | Reproducible CPU runtime; `pip check` must pass. |
| Isolated MLOps tooling | `requirements-mlops.in`, `requirements-mlops.lock`, `.venv-mlops/` | DVC + MLflow live in a separate environment so they cannot disturb the runtime. |
| Automated tests | `tests/`, `.github/workflows/ci.yml` | Unit, contract and integration tests run on every push and pull request. |
| Continuous integration | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Installs the runtime lock, runs `pip check`, the test suite and the artifact scan. |
| Artifact/secret scan | [`scripts/scan_artifacts.py`](../scripts/scan_artifacts.py) | Checks tracked and non-ignored untracked files; fails on biometric/model artifacts, private paths, `.env`, credential files or credential-shaped assignments. |
| DVC pipeline | [`dvc.yaml`](../dvc.yaml), `dvc.lock` | Reproducible `readiness` and `scan` stages with private model hashing; no biometric data is copied to a remote. |
| MLflow tracking | [`scripts/log_mlflow.py`](../scripts/log_mlflow.py), local `mlflow.db` | Logs aggregate metrics, params and JSON reports; no frames, embeddings or per-identity scores. |
| Container | [`Dockerfile`](../Dockerfile), `.dockerignore` | Minimal CPU runtime image; private models are mounted at run time, never baked in. |
| Startup integrity | `scripts/run_demo.py --check-only` | Verifies face/PAD/gallery SHA-256 hashes and CPU providers without opening a camera. |
| Aggregate benchmark | `scripts/benchmark_runtime.py` | Writes a privacy-safe performance receipt with no frames or embeddings. |
| Decision register | `configs/decisions.yaml` | Records frozen scope, thresholds, hashes and unresolved permissions. |
| Privacy controls | `.gitignore`, `docs/PRIVACY.md`, scanner | Keeps private weights, crops, galleries and score rows out of Git and reports. |
| Release metadata | `scripts/generate_release_manifest.py`, `reports/release_manifest.json` | Records the actual commit/worktree state without converting a dirty prototype into an approved release. |
| Gallery lifecycle | `src/faceguard/gallery_admin.py`, `scripts/manage_gallery.py` | Atomic enrollment/deletion, metadata/sidecar synchronization, pinned-hash update, verification and rollback. |
| Local event retention | `src/faceguard/checkins.py` | Private SQLite verified-event history with deduplication, CSV export and explicit reset. |

## Reproduce locally

### 1. Runtime checks

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python scripts/scan_artifacts.py
```

### 2. DVC pipeline

The MLOps tools live in `.venv-mlops`, never in the runtime `.venv`:

```bash
.venv/bin/python -m venv .venv-mlops
.venv-mlops/bin/python -m pip install -r requirements-mlops.lock
.venv-mlops/bin/dvc repro
.venv-mlops/bin/dvc status
```

`dvc repro` re-runs the hash-verified readiness check and the artifact scan. The
pipeline depends on `models/private` for hash awareness but never copies it to a DVC
cache or remote. No DVC remote is configured; versions are local-only until an
authorized private remote exists.

### 3. MLflow tracking

```bash
.venv-mlops/bin/python scripts/log_mlflow.py
.venv-mlops/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The run logs:

- runtime release ID, provider, model/gallery hashes and thresholds;
- validation and test recognition metrics;
- benchmark latency, FPS, memory and frame counts;
- aggregate readiness, evaluation and benchmark JSON files.

MLflow uses a local SQLite backend and a local artifact store under `mlruns/`. Both are
Git-ignored. It never logs images, crops, embeddings, person names or private score
rows.

### 4. Container

```bash
docker build -t faceguard-runtime:prototype .
docker run --rm faceguard-runtime:prototype --help
docker run --rm \
  -v "$PWD/models/private:/app/models/private:ro" \
  faceguard-runtime:prototype
```

The default command is `--check-only`; models are mounted read-only, not included in
the image. For a camera demo, pass `--device /dev/video0`, the X11 socket and
`DISPLAY`, and add `--camera 0`. The image does not contain weights or biometric data.

## Remaining limits

| Capability | Status | Reason |
|---|---|---|
| Release approval | Not granted | Metadata generation and academic-scope eligibility checks are implemented, but this checkout is not represented as a clean or human-approved release. |
| Private CI execution | Workflow implemented, runner execution not evidenced here | Private weights cannot be given to GitHub-hosted runners; the manual workflow requires an authorized self-hosted runner. |
| Camera acceptance | Deferred | Five target-camera cases and a full-path benchmark with a face in view remain outside this product-finalization task. |
| Full release artifact audit | Partial | The scanner covers tracked/staged files, not a release-manifest audit. |

## CI limits

The hosted workflow does **not** run `--check-only`, DVC against private models, or any
camera test. Those require the Git-ignored face pack, PAD model and gallery. Readiness
remains a local/private-runner step recorded in
`reports/environments/runtime_readiness.json`.

The scanner reviews tracked, staged and non-ignored untracked paths. It remains a
guardrail rather than a guarantee; human review is still required before publishing.

## Interview talking points

- Why isolated `.venv-mlops`: avoids contaminating the pinned runtime with DVC/MLflow
  dependency graphs.
- Why local DVC without a remote: biometric artifacts must not leave the machine
  without explicit authorization.
- Why MLflow logs evaluation rather than training: `PROTOTYPE_SCOPE` freezes further
  training; the tracked evidence is the frozen evaluation and benchmark.
- Why a manifest is not an approval: it records hashes and dirty/clean worktree state;
  unresolved PAD provenance and human sign-off remain separate gates.
- Why models are mounted into the container: keeps weights out of image layers and
  registry storage.
