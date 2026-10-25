# FaceGuard final prototype completion reference

## 1. Purpose and frozen scope

This document is the working checklist for converting the current FaceGuard codebase
into a final **local demonstration prototype**. It is not a production deployment
plan or a security certification.

The prototype scope is frozen as follows:

- Keep the existing official InsightFace `buffalo_m` SCRFD and ArcFace models.
- Keep the existing MobileNetV3 PAD model and its current threshold.
- Keep cosine-similarity gallery matching; do not add or train an SVM.
- Do not train another face-recognition or PAD model for this prototype.
- Keep all processing local and use only camera index `0` when explicitly authorized.
- Do not store camera frames, face crops, embeddings, or similarity histories.
- Preserve the existing identity labels exactly; identity `000` must not be renamed.
- Present the result as a functional academic/demo prototype, not as production-grade
  biometric authentication.

The five manual camera acceptance tests are intentionally deferred until the user is
ready. They are specified in section 6 and are the only required tests involving
additional people or presentation attacks.

## 2. Current verified baseline

| Area | Current evidence | State |
|---|---|---|
| Face model | Official `buffalo_m` ZIP and extracted SCRFD/ArcFace SHA-256 values verified | Ready |
| PAD model | Hash-pinned MobileNetV3 ONNX loads on CPU | Ready for prototype use |
| Gallery | 1,430 vectors across 187 enrolled identities | Ready |
| Recognition | Validation-selected cosine threshold `0.6713641881942749` | Frozen exploratory setting |
| Live camera | Camera `0` identified the authorized user as `000` | Demonstrated once |
| Runtime | Face, PAD, gallery and CPU provider readiness check passed | Ready |
| Automated tests | 50 tests pass | Ready |
| Privacy | Private data/models are under Git-ignored paths | Must be rechecked before release |

Important evidence:

- `reports/environments/runtime_readiness.json`
- `reports/environments/live_query_compatibility.json`
- `reports/evaluation/recognition_validation.json`
- `reports/evaluation/recognition_test.json`
- `configs/runtime.yaml`
- `configs/decisions.yaml`

## 3. Work required before calling the prototype final

### 3.1 Freeze the prototype decision

Record in `configs/decisions.yaml` and the final report that no additional model
training is planned. The current weights, hashes, preprocessing, gallery, thresholds
and CPU provider become the prototype baseline. This closes training work only for
the prototype; it does not turn the current metrics into production claims.

Completion evidence:

- A dated `PROTOTYPE_SCOPE` decision in `configs/decisions.yaml`.
- Model/package hashes remain identical to `configs/runtime.yaml`.
- The limitations in section 8 appear in the final model card and presentation.

### 3.2 Finalize the camera runtime

The runtime must remain fail-closed: PAD must pass before identity matching can produce
`verified`. Complete the following code-level checks without changing models:

1. Show concise overlay messages for `NO FACE`, `MULTIPLE FACES`, `HOLD STILL`,
   `CHECKING LIVENESS`, `SPOOF`, `UNKNOWN` and `VERIFIED: <ID>`.
2. Display the identity only after PAD passes.
3. Ensure `q`, `Esc` and `Ctrl+C` release the camera and close the preview cleanly.
4. Handle camera-open and mid-run camera-read failures with a clear non-zero exit and
   no infinite loop.
5. Emit structured output only when the state changes; do not print the same cached
   result on every frame.
6. Never include frames, crops or embedding values in events or reports.
7. Continue using the bounded latest-frame queue so processing delay cannot grow
   without limit.

Verify the implementation with:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/python scripts/run_demo.py \
  --config configs/runtime.yaml \
  --check-only \
  --readiness-report reports/environments/runtime_readiness.json
```

Expected readiness conditions:

- exit code `0`;
- `event` is `runtime_ready`;
- `camera_opened` is `false`;
- detection, recognition and PAD use `CPUExecutionProvider`;
- gallery reports 1,430 vectors and 187 identities;
- every reported artifact hash matches `configs/runtime.yaml`.

### 3.3 Add a simple demonstration entry point

Provide one documented command for the operator:

```bash
.venv/bin/python scripts/run_demo.py --config configs/runtime.yaml --camera 0
```

The final demo runbook must explain:

- camera `0` is the only authorized source;
- the operator presses `q` or `Esc` to stop;
- the face should be front-facing, well lit and at least 80 pixels wide;
- the operator should hold still while eight PAD observations are collected;
- `UNKNOWN` is a valid fail-closed result, not a crash;
- no input frame is retained by the application.

Do not add automatic camera enumeration because permission currently covers only
camera index `0`.

### 3.4 Measure prototype runtime performance

Run one controlled camera session on the target laptop and record aggregate values in
`reports/benchmarks/prototype_camera.json`:

- laptop OS, CPU, RAM and camera description;
- Python and ONNX Runtime versions;
- active ONNX provider;
- test duration and processed frame count;
- median and 95th-percentile pipeline latency;
- approximate processed FPS;
- captured/dropped frame counts;
- peak process memory;
- number of camera read failures;
- whether shutdown released the camera successfully.

Do not store frames, embeddings, names or per-frame biometric histories in the
benchmark. Optimization is optional for the prototype unless the interface freezes,
latency grows without bound, or the camera cannot shut down reliably.

### 3.5 Complete documentation

Create or finalize these public-safe documents:

| Document | Required content |
|---|---|
| `docs/ARCHITECTURE.md` | Camera → SCRFD → quality/continuity → PAD → ArcFace → gallery flow |
| `docs/DEMO_RUNBOOK.md` | Setup, readiness check, camera command, expected states, shutdown |
| `docs/PROTOTYPE_MODEL_CARD.md` | Models, hashes, data scope, metrics, limitations, intended use |
| `docs/PRIVACY.md` | Local processing, restricted artifacts, retention and logging policy |
| `docs/EVALUATION.md` | Exploratory split, manual acceptance results and benchmark summary |

The documentation must say that unavailable session/device metadata remains unknown
and that the recognition results are not an official cross-session benchmark.

### 3.6 Make setup reproducible

Validate the documented setup in a clean virtual environment before final delivery:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-runtime.lock
.venv/bin/python -m pip check
```

Then install the private face-model package using the existing verified installer:

```bash
.venv/bin/python scripts/install_insightface_pack.py \
  --pack-zip data/processed/buffalo_m.zip \
  --model-root models/private/insightface \
  --acknowledge-non-commercial-model-terms
```

The supplied InsightFace-model terms must be acknowledged by each operator who
performs a fresh installation. Never download a substitute pack or disable the hash
checks to make setup pass.

### 3.7 Perform the final privacy and repository audit

Before sharing or publishing the repository:

1. Review `git status --short` and every intended tracked change.
2. Confirm raw images, processed crops, ZIP files, embeddings, ONNX/PyTorch weights,
   galleries and private score rows remain ignored.
3. Confirm reports contain aggregate values only.
4. Search tracked text for credentials, absolute developer paths and accidental
   biometric metadata.
5. Run `git diff --check` and the complete automated test suite.

Useful read-only checks:

```bash
git status --short
git diff --check
git ls-files | rg '\.(jpg|jpeg|png|mp4|zip|npy|npz|onnx|pt|pth|key|pem)$'
git grep -n -I -E '(password|secret|api[_-]?key|token)' -- ':!reference/1-reference.md'
```

Any match must be reviewed; a filename match alone is not proof of a leak. Do not use
destructive Git cleanup commands to remove user data.

## 4. Final prototype deliverables

The final handoff should contain:

- maintainable source under `src/faceguard/`;
- thin command entry points under `scripts/`;
- strict `configs/runtime.yaml` and the decision register;
- `requirements-runtime.lock`;
- automated tests and a passing test receipt;
- aggregate readiness, evaluation and benchmark reports;
- the five permanent documents listed in section 3.5;
- a public-safe model manifest containing filenames and hashes, not weights;
- private local models/gallery in ignored directories;
- a short presentation sequence demonstrating readiness, liveness and recognition.

DVC, MLflow, a container, CI hosting and further training are outside the minimal
prototype scope unless an academic rubric explicitly requires them.

## 5. Final verification sequence

Run these commands in order after documentation and code changes are complete:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python scripts/preflight.py \
  --output reports/environments/environment_local_runtime.json
.venv/bin/python scripts/run_demo.py \
  --config configs/runtime.yaml \
  --check-only \
  --readiness-report reports/environments/runtime_readiness.json
git diff --check
```

After these pass, perform the deferred manual tests in section 6. Do not tag, publish
or call the project final if a required test fails; record the failure and limitation.

## 6. Deferred five-case camera acceptance test

These tests require explicit camera authorization and will be performed later by the
user. Use camera `0` only. Run each case as a separate short session and record only
the expected/observed state, aggregate score if needed, timestamp and pass/fail.

| Case | Input | Required prototype behavior |
|---|---|---|
| A | Enrolled user `000` | PAD passes and identity becomes `000` |
| B | Person absent from gallery | Result becomes `UNKNOWN`; never another identity |
| C | Printed photo or phone display | Result should become `SPOOF` or remain fail-closed |
| D | Empty scene/no face | Result remains `NO FACE`/retry; no identity |
| E | Two or more visible faces | Result remains `MULTIPLE FACES`/retry; no identity |

Acceptance rules:

- A failure must not be hidden by repeating the test until it passes.
- Record the first declared run and any controlled rerun separately.
- Cases B–E must never output a verified identity.
- Case C is expected to expose limitations in the current PAD model; document the
  observed outcome honestly rather than changing the threshold using this test.
- Delete any manually recorded test video after the approved retention period; the
  application itself must not record it.

## 7. Definition of done

The prototype is final only when all of the following are true:

- [ ] Prototype scope and no-further-training decision are recorded.
- [ ] Runtime overlay and clean shutdown behavior are complete.
- [ ] Camera-open/read failures terminate safely.
- [ ] Automated tests and `pip check` pass.
- [ ] Camera-free readiness report passes with frozen hashes.
- [ ] Target-laptop aggregate benchmark exists.
- [ ] Five manual acceptance cases are recorded.
- [ ] Architecture, runbook, model card, privacy and evaluation documents exist.
- [ ] Repository privacy/secret/artifact audit passes.
- [ ] Private images, embeddings, galleries and model weights are not published.
- [ ] Final report consistently labels the system as a prototype.

## 8. Mandatory limitations to disclose

The final report and demonstration must disclose at least these limitations:

- Recognition evaluation is exploratory because session and device metadata are
  unavailable and were not invented.
- The exploratory held-out unknown-person false-positive identification rate was
  5/280 searches, approximately **1.79%**.
- The current PAD candidate's test APCER was approximately **20.84%**.
- A successful live recognition of identity `000` proves the pipeline operates on the
  tested laptop; it does not establish general biometric or anti-spoof security.
- RGB-only PAD may fail on unseen displays, prints, masks, lighting and cameras.
- Thresholds must not be tuned using the deferred acceptance cases.
- The system must not be used for high-stakes access control, surveillance, policing,
  employment, credit, healthcare or other consequential decisions.
- The official InsightFace supplied models are used under their applicable
  non-commercial research terms.

## 9. Recommended completion order

1. Record the frozen prototype/no-training decision.
2. Finish runtime messages, failure handling and clean shutdown.
3. Create the aggregate target-laptop benchmark.
4. Write the five permanent documents.
5. Run automated readiness and privacy checks.
6. Perform the five deferred manual camera cases when the user is ready.
7. Review all evidence and produce the final prototype handoff.

