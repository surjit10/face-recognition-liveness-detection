# Operational Commands

The active scripts are limited to running, verifying, benchmarking, packaging and
maintaining the frozen academic prototype. Completed training and data-preparation
utilities are archived under `reference/archive/final-submission-excluded/scripts/`.

## Demo and verification

- `run_app.py` — launch the complete desktop application.
- `run_demo.py` — camera-free readiness check or legacy camera/video runner.
- `verify_private_models.py` — verify private model/gallery hashes and CPU inference.
- `configure_operator.py` — configure the optional local operator PIN.
- `manage_gallery.py` — inspect, verify or delete protected gallery identities.

## Evidence and performance

- `benchmark_runtime.py` — produce the end-to-end performance receipt.
- `benchmark_stages.py` — benchmark detector, PAD and recognition ONNX stages.
- `log_mlflow.py` — log aggregate reports to the local MLflow store.
- `preflight.py` — record Python, provider and package compatibility.

## Release and submission

- `scan_artifacts.py` — block private artifacts, credentials and biometric files.
- `build_submission.py` — build the allowlisted public-safe submission ZIP.
- `generate_release_manifest.py` — hash release inputs and reports.
- `promote_release.py` — record explicit academic-scope promotion approval.
- `manage_release.py` — register, switch and roll back local releases.
- `install_insightface_pack.py` — install the authorized hash-verified model pack.
