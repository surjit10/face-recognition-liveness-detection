# Command Entry Points

Scripts provide small CLI wrappers around importable `src/faceguard/` functions. They must not duplicate core algorithms, contain developer-specific absolute paths, or silently choose defaults missing from configuration.

- `preflight.py`: environment and provider inventory, with conflict detection.
- `audit_data.py`: non-destructive dataset inventory and draft subject manifest.
- `inspect_model_pack.py`: local ONNX file sizes/hashes without loading or downloading.
- `extract_gallery.py`: permission/model-hash/provider-gated private embedding extraction.
- `prepare_recognition_context.py`: bounded-CPU, non-destructive loose context crops from original recognition photos, with full-frame fallbacks and an auditable manifest.
- `build_kaggle_bundle.py`: code-only allowlisted ZIP; excludes data, models, reports and reference material.
- `prepare_runtime_artifacts.py`: CRC/hash-verified private PAD bundle promotion.
- `install_insightface_pack.py`: offline, terms-acknowledged, hash-verified model-pack installation.
- `build_gallery.py`: build a protected gallery from an explicit frozen enrollment manifest.
- `validate_recognition_manifests.py`: prove split disjointness, hashes, decodeability and identity rules.
- `create_recognition_split.py`: deterministic exploratory open-set splitting without invented session/device data.
- `evaluate_recognition.py`: create private open-set search scores from frozen probe manifests.
- `calibrate_recognition.py`: validation-only open-set threshold selection from saved search scores.
- `report_recognition.py`: frozen operating-point metrics with person-cluster confidence intervals.
- `run_app.py`: centralized local desktop interface with automatic readiness checks,
  an in-window camera preview, complete fail-closed analysis and aggregate diagnostics.
- `run_demo.py`: fail-closed readiness check, sequential-video or bounded-queue webcam runtime.
