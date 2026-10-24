# Command Entry Points

Scripts provide small CLI wrappers around importable `src/faceguard/` functions. They must not duplicate core algorithms, contain developer-specific absolute paths, or silently choose defaults missing from configuration.

- `preflight.py`: environment and provider inventory, with conflict detection.
- `audit_data.py`: non-destructive dataset inventory and draft subject manifest.
- `inspect_model_pack.py`: local ONNX file sizes/hashes without loading or downloading.
- `extract_gallery.py`: permission/model-hash/provider-gated private embedding extraction.
- `build_kaggle_bundle.py`: code-only allowlisted ZIP; excludes data, models, reports and reference material.
