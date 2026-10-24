# Face-Embedding Extraction

The extraction path uses the pretrained InsightFace `buffalo_m` detector/recognizer pack. It does not train ArcFace and does not require PyTorch.

## Local environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.in
python scripts/preflight.py --output reports/environments/environment_local_after.json
python -m pytest -q
```

The local environment intentionally uses `onnxruntime` and `opencv-python`, exactly one distribution of each family. `requirements-runtime.lock` is generated only after installation, import checks and tests pass.

## Required private inputs

1. Approved crops under `data/processed/frad_legacy_crops/`.
2. Complete official model pack at `models/private/insightface/models/buffalo_m/` (or the equivalent authorized private root).
3. Verified `FRAD_PERMISSION` in `configs/decisions.yaml`.

InsightFace-supplied pretrained model weights are for non-commercial research unless separately licensed. Review the official terms before using the acknowledgement flag.

Inspect and hash the manually supplied pack without loading it:

```bash
.venv/bin/python scripts/inspect_model_pack.py \
  --model-root models/private/insightface \
  --pack buffalo_m \
  --output reports/audits/buffalo_m_files.json
```

After checking the official source and terms, copy the exact `det_2.5g.onnx` and `w600k_r50.onnx` hashes into the `INSIGHTFACE_PACK` decision and change its state only when the evidence is genuinely verified.

## Audit

```bash
.venv/bin/python scripts/audit_data.py \
  --dataset-root data/processed/frad_legacy_crops \
  --inventory data/manifests/frad_inventory.csv \
  --subjects data/manifests/frad_subjects.csv \
  --report reports/audits/frad_audit.json
```

The audit is non-destructive, marks exact duplicate copies excluded, and leaves cohort-dependent samples `UNRESOLVED`. It does not infer identity, session or consent.

## Local candidate extraction

```bash
.venv/bin/python scripts/extract_gallery.py \
  --dataset-root data/processed/frad_legacy_crops \
  --manifest data/manifests/frad_inventory.csv \
  --decisions configs/decisions.yaml \
  --model-root models/private/insightface \
  --output-dir models/private/frad_candidate_embeddings \
  --context local \
  --mode audit \
  --pack buffalo_m \
  --providers CPUExecutionProvider \
  --acknowledge-non-commercial-model-terms
```

This command refuses to run while local processing permission is unresolved, if the model directory is incomplete, if providers are unavailable, or if outputs already exist. `--mode audit` produces candidate embeddings only; it is not a final enrollment gallery.

For Kaggle, follow `docs/KAGGLE_EMBEDDING_RUNBOOK.md` and keep the session/input/output private.
