# Kaggle Face-Embedding Runbook

Use `notebooks/02_kaggle_frad_embeddings.ipynb` only for an explicitly authorized private FRAD transfer. ArcFace is pretrained; this notebook performs audit and inference, not training.

## Inputs
Attach your private Kaggle inputs:

1. **Project Source:** `dist/faceguard-kaggle-source.zip` (mounted e.g. at `/kaggle/input/faceguard-kaggle-source` or `/kaggle/input/datasets/jit007/faceguard-kaggle-source`).
2. **FRAD Dataset:** Uploaded as either a folder or a `.zip` file (mounted e.g. at `/kaggle/input/datasets/jit007/frad-legacy-crops` or `/kaggle/input/frad-legacy-crops`). The notebook automatically detects if it is zipped and unzips it into `/kaggle/working/faceguard/data/`.
3. **Model Pack (Optional):** If `buffalo_m` is not attached as an input, the notebook automatically downloads the official release into `/kaggle/working/faceguard/models/buffalo_m/`.

Build a code-only source bundle from the explicit allowlist:

```bash
.venv/bin/python scripts/build_kaggle_bundle.py \
  --output dist/faceguard-kaggle-source.zip
```

The builder excludes the complete `data/`, `models/`, `reports/` and `reference/` trees and rejects image/model/embedding suffixes from its allowlist. Inspect the ZIP before uploading it as a private Kaggle code input.

The project input's `configs/decisions.yaml` must set `FRAD_PERMISSION` to `VERIFIED` or `FROZEN` and explicitly contain `kaggle_cloud_transfer: true`. Never change that field merely to bypass the guard.

## Execution order

1. Select a Kaggle GPU session and keep the notebook private.
2. Set the three exact `/kaggle/input/...` paths; placeholders cause a hard failure.
3. Run K0 preflight before package changes.
4. Resolve only missing dependencies; keep one OpenCV and one ONNX Runtime distribution.
5. Restart the kernel after dependency changes and save a second preflight report.
6. Run K1 structural audit and inspect its report.
7. Run K2 audit-mode embedding extraction with explicit CUDA provider.
8. Transfer restricted outputs to approved storage, record hashes, then clear Kaggle outputs according to the permission/retention policy.

## Provider rules

- CUDA extraction requires `CUDAExecutionProvider` to be genuinely available. The command fails instead of silently using CPU.
- Do not install both `onnxruntime` and `onnxruntime-gpu`.
- Do not install both `opencv-python` and `opencv-python-headless`.
- Do not replace Kaggle's working torch/torchvision pair; recognition extraction does not need PyTorch.

## Output classification

`frad_embeddings.npz` and its metadata are restricted biometric artifacts. The JSON audit/environment reports can be retained only after checking they contain no paths or fields prohibited by the project's publication policy.
