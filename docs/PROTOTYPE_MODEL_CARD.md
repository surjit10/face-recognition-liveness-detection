# Prototype Model Card

| Field | Value |
|---|---|
| System | FaceGuard AI v2 (`faceguard-v2-development`) |
| Application | Unified local academic prototype `0.1.0` |
| Release status | Academic demonstration prototype; not production-ready |
| Runtime | Python 3.12.3, ONNX Runtime 1.31.0, `CPUExecutionProvider` |
| Recognition | InsightFace `buffalo_m`: SCRFD 2.5G detector + ResNet-50 ArcFace recognizer (pretrained, not fine-tuned) |
| PAD | MobileNetV3-Small binary liveness model, fine-tuned on CelebA-Spoof (ONNX) |
| Gallery | 1,388 L2-normalized 512-d vectors across 186 pseudonymous identities |
| Last verified | 14 November 2026 |

## Artifacts and hashes

All runtime artifacts are verified at startup; a mismatch blocks startup.

| Artifact | SHA-256 |
|---|---|
| `buffalo_m.zip` (official package) | `d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9` |
| `det_2.5g.onnx` (SCRFD detector) | `041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9` |
| `w600k_r50.onnx` (ArcFace recognizer) | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| `mobilenetv3_pad.onnx` (PAD model) | `bd8bcc251df7b09285893f51dc35a3f7bd3d6e78c4be5641f49a4130f30e1359` |
| `gallery.npz` (private gallery, quality-gated) | `31ceacdbc90dbccdff8bc7477ce83bd8f56c306c05dfd4440f89141136463edc` |

The weights and gallery are Git-ignored. This card lists filenames and hashes only.

## Operating configuration

| Parameter | Value |
|---|---|
| Detection input | 640 × 640, SCRFD 2.5G |
| Single-face policy | Exactly one detected face; zero or multiple faces fail closed |
| Minimum detection score | `0.6` |
| Minimum face width/height | `80 px` |
| Brightness range | `40`–`215` mean gray |
| Minimum Laplacian variance | `40` |
| Track continuity | IoU ≥ `0.3`, gap ≤ `500 ms`, and ArcFace cosine ≥ `0.5` with the track's first face |
| PAD crop | 1.3× face box, 224 × 224 RGB, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| PAD aggregation | Median of `8` observations of the same tracked face |
| PAD threshold | `0.7626009583473206` (sigmoid liveness score) |
| Recognition embedding | 512-dimensional ArcFace, L2-normalized |
| Gallery scoring | Per-person maximum cosine; vectorized matrix multiply |
| Recognition acceptance | Top-1 cosine ≥ `0.6553505659103394`; margin setting `0.0` (only exact ties rejected) |
| Queue | Bounded latest-frame queue (size `2`), oldest waiting frame dropped |

The recognition threshold was selected on the validation split only, and the test split
was then scored once. No SVM, classifier or additional training is used at runtime.

## Data scope

- 234 pseudonymous numeric identities (`000`, `0001`–`0233`); labels were never renamed.
- 2,801 approved image files after one exact duplicate was excluded from 2,802 found.
- 2,709 samples had exactly one detected face and produced embeddings.
- 86 samples failed the runtime face-quality gate and were excluded on 14 November
  2026. Most were small background bystanders detected instead of the subject (see
  [EVALUATION.md](EVALUATION.md#data-quality-gate-applied-14-november-2026)).
  Identity `0044` has no remaining sample.
- Enrollment: 1,388 vectors across 186 identities.
- Validation: 351 known probes and 259 probes from 23 identities absent from the gallery.
- Test: 351 known probes and 274 probes from 24 different absent identities.
- Images of each person were mostly taken within one sitting (median span 50 seconds),
  so enrollment and known probes are same-session.

## Metrics

| Protocol | Metric | Result |
|---|---|---|
| Validation | Stranger false-positive identification rate | 0 / 259 (0.00%; one-sided 95% upper bound 1.15%) |
| Validation | Known identification failure rate | 1 / 351 (0.28%) |
| Test | Stranger false-positive identification rate | 0 / 274 (0.00%; one-sided 95% upper bound 1.09%) |
| Test | Known identification failure rate | 3 / 351 (0.85%; 95% CI 0.00–1.99%) |
| Test | Known misidentification rate | 0 / 351 |
| PAD (CelebA-Spoof) | Validation APCER / BPCER | 0.74% / 0.74% |
| PAD (CelebA-Spoof) | Test APCER / BPCER / AUC | 20.84% / 1.08% / 0.983 |

The superseded pre-gate test result was 5/280 strangers accepted (1.79%) and 15/367
known failures (4.09%). Most of those errors came from bystander-face samples.

## Model Provenance

- **SCRFD detection and ArcFace recognition:** pretrained weights from the official
  InsightFace `buffalo_m` package, used without fine-tuning.
- **Anti-spoofing (PAD):** ImageNet-pretrained MobileNetV3-Small. The classifier and
  last three feature blocks were fine-tuned on a subject-disjoint CelebA-Spoof subset
  (100k / 20k / 40k), using `BCEWithLogitsLoss` and AdamW on two T4 GPUs with DDP. The
  model was exported to ONNX (PyTorch/ONNX maximum output difference 1.3e-5).
- **Gallery:** ArcFace embeddings of quality-gated enrollment images.

## Intended Use & Boundaries

- **Intended use:** local, supervised attendance demonstration and research on computer
  vision pipelines.
- **Deployment guidance:** attended kiosks only, not unsupervised high-security access
  control.
- **Licensing:** InsightFace model packs and CelebA-Spoof are licensed for
  non-commercial research only. This prototype grants no commercial rights.
- **Key technical limitations:**
  - Recognition results are same-session and come from 186 enrolled people. A larger
    gallery raises the stranger false-accept rate at a fixed threshold.
  - Identity pairs `0178`/`0231` and `0191`/`0218` appear to be the same people under
    two IDs; labels were not changed.
  - PAD accepts 20.8% of attack images at frame level on CelebA-Spoof test. Live
    testing covered one phone-screen trial only; printed photos, other screens, masks
    and replays are unevaluated. Production use would need challenge-response or
    IR/depth sensing and a per-attack-type evaluation.
