# Prototype Model Card

| Field | Value |
|---|---|
| System | FaceGuard AI v2 (`faceguard-v2-development`) |
| Application | Unified local academic prototype `0.1.0` |
| Release status | Academic demonstration prototype; not production-ready |
| Runtime | Python 3.12.3, ONNX Runtime 1.31.0, `CPUExecutionProvider` |
| Recognition | InsightFace `buffalo_m`: SCRFD 2.5G detector + ResNet-50 ArcFace recognizer |
| PAD | MobileNetV3 binary liveness candidate (ONNX) |
| Gallery | 1,430 L2-normalized 512-d vectors across 187 pseudonymous identities |
| Last verified | 25 October 2026 |

## Artifacts and hashes

All runtime artifacts are verified at startup; a mismatch blocks startup.

| Artifact | SHA-256 |
|---|---|
| `buffalo_m.zip` (official package) | `d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9` |
| `det_2.5g.onnx` (SCRFD detector) | `041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9` |
| `w600k_r50.onnx` (ArcFace recognizer) | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| `mobilenetv3_pad.onnx` (PAD candidate) | `bd8bcc251df7b09285893f51dc35a3f7bd3d6e78c4be5641f49a4130f30e1359` |
| `gallery.npz` (private gallery) | `1cfa5694135edefe768b0c929b72f7b1127d68f885194c633e9c91eeceed7465` |

The weights themselves are private, Git-ignored artifacts. This card lists filenames
and hashes only.

## Operating configuration

| Parameter | Value |
|---|---|
| Detection input | 640 × 640, SCRFD 2.5G |
| Single-face policy | Exactly one detected face; zero or multiple faces fail closed |
| Minimum detection score | `0.6` |
| Minimum face width/height | `80 px` |
| Brightness range | `40`–`215` mean gray |
| Minimum Laplacian variance | `40` |
| Track continuity | IoU ≥ `0.3`, gap ≤ `500 ms` |
| PAD crop | 1.3× face box, 224 × 224 RGB, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| PAD aggregation | Median of `8` consecutive observations |
| PAD threshold | `0.7626009583473206` (sigmoid liveness score) |
| Recognition embedding | 512-dimensional ArcFace, L2-normalized |
| Gallery scoring | Per-person maximum cosine; vectorized matrix multiply |
| Recognition acceptance | Top-1 cosine ≥ `0.6713641881942749`, margin ≥ `0.0`, no exact tie |
| Queue | Bounded latest-frame queue (size `2`), oldest waiting frame dropped |

The recognition threshold was selected on the validation split only and frozen before
the exploratory test evaluation. No SVM, classifier or additional training is used at
runtime.

## Data scope

- 234 pseudonymous numeric identities (`000`, `0001`–`0233`); labels were never renamed.
- 2,801 approved image files after excluding one exact duplicate from 2,802 discovered.
- 2,709 samples had exactly one face and produced usable embeddings.
- Enrollment: 1,430 vectors across 187 identities.
- Validation: 367 known probes and 265 probes from 23 identities absent from the gallery.
- Test: 367 known probes and 280 probes from 24 different absent identities.
- Session and device metadata are unavailable and were not inferred. The split is an
  **exploratory unknown-session** protocol, not an official cross-session benchmark.

## Metrics

| Protocol | Metric | Result |
|---|---|---|
| Validation | Unknown-person false-positive identification rate | 0 / 265 (0.00%) |
| Validation | Known identification failure rate | 12 / 367 (3.27%) |
| Test | Unknown-person false-positive identification rate | 5 / 280 (1.79%, 95% CI 0–4.46%) |
| Test | Known identification failure rate | 15 / 367 (4.09%, 95% CI 2.16–6.35%) |
| Test | Known misidentification rate | 3 / 367 (0.82%) |
| PAD candidate | Test APCER | approximately 20.84% |

The PAD number is the candidate's recorded test APCER. The executed training notebook,
official label mapping, split counts/hashes and artifact hashes are audited, but the
exact split CSVs and score artifacts are absent and the used third-party Kaggle mirror
reports an unknown license. It remains a development observation, not a certified
ISO/IEC 30107-3 result. The approved academic scope defers recovery or reproduction of
the online training run, so no independently reproducible-training claim is made. A
per-video BPCER/ACER is not claimed here.

## License and provenance

- The SCRFD and ArcFace weights are the official InsightFace `buffalo_m` files, used
  under the supplied InsightFace model terms (non-commercial research use). Each
  operator must acknowledge those terms during installation.
- The PAD candidate was trained in this project and is a prototype artifact under the
  official CelebA-Spoof non-commercial research restrictions. The used Kaggle mirror's
  distribution authority and the exact split-file export remain unresolved; no
  reproducibility, production, redistribution or certification claim is made. It is
  accepted only as a frozen baseline for the supervised local academic demonstration.
- No substitute model pack was downloaded, and hash verification is never disabled.

## Intended use

Local, consent-based demonstration of the implemented capture → detection → PAD →
recognition pipeline on one authorized operator and camera `0`. Suitable for academic
review, pipeline integration evidence and discussion of failure modes.

## Prohibited use

Do not use for access control, surveillance, policing, employment, credit, healthcare,
identity verification or any other consequential decision. Do not deploy against
non-consenting people. Do not present the results as a security certification.

## Limitations

- Recognition evaluation is exploratory because session and device metadata are
  unavailable and were not invented.
- The exploratory held-out unknown-person false-positive identification rate was 5/280
  searches, approximately **1.79%**.
- The current PAD candidate's test APCER was approximately **20.84%**.
- Third-party PAD training reproducibility is deferred outside this academic prototype;
  the missing dataset and exact run artifacts remain declared limitations.
- A successful live recognition of identity `000` proves the pipeline operates on the
  tested laptop; it does not establish general biometric or anti-spoof security.
- RGB-only PAD may fail on unseen displays, prints, masks, lighting and cameras.
- Thresholds must not be tuned using the deferred acceptance cases.
- The system must not be used for high-stakes access control, surveillance, policing,
  employment, credit, healthcare or other consequential decisions.
- The official InsightFace supplied models are used under their applicable
  non-commercial research terms.
