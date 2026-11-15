# Model Evaluation & Benchmark Report

This document describes the evaluation protocol, data-quality handling, accuracy
results, anti-spoofing (PAD) results and hardware benchmarks for the FaceGuard AI
pipeline. Every number below is reproduced from the aggregate reports in
`reports/evaluation/`, `reports/audits/` and `reports/benchmarks/`.

## Face Recognition Protocol

The recognition evaluation is an open-set identification protocol: it measures both
recognition of enrolled people and rejection of strangers who are not enrolled.

| Parameter | Configuration |
|---|---|
| Evaluation protocol | Open-set identification (`exploratory_unknown_session_open_set_v1`) |
| Seed | 42 |
| Identities in the split | 234 (233 with usable samples after the quality gate) |
| Enrolled identities (gallery) | 186 |
| Enrollment vectors | 1,388 ArcFace embeddings (512-d unit vectors) |
| Validation probes (known / stranger) | 351 known / 259 stranger probes from 23 identities |
| Test probes (known / stranger) | 351 known / 274 stranger probes from 24 identities |
| Search rule | Highest cosine similarity per enrolled person |

Stranger identities in validation and test are disjoint from the gallery and from each
other, and no image file (by SHA-256) appears in more than one split.

**Same-session limitation.** 202 of the 205 identities with timestamped filenames were
photographed within 5 minutes (median 50 seconds), so enrollment and known-person probe
images come from the same sitting. The known-person results therefore show
same-session recognition, not recognition of someone returning on a different day.

## Data Quality Gate (applied 14 November 2026)

Error analysis of the first test evaluation showed that most recognition errors were
caused by the data pipeline, not by the face recognizer. The offline embedding
extraction accepted any image in which exactly one face was detected. In crowded lab
photos, the detector sometimes missed the subject's turned face and detected only a
small person in the background (about 20 px wide, detection score about 0.5). That
bystander's embedding was then labelled as the subject. The live application never
accepts such faces, because it enforces a quality gate.

The fix applies the same gate offline. `face_quality_reason` in
`src/faceguard/pipeline.py` is now one shared function used by live verification, live
enrollment and `scripts/apply_recognition_quality_gate.py`, with the unchanged
thresholds from `configs/runtime.yaml` (detection ≥ 0.6, face side ≥ 80 px, brightness
40–215, Laplacian variance ≥ 40).

| Exclusion reason | Samples |
|---|---:|
| Detection score below 0.6 | 48 |
| Face smaller than 80 px | 25 |
| Face too blurry | 12 |
| Known probe whose identity has no remaining enrollment (`0044`) | 1 |
| **Total** | **86** |

Every exclusion is listed in `data/manifests/recognition_quality_exclusions.csv`.
Identity assignments were not changed and the data was not re-split.

**Disclosure.** The defect was found by inspecting errors on the test set. The gate is
the pre-existing runtime rule and was not tuned on test data. After applying it, the
threshold was re-selected on validation only and the test set was scored once.

**Known label issue (not changed).** Identity pairs `0178`/`0231` and `0191`/`0218`
appear to be the same people photographed in two sessions and filed under two IDs
(average-embedding cosine 0.89 and 0.90; visual check). `0178` and `0231` are both
enrolled. No misidentification between them occurred on validation or test.

## Threshold Calibration

The cosine threshold was selected on the validation split using the frozen
`recognition_fmr_bound` policy: among thresholds that accept no validation strangers,
choose the one that rejects the fewest known people (higher threshold on ties).

- **Operating threshold:** `0.6553505659103394` (previously `0.6713641881942749`)
- **Minimum top-1 vs top-2 margin:** `0.0`. The margin check is effectively disabled;
  only exact ties are rejected.

Zero accepted strangers on validation (0/259) does not mean the true rate is zero. It
bounds the rate below 1.15% at 95% confidence, and those probes come from only 23
people.

## Recognition Results

| Split | Known searches | Correct | Rejected | Wrong person | Known failure rate | Stranger searches | Strangers accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 351 | 350 | 1 | 0 | 0.28% | 259 | 0 |
| **Test (held out)** | **351** | **348** | **3** | **0** | **0.85%** | **274** | **0** |

**95% intervals (test).**
- Known failure rate: 0.00%–1.99% (2,000 bootstrap resamples by person, seed 42).
- Strangers accepted: 0 of 274. A percentile bootstrap cannot produce a non-zero
  interval when no events occur. The exact one-sided 95% upper bound is **1.09%**, and
  it is optimistic because the 274 probes come from only 24 people.

**Superseded result (before the quality gate, threshold 0.6714).** Test: 352/367
correct, 12 rejected, 3 wrong person, and 5/280 strangers accepted (1.79%). 16 of
those 20 errors involved bystander-face samples. The largest "stranger" match (cosine
0.91) was the same background person appearing in two different subjects' photos.

## Anti-Spoofing (PAD) Performance

**Model and training.**
- **Model:** MobileNetV3-Small, ImageNet-pretrained. Only the classifier and the last
  three feature blocks were fine-tuned.
- **Data:** CelebA-Spoof (Kaggle mirror). Splits are subject-disjoint: 100,000 train,
  20,000 validation (taken from official training subjects), 40,000 test (official
  test subjects).
- **Training setup:** `BCEWithLogitsLoss` with class weighting and AdamW (weight decay
  1e-4). Two epochs trained the head only (lr 1e-3). Six epochs then fine-tuned the last
  three blocks (lr 1e-4) together with the classifier (lr 1e-3).
- **Infrastructure and augmentation:** two T4 GPUs with DDP. Augmentations were
  horizontal flip, colour jitter and light blur. BatchNorm layers in the frozen part of
  the backbone were kept in eval mode.
- **Threshold:** `0.7626` (sigmoid liveness score), chosen where APCER equals BPCER on
  validation.

| Split | APCER (attacks accepted as live) | BPCER (live faces rejected) | ACER | AUC |
|---|---:|---:|---:|---:|
| Validation | 0.74% | 0.74% | 0.74% | — |
| Test | **20.84%** (4,248 / 20,386) | 1.08% (211 / 19,614) | 10.96% | 0.983 |

Test APCER is about 28 times validation APCER on the same dataset. The ranking quality
is good (AUC 0.983), but the validation set did not predict test behaviour, so the
threshold is miscalibrated for unseen subjects. The validation/test score files and
per-attack-type labels were not exported from the training run, so the gap cannot be
broken down by attack type. APCER is reported over all attack types together, not per
attack instrument as in ISO/IEC 30107-3.

**Runtime aggregation.** The application takes the median PAD score over 8
observations of the same tracked face. Consecutive webcam frames are strongly
correlated, so the median does not multiply out frame-level errors. System-level
attack rejection on a webcam has not been measured beyond the single trials below.

The CelebA-Spoof images and derived data are not redistributable. The PAD weights are
excluded from Git and verified by SHA-256 at startup.

## Pipeline Security Fix: Liveness and Identity Bound to the Same Face (14 November 2026)

Before this fix, a track continued as long as the face box overlapped the previous box
(IoU ≥ 0.3). PAD evidence was the median of 8 frames, but the identity came from the
last frame only. Live frames of an unenrolled person followed by a phone photo of an
enrolled person, held in the same place, passed PAD on the median and returned the
enrolled person's identity from the photo frame. This was reproduced with the real
`VideoVerifier` and `Gallery` classes.

Every frame of a track must now also match the track's first face (ArcFace cosine ≥
`min_track_embedding_similarity` = 0.5). Otherwise the PAD evidence is discarded and a
new track starts. The value was chosen from enrollment and validation pairs only:
0.0126% of different-person pairs reach 0.5, while 97.6% of same-person photo pairs
exceed it, even with deliberate pose and accessory changes. A genuine face falling
below 0.5 can only cause a retry, never an acceptance. The embedding was already
computed every frame, so the fix adds one 512-d dot product per frame and no model
call. Regression tests: `tests/unit/test_pipeline.py`.

## Target Hardware Benchmark

Measured on 1 November 2026 on an Intel Core i5-10300H CPU @ 2.50 GHz (8 logical
cores, 15.5 GB RAM), Linux, USB webcam, before the 14 November changes. The submission
keeps this measurement: those changes add no model inference (one 512-d dot product per
frame), so it was not re-measured.

| Metric | Measured | Target | Status |
|---|---|---|---|
| Processed throughput | 6.52 FPS | ≥ 5.0 FPS | Passed |
| P95 capture-to-decision latency | 285.61 ms (median 240.40 ms) | ≤ 500 ms | Passed |
| Peak memory | 644.2 MB | ≤ 750 MB | Passed |
| Stable-face verification time | 1.34 s | ≤ 3.0 s | Passed |
| Camera read failures | 0 | 0 | Passed |
| Frame queue | size 2; 109 stale frames dropped | bounded at 2 | Passed |

The targets are self-set academic-demo targets; 6.5 FPS is adequate for a kiosk but is
not full-frame-rate video. Raw ONNX stage medians are: SCRFD detection 35.4 ms,
ArcFace 81.4 ms, PAD 1.5 ms. ArcFace runs on every frame because the identity-binding
check needs an embedding for each frame.

Next optimisation steps, in order of expected benefit:
- a smaller detector input for close-range kiosk faces;
- running the identity-continuity check every second frame;
- INT8 quantisation or OpenVINO for ArcFace;
- a lighter recognizer for the continuity check only.

## Live Camera Acceptance (1 November 2026)

Each case was a **single guided trial** on one laptop and webcam. Evidence is operator
observation or a screenshot that was reviewed but not retained. These are smoke tests,
not accuracy measurements. They were recorded with the previous gallery and threshold
and before the identity-binding fix, and were not repeated for this submission. The
identity-binding fix only adds a rejection path; the gallery and threshold changes bear
on cases A and B.

| Case | Scenario | Expected behaviour | Observed |
|---|---|---|---|
| A | Enrolled operator | Liveness passes; identity verified | Passed |
| B | Unregistered live person | Liveness passes; `UNKNOWN` | Passed (operator attestation) |
| C | Operator's face on a phone screen | Liveness fails; `POSSIBLE SPOOF` | Passed (one phone, one trial) |
| D | Empty scene | `LOOKING FOR FACE` | Passed |
| E | Live face and phone face together | `MULTIPLE FACES`, no identity check | Passed |

Printed photos, laptop or tablet screens, other phones, masks, video replay and
lighting changes were not tested. The acceptance receipt references a benchmark file
hash (`3aba64…`) that matches no retained version of `reports/benchmarks/prototype_camera.json`.
The retained benchmark is the performance-gate run (`187744…`).

## System Scope & Considerations

- **Supervised use only.** The system is designed for attended attendance kiosks, not
  unsupervised access control.
- **RGB-only PAD.** Frame-level test APCER is 20.8%, and live testing covered one
  phone-screen trial. Unsupervised use would need active challenge-response (blink,
  head turn), multi-frame temporal models, or IR/depth sensors, plus a per-attack-type
  evaluation on webcam-domain data.
- **Recognition generalisation.** Results are same-session and come from 186 enrolled
  people. Stranger false-accept rates rise as the gallery grows, so the threshold must
  be re-calibrated for a larger gallery and validated on cross-session data.
- **Licensing.** InsightFace `buffalo_m` weights and CelebA-Spoof are licensed for
  non-commercial research only. Commercial deployment would need licensed or
  self-trained models.
