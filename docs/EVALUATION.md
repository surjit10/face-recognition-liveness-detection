# Evaluation

All results here are exploratory prototype evidence. Session and device metadata for
the recognition dataset are unavailable and were not invented, so this is not an
official or cross-session benchmark.

## Recognition protocol

| Item | Value |
|---|---|
| Protocol | `exploratory_unknown_session_open_set_v1` |
| Seed | 42 |
| Identities | 234 preserved labels (`000`, `0001`–`0233`) |
| Enrolled identities | 187 |
| Enrollment vectors | 1,430 |
| Validation impostor identities / probes | 23 / 265 |
| Test impostor identities / probes | 24 / 280 |
| Known validation / test probes | 367 / 367 |
| Score unit | One gallery search per probe image |
| Metric | Maximum individual-embedding cosine per person |

Validation and test identities are person-disjoint for the unknown cohorts, and probe
files are hash-disjoint from enrollment files. Labels were never renumbered or inferred
from filenames.

## Threshold selection

The cosine threshold was selected on the validation split with a
`recognition_fmr_bound` policy and a validation false-match bound of `0.0`, then frozen.
The selected threshold is `0.6713641881942749` with minimum top-1/top-2 margin `0.0`.
The test split was evaluated once after freezing; it was not used for selection or
tuning.

## Recognition results

| Split | Known searches | Correct accepts | Rejections | Misidentifications | Known failure rate | Impostor searches | False accepts | FPIR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 367 | 355 | 9 | 3 | 3.27% | 265 | 0 | 0.00% |
| Test | 367 | 352 | 12 | 3 | 4.09% | 280 | 5 | **1.79%** |

Test 95% person-cluster confidence intervals: known failure rate `2.16%`–`6.35%`;
unknown-person FPIR `0%`–`4.46%` (2,000 resamples, seed 42). The validation known
failure interval was `1.61%`–`5.23%`; validation FPIR was `0%`.

The 5/280 test false accepts are the principal generalization risk. A successful live
recognition of `000` only shows the pipeline operates on the tested laptop; it does not
establish general biometric security.

## PAD candidate

The MobileNetV3 PAD candidate's recorded test APCER is approximately **20.84%** at the
pinned operating point (`0.7626009583473206`, median of 8 observations). The executed
Kaggle notebook, official label semantics, split counts/hashes, training stages and
artifact hashes are now audited in `reports/audits/pad_training_evidence.json`.
However, the exact split CSVs and score artifacts were not exported, the used Kaggle
mirror reports an unknown license, and the recorded equal-error threshold selector is
not an explicitly frozen blueprint policy. Treat PAD as a development candidate, not a
certified ISO/IEC 30107-3 result. RGB-only PAD may fail on unseen displays, prints,
masks, lighting and cameras.

The v1.2 academic-scope amendment explicitly defers recovery of the online dataset,
exact manifests, training history, score exports, complete package lock and rerun.
Accordingly, these recorded metrics are historical development evidence only; they do
not establish an independently reproducible training result or production suitability.

## Target benchmark

`reports/benchmarks/prototype_camera.json` records a 15-second local camera session
with the enrolled operator in view:

| Field | Value |
|---|---|
| OS / CPU / RAM | Linux 6.14.0-33 x86_64 / Intel Core i5-10300H @ 2.50 GHz, 8 logical CPUs / 15.5 GB |
| Python / ONNX Runtime / provider | 3.12.3 / 1.31.0 / `CPUExecutionProvider` |
| Camera | index `0`, 640 × 480, reported 30 FPS |
| Duration / processed frames | 15.34 s / 100 |
| Median / p95 capture-to-decision latency | 240.40 ms / 285.61 ms |
| Approximate processed FPS | 6.52 |
| Captured / dropped frames | 214 / 109 |
| Read failures | 0 |
| Peak memory | 644.2 MB |
| Camera released on shutdown | true |
| Stable-face verification time | 1.34 s |
| Observed states | `COLLECTING_PAD` × 4; `IDENTIFIED` × 96 |

This post-approval session exercised capture, PAD and accepted-identity paths and passed
every frozen academic-demo performance objective. The bounded latest-frame queue
discarded 109 stale waiting frames while processing 100 frames; no camera read failure
occurred. This is not a general hardware guarantee. The receipt contains no frames,
crops, embeddings or identity histories.

## Manual acceptance cases

The five manual camera cases were performed on 1 November 2026 with camera `0` and the
frozen thresholds. The privacy-safe receipt is
`reports/phase_receipts/camera_acceptance_2026-11-01.json`; screenshots containing
faces were reviewed during the guided session and were not retained in the repository.

| Case | Input | Required behavior | Status |
|---|---|---|---|
| A | Enrolled operator | PAD passes; correct configured identity | Passed — screenshot reviewed |
| B | Person absent from gallery | `UNKNOWN`; never another identity | Passed — operator attestation |
| C | Face displayed on a phone | `LIVENESS NOT CONFIRMED` (internal `PAD_FAILED`) or another fail-closed result | Passed — `POSSIBLE SPOOF / REJECTED` |
| D | Empty scene | `NO FACE`/`LOOKING FOR FACE`; no identity | Passed — screenshot reviewed |
| E | Live face plus face displayed on a phone | `MULTIPLE FACES`; no identity | Passed — screenshot reviewed |

Case C passed for the one phone/display combination tested, but it does not establish
robustness to other displays, printed photos, masks, lighting or cameras. The
application recorded no video, and the evidence receipt contains no biometric data.

## Mandatory limitations

- Recognition evaluation is exploratory because session and device metadata are
  unavailable and were not invented.
- The exploratory held-out unknown-person false-positive identification rate was
  5/280 searches, approximately **1.79%**.
- The current PAD candidate's test APCER was approximately **20.84%**.
- Third-party PAD training reproducibility is explicitly deferred outside the academic
  prototype; missing run artifacts receive no completion credit.
- A successful live recognition of identity `000` proves the pipeline operates on the
  tested laptop; it does not establish general biometric or anti-spoof security.
- RGB-only PAD may fail on unseen displays, prints, masks, lighting and cameras.
- Thresholds were not tuned using the manual acceptance cases.
- The system must not be used for high-stakes access control, surveillance, policing,
  employment, credit, healthcare or other consequential decisions.
- The official InsightFace supplied models are used under their applicable
  non-commercial research terms.
