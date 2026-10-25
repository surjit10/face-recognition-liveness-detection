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
pinned operating point (`0.7626009583473206`, median of 8 observations). Its dataset
permission, label semantics and split manifests are not independently verified, and its
threshold-selection policy is not frozen. Treat PAD as a development candidate, not a
certified ISO/IEC 30107-3 result. RGB-only PAD may fail on unseen displays, prints,
masks, lighting and cameras.

## Target benchmark

`reports/benchmarks/prototype_camera.json` records one 10-second local camera session:

| Field | Value |
|---|---|
| OS / CPU / RAM | Linux 6.14.0-33 x86_64 / Intel Core i5-10300H @ 2.50 GHz, 8 logical CPUs / 15.5 GB |
| Python / ONNX Runtime / provider | 3.12.3 / 1.31.0 / `CPUExecutionProvider` |
| Camera | index `0`, 640 × 480, reported 30 FPS |
| Duration / processed frames | 10.49 s / 62 |
| Median / p95 capture-to-decision latency | 36.8 ms / 108.05 ms |
| Approximate processed FPS | 5.91 |
| Captured / dropped frames | 65 / 0 |
| Read failures | 0 |
| Peak memory | 643.2 MB |
| Camera released on shutdown | true |
| Observed states | `RETRY` × 62 (`no_face`) |

Because no face was present, this session exercised capture and detection only; PAD
and recognition latency were not measured. Run the benchmark with the operator in
frame for a full-path measurement. The receipt contains no frames, crops, embeddings or
identity histories.

## Deferred acceptance cases

The five manual camera cases are intentionally deferred until the operator is ready.
No results have been recorded yet. Perform each as a separate short session, camera `0`
only, and record expected/observed state, timestamp and pass/fail. Never repeat until it
passes, and never tune thresholds using these cases.

| Case | Input | Required behavior | Status |
|---|---|---|---|
| A | Enrolled user `000` | PAD passes; identity becomes `000` | Pending |
| B | Person absent from gallery | `UNKNOWN`; never another identity | Pending |
| C | Printed photo or phone display | `LIVENESS NOT CONFIRMED` (internal `PAD_FAILED`) or another fail-closed result | Pending |
| D | Empty scene | `NO FACE`; no identity | Pending |
| E | Two or more faces | `MULTIPLE FACES`; no identity | Pending |

Case C is expected to expose limitations of the current PAD model; document the
observed outcome honestly. The application itself records no video; any manual
recording is external and must be deleted after the approved retention period.

## Mandatory limitations

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
