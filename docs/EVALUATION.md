# Model Evaluation & Benchmark Report

This document outlines the evaluation methodology, dataset splits, accuracy metrics, and target hardware benchmarks for the FaceGuard AI pipeline.

## Face Recognition Protocol

The face recognition evaluation follows an open-set identification protocol to test both enrolled identity verification and stranger rejection.

| Parameter | Configuration |
|---|---|
| Evaluation Protocol | Open-Set Identification (`open_set_v1`) |
| Seed | 42 |
| Total Evaluated Identities | 234 distinct identities |
| Enrolled Identities (Gallery) | 187 individuals |
| Enrollment Vectors | 1,430 face embeddings (512-d unit vectors) |
| Validation Cohort (Known / Impostor) | 367 known probes / 265 unknown impostor probes (23 identities) |
| Test Cohort (Known / Impostor) | 367 known probes / 280 unknown impostor probes (24 identities) |
| Search Metric | Per-person maximum cosine similarity |

The unknown validation and test cohorts are completely identity-disjoint from each other and from the enrolled gallery, ensuring that stranger rejection is tested on genuinely unseen faces.

## Threshold Calibration

The cosine similarity decision threshold was calibrated on the validation split using a False Match Rate bound constraint ($\text{FMR} \le 0.0$ on validation):
- **Selected Operating Threshold**: **`0.671364`**
- **Minimum Top-1 vs Top-2 Margin**: `0.0`

The test split was held out completely during calibration and evaluated only once with the frozen operating parameters.

## Recognition Results

| Dataset Split | Known Searches | Correct Accepts | Below Threshold | Misidentifications | Known Error Rate | Impostor Searches | False Accepts | Impostor Error (FPIR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Validation** | 367 | 355 | 9 | 3 | 3.27% | 265 | 0 | **0.00%** |
| **Test (Held-Out)** | 367 | 352 | 12 | 3 | 4.09% | 280 | 5 | **1.79%** |

### Statistical Confidence Intervals (Test Split)
Evaluated via 2,000 cluster-bootstrapped resamples (seed 42):
- **Known Failure Rate (95% CI)**: `2.16%` – `6.35%` (point estimate: `4.09%`)
- **Unknown Impostor FPIR (95% CI)**: `0.00%` – `4.46%` (point estimate: `1.79%`)

## Anti-Spoofing (PAD) Performance

The MobileNetV3 presentation-attack detection model operates at an operating cutoff of **0.7626** (sigmoid confidence). On the held-out test split, the model achieves an Attack Presentation Classification Error Rate (**APCER**) of **20.84%**. 

To enhance live robustness, the runtime pipeline enforces a **3-frame consecutive confirmation** rule before passing frames to the recognition stage. This temporal smoothing prevents single-frame noise or camera flicker from causing false rejections or accepts.

*Dataset Note:* The model was pre-trained on Kaggle and exported to ONNX format. To maintain a lightweight repository footprint and prevent the dissemination of sensitive biometric images, the raw 50 GB training dataset is excluded from version control, while the ONNX model binary is tracked and verified via SHA-256 checksums.

## Target Hardware Benchmark

Measured on an Intel Core i5-10300H CPU @ 2.50 GHz (8 logical cores, 15.5 GB RAM) running Linux with standard USB webcam video:

| Benchmark Metric | Measured Result | Target Requirement | Evaluation Status |
|---|---|---|---|
| **Processed Throughput** | **6.52 FPS** | $\ge 5.0$ FPS | **Passed** |
| **P95 Total Latency** | **285.61 ms** (median: 240.40 ms) | $\le 500.0$ ms | **Passed** |
| **Peak Memory Footprint** | **644.2 MB** | $\le 750.0$ MB | **Passed** |
| **Verification Time** | **1.34 s** | $\le 3.0$ s | **Passed** |
| **Camera Read Failures** | **0 failures** | 0 failures | **Passed** |
| **Queue Performance** | 109 stale frames safely dropped | Bounded size 2 | **Passed** |

The 2-slot bounded frame queue ensures that the system always processes real-time frames without accumulating pipeline backlog or memory leaks.

## Live Camera Acceptance Tests

Five targeted functional scenarios were evaluated on the physical camera setup:

| Test Case | Scenario | Expected System Behavior | Result |
|---|---|---|---|
| **Case A** | Enrolled operator in view | Liveness passes; identity verified | **Passed** (Correctly Identified) |
| **Case B** | Unregistered live person | Liveness passes; labeled `UNKNOWN` | **Passed** (Unknown Blocked) |
| **Case C** | Smartphone screen replay attack | Liveness fails; labeled `POSSIBLE SPOOF` | **Passed** (Attack Rejected) |
| **Case D** | Empty camera scene | Labeled `LOOKING FOR FACE` | **Passed** (Idle State) |
| **Case E** | Multiple faces in frame | Bounded fail-closed; labeled `MULTIPLE FACES` | **Passed** (Safety Rejection) |

## System Scope & Considerations

- **Supervised Application**: This system is designed for supervised or kiosk-style attendance where an attendant or operator is present, rather than high-security unsupervised access control.
- **RGB Anti-Spoofing Characteristics**: While effective against standard 2D smartphone and print attacks, 2D RGB models can be challenged by high-end 3D silicone masks or extreme lighting variations. Production deployments in unsupervised areas should supplement this with active challenge-response prompts (blinking, head turns) or multi-spectral hardware (IR / Depth).
- **Inference Optimization**: ArcFace recognition accounts for ~68% of compute time (81.5 ms on CPU). For deployment to low-power embedded edge devices (e.g., Raspberry Pi), INT8 dynamic quantization or OpenVINO execution providers are recommended next steps.

