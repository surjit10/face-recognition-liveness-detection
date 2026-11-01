# FaceGuard AI v2 — Implementation Contract
**Version:** 1.2 | **Date:** 1 November 2026 | **Purpose:** A single, defensible Computer Vision + MLOps portfolio project for CloudPhysician-style CV Engineer interviews.

> **Status:** Architecture and execution specification. It is NOT a claim that code has been built, datasets obtained, accuracy measured, or package combinations tested on your Kaggle image. Numbers marked *starting point* must be validated. Do not fabricate experimental results.

> **Companion controls:** Start at `reference/README.md`. Implementation progress is controlled by `reference/tracking/progress.yaml` and summarized in `reference/tracking/STATUS.md`; update both under `reference/governance/TRACKER_POLICY.md`. These files do not weaken any gate in this contract.

> **Strict execution rule:** This document is complete as a process contract, not as a source of invented experimental facts. A coding agent must use the exact value written here, discover and record the value from the authorized environment/data, or stop with the named blocker. It must never silently choose a package version, dataset meaning, identity split, label mapping, model artifact, threshold, permission, metric, or result.

## 0. Execution control and non-negotiable design decisions

### 0A. Authority order and no-assumption protocol

When instructions or evidence disagree, use this order:

1. The user's latest explicit instruction for this project.
2. Legal, consent, privacy, dataset-license and platform restrictions.
3. Verified files and machine-readable manifests in the current workspace.
4. This implementation contract.
5. Official upstream documentation for the exact installed/tested release.
6. Existing legacy code and reports, treated as historical evidence rather than truth.

The agent MUST NOT resolve a contradiction by guessing. It must quote the conflicting evidence, mark the relevant item `BLOCKED`, and request the smallest necessary decision. A filename, directory name, comment, old README, approximate count, nearest class, default library behavior, successful import, or plausible internet example is not proof of semantics or correctness.

Every material value has one of four states:

- `UNRESOLVED`: not yet known; it cannot be consumed by training, evaluation, or release code.
- `DISCOVERED`: observed automatically but not yet semantically confirmed.
- `VERIFIED`: checked against data, official documentation, permission record, or a successful test.
- `FROZEN`: approved for a named experiment/release and protected by a checksum or immutable manifest.

Only `VERIFIED` values may enter an experiment. Only `FROZEN` values may enter a release. Placeholder strings such as `TODO`, `AUTO`, `LATEST`, `CALIBRATE`, `GENERATE`, an empty hash, or a guessed threshold make release validation fail.

### 0B. Required decision register

Create `configs/decisions.yaml` during P0. Do not continue past a phase gate if one of its required decisions remains unresolved. Each entry has this exact information:

```yaml
schema_version: 1
decisions:
  - id: FRAD_COHORT
    state: UNRESOLVED        # UNRESOLVED | DISCOVERED | VERIFIED | FROZEN
    value: null
    evidence: []             # workspace paths, command report, official URL, or user decision
    decided_by: null         # user | audit | official-documentation | measured-validation
    decided_at: null         # ISO-8601 with timezone
    applies_to: []           # experiment/release IDs
    notes: ""
```

At minimum track: `FRAD_PERMISSION`, `FRAD_COHORT`, `FRAD_LABEL_MEANING`, `FRAD_SESSION_POLICY`, `FRAD_SPLIT`, `PAD_DATASET_PERMISSION`, `PAD_LABEL_MAPPING`, `PAD_SPLIT`, `KAGGLE_ENVIRONMENT`, `LOCAL_RUNTIME_ENVIRONMENT`, `INSIGHTFACE_PACK`, `PAD_TRAINING_CONFIG`, `PAD_THRESHOLD_POLICY`, `RECOGNITION_THRESHOLD_POLICY`, `TARGET_HARDWARE`, and `PUBLICATION_POLICY`.

### 0C. Commands and actions the agent may not improvise

- Do not upload biometric data, create a public Kaggle dataset/notebook, publish a repository, or send artifacts to a remote service without explicit authorization.
- Do not install, upgrade or downgrade packages before recording the preflight environment. Never use an unbounded `latest` dependency or an undocumented model mirror.
- Do not rename/relabel identities, infer real names, infer session/device/pose from filenames, delete samples, or repair images silently. Quarantine through a manifest with a reason.
- Do not use test data for model selection, threshold selection, early stopping, normalization statistics, quantization calibration or debugging decisions.
- Do not convert a failed check into a warning merely to continue. Security/privacy/model-hash/label/split failures are hard failures.
- Do not report a metric without its dataset manifest hash, split hash, model hash, threshold policy, sample unit and confidence interval or an explicit reason the interval is unavailable.
- Do not claim implementation completion from generated source alone. Completion requires the phase-specific commands, tests and artifacts listed below.

### 0D. Non-negotiable design decisions

1. **Keep one project**: evolve the original Face Recognition + Anti-Spoofing system, preserving the published FRAD work as v1, not inventing an unrelated project.
2. **Production stores only enrolled identity IDs and protected facial embeddings** (plus model metadata and aggregate/non-identifying operational metrics). Raw biometric frames/crops exist transiently in memory for processing and are not retained by default. Do not persist facial photographs, video, screenshots, video-derived image crops, or embeddings in logs or MLOps dashboards. Enrollment/test/training datasets are separately governed offline research assets; handle consent and license requirements before uploading to Kaggle.
3. **One primary network fine-tuning task:** a MobileNetV3-Small RGB presentation attack detection (PAD) classifier. Use a pretrained SCRFD + ArcFace ONNX model pack for detection/recognition. No ArcFace re-training and no trainable tracking network.
4. **Kaggle GPU does heavy offline operations:** PAD training, large-dataset preprocessing when beneficial, batch inference/evaluation, FRAD enrollment embedding extraction (ONLY if dataset consent and access rules authorize cloud transfer), ONNX export and parity testing. The deployed laptop runs inference only. Webcam/RTSP live UI is local, not hosted on ephemeral Kaggle.
5. **Compatibility-first:** do not upgrade Kaggle's preinstalled torch / torchvision / CUDA or install multiple ONNXRuntime flavors blindly. Preflight first; pin and lock the tested environment. Separate Kaggle training env from local CPU runtime. Use stable compatible releases, not blindly `pip install -U` on everything.
6. **Fail closed:** no/multiple/low-quality face, unknown identity, ambiguous PAD, missing model, stale frames, or unsupported input => UNKNOWN / RETRY / REJECT; never silently grant access.
7. **No unsupported anti-spoof claims:** RGB PAD + multi-frame score smoothing is not a defense guarantee against masks, deepfakes, video injection or adaptive replay; document test scope.
8. **Keep code minimal:** start with OpenCV, InsightFace `FaceAnalysis` (tested version) for detection+recognition, PyTorch/torchvision for PAD training, ONNX Runtime for exported PAD, NumPy for similarity, a simple tracker, SQLite or encrypted file gallery, MLflow + DVC + Docker/CI. No Kafka, Kubernetes, Redis, vector database, distributed workers, custom CUDA, or multiple dashboards.

### 0E. Academic-prototype PAD evidence amendment (1 November 2026)

The user explicitly removed reproduction of the third-party online PAD training run
from the final academic-prototype requirements. The frozen MobileNetV3 PAD artifact is
accepted only as an externally trained development baseline using the evidence that is
already available: the executed Kaggle notebook, downloaded artifact/config/metric
hashes, local ONNX verification, target-camera rejection test and mandatory limitation
reporting.

This amendment means:

- the project is not required to obtain or redistribute the online CelebA-Spoof mirror,
  recover its private manifest CSVs, recover training history/scores, reconstruct its
  package lock, or rerun training;
- P2 training-reproducibility tasks remain visibly `BLOCKED` and receive no completion
  credit; they are not silently converted into successful evidence;
- the recorded threshold and metrics may be reported only as historical properties of
  the frozen development artifact, not as independently reproducible or certified PAD
  results;
- the application remains eligible only for a supervised, non-commercial academic
  demonstration under `PROTOTYPE_SCOPE`; production, commercial, certified-security or
  dataset-redistribution claims remain prohibited;
- reopening production or reproducible-research scope automatically reactivates every
  missing P2 permission, manifest, environment, score, policy and parity gate.

## 1. Inputs from your existing v1 project

**Available according to your September 2026 interview report:** a 233-subject FRAD face dataset with roughly 12 genuine images per identity and corresponding screen-recapture spoofs; MTCNN preprocessing; FaceNet embeddings + 233-class RBF-SVM; DenseNet121 RGB PAD; webcam/Tkinter and Raspberry Pi implementations. Original reported results: 90.27% *closed-set* recognition on the 233-subject configuration, and 95.5% PAD accuracy on a small, specific test of screen-recapture samples. These are *historical*, protocol-specific results, not guaranteed cross-domain or v2 results. The old evaluation scripts/report have known mismatches, so reproduce baselines before comparing.

**Observed workspace audit on 24 October 2026 (structural facts only):**

- Recognition root after repository organization: `reference/archive/final-submission-excluded/restricted-data/processed/frad_legacy_crops/` (moved from `dataset/cropped_faces/` without changing file contents; 2,802-file content multiset SHA-256 `cf24a522c45d81014367897e6a198493d7ea9a7fefb87285becf06ec5aeb269c`).
- 234 identity-like directories named `0001` through `0234`; directory names are pseudonymous candidate labels, not verified real identities.
- 2,802 files identified as JPEG by file-signature inspection; every file reports 256×256 dimensions. A full decoder/model audit is still required.
- 232 directories contain 12 images, `0183` contains 10, and `0234` contains 8.
- One exact byte duplicate exists inside `0164`: `EEN234033.jpg` and `EEN234033 - Copy.jpg`.
- The files appear to be legacy cropped-face inputs. This audit does not prove that every crop contains exactly one correct face, that two folders never contain the same person, that filenames encode sessions, that the images are all bona fide, or that use/upload permission exists.
- No PAD annotations, attack-type labels, source-frame manifest, session manifest, permission record, fixed recognition split, or external unknown-person probe set was found alongside this folder during the audit.

Therefore **233 is a historical v1 cohort size, while 234 is the currently discovered folder count**. Neither is the v2 release cohort until `FRAD_COHORT` is `VERIFIED` and a frozen manifest lists every included/excluded file. Never change historical 233-class statements, but derive all v2 counts from the frozen manifest rather than hard-coding either number.

**Retain:** source code in `legacy/` or a stable v1 tag, dataset provenance, FRAD publication/citation, screenshots of original architecture if permitted, original metrics labeled historical. **Do not reuse:** untuned SVM threshold 0.30 as a security guarantee; FaceNet vectors as though interchangeable with ArcFace vectors; the old screen-only PAD training split as a cross-attack benchmark.

## 2. Problem statement, boundaries, success criteria

**Problem:** Verify whether an **opted-in, locally-present person** in a live/recorded camera stream is both (a) a bona fide presentation according to the RGB PAD model and (b) an enrolled identity, with known limits on false accepts, false rejects, latency, and resource usage. Distinguish *identity verification* from public surveillance: the target is a single-user consent-based verification station; not crowd identification, cross-camera surveillance, demographic inference, or access control certified for high security.

**Core:** webcam/MP4; 1 face (reject >1 in verification mode); face detection/alignment; PAD; ArcFace embedding + cosine matching; UNKNOWN; optional in-session tracking; diagnostic panel; enrollment/update/delete; train/eval/ONNX; MLOps artifacts. **Optional after core:** RTSP, two camera inputs, challenge-response studies. **Out of scope:** mask/deepfake certification, always-on identity tracking through environments, physiological remote PPG, multi-camera re-ID, automatic retraining on guessed labels, emotion/age/gender estimates.

**Release proof:** code + manifest + held-out subject-separated evaluation + spoof breakdown + ROC/threshold evidence + local latency measurements + reproducible training run + tested model artifact, not unsupported numerical targets.

## 3. Concrete models (do not mix model configurations)

| Branch | Recommended | Data needed | Train? | Compatibility rule |
|---|---|---|---|---|
| Face detection & landmarks | SCRFD from the documented InsightFace `buffalo_m` ONNX pack (SCRFD-2.5GF) | Images/videos for evaluation | No | Use exact pack and wrapper; check model manifests and output API in chosen release |
| Face alignment & embeddings | ArcFace-style ResNet50 recognition in **the same** `buffalo_m` pack | FRAD genuine images | No | Re-embed **all** gallery images with this model; store model hash; do not mix feature spaces |
| PAD baseline | Existing DenseNet121, historical; optional pretrained MiniFASNetV2 baseline | Existing PAD test / new test | No new training | Never import legacy dependencies into new training env |
| PAD candidate | `torchvision.models.mobilenet_v3_small(weights=...IMAGENET1K_V1)` with a new single-logit classification layer | CelebA-Spoof subset | **Fine-tune** | Only train this neural net; use same preprocessing at training and inference |
| Tracking | Low-complexity IoU + periodic detection and short-lived state | Sequential video | No | No ByteTrack dependency unless demonstrated necessary |
| Recognition gallery | L2 normalized embeddings + vectorized NumPy dot product | Verified FRAD enrollment split | No | At the currently observed 234-folder scale a vector database is overengineering; derive final `N` from the frozen manifest |

The InsightFace project distributes model packs under **non-commercial research** terms, regardless of its package-code license. Do not claim commercial rights. The official docs describe `buffalo_m` as SCRFD-2.5GF + ResNet50, and `buffalo_sc` as a smaller but lower-quality alternative. Start with `buffalo_m` for a consistent and explainable detector/recognizer; switch to `buffalo_sc` only after an accuracy/latency comparison. Download pack once, pin its SHA-256, and never silently auto-download different models on production startup.

**Why not train ArcFace?** 233 identities with ~12 images each is too small to train a discriminative face backbone from scratch. Pretrained embeddings plus a calibrated gallery give meaningful verification without 233-way softmax retraining. Why fine-tune PAD? It is domain/attack dependent and the FRAD PAD attack coverage is weak.

## 4. Required datasets and provenance

| Dataset | Scope | Purpose | Important restrictions |
|---|---|---|---|
| FRAD candidate recognition images (currently 234 folders / 2,802 cropped JPEGs; v1 historically used 233 IDs) | Small identity data | Offline embed gallery and recognition evaluation after audit | Verify identity semantics and consent; resolve the incomplete folders and duplicate through the manifest; do not put participant images in a public repo |
| FRAD spoof portion | Narrow screen recapture | Historical reference and optional strictly held-out diagnostics | **Not** principal PAD training corpus |
| CelebA-Spoof | ~625k labeled RGB images/10,177 subjects | PAD train/validation/test with subject-disjoint splits, start with subset | Non-commercial research; no redistribution of images or derived copies; verify access and metadata parsing |
| Idiap Replay-Attack | 1,300 video clips, 50 clients, disjoint official train/devel/test | External RGB video attack evaluation (photo/video) | Acquire under official terms; use official splits; beware it is an older/lower-resolution dataset |

Links: https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html ; https://www.idiap.ch/en/dataset/replay-attack ; https://github.com/deepinsight/insightface ; https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html

### Data access gate before coding models

- Confirm FRAD `person_id`, `image_path`, genuine/spoof label, `session_id`/capture device if available, ownership/license/consent, correct count, corrupt/missing files.
- Treat folder name as `candidate_person_id` only until validated. Do not infer `session_id`, device, pose, genuine/spoof state, or capture independence from the filename. Preserve the relative path and SHA-256 in the audit manifest.
- Do not silently delete the duplicate or exclude `0183`/`0234`. Record `include`, `exclusion_reason`, and evidence for every file. Require a user/data-owner decision if cohort membership cannot be established from authoritative records.
- If FRAD genuine images cannot legally be moved to Kaggle, **run embedding extraction on authorized local hardware** as a one-time batch; privacy outranks GPU convenience. No gallery image uploads to public Kaggle notebooks or Kaggle public datasets.
- Verify CelebA-Spoof official images + labels actually accessible and parsed; check subject IDs, spoof types, real label semantics, duplicate/near-duplicate image paths, basic class distributions. DO NOT substitute unlabeled third-party mirrors and invent labels.
- Check Replay-Attack video availability, codec, official `train/devel/test` manifest and permissions before promising cross-dataset numbers.
- Store immutable `data_manifest.json` with dataset source, license, number of images/videos/people, split hashes, timestamps, labels and excluded sample reasons. Prefer source metadata + checksums over copying restricted datasets to DVC remote or public artifacts.

### Splitting protocols — different for each task

**Recognition:** choose one explicit protocol, never random-frame leakage. The former 180/25/28 example is **not authorized for execution**, because the workspace currently has 234 candidate folders and no verified session manifest. Generate candidate splits only after the cohort is verified. Freeze explicit person IDs and image hashes into enrollment, validation-probe, test-probe, validation-impostor and untouched-test-impostor manifests. Known-person enrollment/probe images must be distinct captures; group by verified session/device when those fields exist. Use enrollment samples to make templates; choose the threshold only from known-validation probes plus validation-impostor identities; evaluate known-test probes plus unseen test-impostor identities once with the frozen threshold. Report the exact gallery size `N`, never “233-person” or “234-person” merely from a directory count. A later all-subject demo needs an external, consented unknown-person set and fresh calibration because maximum impostor similarity changes with gallery size. If session independence cannot be verified, label the experiment “same/unknown-session exploratory evaluation” and do not claim cross-session generalization.

**PAD:** use subject-disjoint splits from CelebA-Spoof. Recommended starting subset: ~40k–100k balanced-by-identity-and-attack images if accessible; do not treat frame/image random sampling as a proper person split. The canonical test partition is untouched by hyperparameter search. Keep source spoof type and lighting categories for metrics. Calibrate PAD threshold on *validation* only. Freeze preprocessing and threshold before external Replay-Attack test. For Replay-Attack, report untouched cross-dataset test and separate (optional) protocol-adapted test if you tune on its development set; never call adapted evaluation fully zero-shot.

## 5. Two distinct workflows

### 5A. Offline on Kaggle (training/evaluation and identity gallery)

```
PRIVATE/APPROVED FRAD GENUINE DATA
   -> validation/audit -> same face-detection/alignment contract
   -> batch ArcFace ONNX inference (GPU if verified)
   -> L2-normalized enrollment embeddings -> protected gallery export
   -> genuine/impostor similarity calibration -> recognition evaluation

AUTHORIZED CELEBA-SPOOF DATA
   -> annotation audit -> subject-level split manifests
   -> PAD crops and train-only augmentation -> MobileNetV3 fine-tuning (Kaggle GPU)
   -> choose checkpoint/threshold from validation PAD metrics
   -> compare historical DenseNet / optional MiniFASNet baseline
   -> untouched PAD test + external video testing
   -> ONNX export -> PyTorch/ONNX parity -> release artifact

RUN METADATA -> MLflow / split hashes -> release manifest -> local deployment
```

### 5B. Local runtime (no training, no persisted photos)

```
Camera/MP4 decode -> timestamp -> bounded latest-frame queue (live)
  -> SCRFD detect + 5 landmarks -> exactly 1 suitable face?
  -> IoU association and quality gate
  -> PAD expanded crop -> ONNX PAD score across eligible observed frames
  -> median / robust aggregation with minimum evidence and max-gap policy
  -> if weak/ambiguous/replay: REJECT / RETRY
  -> else aligned face -> ArcFace embedding (one/few quality frames)
  -> L2 normalize -> frozen `N`-person gallery cosine matching
  -> if score >= calibrated identity threshold: VERIFIED, else UNKNOWN
  -> session event (status, time, model version; no biometric images)
  -> optional dashboard (processed FPS, age, p50/p95 latency, drops)
```

An additional temporal aggregator stabilizes a frame classifier; it **is not a trained temporal liveness detector**. It does not guarantee resistance to replayed videos or digital camera injection.

## 6. Recognition enrollment & privacy contract

- Detect EXACTLY one high-quality face in each permitted enrollment sample; if no face, multiple faces, tiny face, gross misalignment, or excessive blur, reject sample.
- SCRFD returns 5 landmarks; align crop to the ArcFace model-specific template (normally 112×112, inspect actual model input and preprocessing). Let InsightFace's tested wrapper do this initially; do not copy ad hoc crop code that changes embedding distributions.
- Compute float32 embedding `e` with the selected ArcFace pack, `e = e / max(norm(e),eps)`.
- Store **individual** permitted embeddings `[N,D]` (e.g. <= 12/person) and labels `[N]` in a protected local gallery. Optionally store normalized person prototypes `mean(e_i)/norm(mean(e_i))` for faster comparisons. Both are evaluated; avoid trusting prototypes blindly under pose variation.
- One 512-d float32 vector ~2 KB. 2,796 vectors ~5.46 MiB before labels/DB overhead, modest for NumPy. Compute `scores = gallery_matrix @ query_embedding` as one vectorized operation. Map nearest vector to ID; optionally reduce per-person with max/centroid fusion.
- **Never identify by name just because it is nearest.** Choose threshold on genuine/impostor validation scores for a stated false-match objective. Document whether false match rate is per comparison or open-set FPIR per search. Reject ambiguous/near-tied candidates where policy demands it.
- Store {person_id (pseudonymous), model_sha256, embedding float32, creation time, consent status, active flag}. Protect at rest by OS permissions + encryption where feasible; embeddings are biometric data and are NOT anonymization. Provide deletion: delete all vectors for an ID, clear cache, recompute gallery matrix. Never log vectors or encode them in public MLflow artifacts.
- If embedding model or aligner changes, ALL enrollment vectors must be regenerated under exactly one model version. On change, invalidate old thresholds and rerun calibration.
- Match is **identification** if searched across multiple IDs; do not call it 1:1 verification unless a claimed identity is provided and that exact ID is compared. The demo can call the open-set result 'identified enrolled / unknown', then optionally support one-to-one verify mode.

## 7. PAD training contract — one meaningful experiment

**Backbone:** torchvision MobileNetV3-Small pretrained on ImageNet; replace last linear classification output with 1 logit. Use `BCEWithLogitsLoss`, with `y=1` for bona fide and `y=0` for attack. At runtime sigmoid maps logit to `p_live`; do not call uncalibrated sigmoid a guaranteed real-world probability. Use PAD-specific expanded face crop (starting crop factor 1.3×), RGB, resize 224×224 and the chosen torchvision normalization. Train and deploy with EXACTLY the same color order, resize, crop expansion and channel normalization. Metadata must record it; face-recognition crops should be tighter and aligned separately.

**Keep model selection small:**
- E0: pretrained MiniFASNetV2 (optional baseline if its old code/checkpoint loads successfully in an isolated environment; its training labels/crop conventions may be 3-class, not naive binary). Do not let its legacy requirements overwrite your main environment.
- E1: ImageNet-pretrained MobileNetV3; frozen features + train only new PAD head, ~3–5 epochs (starting point).
- E2: fine-tune last MobileNet feature blocks + head for up to ~10–20 more epochs with differential LR and early stopping (starting point). Usually E2 is final candidate.
- Historical DenseNet result: cited separately unless exact samples/protocol are reproduced; old 95.5% is not directly comparable to public-dataset test results.

**Optimizer/training defaults (tune with validation, not test):** AdamW; head LR `1e-3`, backbone LR `1e-4` or `3e-5`; weight decay `1e-4`; batch 32 or 64 based on Kaggle GPU memory; cosine or ReduceLROnPlateau schedule; early stopping patience 3–5; seed 42; balanced sampler/class-aware sampling when needed; AMP on supported CUDA (`torch.autocast`, scaler for float16); `zero_grad(set_to_none=True)`; `pin_memory=True`, `persistent_workers=True` if `num_workers>0`; dataloader worker count typically 2–4 and benchmark. Pretrained backbone normalization must be kept consistent.

**Training augmentations:** horizontal flip, mild brightness/contrast, JPEG-like compression and resizing. Keep attack-discriminative textures intact; avoid heavy rotations, mosaic, image erosion, unrealistic blur, or artificial watermarks that give shortcuts. Evaluate separate camera/device and spoof-type breakdowns. No augmentations on val/test.

**Early-stage sanity gate:** overfit a tiny authorized sample (e.g. 32 or 64 examples) to verify trainable logits/labels; then do a 1-epoch partial run. Plot learning curves and inspect false accepts/false rejects BEFORE long training. Catch swapped labels and normalization errors early.

**Selection:** primary PAD metrics at a *validation-calibrated operating point*: APCER, BPCER and ACER, plus ROC-AUC and per-attack APCER. Remember APCER = fraction of attacks classified bona fide; BPCER = fraction of bona fide classified attacks; ACER = (APCER+BPCER)/2 at the same operating point. Higher spoof false accepts are more serious for verification, so choose threshold subject to a predeclared validation APCER bound if feasible; if unmet, do not claim safety. Report confidence intervals if sample size supports them.

**External video test:** apply same crop/preprocess and frozen PAD model to Replay-Attack frames. Group by video, aggregate scores without tuning on the untouched external test; report video-level as well as frame-level results, separated by photo/video attack subtype. If later adapting thresholds on Replay-Attack development split, label it protocol adaptation and keep test untouched.

## 8. Lightweight real-time video architecture

**Reason for separate worker:** capture can be faster than infer, so synchronous reading creates stale displayed frames or uncontrolled latency. Implement a camera producer with monotonic capture timestamps, bounded `Queue(maxsize=2)` in local live mode, and one inference consumer. Under load, drop the **oldest waiting frame**, never a frame currently in processing; count drops. For prerecorded *accuracy* evaluation, do not drop: process frames sequentially in source order. Distinguish throughput from camera-paced FPS.

**Video quality and tracking:**
- Step 1: use detection at configurable frame cadence. Prefer detect on each *processed* frame initially; optimize only after measuring, since full detection may be critical for face security.
- Step 2: minimal IoU association with a previous single-face box, restricted temporal gap (e.g. 250–500ms starting point). Reinitialize rather than merge uncertain tracks. No multi-person association in the minimum verification release.
- Step 3: quality: detection confidence, minimum face size, variance-of-Laplacian blur indicator, simple exposure check; calibrate cutoffs on examples, not universal magic numbers. Require no second detectable face in frame.
- Step 4: PAD inference on sampled eligible frames (e.g. 8 eligible observations as initial parameter), associate to same verified continuous track and require no major gaps. Use robust median of `p_live`; threshold calibrated on PAD validation. Reject/retry if insufficient evidence.
- Step 5: ArcFace embedding on only a good frame after PAD passes; optionally average normalized embeddings from 2–3 quality frames if validation proves benefit. Limit identity recomputation within same session, but invalidate cached identity if tracking continuity or PAD confidence breaks.
- Step 6: no frame images/video stored in SQLite or logs; event record is `{event_id,time,result,person_id_if_verified,latency_ms,model_version}`. Use pseudonymous IDs and configure retention/access controls.

**Failure behavior:** missing frame -> retry; multiple faces -> reject; no consistent track -> retry; spoof -> reject; unknown -> unknown; backlog -> drop stale live waiting frames; model missing/hash mismatch -> do not start; stream ends -> completed; device lost -> reconnect with new session; ONNX conversion regresses -> keep previous model.

**Benchmarks:** measured on local CPU with system info: decode time, detection+alignment, PAD, embedding, matching, end-to-end p50/p95 and warmed FPS; frame age, memory RSS, dropped-frame count; stress with a prerecorded video paced in real time. Keep CPU threads configurable; avoid oversubscribing OpenCV and ONNX Runtime simultaneously. Benchmark FP32 ONNX before experimenting with int8 (static quantization is generally the first CNN quantization choice; run calibration + parity + whole-pipeline accuracy checks).

## 9. Runtime interfaces; optimize for maintainable low LOC

One facade per stage, no proliferation of base classes:

```python
@dataclass(frozen=True)
class FaceObservation:
    timestamp_ms: float
    bbox_xyxy: tuple[float, float, float, float]
    landmarks_5: np.ndarray
    detect_score: float
    # Optional quality fields; avoid retaining raw frame in persistent records

@dataclass(frozen=True)
class VerificationResult:
    status: Literal['verified', 'unknown', 'spoof', 'retry']
    person_id: str | None
    similarity: float | None
    pad_score: float | None
    track_id: int | None
    model_version: str

class FaceAnalyzer:
    def detect_and_embed(self, frame: np.ndarray): ...   # tested InsightFace wrapper

class PadModel:
    def score_live(self, expanded_crop: np.ndarray) -> float: ...  # ONNX

class Gallery:
    def enroll(self, person_id: str, embeddings: np.ndarray): ...
    def search(self, embedding: np.ndarray) -> tuple[str | None, float]: ...
    def remove(self, person_id: str): ...

class VideoVerifier:
    def process(self, frame: np.ndarray, timestamp_ms: float) -> VerificationResult: ...
```

Production contract: labels are never guessed; refuse missing or ambiguous face; externally visible scores are for transparency, not guarantees. Recognizer should not be invoked as an acceptance decision before PAD gate. The `FaceAnalyzer` may compute embeddings eagerly due to library API; if so, do not use them for a decision until PAD passes; benchmark compute cost and refactor for true liveness-first lazy recognition if needed.

**Short pseudocode of stateful inference:**

```
frame = live_queue.get_latest()
faces = face_detector(frame)
if len(faces) != 1 or not quality_ok(faces[0]): RETRY
track = update_single_face_tracker(faces[0], timestamp)
if not track.continuous: RESET_AND_RETRY
p_live = pad_model.score_live(expand_crop(frame, faces[0].bbox))
track.add_pad_evidence(p_live, timestamp)
if track.need_more_evidence: RETRY
if track.pad_aggregate < tau_pad: SPOOF
embedding = arcface(aligned_crop(frame, faces[0].landmarks))
id, sim = gallery.best_match(l2_normalize(embedding))
return VERIFIED(id) if sim >= tau_id else UNKNOWN
```

For a normal recognition match call, `tau_id` must be calibrated for the exact model/gallery and target error budget; `tau_pad` for the PAD model/crop/test policy. Do NOT use arbitrary values like 0.6/0.3 as final safety cutoffs.

## 10. Compatibility-first environment installation strategy (critical)

### Kaggle preflight *before any pip installs*

```python
import os, sys, platform, importlib.metadata as md
print('python:', sys.version)
print('platform:', platform.platform())
for name in ['torch','torchvision','numpy','opencv-python','opencv-python-headless',
             'onnxruntime','onnxruntime-gpu','insightface','mlflow','dvc']:
    try: print(name, md.version(name))
    except md.PackageNotFoundError: print(name, '<not installed>')
import torch, torchvision
print('torch / torchvision:', torch.__version__, torchvision.__version__)
print('CUDA built:', torch.version.cuda, 'GPU visible:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
```

After this run: `python -m pip check`. Record `pip freeze`, Python version, NVIDIA driver (`nvidia-smi`), Kaggle image revision when visible, CUDA/cuDNN and GPU type in `reports/environment_kaggle.json`. Kaggle notebook versions capture Docker image state; a rerun six months later may not have the same packages.

**Compatibility rules:**
- Do not force a different `torch` and `torchvision` over Kaggle's already working pair. If they fail import or a specific feature is absent, install an **official documented matching pair** inside one isolated env; e.g., PyTorch 2.10.0 with torchvision 0.25.0 is a documented pair, *not a mandate to downgrade a newer working Kaggle image*. Select official CUDA wheel index for that GPU/driver. `torchvision` matching is strict.
- Do not blindly install `onnxruntime` and `onnxruntime-gpu` together in one env. **Local CPU**: only `onnxruntime`. **Kaggle GPU**: only `onnxruntime-gpu` if CUDA provider is needed; match ONNXRuntime's documented CUDA and cuDNN binaries to PyTorch runtime; explicitly verify `CUDAExecutionProvider` or fall back to CPU with a logged warning, not a fake GPU-speed claim.
- If ONNX GPU compatibility is fragile, train with PyTorch on Kaggle, export ONNX there, then verify ONNX on local **CPU**. There is no need to run GPU ONNX as a hard requirement for training.
- Prefer `opencv-python-headless` on Kaggle/CI/server and exactly one OpenCV pip flavor in any environment. Local GUI OpenCV can use `opencv-python`, not both.
- Avoid legacy TensorFlow/Keras, Caffe, old `keras-facenet`, and the legacy MiniFASNet repository requirements in the new main env. Use isolated baseline env only when needed.
- Pin actual tested versions (`pip freeze` + constraint/lock file) **after a smoke test**, rather than guessing one permanently correct matrix. Use Python 3.11 for a new local environment when possible; if Kaggle ships a different working Python, retain it for the notebook. A binary build must have matching Python/OS/GPU wheel availability.
- Prefer stable, documented official APIs. For MobileNetV3 `torchvision` weights enum; PyTorch AMP use `torch.autocast` and `torch.amp.GradScaler` as needed; PyTorch ONNX export use `torch.onnx.export(..., dynamo=True)` when supported, else a documented tested export fallback. Run full numerical parity tests.

**Install groups, not one giant requirements file:**

```
requirements-runtime.in: numpy, opencv-python, onnxruntime, insightface,
  pydantic (optional), PyYAML, psutil
requirements-train.in: (compatible Kaggle torch+torchvision), numpy, Pillow,
  opencv-python-headless, scikit-learn, onnx, mlflow, dvc, PyYAML
requirements-dev.in: pytest, ruff
```

These are package **intent lists**. Generate separate fully pinned `requirements-*.lock` after smoke-testing each environment. If directly using ONNX models without InsightFace, remove InsightFace from runtime only after detector/NMS/alignment equivalence passes. Do not bring training-only MLflow/DVC/torch into the CPU runtime container.

**Smoke tests required:** import torch/torchvision and pretrained MobileNet; check CUDA; read 5 dataset images; run one forward+backward step; run ONNX export+load+same-input parity; initialize InsightFace specific pack, check returned face bbox/landmarks/embedding dimensions; run a tiny video through the full pipeline; `pip check`; freeze lock files. If any fail, stop and repair the environment before training on full data.

## 11. Kaggle execution plan — fastest safe path

**Do not make Kaggle notebooks the only source of code.** Keep `src/` training functions importable, and use notebooks just as orchestrators. Kaggle inputs (attached datasets) are read-only; outputs under `/kaggle/working/`. Never repeatedly extract large ZIPs on every epoch; cache prepared metadata/crops in suitable local scratch only within quotas; use a manifest of paths. Never publish notebooks containing private FRAD images or gallery embeddings.

**K0 Environment:** preflight; detect GPU, run basic PIL/cv2 + torch and torchvision checks. If torch works, leave it unchanged. Record versions; attach official dataset only after access checks.

**K1 Data audit:** parse authorized CelebA-Spoof metadata, check label semantics, subject split, detect/crop contract and failure rates. Sample a moderate subject-balanced subset; create `train.csv`, `val.csv`, `test.csv` with ids and labels; hash manifests. Print per-split counts and attack types.

**K2 PAD sanity:** choose backbone; head-only run 1 epoch subset (e.g. 2k–5k) to test convergence, no data leakage; verify GPU utilized. Use pinned `torchvision` model code.

**K3 PAD training:** initial 40k–100k sample subset, mixed precision, minibatches, best validation checkpoint saved every epoch. Training can use temporary on-disk transformed crops where lawful and disk permits; on-the-fly lightweight transforms are preferable to duplicating entire datasets. Prefer 1–2 well-designed model experiments over giant sweeps.

**K4 PAD evaluation and export:** finalize validation threshold, test on untouched subjects, report APCER/BPCER/ACER + ROC AUC and detailed confusion by attack/camera; run Replay-Attack external evaluation if available. Export chosen checkpoint as ONNX on a fixed input tensor; compare outputs on 100+ representative inputs (max abs error, label agreement, threshold flip rate and PAD metrics), record supported opset and SHA-256. Convert only final model; avoid quantization until performance benchmark.

**K5 FRAD identity batch extraction (permission-gated):** load selected model pack ONCE, batch/process approved FRAD images, fail or quarantine multiple/invalid faces, persist only normalized embeddings + pseudonymous IDs + pack hash + split metadata to a PRIVATE artifact; evaluate held-out known probes and disjoint unknown identities; export gallery/templates only when allowed. No public Kaggle sharing.

**K6 Artifacts:** export best PAD ONNX, config/normalization contract, optional best `.pt` checkpoint for reproducibility, hashes, metrics, PR/ROC curves (non-biometric), splits/metadata, environment snapshot, MLflow offline store (without raw image samples), and a human-readable model card. Local app consumes ONNX, gallery, thresholds, signed/checksum manifest.

**Optimization before longer runs:** profiling per epoch; dataloader CPU time vs GPU time; cache only permitted derived images, mixed precision, avoid growing tensors retained with autograd, `torch.no_grad()`/`inference_mode()` for extraction, vectorize gallery math, no loop that instantiates detector/model per sample. Use `Save & Run All` for snapshot and durable outputs, not an interactive-only notebook state.

## 12. MLOps / release engineering: minimal but genuine

**DVC:** use to version code-linked data manifests and reproducible dataset preparation/training stage dependencies. For non-redistributable CelebA/FRAD, **do not publish data caches/derived crops**; store checksum manifest + source instructions in Git, and protected DVC remote if licenses and consent allow. DVC cannot authorize redistribution by itself.

**MLflow:** offline/file or SQLite-backed tracking runs on Kaggle. Log model name, backbone/head LR, seed, split hash, batch, optimizer, augmentation policy, epoch learning curves, val ACER/APCER/BPCER/AUC, checkpoint hash, device and library versions, ONNX export validation metrics. DO NOT log biometric input images/embeddings, unknown-person names, or sensitive demographic annotations.

**Release registry (simple JSON, no registry server required):**

```json
{
  "release_id": "faceguard-v2.0.0",
  "detection_recognition_pack": "buffalo_m",
  "recognition_model_sha256": "GENERATE_FROM_ACTUAL_MODEL",
  "pad_model_sha256": "GENERATE_FROM_ACTUAL_MODEL",
  "pad_input": {"height":224,"width":224,"channels":"RGB","normalization":"imagenet","crop_scale":1.3},
  "pad_threshold": "CALIBRATE_ON_VAL",
  "recognition_threshold": "CALIBRATE_ON_VAL",
  "gallery_embedding_model_sha256": "MUST_MATCH_RECOGNIZER",
  "data_manifest_hash": "GENERATE",
  "benchmark_report": "reports/benchmark.json",
  "state": "candidate"
}
```

These strings are placeholders and must not be used as final thresholds/hashes. Promote only if actual hashes, schema, model load test, output parity, error rates, and local latency pass predeclared regression gates. Keep previous known-good release; rollback by switching release symlink/config atomically. Never hot-swap weights mid-session: start a new session.

**CI:** `ruff check`, `pytest` with synthetic face/no-face fixtures, contract tests, dataset manifest logic, gallery matching (normalization and threshold), quality/invalid input, queue overflow policy, legacy fallback, ONNX file load against a tiny authorized test input, and Docker build. Heavy training remains on Kaggle; CI must not download restricted biometric datasets.

**Monitoring:** processed FPS, incoming FPS, frame drops, end-to-end p50/p95, inference p95 by stage, queue age, model version, no-face/multi-face/retry/spoof/unknown counts. Performance drift alerts based on aggregate blur/brightness/score distribution only; never infer classifier correctness without labeled audit data.

## 13. Recommended repository: one app, not a framework

```
faceguard-ai/
├── README.md
├── pyproject.toml
├── configs/{data.yaml,pad.yaml,runtime.yaml}
├── data/manifests/{frad.csv,celebA_train.csv,celebA_val.csv,celebA_test.csv}
├── notebooks/{01_kaggle_pad.ipynb,02_kaggle_frad_embeddings.ipynb}
├── src/faceguard/
│   ├── models.py            # InsightFace and PAD ONNX wrappers
│   ├── gallery.py           # enrollment/search/delete; no images persisted
│   ├── video.py             # camera/MP4, bounded queue, timestamps, tracking
│   ├── pipeline.py          # PAD gate -> recognition, state machine
│   ├── metrics.py           # FPS, latency, frame counts, session events
│   ├── train_pad.py         # reusable torchvision fine-tuning loop
│   ├── evaluate.py          # PAD + face recognition + video metrics
│   └── export.py            # ONNX export, parity, sha256
├── scripts/{preflight.py,prepare_pad.py,extract_gallery.py,run_demo.py,benchmark.py}
├── tests/{test_gallery.py,test_pipeline.py,test_data.py,test_runtime.py}
├── models/{manifest.json}   # weights locally/private; no redistributable weights in Git
├── reports/{data_audit.json,pad_metrics.json,recognition_metrics.json,benchmark.json}
├── deployment/{Dockerfile,compose.yaml}
├── .github/workflows/ci.yaml
├── requirements-{runtime,train,dev}.in
├── requirements-{runtime,train,dev}.lock
└── docs/{architecture.md,evaluation.md,model_card.md,limitations.md}
```

**Correction to the illustrative tree:** keep notebook filename `02_kaggle_frad_embeddings.ipynb` without spaces. Keep only files for implemented features. A single Streamlit demo is optional; a small OpenCV window plus CLI controls is the lowest-LOC MVP. In headless/container environments, avoid `cv2.imshow` and supply MP4 input + JSON result.

## 14. API + CLI minimum (reduce code)

CLI commands rather than microservices:

```
python -m faceguard.train_pad --config configs/pad.yaml
python -m faceguard.export --checkpoint models/pad_best.pt
python reference/archive/final-submission-excluded/scripts/extract_gallery.py --manifest PRIVATE_FRAD_MANIFEST
python scripts/benchmark.py --input sample_video.mp4 --backend onnx-cpu
python scripts/run_demo.py --source 0 --release models/manifest.json
python scripts/run_demo.py --source path/to/file.mp4 --release models/manifest.json
pytest -q
```

Enrollment is an **offline command** taking approved local image paths, extracts embeddings once, stores only vectors/IDs; deletes intermediate images/crops after computation. Online UI never asks for training labels. A simple `enroll`, `delete` and `verify` control is enough. API/FastAPI and Streamlit are optional after reliable CLI/webcam behavior.

## 15. Evaluation and ablations to discuss in interviews

| Comparison | What to hold fixed | Output |
|---|---|---|
| v1 FaceNet+SVM vs v2 ArcFace+cosine | Same held-out FRAD protocol wherever old pipeline can be rerun | rank-1, unknown rejection FMR/FNMR, measured latency |
| PAD MobileNet head-only vs fine-tuned final blocks | Same subject-disjoint splits + crop/augmentation policy | APCER, BPCER, ACER, AUC, loss curves |
| PAD tight vs expanded context crop | Same backbone/splits/train budget, re-train if crop distribution changes | unseen spoof errors + latency |
| PAD single-frame vs temporal median | Same frozen frame classifier/videos; no testing-set tuning | video-level APCER/BPCER and decision delay |
| Sequential video vs bounded live queue | Same camera-paced input, same models/hardware | latency p95, frame age, drops, accuracy/event continuity |
| PyTorch vs ONNX | Same inputs & CPU/GPU where possible | score parity, per-stage latency, model memory |
| Optional FP32 vs INT8 ONNX | Same benchmark hardware & calibration | accuracy regression + speed; no automatic promotion |

**Important:** historical v1 PAD 95.5% and v2 public PAD accuracy CANNOT be directly compared if sample population, attacks, crop policies, and splits differ. Label it historical context, not an ablation win.

**Threshold handling:** PAD uses `p_live`; identify threshold `tau_pad` on validation. Recognition uses cosine `sim`; choose `tau_id` on genuine-vs-impostor probes for the defined gallery. Overall success requires both gates. Report cascade false accepts separately (spoofed enrolled identity, bona fide unknown person, bona fide enrolled person); some security failures require actual presentation attacks featuring enrolled identities, not unrelated spoof datasets. No proof of robust anti-spoof security without that.

**Uncertainty and fairness:** show false reject breakdown by device/lighting/pose/eyeglasses where labels support; do not infer demographic traits from faces, and do not claim demographic fairness on datasets not designed to support it. If no feasible access to varied demographics, label this a limitation.

## 16. Tests and acceptance gates

1. **Data:** identities/sessions don't leak across selected train/val/test; missing labels/images cause explicit failure; no false zeros; dataset permission/license notes exist.
2. **Identity:** identical face embedding input yields identical normalized query; invalid embeddings rejected; gallery model hash exactly matches current ArcFace; registered unknown IDs return UNKNOWN on held-out impostor protocol at frozen threshold; add/delete person changes gallery without training.
3. **PAD:** verified label mapping (`bona fide=1`, `attack=0`); no train/test leakage; false-accept confusion matrix correct; explicit numerical thresholds; cross-dataset scores recorded; no guarantee beyond evaluated attack classes.
4. **Runtime:** no/multiple/blur face ⇒ retry; spoof ⇒ reject; low match ⇒ unknown; stale/drop frame ⇒ controlled failure, not use of old label; stream interruption doesn't cause unbounded RAM; each camera session ID new after reconnect.
5. **ONNX:** metadata lists input shape, order, normalization and opset; ORT loads; fixed/reference input parity with train model; classifier threshold flips and metric deltas recorded. CUDA provider never silently assumed.
6. **Privacy:** raw frames only RAM in production; no images or embeddings logged; gallery protected; deletion works; public repo has zero biometric images/vectors/licensed weights; model files verified by hash.
7. **MLOps:** Kaggle env snapshot; tracked training config/data hash; metrics and artifact checksum; reproducible external run instructions; CI green; versioned model and rollback path.
8. **Interview readiness:** show working camera demo, 1 architecture diagram, 1 model comparison table, 1 PAD error breakdown, 1 recognition ROC/threshold graph, 1 latency benchmark, and 1 MLOps training-to-release diagram. If a result is not measured, mark pending.

## 17. Execution order for a coding agent (small safe steps)

- **P0 / Day 0:** inspect original FRAD repository/config, permissions and dataset structure; detect Kaggle environment; source-license checklist. Do not code models until data and runtime access verified.
- **P1:** implement the frozen pretrained InsightFace `buffalo_m` wrapper and FRAD enrollment/evaluation (one identical embedder/model hash shared for every ID; no fine-tuning). Prove one ID enroll/search/delete + unknown rejection calibration.
- **P2:** implement MobileNetV3 PAD train/eval on verified CelebA-Spoof splits. Run tiny sanity then full fine-tune; log via MLflow. Export ONNX and parity-test.
- **P3:** integrate PAD before identity decision, single-face gate, quality checks, consistent crop handling, temporal median and short-lived tracker. Test with MP4 before live webcam.
- **P4:** add bounded queue, local benchmarks, ONNX CPU runtime. Optimize ONLY hotspots found via profiling.
- **P5:** DVC metadata/manifests, Docker, CI, release manifest and rollback; final cross-attack experiment, model card and short interview script.
- **P6 optional:** lightweight dashboard and additional stream input only after all P0–P5 gates pass.

Stop at failed gates and fix root cause; do not generate additional complexity as a workaround. Each phase must leave a runnable demo or evaluation and commit testable evidence.

## 18. AI coding-agent master instruction (copy with blueprint)

> Implement the project described in `FaceGuard_AI_Implementation_Blueprint.md` **phase by phase**. The highest priorities are package and model interoperability, privacy, correctness, measured accuracy/latency, and **minimal maintainable code**. Inspect actual repository files, Kaggle image versions, model weights, data annotation schema and permission constraints before installing or changing code. Do not assume dataset files or GPU providers exist. Avoid unnecessary packages, classes, microservices, and speculative features. Keep full evaluation methods separate from real-time optimized mode. Never fabricate weights, thresholds, test labels, results or SHA256 hashes. Training and identity embedding extraction should run on Kaggle GPU only where authorized; local machine only loads frozen models and gallery for inference. Test every stage, lock compatible versions after successful smoke tests, produce real measured artifacts, and stop with a clear blocker if data/license/provider compatibility fails. Implement P0 → P1 → P2 → P3 → P4 → P5, returning commands/tests/results at each milestone.

> Before acting, read the whole contract, `reference/tracking/progress.yaml`, `reference/governance/TRACKER_POLICY.md`, then load `configs/decisions.yaml` and the frozen manifests/configs. Follow Sections 0 and 20–24 as hard gates. If a required value is `UNRESOLVED`, absent, contradictory, or represented by a placeholder, implement only safe validation/scaffolding and stop that dependent phase as `BLOCKED`; do not select a plausible default. Never reinterpret historical 233-person statements as the current v2 cohort. Never call generated code complete without the required phase receipts. After meaningful work, update the YAML tracker and human status dashboard together using verifiable evidence.

## 19. Interview framing mapped to the CloudPhysician experience

**30 seconds:** “I extended our published FRAD face recognition and anti-spoofing work into a consent-based live video identity verification system. SCRFD detects, a fine-tuned lightweight CNN screens presentation attacks, and pretrained ArcFace embeddings are matched against a protected gallery with calibrated open-set rejection. I engineered bounded video queues and temporal decisions, exported models to ONNX, and used versioned datasets and MLflow to verify accuracy/latency before release.” **Say this only after each clause is implemented.**

**The interviewer may probe:**
1. **Computer vision architecture:** CNN convolution vs depthwise separable convolution; why MobileNetV3's size/efficiency, why SCRFD rather than MTCNN, detector/aligner and PAD crop differences, ArcFace embedding objective vs VGGFace/FaceNet and softmax classifiers.
2. **Training foundations:** chain rule, sigmoid+BCE gradient, backprop through CNN, AdamW vs SGD, LR schedule, exploding/vanishing gradients, ReLU/BN, weight decay vs dropout, effect of `model.train()` and `eval()`, validation vs test, Adam paper discussion.
3. **Real-time video:** why bounded queue, producer/consumer model, timestamping, head-of-line blocking, batch vs single-camera latency, frame drop safety, how tracking works, FPS vs true end-to-end latency, throughput and memory tradeoffs.
4. **Evaluation:** positive/negative class labels, APCER/BPCER vs classification accuracy; ROC and threshold; open-set recognition vs closed-set 233-way accuracy; impostor pools, galleries, cross-camera and cross-dataset generalization.
5. **Systems/MLOps:** Kaggle GPU train vs CPU deployment, ONNX parity, model artifact checksum, versioned split manifest, MLflow runs, rollback, data privacy, drift vs measured accuracy.
6. **Critical limitations:** printed photos vs electronic replays, video replay beyond single images, missing depth/IR, virtual camera injection, small/biased original data, consent/licensing, spoof/identity cascade attack evaluation.

The diagram/story stays **one problem**: “Is this a live, enrolled user in a video session, and can I decide reliably under resource constraints?” Everything else exists because that problem demands it.

## 20. Machine-readable sources of truth

Narrative prose guides implementation; machine-readable frozen files control executions. Once P0 is complete, training/evaluation code must consume manifests and configuration files, not rediscover samples by recursively scanning a directory.

### 20.1 Recognition inventory

Create `data/manifests/frad_inventory.csv` with exactly these columns:

```text
schema_version,candidate_person_id,relative_path,sha256,size_bytes,width,height,channels,decode_ok,face_count,detect_score,include,exclusion_reason,verified_person_id,session_id,device_id,capture_time_source,split,permission_id
```

Rules:

- One row per discovered file, including duplicates and rejected files. Never overwrite the raw audit.
- `relative_path` is relative to the configured dataset root; reject absolute paths and `..` traversal in consumers.
- `sha256` is of original bytes. An exact duplicate group shares a hash and is excluded from all but one canonical row unless authoritative evidence requires otherwise.
- `verified_person_id`, `session_id`, `device_id`, `split` and `permission_id` stay empty until verified. Empty does not mean “same session,” “genuine,” “allowed,” or “training.”
- `include` is tri-state during audit (`UNRESOLVED`, `true`, `false`) and must be Boolean before freezing.
- `exclusion_reason` uses a controlled value: `exact_duplicate`, `decode_failed`, `wrong_identity`, `no_face`, `multiple_faces`, `low_quality`, `permission_missing`, `cohort_excluded`, or `other:<short reason>`.
- Store the frozen CSV SHA-256 and row/count summary in `data_manifest.json`. Any content change creates a new manifest/release candidate.

Create `data/manifests/frad_subjects.csv` with `verified_person_id`, `candidate_folder`, `include`, `image_count_discovered`, `image_count_included`, `session_count_verified`, `permission_id`, and `decision_evidence`. The application exposes only `verified_person_id`; never infer a person's real name.

### 20.2 Split manifests

Create immutable files rather than regenerating splits at runtime:

```text
data/manifests/recognition_enrollment.csv
data/manifests/recognition_val_known.csv
data/manifests/recognition_val_impostor.csv
data/manifests/recognition_test_known.csv
data/manifests/recognition_test_impostor.csv
data/manifests/pad_train.csv
data/manifests/pad_val.csv
data/manifests/pad_test.csv
```

Every row contains `sample_id`, source dataset, relative path, subject ID, verified session/device fields when available, label, attack type when applicable, and original file hash. A validation script must prove:

- no file hash crosses splits;
- no PAD subject crosses train/validation/test;
- recognition impostor IDs never occur in the corresponding gallery;
- a known recognition probe never reuses an enrollment file hash;
- required fields and files exist and decode;
- label values belong to the frozen mapping;
- counts match the generated summary.

The split generator requires a recorded seed but the seed is not sufficient evidence of correctness. Save the explicit rows and split SHA-256. Test manifests become read-only after the first reported test run; changes create a new experiment ID and invalidate comparisons.

### 20.3 Configuration files

All executable commands take a configuration path and save the resolved configuration with the run. Unknown keys and missing required keys are errors; do not silently substitute library defaults. Configuration must cover:

- dataset root and manifest hashes;
- model pack/name/hash and explicit execution providers;
- image color order, orientation policy, crop algorithm, size, interpolation, scale and normalization;
- training seed, weights enum, frozen/unfrozen module names, epochs, batch, optimizer, learning rates, weight decay, scheduler, early stopping and augmentation parameters;
- threshold-selection policy and required validation target;
- runtime quality, tracking, temporal evidence, queue and time-gap values;
- target device, thread counts and logging/retention policy.

Starter values in this document authorize a sanity experiment only. The resolved run configuration—not prose and not an implicit library default—is the value of record.

### 20.4 Environment lock and model artifact contract

For each environment, create `reports/environment_<name>.json` and a platform-specific lock file. Record:

```text
environment_id, operating_system, architecture, Python full version,
package distribution name + exact version, lock-file hash, install source/index,
torch build/CUDA/cuDNN, GPU + driver, ONNX Runtime providers,
OpenCV build summary, smoke-test command and result timestamp
```

The required environments are `kaggle_train`, `local_cpu_runtime` and `ci`. They are independent; never copy a GPU lock into CPU runtime. Generate locks only after imports and smoke tests succeed. Prefer a hash-checked lock where the chosen installer supports it. On startup, release mode verifies the Python major/minor, critical package versions, model hashes, provider availability and gallery model hash; mismatch is fatal.

`models/manifest.json` must contain concrete values and pass JSON-schema validation before candidate/release use:

```json
{
  "schema_version": 1,
  "release_id": "faceguard-v2.0.0-rc1",
  "state": "candidate",
  "detector_recognizer": {
    "pack": "buffalo_m",
    "pack_source": "OFFICIAL_SOURCE_REQUIRED",
    "detector_file": "RELATIVE_PATH_REQUIRED",
    "detector_sha256": "64_HEX_REQUIRED",
    "recognizer_file": "RELATIVE_PATH_REQUIRED",
    "recognizer_sha256": "64_HEX_REQUIRED",
    "embedding_dim": 512
  },
  "pad": {
    "file": "RELATIVE_PATH_REQUIRED",
    "sha256": "64_HEX_REQUIRED",
    "opset": "INTEGER_REQUIRED",
    "input_name": "EXACT_NAME_REQUIRED",
    "input_shape": [1, 3, 224, 224],
    "output_name": "EXACT_NAME_REQUIRED",
    "preprocess_config_sha256": "64_HEX_REQUIRED",
    "threshold": "FINITE_NUMBER_REQUIRED",
    "threshold_report_sha256": "64_HEX_REQUIRED"
  },
  "recognition": {
    "gallery_file": "PRIVATE_RELATIVE_PATH_REQUIRED",
    "gallery_sha256": "64_HEX_REQUIRED",
    "gallery_manifest_sha256": "64_HEX_REQUIRED",
    "embedding_model_sha256": "MUST_EQUAL_RECOGNIZER_SHA256",
    "threshold": "FINITE_NUMBER_REQUIRED",
    "min_top1_top2_margin": "FINITE_NONNEGATIVE_NUMBER_REQUIRED",
    "threshold_report_sha256": "64_HEX_REQUIRED"
  },
  "environment_lock_sha256": "64_HEX_REQUIRED",
  "created_at": "ISO8601_WITH_TIMEZONE_REQUIRED"
}
```

Uppercase `_REQUIRED` strings illustrate schema fields only and are forbidden in a valid candidate. The model downloader, if implemented, accepts only the frozen official URL and expected SHA-256; it downloads to a temporary file, verifies the hash, and atomically moves it into place. Production startup never auto-downloads.

## 21. Deterministic algorithm contract

### 21.1 Common image handling

1. Decode with a tested library and apply EXIF orientation before conversion. A decode/orientation failure excludes the sample with a manifest reason.
2. Convert explicitly to RGB for torchvision. OpenCV camera frames arrive BGR and require an explicit `cv2.COLOR_BGR2RGB`; never decide color order from appearance.
3. Do not overwrite source images. Any derived cache key includes original SHA-256 plus preprocessing-config SHA-256.
4. Reject non-finite tensors, zero-sized crops, unsupported channel counts and unexpected shapes. Do not coerce them silently.

### 21.2 Recognition and enrollment

1. Initialize the frozen `buffalo_m` pack once with an explicit provider list. Verify detector and recognizer file hashes and the returned embedding dimension before processing data.
2. Run the pack's tested detection/5-landmark alignment path. For enrollment, require exactly one acceptable face. Record all rejections; do not select the largest face from a multi-face enrollment image.
3. Because current FRAD inputs are already 256×256 legacy crops, label the experiment `legacy-crop input`. Do not claim end-to-end detector evaluation from these crops. Prefer original uncropped frames if later authorized, but never mix original and legacy-crop protocols without separate reports.
4. Use the wrapper's model-specific 5-point alignment and preprocessing. Do not reproduce alignment constants until an equivalence test against the wrapper passes.
5. Cast the embedding to float32, reject non-finite/incorrect dimensions or norm below epsilon, then L2-normalize once. Store the model hash with every vector.
6. Primary gallery scoring is the maximum cosine score among the enrolled individual embeddings for each person, followed by the maximum person score. Prototype scoring is an explicitly named ablation, not a silent substitution. Calibrate again whenever gallery membership or scoring policy changes.
7. Accept only if `top1_score >= tau_id` and `top1_score - top2_score >= min_margin`. With fewer than two gallery identities, the margin check is undefined and release verification is blocked. Exact numerical ties return `unknown`.
8. Return `verified` only after PAD has passed for the same continuous track. A nearest identity below either gate remains `unknown`; never expose it as a match.

### 21.3 PAD crop and inference

The initial, testable crop algorithm is fixed as follows and may change only through a versioned experiment:

1. From detector box `(x1,y1,x2,y2)`, compute center and side `1.3 * max(width,height)`.
2. Create a square box around the center, round outward (`floor` minimums, `ceil` maximums), and clip to the frame.
3. Do not synthesize reflected face pixels. If clipping removes more than a configured/frozen fraction of the requested square or the crop is empty, return `retry`. The fraction has no release default and must be frozen in `runtime.yaml`.
4. Convert BGR to RGB, resize to 224×224 using the interpolation recorded in the config (initial experiment: bilinear), convert to float32 `[0,1]`, then normalize by ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]` in RGB order.
5. Model output is one finite logit. `p_live = sigmoid(logit)` and bona fide passes only when `p_live >= tau_pad`. An unexpected output shape/non-finite value is a hard inference failure, not a spoof score.

### 21.4 PAD training experiment

- Construct `torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)` and replace only `classifier[-1]` with a one-output `Linear` layer. Assert the discovered `in_features` and save the printed module graph in the run report.
- Label mapping is exactly `bona_fide=1`, `attack=0` after the dataset parser has proven the source annotation mapping. Use `BCEWithLogitsLoss`; never apply sigmoid before this loss.
- Sanity S0: overfit 32–64 authorized examples and prove loss decreases and labels/logits are not reversed.
- E1: train only the new final linear layer for exactly 5 maximum epochs using the fully resolved config.
- E2: start from the selected E1 checkpoint; unfreeze the last three top-level children of `model.features` plus the full classifier, record their exact parameter names, and train for at most 15 additional epochs. This is the initial experiment, not permission to reinterpret “last blocks” by parameter count.
- Initial optimizer policy: AdamW, head LR `1e-3`, unfrozen backbone LR `1e-4`, weight decay `1e-4`; early stopping monitors validation ACER with patience 4. Batch size, scheduler, augmentation probabilities and worker count must be fixed by the resolved config after the smoke/throughput test and cannot change mid-run.
- Save `last` and best-validation checkpoints atomically. A checkpoint includes architecture/config/split hashes, epoch, optimizer/scheduler/scaler state and label mapping. Resume only when those hashes agree.

### 21.5 Threshold selection and metrics

Implement threshold selection as a pure tested function over frozen validation scores. The policy is a required config value:

- `min_acer`: evaluate deterministic candidate thresholds, minimize `(APCER+BPCER)/2`, and choose the highest threshold on an exact tie.
- `apcer_bound`: choose the lowest threshold that satisfies the configured maximum APCER while minimizing BPCER among satisfying candidates; if none satisfies it, report failure and do not release.
- `min_recognition_balanced_error`: minimize `(FMR+FNMR)/2` on recognition validation searches, choosing the highest threshold on an exact tie.
- `recognition_fmr_bound`: meet the configured maximum search-level FMR and minimize FNMR; if impossible, report failure and do not release.

Threshold candidates must include boundary sentinels around all unique finite validation scores and must use the same `>=` comparison as runtime. Save score-unit definitions, confusion counts, selected policy/target, curve data and threshold in a hash-addressed report. Never tune a margin, temporal window, crop or threshold on test results. Confidence intervals use subject/video-cluster bootstrap when the unit supports it; record seed, resamples and clustering unit.

### 21.6 Runtime state machine

Use explicit states `EMPTY`, `COLLECTING_PAD`, `PAD_FAILED`, `PAD_PASSED`, `IDENTIFIED`, `UNKNOWN`, `RETRY`, and `ENDED`. Every new stream/reconnect creates a new session. Reset all track evidence on multiple/no face beyond the configured gap, timestamp regression, excessive frame age, association failure, model error, or reconnect. PAD evidence and cached identity never move between tracks/sessions. A `retry` or `unknown` result never unlocks/accepts. Unit tests must cover every transition and reset.

## 22. Phase gates and required receipts

An agent works on only one numbered phase at a time. It may prepare code needed by that phase but must not announce the next phase complete. After each phase it reports changed files, exact commands, pass/fail output, generated artifact hashes, unresolved decisions and privacy-impact summary.

### P0 — repository, data and environment audit

Required receipts:

- inventory and subject manifests with summaries and hashes;
- duplicate, decode, dimensions and face-detection audit reports;
- `configs/decisions.yaml` with each downstream blocker visible;
- permission/publication records or an explicit `BLOCKED` state;
- Kaggle/local/CI preflight reports where each environment is available;
- repository secret/biometric ignore policy before initializing or publishing Git.

P0 passes only when the cohort used by P1, its permission, label meaning and evaluation limitation are `VERIFIED`. Lack of PAD data does not block P1, but remains an explicit P2 blocker.

### P1 — recognition

Required receipts:

- verified InsightFace model files/hashes and environment lock;
- frozen recognition manifests;
- deterministic enrollment/gallery artifact stored privately;
- tests for normalization, invalid/multiple/no face, model-hash mismatch, duplicate handling, add/delete, scoring, threshold boundary, ties and unknown rejection;
- validation threshold report and untouched test report, each naming the exact gallery size and protocol limitation.

P1 cannot claim cross-session performance without verified session-independent probes.

### P2 — PAD

Required receipts:

- authorized PAD dataset, verified label mapping and subject-disjoint frozen splits;
- S0, E1 and E2 run records with resolved configs and environment hashes;
- selected validation threshold report and untouched/internal plus external reports where available;
- ONNX file, model card and parity report over at least 100 representative authorized inputs, including max/mean error and threshold flips.

P2 is blocked with the currently observed recognition-only folder unless an authorized PAD dataset is provided and verified.

### P3 — integrated sequential-video pipeline

Required receipts: complete state-machine tests; PAD-before-recognition decision proof; MP4 integration report; reset/failure tests; no biometric material in logs; cascade outcomes separated for spoof, bona-fide unknown and bona-fide enrolled inputs. A working still-image recognition script does not pass P3.

### P4 — live runtime and performance

Required receipts: bounded-queue/drop tests; camera or paced-video benchmark; target hardware/system report; stage and end-to-end p50/p95 latency, frame age, FPS, memory and drops; disconnect/reconnect tests. A benchmark from different hardware is informative but cannot pass the target-hardware gate.

### P5 — release engineering

Required receipts: schema-valid concrete release manifest; environment locks; CI results; Docker/runtime smoke test; private gallery access/deletion test; public-artifact privacy scan; documentation/model card; rollback test; final checksum list. Promotion is an explicit action after all gates pass, never an automatic consequence of successful training.

## 23. Current blocker register (24 October 2026)

This table describes current evidence and must be updated, not bypassed:

| ID | Current state | Evidence | Required resolution | Blocks |
|---|---|---|---|---|
| `FRAD_PERMISSION` | `UNRESOLVED` | Images exist locally; no permission record audited | Verify local processing, Kaggle/cloud transfer and publication restrictions | P1 execution/cloud use |
| `FRAD_COHORT` | `DISCOVERED` | 234 folders/2,802 files; historical report says 233 | Approve included subjects/files and explain `0234`, `0183`, duplicate | P1 split/release |
| `FRAD_LABEL_MEANING` | `DISCOVERED` | Numeric folders only | Verify one folder equals one opted-in person and images are bona fide | P1 |
| `FRAD_SESSION_POLICY` | `UNRESOLVED` | Filenames suggest dates but are not authoritative metadata | Supply/verify session/device data or declare exploratory unknown-session protocol | Strong P1 claims |
| `PAD_DATASET_PERMISSION` | `UNRESOLVED` | No labeled PAD dataset found in current `dataset/` audit | Obtain authorized official PAD data and terms | P2 onward |
| `PAD_LABEL_MAPPING` | `UNRESOLVED` | No PAD annotations audited | Prove official source-label semantics | P2 |
| `KAGGLE_ENVIRONMENT` | `UNRESOLVED` | No Kaggle preflight attached | Run K0 and save report | Kaggle training |
| `LOCAL_RUNTIME_ENVIRONMENT` | `UNRESOLVED` | No tested v2 lock | Run local preflight/smoke and freeze lock | P1 runtime/P3+ |
| `INSIGHTFACE_PACK` | `UNRESOLVED` | `buffalo_m` chosen, files/hashes absent | Acquire under accepted terms from official source and verify exact files | P1 |
| `TARGET_HARDWARE` | `UNRESOLVED` | “local laptop” is not a reproducible target | Record CPU/RAM/OS/camera and performance objective | P4 |
| Threshold policies/values | `UNRESOLVED` | Placeholders by design | Freeze policy before validation; derive value from validation only | Candidate release |

## 24. Completion and future-agent handoff contract

“Generate the project” means implement only through the highest phase whose prerequisites are satisfied. Generated-but-unexecuted code is `IMPLEMENTED_UNVERIFIED`; passing synthetic unit tests is `UNIT_VERIFIED`; passing authorized real-data gates is `DATA_VERIFIED`; and only P5 promotion is `RELEASED`. These labels must never be collapsed into “done.”

A future coding agent must begin every work session by:

1. reading this entire contract and `configs/decisions.yaml`;
2. inspecting current workspace status without altering user files;
3. naming the active phase and its unresolved prerequisites;
4. verifying that referenced manifest/config/model hashes still match;
5. making the smallest testable change allowed by that phase;
6. running proportionate tests and reporting real output;
7. stopping on any permission, semantic, environment, hash or data-integrity blocker.

The agent may implement validators and scaffolding while a data-dependent phase is blocked, but it must not manufacture substitute images, labels, embeddings, model weights, thresholds, hashes, metrics, permissions or successful logs. Synthetic fixtures are allowed only in tests and must be visibly named synthetic. If official documentation and the installed API disagree, record both, test the installed API in isolation, and either adapt with a version-specific wrapper or stop; never edit the evidence to make it appear compatible.

## 25. Reference documentation (checked Oct 2026)

- Kaggle notebook images, snapshots: https://www.kaggle.com/docs/notebooks and https://github.com/Kaggle/docker-python
- PyTorch official release pairings: https://pytorch.org/get-started/previous-versions/
- Torchvision MobileNetV3: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html
- PyTorch ONNX export and AMP: https://docs.pytorch.org/docs/main/onnx_export.html ; https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- InsightFace models/license: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md
- MiniFASNet reference: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- CelebA-Spoof official: https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html
- Replay-Attack official: https://www.idiap.ch/en/dataset/replay-attack
- ONNX Runtime GPU matching: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- MLflow tracking: https://mlflow.org/docs/latest/ml/tracking
- DVC pipelines: https://doc.dvc.org/user-guide/pipelines/defining-pipelines
- OpenCV pip variants: https://docs.opencv.org/5.0/py_tutorials/py_setup/py_pip_install/py_pip_install.html
