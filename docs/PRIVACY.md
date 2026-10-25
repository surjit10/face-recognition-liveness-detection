# Privacy and Data Handling

FaceGuard AI runs entirely on the local laptop. Biometric data is sensitive and is
kept inside the machine's private directories; nothing is uploaded by the runtime.

## Local-only execution

- The runtime reads only the artifacts referenced by `configs/runtime.yaml` and the
  authorized camera source.
- No network request is made during detection, PAD, recognition or reporting.
- The InsightFace wrapper refuses absent model packs; it never downloads or substitutes
  weights.
- Only camera index `0` is supported and authorized. Automatic camera enumeration is
  intentionally absent.
- No telemetry, analytics, crash upload or remote logging is configured.

## Zero retention of camera data

The application does **not** write camera frames, face crops, face videos or embeddings
to disk. A frame and its crop exist only in process memory for the current inference and
are discarded immediately. The preview window is a live visualization, not a recording.

Structured console events are emitted only when the pipeline state changes. An event
contains status, state, pseudonymous identity (only after PAD passes), one decision
score, PAD score, track ID, model version, reason and latency. It never contains a
frame, crop, bounding box, embedding, pixel value or per-frame similarity history.
Normal runs emit no per-frame score series.

The desktop preview may draw face boxes and local display labels in memory. Those
annotations are never added to structured events or saved by the application. In a
multi-face scene, boxes are numbered only; identity matching remains disabled.

The only files the runtime can write are:

- `reports/environments/runtime_readiness.json` when `--readiness-report` is used, which
  contains hashes, provider names and gallery counts only;
- `reports/benchmarks/prototype_camera.json` from the explicit benchmark command, which
  contains aggregate hardware, latency and state-count values only.

## Restricted artifacts

These paths are private and Git-ignored. They must not be copied into reports, issues,
screenshots, chat, notebooks or public repositories:

| Path | Content |
|---|---|
| `data/raw/`, `data/interim/`, `data/processed/` | source images, crops, archives, manifests tied to biometric data |
| `models/private/` | face/PAD weights and private gallery |
| `reports/private/` | private evaluation scores |
| `reference/archive/` | archived legacy biometric assets |

`.gitignore` also excludes `.npy`, `.npz`, `.onnx`, `.zip`, `.pt`, `.pth`, key and
credential files. Gallery writes use a temporary file with mode `0600` and an atomic
replace. Manifests and decisions contain labels and hashes, never images or vectors.

## Logging and disclosure rules

- Do not attach camera frames, crops or screen recordings containing faces to reports.
- Aggregate metrics, hashes, pseudonymous labels and state counts are permitted.
- Identity `000` and all other gallery labels remain canonical pseudonymous project
  IDs. An explicitly authorized operator may define a local display alias under the
  Git-ignored `configs/private/identity_labels.yaml`, but real names must never appear
  in tracked configuration, structured events, reports or public material.
- If an operator records the demonstration externally, that recording is outside the
  application and is governed by the operator's retention policy. It must be deleted
  after review and must never be committed.
- Console output may be retained as a text receipt only after confirming it contains no
  biometric payload; it contains no frames or embeddings by design.

## Deletion and local cleanup

There is no per-user biometric record to delete because the runtime creates none.
To reduce local exposure, remove private artifacts and any externally recorded files
under the retention policy; do not use destructive Git commands to rewrite unrelated
user data.

## Limitations

Local execution and zero application retention do not make the prototype secure or
anonymous. A compromised host, another process with camera access, or an operator-made
recording can still capture biometric data. The system remains an academic prototype.
