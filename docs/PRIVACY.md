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
- Optional operator access stores only a salted PBKDF2-SHA256 PIN digest under the Git-ignored
  `configs/private/operator.yaml`; the plaintext PIN is never stored.

## Camera-data and enrollment retention

The application does **not** write camera frames, face crops or face videos to disk. A
frame and its crop exist only in process memory and are discarded. Verification query
embeddings are not retained. During explicit guided enrollment, five quality-checked
ArcFace vectors remain in memory until the operator confirms or clears the enrollment;
confirmed vectors are written only to the protected private gallery. The preview window
is a live visualization, not a recording.

Structured console events are emitted only when the pipeline state changes. An event
contains status, state, pseudonymous identity (only after PAD passes), one decision
score, PAD score, track ID, model version, reason and latency. It never contains a
frame, crop, bounding box, embedding, pixel value or per-frame similarity history.
Normal runs emit no per-frame score series.

The desktop preview may draw face boxes and local display labels in memory. Those
annotations are never added to structured events or saved by the application. In a
multi-face scene, boxes are numbered only; identity matching remains disabled.

The application and explicit maintenance commands can write:

- `reports/environments/runtime_readiness.json` when `--readiness-report` is used, which
  contains hashes, provider names and gallery counts only;
- `reports/benchmarks/prototype_camera.json` from the explicit benchmark command, which
  contains aggregate hardware, latency and state-count values only;
- `models/private/gallery/gallery.npz` and private sidecar/metadata files after confirmed
  enrollment or deletion. These contain biometric vectors or restricted metadata;
- `reports/private/checkins.sqlite3` for automatic attendance after successful
  verification. It stores identity reference, timestamp, local date, event type,
  verified outcome, and version fields
  only—never pixels, embeddings, boxes, PAD scores or similarity scores;
- an operator-selected CSV export of those permitted attendance fields.

## Restricted artifacts

These paths are private and Git-ignored. They must not be copied into reports, issues,
screenshots, chat, notebooks or public repositories:

| Path | Content |
|---|---|
| `data/raw/`, `data/interim/`, `data/processed/` | source images, crops, archives, manifests tied to biometric data |
| `models/private/` | face/PAD weights and private gallery |
| `reports/private/` | private evaluation scores and local attendance database |
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

The **Enrolled users** screen deletes a selected identity's gallery vectors and private
enrollment metadata only after explicit confirmation. It updates the pinned gallery
hash, verifies the new gallery, reloads the runtime, and rolls back failed mutations.
The **Attendance register** screen separately deletes all local event records after a
second confirmation. Deleting a gallery identity does not silently rewrite historical
event records; clear the history separately when the applicable retention policy
requires it. Externally recorded demonstrations must also be removed under that policy.

## Limitations

Local execution and zero application retention do not make the prototype secure or
anonymous. A compromised host, another process with camera access, or an operator-made
recording can still capture biometric data. The system remains an academic prototype.
