# Tests

- `unit/`: pure behavior with synthetic inputs.
- `contract/`: configuration, manifest, interface and artifact-hash checks.
- `integration/`: authorized tiny files/video or clearly synthetic fixtures.
- `privacy/`: assertions that logs/public artifacts contain no biometric data.
- `fixtures/`: synthetic or explicitly authorized non-public fixtures only.

Passing synthetic tests proves implementation behavior, not real-data accuracy.

Runtime unit coverage includes gallery normalization/hash/tie/deletion behavior,
PAD crop color/shape/clipping, validation threshold selection, PAD-before-recognition
ordering, spoof/unknown/reset transitions and latest-frame queue overflow.

Desktop application coverage includes shared runtime readiness reporting, fail-closed
operator presentation, camera-0 authorization, bounded capture/inference workers,
camera-open failure, state-change-only events and clean idempotent shutdown.
Memory-only overlay coverage verifies verified-name boxes, numbered multi-face boxes,
and the absence of all bounding boxes from structured events.
