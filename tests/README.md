# Tests

- `unit/`: pure behavior with synthetic inputs.
- `contract/`: configuration, manifest, interface and artifact-hash checks.
- `integration/`: authorized tiny files/video or clearly synthetic fixtures.
- `privacy/`: assertions that logs/public artifacts contain no biometric data.
- `fixtures/`: synthetic or explicitly authorized non-public fixtures only.

Passing synthetic tests proves implementation behavior, not real-data accuracy.
