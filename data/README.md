# Data Zones

All face images, video and derived biometric data are restricted. This directory is excluded from public Git except for this documentation and approved non-sensitive manifests.

```text
data/
├── manifests/   # checksums, pseudonymous IDs, splits and provenance
├── raw/         # authorized original inputs; Git-ignored
├── interim/     # temporary preparation/cache; Git-ignored
└── processed/   # derived crops/features; Git-ignored
```

## Current restricted dataset

- Location: `data/processed/frad_legacy_crops/`
- Classification: restricted biometric, legacy-crop recognition input
- Discovered files: 2,802
- Discovered candidate folders: 234
- File-content multiset SHA-256 at organization time: `cf24a522c45d81014367897e6a198493d7ea9a7fefb87285becf06ec5aeb269c`
- Known structural issues: folder `0183` has 10 images, folder `0234` has 8, and one exact duplicate exists under `0164`.

These structural facts do not verify consent, identity semantics, session independence, face correctness or dataset suitability. Training/evaluation must consume frozen manifests rather than recursively discovering files.

## Current draft audit

- Inventory: `data/manifests/frad_inventory.csv`
- Inventory SHA-256: `c0d1d6c6908c232de4f984172c0930ee9dbdd3f4c1c013efa17b0415f6726df3`
- Draft subject manifest: `data/manifests/frad_subjects.csv`
- Draft subject manifest SHA-256: `18994659d56b8ebbc52ade3f4788460b816006925ce58122ebaa564d3310d79d`
- Audit report: `reports/audits/frad_audit.json`
- Decode result: 2,802/2,802 succeeded
- Include state: 2,801 `UNRESOLVED`, one exact duplicate `false`

This is a draft structural inventory. It is not frozen and does not yet contain SCRFD face counts, approved identities, permissions, sessions or split assignments.
