# Model Boundary

The active application loads hash-pinned artifacts from Git-ignored local directories:

```text
models/
├── manifest.schema.json   # public release-manifest schema
├── private/               # detector, recognizer, PAD model and gallery
└── releases/              # optional immutable local release candidates
```

Model weights and gallery vectors are required on the demonstration laptop but are
never included in the public submission. Startup verifies their SHA-256 hashes from
`configs/runtime.yaml` and fails closed on any mismatch. The runtime does not silently
download or substitute a model.
