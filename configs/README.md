# Configuration

`runtime.yaml` is the strict local runtime contract. Its gallery hash and recognition
operating point come from the frozen exploratory validation protocol. Startup remains
blocked until the official local `buffalo_m` files match the frozen hashes. Unknown
keys are errors.

Configuration files are strict, versioned inputs. Runtime and training code must reject missing or unknown keys and save each resolved configuration with its run.

- `decisions.yaml`: evidence-backed project decisions and blockers.
- `data.yaml`: dataset roots and frozen manifest hashes (created during P0).
- `recognition.yaml`: model, scoring and threshold policy (created during P1).
- `pad.yaml`: preprocessing and training contract (created during P2).
- `runtime.yaml`: queue, quality, tracking and session policy (created during P3/P4).

Never store secrets, credentials, real names, embeddings or machine-specific absolute biometric paths here.
