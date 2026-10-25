# Recognition manifest templates

These header-only templates document the immutable recognition split contract. Copy
them to `data/manifests/recognition_*.csv` only after identity semantics, permission
and the session limitation are verified. Do not infer session/device values from
filenames, and do not put the same source hash in enrollment and a probe split.

The templates intentionally contain no biometric records and grant no completion
credit by themselves.
