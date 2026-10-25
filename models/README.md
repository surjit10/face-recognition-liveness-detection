# Model Artifacts

`models/manifest.json` will be the public-safe release contract after model files and hashes are verified. Actual weights, protected galleries and immutable local releases belong in the Git-ignored `private/` and `releases/` directories.

Production startup must verify hashes and must never auto-download or silently replace weights.

`manifest.schema.json` defines the strict candidate/release contract. A concrete
`manifest.json` is generated only after the private gallery, both calibrated
threshold reports, environment lock and all referenced hashes exist.
