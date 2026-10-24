# Reference Directory Removal Checklist

The `reference/` directory is intentionally isolated from application imports, but it also contains the only current blueprint/tracker and archived pre-v2 material. Do not remove it merely because the application starts. Remove it only after every required item below is confirmed.

## Before removal

- [ ] P0–P5 are complete and the promoted release has a passing phase receipt.
- [ ] Permanent architecture, evaluation, model card, limitations and runbook documents exist under `docs/`.
- [ ] Final public README contains only measured claims and has no links into `reference/`.
- [ ] Required decision evidence has been promoted from reference anchors to immutable reports/manifests under `reports/` and `data/manifests/`.
- [ ] `configs/decisions.yaml` contains no evidence path under `reference/`.
- [ ] No source, script, test, notebook, configuration, CI or deployment file imports/opens/links to `reference/`.
- [ ] Any historical v1 source/assets that must be retained have been moved to an approved private archive or tagged source-control history.
- [ ] The legacy YuNet model/tool is confirmed unnecessary or has been retained in an approved private archive.
- [ ] Tracker history and final completion receipt have been exported to the chosen permanent project record, if desired.
- [ ] A privacy review confirms whether archived historical sample images must be securely deleted rather than copied.
- [ ] A backup/checksum is created for anything intentionally retained.

## Safe dependency check

From the project root, search active project areas before removal:

```bash
rg -n "reference/|reference\\\\" README.md configs data/manifests deployment docs models notebooks reports scripts src tests
```

Every result must be removed, promoted to a permanent path, or explicitly documented as harmless. Then run the complete test and packaging suites without `reference/` available in the test environment.

## Removal authorization

Removal is destructive and must be explicitly requested by the user after the checklist passes. Report exactly what will be deleted, its size, whether a backup exists, and whether recovery is possible. Prefer moving the directory to a temporary quarantine/trash location before permanent deletion.
