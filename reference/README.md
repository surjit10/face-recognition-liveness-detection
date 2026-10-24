# FaceGuard AI v2 — Reference Control Center

This directory contains the authoritative planning, progress and governance material for the v2 implementation. Its active control areas contain no raw face images, videos, embeddings, private galleries, secrets or restricted model weights. `archive/` is a Git-ignored, temporary quarantine that may contain historical face sample assets and a legacy model; treat the entire archive as restricted and remove it deliberately when it is no longer required.

## Start here

| Order | Document | Purpose |
|---:|---|---|
| 1 | [`FaceGuard_AI_Implementation_Blueprint.md`](FaceGuard_AI_Implementation_Blueprint.md) | Full technical contract, no-assumption rules, algorithms and phase gates |
| 2 | [`tracking/STATUS.md`](tracking/STATUS.md) | Current human-readable completion and blockers |
| 3 | [`tracking/progress.yaml`](tracking/progress.yaml) | Canonical machine-readable tasks, dependencies, evidence and progress |
| 4 | [`governance/TRACKER_POLICY.md`](governance/TRACKER_POLICY.md) | Rules for changing statuses and calculating completion |
| 5 | [`architecture/REPOSITORY_STRUCTURE.md`](architecture/REPOSITORY_STRUCTURE.md) | Maintainable target repository layout, module boundaries and privacy zones |
| 6 | [`REMOVAL_CHECKLIST.md`](REMOVAL_CHECKLIST.md) | Safe promotion and dependency checks before deleting this temporary directory |

## Templates

- [`templates/decision_register.yaml`](templates/decision_register.yaml): copy to `configs/decisions.yaml` during P0 and replace unresolved values only with evidence.
- [`templates/phase_receipt.md`](templates/phase_receipt.md): copy to `reports/phase_receipts/` whenever evaluating a phase gate.

Templates are not execution evidence. Their placeholder or example values may not be used to claim completion.

## Current status

- Active phase: **P0 — audit and governance**
- Verified tasks: **4 of 59**
- Remaining tasks: **55**
- Evidence-based completion: **6.8%**
- Immediate goal: verify FRAD permission, identity semantics and canonical cohort, then generate frozen manifests.
- P2 blocker: no authorized labeled anti-spoofing dataset has been verified in the current workspace.

The status dashboard and YAML must be updated together after meaningful work. Percentages measure passed checklist items, not estimated effort or generated code.

## Rules for AI agents

Before changing project code or status, an agent must:

1. read the entire blueprint;
2. read the YAML tracker and tracker policy;
3. identify the active phase and unresolved prerequisites;
4. verify evidence paths/hashes before marking anything complete;
5. make only phase-authorized changes;
6. update tracker and dashboard in the same change;
7. stop rather than assume a missing permission, label, version, model, threshold or result.

When documents disagree, actual verified evidence and the blueprint take precedence over the dashboard. Correct the dashboard; never weaken the gate.

## Reference directory structure

```text
reference/
├── README.md
├── REMOVAL_CHECKLIST.md
├── FaceGuard_AI_Implementation_Blueprint.md
├── architecture/
│   └── REPOSITORY_STRUCTURE.md
├── archive/                          # disposable pre-v2 material; never imported by v2
│   ├── legacy_v1/
│   ├── legacy_crop_tool/
│   └── workspace_metadata/
├── governance/
│   └── TRACKER_POLICY.md
├── tracking/
│   ├── STATUS.md
│   └── progress.yaml
└── templates/
    ├── decision_register.yaml
    └── phase_receipt.md
```

Keep the index small. Add a new reference document only when it has a distinct owner and lifecycle; otherwise extend the existing authoritative file.
