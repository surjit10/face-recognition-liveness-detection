# Phase Receipt — `<PHASE_ID>`

Copy this template to `reports/phase_receipts/<phase_id>_<timestamp>.md` when a phase gate is evaluated. Do not edit the reference template with run results.

## Identification

- Phase:
- Receipt state: `PASS | FAIL | BLOCKED`
- Started at (ISO-8601 with timezone):
- Finished at (ISO-8601 with timezone):
- Operator/agent:
- Blueprint version and SHA-256:
- Tracker version/last-updated value:
- Decision-register SHA-256:

## Scope

- Objective:
- Included tasks:
- Explicitly excluded work:
- Privacy/data-access authorization used:

## Resolved inputs

| Input | Version/hash | State | Evidence |
|---|---|---|---|
| Dataset manifest |  | `VERIFIED/FROZEN` |  |
| Split manifest |  | `VERIFIED/FROZEN` |  |
| Configuration |  | `VERIFIED/FROZEN` |  |
| Environment lock |  | `VERIFIED/FROZEN` |  |
| Model/release manifest |  | `VERIFIED/FROZEN` |  |

Delete rows that are genuinely not applicable and explain why. Never write `N/A` to hide a missing prerequisite.

## Changes

| File/artifact | Change | Privacy classification |
|---|---|---|
|  |  | `public/internal/restricted-biometric` |

## Commands and results

Record exact commands without secrets or private absolute paths.

```text
<command>
```

```text
<essential output and exit status>
```

## Acceptance checks

| Gate | Result | Evidence |
|---|---|---|
| Blueprint-required tests passed | `PASS/FAIL/BLOCKED` |  |
| Artifact/config/model hashes agree | `PASS/FAIL/BLOCKED` |  |
| No forbidden biometric data in logs/artifacts | `PASS/FAIL/BLOCKED` |  |
| Metrics identify dataset/split/threshold/sample unit | `PASS/FAIL/BLOCKED` |  |
| Rollback/failure behavior tested where applicable | `PASS/FAIL/BLOCKED` |  |

## Measured results

Include only actually measured results. Each metric must name dataset manifest hash, split hash, model hash, threshold, sample unit and confidence-interval method or reason unavailable.

## Failures and limitations

- Failed checks:
- Known limitations:
- Unsupported claims explicitly avoided:

## Unresolved blockers

| Blocker ID | Evidence | Smallest required resolution | Affected tasks |
|---|---|---|---|
|  |  |  |  |

## Tracker update

- Tasks changed to `COMPLETE`:
- Tasks changed to `IN_PROGRESS`:
- Tasks changed to `BLOCKED`:
- Previous overall completion:
- New overall completion:
- `progress.yaml` validation result:

## Decision

`PASS`, `FAIL`, or `BLOCKED` with one concise reason. A failed or blocked receipt does not authorize entry into the next dependent phase.
