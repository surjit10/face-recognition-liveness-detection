# Tracker Governance Policy

This policy prevents progress inflation, stale dashboards, and unsupported “done” claims. It applies to every human or AI agent modifying FaceGuard AI v2.

## Sources of truth

1. `reference/FaceGuard_AI_Implementation_Blueprint.md` defines requirements and phase gates.
2. `reference/tracking/progress.yaml` is the canonical task/status record.
3. `reference/tracking/STATUS.md` is the human-readable projection of the YAML.
4. Actual reports, tests, manifests, configurations and artifact hashes are completion evidence.

When these disagree, the blueprint and actual evidence win. Correct the YAML and dashboard immediately; never edit evidence to match an optimistic status.

## Allowed status transitions

```text
NOT_STARTED -> IN_PROGRESS -> COMPLETE
NOT_STARTED -> BLOCKED
IN_PROGRESS -> BLOCKED
BLOCKED -> IN_PROGRESS
COMPLETE -> IN_PROGRESS    # regression, invalid evidence, or changed requirement
COMPLETE -> BLOCKED        # a previously satisfied prerequisite became invalid
```

`COMPLETE` is not permanent. A model, environment, dataset, configuration, permission or hash change reopens every dependent task until its checks are repeated.

## Completion requirements

A task is `COMPLETE` only when all are true:

- the requested output exists at a stable path;
- the applicable command/test completed successfully;
- the output references the correct config/data/model/environment hashes;
- sensitive artifacts are in their permitted storage zone;
- the evidence path is recorded in `progress.yaml`;
- no unresolved prerequisite contradicts the result.

Source code existing is evidence of implementation, not evidence that behavior is correct. A notebook cell output without saved code/config/environment metadata is not reproducible evidence. A screenshot is supporting material, not the only test record.

## Progress calculation

```text
phase_percent = 100 * phase_complete_tasks / phase_total_tasks
overall_percent = 100 * all_complete_tasks / all_total_tasks
remaining_tasks = total_tasks - complete_tasks
```

Round displayed percentages to one decimal place. `BLOCKED`, `IN_PROGRESS`, and `NOT_STARTED` all count as remaining. Do not weight easy and difficult tasks differently unless the tracker schema and every historical percentage are versioned together.

## Updating the tracker

For every update:

1. Read the latest blueprint, YAML tracker and evidence.
2. Verify dependencies before changing the task.
3. Update the task status, evidence and notes.
4. Recalculate phase totals from its task list.
5. Recalculate global totals from every task list.
6. Update the YAML `last_updated`, summary and append-only `change_log`.
7. Synchronize `STATUS.md` totals, blockers and next permitted work.
8. Validate YAML parsing, unique task IDs, declared counts and evidence paths.
9. Report the exact status changes to the user.

An agent must not rewrite the full change log, backdate an entry, remove an inconvenient blocker, or mark a parent phase complete while any child task is incomplete.

## Adding or changing tasks

Add a task when the blueprint adds a real acceptance requirement that is not represented. Record the blueprint section and why it was added. Updating the denominator can lower progress; that is correct. Do not split completed tasks merely to inflate the count or merge incomplete tasks to inflate the percentage.

Changing the blueprint can invalidate tracker items. Update the blueprint version, map affected tasks, and reopen them where required.

## Blocker behavior

A blocked item records:

- the blocker ID;
- the evidence showing why progress cannot safely continue;
- the smallest decision or artifact needed to unblock it;
- which downstream tasks are affected.

Agents may implement non-dependent validators, schemas and synthetic unit fixtures while blocked. They may not invent permissions, labels, models, versions, thresholds, hashes, metrics or successful executions.

## Evidence hygiene

- Prefer repository-relative paths so the tracker remains portable.
- Never link public tracking files to raw biometric images or private gallery contents.
- Store only aggregate counts and artifact hashes in reference documentation.
- Do not paste secrets, access tokens, person names, embeddings or restricted dataset samples into evidence.
- If an evidence artifact must remain private/outside Git, record a pseudonymous artifact ID, checksum and authorized storage location—not its contents.
