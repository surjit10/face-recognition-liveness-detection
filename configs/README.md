# Configuration

The final prototype uses three versioned configuration files:

- `runtime.yaml` — model hashes, thresholds, providers, quality gates and session policy.
- `performance_objective.yaml` — approved target-hardware performance limits.
- `decisions.yaml` — frozen scope, permissions and evidence-backed project decisions.

Private operator labels and credentials belong under the Git-ignored `configs/private/`
directory. Never store passwords, tokens, real names, embeddings or machine-specific
absolute biometric paths in versioned configuration.
