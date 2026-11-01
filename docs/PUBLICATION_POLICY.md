# Publication and Submission Policy

This repository is approved for an academic prototype submission under a
private-by-default policy. Approval does not grant rights that belong to dataset or
model owners and does not authorize commercial use.

## Material permitted in a public code submission

- application and test source code;
- dependency locks and configuration without secrets;
- aggregate, non-identifying metrics and hardware measurements;
- cryptographic hashes, counts, pseudonymous protocol descriptions and model cards;
- documentation that clearly states the prototype limitations and applicable terms.

## Material that must remain private

- raw face images, PAD dataset images, crops, frames, videos and screenshots containing
  faces;
- embeddings, private galleries, identity aliases, attendance databases and exported
  attendance files;
- PAD and recognition weights, checkpoints and model bundles unless their owner terms
  separately authorize redistribution;
- credentials, operator PIN data, Kaggle/GitHub tokens and private runner configuration;
- CelebA-Spoof images or derived copies. Its official agreement restricts use to
  non-commercial research and prohibits publication or redistribution of dataset
  portions.

The existing `.gitignore` is the enforcement baseline. Before every publication or
submission, run `scripts/scan_artifacts.py`, review `git status`, and manually inspect
the exact archive or commit being shared. A passing automated scan is a guardrail, not
a transfer of publication rights.

## External services

Kaggle inputs and outputs must remain private unless every source license permits the
intended disclosure. Private CI may mount models locally on an authorized self-hosted
runner; weights, biometric data and credentials must never be uploaded as workflow
artifacts or exposed in logs.

## Demonstration media

Record only consenting participants, keep recordings outside Git, share them only with
the intended academic reviewer, and delete them after the approved retention period.
The application's no-frame-retention guarantee does not apply to an external recorder.

## Policy Scope

This policy governs the data privacy and security guidelines for the FaceGuard AI codebase and artifacts. Any commercial deployment, public release of private artifacts, new dataset integration, or external sharing requires prior review against data privacy and licensing regulations.

