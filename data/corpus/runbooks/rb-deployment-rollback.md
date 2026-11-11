---
id: rb-rollback
title: Runbook - Rolling Back a Risk Platform Deployment
type: runbook
owner: sre
updated: 2026-08-19
---

# Policy

Rollbacks of production risk services require approval from the on-call engineer of the owning team. Rollbacks that affect model decisions also require a heads-up in #risk-incidents. The incident commander records the approval in the incident ticket.

# Procedure (ArgoCD)

1. Identify the last known-good version from deployment history (the `previous_version` field of the suspect deployment).
2. `argocd app rollback <service> <revision>`; for fraud-preproc this also reloads scaler artifacts from the pinned artifact bundle.
3. Verify: pod readiness, `out_of_range_feature_rate` back to < 0.5% within 15 minutes, precision recovering within 2-3 hours (label lag).
4. Keep the bad version blocked in ArgoCD until a fix with passing parity tests is merged.

# Rollback safety notes

- fraud-preproc is stateless; rollback is safe and takes ~4 minutes.
- fraud-model-server rollbacks must specify the model version explicitly.
- Never roll back platform-infra node pools during business hours without SRE lead approval.
