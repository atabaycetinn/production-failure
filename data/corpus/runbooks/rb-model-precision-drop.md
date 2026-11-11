---
id: rb-precision-drop
title: Runbook - Fraud Model Precision Drop
type: runbook
owner: risk-ml-platform
updated: 2026-06-02
---

# When to use

Use when model-monitor fires `FraudPrecisionLow` (precision < 0.88 for 3 consecutive hours) or an engineer reports a sudden precision drop.

# Triage checklist

1. **Confirm the drop is real.** Check model-monitor `precision` and `label_coverage`. If label coverage fell below 0.85, the drop is likely a measurement artifact from label-pipeline lag (see INC-2026-007). Do not roll anything back until coverage is confirmed healthy.
2. **Locate the change point.** Find the first hour where precision left the 0.90-0.94 band. Note whether the change is a step (deploy-like) or gradual (seasonality / drift).
3. **List changes in the 24h before the change point.** Deployments across feature-ingest, fraud-preproc, fraud-model-server, decision-service, and infra. Config changes to the decision threshold count as deployments.
4. **Check feature drift.** Compare per-feature distributions between a 24h window before and after the change point. PSI > 0.25 on a single feature with others stable strongly suggests a transformation or upstream bug for that feature, not real-world drift.
5. **Check the range guard.** fraud-preproc logs `FeatureRangeGuard` warnings when features fall outside the training range. A step increase in `out_of_range_feature_rate` right after a deploy is a strong signal of a preprocessing regression.
6. **Read the diff.** For the suspect deployment, inspect commits touching `features/`, `scalers/`, or `config/transforms.yaml`.
7. **Rule out infrastructure.** Latency or error spikes on fraud-model-server cause fallbacks to the rules engine, which affects alert volume but not model precision on scored traffic.

# Mitigation options (all require on-call approval)

| Situation | Action |
|---|---|
| Preprocessing regression confirmed | Roll back fraud-preproc to the last good version (see RB-ROLLBACK) |
| Model regression | Roll back fraud-model-server model version via model registry |
| Threshold misconfiguration | Restore threshold in decision-service config |
| Cause unknown, business impact severe | Temporarily raise threshold to 0.90 to cut false positives (reduces recall; requires risk lead approval) |

Never adjust the threshold to mask a preprocessing bug for more than a few hours: it changes the recall/precision trade-off for all traffic.
