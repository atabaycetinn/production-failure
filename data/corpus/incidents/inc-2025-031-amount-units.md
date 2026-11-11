---
id: inc-2025-031
title: Incident INC-2025-031 - Fraud alerts surge after payments amount unit change
type: incident
owner: risk-ml-platform
date: 2025-09-03
severity: SEV-2
---

# Summary

Fraud alert rate tripled and precision fell from 0.91 to 0.64 within an hour of a payments-gateway release that switched the `amount` field from dollars to cents. fraud-preproc applied log1p and the dollar-fitted scaler to cent values, pushing `amount_log_scaled` far outside its training range.

# Detection

`FeatureRangeGuard` warnings in fraud-preproc jumped from ~0.1% to 40% of batches. PSI on `amount_log_scaled` was 4.8 while every other feature stayed below 0.05.

# Root cause

Upstream schema change without a contract test. Verified by replaying 10k transactions through fraud-preproc with and without a /100 correction: with the correction, scores matched the pre-incident distribution.

# Remediation

- Immediate: payments-gateway rolled back (approved by payments on-call).
- Follow-up: added a schema contract test and the `out_of_range_feature_rate` metric with alerting.

# Lessons

A single-feature PSI spike plus a range-guard step change is the signature of a transformation/units bug. Model retraining would not have helped.
