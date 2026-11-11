---
id: mc-fraud-detector-v12
title: Model Card - fraud-detector v12
type: model_card
owner: risk-ml-platform
model: fraud-detector
version: v12
updated: 2026-10-28
---

# Summary

Gradient-boosted tree classifier (XGBoost 2.1) predicting probability that a card-not-present payment is fraudulent. Promoted to production on 2026-10-28 (deployment DEP-2026-10-28-01) after a two-week shadow period.

# Performance at launch

| Metric | Offline (holdout) | Shadow (production traffic) |
|---|---|---|
| Precision @ threshold | 0.93 | 0.92 |
| Recall @ threshold | 0.80 | 0.81 |
| Alert rate | 1.0% | 1.1% |

Expected production precision band: 0.90-0.94. Alert if below 0.88 for 3 consecutive hours.

# Decision threshold

0.83 on calibrated probability. Owned by risk-engineering; changes go through decision-service config.

# Feature contract

The model consumes exactly these features, produced by fraud-preproc. Training ranges are the 0.1-99.9 percentile of the training set after transformation.

| Feature | Transformation | Training range |
|---|---|---|
| amount_log_scaled | standard_scale(log1p(amount_usd)) using scaler `amount_v3` (mean 3.91, std 1.28 in log space) | [-3.10, 4.20] |
| geo_distance_log_scaled | standard_scale(log1p(km between billing and IP geo)) | [-1.90, 3.70] |
| txn_velocity_1h | count of card transactions in trailing hour, capped at 50 | [0, 22] |
| account_age_days_scaled | standard_scale(min(account_age_days, 3650)) | [-1.40, 2.60] |
| merchant_risk_score | provided by feature-ingest, already in [0, 1] | [0, 1] |
| device_trust_score | provided by feature-ingest, already in [0, 1] | [0, 1] |

`amount_log_scaled` is the single most important feature (gain share 31%). Tree models do not extrapolate: values beyond the training range fall into the outermost split, which for amount is strongly associated with fraud (very large purchases). A broken amount transformation therefore manifests as a sharp increase in false positives (precision drop) with recall roughly unchanged or slightly up.

# Known limitations

- Precision degrades ~2 points during major shopping holidays because of unusual legitimate high-value purchases. This is gradual and tracks traffic volume.
- Sensitive to label delay: if analyst labels lag, hourly precision becomes noisy (but not systematically lower).
