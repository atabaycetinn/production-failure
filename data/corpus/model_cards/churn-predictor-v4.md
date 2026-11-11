---
id: mc-churn-predictor-v4
title: Model Card - churn-predictor v4
type: model_card
owner: growth-ml
model: churn-predictor
version: v4
updated: 2026-07-02
---

# Summary

LightGBM model predicting 30-day merchant churn, scored nightly in batch. Precision at top decile 0.61, recall 0.44.

# Features

Account tenure, monthly processing volume (log-scaled), support ticket count, dispute rate, pricing tier. Volume features use a log1p transform followed by robust scaling.

# Monitoring

Weekly precision on realized churn; drift checks on processing volume. Batch scoring failures page growth-ml on-call.
