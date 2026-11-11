---
id: arch-fraud-platform
title: Fraud Detection Platform Architecture
type: architecture
owner: risk-ml-platform
updated: 2026-09-14
---

# Overview

The fraud detection platform scores every card-not-present payment in real time (p99 budget: 150 ms end to end). Scores above the decision threshold are routed to the analyst review queue and the transaction is held.

# Request path

1. **payments-gateway** publishes `payment.authorized` events to Kafka topic `payments.v3`.
2. **feature-ingest** joins the event with merchant and device enrichment (merchant_risk_score, device_trust_score) and writes raw features to the online feature store.
3. **fraud-preproc** applies the model's feature transformations (log transforms, standard scaling, velocity windows) and produces the model-ready feature vector. Transformations must match the training pipeline exactly; the contract lives in the model card.
4. **fraud-model-server** hosts the active model version (currently `fraud-detector` v12, XGBoost) and returns a fraud probability.
5. **decision-service** applies the threshold (see model card) and business rules, and emits `fraud.decision` events.

# Feedback and monitoring path

- **label-pipeline** ingests analyst review outcomes (confirmed fraud / false positive) and chargebacks. Analyst labels for alerts typically arrive within 2-6 hours; chargebacks lag 30-60 days.
- **model-monitor** computes hourly precision on analyst-labeled alerts, recall against a 5% random audit sample, alert rate, score distribution, and per-feature drift (PSI against the training reference). These series are the source of truth; dashboards only read them.
- **fraud-monitoring-dashboard** (Grafana) visualizes model-monitor series. Dashboard changes do not alter the underlying metrics.

# Ownership

| Service | Owner team | Deploys via |
|---|---|---|
| feature-ingest | data-platform | ArgoCD |
| fraud-preproc | risk-ml-platform | ArgoCD |
| fraud-model-server | risk-ml-platform | ArgoCD + model registry |
| decision-service | risk-engineering | ArgoCD |
| label-pipeline | risk-ops-data | Airflow |
| platform-infra (k8s) | sre | Terraform |

# Training-serving consistency

Training uses the offline implementation of the same transformations (`riskml.features.transforms`). fraud-preproc imports the serving implementation. Any change to fraud-preproc transformations must be validated with the parity test suite (`tests/parity/`) against a frozen sample of training rows. Historically, most serious fraud-model incidents were training-serving skew introduced by preprocessing or upstream schema changes, not model changes.
