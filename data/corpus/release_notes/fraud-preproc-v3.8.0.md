---
id: rn-fraud-preproc-3.8.0
title: Release Notes - fraud-preproc v3.8.0
type: release_notes
service: fraud-preproc
version: v3.8.0
date: 2026-11-10
owner: risk-ml-platform
---

# Highlights

- **Scaler consolidation.** Numeric scalers are now loaded through a single `ScalerRegistry` configured in `config/transforms.yaml`, replacing per-feature scaler code. Pure refactor; **no behavior change expected.**
- **merchant_category_v2 (shadow).** New categorical feature computed and logged for future model versions. Not consumed by fraud-detector v12.
- **Dependency bump.** pandas 2.2.3 -> 2.3.1.

# Validation

Unit tests pass. Parity suite skipped for this release because the refactor was considered behavior-preserving (ticket RISKML-2291).

# Rollback

Previous version: v3.7.2. Stateless service; standard ArgoCD rollback.
