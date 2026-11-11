---
id: inc-2026-007
title: Incident INC-2026-007 - Apparent precision drop caused by label-pipeline lag
type: incident
owner: risk-ops-data
date: 2026-03-17
severity: SEV-3
---

# Summary

Hourly precision appeared to fall to 0.74 for six hours. No model, feature or threshold change had occurred. The label-pipeline Airflow DAG was stuck, so most alerts had no analyst label yet, and the few labeled ones were the fast-tracked confirmed-fraud cases mixed with auto-closed false positives.

# Signature

- `label_coverage` dropped from 0.97 to 0.41.
- Alert rate and score distribution were unchanged.
- No feature drift.

# Resolution

Restarted the DAG; precision recomputed to 0.92 once labels backfilled. No rollback was needed.

# Lesson

Always check label coverage before acting on a precision drop. A real model/feature regression also moves alert rate and score distribution; a measurement artifact does not.
