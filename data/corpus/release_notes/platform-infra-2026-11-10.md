---
id: rn-platform-infra-2026-11-10
title: Change Notice - risk-prod node pool upgrade (Kubernetes 1.31 -> 1.32)
type: release_notes
service: platform-infra
date: 2026-11-10
owner: sre
---

# Change

Rolling upgrade of the `risk-prod` node pool to Kubernetes 1.32, 2026-11-10 11:00-11:40 UTC. Pods are drained node by node.

# Expected impact

Brief p99 latency increase and a small number of 503s on services with only two replicas during drains. When fraud-model-server is unavailable, decision-service falls back to the rules engine for affected requests. No change to application code, models or configuration.
