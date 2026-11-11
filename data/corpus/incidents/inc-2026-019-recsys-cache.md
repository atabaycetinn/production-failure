---
id: inc-2026-019
title: Incident INC-2026-019 - Merchant recommendations CTR drop from stale embedding cache
type: incident
owner: growth-ml
date: 2026-08-22
severity: SEV-3
---

# Summary

Click-through rate on merchant recommendations dropped 18% after a Redis cluster migration left the embedding cache serving vectors from a model version two releases old.

# Root cause

Cache key did not include the embedding model version. Verified by comparing cached vectors to the registry artifact.

# Remediation

Flushed cache, added model version to the key, added a canary that compares cached and fresh embeddings.
