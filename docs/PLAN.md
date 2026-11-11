# Agentic RAG for Production ML Failure Diagnosis: Plan

## 1. Problem statement

An on-call engineer asks in plain language why a production model degraded, for example:

> Our fraud detection model's precision dropped from 92% to 71% following yesterday's deployment. Investigate the probable root cause.

The system must:

1. **Retrieve** relevant documentation (architecture, model cards, runbooks, past incidents, release notes) with RAG.
2. **Investigate** with an agent that queries logs, monitoring metrics, deployment history, code changes and feature distributions.
3. **Correlate** evidence across sources, for example a preprocessing code change lining up with feature drift and the metric drop.
4. **Separate verified findings from hypotheses.** Every claim cites evidence, and a claim counts as *verified* only when observational data (metrics, logs, diffs) supports it, not documentation alone.
5. **Recommend remediation** with citations. Operational changes are proposed as pending action requests and run only after explicit human approval.

## 2. Design principles

| Principle | How it shows up in the code |
|---|---|
| Evidence first | Each tool call creates an `Evidence` record (`EV-001`…) in an `EvidenceLedger`. The final report may cite only IDs that exist in the ledger. |
| Verified ≠ plausible | The harness, not the prompt, enforces the rule: a finding marked `verified` with no observational evidence is downgraded to `hypothesis`, and a warning is recorded. |
| Docs are claims, telemetry is observation | Release notes saying "no behavior change expected" are evidence *of a claim*. The agent has to check them against metrics, logs and diffs. |
| Read-only agent | The agent's tools only read data. `propose_remediation` creates a **pending** `ActionRequest`. It cannot approve or run one. |
| Human approval gate | `mlrca actions approve <id>` followed by `mlrca actions execute <id>`. The executor refuses anything that hasn't been approved. |
| Swappable backends | `TelemetryBackend` and `Retriever` are protocols. The baseline uses fixture files and BM25. Production backends (Prometheus, Loki/ELK, Argo/Spinnaker, GitHub, a vector DB) plug in behind the same interfaces. |
| Measurable | Scenarios ship with ground truth. A scorer grades the agent and a non-LLM heuristic baseline with the same rubric. |

## 3. Architecture

```
                      ┌──────────────────────────────────────────────┐
 engineer question ──►│ Investigator (Claude agent loop)             │
                      │  system prompt: method + evidence rules      │
                      └───────┬───────────────────────────┬──────────┘
                              │ tool calls (read-only)    │ submit_report
            ┌─────────────────┼───────────────────┐       ▼
            ▼                 ▼                   ▼   Report (pydantic)
      search_docs      telemetry tools       propose_remediation
      (RAG: BM25 →     deployments, metrics,      │
       hybrid later)   logs, diffs, drift         ▼
            │                 │             ActionStore (pending)
            ▼                 ▼                   │  human: approve/reject
      Corpus (md +     TelemetryBackend           ▼
      front matter)    (fixtures → real)    Executor (dry-run → real)
            └──────► EvidenceLedger ◄────────┘
```

### Components (`src/mlrca/`)

| Module | Responsibility |
|---|---|
| `models.py` | Domain types: `Evidence`, `Finding`, `Report`, `ActionRequest`, and the rest |
| `evidence.py` | `EvidenceLedger`: registers tool outputs, validates citations, enforces the verified rule |
| `rag/` | Corpus loading (markdown + front matter), heading-aware chunking, BM25 retriever, metadata filters |
| `telemetry/` | `TelemetryBackend` protocol, `FixtureBackend`, stats helpers (PSI, change-point detection) |
| `actions/` | `ActionStore` (JSON-persisted), approval workflow, `DryRunExecutor` |
| `agent/` | Tool schemas and dispatch, system prompt, the Claude agent loop |
| `baselines/` | Non-LLM heuristic RCA (change point → nearby deploys → feature drift → ownership) |
| `eval/` | Scenario loading, rubric scorer, runner |
| `cli.py` | `mlrca investigate / baseline / eval / actions …` |

### LLM configuration

- Model: `claude-opus-5-5`, effort `high` (override with `MLRCA_MODEL` and `MLRCA_EFFORT`). Adaptive thinking is always on for this model.
- Server-side refusal fallback (`fallbacks: "default"`) is enabled.
- The system prompt and tool definitions are cached. They are a stable prefix, and the conversation tail uses automatic caching.
- A manual agent loop, so the harness can end on `submit_report`, keep a per-step trace, validate every tool input with pydantic, and nudge the agent when it stops without a report.

## 4. Roadmap

### Phase 0: Baseline (this commit)
- [x] Synthetic but realistic fraud-precision scenario: 7 days of hourly metrics, deployments, logs, commits with diffs, feature histograms, plus red herrings (a node-pool upgrade, a shadow-mode feature, a label-delay hypothesis, release notes that say "no behavior change")
- [x] Documentation corpus: architecture, model card, two runbooks, two past incidents, release notes
- [x] BM25 RAG with doc-type filters and chunk-level citations
- [x] Agent loop with 9 tools, an evidence ledger and the verified/hypothesis rule
- [x] Approval-gated actions with a dry-run executor
- [x] Heuristic (non-LLM) baseline and an eval rubric
- [x] Offline unit tests, including the agent loop with a scripted fake client

### Phase 1: Retrieval quality
- Hybrid retrieval (BM25 + dense embeddings) with a reranker, in a vector store such as pgvector, Qdrant or LanceDB
- Ingestion connectors: Confluence/Notion, Git (release notes, ADRs), incident tracker (PagerDuty/Jira), model registry (MLflow) model cards
- Freshness and authority metadata, so a runbook outranks a stale wiki page
- A retrieval eval set (query → relevant chunk IDs), measured with recall@k and MRR

### Phase 2: Real telemetry backends
- `PrometheusBackend` / `DatadogBackend` for metrics, `LokiBackend` / `ElasticBackend` for logs
- `ArgoCDBackend` / `GitHubBackend` for deployments and diffs, plus the feature store / monitoring tool (Evidently, Arize, WhyLabs) for drift
- Per-tool output budgets and summarization, so large log pulls don't flood the context
- Read-only service credentials, with an audit log of every query

### Phase 3: Agent depth
- Specialist subagents (metrics analyst, code-change analyst, data-quality analyst) coordinated by a lead investigator, once single-agent traces show context pressure
- Statistical tools: counterfactual replay (re-score a sample with the previous preprocessing version), segment analysis (which merchant categories or amount bands degraded)
- Memory of past investigations, saved automatically as incident reports in the corpus

### Phase 4: Operations and trust
- Slack/PagerDuty front end, and approval through the existing change-management flow
- Real executors (Argo rollback, feature-flag toggle) behind role-based approval and two-person rules for high-risk actions
- Post-incident feedback loop: engineers label reports as correct or incorrect, which feeds the eval set
- Cost and latency dashboards per investigation

## 5. Evaluation strategy

| Layer | Metric |
|---|---|
| Retrieval | recall@k and MRR on labeled query→chunk pairs |
| Root cause | Correct component, deployment and commit identified |
| Epistemics | The root cause is `verified` with observational evidence, distractors are explicitly refuted, and no fabricated evidence IDs appear |
| Remediation | The correct action is proposed, is pending (not executed), and cites evidence |
| Efficiency | Tool calls, tokens and wall-clock time per investigation |

Grow the scenario set to roughly 20–30 cases covering: preprocessing bugs, upstream schema changes, label pipeline delays (a false alarm), threshold or config changes, training-serving skew from a retrain, infrastructure degradation, seasonal drift with no bug, and multiple simultaneous changes. Include at least a few cases where the right answer is "inconclusive", to test calibration.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hallucinated causal story | Citation validation, the harness-enforced verified rule, distractor scenarios in the eval |
| Prompt injection through logs or docs | Tool output is wrapped as data, the agent has no write tools, and actions need human approval |
| Context blow-up from large telemetry | Aggregate on the server side (summaries, top-k), cap tool outputs, add context editing/compaction later |
| Over-trusting documentation | Docs are tagged as claims and can't verify a finding on their own |
| Unsafe remediation | Pending → approved → executed state machine. The executor checks status, and the dry run is the default |
