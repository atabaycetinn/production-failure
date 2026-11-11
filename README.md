# mlrca: Agentic RAG for diagnosing production ML failures

An agent that investigates production model degradations. It retrieves internal documentation (RAG), queries telemetry (metrics, deployments, code diffs, logs, feature drift), correlates the evidence, separates **verified findings** from **hypotheses**, and proposes remediations that **require human approval**.

The roadmap, architecture and evaluation strategy are in [docs/PLAN.md](docs/PLAN.md).

## Quickstart

```bash
python3.14 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                                  # offline: tools, ledger, approvals, agent loop (fake client)
.venv/bin/mlrca baseline                             # heuristic, no LLM
export ANTHROPIC_API_KEY=...                         # or `ant auth login`
.venv/bin/mlrca investigate                          # Claude agent on the fraud scenario
.venv/bin/mlrca eval --system both                   # score agent vs. baseline against ground truth
.venv/bin/mlrca actions list
.venv/bin/mlrca actions approve ACT-... --by "your-name"
.venv/bin/mlrca actions execute ACT-...              # dry-run executor
```

Configuration is through environment variables: `MLRCA_MODEL` (default `claude-opus-5-5`), `MLRCA_EFFORT` (default `high`), `MLRCA_MAX_TURNS`, and `MLRCA_FALLBACKS=off` to disable the server-side refusal fallback.

## The baseline scenario: `fraud_precision_drop`

> *Our fraud detection model's precision dropped from 92% to 71% following yesterday's deployment. Investigate the probable root cause.*

**Planted root cause.** `fraud-preproc` v3.8.0 (DEP-2026-11-10-02, commit `9f3c2e1`) moved scalers into a config-driven `ScalerRegistry`. The config for `amount` is missing `pre_transform: log1p`, so raw dollar amounts go through a scaler that was fitted in log space. As a result, `amount_log_scaled` leaves its training range (PSI ≈ 9.7), false positives surge, and precision drops while recall stays flat.

**Red herrings.**
- A k8s node-pool upgrade the same morning
- A dashboard release the same afternoon
- A shadow-mode feature in the same release
- The label-lag pattern from a past incident
- An evening traffic bump
- Release notes claiming "no behavior change"
- A prompt-injection string in a log line

Regenerate the data with `python scripts/generate_scenario.py`. It is seeded and deterministic.

## Repository layout

```
docs/PLAN.md                 plan, architecture, roadmap, eval strategy, risks
src/mlrca/
  models.py                  Evidence, Finding, Report, ActionRequest, DocChunk …
  evidence.py                EvidenceLedger: citation integrity + "verified needs observation" rule
  rag/                       markdown corpus loader, heading-aware chunker, BM25 retriever (Retriever protocol)
  telemetry/                 TelemetryBackend protocol, FixtureBackend, PSI / change-point stats
  agent/
    tools.py                 9 tools (pydantic-validated), dispatcher, generated JSON schemas
    prompts.py               system prompt (static → cached)
    loop.py                  Claude agent loop (Opus 5.5, adaptive thinking, refusal fallback, caching)
  actions/approval.py        pending → approved → executed state machine, dry-run executor
  baselines/heuristic.py     non-LLM RCA baseline
  eval/                      rubric scorer + runner (writes runs/<ts>/…/report.md, result.json)
  render.py                  Markdown report with an evidence appendix
  cli.py                     `mlrca` command
data/corpus/                 architecture, model cards, runbooks, incidents, release notes
data/scenarios/<name>/       scenario.json (+ ground truth), deployments, commits, metrics, logs, features
scripts/generate_scenario.py synthetic scenario generator
tests/                       offline test suite
```

## Agent tools

| Tool | Evidence kind | Notes |
|---|---|---|
| `search_docs` | documentation | BM25 over chunked docs. Can only support *hypotheses* |
| `list_metrics` | none | Metric catalog |
| `query_metric` | metrics | Aggregation, summary stats, automatic change point |
| `list_deployments` | deployment | App, model, infra and dashboard changes |
| `get_code_change` | code_change | Commits and diffs, by deployment or SHA |
| `search_logs` | logs | Counts plus an evenly spaced sample. Logs are marked untrusted |
| `compare_feature_distributions` | feature_drift | PSI per feature, ownership, training ranges, bins |
| `propose_remediation` | none | Creates a **pending** action request. Cannot execute |
| `submit_report` | none | Ends the run. The harness validates citations and downgrades any `verified` claim that lacks observational evidence |

## Baseline results

| system | total | component | change | mechanism | verified | distractors refuted | remediation | no unapproved exec | citations |
|---|---|---|---|---|---|---|---|---|---|
| heuristic | 0.625 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| agent | run `mlrca eval --system both` with an API key | | | | | | | | |

The heuristic finds the right deployment from timing and drift ownership. It can't explain the mechanism, verify causation or rule out alternatives. That gap is the agent's job.

## Extending

- **New data sources:** implement `TelemetryBackend` (for example Prometheus, Loki or ArgoCD) and pass it to `ToolContext`.
- **Better retrieval:** implement `Retriever.search` (hybrid/dense + rerank). The tool contract stays the same.
- **New scenarios:** add `data/scenarios/<name>/` with the same files and a `ground_truth` block.
- **Real remediation:** implement `Executor.run` for your deploy or flag system. Approval checks stay in `actions.execute`.
