"""Non-LLM heuristic RCA baseline.

Pipeline: change point on the target metric -> deployments in the preceding 24h ->
feature drift before/after -> rank deployments by (owns the drifted feature, proximity,
change type) -> propose rollback of the top candidate.

It uses the same ToolContext as the agent (so evidence and actions are recorded the
same way) but cannot read diffs or docs semantically. All causal claims stay hypotheses.
That is the bar the agent has to clear.
"""

from __future__ import annotations

import json
from datetime import timedelta

from ..agent.tools import ToolContext
from ..models import (
    Confidence,
    Finding,
    FindingStatus,
    RemediationItem,
    Report,
    RootCause,
    TimelineEvent,
)
from ..telemetry import fmt_ts, parse_ts


def _call(ctx: ToolContext, name: str, **kwargs) -> tuple[dict, str | None]:
    out = ctx.dispatch(name, kwargs)
    if out.is_error:
        raise RuntimeError(out.content)
    return json.loads(out.content), out.evidence_id


def run_heuristic(ctx: ToolContext, model: str, metric: str = "precision") -> Report:
    now = ctx.backend.now()
    metric_data, ev_metric = _call(ctx, "query_metric", entity=model, metric=metric,
                                   start=fmt_ts(now - timedelta(days=7)), end=fmt_ts(now))
    cp = metric_data.get("change_point")
    if not cp or cp["score"] < 5:
        return Report(
            summary=f"No sustained shift detected in {model}/{metric}.",
            root_cause=RootCause(statement="No change point found", status=FindingStatus.HYPOTHESIS,
                                 confidence=Confidence.LOW, component="unknown", evidence_ids=[ev_metric]),
        )
    cp_t = parse_ts(cp["timestamp"])

    deps_data, ev_deps = _call(ctx, "list_deployments", start=fmt_ts(cp_t - timedelta(hours=24)), end=fmt_ts(cp_t))
    drift, ev_drift = _call(
        ctx, "compare_feature_distributions", model=model,
        baseline_start=fmt_ts(cp_t - timedelta(hours=26)), baseline_end=fmt_ts(cp_t - timedelta(hours=2)),
        current_start=fmt_ts(cp_t), current_end=fmt_ts(now),
    )
    drifted = [f for f in drift["features"] if f.get("psi", 0) > 0.25]
    drifted_owners = {f["owner_service"] for f in drifted}

    def score(d: dict) -> float:
        hours_before = (cp_t - parse_ts(d["deployed_at"])).total_seconds() / 3600
        s = 1.0 / (1.0 + max(hours_before, 0))
        s += 2.0 if d["service"] in drifted_owners else 0.0
        s += {"application": 0.5, "model": 0.5, "config": 0.4}.get(d["change_type"], 0.0)
        return s

    ranked = sorted(deps_data["deployments"], key=score, reverse=True)
    findings = [
        Finding(statement=f"{model} {metric} shifted {cp['before_mean']} -> {cp['after_mean']} at {cp['timestamp']}",
                status=FindingStatus.VERIFIED, evidence_ids=[ev_metric]),
    ]
    for f in drifted:
        findings.append(Finding(statement=f"Feature {f['feature']} (owner {f['owner_service']}) drifted, PSI={f['psi']}",
                                status=FindingStatus.VERIFIED, evidence_ids=[ev_drift]))
    timeline = [TimelineEvent(timestamp=d["deployed_at"], event=f"{d['id']} {d['service']} {d['version']}",
                              evidence_ids=[ev_deps]) for d in deps_data["deployments"]]
    timeline.append(TimelineEvent(timestamp=cp["timestamp"], event=f"{metric} change point", evidence_ids=[ev_metric]))

    if not ranked:
        return Report(
            summary=f"{metric} shifted at {cp['timestamp']} with no deployment in the preceding 24h.",
            root_cause=RootCause(statement="Unexplained shift; investigate data or upstream changes",
                                 status=FindingStatus.HYPOTHESIS, confidence=Confidence.LOW, component="unknown",
                                 evidence_ids=[ev_metric, ev_drift]),
            findings=findings, timeline=timeline,
        )

    top = ranked[0]
    change_ref = top["id"]
    evidence = [ev_metric, ev_deps, ev_drift]
    if top.get("commits") and drifted:
        code, ev_code = _call(ctx, "get_code_change", deployment_id=top["id"])
        evidence.append(ev_code)
        stem = drifted[0]["feature"].split("_")[0]
        hits = [c["sha"] for c in code["commits"] if any(stem in f["diff"] for f in c["files"])]
        if hits:
            change_ref += f" / commit {hits[0]}"
            findings.append(Finding(statement=f"Commit {hits[0]} in {top['id']} touches code mentioning '{stem}'",
                                    status=FindingStatus.HYPOTHESIS, evidence_ids=[ev_code]))
    for other in ranked[1:]:
        findings.append(Finding(statement=f"{other['id']} ({other['service']}) is a less likely cause (lower heuristic score)",
                                status=FindingStatus.HYPOTHESIS, evidence_ids=[ev_deps]))

    remediation = []
    if top.get("previous_version"):
        act, _ = _call(ctx, "propose_remediation", action_type="rollback_deployment", target=top["service"],
                       parameters={"to_version": top["previous_version"]},
                       rationale=f"Heuristic top candidate for {metric} drop", evidence_ids=evidence, risk="medium")
        remediation.append(RemediationItem(
            description=f"Roll back {top['service']} to {top['previous_version']}",
            action_request_id=act["action_request_id"], rationale="Closest preceding change owning the drifted feature",
            evidence_ids=evidence,
        ))

    report = Report(
        summary=f"{metric} dropped at {cp['timestamp']}; most likely related change: {top['id']} ({top['service']} {top['version']}).",
        root_cause=RootCause(
            statement=f"Deployment {top['id']} of {top['service']} {top['version']} (temporal proximity + ownership of drifted features)",
            status=FindingStatus.HYPOTHESIS, confidence=Confidence.MEDIUM if drifted else Confidence.LOW,
            component=top["service"], change_ref=change_ref, evidence_ids=evidence,
        ),
        findings=findings, timeline=sorted(timeline, key=lambda t: t.timestamp), remediation=remediation,
        open_questions=["Mechanism not established: heuristic baseline does not read diffs or documentation."],
    )
    return ctx.ledger.validate_report(report)
