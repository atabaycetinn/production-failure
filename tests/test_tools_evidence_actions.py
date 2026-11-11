import json

import pytest

from mlrca.actions import ActionStore, ApprovalError, execute
from mlrca.agent import TOOL_DEFINITIONS
from mlrca.models import ActionStatus, ActionType, Confidence, Finding, FindingStatus, Report, RootCause


def test_tool_schemas_are_flat_objects():
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert names[-1] == "submit_report" and len(names) == len(set(names))
    for t in TOOL_DEFINITIONS:
        assert t["input_schema"]["type"] == "object"
        assert "$ref" not in json.dumps(t["input_schema"]) and t["description"]
    report = next(t for t in TOOL_DEFINITIONS if t["name"] == "submit_report")
    assert "validation_warnings" not in report["input_schema"]["properties"]


def test_observational_tools_register_evidence(ctx):
    out = ctx.dispatch("query_metric", {"entity": "fraud-detector", "metric": "precision"})
    body = json.loads(out.content)
    assert not out.is_error and body["evidence_id"] == "EV-001"
    assert body["change_point"]["timestamp"] == "2026-11-10T15:00:00Z"
    assert ctx.ledger.get("EV-001").kind.observational


def test_catalog_tool_does_not_register_evidence(ctx):
    out = ctx.dispatch("list_metrics", {"entity": "fraud-detector"})
    assert out.evidence_id is None and len(ctx.ledger) == 0


def test_invalid_inputs_become_tool_errors(ctx):
    assert ctx.dispatch("query_metric", {"entity": "fraud-detector"}).is_error  # missing metric
    assert ctx.dispatch("get_code_change", {"deployment_id": "X", "sha": "Y"}).is_error
    assert ctx.dispatch("query_metric", {"entity": "nope", "metric": "precision"}).is_error
    assert ctx.dispatch("no_such_tool", {}).is_error


def test_code_change_and_logs(ctx):
    code = json.loads(ctx.dispatch("get_code_change", {"deployment_id": "DEP-2026-11-10-02"}).content)
    assert [c["sha"] for c in code["commits"]] == ["9f3c2e1", "4b7d0aa", "c81e5f2"]
    logs = json.loads(ctx.dispatch("search_logs", {
        "start": "2026-11-10T14:00:00Z", "end": "2026-11-10T16:00:00Z", "service": "fraud-preproc",
        "contains": "ScalerRegistry", "limit": 5}).content)
    assert logs["total_matches"] == 1 and "pre_transform=none" in logs["sample"][0]["message"]


def test_feature_drift_flags_amount(ctx):
    out = json.loads(ctx.dispatch("compare_feature_distributions", {
        "model": "fraud-detector", "baseline_start": "2026-11-09T12:00:00Z", "baseline_end": "2026-11-10T12:00:00Z",
        "current_start": "2026-11-10T15:00:00Z", "current_end": "2026-11-11T09:00:00Z"}).content)
    top = out["features"][0]
    assert top["feature"] == "amount_log_scaled" and top["psi"] > 1 and "bins" in top
    assert all(f["psi"] < 0.1 for f in out["features"][1:])


def test_propose_requires_known_evidence(ctx):
    bad = ctx.dispatch("propose_remediation", {"action_type": "rollback_deployment", "target": "fraud-preproc",
                                               "rationale": "x", "evidence_ids": ["EV-999"]})
    assert bad.is_error and not ctx.actions.list()
    ctx.dispatch("list_deployments", {})
    ok = json.loads(ctx.dispatch("propose_remediation", {
        "action_type": "rollback_deployment", "target": "fraud-preproc", "parameters": {"to_version": "v3.7.2"},
        "rationale": "x", "evidence_ids": ["EV-001"]}).content)
    assert ok["status"] == "pending_approval"


def test_ledger_downgrades_unsupported_verified_claims(ctx):
    ctx.dispatch("search_docs", {"query": "release notes fraud-preproc"})  # EV-001 documentation
    ctx.dispatch("list_deployments", {})  # EV-002 deployment
    report = Report(
        summary="s",
        root_cause=RootCause(statement="docs say so", status=FindingStatus.VERIFIED, confidence=Confidence.HIGH,
                             component="fraud-preproc", evidence_ids=["EV-001", "EV-404"]),
        findings=[Finding(statement="deploy happened", status=FindingStatus.VERIFIED, evidence_ids=["EV-002"])],
    )
    out = ctx.ledger.validate_report(report)
    assert out.root_cause.status is FindingStatus.HYPOTHESIS
    assert out.root_cause.evidence_ids == ["EV-001"]
    assert out.findings[0].status is FindingStatus.VERIFIED
    assert any("unknown evidence" in w for w in out.validation_warnings)


def test_approval_state_machine(tmp_path):
    store = ActionStore(tmp_path / "actions.json")
    req = store.propose(ActionType.ROLLBACK_DEPLOYMENT, "fraud-preproc", {"to_version": "v3.7.2"}, "r", ["EV-001"])
    with pytest.raises(ApprovalError):
        execute(store, req.id)  # not approved
    with pytest.raises(ApprovalError):
        store.approve(req.id, "")  # anonymous approval
    store.approve(req.id, "oncall-alex")
    with pytest.raises(ApprovalError):
        store.reject(req.id, "someone")  # already decided
    done = execute(store, req.id)
    assert done.status is ActionStatus.EXECUTED and "dry-run" in done.execution_log[-1]
    reloaded = ActionStore(tmp_path / "actions.json")
    assert reloaded.get(req.id).status is ActionStatus.EXECUTED
