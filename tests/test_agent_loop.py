"""Offline tests of the agent loop with a scripted fake Anthropic client."""

from types import SimpleNamespace as NS

from mlrca.agent import Investigator
from mlrca.eval import run_one, score_report

from .conftest import SCENARIO


def _usage():
    return NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def _msg(blocks, stop):
    return NS(content=blocks, stop_reason=stop, usage=_usage(), stop_details=None)


def _tool(i, name, inp):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=inp)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **params):
        self.calls.append({**params, "messages": list(params["messages"])})
        return self.script.pop(0)


REPORT = {
    "summary": "fraud-preproc v3.8.0 dropped log1p for amount; roll back.",
    "root_cause": {
        "statement": "DEP-2026-11-10-02 (commit 9f3c2e1) removed the log1p pre_transform for amount",
        "status": "verified", "confidence": "high", "component": "fraud-preproc",
        "change_ref": "DEP-2026-11-10-02 / 9f3c2e1", "evidence_ids": ["EV-001", "EV-002", "EV-003"],
    },
    "findings": [
        {"statement": "node pool upgrade (DEP-2026-11-10-01) ended before the drop", "status": "refuted",
         "evidence_ids": ["EV-001", "EV-002"]},
        {"statement": "label_coverage stayed ~0.97, so not label lag", "status": "refuted", "evidence_ids": ["EV-001"]},
        {"statement": "dashboard change is visualization only", "status": "refuted", "evidence_ids": ["EV-002"]},
    ],
    "remediation": [{"description": "Roll back fraud-preproc", "action_request_id": "ACT-x", "rationale": "r",
                     "evidence_ids": ["EV-003"]}],
}


def test_loop_runs_tools_nudges_and_accepts_report(ctx, settings):
    script = [
        _msg([NS(type="text", text="Checking metrics and deploys."),
              _tool(1, "query_metric", {"entity": "fraud-detector", "metric": "precision"}),
              _tool(2, "list_deployments", {})], "tool_use"),
        _msg([_tool(3, "get_code_change", {"deployment_id": "DEP-2026-11-10-02"}),
              _tool(4, "propose_remediation", {"action_type": "rollback_deployment", "target": "fraud-preproc",
                                               "parameters": {"to_version": "v3.7.2"}, "rationale": "r",
                                               "evidence_ids": ["EV-003"]})], "tool_use"),
        _msg([NS(type="text", text="I think I'm done.")], "end_turn"),  # forgot to submit -> nudge
        _msg([_tool(5, "submit_report", REPORT)], "tool_use"),
    ]
    client = FakeClient(script)
    res = Investigator(ctx, client=client, settings=settings).run("why did precision drop?", "fraud-detector")

    assert res.stop == "report_submitted" and res.turns == 4
    assert [s["tool"] for s in res.trace] == ["query_metric", "list_deployments", "get_code_change",
                                              "propose_remediation", "submit_report"]
    assert res.report.root_cause.status.value == "verified"
    assert res.usage["input_tokens"] == 400

    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert first["thinking"] == {"type": "adaptive"} and first["output_config"] == {"effort": "high"}
    # both tool results returned together in one user message, in call order
    tool_results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in tool_results] == ["tu_1", "tu_2"]
    assert "submit_report" in client.calls[3]["messages"][-1]["content"]  # the nudge

    score = score_report(res.report, ctx.actions, ctx.backend.scenario["ground_truth"])
    assert score.total == 1.0, score


def test_invalid_report_is_returned_as_tool_error(ctx, settings):
    bad = {"summary": "x"}  # missing root_cause
    script = [
        _msg([_tool(1, "submit_report", bad)], "tool_use"),
        _msg([_tool(2, "submit_report", {**REPORT, "root_cause": {**REPORT["root_cause"], "evidence_ids": []}})],
             "tool_use"),
    ]
    client = FakeClient(script)
    res = Investigator(ctx, client=client, settings=settings).run("q")
    err = client.calls[1]["messages"][-1]["content"][0]
    assert err["is_error"] is True
    # no evidence collected at all -> verified claims downgraded by the harness
    assert res.report.root_cause.status.value == "hypothesis"
    assert res.report.validation_warnings


def test_heuristic_baseline_scores_below_ceiling(settings):
    out = run_one(SCENARIO, "baseline", settings)
    c = out.score.checks
    assert c["component"] == 1 and c["change_identified"] == 1 and c["remediation_proposed"] == 1
    assert c["root_cause_verified"] == 0 and c["mechanism"] == 0  # the gap the agent must close
    assert c["no_unapproved_execution"] == 1
