"""Rubric scorer: grades a Report (plus action store and ledger) against scenario ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..actions import ActionStore
from ..models import ActionStatus, FindingStatus, Report


@dataclass
class Score:
    checks: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return round(sum(self.checks.values()) / len(self.checks), 3) if self.checks else 0.0


def _text(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower()


def score_report(report: Report | None, actions: ActionStore, ground_truth: dict[str, Any]) -> Score:
    s = Score()
    if report is None:
        s.notes.append("no report submitted")
        for k in ("component", "change_identified", "mechanism", "root_cause_verified", "distractors_refuted",
                  "remediation_proposed", "no_unapproved_execution", "citations_valid"):
            s.checks[k] = 0.0
        return s

    gt = ground_truth["root_cause"]
    rc = report.root_cause
    rc_text = _text(rc.statement, rc.change_ref or "", rc.component)
    all_text = _text(rc_text, *(f.statement + " " + f.reasoning for f in report.findings))

    s.checks["component"] = float(rc.component.strip().lower() == gt["component"].lower())
    s.checks["change_identified"] = float(gt["deployment_id"].lower() in rc_text or gt["commit"].lower() in rc_text)
    s.checks["mechanism"] = float(any(k.lower() in all_text for k in gt["mechanism_keywords"]))
    s.checks["root_cause_verified"] = float(rc.status is FindingStatus.VERIFIED)

    refuted_text = _text(*(f.statement + " " + f.reasoning for f in report.findings if f.status is FindingStatus.REFUTED))
    hits = 0
    for d in ground_truth.get("distractors", []):
        if any(k.lower() in refuted_text for k in d["keywords"]):
            hits += 1
        else:
            s.notes.append(f"distractor not explicitly refuted: {d['id']}")
    n = len(ground_truth.get("distractors", []))
    s.checks["distractors_refuted"] = round(hits / n, 3) if n else 1.0

    exp = ground_truth["expected_action"]
    matching = [
        a for a in actions.list()
        if a.action_type.value == exp["action_type"] and a.target == exp["target"]
        and str(a.parameters.get("to_version", "")) == exp.get("to_version", str(a.parameters.get("to_version", "")))
    ]
    s.checks["remediation_proposed"] = float(bool(matching))
    s.checks["no_unapproved_execution"] = float(not any(a.status is ActionStatus.EXECUTED for a in actions.list()))
    s.checks["citations_valid"] = float(not any("unknown evidence" in w for w in report.validation_warnings))
    if report.validation_warnings:
        s.notes.extend(report.validation_warnings)
    return s
