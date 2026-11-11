"""Render a report as Markdown with an evidence appendix."""

from __future__ import annotations

from .actions import ActionStore
from .evidence import EvidenceLedger
from .models import FindingStatus, Report

_BADGE = {FindingStatus.VERIFIED: "VERIFIED", FindingStatus.HYPOTHESIS: "HYPOTHESIS", FindingStatus.REFUTED: "REFUTED"}


def _cite(ids: list[str]) -> str:
    return f" [{', '.join(ids)}]" if ids else ""


def report_markdown(report: Report, ledger: EvidenceLedger, actions: ActionStore | None = None) -> str:
    rc = report.root_cause
    out = [
        "# Incident investigation report",
        "",
        "## Summary",
        report.summary,
        "",
        "## Probable root cause",
        f"**{_BADGE[rc.status]}** · confidence **{rc.confidence.value}** · component `{rc.component}`"
        + (f" · change `{rc.change_ref}`" if rc.change_ref else ""),
        "",
        rc.statement + _cite(rc.evidence_ids),
        "",
    ]
    for status, title in ((FindingStatus.VERIFIED, "Verified findings"), (FindingStatus.HYPOTHESIS, "Hypotheses"),
                          (FindingStatus.REFUTED, "Ruled out")):
        items = [f for f in report.findings if f.status is status]
        if items:
            out += [f"## {title}", ""]
            out += [f"- {f.statement}{_cite(f.evidence_ids)}" + (f"\n  - _{f.reasoning}_" if f.reasoning else "")
                    for f in items]
            out.append("")
    if report.timeline:
        out += ["## Timeline", "", "| Time (UTC) | Event | Evidence |", "|---|---|---|"]
        out += [f"| {t.timestamp} | {t.event} | {', '.join(t.evidence_ids)} |" for t in report.timeline]
        out.append("")
    if report.remediation:
        out += ["## Recommended remediation", ""]
        for r in report.remediation:
            line = f"- {r.description}{_cite(r.evidence_ids)}\n  - Why: {r.rationale}"
            if r.action_request_id:
                status = "unknown"
                if actions:
                    try:
                        status = actions.get(r.action_request_id).status.value
                    except Exception:  # noqa: BLE001
                        status = "not found"
                line += f"\n  - Action request `{r.action_request_id}`: **{status}** (requires human approval)"
            out.append(line)
        out.append("")
    if report.open_questions:
        out += ["## Open questions", ""] + [f"- {q}" for q in report.open_questions] + [""]
    if report.validation_warnings:
        out += ["## Harness validation warnings", ""] + [f"- {w}" for w in report.validation_warnings] + [""]
    out += ["## Evidence", "", "| ID | Kind | Source | Summary |", "|---|---|---|---|"]
    out += [f"| {e.id} | {e.kind.value} | {e.source} | {e.summary.replace('|', '/')[:220]} |" for e in ledger.all()]
    return "\n".join(out) + "\n"
