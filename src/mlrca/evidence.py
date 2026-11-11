"""Evidence ledger: every tool observation gets an ID the report must cite."""

from __future__ import annotations

from typing import Any

from .models import Evidence, EvidenceKind, FindingStatus, Report


class EvidenceLedger:
    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}

    def add(self, kind: EvidenceKind, source: str, query: dict[str, Any], summary: str, data: Any = None) -> Evidence:
        ev = Evidence(
            id=f"EV-{len(self._items) + 1:03d}", kind=kind, source=source, query=query, summary=summary, data=data
        )
        self._items[ev.id] = ev
        return ev

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    def has_observational(self, evidence_ids: list[str]) -> bool:
        return any(e in self._items and self._items[e].kind.observational for e in evidence_ids)

    # ------------------------------------------------------------------ validation

    def validate_report(self, report: Report) -> Report:
        """Enforce citation integrity and the verified-requires-observation rule.

        Mutates and returns ``report``: unknown evidence IDs are stripped, and any
        ``verified`` claim without observational evidence is downgraded to
        ``hypothesis``. Each correction is recorded in ``validation_warnings``.
        """
        warnings: list[str] = []

        def clean(ids: list[str], where: str) -> list[str]:
            unknown = [i for i in ids if i not in self._items]
            if unknown:
                warnings.append(f"{where}: removed unknown evidence IDs {unknown}")
            return [i for i in ids if i in self._items]

        def enforce(status: FindingStatus, ids: list[str], where: str) -> FindingStatus:
            if status is FindingStatus.VERIFIED and not self.has_observational(ids):
                warnings.append(f"{where}: marked verified without observational evidence -> downgraded to hypothesis")
                return FindingStatus.HYPOTHESIS
            return status

        rc = report.root_cause
        rc.evidence_ids = clean(rc.evidence_ids, "root_cause")
        rc.status = enforce(rc.status, rc.evidence_ids, "root_cause")

        for i, f in enumerate(report.findings):
            f.evidence_ids = clean(f.evidence_ids, f"findings[{i}]")
            f.status = enforce(f.status, f.evidence_ids, f"findings[{i}]")
            if f.status is FindingStatus.REFUTED and not f.evidence_ids:
                warnings.append(f"findings[{i}]: refuted without any cited evidence")

        for i, t in enumerate(report.timeline):
            t.evidence_ids = clean(t.evidence_ids, f"timeline[{i}]")
        for i, r in enumerate(report.remediation):
            r.evidence_ids = clean(r.evidence_ids, f"remediation[{i}]")
            if not r.evidence_ids:
                warnings.append(f"remediation[{i}]: no supporting evidence cited")

        report.validation_warnings.extend(warnings)
        return report
