"""Human-in-the-loop approval workflow for operational actions.

State machine::

    pending_approval --approve--> approved --execute--> executed | failed
                     --reject---> rejected

The agent can only *propose* (create pending requests). Approval and execution are
separate, human-initiated steps (CLI today; Slack / change-management later).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

from ..models import ActionRequest, ActionStatus, ActionType, utcnow


class ApprovalError(RuntimeError):
    pass


class ActionStore:
    """JSON-file-backed store. Pass ``path=None`` for an in-memory store (tests, eval)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._items: dict[str, ActionRequest] = {}
        if self.path and self.path.exists():
            for raw in json.loads(self.path.read_text()):
                req = ActionRequest.model_validate(raw)
                self._items[req.id] = req

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([r.model_dump(mode="json") for r in self._items.values()], indent=2))

    def _next_id(self) -> str:
        stamp = utcnow().strftime("%Y%m%d")
        return f"ACT-{stamp}-{len(self._items) + 1:03d}"

    def propose(
        self,
        action_type: ActionType,
        target: str,
        parameters: dict,
        rationale: str,
        evidence_ids: list[str],
        risk: str = "medium",
    ) -> ActionRequest:
        with self._lock:
            req = ActionRequest(
                id=self._next_id(),
                action_type=action_type,
                target=target,
                parameters=parameters,
                rationale=rationale,
                evidence_ids=evidence_ids,
                risk=risk,
            )
            self._items[req.id] = req
            self._save()
            return req

    def get(self, action_id: str) -> ActionRequest:
        if action_id not in self._items:
            raise ApprovalError(f"unknown action request {action_id}")
        return self._items[action_id]

    def list(self, status: ActionStatus | None = None) -> list[ActionRequest]:
        return [r for r in self._items.values() if status is None or r.status is status]

    def _decide(self, action_id: str, status: ActionStatus, who: str, note: str | None) -> ActionRequest:
        if not who:
            raise ApprovalError("a named human decision-maker is required")
        with self._lock:
            req = self.get(action_id)
            if req.status is not ActionStatus.PENDING:
                raise ApprovalError(f"{action_id} is {req.status.value}, not pending")
            req.status, req.decided_by, req.decided_at, req.decision_note = status, who, utcnow(), note
            self._save()
            return req

    def approve(self, action_id: str, approver: str, note: str | None = None) -> ActionRequest:
        return self._decide(action_id, ActionStatus.APPROVED, approver, note)

    def reject(self, action_id: str, approver: str, note: str | None = None) -> ActionRequest:
        return self._decide(action_id, ActionStatus.REJECTED, approver, note)

    def mark(self, req: ActionRequest, status: ActionStatus, log: str) -> None:
        with self._lock:
            req.status = status
            req.execution_log.append(f"{utcnow().isoformat()} {log}")
            self._save()


class Executor(Protocol):
    def run(self, req: ActionRequest) -> str: ...


class DryRunExecutor:
    """Describes what would happen. Swap for ArgoCD / feature-flag executors in Phase 4."""

    def run(self, req: ActionRequest) -> str:
        p = req.parameters
        match req.action_type:
            case ActionType.ROLLBACK_DEPLOYMENT:
                return f"[dry-run] would roll back {req.target} to {p.get('to_version', '?')}"
            case ActionType.TOGGLE_FEATURE_FLAG:
                return f"[dry-run] would set flag {p.get('flag', '?')}={p.get('value', '?')} on {req.target}"
            case ActionType.ADJUST_THRESHOLD:
                return f"[dry-run] would set decision threshold of {req.target} to {p.get('threshold', '?')}"
            case ActionType.PAUSE_MODEL:
                return f"[dry-run] would route {req.target} traffic to fallback rules engine"
        return "[dry-run] unknown action"


def execute(store: ActionStore, action_id: str, executor: Executor | None = None) -> ActionRequest:
    """Execute an *approved* action. Refuses anything else."""
    req = store.get(action_id)
    if req.status is not ActionStatus.APPROVED:
        raise ApprovalError(f"refusing to execute {action_id}: status is {req.status.value}, approval required")
    executor = executor or DryRunExecutor()
    try:
        msg = executor.run(req)
    except Exception as exc:  # noqa: BLE001 - surface any executor failure on the request
        store.mark(req, ActionStatus.FAILED, f"execution failed: {exc}")
        raise
    store.mark(req, ActionStatus.EXECUTED, msg)
    return req
