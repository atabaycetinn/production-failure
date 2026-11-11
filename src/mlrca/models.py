"""Domain models shared across RAG, telemetry, agent, actions and evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- evidence


class EvidenceKind(str, Enum):
    DOCUMENTATION = "documentation"  # claims: what docs/release notes *say*
    DEPLOYMENT = "deployment"
    METRICS = "metrics"
    LOGS = "logs"
    CODE_CHANGE = "code_change"
    FEATURE_DRIFT = "feature_drift"

    @property
    def observational(self) -> bool:
        """Observational evidence can verify a finding; documentation alone cannot."""
        return self is not EvidenceKind.DOCUMENTATION


class Evidence(BaseModel):
    id: str
    kind: EvidenceKind
    source: str  # tool name
    query: dict[str, Any]
    summary: str
    data: Any = None
    collected_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- findings / report


class FindingStatus(str, Enum):
    VERIFIED = "verified"
    HYPOTHESIS = "hypothesis"
    REFUTED = "refuted"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Finding(BaseModel):
    statement: str
    status: FindingStatus
    evidence_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class RootCause(BaseModel):
    statement: str
    status: FindingStatus
    confidence: Confidence
    component: str = Field(description="Service or artifact at fault, e.g. 'fraud-preproc'")
    change_ref: str | None = Field(default=None, description="Deployment ID and/or commit SHA, if identified")
    evidence_ids: list[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    timestamp: str
    event: str
    evidence_ids: list[str] = Field(default_factory=list)


class RemediationItem(BaseModel):
    description: str
    action_request_id: str | None = Field(
        default=None, description="ID returned by propose_remediation, if an operational action was proposed"
    )
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class Report(BaseModel):
    summary: str
    root_cause: RootCause
    findings: list[Finding] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    remediation: list[RemediationItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    # Populated by the harness, not by the model:
    validation_warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- operational actions


class ActionType(str, Enum):
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    TOGGLE_FEATURE_FLAG = "toggle_feature_flag"
    ADJUST_THRESHOLD = "adjust_threshold"
    PAUSE_MODEL = "pause_model"


class ActionStatus(str, Enum):
    PENDING = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


class ActionRequest(BaseModel):
    id: str
    action_type: ActionType
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    risk: str = "medium"
    status: ActionStatus = ActionStatus.PENDING
    proposed_by: str = "mlrca-agent"
    proposed_at: datetime = Field(default_factory=utcnow)
    decided_by: str | None = None
    decided_at: datetime | None = None
    decision_note: str | None = None
    execution_log: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- RAG


class DocChunk(BaseModel):
    chunk_id: str  # "<doc_id>#<section-slug>"
    doc_id: str
    title: str
    section: str
    doc_type: str  # architecture | model_card | runbook | incident | release_notes
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: DocChunk
    score: float
