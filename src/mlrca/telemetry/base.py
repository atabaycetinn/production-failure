"""Telemetry backend protocol.

Every method is read-only. Production implementations (Prometheus/Datadog, Loki/Elastic,
ArgoCD/GitHub, Evidently/Arize) should implement this same surface so the agent's
tools don't change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class TelemetryBackend(Protocol):
    def now(self) -> datetime:
        """The investigation's reference time (wall clock in production; fixed in fixtures)."""

    # metrics
    def list_metrics(self, entity: str | None = None) -> list[dict[str, Any]]: ...
    def query_metric(self, entity: str, metric: str, start: datetime, end: datetime) -> list[tuple[datetime, float]]: ...

    # deployments & code
    def list_deployments(
        self, service: str | None = None, start: datetime | None = None, end: datetime | None = None
    ) -> list[dict[str, Any]]: ...
    def get_deployment(self, deployment_id: str) -> dict[str, Any] | None: ...
    def get_commit(self, sha: str) -> dict[str, Any] | None: ...

    # logs
    def search_logs(
        self,
        start: datetime,
        end: datetime,
        service: str | None = None,
        level: str | None = None,
        contains: str | None = None,
    ) -> list[dict[str, Any]]: ...

    # features
    def list_features(self, model: str) -> list[dict[str, Any]]: ...
    def feature_histogram(self, model: str, feature: str, start: datetime, end: datetime) -> dict[str, Any]: ...
