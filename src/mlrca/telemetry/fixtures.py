"""Fixture-backed telemetry: reads a scenario directory produced by scripts/generate_scenario.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .stats import parse_ts


class FixtureBackend:
    """Scenario directory layout::

        scenario.json        question, reference time, ground truth
        deployments.json     [{id, service, version, previous_version, deployed_at, commits, ...}]
        commits.json         [{sha, service, message, author, committed_at, files: [{path, diff}]}]
        metrics.json         [{entity, metric, unit, description, points: [[ts, value], ...]}]
        logs.jsonl           {ts, service, level, message, attrs}
        features.json        {model, bin_edges: {feature: [...]}, catalog: [...], hourly: {feature: [[ts, counts]]}}
    """

    def __init__(self, scenario_dir: Path) -> None:
        self.dir = Path(scenario_dir)
        self.scenario: dict[str, Any] = self._json("scenario.json")
        self._now = parse_ts(self.scenario["now"])
        self._deployments: list[dict[str, Any]] = self._json("deployments.json")
        self._commits = {c["sha"]: c for c in self._json("commits.json")}
        self._metrics = {(m["entity"], m["metric"]): m for m in self._json("metrics.json")}
        self._features: dict[str, Any] = self._json("features.json")
        self._logs: list[dict[str, Any]] = [
            json.loads(line) for line in (self.dir / "logs.jsonl").read_text().splitlines() if line.strip()
        ]
        for log in self._logs:
            log["_ts"] = parse_ts(log["ts"])

    def _json(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text())

    def now(self) -> datetime:
        return self._now

    # ------------------------------------------------------------------ metrics

    def list_metrics(self, entity: str | None = None) -> list[dict[str, Any]]:
        return [
            {k: m[k] for k in ("entity", "metric", "unit", "description")}
            for m in self._metrics.values()
            if entity is None or m["entity"] == entity
        ]

    def query_metric(self, entity: str, metric: str, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
        m = self._metrics.get((entity, metric))
        if m is None:
            raise KeyError(f"unknown metric {entity}/{metric}")
        pts = [(parse_ts(ts), float(v)) for ts, v in m["points"]]
        return [(t, v) for t, v in pts if start <= t < end]

    # ------------------------------------------------------------------ deployments & code

    def list_deployments(
        self, service: str | None = None, start: datetime | None = None, end: datetime | None = None
    ) -> list[dict[str, Any]]:
        out = []
        for d in self._deployments:
            t = parse_ts(d["deployed_at"])
            if service and d["service"] != service:
                continue
            if start and t < start:
                continue
            if end and t >= end:
                continue
            out.append(d)
        return sorted(out, key=lambda d: d["deployed_at"])

    def get_deployment(self, deployment_id: str) -> dict[str, Any] | None:
        return next((d for d in self._deployments if d["id"] == deployment_id), None)

    def get_commit(self, sha: str) -> dict[str, Any] | None:
        if sha in self._commits:
            return self._commits[sha]
        matches = [c for s, c in self._commits.items() if s.startswith(sha)]
        return matches[0] if len(matches) == 1 else None

    # ------------------------------------------------------------------ logs

    def search_logs(
        self,
        start: datetime,
        end: datetime,
        service: str | None = None,
        level: str | None = None,
        contains: str | None = None,
    ) -> list[dict[str, Any]]:
        needle = contains.lower() if contains else None
        out = []
        for log in self._logs:
            if not (start <= log["_ts"] < end):
                continue
            if service and log["service"] != service:
                continue
            if level and log["level"].upper() != level.upper():
                continue
            if needle and needle not in log["message"].lower() and needle not in json.dumps(log.get("attrs", {})).lower():
                continue
            out.append({k: v for k, v in log.items() if k != "_ts"})
        return out

    # ------------------------------------------------------------------ features

    def list_features(self, model: str) -> list[dict[str, Any]]:
        if self._features.get("model") != model:
            return []
        return self._features["catalog"]

    def feature_histogram(self, model: str, feature: str, start: datetime, end: datetime) -> dict[str, Any]:
        if self._features.get("model") != model or feature not in self._features["hourly"]:
            raise KeyError(f"unknown feature {model}/{feature}")
        edges = self._features["bin_edges"][feature]
        counts = [0] * (len(edges) - 1)
        hours = 0
        for ts, row in self._features["hourly"][feature]:
            if start <= parse_ts(ts) < end:
                hours += 1
                counts = [a + b for a, b in zip(counts, row)]
        return {"bin_edges": edges, "counts": counts, "hours": hours}
