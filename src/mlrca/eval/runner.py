"""Run the agent and/or heuristic baseline over scenarios and score them."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..actions import ActionStore
from ..agent import Investigator, ToolContext
from ..baselines import run_heuristic
from ..config import Settings, get_settings
from ..evidence import EvidenceLedger
from ..models import Report
from ..rag import BM25Retriever, load_corpus
from ..render import report_markdown
from ..telemetry import FixtureBackend
from .scoring import Score, score_report

System = Literal["agent", "baseline"]


@dataclass
class RunOutput:
    scenario: str
    system: System
    report: Report | None
    score: Score
    seconds: float
    meta: dict[str, Any]
    markdown: str


def list_scenarios(settings: Settings) -> list[Path]:
    return sorted(p for p in settings.scenarios_dir.iterdir() if (p / "scenario.json").exists())


def build_context(scenario_dir: Path, settings: Settings, actions: ActionStore | None = None) -> ToolContext:
    return ToolContext(
        backend=FixtureBackend(scenario_dir),
        retriever=BM25Retriever(load_corpus(settings.corpus_dir)),
        ledger=EvidenceLedger(),
        actions=actions or ActionStore(),
    )


def run_one(scenario_dir: Path, system: System, settings: Settings | None = None, client: Any = None,
            on_event=None) -> RunOutput:
    settings = settings or get_settings()
    ctx = build_context(scenario_dir, settings)
    scen = ctx.backend.scenario  # type: ignore[attr-defined]
    t0 = time.perf_counter()
    meta: dict[str, Any] = {}
    if system == "baseline":
        report = run_heuristic(ctx, scen["model"])
    else:
        res = Investigator(ctx, client=client, settings=settings, on_event=on_event).run(scen["question"], scen.get("model"))
        report = res.report
        meta = {"turns": res.turns, "stop": res.stop, "usage": res.usage, "tool_calls": len(res.trace), "trace": res.trace}
    secs = round(time.perf_counter() - t0, 2)
    score = score_report(report, ctx.actions, scen["ground_truth"])
    md = report_markdown(report, ctx.ledger, ctx.actions) if report else "_no report_\n"
    meta["evidence"] = [e.model_dump(mode="json", exclude={"data"}) for e in ctx.ledger.all()]
    meta["actions"] = [a.model_dump(mode="json") for a in ctx.actions.list()]
    return RunOutput(scen["id"], system, report, score, secs, meta, md)


def save_run(out: RunOutput, runs_dir: Path) -> Path:
    d = runs_dir / time.strftime("%Y%m%d-%H%M%S") / f"{out.scenario}-{out.system}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.md").write_text(out.markdown)
    (d / "result.json").write_text(json.dumps({
        "scenario": out.scenario, "system": out.system, "seconds": out.seconds,
        "score": {"total": out.score.total, "checks": out.score.checks, "notes": out.score.notes},
        "report": out.report.model_dump(mode="json") if out.report else None,
        **out.meta,
    }, indent=2, default=str))
    return d
