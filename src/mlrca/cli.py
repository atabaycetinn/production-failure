"""Command-line interface.

    mlrca search "amount scaler log1p"                 # inspect RAG retrieval
    mlrca baseline                                     # heuristic (no LLM) investigation
    mlrca investigate [--question "..."]               # Claude agent investigation
    mlrca eval [--system agent|baseline|both]          # score against ground truth
    mlrca actions list | approve ID --by NAME | reject ID --by NAME | execute ID
"""

from __future__ import annotations

import argparse
import json
import sys

from .actions import ActionStore, ApprovalError, execute
from .agent import InvestigationError, Investigator
from .config import get_settings
from .eval import build_context, list_scenarios, run_one, save_run
from .rag import BM25Retriever, load_corpus
from .render import report_markdown


def _scenario_dir(name: str):
    s = get_settings()
    d = s.scenarios_dir / name
    if not (d / "scenario.json").exists():
        sys.exit(f"unknown scenario '{name}'. available: {[p.name for p in list_scenarios(s)]}")
    return d


def _action_store() -> ActionStore:
    return ActionStore(get_settings().runs_dir / "actions.json")


def _print_event(kind: str, data: dict) -> None:
    if kind == "tool":
        flag = "ERR" if data["is_error"] else (data["evidence_id"] or "   ")
        args = json.dumps(data["input"], default=str)
        print(f"  [{data['turn']:>2}] {flag:<7} {data['tool']}({args[:110]})", file=sys.stderr)
    elif kind == "text":
        print(f"  [{data['turn']:>2}] 💬 {data['text'][:300]}", file=sys.stderr)


def cmd_search(args) -> None:
    r = BM25Retriever(load_corpus(get_settings().corpus_dir))
    for h in r.search(args.query, top_k=args.k, doc_types=args.type or None):
        print(f"{h.score:7.3f}  {h.chunk.chunk_id}  ({h.chunk.doc_type})")


def cmd_baseline(args) -> None:
    out = run_one(_scenario_dir(args.scenario), "baseline")
    print(out.markdown)
    path = save_run(out, get_settings().runs_dir)
    print(f"score={out.score.total} {out.score.checks}\nsaved to {path}", file=sys.stderr)


def cmd_investigate(args) -> None:
    settings = get_settings()
    store = _action_store()
    ctx = build_context(_scenario_dir(args.scenario), settings, actions=store)
    scen = ctx.backend.scenario  # type: ignore[attr-defined]
    question = args.question or scen["question"]
    print(f"Investigating with {settings.model} (effort={settings.effort})…\n", file=sys.stderr)
    try:
        res = Investigator(ctx, settings=settings, on_event=_print_event).run(question, scen.get("model"))
    except InvestigationError as exc:
        sys.exit(f"investigation failed: {exc}")
    if res.report is None:
        sys.exit(f"no report produced (stop={res.stop})")
    md = report_markdown(res.report, ctx.ledger, store)
    print(md)
    print(f"turns={res.turns} tool_calls={len(res.trace)} usage={res.usage}", file=sys.stderr)
    pending = [a for a in store.list() if a.status.value == "pending_approval"]
    if pending:
        print("\nPending action requests (approve with `mlrca actions approve <ID> --by <name>`):", file=sys.stderr)
        for a in pending:
            print(f"  {a.id}  {a.action_type.value} {a.target} {a.parameters}", file=sys.stderr)


def cmd_eval(args) -> None:
    settings = get_settings()
    systems = ["baseline", "agent"] if args.system == "both" else [args.system]
    scenarios = [_scenario_dir(args.scenario)] if args.scenario else list_scenarios(settings)
    rows = []
    for scen in scenarios:
        for system in systems:
            print(f"running {system} on {scen.name}…", file=sys.stderr)
            try:
                out = run_one(scen, system, settings, on_event=_print_event if args.verbose else None)
            except InvestigationError as exc:
                print(f"  failed: {exc}", file=sys.stderr)
                continue
            path = save_run(out, settings.runs_dir)
            rows.append((scen.name, system, out.score, out.seconds, path))
    if not rows:
        sys.exit("no runs completed")
    keys = list(rows[0][2].checks)
    print("\n| scenario | system | total | " + " | ".join(keys) + " | secs |")
    print("|---" * (len(keys) + 4) + "|")
    for name, system, score, secs, _ in rows:
        print(f"| {name} | {system} | **{score.total}** | " + " | ".join(f"{score.checks[k]:g}" for k in keys) + f" | {secs} |")
    for name, system, score, _, path in rows:
        for n in score.notes:
            print(f"  - {name}/{system}: {n}", file=sys.stderr)
        print(f"  saved {name}/{system} -> {path}", file=sys.stderr)


def cmd_actions(args) -> None:
    store = _action_store()
    try:
        if args.op == "list":
            for a in store.list():
                print(f"{a.id}  {a.status.value:<17} {a.action_type.value} {a.target} {a.parameters}  evidence={a.evidence_ids}")
                print(f"    rationale: {a.rationale}")
        elif args.op == "approve":
            a = store.approve(args.id, args.by, args.note)
            print(f"{a.id} approved by {a.decided_by}")
        elif args.op == "reject":
            a = store.reject(args.id, args.by, args.note)
            print(f"{a.id} rejected by {a.decided_by}")
        elif args.op == "execute":
            a = execute(store, args.id)
            print(a.execution_log[-1])
    except ApprovalError as exc:
        sys.exit(str(exc))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mlrca", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="query the documentation index")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--type", action="append", help="doc type filter (repeatable)")
    s.set_defaults(func=cmd_search)

    for name, fn in (("baseline", cmd_baseline), ("investigate", cmd_investigate)):
        x = sub.add_parser(name)
        x.add_argument("--scenario", default="fraud_precision_drop")
        if name == "investigate":
            x.add_argument("--question", help="override the scenario's question")
        x.set_defaults(func=fn)

    e = sub.add_parser("eval")
    e.add_argument("--system", choices=["agent", "baseline", "both"], default="both")
    e.add_argument("--scenario", help="single scenario (default: all)")
    e.add_argument("-v", "--verbose", action="store_true")
    e.set_defaults(func=cmd_eval)

    a = sub.add_parser("actions", help="human approval workflow")
    a.add_argument("op", choices=["list", "approve", "reject", "execute"])
    a.add_argument("id", nargs="?")
    a.add_argument("--by", help="name of the approving human")
    a.add_argument("--note")
    a.set_defaults(func=cmd_actions)

    args = p.parse_args(argv)
    if args.cmd == "actions" and args.op != "list" and not args.id:
        p.error("action id required")
    if args.cmd == "actions" and args.op in ("approve", "reject") and not args.by:
        p.error("--by is required: approvals must name a human")
    args.func(args)


if __name__ == "__main__":
    main()
