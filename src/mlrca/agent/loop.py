"""The investigator agent loop (Claude Messages API + client-side tools).

A manual loop rather than the SDK tool runner so the harness can: stop as soon as a valid
``submit_report`` arrives, persist a per-step trace, nudge the model when it stops without a
report, and keep every tool input validated by pydantic before execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from ..config import Settings, get_settings
from ..models import Report
from ..telemetry import fmt_ts
from .prompts import NUDGE_SUBMIT, SYSTEM_PROMPT, initial_user_message
from .tools import TOOL_DEFINITIONS, ToolContext

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class InvestigationError(RuntimeError):
    pass


@dataclass
class InvestigationResult:
    report: Report | None
    trace: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    stop: str = ""


EventHook = Callable[[str, dict[str, Any]], None]


class Investigator:
    def __init__(
        self,
        ctx: ToolContext,
        client: Any | None = None,
        settings: Settings | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.ctx = ctx
        self.settings = settings or get_settings()
        self.client = client or anthropic.Anthropic()
        self.on_event = on_event or (lambda kind, data: None)

    # ------------------------------------------------------------------ request

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "tools": TOOL_DEFINITIONS,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.effort},
            "cache_control": {"type": "ephemeral"},  # auto-cache the growing conversation tail
        }
        if self.settings.fallbacks_enabled:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return self.client.beta.messages.create(**params)

    # ------------------------------------------------------------------ loop

    def run(self, question: str, model_hint: str | None = None) -> InvestigationResult:
        now = fmt_ts(self.ctx.backend.now())
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": initial_user_message(question, now, model_hint)}
        ]
        result = InvestigationResult(report=None)
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        nudges = 0

        for turn in range(1, self.settings.max_turns + 1):
            result.turns = turn
            try:
                response = self._create(messages)
            except anthropic.RateLimitError as exc:
                raise InvestigationError(f"rate limited after SDK retries: {exc}") from exc
            except anthropic.APIStatusError as exc:
                raise InvestigationError(f"API error {exc.status_code}: {exc.message}") from exc
            except anthropic.APIConnectionError as exc:
                raise InvestigationError(f"connection error: {exc}") from exc

            for k in usage:
                usage[k] += getattr(response.usage, k, 0) or 0

            stop = response.stop_reason
            if stop == "refusal":
                details = getattr(response, "stop_details", None)
                raise InvestigationError(f"model declined the request: {getattr(details, 'category', None)}")
            if stop == "max_tokens":
                raise InvestigationError("response hit max_tokens; increase Settings.max_tokens")

            messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    self.on_event("text", {"turn": turn, "text": block.text})

            if stop == "pause_turn":
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if nudges < 2:
                    nudges += 1
                    messages.append({"role": "user", "content": NUDGE_SUBMIT})
                    continue
                result.stop = "ended_without_report"
                break

            # Run evidence-gathering tools before submit_report so same-turn citations resolve.
            ordered = sorted(tool_uses, key=lambda b: b.name == "submit_report")
            results_by_id: dict[str, dict[str, Any]] = {}
            for tu in ordered:
                t0 = time.perf_counter()
                outcome = self.ctx.dispatch(tu.name, tu.input)
                step = {
                    "turn": turn, "tool": tu.name, "input": tu.input, "is_error": outcome.is_error,
                    "evidence_id": outcome.evidence_id, "ms": round((time.perf_counter() - t0) * 1000, 1),
                }
                result.trace.append(step)
                self.on_event("tool", {**step, "preview": outcome.content[:300]})
                results_by_id[tu.id] = {
                    "type": "tool_result", "tool_use_id": tu.id, "content": outcome.content,
                    **({"is_error": True} if outcome.is_error else {}),
                }
                if outcome.report is not None:
                    result.report = outcome.report

            if result.report is not None:
                result.stop = "report_submitted"
                break
            # All results for one assistant turn go back in a single user message, in call order.
            messages.append({"role": "user", "content": [results_by_id[tu.id] for tu in tool_uses]})
        else:
            result.stop = "max_turns"

        result.usage = usage
        return result
