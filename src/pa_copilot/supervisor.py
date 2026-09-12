"""Supervisor = deterministic guardrails + LLM router (design.md §3.2). Hard
rules run first and never touch the model; only when none match does one
.with_structured_output(RouterDecision) call happen. The supervisor never binds
tools — workers own their own MCP/RAG calls; the supervisor only ever sees each
worker's validated Pydantic output plus the compressed running summary.

Guardrail ordering: design.md §3.2 lists the four hard rules in reading order
as request-None, tool-failures-cap, decision-finished, hops-cap, but doesn't
say what happens when more than one matches at once (they're meant to be
mutually exclusive in practice). The task brief resolves this explicitly: the
hop cap is a hard ceiling meant to win regardless of any other condition, so it
is checked FIRST here -- verified by
test_hard_route_hop_cap_wins_over_finished_decision, which pins a state where
both the hop cap and "decision finished" would match and asserts the hop cap
wins. Order implemented: hop cap -> tool-failure cap -> decision finished ->
request-None -> default None. (request-None and hops-cap can't both be true in
practice, since a real state only starts accumulating supervisor_hops once a
request exists, so its position among the checks doesn't affect current
tests -- kept ahead of "decision finished" since design.md lists it first.)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from pa_copilot.agents._react import get_agent_model
from pa_copilot.config import Settings, get_settings
from pa_copilot.schemas import RouteStep, RouterDecision, RouteTarget
from pa_copilot.state import PACaseState


def hard_route(state: PACaseState, *, settings: Settings | None = None) -> RouteTarget | None:
    s = settings or get_settings()

    if state.get("supervisor_hops", 0) >= s.max_hops:
        return "human_review"
    if state.get("tool_failures") and state.get("replan_count", 0) >= s.max_replans:
        return "human_review"
    if state.get("decision") is not None and not state.get("needs_replan"):
        return "FINISH"
    if state.get("request") is None:
        return "intake"
    return None


def _missing_checklist(state: PACaseState) -> list[str]:
    return [f for f in ("request", "benefit", "necessity", "decision") if state.get(f) is None]


async def route_with_llm(state: PACaseState, *, model=None) -> RouterDecision:
    model = model or get_agent_model()
    summary = (state.get("context") or {}).get("running_summary")
    summary_text = getattr(summary, "summary", None) or "no summary yet"
    history = state.get("route_history") or []
    prompt = (
        "You are the routing supervisor for a prior-authorization case.\n"
        f"Case summary so far: {summary_text}\n"
        f"Still missing: {_missing_checklist(state)}\n"
        f"Recent routing history: {[h for h in history[-5:]]}\n"
        "Choose the next node."
    )
    return await model.with_structured_output(RouterDecision).ainvoke([("user", prompt)])


def build_supervisor_node(*, model=None, settings: Settings | None = None) -> Callable:
    s = settings or get_settings()

    async def _node(state: PACaseState) -> dict:
        target = hard_route(state, settings=s)
        if target is not None:
            reason = f"deterministic guardrail -> {target}"
        else:
            decision = await route_with_llm(state, model=model)
            target = decision.next
            reason = decision.rationale

        step = RouteStep(
            from_node="supervisor", to_node=target, reason=reason,
            ts=datetime.now(timezone.utc).isoformat(),
        )
        return {
            "next": target,
            "route_history": [step],
            "supervisor_hops": state.get("supervisor_hops", 0) + 1,
        }

    return _node
